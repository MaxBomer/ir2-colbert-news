"""ColBERT adapter for news recommendation with late interaction MaxSim."""
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add colbert to path if needed
# __file__ is baseline/model/ColBERT/__init__.py
# We need to go up to baseline/../colbert
baseline_dir = Path(__file__).parent.parent.parent  # baseline/
project_root = baseline_dir.parent  # ir2-colbert-news/
colbert_path = project_root / 'colbert'
if str(colbert_path) not in sys.path:
    sys.path.insert(0, str(colbert_path))

from pylate import models as colbert_models
from transformers import AutoTokenizer

from config import NRMSbertConfig
from model.base import BaseNewsRecommendationModel
from model.NRMSbert.user_encoder import UserEncoder


def maxsim_score(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Compute ColBERT MaxSim score between query and document token embeddings.
    
    Args:
        query_embeddings: [batch_size, num_query_tokens, embedding_dim] or [num_query_tokens, embedding_dim]
        doc_embeddings: [batch_size, num_doc_tokens, embedding_dim] or [num_doc_tokens, embedding_dim]
        
    Returns:
        Score tensor: [batch_size] or scalar
    """
    # Handle single sample case
    if query_embeddings.dim() == 2:
        query_embeddings = query_embeddings.unsqueeze(0)
    if doc_embeddings.dim() == 2:
        doc_embeddings = doc_embeddings.unsqueeze(0)
    
    batch_size = query_embeddings.shape[0]
    num_query_tokens = query_embeddings.shape[1]
    num_doc_tokens = doc_embeddings.shape[1]
    embedding_dim = query_embeddings.shape[2]
    
    # Normalize embeddings (ColBERT uses cosine similarity)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)
    
    # Compute similarity matrix: [batch_size, num_query_tokens, num_doc_tokens]
    # Using matrix multiplication: Q @ D^T
    similarity_matrix = torch.bmm(query_embeddings, doc_embeddings.transpose(1, 2))
    
    # MaxSim: for each query token, take max similarity with any doc token
    # [batch_size, num_query_tokens]
    max_similarities = similarity_matrix.max(dim=2)[0]
    
    # Sum over query tokens: [batch_size]
    scores = max_similarities.sum(dim=1)
    
    return scores


class ColBERTNewsRecommendationModel(BaseNewsRecommendationModel):
    """ColBERT adapter for news recommendation with late interaction MaxSim.
    
    Uses ColBERT's token-level embeddings with MaxSim scoring instead of pooling.
    Implements proper late interaction as described in the ColBERT paper.
    """
    
    def __init__(self, config: NRMSbertConfig) -> None:
        """Initialize ColBERT adapter.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # Initialize ColBERT model
        colbert_model_name = config.colbert_model_name if config.colbert_model_name is not None else config.pretrained_model_name
        embedding_size = getattr(config, 'colbert_embedding_dim', None)
        try:
            self.colbert_model = colbert_models.ColBERT(
                model_name_or_path=colbert_model_name,
                device=None,  # Will be set when moved to device
                embedding_size=embedding_size,
                query_length=config.colbert_max_query_tokens,
                document_length=config.colbert_max_doc_tokens,
            )
        except Exception as e:
            try:
                self.colbert_model = colbert_models.ColBERT(
                    model_name_or_path=colbert_model_name,
                    device=None,
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to initialize ColBERT model with '{colbert_model_name}'. "
                    f"Original error: {e}. Secondary error: {e2}"
                ) from e2
        
        # Get embedding dimension
        if embedding_size is not None:
            self.embedding_dim = embedding_size
        else:
            self.embedding_dim = config.colbert_embedding_dim if config.colbert_embedding_dim is not None else 128
        
        # Initialize tokenizer for converting token IDs back to text
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
        
        # Token limits
        self.max_query_tokens = config.colbert_max_query_tokens
        self.max_doc_tokens = config.colbert_max_doc_tokens
        
        # Embedding cache (LRU cache)
        self.enable_caching = config.colbert_enable_caching
        self.cache_size = config.colbert_cache_size
        self._embedding_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        
        # User encoder - we'll aggregate token embeddings before passing to it
        # For now, we'll use mean pooling of token embeddings for user encoding
        # This could be improved with attention over tokens
        self.user_encoder = UserEncoder(config, embedding_dim=self.embedding_dim)
    
    def _prune_tokens(self, input_ids: torch.Tensor, max_tokens: int) -> torch.Tensor:
        """Truncate tokens to max_tokens length (ColBERT handles special tokens internally).
        
        Args:
            input_ids: [batch_size, seq_len] or [seq_len]
            max_tokens: Maximum number of tokens to keep
            
        Returns:
            Truncated input_ids: [batch_size, max_tokens] or [max_tokens]
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            was_1d = True
        else:
            was_1d = False
        
        batch_size, seq_len = input_ids.shape
        
        # Simply truncate to max_tokens (ColBERT will handle special tokens)
        if seq_len > max_tokens:
            input_ids = input_ids[:, :max_tokens]
        elif seq_len < max_tokens:
            # Pad if needed
            padding = torch.zeros(batch_size, max_tokens - seq_len, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        
        if was_1d:
            input_ids = input_ids.squeeze(0)
        return input_ids
    
    def _token_ids_to_text(self, input_ids: torch.Tensor) -> List[str]:
        """Convert token IDs back to text.
        
        Args:
            input_ids: Tensor with shape [batch_size, seq_len] or [seq_len]
            
        Returns:
            List of text strings
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        texts = []
        for ids in input_ids:
            # Remove padding tokens (0) only - keep special tokens for ColBERT
            ids = ids[ids != 0]
            # Decode (skip_special_tokens=False to preserve them, but ColBERT will handle)
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        
        return texts
    
    def _get_cache_key(self, text: str, is_query: bool) -> str:
        """Generate cache key for text."""
        return f"{'query' if is_query else 'doc'}:{text}"
    
    def _encode_with_colbert(
        self,
        texts: List[str],
        is_query: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode texts with ColBERT, returning token-level embeddings.
        
        Args:
            texts: List of text strings
            is_query: Whether encoding as queries (True) or documents (False)
            device: Device to use
            
        Returns:
            Token embeddings: [batch_size, num_tokens, embedding_dim]
        """
        # Check cache if enabled
        if self.enable_caching:
            cached_embeddings = []
            texts_to_encode = []
            cache_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, is_query)
                if cache_key in self._embedding_cache:
                    cached_embeddings.append((i, self._embedding_cache[cache_key]))
                    # Move to end (LRU)
                    self._embedding_cache.move_to_end(cache_key)
                else:
                    texts_to_encode.append((i, text))
                    cache_indices.append(i)
            
            # Encode uncached texts
            if texts_to_encode:
                texts_list = [text for _, text in texts_to_encode]
                embeddings_list = self.colbert_model.encode(
                    sentences=texts_list,
                    batch_size=len(texts_list),
                    convert_to_tensor=True,
                    convert_to_numpy=False,
                    is_query=is_query,
                    device=str(device),
                    show_progress_bar=False,
                )
                
                # Process embeddings (ColBERT returns list of [num_tokens, embedding_dim])
                if isinstance(embeddings_list, list):
                    for (orig_idx, _), emb in zip(texts_to_encode, embeddings_list):
                        if isinstance(emb, torch.Tensor):
                            emb_tensor = emb.to(device)
                        else:
                            emb_tensor = torch.from_numpy(emb).to(device)
                        
                        # Cache it
                        cache_key = self._get_cache_key(texts_list[texts_to_encode.index((orig_idx, _))], is_query)
                        self._embedding_cache[cache_key] = emb_tensor.cpu()
                        # Enforce cache size limit
                        if len(self._embedding_cache) > self.cache_size:
                            self._embedding_cache.popitem(last=False)
                        
                        cached_embeddings.append((orig_idx, emb_tensor))
                else:
                    # Single tensor case
                    if embeddings_list.dim() == 2:
                        embeddings_list = embeddings_list.unsqueeze(0)
                    for i, (orig_idx, _) in enumerate(texts_to_encode):
                        emb_tensor = embeddings_list[i].to(device)
                        cache_key = self._get_cache_key(texts_list[i], is_query)
                        self._embedding_cache[cache_key] = emb_tensor.cpu()
                        if len(self._embedding_cache) > self.cache_size:
                            self._embedding_cache.popitem(last=False)
                        cached_embeddings.append((orig_idx, emb_tensor))
            
            # Sort by original index and stack
            cached_embeddings.sort(key=lambda x: x[0])
            embeddings = torch.stack([emb for _, emb in cached_embeddings], dim=0)
        else:
            # No caching
            embeddings_list = self.colbert_model.encode(
                sentences=texts,
                batch_size=len(texts),
                convert_to_tensor=True,
                convert_to_numpy=False,
                is_query=is_query,
                device=str(device),
                show_progress_bar=False,
            )
            
            # Process embeddings
            if isinstance(embeddings_list, list):
                embeddings = []
                for emb in embeddings_list:
                    if isinstance(emb, torch.Tensor):
                        embeddings.append(emb.to(device))
                    else:
                        embeddings.append(torch.from_numpy(emb).to(device))
                embeddings = torch.stack(embeddings, dim=0)
            elif isinstance(embeddings_list, torch.Tensor):
                embeddings = embeddings_list.to(device)
                if embeddings.dim() == 2:
                    embeddings = embeddings.unsqueeze(0)
            else:
                # numpy array
                embeddings = torch.from_numpy(embeddings_list).to(device)
                if embeddings.dim() == 2:
                    embeddings = embeddings.unsqueeze(0)
        
        return embeddings  # [batch_size, num_tokens, embedding_dim]
    
    def forward(
        self,
        candidate_news: List[Dict[str, torch.Tensor]],
        clicked_news: List[Dict[str, torch.Tensor]],
        clicked_news_mask: List[List[int]],
    ) -> torch.Tensor:
        """Forward pass using ColBERT late interaction MaxSim.
        
        Args:
            candidate_news: List of (1 + K) candidate news dictionaries
            clicked_news: List of clicked news dictionaries
            clicked_news_mask: List of mask lists indicating real vs padded news
                
        Returns:
            Click probability tensor with shape [batch_size, 1 + K]
        """
        device = next(self.parameters()).device
        batch_size = candidate_news[0]["title"].shape[0]
        num_candidates = len(candidate_news)
        
        # Encode candidate news as documents (keep token-level embeddings)
        all_candidate_input_ids = torch.cat([x["title"][:, 0] for x in candidate_news], dim=0)
        # Prune tokens for documents
        all_candidate_input_ids = self._prune_tokens(all_candidate_input_ids, self.max_doc_tokens)
        all_candidate_texts = self._token_ids_to_text(all_candidate_input_ids)
        
        candidate_token_embeddings = self._encode_with_colbert(
            all_candidate_texts,
            is_query=False,
            device=device,
        )  # [batch_size * (1+K), num_doc_tokens, embedding_dim]
        
        # Reshape: [batch_size, 1+K, num_doc_tokens, embedding_dim]
        candidate_token_embeddings = candidate_token_embeddings.view(
            batch_size, num_candidates, -1, self.embedding_dim
        )
        
        # Encode clicked news as queries (keep token-level embeddings)
        all_clicked_input_ids = torch.cat([x["title"][:, 0] for x in clicked_news], dim=0)
        # Prune tokens for queries
        all_clicked_input_ids = self._prune_tokens(all_clicked_input_ids, self.max_query_tokens)
        all_clicked_texts = self._token_ids_to_text(all_clicked_input_ids)
        
        clicked_token_embeddings = self._encode_with_colbert(
            all_clicked_texts,
            is_query=True,
            device=device,
        )  # [batch_size * num_clicked, num_query_tokens, embedding_dim]
        
        num_clicked = len(clicked_news)
        # Reshape: [batch_size, num_clicked, num_query_tokens, embedding_dim]
        clicked_token_embeddings = clicked_token_embeddings.view(
            batch_size, num_clicked, -1, self.embedding_dim
        )
        
        # Apply mask to clicked news
        clicked_news_mask_array = np.array(clicked_news_mask, dtype=np.float32)
        clicked_news_mask_tensor = torch.from_numpy(clicked_news_mask_array).to(device).transpose(0, 1)
        # [batch_size, num_clicked, 1, 1]
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1).unsqueeze(-1)
        clicked_token_embeddings = clicked_token_embeddings * expanded_mask
        
        # Use clicked token embeddings directly as query for MaxSim
        # Flatten clicked tokens: [batch_size, num_clicked * num_query_tokens, embedding_dim]
        clicked_tokens_flat = clicked_token_embeddings.view(batch_size, -1, self.embedding_dim)
        
        # Apply mask: zero out padded clicked news tokens
        # [batch_size, num_clicked, num_query_tokens] -> [batch_size, num_clicked * num_query_tokens]
        mask_flat = clicked_news_mask_tensor.unsqueeze(-1).expand(-1, -1, clicked_token_embeddings.shape[2])
        mask_flat = mask_flat.reshape(batch_size, -1).unsqueeze(-1)
        clicked_tokens_flat = clicked_tokens_flat * mask_flat
        
        # Remove zero-padded tokens (where mask is 0) to avoid affecting MaxSim
        # For each batch, keep only non-zero tokens
        # This is more efficient than computing with zeros
        batch_query_embeddings = []
        for b in range(batch_size):
            mask_b = mask_flat[b, :, 0] > 0  # [num_total_tokens]
            if mask_b.any():
                query_b = clicked_tokens_flat[b][mask_b]  # [num_active_tokens, embedding_dim]
            else:
                # All masked - use a single zero vector
                query_b = torch.zeros(1, self.embedding_dim, device=device)
            batch_query_embeddings.append(query_b)
        
        # Compute MaxSim scores for each candidate
        scores = []
        for i in range(num_candidates):
            # [batch_size, num_doc_tokens, embedding_dim]
            doc_embeddings = candidate_token_embeddings[:, i, :, :]
            
            # Compute MaxSim for each batch item separately (due to variable query lengths)
            batch_scores = []
            for b in range(batch_size):
                query_emb = batch_query_embeddings[b].unsqueeze(0)  # [1, num_query_tokens, embedding_dim]
                doc_emb = doc_embeddings[b:b+1, :, :]  # [1, num_doc_tokens, embedding_dim]
                score = maxsim_score(query_emb, doc_emb)  # [1]
                batch_scores.append(score.squeeze(0))
            
            scores.append(torch.stack(batch_scores))  # [batch_size]
        
        # Stack: [batch_size, 1+K]
        click_probability = torch.stack(scores, dim=1)
        
        return click_probability
    
    def get_news_vector(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get news vector representation (mean pooled for compatibility).
        
        Args:
            news: Dictionary containing "title" tensor
            
        Returns:
            News vector with shape [batch_size, embedding_dim]
        """
        device = next(self.parameters()).device
        input_ids = news["title"][:, 0]
        input_ids = self._prune_tokens(input_ids, self.max_doc_tokens)
        texts = self._token_ids_to_text(input_ids)
        
        token_embeddings = self._encode_with_colbert(
            texts,
            is_query=False,
            device=device,
        )  # [batch_size, num_tokens, embedding_dim]
        
        # Mean pool for compatibility with existing evaluation code
        return token_embeddings.mean(dim=1)  # [batch_size, embedding_dim]
    
    def get_user_vector(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        """Get user vector representation.
        
        Args:
            clicked_news_vector: Tensor with shape [batch_size, num_clicked_news_a_user, embedding_dim]
            
        Returns:
            User vector with shape [batch_size, embedding_dim]
        """
        return self.user_encoder(clicked_news_vector)
    
    def get_prediction(
        self, news_vector: torch.Tensor, user_vector: torch.Tensor
    ) -> torch.Tensor:
        """Get click prediction for a single user and news candidates.
        
        Note: This method uses mean-pooled vectors for compatibility.
        For proper MaxSim, use forward() method.
        
        Args:
            news_vector: Tensor with shape [candidate_size, embedding_dim]
            user_vector: Tensor with shape [embedding_dim]
            
        Returns:
            Click probability tensor with shape [candidate_size]
        """
        # For compatibility, use dot product on pooled vectors
        # This is not ideal but needed for evaluation code
        user_expanded = user_vector.unsqueeze(0)  # [1, embedding_dim]
        news_expanded = news_vector.unsqueeze(0)  # [1, candidate_size, embedding_dim]
        
        # Dot product similarity
        scores = torch.bmm(news_expanded, user_expanded.unsqueeze(-1)).squeeze(-1).squeeze(0)
        return scores
