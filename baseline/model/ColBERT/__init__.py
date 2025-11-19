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
        colbert_model_name = (
            config.colbert_model_name
            if config.colbert_model_name is not None
            else config.pretrained_model_name
        )
        embedding_size = getattr(config, 'colbert_embedding_dim', None)
        try:
            self.colbert_model = colbert_models.ColBERT(
                model_name_or_path=colbert_model_name,
                device=None,
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

        # Get embedding dimension from the ColBERT sentence embeddings
        if hasattr(self.colbert_model, "get_sentence_embedding_dimension"):
            # Works with Pylate
            self.embedding_dim = self.colbert_model.get_sentence_embedding_dimension()
        else:
            # Default fallback
            self.embedding_dim = getattr(config, "colbert_embedding_dim", 128)

        # Tokenizer for turning IDs into text
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

        # Sets token limit
        self.max_query_tokens = config.colbert_max_query_tokens
        self.max_doc_tokens = config.colbert_max_doc_tokens

        # Embedding cache
        self.enable_caching = config.colbert_enable_caching
        self.cache_size = config.colbert_cache_size
        self._embedding_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        # Define user encoder
        self.user_encoder = UserEncoder(config, embedding_dim=self.embedding_dim)

        # Add trainable scoring layer for MaxSim output
        self.scoring_layer = nn.Linear(1, 1)
    
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
    
    def _encode_with_colbert(self, texts, is_query, device):
        """
        Trainable ColBERT embedding function.
        Compatible with ColBERT-v1, v2, and PyLaTe wrapper.
        
        Uses PyLaTe's internal tokenize and forward methods to ensure:
        1. [Q]/[D] markers are added
        2. Query padding uses [MASK] tokens (for query expansion)
        3. Linear projection layer is applied
        4. Normalization is handled if configured (though we normalize again in MaxSim)
        """
        # Use PyLaTe's tokenize to handle markers ([Q]/[D]) and query expansion ([MASK] padding)
        features = self.colbert_model.tokenize(texts, is_query=is_query)
        
        # Move all features to device
        features = {k: v.to(device) for k, v in features.items()}

        # Forward pass through SentenceTransformer modules (Transformer + Dense)
        # This ensures the projection layer is applied
        outputs = self.colbert_model(features)

        # Extract token embeddings
        token_embeddings = outputs["token_embeddings"]

        return token_embeddings
    
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
        # Convert to text (Pylate handles truncation/padding/markers internally)
        all_candidate_texts = self._token_ids_to_text(all_candidate_input_ids)
        
        candidate_token_embeddings = self._encode_with_colbert(
            all_candidate_texts,
            is_query=False,
            device=device,
        )  # [batch_size * (1+K), num_doc_tokens, embedding_dim]
        
        # Reshape: [batch_size, 1+K, num_doc_tokens, embedding_dim]
        candidate_token_embeddings = (
            candidate_token_embeddings.view(
                num_candidates, batch_size, -1, self.embedding_dim
            )
            .transpose(0, 1)
            .contiguous()
        )
        
        # Encode clicked news as queries (keep token-level embeddings)
        all_clicked_input_ids = torch.cat([x["title"][:, 0] for x in clicked_news], dim=0)
        # Convert to text (Pylate handles truncation/padding/markers internally)
        all_clicked_texts = self._token_ids_to_text(all_clicked_input_ids)
        
        clicked_token_embeddings = self._encode_with_colbert(
            all_clicked_texts,
            is_query=True,
            device=device,
        )  # [batch_size * num_clicked, num_query_tokens, embedding_dim]
        
        num_clicked = len(clicked_news)
        # Reshape: [batch_size, num_clicked, num_query_tokens, embedding_dim]
        clicked_token_embeddings = (
            clicked_token_embeddings.view(
                num_clicked, batch_size, -1, self.embedding_dim
            )
            .transpose(0, 1)
            .contiguous()
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
        click_probability = torch.stack(scores, dim=1)  # [B, 1+K]
        
        # Pass through trainable layer so gradients flow
        click_probability = self.scoring_layer(click_probability.unsqueeze(-1)).squeeze(-1)
        
        return click_probability
    
    def get_news_vector(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get news vector representation for evaluation.
    
        Uses mean pooling over token embeddings so dimensionality stays fixed.
        """
        device = next(self.parameters()).device
    
        # Convert token IDs back to text
        input_ids = news["title"][:, 0]
        # Pylate handles truncation internally
        texts = self._token_ids_to_text(input_ids)
    
        # Use no-grad
        with torch.no_grad():
            embeddings = self.colbert_model.encode(
                sentences=texts,
                batch_size=len(texts),
                convert_to_tensor=True,
                convert_to_numpy=False,
                is_query=False,
                device=str(device),
                show_progress_bar=False,
            )
    
        # Normalize to a list of tensors
        if isinstance(embeddings, list):
            tensors = []
            for emb in embeddings:
                if not isinstance(emb, torch.Tensor):
                    emb = torch.from_numpy(emb)
                emb = emb.to(device)
    
                # emb has shape [seq_len, dim]
                pooled = emb.mean(dim=0) 
                tensors.append(pooled)
    
            # news_vector has shape [batch, dim]
            news_vectors = torch.stack(tensors, dim=0)
    
        elif isinstance(embeddings, torch.Tensor):
            # If embeddings is [batch, seq_len, dim], we pool
            if embeddings.dim() == 3:
                news_vectors = embeddings.mean(dim=1)
            else:
                news_vectors = embeddings.unsqueeze(0)
    
        else:
            # Otherwise fall back on a numpy array
            emb = torch.from_numpy(embeddings).to(device)
            if emb.dim() == 3:
                news_vectors = emb.mean(dim=1)
            else:
                news_vectors = emb.unsqueeze(0)
    
        return news_vectors
    
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
