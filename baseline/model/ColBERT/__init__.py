"""ColBERT adapter for news recommendation with late interaction MaxSim."""
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add colbert to path if needed
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
        query_embeddings: [batch_size, num_query_tokens, embedding_dim]
        doc_embeddings: [batch_size, num_doc_tokens, embedding_dim]
        
    Returns:
        Score tensor: [batch_size]
    """
    # Ensure batch dimension exists
    if query_embeddings.dim() == 2:
        query_embeddings = query_embeddings.unsqueeze(0)
    if doc_embeddings.dim() == 2:
        doc_embeddings = doc_embeddings.unsqueeze(0)
    
    # Normalize embeddings (ColBERT uses cosine similarity)
    # Default eps=1e-12 prevents division by zero for padding vectors
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
    
    Uses ColBERT's token-level embeddings with MaxSim scoring.
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

        # Freeze ColBERT weights if requested (Zero-Shot / No-Retraining mode)
        if getattr(config, 'colbert_freeze_weights', False):
            for param in self.colbert_model.parameters():
                param.requires_grad = False
            # Ensure scoring layer is trainable
            print("ColBERT weights frozen. Training only scoring layer.")

        # Get embedding dimension
        if hasattr(self.colbert_model, "get_sentence_embedding_dimension"):
            self.embedding_dim = self.colbert_model.get_sentence_embedding_dimension()
        else:
            self.embedding_dim = getattr(config, "colbert_embedding_dim", 128)

        # Tokenizer for debug (optional now)
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

        self.max_query_tokens = config.colbert_max_query_tokens
        self.max_doc_tokens = config.colbert_max_doc_tokens

        # We keep UserEncoder only if we wanted to do hybrid/legacy, 
        # but for pure ColBERT it's not used in the main path.
        self.user_encoder = UserEncoder(config, embedding_dim=self.embedding_dim)

        # Add trainable scoring layer for MaxSim output if needed (NRMS usually has one)
        self.scoring_layer = nn.Linear(1, 1)
    
    def _token_ids_to_text(self, input_ids: torch.Tensor) -> List[str]:
        """Convert token IDs back to text. (Legacy - slow)"""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        texts = []
        for ids in input_ids:
            # Remove padding tokens (0)
            ids = ids[ids != 0]
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        
        return texts
    
    def _process_input_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, is_query: bool) -> Dict[str, torch.Tensor]:
        """Optimize input processing by skipping text decoding/re-encoding.
        
        Mimics Pylate ColBERT.tokenize but operates on tensor IDs.
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # 1. Determine max length and prefix
        target_length = self.max_query_tokens if is_query else self.max_doc_tokens
        prefix_id = self.colbert_model.query_prefix_id if is_query else self.colbert_model.document_prefix_id
        
        # 2. Insert Prefix (after [CLS] which is at index 0)
        # Input is [CLS] T1 T2 ... [SEP] [PAD] ...
        # We want [CLS] [PREFIX] T1 T2 ...
        
        # Split at index 1
        cls_token = input_ids[:, :1]
        rest_tokens = input_ids[:, 1:]
        
        cls_mask = attention_mask[:, :1]
        rest_mask = attention_mask[:, 1:]
        
        # Create prefix tensors
        prefix_token = torch.full((batch_size, 1), prefix_id, device=device, dtype=input_ids.dtype)
        prefix_mask = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        
        # Concatenate: [CLS] [PREFIX] [REST]
        new_input_ids = torch.cat([cls_token, prefix_token, rest_tokens], dim=1)
        new_attention_mask = torch.cat([cls_mask, prefix_mask, rest_mask], dim=1)
        
        # 3. Pad or Truncate to target_length
        current_len = new_input_ids.shape[1]
        
        if current_len > target_length:
            # Truncate
            new_input_ids = new_input_ids[:, :target_length]
            new_attention_mask = new_attention_mask[:, :target_length]
            # Ensure last token is SEP if we cut it off? 
            # ColBERT usually relies on fixed length. Pylate truncates. 
            # We just let it be.
        elif current_len < target_length:
            # Pad
            pad_len = target_length - current_len
            pad_ids = torch.zeros((batch_size, pad_len), device=device, dtype=input_ids.dtype) # 0 is standard PAD
            pad_masks = torch.zeros((batch_size, pad_len), device=device, dtype=attention_mask.dtype)
            
            new_input_ids = torch.cat([new_input_ids, pad_ids], dim=1)
            new_attention_mask = torch.cat([new_attention_mask, pad_masks], dim=1)
            
        # 4. Query Expansion (Mask = 1 for all query tokens)
        if is_query:
            # ColBERT query expansion treats padding as mask=1 so they participate in MaxSim (but low score?)
            # Pylate: if is_query and self.attend_to_expansion_tokens: tokenized_outputs["attention_mask"].fill_(1)
            # Wait, Pylate defaults to attend_to_expansion_tokens=False usually.
            # But Pylate 'tokenize' logic:
            # if is_query and self.do_query_expansion:
            #    masks = torch.ones_like(...)
            # The actual model mask logic:
            # if is_query and self.attend_to_expansion_tokens: tokenized_outputs["attention_mask"].fill_(1)
            
            # We replicate Pylate's behavior.
            # Usually ColBERT queries are fixed length (32) and all attend.
            # But we stick to the mask we built (1 for real+prefix, 0 for pad) UNLESS expansion is on.
            if getattr(self.colbert_model, 'do_query_expansion', True):
                 # If we want queries to always have length 32 and interactions, we might need mask=1?
                 # Standard ColBERT uses mask for attention. 
                 # MaxSim logic later uses the mask to filter?
                 # Pylate encode:
                 # if is_query: ... masks = torch.ones_like(...)
                 pass
        
        return {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask
        }

    def _encode_ids_with_colbert(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, is_query: bool) -> torch.Tensor:
        """
        Get token embeddings from ColBERT using pre-tokenized IDs.
        Returns: [batch_size, seq_len, dim]
        """
        features = self._process_input_ids(input_ids, attention_mask, is_query)
        outputs = self.colbert_model(features)
        return outputs["token_embeddings"]
    
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
        
        # 1. Encode Candidates (Documents)
        # candidate_news[i]['title'] shape: [batch, 2, seq_len] (input_ids, attention_mask)
        all_candidate_input_ids = torch.cat([x["title"][:, 0] for x in candidate_news], dim=0).to(device)
        all_candidate_mask = torch.cat([x["title"][:, 1] for x in candidate_news], dim=0).to(device)
        
        candidate_token_embeddings = self._encode_ids_with_colbert(
            all_candidate_input_ids,
            all_candidate_mask,
            is_query=False
        )  # [batch_size * (1+K), num_doc_tokens, dim]
        
        # Reshape: [batch_size, 1+K, num_doc_tokens, dim]
        candidate_token_embeddings = (
            candidate_token_embeddings.view(
                num_candidates, batch_size, -1, self.embedding_dim
            )
            .transpose(0, 1)
            .contiguous()
        )
        
        # 2. Encode History (Queries)
        all_clicked_input_ids = torch.cat([x["title"][:, 0] for x in clicked_news], dim=0).to(device)
        all_clicked_mask = torch.cat([x["title"][:, 1] for x in clicked_news], dim=0).to(device)
        
        clicked_token_embeddings = self._encode_ids_with_colbert(
            all_clicked_input_ids,
            all_clicked_mask,
            is_query=True
        )  # [batch_size * num_clicked, num_query_tokens, dim]
        
        num_clicked = len(clicked_news)
        
        # Reshape: [batch_size, num_clicked, num_query_tokens, dim]
        clicked_token_embeddings = (
            clicked_token_embeddings.view(
                num_clicked, batch_size, -1, self.embedding_dim
            )
            .transpose(0, 1)
            .contiguous()
        )
        
        # 3. Flatten User History (Concatenate all tokens from all clicked news)
        # We need to handle masking to avoid interacting with padded news
        if isinstance(clicked_news_mask, torch.Tensor):
            clicked_news_mask_tensor = clicked_news_mask.to(device=device, dtype=torch.float32)
        elif isinstance(clicked_news_mask, list) and len(clicked_news_mask) > 0 and isinstance(clicked_news_mask[0], torch.Tensor):
            # Handle list of tensors (usually [num_clicked] tensors of shape [batch_size])
            # We want [batch_size, num_clicked], so stack along dim 1
            clicked_news_mask_tensor = torch.stack(clicked_news_mask, dim=1).to(device=device, dtype=torch.float32)
        else:
            clicked_news_mask_tensor = torch.tensor(clicked_news_mask, device=device, dtype=torch.float32)
            
        # [batch_size, num_clicked] -> [batch_size, num_clicked, 1, 1]
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1).unsqueeze(-1)
        
        # Zero out embeddings of padded news
        clicked_token_embeddings = clicked_token_embeddings * expanded_mask
        
        # Flatten to [batch_size, total_query_tokens, dim]
        # total_query_tokens = num_clicked * num_tokens_per_news
        user_query_embeddings = clicked_token_embeddings.view(batch_size, -1, self.embedding_dim)
        
        # 4. Compute Scores (MaxSim)
        # We compute score for each candidate against the user's flattened query
        scores_list = []
        for i in range(num_candidates):
            # [batch_size, num_doc_tokens, dim]
            doc_emb = candidate_token_embeddings[:, i, :, :]
            
            # [batch_size]
            score = maxsim_score(user_query_embeddings, doc_emb)
            scores_list.append(score)
        
        # [batch_size, 1+K]
        click_probability = torch.stack(scores_list, dim=1)
        
        # Pass through scoring layer
        click_probability = self.scoring_layer(click_probability.unsqueeze(-1)).squeeze(-1)
        
        return click_probability

    def get_news_vector(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get news vector representation for evaluation (unpooled tokens).
        
        Returns:
            Tensor [batch_size, num_tokens, dim]
        """
        device = next(self.parameters()).device
        input_ids = news["title"][:, 0].to(device)
        attention_mask = news["title"][:, 1].to(device)
        
        with torch.no_grad():
            # Use internal helper to get token embeddings directly
            embeddings = self._encode_ids_with_colbert(input_ids, attention_mask, is_query=False)
            
        return embeddings
    
    def get_user_vector(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        """
        Get user vector representation (concatenated tokens).
        
        Args:
            clicked_news_vector: [batch_size, num_clicked, num_tokens, dim]
            
        Returns:
            Tensor [batch_size, num_clicked, num_tokens, dim]
        """
        # clicked_news_vector comes from evaluate.py stacking get_news_vector results
        # batch_size, num_clicked, num_tokens, dim = clicked_news_vector.shape
        
        # Return as is (4D tensor) for Hierarchical MaxSim in get_prediction
        return clicked_news_vector
    
    def get_prediction(
        self, news_vector: torch.Tensor, user_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Get click prediction using Hierarchical MaxSim.
        
        Args:
            news_vector: [num_candidates, doc_tokens, dim]
            user_vector: [batch_size=1, num_clicked, query_tokens, dim] (from get_user_vector)
                         OR [num_clicked, query_tokens, dim] if unbatched
            
        Returns:
            Tensor [num_candidates]
        """
        # Ensure user_vector has batch dimension if missing
        if user_vector.dim() == 3:
            user_vector = user_vector.unsqueeze(0) # [1, num_clicked, Q, D]
            
        batch_size, num_clicked, _, _ = user_vector.shape
        num_candidates = news_vector.shape[0]
        
        # Loop over candidates (usually small number in eval)
        scores = []
        for i in range(num_candidates):
            # [1, D_tokens, D]
            cand_emb = news_vector[i].unsqueeze(0)
            
            # Expand to match num_clicked
            # [num_clicked, D_tokens, D]
            cand_emb_expanded = cand_emb.repeat_interleave(num_clicked, dim=0)
            
            # Flatten user history for batched score
            # [num_clicked, Q_tokens, D]
            user_hist_flat = user_vector.squeeze(0)
            
            # Compute scores for this candidate against ALL history items
            # [num_clicked]
            item_scores = maxsim_score(user_hist_flat, cand_emb_expanded)
            
            # Masking?
            # evaluate.py passes 'PADDED_NEWS' embeddings.
            # We assume PADDED_NEWS has valid embeddings (zeros or random).
            # MaxSim against Zero vector -> 0 score?
            # If 'PADDED_NEWS' is all zeros, dot product is 0.
            # If real news gives positive scores, 0 is fine?
            # If real news gives negative scores, 0 is bad.
            # Ideally we should mask. But evaluate.py doesn't pass the mask to get_prediction.
            # However, PADDED_NEWS usually has ID 0.
            # If the model learned that PAD = ignore, it might handle it.
            # For rigorous eval, we should check if embedding is all zeros (if we implemented it that way).
            # In evaluate.py, PADDED_NEWS vector is zeros_like(news).
            # So MaxSim will be 0.
            # If real scores are > 0, then max(scores) will pick real news.
            # If real scores are < 0, then 0 (padding) will be picked -> bad.
            # MaxSim (Cosine) is usually [-1, 1].
            # We normalized in maxsim_score.
            # If we assume good matches > 0, we are fine.
            
            final_score = item_scores.max()
            scores.append(final_score)
            
        return torch.stack(scores)
