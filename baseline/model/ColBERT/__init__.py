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

        # Get embedding dimension
        if hasattr(self.colbert_model, "get_sentence_embedding_dimension"):
            self.embedding_dim = self.colbert_model.get_sentence_embedding_dimension()
        else:
            self.embedding_dim = getattr(config, "colbert_embedding_dim", 128)

        # Tokenizer for turning IDs into text
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

        self.max_query_tokens = config.colbert_max_query_tokens
        self.max_doc_tokens = config.colbert_max_doc_tokens

        # We keep UserEncoder only if we wanted to do hybrid/legacy, 
        # but for pure ColBERT it's not used in the main path.
        # Keeping it initialized to avoid breaking any implicit dependencies if any.
        self.user_encoder = UserEncoder(config, embedding_dim=self.embedding_dim)

        # Add trainable scoring layer for MaxSim output if needed (NRMS usually has one)
        self.scoring_layer = nn.Linear(1, 1)
    
    def _token_ids_to_text(self, input_ids: torch.Tensor) -> List[str]:
        """Convert token IDs back to text."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        texts = []
        for ids in input_ids:
            # Remove padding tokens (0)
            ids = ids[ids != 0]
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        
        return texts
    
    def _encode_with_colbert(self, texts: List[str], is_query: bool, device: torch.device) -> torch.Tensor:
        """
        Get token embeddings from ColBERT.
        Returns: [batch_size, seq_len, dim]
        """
        # Pylate tokenize handles markers and padding
        features = self.colbert_model.tokenize(texts, is_query=is_query)
        features = {k: v.to(device) for k, v in features.items()}
        
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
        all_candidate_input_ids = torch.cat([x["title"][:, 0] for x in candidate_news], dim=0)
        all_candidate_texts = self._token_ids_to_text(all_candidate_input_ids)
        
        candidate_token_embeddings = self._encode_with_colbert(
            all_candidate_texts,
            is_query=False,
            device=device,
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
        all_clicked_input_ids = torch.cat([x["title"][:, 0] for x in clicked_news], dim=0)
        all_clicked_texts = self._token_ids_to_text(all_clicked_input_ids)
        
        clicked_token_embeddings = self._encode_with_colbert(
            all_clicked_texts,
            is_query=True,
            device=device,
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
        else:
            clicked_news_mask_tensor = torch.tensor(clicked_news_mask, device=device, dtype=torch.float32)
            
        # [batch_size, num_clicked] -> [batch_size, num_clicked, 1, 1]
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1).unsqueeze(-1)
        
        # Zero out embeddings of padded news
        clicked_token_embeddings = clicked_token_embeddings * clicked_news_mask_tensor
        
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
        input_ids = news["title"][:, 0]
        texts = self._token_ids_to_text(input_ids)
        
        with torch.no_grad():
            # Use internal helper to get token embeddings directly
            embeddings = self._encode_with_colbert(texts, is_query=False, device=device)
            
        return embeddings
    
    def get_user_vector(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        """
        Get user vector representation (concatenated tokens).
        
        Args:
            clicked_news_vector: [batch_size, num_clicked, num_tokens, dim]
            
        Returns:
            Tensor [batch_size, num_clicked * num_tokens, dim]
        """
        # clicked_news_vector comes from evaluate.py stacking get_news_vector results
        batch_size, num_clicked, num_tokens, dim = clicked_news_vector.shape
        
        # Flatten the clicked news and tokens
        return clicked_news_vector.view(batch_size, -1, dim)
    
    def get_prediction(
        self, news_vector: torch.Tensor, user_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Get click prediction using MaxSim.
        
        Args:
            news_vector: [num_candidates, doc_tokens, dim]
            user_vector: [total_user_tokens, dim]
            
        Returns:
            Tensor [num_candidates]
        """
        # Expand user_vector to match num_candidates
        num_candidates = news_vector.shape[0]
        
        # [num_candidates, total_user_tokens, dim]
        user_vector_expanded = user_vector.unsqueeze(0).expand(num_candidates, -1, -1)
        
        # Compute MaxSim
        # returns [num_candidates]
        scores = maxsim_score(user_vector_expanded, news_vector)
        
        return scores
