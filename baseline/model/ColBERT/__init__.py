"""ColBERT adapter for news recommendation."""
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

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
from model.general.click_predictor.dot_product import DotProductClickPredictor
from model.NRMSbert.user_encoder import UserEncoder


class ColBERTNewsRecommendationModel(BaseNewsRecommendationModel):
    """ColBERT adapter for news recommendation.
    
    Wraps ColBERT model to work with the news recommendation pipeline.
    Converts tokenized BERT inputs to text and uses ColBERT for encoding.
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
        # Use embedding_size from config if specified, otherwise let ColBERT use default (128)
        embedding_size = getattr(config, 'colbert_embedding_dim', None)
        try:
            self.colbert_model = colbert_models.ColBERT(
                model_name_or_path=colbert_model_name,
                device=None,  # Will be set when moved to device
                embedding_size=embedding_size,  # Specify embedding size if provided
            )
        except Exception as e:
            # If ColBERT initialization fails, try without embedding_size
            # This might happen if the model structure doesn't support it
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
        # Use config value if specified, otherwise default to 128 (ColBERT default)
        # or word_embedding_dim from config
        if embedding_size is not None:
            self.embedding_dim = embedding_size
        else:
            # Default to 128 (ColBERT's default embedding_size)
            # Check explicitly for None since getattr returns None if attribute exists but is None
            self.embedding_dim = config.colbert_embedding_dim if config.colbert_embedding_dim is not None else 128
        
        # Initialize tokenizer for converting token IDs back to text
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
        
        # User encoder (same as NRMSbert, but with ColBERT embedding dim)
        self.user_encoder = UserEncoder(config, embedding_dim=self.embedding_dim)
        
        # Click predictor
        self.click_predictor = DotProductClickPredictor()
    
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
            # Remove padding tokens (0) and special tokens
            ids = ids[ids != 0]
            # Decode
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        
        return texts
    
    def _encode_news_with_colbert(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode news using ColBERT.
        
        Args:
            news: Dictionary containing "title" tensor with shape [batch_size, 2, num_words_title]
                where [:, 0, :] are input_ids and [:, 1, :] are attention_mask
                
        Returns:
            News vector with shape [batch_size, embedding_dim]
        """
        device = next(self.parameters()).device
        
        # Extract input_ids
        input_ids = news["title"][:, 0]  # [batch_size, seq_len]
        
        # Convert token IDs to text
        texts = self._token_ids_to_text(input_ids)
        
        # Encode with ColBERT (as documents, not queries)
        # ColBERT returns multi-vector embeddings (list of tensors)
        embeddings = self.colbert_model.encode(
            sentences=texts,
            batch_size=len(texts),
            convert_to_tensor=True,
            convert_to_numpy=False,
            is_query=False,  # Encode as documents
            device=str(device),
            show_progress_bar=False,  # Disable progress bar during training
        )
        
        # ColBERT returns multi-vector embeddings
        # We need to pool them to get a single vector
        # If embeddings is a list, each element is [seq_len, embedding_dim]
        # If embeddings is a tensor, it might be [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim]
        if isinstance(embeddings, list):
            # Pool each multi-vector embedding to a single vector
            pooled_embeddings = []
            for emb in embeddings:
                if isinstance(emb, torch.Tensor):
                    # Average pool over sequence dimension
                    pooled = emb.mean(dim=0)  # [embedding_dim]
                else:
                    # numpy array
                    pooled = torch.from_numpy(emb.mean(axis=0))
                pooled_embeddings.append(pooled)
            embeddings = torch.stack(pooled_embeddings, dim=0)  # [batch_size, embedding_dim]
        elif isinstance(embeddings, torch.Tensor):
            if embeddings.dim() == 3:
                # [batch_size, seq_len, embedding_dim]
                embeddings = embeddings.mean(dim=1)  # [batch_size, embedding_dim]
            elif embeddings.dim() == 2:
                # [seq_len, embedding_dim] - single sample
                embeddings = embeddings.mean(dim=0, keepdim=True)  # [1, embedding_dim]
        
        # Ensure correct shape
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        return embeddings.to(device)
    
    def forward(
        self,
        candidate_news: List[Dict[str, torch.Tensor]],
        clicked_news: List[Dict[str, torch.Tensor]],
        clicked_news_mask: List[List[int]],
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            candidate_news: List of (1 + K) candidate news dictionaries
            clicked_news: List of clicked news dictionaries
            clicked_news_mask: List of mask lists indicating real vs padded news
                
        Returns:
            Click probability tensor with shape [batch_size, 1 + K]
        """
        device = next(self.parameters()).device
        
        # Batch encode all candidate news at once
        # Stack all candidate news input_ids: [batch_size * (1+K), seq_len]
        all_candidate_input_ids = torch.cat([x["title"][:, 0] for x in candidate_news], dim=0)
        all_candidate_texts = self._token_ids_to_text(all_candidate_input_ids)
        
        # Encode all candidates in one batch
        batch_size = candidate_news[0]["title"].shape[0]
        num_candidates = len(candidate_news)
        
        candidate_embeddings = self.colbert_model.encode(
            sentences=all_candidate_texts,
            batch_size=len(all_candidate_texts),
            convert_to_tensor=True,
            convert_to_numpy=False,
            is_query=False,
            device=str(device),
            show_progress_bar=False,  # Disable progress bar during training
        )
        
        # Pool and reshape: [batch_size, 1+K, embedding_dim]
        if isinstance(candidate_embeddings, list):
            candidate_embeddings = torch.stack([emb.mean(dim=0) if isinstance(emb, torch.Tensor) else torch.from_numpy(emb.mean(axis=0)) for emb in candidate_embeddings], dim=0)
        elif candidate_embeddings.dim() == 3:
            candidate_embeddings = candidate_embeddings.mean(dim=1)
        elif candidate_embeddings.dim() == 2:
            candidate_embeddings = candidate_embeddings.mean(dim=0, keepdim=True) if candidate_embeddings.shape[0] == 1 else candidate_embeddings
        
        candidate_news_vector = candidate_embeddings.view(batch_size, num_candidates, -1).to(device)
        
        # Batch encode all clicked news at once
        all_clicked_input_ids = torch.cat([x["title"][:, 0] for x in clicked_news], dim=0)
        all_clicked_texts = self._token_ids_to_text(all_clicked_input_ids)
        
        clicked_embeddings = self.colbert_model.encode(
            sentences=all_clicked_texts,
            batch_size=len(all_clicked_texts),
            convert_to_tensor=True,
            convert_to_numpy=False,
            is_query=False,
            device=str(device),
            show_progress_bar=False,  # Disable progress bar during training
        )
        
        # Pool and reshape: [batch_size, num_clicked_news_a_user, embedding_dim]
        if isinstance(clicked_embeddings, list):
            clicked_embeddings = torch.stack([emb.mean(dim=0) if isinstance(emb, torch.Tensor) else torch.from_numpy(emb.mean(axis=0)) for emb in clicked_embeddings], dim=0)
        elif clicked_embeddings.dim() == 3:
            clicked_embeddings = clicked_embeddings.mean(dim=1)
        elif clicked_embeddings.dim() == 2:
            clicked_embeddings = clicked_embeddings.mean(dim=0, keepdim=True) if clicked_embeddings.shape[0] == 1 else clicked_embeddings
        
        num_clicked = len(clicked_news)
        clicked_news_vector = clicked_embeddings.view(batch_size, num_clicked, -1).to(device)
        
        # Apply mask to clicked news vectors
        clicked_news_mask_array = np.array(clicked_news_mask, dtype=np.float32)
        clicked_news_mask_tensor = torch.from_numpy(clicked_news_mask_array).to(device).transpose(0, 1)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_news_vector = clicked_news_vector * expanded_mask
        
        # Encode user: [batch_size, embedding_dim]
        user_vector = self.user_encoder(clicked_news_vector)
        
        # Predict click probability: [batch_size, 1 + K]
        click_probability = self.click_predictor(candidate_news_vector, user_vector)
        
        return click_probability
    
    def get_news_vector(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get news vector representation.
        
        Args:
            news: Dictionary containing "title" tensor
            
        Returns:
            News vector with shape [batch_size, embedding_dim]
        """
        return self._encode_news_with_colbert(news)
    
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
        
        Args:
            news_vector: Tensor with shape [candidate_size, embedding_dim]
            user_vector: Tensor with shape [embedding_dim]
            
        Returns:
            Click probability tensor with shape [candidate_size]
        """
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)
        ).squeeze(dim=0)
