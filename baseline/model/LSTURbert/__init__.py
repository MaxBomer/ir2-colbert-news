"""LSTURbert model implementation with long/short-term user modeling."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NRMSbertConfig
from model.base import BaseNewsRecommendationModel
from model.general.click_predictor.dot_product import DotProductClickPredictor
from model.LSTURbert.news_encoder import NewsEncoder
from model.LSTURbert.user_encoder import UserEncoder


class LSTURbert(BaseNewsRecommendationModel):
    """Long and Short-Term User Representation using BERT.
    
    Uses GRU for short-term user modeling and user embeddings for long-term modeling.
    """
    
    def __init__(self, config: NRMSbertConfig) -> None:
        """Initialize LSTURbert model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()
        
        # User embedding
        if config.long_short_term_method == 'ini':
            user_embedding_dim = config.num_filters * 3
        else:
            user_embedding_dim = int(config.num_filters * 1.5)
        
        self.user_embedding = nn.Embedding(
            config.num_users,
            user_embedding_dim,
            padding_idx=0
        )
    
    def forward(
        self,
        candidate_news: list[dict[str, torch.Tensor]],
        clicked_news: list[dict[str, torch.Tensor]],
        clicked_news_mask: list[list[int]],
        user: torch.Tensor | None = None,
        clicked_news_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            candidate_news: List of (1 + K) candidate news dictionaries
            clicked_news: List of clicked news dictionaries
            clicked_news_mask: List of mask lists indicating real vs padded news
            user: User IDs with shape [batch_size] (required for LSTUR)
            clicked_news_length: Actual clicked news length [batch_size] (required for LSTUR)
                
        Returns:
            Click probability tensor with shape [batch_size, 1 + K]
        """
        device = next(self.parameters()).device
        
        if user is None or clicked_news_length is None:
            raise ValueError("LSTUR requires 'user' and 'clicked_news_length' arguments")
        
        # Encode candidate news: [batch_size, 1 + K, num_filters * 3]
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1
        )
        
        # Get user embedding
        if self.config.long_short_term_method == 'ini':
            # Initialization: apply dropout to user embedding
            user_emb = F.dropout(
                self.user_embedding(user.to(device)),
                p=self.config.masking_probability,
                training=self.training
            )
        else:
            # Concatenation: no dropout
            user_emb = self.user_embedding(user.to(device))
        
        # Encode clicked news: [batch_size, num_clicked_news_a_user, num_filters * 3]
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1
        )
        
        # Apply mask to clicked news vectors
        clicked_news_mask_array = np.array(clicked_news_mask, dtype=np.float32)
        clicked_news_mask_tensor = torch.from_numpy(clicked_news_mask_array).to(device).transpose(0, 1)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_news_vector = clicked_news_vector * expanded_mask
        
        # Encode user: [batch_size, num_filters * 3]
        user_vector = self.user_encoder(user_emb, clicked_news_length.to(device), clicked_news_vector)
        
        # Predict click probability: [batch_size, 1 + K]
        click_probability = self.click_predictor(candidate_news_vector, user_vector)
        
        return click_probability
    
    def get_news_vector(self, news: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get news vector representation.
        
        Args:
            news: Dictionary containing news attributes
            
        Returns:
            News vector with shape [batch_size, num_filters * 3]
        """
        return self.news_encoder(news)
    
    def get_user_vector(
        self,
        clicked_news_vector: torch.Tensor,
        user: torch.Tensor | None = None,
        clicked_news_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get user vector representation.
        
        Args:
            clicked_news_vector: Tensor with shape [batch_size, num_clicked_news_a_user, num_filters * 3]
            user: User IDs with shape [batch_size]
            clicked_news_length: Actual clicked news length [batch_size]
            
        Returns:
            User vector with shape [batch_size, num_filters * 3]
        """
        if user is None or clicked_news_length is None:
            raise ValueError("LSTUR requires 'user' and 'clicked_news_length' arguments")
        
        device = next(self.parameters()).device
        user_emb = self.user_embedding(user.to(device))
        return self.user_encoder(user_emb, clicked_news_length.to(device), clicked_news_vector)
    
    def get_prediction(
        self, news_vector: torch.Tensor, user_vector: torch.Tensor
    ) -> torch.Tensor:
        """Get click prediction for a single user and news candidates.
        
        Args:
            news_vector: Tensor with shape [candidate_size, num_filters * 3]
            user_vector: Tensor with shape [num_filters * 3]
            
        Returns:
            Click probability tensor with shape [candidate_size]
        """
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)
        ).squeeze(dim=0)

