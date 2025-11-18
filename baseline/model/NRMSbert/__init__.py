"""NRMSbert model implementation."""
import numpy as np
import torch
import torch.nn as nn

from config import NRMSbertConfig
from model.base import BaseNewsRecommendationModel
from model.general.click_predictor.dot_product import DotProductClickPredictor
from model.NRMSbert.news_encoder import NewsEncoder
from model.NRMSbert.user_encoder import UserEncoder


class NRMSbert(BaseNewsRecommendationModel):
    """Neural News Recommendation with Multi-Head Self-Attention using BERT.
    
    Takes 1 + K candidate news and a list of user clicked news,
    produces click probability for each candidate.
    """
    
    def __init__(self, config: NRMSbertConfig) -> None:
        """Initialize NRMSbert model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()
    
    def forward(
        self,
        candidate_news: list[dict[str, torch.Tensor]],
        clicked_news: list[dict[str, torch.Tensor]],
        clicked_news_mask: list[list[int]],
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            candidate_news: List of (1 + K) candidate news dictionaries
                Each dict contains "title" tensor with shape [batch_size, 2, num_words_title]
            clicked_news: List of clicked news dictionaries
                Each dict contains "title" tensor with shape [batch_size, 2, num_words_title]
            clicked_news_mask: List of mask lists indicating real vs padded news
                Each inner list has length num_clicked_news_a_user (0 for padding, 1 for real)
                
        Returns:
            Click probability tensor with shape [batch_size, 1 + K]
        """
        device = next(self.parameters()).device
        
        # Encode candidate news: [batch_size, 1 + K, word_embedding_dim]
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1
        )
        
        # Encode clicked news: [batch_size, num_clicked_news_a_user, word_embedding_dim]
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1
        )
        
        # Apply mask to clicked news vectors
        # Convert to numpy first to handle any mixed types from DataLoader batching
        clicked_news_mask_array = np.array(clicked_news_mask, dtype=np.float32)
        clicked_news_mask_tensor = torch.from_numpy(clicked_news_mask_array).to(device).transpose(0, 1)  # [batch_size, num_clicked_news_a_user]
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)  # [batch_size, num_clicked_news_a_user, 1]
        clicked_news_vector = clicked_news_vector * expanded_mask
        
        # Encode user: [batch_size, word_embedding_dim]
        user_vector = self.user_encoder(clicked_news_vector)
        
        # Predict click probability: [batch_size, 1 + K]
        click_probability = self.click_predictor(candidate_news_vector, user_vector)
        
        return click_probability
    
    def get_news_vector(self, news: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get news vector representation.
        
        Args:
            news: Dictionary containing "title" tensor with shape [batch_size, 2, num_words_title]
            
        Returns:
            News vector with shape [batch_size, word_embedding_dim]
        """
        return self.news_encoder(news)
    
    def get_user_vector(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        """Get user vector representation.
        
        Args:
            clicked_news_vector: Tensor with shape [batch_size, num_clicked_news_a_user, word_embedding_dim]
            
        Returns:
            User vector with shape [batch_size, word_embedding_dim]
        """
        return self.user_encoder(clicked_news_vector)
    
    def get_prediction(
        self, news_vector: torch.Tensor, user_vector: torch.Tensor
    ) -> torch.Tensor:
        """Get click prediction for a single user and news candidates.
        
        Args:
            news_vector: Tensor with shape [candidate_size, word_embedding_dim]
            user_vector: Tensor with shape [word_embedding_dim]
            
        Returns:
            Click probability tensor with shape [candidate_size]
        """
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)
        ).squeeze(dim=0)
