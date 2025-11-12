"""Base model interface for news recommendation models."""
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn


class BaseNewsRecommendationModel(nn.Module, ABC):
    """Abstract base class for news recommendation models.
    
    All models must implement this interface to be compatible with the training
    and evaluation pipelines.
    """
    
    @abstractmethod
    def forward(
        self,
        candidate_news: List[Dict[str, torch.Tensor]],
        clicked_news: List[Dict[str, torch.Tensor]],
        clicked_news_mask: List[List[int]],
    ) -> torch.Tensor:
        """Forward pass for training.
        
        Args:
            candidate_news: List of (1 + K) candidate news dictionaries
            clicked_news: List of clicked news dictionaries
            clicked_news_mask: List of mask lists indicating real vs padded news
                
        Returns:
            Click probability tensor with shape [batch_size, 1 + K]
        """
        pass
    
    @abstractmethod
    def get_news_vector(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get news vector representation.
        
        Args:
            news: Dictionary containing news data
            
        Returns:
            News vector with shape [batch_size, embedding_dim]
        """
        pass
    
    @abstractmethod
    def get_user_vector(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        """Get user vector representation.
        
        Args:
            clicked_news_vector: Tensor with shape [batch_size, num_clicked_news_a_user, embedding_dim]
            
        Returns:
            User vector with shape [batch_size, embedding_dim]
        """
        pass
    
    @abstractmethod
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
        pass
