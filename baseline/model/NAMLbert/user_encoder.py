"""User encoder for NAMLbert model using additive attention."""
import torch
import torch.nn as nn

from config import NRMSbertConfig
from model.general.attention.additive import AdditiveAttention


class UserEncoder(nn.Module):
    """User encoder using additive attention over clicked news."""
    
    def __init__(self, config: NRMSbertConfig) -> None:
        """Initialize user encoder.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.additive_attention = AdditiveAttention(
            config.query_vector_dim,
            config.num_filters
        )
    
    def forward(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        """Encode user representation from clicked news vectors.
        
        Args:
            clicked_news_vector: Tensor with shape [batch_size, num_clicked_news_a_user, num_filters]
            
        Returns:
            User vector with shape [batch_size, num_filters]
        """
        return self.additive_attention(clicked_news_vector)







