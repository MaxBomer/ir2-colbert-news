"""User encoder for NRMSbert model."""
import torch
import torch.nn as nn

from config import NRMSbertConfig
from model.general.attention.additive import AdditiveAttention
from model.general.attention.multihead_self import MultiHeadSelfAttention


class UserEncoder(nn.Module):
    """Encoder for user representation using attention mechanisms."""
    
    def __init__(self, config: NRMSbertConfig) -> None:
        """Initialize user encoder.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        self.multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads
        )
        self.additive_attention = AdditiveAttention(
            config.query_vector_dim, config.word_embedding_dim
        )
    
    def forward(self, user_vector: torch.Tensor) -> torch.Tensor:
        """Encode user representation from clicked news vectors.
        
        Args:
            user_vector: Tensor with shape [batch_size, num_clicked_news_a_user, word_embedding_dim]
            
        Returns:
            User vector with shape [batch_size, word_embedding_dim]
        """
        # Apply multi-head self-attention
        multihead_user_vector = self.multihead_self_attention(user_vector)
        
        # Apply additive attention to get final user vector
        final_user_vector = self.additive_attention(multihead_user_vector)
        
        return final_user_vector
