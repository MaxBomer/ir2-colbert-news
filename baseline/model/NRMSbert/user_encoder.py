"""User encoder for NRMSbert model."""
import torch
import torch.nn as nn

from config import NRMSbertConfig
from model.general.attention.additive import AdditiveAttention
from model.general.attention.multihead_self import MultiHeadSelfAttention


class UserEncoder(nn.Module):
    """Encoder for user representation using attention mechanisms."""
    
    def __init__(self, config: NRMSbertConfig, embedding_dim: int | None = None) -> None:
        """Initialize user encoder.
        
        Args:
            config: Configuration object
            embedding_dim: Embedding dimension (defaults to config.word_embedding_dim)
        """
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim if embedding_dim is not None else config.word_embedding_dim
        
        self.multihead_self_attention = MultiHeadSelfAttention(
            self.embedding_dim, config.num_attention_heads
        )
        self.additive_attention = AdditiveAttention(
            config.query_vector_dim, self.embedding_dim
        )
    
    def forward(self, user_vector: torch.Tensor) -> torch.Tensor:
        """Encode user representation from clicked news vectors.
        
        Args:
            user_vector: Tensor with shape [batch_size, num_clicked_news_a_user, embedding_dim]
            
        Returns:
            User vector with shape [batch_size, embedding_dim]
        """
        # Apply multi-head self-attention
        multihead_user_vector = self.multihead_self_attention(user_vector)
        
        # Apply additive attention to get final user vector
        final_user_vector = self.additive_attention(multihead_user_vector)
        
        return final_user_vector
