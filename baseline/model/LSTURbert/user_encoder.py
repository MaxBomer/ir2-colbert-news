"""User encoder for LSTURbert model using GRU."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from config import NRMSbertConfig


class UserEncoder(nn.Module):
    """User encoder using GRU for long/short-term user modeling."""
    
    def __init__(self, config: NRMSbertConfig) -> None:
        """Initialize user encoder.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # GRU hidden size depends on method
        if config.long_short_term_method == 'ini':
            # Initialization: user embedding and GRU both use num_filters * 3
            gru_hidden_size = config.num_filters * 3
        else:
            # Concatenation: GRU hidden is num_filters * 1.5
            gru_hidden_size = int(config.num_filters * 1.5)
        
        self.gru = nn.GRU(
            config.num_filters * 3,  # Input: news vector size
            gru_hidden_size,
            batch_first=True
        )
    
    def forward(
        self,
        user: torch.Tensor,
        clicked_news_length: torch.Tensor,
        clicked_news_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Encode user representation.
        
        Args:
            user: User embedding vector
                - ini: [batch_size, num_filters * 3]
                - con: [batch_size, num_filters * 1.5]
            clicked_news_length: [batch_size] - actual length of clicked news
            clicked_news_vector: [batch_size, num_clicked_news_a_user, num_filters * 3]
            
        Returns:
            User vector with shape [batch_size, num_filters * 3]
        """
        # Ensure no zero lengths (causes issues with pack_padded_sequence)
        clicked_news_length = clicked_news_length.clone()
        clicked_news_length[clicked_news_length == 0] = 1
        
        if self.config.long_short_term_method == 'ini':
            # Initialization method: use user embedding as initial hidden state
            packed_clicked_news_vector = pack_padded_sequence(
                clicked_news_vector,
                clicked_news_length.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            
            # user: [batch_size, num_filters * 3] -> [1, batch_size, num_filters * 3]
            initial_hidden = user.unsqueeze(dim=0)
            
            _, last_hidden = self.gru(packed_clicked_news_vector, initial_hidden)
            
            # last_hidden: [1, batch_size, num_filters * 3]
            return last_hidden.squeeze(dim=0)
        else:
            # Concatenation method: concatenate GRU output with user embedding
            packed_clicked_news_vector = pack_padded_sequence(
                clicked_news_vector,
                clicked_news_length.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            
            _, last_hidden = self.gru(packed_clicked_news_vector)
            
            # last_hidden: [1, batch_size, num_filters * 1.5]
            # user: [batch_size, num_filters * 1.5]
            # Concatenate: [batch_size, num_filters * 3]
            return torch.cat([last_hidden.squeeze(dim=0), user], dim=1)




