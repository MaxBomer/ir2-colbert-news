"""News encoder for NRMSbert model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from config import NRMSbertConfig
from model.general.attention.additive import AdditiveAttention
from model.general.attention.multihead_self import MultiHeadSelfAttention


def _init_weights(module: nn.Module) -> None:
    """Initialize weights for custom layers.
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight.data)
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        if module.bias is not None:
            module.bias.data.zero_()


class NewsEncoder(nn.Module):
    """Encoder for news articles using BERT and attention mechanisms."""
    
    def __init__(self, config: NRMSbertConfig) -> None:
        """Initialize news encoder.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # Load BERT model
        bert = AutoModel.from_pretrained(config.pretrained_model_name)
        self.dim = bert.config.hidden_size
        self.bert = bert
        
        # Freeze all layers except the last `config.finetune_layers` layers
        num_layers = len(self.bert.encoder.layer)
        for i, layer in enumerate(self.bert.encoder.layer):
            should_finetune = i >= num_layers - config.finetune_layers
            for param in layer.parameters():
                param.requires_grad = should_finetune
        
        # Pooler for [CLS] token
        self.pooler = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(0.1),
            nn.LayerNorm(self.dim),
            nn.SiLU(),
        )
        self.pooler.apply(_init_weights)
        
        # Attention mechanisms
        self.multihead_self_attention = MultiHeadSelfAttention(
            self.dim, config.num_attention_heads
        )
        self.additive_attention = AdditiveAttention(
            config.query_vector_dim, self.dim
        )
    
    def forward(self, news: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode news article.
        
        Args:
            news: Dictionary containing tokenized title
                - "title": Tensor with shape [batch_size, 2, num_words_title]
                  where [:, 0, :] are input_ids and [:, 1, :] are attention_mask
                  
        Returns:
            News vector with shape [batch_size, word_embedding_dim]
        """
        device = next(self.parameters()).device
        
        # Extract input_ids and attention_mask from title tensor
        news_input = {
            "input_ids": news["title"][:, 0].to(device),
            "attention_mask": news["title"][:, 1].to(device)
        }
        
        # Get BERT embeddings
        bert_output = self.bert(**news_input)[0]  # All token embeddings
        news_vector = bert_output[:, 0]  # [CLS] token representation
        
        # Apply pooler
        news_vector = self.pooler(news_vector)
        
        # Apply multi-head self-attention
        multihead_news_vector = self.multihead_self_attention(news_vector)
        multihead_news_vector = F.dropout(
            multihead_news_vector,
            p=self.config.dropout_probability,
            training=self.training
        )
        
        # Apply additive attention
        final_news_vector = self.additive_attention(multihead_news_vector)
        
        return final_news_vector
