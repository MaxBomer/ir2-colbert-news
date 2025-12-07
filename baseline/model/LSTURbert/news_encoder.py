"""News encoder for LSTURbert model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from config import NRMSbertConfig
from model.general.attention.additive import AdditiveAttention


def _init_weights(module: nn.Module) -> None:
    """Initialize weights for custom layers."""
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
    """News encoder for LSTUR using BERT + CNN + additive attention."""
    
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
        
        # Category embedding
        self.category_embedding = nn.Embedding(
            config.num_categories,
            config.num_filters,
            padding_idx=0
        )
        
        # CNN for title encoding
        assert config.window_size >= 1 and config.window_size % 2 == 1
        self.title_cnn = nn.Conv2d(
            1,
            config.num_filters,
            (config.window_size, config.word_embedding_dim),
            padding=(int((config.window_size - 1) / 2), 0)
        )
        
        # Additive attention for title
        self.title_attention = AdditiveAttention(
            config.query_vector_dim,
            config.num_filters
        )
    
    def forward(self, news: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode news article.
        
        Args:
            news: Dictionary containing:
                - "category": [batch_size]
                - "subcategory": [batch_size]
                - "title": [batch_size, 2, num_words_title]
                
        Returns:
            News vector with shape [batch_size, num_filters * 3]
        """
        device = next(self.parameters()).device
        
        # Part 1: Category vector
        category_vector = self.category_embedding(news['category'].to(device))
        
        # Part 2: Subcategory vector
        subcategory_vector = self.category_embedding(news['subcategory'].to(device))
        
        # Part 3: Title vector with BERT + CNN + attention
        news_input = {
            "input_ids": news["title"][:, 0].to(device),
            "attention_mask": news["title"][:, 1].to(device)
        }
        
        # Get BERT token embeddings
        bert_output = self.bert(**news_input)[0]  # [batch_size, seq_len, dim]
        
        # Apply CNN to token embeddings
        # Reshape for CNN: [batch_size, 1, seq_len, dim]
        cnn_input = bert_output.unsqueeze(1)
        
        convoluted_title_vector = self.title_cnn(cnn_input.float()).squeeze(dim=3)
        
        # Apply activation and dropout
        activated_title_vector = F.dropout(
            F.relu(convoluted_title_vector),
            p=self.config.dropout_probability,
            training=self.training
        )
        
        # Apply additive attention
        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2)
        )
        
        # Concatenate: [batch_size, num_filters * 3]
        news_vector = torch.cat(
            [category_vector, subcategory_vector, weighted_title_vector],
            dim=1
        )
        
        return news_vector

