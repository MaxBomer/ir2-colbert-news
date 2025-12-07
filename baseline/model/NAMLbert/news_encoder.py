"""News encoder for NAMLbert model with multi-view learning."""
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


class TextEncoder(nn.Module):
    """Text encoder using BERT + CNN + additive attention."""
    
    def __init__(
        self,
        word_embedding_dim: int,
        num_filters: int,
        window_size: int,
        query_vector_dim: int,
        dropout_probability: float,
        config: NRMSbertConfig,
    ) -> None:
        """Initialize text encoder.
        
        Args:
            word_embedding_dim: BERT hidden size
            num_filters: Number of CNN filters
            window_size: CNN window size
            query_vector_dim: Query vector dimension for attention
            dropout_probability: Dropout probability
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.dropout_probability = dropout_probability
        
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
        
        # CNN for text encoding
        self.cnn = nn.Conv2d(
            1,
            num_filters,
            (window_size, word_embedding_dim),
            padding=(int((window_size - 1) / 2), 0)
        )
        
        # Additive attention
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)
    
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text using BERT + CNN + attention.
        
        Args:
            text: Tensor with shape [batch_size, 2, num_words]
                where [:, 0, :] are input_ids and [:, 1, :] are attention_mask
                
        Returns:
            Text vector with shape [batch_size, num_filters]
        """
        device = next(self.parameters()).device
        
        # Extract input_ids and attention_mask
        news_input = {
            "input_ids": text[:, 0].to(device),
            "attention_mask": text[:, 1].to(device)
        }
        
        # Get BERT embeddings
        bert_output = self.bert(**news_input)[0]  # All token embeddings [batch_size, seq_len, dim]
        
        # Use token embeddings (excluding [CLS] and [SEP] if present)
        # The CNN can handle variable sequence lengths due to padding
        batch_size, seq_len, dim = bert_output.shape
        
        # Use all token embeddings (or exclude special tokens)
        # For CNN, we want [batch_size, 1, seq_len, dim]
        # The CNN padding will handle the rest
        token_embeddings = bert_output  # [batch_size, seq_len, dim]
        
        # Reshape for CNN: [batch_size, 1, seq_len, dim]
        cnn_input = token_embeddings.unsqueeze(1)
        
        convoluted_text_vector = self.cnn(cnn_input.float()).squeeze(dim=3)
        
        # Apply activation and dropout
        activated_text_vector = F.dropout(
            F.relu(convoluted_text_vector),
            p=self.dropout_probability,
            training=self.training
        )
        
        # Apply additive attention: [batch_size, num_filters]
        text_vector = self.additive_attention(activated_text_vector.transpose(1, 2))
        
        return text_vector


class ElementEncoder(nn.Module):
    """Element encoder for category/subcategory embeddings."""
    
    def __init__(
        self,
        embedding: nn.Embedding,
        linear_input_dim: int,
        linear_output_dim: int,
    ) -> None:
        """Initialize element encoder.
        
        Args:
            embedding: Category embedding layer
            linear_input_dim: Input dimension
            linear_output_dim: Output dimension
        """
        super().__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)
    
    def forward(self, element: torch.Tensor) -> torch.Tensor:
        """Encode element.
        
        Args:
            element: Category indices with shape [batch_size]
            
        Returns:
            Element vector with shape [batch_size, linear_output_dim]
        """
        return F.relu(self.linear(self.embedding(element)))


class NewsEncoder(nn.Module):
    """Multi-view news encoder for NAMLbert."""
    
    def __init__(self, config: NRMSbertConfig) -> None:
        """Initialize news encoder.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # Text encoders for title and abstract
        text_encoders_candidates = ['title', 'abstract']
        self.text_encoders = nn.ModuleDict({
            name: TextEncoder(
                config.word_embedding_dim,
                config.num_filters,
                config.window_size,
                config.query_vector_dim,
                config.dropout_probability,
                config
            )
            for name in (set(config.dataset_attributes['news']) & set(text_encoders_candidates))
        })
        
        # Category embedding
        category_embedding = nn.Embedding(
            config.num_categories,
            config.category_embedding_dim,
            padding_idx=0
        )
        
        # Element encoders for category and subcategory
        element_encoders_candidates = ['category', 'subcategory']
        self.element_encoders = nn.ModuleDict({
            name: ElementEncoder(
                category_embedding,
                config.category_embedding_dim,
                config.num_filters
            )
            for name in (set(config.dataset_attributes['news']) & set(element_encoders_candidates))
        })
        
        # Final attention if multiple views
        if len(config.dataset_attributes['news']) > 1:
            self.final_attention = AdditiveAttention(
                config.query_vector_dim,
                config.num_filters
            )
    
    def forward(self, news: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode news article with multi-view learning.
        
        Args:
            news: Dictionary containing:
                - "title": [batch_size, 2, num_words_title]
                - "abstract": [batch_size, 2, num_words_abstract] (if available)
                - "category": [batch_size]
                - "subcategory": [batch_size]
                
        Returns:
            News vector with shape [batch_size, num_filters]
        """
        device = next(self.parameters()).device
        
        # Encode text views
        text_vectors = [
            encoder(news[name].to(device))
            for name, encoder in self.text_encoders.items()
        ]
        
        # Encode element views
        element_vectors = [
            encoder(news[name].to(device))
            for name, encoder in self.element_encoders.items()
        ]
        
        # Combine all vectors
        all_vectors = text_vectors + element_vectors
        
        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            # Stack and apply attention: [batch_size, num_views, num_filters]
            stacked = torch.stack(all_vectors, dim=1)
            final_news_vector = self.final_attention(stacked)
        
        return final_news_vector

