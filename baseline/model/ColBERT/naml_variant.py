"""ColBERT-NAML variant: Multi-view ColBERT with additive attention user encoder."""
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add colbert to path if needed
baseline_dir = Path(__file__).parent.parent.parent
project_root = baseline_dir.parent
colbert_path = project_root / 'colbert'
if str(colbert_path) not in sys.path:
    sys.path.insert(0, str(colbert_path))

from pylate import models as colbert_models

from config import NRMSbertConfig
from model.base import BaseNewsRecommendationModel
from model.general.attention.additive import AdditiveAttention
from model.ColBERT.__init__ import maxsim_score


class ColBERTNAMLModel(BaseNewsRecommendationModel):
    """ColBERT-NAML: Multi-view ColBERT with additive attention user encoder.
    
    Uses ColBERT for encoding title and abstract separately, combines with
    category/subcategory embeddings, and uses additive attention for user representation.
    """
    
    def __init__(self, config: NRMSbertConfig) -> None:
        """Initialize ColBERT-NAML model."""
        super().__init__()
        self.config = config
        
        # Initialize ColBERT model
        colbert_model_name = config.colbert_model_name or config.pretrained_model_name
        embedding_size = getattr(config, 'colbert_embedding_dim', None)
        
        try:
            self.colbert_model = colbert_models.ColBERT(
                model_name_or_path=colbert_model_name,
                device=None,
                embedding_size=embedding_size,
                query_length=config.colbert_max_query_tokens,
                document_length=config.colbert_max_doc_tokens,
            )
        except Exception as e:
            try:
                self.colbert_model = colbert_models.ColBERT(
                    model_name_or_path=colbert_model_name,
                    device=None,
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to initialize ColBERT model with '{colbert_model_name}'. "
                    f"Original error: {e}. Secondary error: {e2}"
                ) from e2
        
        # Freeze ColBERT weights if requested
        if getattr(config, 'colbert_freeze_weights', False):
            for param in self.colbert_model.parameters():
                param.requires_grad = False
        
        # Get embedding dimension
        if hasattr(self.colbert_model, "get_sentence_embedding_dimension"):
            self.embedding_dim = self.colbert_model.get_sentence_embedding_dimension()
        else:
            self.embedding_dim = getattr(config, "colbert_embedding_dim", 128)
        
        self.max_query_tokens = config.colbert_max_query_tokens
        self.max_doc_tokens = config.colbert_max_doc_tokens
        
        # Category embedding
        self.category_embedding = nn.Embedding(
            config.num_categories,
            config.category_embedding_dim,
            padding_idx=0
        )
        
        # Element encoders for category/subcategory
        self.category_encoder = nn.Linear(config.category_embedding_dim, config.num_filters)
        self.subcategory_encoder = nn.Linear(config.category_embedding_dim, config.num_filters)
        
        # Text projection layer for ColBERT embeddings -> num_filters
        self.text_proj = nn.Linear(self.embedding_dim, config.num_filters)
        
        # Final attention for combining views
        self.final_attention = AdditiveAttention(
            config.query_vector_dim,
            config.num_filters
        )
        
        # User encoder: additive attention
        self.user_encoder = AdditiveAttention(
            config.query_vector_dim,
            config.num_filters
        )
    
    def _process_input_ids(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, is_query: bool
    ) -> Dict[str, torch.Tensor]:
        """Process input IDs for ColBERT encoding."""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        target_length = self.max_query_tokens if is_query else self.max_doc_tokens
        prefix_id = self.colbert_model.query_prefix_id if is_query else self.colbert_model.document_prefix_id
        
        # Insert prefix after [CLS]
        cls_token = input_ids[:, :1]
        rest_tokens = input_ids[:, 1:]
        cls_mask = attention_mask[:, :1]
        rest_mask = attention_mask[:, 1:]
        
        prefix_token = torch.full((batch_size, 1), prefix_id, device=device, dtype=input_ids.dtype)
        prefix_mask = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        
        new_input_ids = torch.cat([cls_token, prefix_token, rest_tokens], dim=1)
        new_attention_mask = torch.cat([cls_mask, prefix_mask, rest_mask], dim=1)
        
        # Pad or truncate to target length
        current_len = new_input_ids.shape[1]
        if current_len > target_length:
            new_input_ids = new_input_ids[:, :target_length]
            new_attention_mask = new_attention_mask[:, :target_length]
        elif current_len < target_length:
            pad_len = target_length - current_len
            pad_ids = torch.zeros((batch_size, pad_len), device=device, dtype=input_ids.dtype)
            pad_masks = torch.zeros((batch_size, pad_len), device=device, dtype=attention_mask.dtype)
            new_input_ids = torch.cat([new_input_ids, pad_ids], dim=1)
            new_attention_mask = torch.cat([new_attention_mask, pad_masks], dim=1)
        
        return {"input_ids": new_input_ids, "attention_mask": new_attention_mask}
    
    def _encode_ids_with_colbert(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, is_query: bool
    ) -> torch.Tensor:
        """Get token embeddings from ColBERT."""
        features = self._process_input_ids(input_ids, attention_mask, is_query)
        outputs = self.colbert_model(features)
        return outputs["token_embeddings"]
    
    def _encode_text_view(
        self, text: torch.Tensor, is_query: bool
    ) -> torch.Tensor:
        """Encode a text view (title or abstract) using ColBERT.
        
        Args:
            text: [batch_size, 2, num_words] where [:, 0, :] are input_ids
            is_query: Whether this is a query (user history) or document (candidate)
            
        Returns:
            Token embeddings: [batch_size, num_tokens, embedding_dim]
        """
        device = next(self.parameters()).device
        input_ids = text[:, 0].to(device)
        attention_mask = text[:, 1].to(device)
        return self._encode_ids_with_colbert(input_ids, attention_mask, is_query)
    
    def _encode_news_multi_view(
        self, news: dict[str, torch.Tensor], is_query: bool = False
    ) -> torch.Tensor:
        """Encode news with multi-view learning.
        
        Args:
            news: Dictionary with title, abstract, category, subcategory
            is_query: Whether encoding for query (user history) or document (candidate)
            
        Returns:
            News vector: [batch_size, num_filters]
        """
        device = next(self.parameters()).device
        views = []
        
        # Encode text views (title, abstract)
        for text_name in ['title', 'abstract']:
            if text_name in news:
                text_embeddings = self._encode_text_view(news[text_name], is_query)
                # Pool text embeddings: mean pooling over tokens
                text_vector = text_embeddings.mean(dim=1)  # [batch_size, embedding_dim]
                # Project to num_filters using the registered layer
                views.append(F.relu(self.text_proj(text_vector)))
        
        # Encode element views (category, subcategory)
        if 'category' in news:
            cat_emb = self.category_embedding(news['category'].to(device))
            views.append(F.relu(self.category_encoder(cat_emb)))
        
        if 'subcategory' in news:
            subcat_emb = self.category_embedding(news['subcategory'].to(device))
            views.append(F.relu(self.subcategory_encoder(subcat_emb)))
        
        # Combine views with attention
        if len(views) == 1:
            return views[0]
        else:
            stacked = torch.stack(views, dim=1)  # [batch_size, num_views, num_filters]
            return self.final_attention(stacked)
    
    def forward(
        self,
        candidate_news: List[Dict[str, torch.Tensor]],
        clicked_news: List[Dict[str, torch.Tensor]],
        clicked_news_mask: List[List[int]],
        user: torch.Tensor | None = None,
        clicked_news_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using ColBERT MaxSim with multi-view learning.
        
        Args:
            candidate_news: List of (1 + K) candidate news dictionaries
            clicked_news: List of clicked news dictionaries
            clicked_news_mask: Mask indicating real vs padded news
            user: Not used (for compatibility)
            clicked_news_length: Not used (for compatibility)
                
        Returns:
            Click probability tensor: [batch_size, 1 + K]
        """
        device = next(self.parameters()).device
        batch_size = candidate_news[0]["title"].shape[0]
        num_candidates = len(candidate_news)
        num_clicked = len(clicked_news)
        
        # Encode candidates: [batch_size, 1+K, num_filters]
        candidate_vectors = torch.stack([
            self._encode_news_multi_view(x, is_query=False)
            for x in candidate_news
        ], dim=1)
        
        # Encode clicked news: [batch_size, num_clicked, num_filters]
        clicked_vectors = torch.stack([
            self._encode_news_multi_view(x, is_query=True)
            for x in clicked_news
        ], dim=1)
        
        # Apply mask
        clicked_news_mask_array = np.array(clicked_news_mask, dtype=np.float32)
        clicked_news_mask_tensor = torch.from_numpy(clicked_news_mask_array).to(device).transpose(0, 1)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_vectors = clicked_vectors * expanded_mask
        
        # Encode user with additive attention: [batch_size, num_filters]
        user_vector = self.user_encoder(clicked_vectors)
        
        # Compute dot product scores: [batch_size, 1+K]
        scores = torch.bmm(
            candidate_vectors,
            user_vector.unsqueeze(-1)
        ).squeeze(-1)
        
        return scores
    
    def get_news_vector(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get news vector for evaluation."""
        return self._encode_news_multi_view(news, is_query=False)
    
    def get_user_vector(
        self,
        clicked_news_vector: torch.Tensor,
        user: torch.Tensor | None = None,
        clicked_news_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get user vector by applying additive attention."""
        return self.user_encoder(clicked_news_vector)
    
    def get_prediction(
        self, news_vector: torch.Tensor, user_vector: torch.Tensor
    ) -> torch.Tensor:
        """Get click prediction using dot product."""
        if user_vector.dim() == 1:
            user_vector = user_vector.unsqueeze(0)
        if news_vector.dim() == 2:
            news_vector = news_vector.unsqueeze(0)
        
        return torch.bmm(
            news_vector,
            user_vector.unsqueeze(-1)
        ).squeeze(-1).squeeze(0)

