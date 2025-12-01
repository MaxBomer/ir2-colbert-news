"""ColBERT adapter for news recommendation with late interaction MaxSim.

Architecture modes (controlled by config flags):
- Default: Flatten user tokens → MaxSim (faithful to original ColBERT)
- +user_attention: Flatten → Self-Attention → MaxSim (David's suggestion)
- +position_embeddings: Add article position embeddings before attention
- +hierarchical_attention: Token-level + Article-level attention
"""
import sys
from pathlib import Path
from typing import Dict, List

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
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention


def maxsim_score(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Compute ColBERT MaxSim score between query and document token embeddings.
    
    Args:
        query_embeddings: [batch_size, num_query_tokens, embedding_dim]
        doc_embeddings: [batch_size, num_doc_tokens, embedding_dim]
        
    Returns:
        Score tensor: [batch_size]
    """
    if query_embeddings.dim() == 2:
        query_embeddings = query_embeddings.unsqueeze(0)
    if doc_embeddings.dim() == 2:
        doc_embeddings = doc_embeddings.unsqueeze(0)
    
    # Normalize embeddings (ColBERT uses cosine similarity)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)
    
    # Compute similarity matrix: [batch_size, num_query_tokens, num_doc_tokens]
    similarity_matrix = torch.bmm(query_embeddings, doc_embeddings.transpose(1, 2))
    
    # MaxSim: for each query token, take max similarity with any doc token
    max_similarities = similarity_matrix.max(dim=2)[0]
    
    # Sum over query tokens: [batch_size]
    return max_similarities.sum(dim=1)


class ColBERTNewsRecommendationModel(BaseNewsRecommendationModel):
    """ColBERT adapter for news recommendation with late interaction MaxSim.
    
    This follows the original ColBERT formula: S_q,d = Σ_i max_j (q_i · d_j)
    where the user's history tokens form the "query" and candidate tokens form the "document".
    
    Extensions:
    - colbert_user_attention: Cross-article self-attention before MaxSim
    - colbert_position_embeddings: Add article position embeddings
    - colbert_hierarchical_attention: Token + article level attention
    """
    
    def __init__(self, config: NRMSbertConfig) -> None:
        """Initialize ColBERT adapter."""
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
            print("ColBERT weights frozen.")

        # Get embedding dimension
        if hasattr(self.colbert_model, "get_sentence_embedding_dimension"):
            self.embedding_dim = self.colbert_model.get_sentence_embedding_dimension()
        else:
            self.embedding_dim = getattr(config, "colbert_embedding_dim", 128)

        self.max_query_tokens = config.colbert_max_query_tokens
        self.max_doc_tokens = config.colbert_max_doc_tokens
        self.num_clicked = config.num_clicked_news_a_user
        
        # Config flags for extensions
        self.use_user_attention = getattr(config, 'colbert_user_attention', False)
        self.use_position_embeddings = getattr(config, 'colbert_position_embeddings', False)
        self.use_hierarchical_attention = getattr(config, 'colbert_hierarchical_attention', False)
        num_heads = getattr(config, 'colbert_attention_heads', 8)
        
        # Extension 1: Cross-article self-attention (David's suggestion)
        if self.use_user_attention or self.use_hierarchical_attention:
            self.token_attention = MultiHeadSelfAttention(self.embedding_dim, num_heads)
            print(f"Token-level attention enabled: {num_heads} heads")
        
        # Extension 2: Article position embeddings
        if self.use_position_embeddings:
            self.article_position_embeddings = nn.Embedding(self.num_clicked, self.embedding_dim)
            print(f"Article position embeddings enabled: {self.num_clicked} positions")
        
        # Extension 3: Hierarchical attention (article-level on top of token-level)
        if self.use_hierarchical_attention:
            self.article_attention = MultiHeadSelfAttention(self.embedding_dim, num_heads)
            self.article_pooling = AdditiveAttention(config.query_vector_dim, self.embedding_dim)
            print(f"Hierarchical attention enabled: token + article level")
            
    def _process_input_ids(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, is_query: bool
    ) -> Dict[str, torch.Tensor]:
        """Process input IDs for ColBERT encoding (adds prefix tokens)."""
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
        """Get token embeddings from ColBERT. Returns: [batch_size, seq_len, dim]"""
        features = self._process_input_ids(input_ids, attention_mask, is_query)
        outputs = self.colbert_model(features)
        return outputs["token_embeddings"]
    
    def _build_token_mask(
        self, article_mask: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Expand article-level mask to token-level mask.
        
        Args:
            article_mask: [batch_size, num_articles] where 1=real, 0=pad
            num_tokens: number of tokens per article
            
        Returns:
            Token mask: [batch_size, num_articles * num_tokens]
        """
        # [batch_size, num_articles, num_tokens]
        expanded = article_mask.unsqueeze(-1).expand(-1, -1, num_tokens)
        # [batch_size, total_tokens]
        return expanded.reshape(article_mask.shape[0], -1)
    
    def _add_position_embeddings(
        self, embeddings: torch.Tensor, num_articles: int, num_tokens: int
    ) -> torch.Tensor:
        """Add article position embeddings to token embeddings.
        
        Args:
            embeddings: [batch_size, num_articles * num_tokens, dim]
            num_articles: number of articles
            num_tokens: tokens per article
            
        Returns:
            Embeddings with position info: [batch_size, num_articles * num_tokens, dim]
        """
        if not self.use_position_embeddings:
            return embeddings
            
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # Create position indices: [0,0,0,...,1,1,1,...,2,2,2,...]
        positions = torch.arange(num_articles, device=device)
        positions = positions.unsqueeze(1).expand(-1, num_tokens).reshape(-1)
        
        # Get position embeddings: [total_tokens, dim]
        pos_emb = self.article_position_embeddings(positions)
        
        # Add to embeddings: [batch_size, total_tokens, dim]
        return embeddings + pos_emb.unsqueeze(0)
    
    def _apply_attention(
        self, 
        embeddings: torch.Tensor, 
        article_mask: torch.Tensor,
        num_articles: int,
        num_tokens: int
    ) -> torch.Tensor:
        """Apply attention over user tokens (with optional hierarchical attention).
        
        Args:
            embeddings: [batch_size, total_tokens, dim]
            article_mask: [batch_size, num_articles] where 1=real, 0=pad
            num_articles: number of articles
            num_tokens: tokens per article
            
        Returns:
            Contextualized embeddings: [batch_size, total_tokens, dim]
        """
        if not (self.use_user_attention or self.use_hierarchical_attention):
            return embeddings
        
        # Skip attention during inference to avoid OOM
        # (evaluation uses doc_tokens=128 vs training query_tokens=32, causing 16x more memory)
        if not self.training:
            return embeddings
            
        batch_size = embeddings.shape[0]
        
        # Build length tensor for attention masking
        # Count real tokens per sample
        token_mask = self._build_token_mask(article_mask, num_tokens)
        lengths = token_mask.sum(dim=1).long()
        
        # Apply token-level self-attention
        contextualized = self.token_attention(embeddings, length=lengths)
        
        if not self.use_hierarchical_attention:
            return contextualized
        
        # Hierarchical: also apply article-level attention
        # Reshape to [batch_size, num_articles, num_tokens, dim]
        reshaped = contextualized.view(batch_size, num_articles, num_tokens, self.embedding_dim)
        
        # Pool each article to single vector: [batch_size, num_articles, dim]
        article_vectors = reshaped.mean(dim=2)  # Simple mean pooling
        
        # Apply article-level attention
        article_lengths = article_mask.sum(dim=1).long()
        article_contextualized = self.article_attention(article_vectors, length=article_lengths)
        
        # Expand back to token level and combine
        # [batch_size, num_articles, 1, dim] -> [batch_size, num_articles, num_tokens, dim]
        article_context = article_contextualized.unsqueeze(2).expand(-1, -1, num_tokens, -1)
        
        # Residual connection: token embeddings + article context
        combined = reshaped + article_context
        
        # Flatten back: [batch_size, total_tokens, dim]
        return combined.view(batch_size, -1, self.embedding_dim)
    
    def forward(
        self,
        candidate_news: List[Dict[str, torch.Tensor]],
        clicked_news: List[Dict[str, torch.Tensor]],
        clicked_news_mask: List[List[int]],
    ) -> torch.Tensor:
        """Forward pass using ColBERT late interaction MaxSim.
        
        Args:
            candidate_news: List of (1 + K) candidate news dictionaries
            clicked_news: List of clicked news dictionaries  
            clicked_news_mask: Mask indicating real vs padded news
                
        Returns:
            Click probability tensor: [batch_size, 1 + K]
        """
        device = next(self.parameters()).device
        batch_size = candidate_news[0]["title"].shape[0]
        num_candidates = len(candidate_news)
        num_clicked = len(clicked_news)
        
        # 1. Encode Candidates (Documents)
        all_candidate_input_ids = torch.cat([x["title"][:, 0] for x in candidate_news], dim=0).to(device)
        all_candidate_mask = torch.cat([x["title"][:, 1] for x in candidate_news], dim=0).to(device)
        
        candidate_token_embeddings = self._encode_ids_with_colbert(
            all_candidate_input_ids, all_candidate_mask, is_query=False
        )
        
        # Reshape: [batch_size, 1+K, num_doc_tokens, dim]
        candidate_token_embeddings = (
            candidate_token_embeddings
            .view(num_candidates, batch_size, -1, self.embedding_dim)
            .transpose(0, 1)
            .contiguous()
        )
        
        # 2. Encode History (Queries)
        all_clicked_input_ids = torch.cat([x["title"][:, 0] for x in clicked_news], dim=0).to(device)
        all_clicked_mask = torch.cat([x["title"][:, 1] for x in clicked_news], dim=0).to(device)
        
        clicked_token_embeddings = self._encode_ids_with_colbert(
            all_clicked_input_ids, all_clicked_mask, is_query=True
        )
        
        num_query_tokens = clicked_token_embeddings.shape[1]
        
        # Reshape: [batch_size, num_clicked, num_query_tokens, dim]
        clicked_token_embeddings = (
            clicked_token_embeddings
            .view(num_clicked, batch_size, -1, self.embedding_dim)
            .transpose(0, 1)
            .contiguous()
        )
        
        # 3. Build article mask
        if isinstance(clicked_news_mask, torch.Tensor):
            article_mask = clicked_news_mask.to(device=device, dtype=torch.float32)
        elif isinstance(clicked_news_mask, list) and len(clicked_news_mask) > 0 and isinstance(clicked_news_mask[0], torch.Tensor):
            article_mask = torch.stack(clicked_news_mask, dim=1).to(device=device, dtype=torch.float32)
        else:
            article_mask = torch.tensor(clicked_news_mask, device=device, dtype=torch.float32)
        
        # Zero out embeddings of padded articles
        expanded_mask = article_mask.unsqueeze(-1).unsqueeze(-1)
        clicked_token_embeddings = clicked_token_embeddings * expanded_mask
        
        # 4. Flatten user tokens: [batch_size, total_query_tokens, dim]
        user_embeddings = clicked_token_embeddings.view(batch_size, -1, self.embedding_dim)
        
        # 5. Add position embeddings (Extension 2)
        user_embeddings = self._add_position_embeddings(user_embeddings, num_clicked, num_query_tokens)
        
        # 6. Apply attention (Extension 1 & 3)
        user_embeddings = self._apply_attention(user_embeddings, article_mask, num_clicked, num_query_tokens)
        
        # 7. Compute MaxSim scores for each candidate
        scores_list = []
        for i in range(num_candidates):
            doc_emb = candidate_token_embeddings[:, i, :, :]
            score = maxsim_score(user_embeddings, doc_emb)
            scores_list.append(score)
        
        return torch.stack(scores_list, dim=1)

    def get_news_vector(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get news token embeddings for evaluation. Returns: [batch_size, num_tokens, dim]"""
        device = next(self.parameters()).device
        input_ids = news["title"][:, 0].to(device)
        attention_mask = news["title"][:, 1].to(device)
        
        with torch.no_grad():
            embeddings = self._encode_ids_with_colbert(input_ids, attention_mask, is_query=False)
        return embeddings
    
    def get_user_vector(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        """Get user vector by flattening and optionally applying attention.
        
        Args:
            clicked_news_vector: [batch_size, num_clicked, num_tokens, dim]
            
        Returns:
            Flattened (and contextualized) user tokens: [batch_size, total_tokens, dim]
        """
        batch_size, num_clicked, num_tokens, dim = clicked_news_vector.shape
        
        # Flatten: [batch_size, num_clicked * num_tokens, dim]
        user_embeddings = clicked_news_vector.reshape(batch_size, -1, dim)
        
        # Detect real vs padded articles by checking for non-zero embeddings
        article_mask = (clicked_news_vector.abs().sum(dim=(2, 3)) > 1e-6).float()
        
        # Add position embeddings
        user_embeddings = self._add_position_embeddings(user_embeddings, num_clicked, num_tokens)
        
        # Apply attention
        user_embeddings = self._apply_attention(user_embeddings, article_mask, num_clicked, num_tokens)
        
        return user_embeddings
    
    def get_prediction(
        self, news_vector: torch.Tensor, user_vector: torch.Tensor
    ) -> torch.Tensor:
        """Get click prediction using MaxSim.
        
        Args:
            news_vector: [num_candidates, doc_tokens, dim]
            user_vector: [batch_size, total_user_tokens, dim] or [total_user_tokens, dim]
            
        Returns:
            Scores: [num_candidates]
        """
        if user_vector.dim() == 2:
            user_vector = user_vector.unsqueeze(0)
        
        num_candidates = news_vector.shape[0]
        user_vector_expanded = user_vector.expand(num_candidates, -1, -1)
        
        return maxsim_score(user_vector_expanded, news_vector)
