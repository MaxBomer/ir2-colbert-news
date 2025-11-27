"""Configuration for NRMSbert model with BERT tiny."""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Configuration for NRMSbert model")
    
    # Data paths (defaults assume running from baseline/ directory)
    parser.add_argument(
        '--current_data_path',
        type=str,
        default='../data',
        help='Path to processed data and checkpoints (default: ../data)'
    )
    parser.add_argument(
        '--original_data_path',
        type=str,
        default='../data/original',
        help='Path to original data directory (default: ../data/original)'
    )
    
    # Model parameters
    parser.add_argument(
        '--model_type',
        type=str,
        default='NRMSbert',
        choices=['NRMSbert', 'ColBERT'],
        help='Model type: NRMSbert or ColBERT'
    )
    parser.add_argument(
        '--pretrained_model_name',
        type=str,
        default='prajjwal1/bert-tiny',
        help='Pretrained model name or path (BERT for NRMSbert, any transformer for ColBERT)'
    )
    parser.add_argument(
        '--colbert_model_name',
        type=str,
        default=None,
        help='ColBERT model name (if different from pretrained_model_name)'
    )
    parser.add_argument(
        '--colbert_embedding_dim',
        type=int,
        default=None,
        help='ColBERT embedding dimension (auto-detected if not specified)'
    )
    parser.add_argument(
        '--colbert_max_query_tokens',
        type=int,
        default=32,
        help='Maximum number of tokens for query encoding (default: 32)'
    )
    parser.add_argument(
        '--colbert_max_doc_tokens',
        type=int,
        default=128,
        help='Maximum number of tokens for document encoding (default: 128)'
    )
    parser.add_argument(
        '--colbert_freeze_weights',
        action='store_true',
        help='Freeze ColBERT weights (zero-shot mode)'
    )
    parser.add_argument(
        '--colbert_user_attention',
        action='store_true',
        help='Enable cross-article self-attention over user tokens (default: False)'
    )
    parser.add_argument(
        '--colbert_position_embeddings',
        action='store_true',
        help='Add article position embeddings before attention (default: False)'
    )
    parser.add_argument(
        '--colbert_hierarchical_attention',
        action='store_true',
        help='Enable hierarchical attention: token-level + article-level (default: False)'
    )
    parser.add_argument(
        '--colbert_attention_heads',
        type=int,
        default=8,
        help='Number of attention heads for ColBERT user attention (default: 8)'
    )
    parser.add_argument(
        '--bert_version',
        type=str,
        default='tiny',
        help='BERT version identifier (tiny, mini, medium, large) - used for data paths'
    )
    parser.add_argument(
        '--word_embedding_dim',
        type=int,
        default=128,
        help='Word embedding dimension (128 for BERT tiny, 768 for BERT base)'
    )
    parser.add_argument(
        '--num_attention_heads',
        type=int,
        default=2,
        help='Number of attention heads (must divide word_embedding_dim)'
    )
    parser.add_argument(
        '--finetune_layers',
        type=int,
        default=2,
        help='Number of BERT layers to fine-tune'
    )
    
    # Training parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Learning rate'
    )
    parser.add_argument(
        '--dropout_probability',
        type=float,
        default=0.2,
        help='Dropout probability'
    )
    parser.add_argument(
        '--negative_sampling_ratio',
        type=int,
        default=2,
        help='Negative sampling ratio (K)'
    )
    
    # Test/quick run parameters
    parser.add_argument(
        '--test_run',
        action='store_true',
        help='Quick test run: train for 10 batches, validate on 100 samples'
    )
    parser.add_argument(
        '--max_batches',
        type=int,
        default=None,
        help='Maximum number of batches to train (overrides num_epochs if set)'
    )
    parser.add_argument(
        '--max_validation_samples',
        type=int,
        default=200000,
        help='Maximum number of validation samples to evaluate (default: 200000)'
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    # Data preprocessing parameters (updated after preprocessing)
    parser.add_argument(
        '--num_categories',
        type=int,
        default=275,
        help='Number of categories (1 + len(category2int))'
    )
    parser.add_argument(
        '--num_words',
        type=int,
        default=70973,
        help='Number of words (1 + len(word2int))'
    )
    parser.add_argument(
        '--num_entities',
        type=int,
        default=12958,
        help='Number of entities (1 + len(entity2int))'
    )
    parser.add_argument(
        '--num_users',
        type=int,
        default=50001,
        help='Number of users (1 + len(user2int))'
    )
    
    return parser.parse_args()


@dataclass
class NRMSbertConfig:
    """Configuration for news recommendation models (NRMSbert, ColBERT, etc.)."""
    
    # Data paths (no defaults - must come first)
    current_data_path: Path
    original_data_path: Path
    
    # Model architecture (no defaults - must come first)
    model_type: str  # 'NRMSbert' or 'ColBERT'
    pretrained_model_name: str
    bert_version: str  # Used for data path organization
    word_embedding_dim: int
    num_attention_heads: int
    finetune_layers: int
    
    # Training hyperparameters (no defaults - must come first)
    batch_size: int
    learning_rate: float
    dropout_probability: float
    negative_sampling_ratio: int
    
    # Test/quick run parameters
    test_run: bool
    max_batches: Optional[int]
    max_validation_samples: int
    
    # Vocabulary sizes (no defaults - must come first)
    num_categories: int
    num_words: int
    num_entities: int
    num_users: int
    
    # Fields with defaults (must come after fields without defaults)
    # ColBERT-specific parameters (optional)
    colbert_model_name: str | None = None
    colbert_embedding_dim: int | None = None
    colbert_max_query_tokens: int = 32
    colbert_max_doc_tokens: int = 128
    colbert_freeze_weights: bool = False
    colbert_user_attention: bool = False  # Cross-article self-attention
    colbert_position_embeddings: bool = False  # Article position embeddings
    colbert_hierarchical_attention: bool = False  # Token + article level attention
    colbert_attention_heads: int = 8  # Attention heads when enabled
    query_vector_dim: int = 200
    num_epochs: int = 100
    num_batches_show_loss: int = 100
    num_batches_validate: int = 1000
    num_workers: int = 4
    num_clicked_news_a_user: int = 50
    num_words_title: int = 20
    num_words_abstract: int = 50
    dataset_attributes: Optional[Dict[str, List[str]]] = None
    seed: int = 2024
    
    def __post_init__(self) -> None:
        """Initialize dataset_attributes after object creation."""
        if self.dataset_attributes is None:
            self.dataset_attributes = {"news": ["title"], "record": []}
        
        # Convert string paths to Path objects
        if isinstance(self.current_data_path, str):
            self.current_data_path = Path(self.current_data_path)
        if isinstance(self.original_data_path, str):
            self.original_data_path = Path(self.original_data_path)
    
    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory path."""
        return self.current_data_path / 'checkpoint' / 'bert' / self.bert_version / self.model_type
    
    @property
    def train_data_path(self) -> Path:
        """Get training data path (processed data organized by bert_version)."""
        return self.current_data_path / self.bert_version / 'train'
    
    @property
    def val_data_path(self) -> Path:
        """Get validation data path (processed data organized by bert_version)."""
        return self.current_data_path / self.bert_version / 'val'
    
    @property
    def test_data_path(self) -> Path:
        """Get test data path (processed data organized by bert_version)."""
        return self.current_data_path / self.bert_version / 'test'


def load_config_values_from_json(current_data_path: Path, bert_version: str) -> Dict[str, int]:
    """Load config values from JSON file if it exists.
    
    Args:
        current_data_path: Path to data directory
        bert_version: BERT version (e.g., 'tiny')
        
    Returns:
        Dictionary with config values, or empty dict if file doesn't exist
    """
    config_json_path = Path(current_data_path) / bert_version / 'config_values.json'
    if config_json_path.exists():
        try:
            with open(config_json_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # Silently fail - will use defaults
            return {}
    return {}


def create_config() -> NRMSbertConfig:
    """Create configuration from command line arguments.
    
    Automatically loads config values from JSON file if available.
    """
    args = parse_args()
    
    # Load config values from JSON file if it exists
    current_data_path = Path(args.current_data_path)
    config_values = load_config_values_from_json(current_data_path, args.bert_version)
    
    # Use JSON values if available, otherwise use command line args or defaults
    num_categories = config_values.get('num_categories', args.num_categories)
    num_words = config_values.get('num_words', args.num_words)
    num_entities = config_values.get('num_entities', args.num_entities)
    num_users = config_values.get('num_users', args.num_users)
    
    # Log if we loaded from JSON
    if config_values:
        config_json_path = current_data_path / args.bert_version / 'config_values.json'
        print(f"Loaded config values from {config_json_path}")
    
    return NRMSbertConfig(
        current_data_path=args.current_data_path,
        original_data_path=args.original_data_path,
        model_type=args.model_type,
        pretrained_model_name=args.pretrained_model_name,
        bert_version=args.bert_version,
        word_embedding_dim=args.word_embedding_dim,
        num_attention_heads=args.num_attention_heads,
        finetune_layers=args.finetune_layers,
        colbert_model_name=args.colbert_model_name,
        colbert_embedding_dim=args.colbert_embedding_dim,
        colbert_max_query_tokens=args.colbert_max_query_tokens,
        colbert_max_doc_tokens=args.colbert_max_doc_tokens,
        colbert_freeze_weights=args.colbert_freeze_weights,
        colbert_user_attention=args.colbert_user_attention,
        colbert_position_embeddings=args.colbert_position_embeddings,
        colbert_hierarchical_attention=args.colbert_hierarchical_attention,
        colbert_attention_heads=args.colbert_attention_heads,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_probability=args.dropout_probability,
        negative_sampling_ratio=args.negative_sampling_ratio,
        test_run=args.test_run,
        max_batches=args.max_batches,
        max_validation_samples=args.max_validation_samples if not args.test_run else 100,
        num_epochs=args.num_epochs,
        num_categories=num_categories,
        num_words=num_words,
        num_entities=num_entities,
        num_users=num_users,
    )


# Create config instance for backward compatibility
config = create_config()

# Export config class for type hints
__all__ = ['config', 'NRMSbertConfig', 'create_config']
