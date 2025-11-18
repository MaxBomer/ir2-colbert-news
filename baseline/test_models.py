#!/usr/bin/env python3
"""Quick test script to verify model loading and training setup."""
import sys
from pathlib import Path

# Add baseline to path
baseline_path = Path(__file__).parent
if str(baseline_path) not in sys.path:
    sys.path.insert(0, str(baseline_path))

from config import create_config
from model.factory import create_model, list_available_models
from utils import get_device


def test_model_loading():
    """Test that models can be loaded."""
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Test NRMSbert
    print("\n1. Testing NRMSbert...")
    try:
        config = create_config()
        config.model_type = 'NRMSbert'
        model = create_model(config).to(device)
        print(f"   ✓ NRMSbert loaded successfully")
        print(f"   Model: {type(model).__name__}")
    except Exception as e:
        print(f"   ✗ NRMSbert failed: {e}")
        return False
    
    # Test ColBERT
    print("\n2. Testing ColBERT...")
    try:
        config = create_config()
        config.model_type = 'ColBERT'
        config.pretrained_model_name = 'prajjwal1/bert-tiny'  # Use small model for testing
        # Set colbert_embedding_dim to match word_embedding_dim for compatibility
        config.colbert_embedding_dim = config.word_embedding_dim
        model = create_model(config).to(device)
        print(f"   ✓ ColBERT loaded successfully")
        print(f"   Model: {type(model).__name__}")
    except Exception as e:
        print(f"   ✗ ColBERT failed: {e}")
        print(f"   Note: Make sure colbert/ directory is accessible")
        return False
    
    print("\n" + "=" * 60)
    print("All models loaded successfully!")
    print("=" * 60)
    print(f"\nAvailable models: {', '.join(list_available_models())}")
    return True


if __name__ == '__main__':
    success = test_model_loading()
    sys.exit(0 if success else 1)

