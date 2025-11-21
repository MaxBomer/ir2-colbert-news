import torch
import sys
from pathlib import Path

# Add baseline to path
sys.path.append(str(Path(__file__).parent / "baseline"))

from baseline.config import NRMSbertConfig
from baseline.model.ColBERT import ColBERTNewsRecommendationModel

def test_colbert_dimensions():
    print("Initializing Config...")
    
    # Check for MPS device (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # Create config with all required parameters dummy values
    config = NRMSbertConfig(
        current_data_path=Path('./data'),
        original_data_path=Path('./data/original'),
        model_type='ColBERT',
        pretrained_model_name='prajjwal1/bert-tiny',
        bert_version='tiny',
        word_embedding_dim=128,
        num_attention_heads=2,
        finetune_layers=2,
        batch_size=2,
        learning_rate=1e-4,
        dropout_probability=0.2,
        negative_sampling_ratio=1,
        test_run=True,
        max_batches=1,
        max_validation_samples=10,
        num_categories=10,
        num_words=100,
        num_entities=10,
        num_users=5,
        # ColBERT specific overrides
        colbert_max_query_tokens=8,
        colbert_max_doc_tokens=12,
        colbert_embedding_dim=128,
    )

    print("Initializing Model...")
    model = ColBERTNewsRecommendationModel(config)
    model.to(device)
    model.eval()

    # --- Dummy Data Setup ---
    batch_size = 2
    num_candidates = 3
    num_clicked = 2
    
    # 1. Test Forward (Training Path)
    print("\n--- Testing Forward (Training Path) ---")
    
    # Candidate News: List of dicts
    candidate_news = []
    for _ in range(num_candidates):
        # shape: [batch_size, 2, seq_len]
        ids = torch.randint(101, 1000, (batch_size, 2, 10)).to(device)
        candidate_news.append({"title": ids})
        
    # Clicked News
    clicked_news = []
    for _ in range(num_clicked):
        ids = torch.randint(101, 1000, (batch_size, 2, 10)).to(device)
        clicked_news.append({"title": ids})
        
    # Mask [batch_size, num_clicked]
    clicked_news_mask = [[1] * num_clicked for _ in range(batch_size)]
    
    try:
        scores = model(candidate_news, clicked_news, clicked_news_mask)
        print(f"Forward Output Shape: {scores.shape}")
        assert scores.shape == (batch_size, num_candidates)
        print("Forward pass successful")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Test Evaluation Path
    print("\n--- Testing Evaluation Path ---")
    
    # A. get_news_vector
    print("Testing get_news_vector...")
    # Single batch of news
    news_batch = {"title": torch.randint(101, 1000, (batch_size, 2, 10)).to(device)}
    
    try:
        news_vecs = model.get_news_vector(news_batch)
        print(f"News Vector Shape: {news_vecs.shape}")
        # Expected: [batch_size, num_doc_tokens, dim]
        assert news_vecs.shape == (batch_size, config.colbert_max_doc_tokens, config.colbert_embedding_dim)
        print("get_news_vector successful")
        
        # B. get_user_vector
        print("Testing get_user_vector...")
        # In evaluate.py, we stack news vectors to create [batch, num_clicked, tokens, dim]
        clicked_vecs_stacked = torch.stack([news_vecs for _ in range(num_clicked)], dim=1)
        
        user_vec = model.get_user_vector(clicked_vecs_stacked)
        print(f"User Vector Shape: {user_vec.shape}")
        # Expected: [batch, num_clicked * num_tokens, dim]
        total_tokens = num_clicked * config.colbert_max_doc_tokens 
        assert user_vec.shape == (batch_size, total_tokens, config.colbert_embedding_dim)
        print("get_user_vector successful")

        # C. get_prediction
        print("Testing get_prediction...")
        # Candidates: [num_candidates (not batch), tokens, dim]
        single_user_vec = user_vec[0] # [total_tokens, dim]
        
        # Simulate candidate vectors for this user
        candidate_vecs = torch.randn(num_candidates, config.colbert_max_doc_tokens, config.colbert_embedding_dim).to(device)
        
        pred_scores = model.get_prediction(candidate_vecs, single_user_vec)
        print(f"Prediction Shape: {pred_scores.shape}")
        assert pred_scores.shape == (num_candidates,)
        print("get_prediction successful")
        
    except Exception as e:
        print(f"Evaluation path failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_colbert_dimensions()
