"""Evaluation script for NRMSbert model."""
import ast
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from ast import literal_eval
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from config import config
from model.base import BaseNewsRecommendationModel
from model.factory import create_model
from utils import load_news_dataset, save_news_dataset, should_pin_memory, get_device, should_display_progress

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def should_display_progress() -> bool:
    """Check if progress bars should be displayed."""
    return sys.stdout.isatty()


def dcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Calculate Discounted Cumulative Gain.
    
    Args:
        y_true: True relevance scores
        y_score: Predicted scores
        k: Top k items to consider
        
    Returns:
        DCG score
    """
    order = np.argsort(y_score)[::-1]
    y_true_ordered = np.take(y_true, order[:k])
    gains = 2**y_true_ordered - 1
    discounts = np.log2(np.arange(len(y_true_ordered)) + 2)
    return float(np.sum(gains / discounts))


def ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain.
    
    Args:
        y_true: True relevance scores
        y_score: Predicted scores
        k: Top k items to consider
        
    Returns:
        nDCG score
    """
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 0.0
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate Mean Reciprocal Rank.
    
    Args:
        y_true: True relevance scores
        y_score: Predicted scores
        
    Returns:
        MRR score
    """
    order = np.argsort(y_score)[::-1]
    y_true_ordered = np.take(y_true, order)
    rr_scores = y_true_ordered / (np.arange(len(y_true_ordered)) + 1)
    sum_true = np.sum(y_true_ordered)
    if sum_true == 0:
        return 0.0
    return float(np.sum(rr_scores) / sum_true)


def calculate_single_user_metric(pair: Tuple[List[int], List[float]]) -> List[float]:
    """Calculate metrics for a single user.
    
    Args:
        pair: Tuple of (y_true, y_pred)
        
    Returns:
        List of [AUC, MRR, nDCG@5, nDCG@10]
    """
    y_true, y_pred = pair
    try:
        auc = roc_auc_score(y_true, y_pred)
        mrr = mrr_score(np.array(y_true), np.array(y_pred))
        ndcg5 = ndcg_score(np.array(y_true), np.array(y_pred), k=5)
        ndcg10 = ndcg_score(np.array(y_true), np.array(y_pred), k=10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4


def _parse_tokenized_title(title_str: str) -> torch.Tensor:
    """Parse tokenized title string to tensor.
    
    Args:
        title_str: String representation of tokenized title
        
    Returns:
        Tensor with shape [2, num_words_title]
    """
    title_dict = ast.literal_eval(title_str)
    input_ids = torch.tensor(title_dict['input_ids'])
    attention_mask = torch.tensor(title_dict['attention_mask'])
    return torch.cat([input_ids.unsqueeze(0), attention_mask.unsqueeze(0)], dim=0)


class NewsDataset(Dataset):
    """Dataset for loading news articles during evaluation."""
    
    def __init__(self, news_path: Path | str) -> None:
        """Initialize news dataset.
        
        Args:
            news_path: Path to parsed news TSV file
        """
        super().__init__()
        
        # Convert Path object to string for pandas
        news_path_str = str(news_path) if isinstance(news_path, Path) else news_path
        
        # Check if file exists
        if not Path(news_path_str).exists():
            raise FileNotFoundError(f"News file not found: {news_path_str}")
        
        self.news_parsed = pd.read_table(
            news_path_str,
            sep='\t',
            usecols=['id', 'title'],
            converters={'title': literal_eval}
        )
        
        self.news2dict: Dict[int, Dict[str, torch.Tensor]] = {}
        for idx, (_, row) in enumerate(self.news_parsed.iterrows()):
            self.news2dict[idx] = {
                'id': row['id'],
                'title': _parse_tokenized_title(str(row['title']))
            }
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.news2dict)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        """Get a news article.
        
        Args:
            idx: Article index
            
        Returns:
            Dictionary containing 'id' and 'title'
        """
        return self.news2dict[idx]


class UserDataset(Dataset):
    """Dataset for loading user click histories during evaluation."""
    
    def __init__(self, behaviors_path: Path | str, user2int_path: Path | str) -> None:
        """Initialize user dataset.
        
        Args:
            behaviors_path: Path to behaviors TSV file
            user2int_path: Path to user2int mapping file
        """
        super().__init__()
        
        # Convert Path objects to strings for pandas
        behaviors_path_str = str(behaviors_path) if isinstance(behaviors_path, Path) else behaviors_path
        user2int_path_str = str(user2int_path) if isinstance(user2int_path, Path) else user2int_path
        
        # Check if files exist
        if not Path(behaviors_path_str).exists():
            raise FileNotFoundError(f"Behaviors file not found: {behaviors_path_str}")
        if not Path(user2int_path_str).exists():
            raise FileNotFoundError(f"User2int file not found: {user2int_path_str}")
        
        self.behaviors = pd.read_table(
            behaviors_path_str,
            sep='\t',
            header=None,
            usecols=[1, 3],
            names=['user', 'clicked_news']
        )
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors.drop_duplicates(inplace=True)
        
        user2int = dict(pd.read_table(user2int_path_str, sep='\t').values.tolist())
        
        for row in self.behaviors.itertuples():
            self.behaviors.at[row.Index, 'user'] = user2int.get(row.user, 0)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.behaviors)
    
    def __getitem__(self, idx: int) -> Dict[str, List[str] | str]:
        """Get a user's click history.
        
        Args:
            idx: User index
            
        Returns:
            Dictionary containing 'clicked_news_string' and 'clicked_news'
        """
        row = self.behaviors.iloc[idx]
        clicked_news = row.clicked_news.split()[:config.num_clicked_news_a_user]
        repeated_times = config.num_clicked_news_a_user - len(clicked_news)
        assert repeated_times >= 0
        
        return {
            "clicked_news_string": row.clicked_news,
            "clicked_news": ['PADDED_NEWS'] * repeated_times + clicked_news,
        }


class BehaviorsDataset(Dataset):
    """Dataset for loading user behaviors during evaluation."""
    
    def __init__(self, behaviors_path: Path | str) -> None:
        """Initialize behaviors dataset.
        
        Args:
            behaviors_path: Path to behaviors TSV file
        """
        super().__init__()
        
        # Convert Path object to string for pandas
        behaviors_path_str = str(behaviors_path) if isinstance(behaviors_path, Path) else behaviors_path
        
        # Check if file exists
        if not Path(behaviors_path_str).exists():
            raise FileNotFoundError(f"Behaviors file not found: {behaviors_path_str}")
        
        self.behaviors = pd.read_table(
            behaviors_path_str,
            sep='\t',
            header=None,
            usecols=range(5),
            names=['impression_id', 'user', 'time', 'clicked_news', 'impressions']
        )
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors.impressions = self.behaviors.impressions.str.split()
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.behaviors)
    
    def __getitem__(self, idx: int) -> Dict[str, List[str] | str]:
        """Get a behavior record.
        
        Args:
            idx: Record index
            
        Returns:
            Dictionary containing impression data
        """
        row = self.behaviors.iloc[idx]
        return {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Attention memory: batch × heads × seq² × 4 bytes (seq = 50 × 32 = 1600)
    # Baseline memory ~66 GB (model + news vectors + attention mask + intermediates)
    # Available: 93 - 66 = ~27 GB for attention scores
    # multiplier=8 (batch=256): 256 × 8 × 1600² × 4 = 21 GB ✓ fits in 27 GB
    batch_size_multiplier: int = 8
    max_count: int = sys.maxsize


def compute_news_vectors(model: BaseNewsRecommendationModel, news_dataset: NewsDataset, device: torch.device, eval_config: EvaluationConfig) -> Dict[str, torch.Tensor]:
    """Compute news article vectors.
    
    Args:
        model: Model instance
        news_dataset: News dataset
        device: Device to run on
        eval_config: Evaluation configuration
        
    Returns:
        Dictionary mapping news IDs to vectors
    """
    news_dataloader = DataLoader(
        news_dataset,
        batch_size=config.batch_size * eval_config.batch_size_multiplier,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=should_pin_memory(device)
    )
    
    news2vector: Dict[str, torch.Tensor] = {}
    progress = tqdm(news_dataloader, desc="Computing news vectors") if should_display_progress() else news_dataloader
    
    for minibatch in progress:
        # Handle both single items and batches
        # DataLoader automatically stacks tensors when batching, so check if title is already a tensor
        if isinstance(minibatch['id'], (list, tuple)):
            news_ids = list(minibatch['id'])
            # If title is already a tensor (batched), use it directly; otherwise stack
            if isinstance(minibatch['title'], torch.Tensor):
                titles = minibatch['title']
            else:
                titles = torch.stack(minibatch['title'])
        else:
            news_ids = [minibatch['id']]
            titles = minibatch['title'].unsqueeze(0)
        
        batch_dict = {'title': titles}
        
        if any(news_id not in news2vector for news_id in news_ids):
            news_vectors = model.get_news_vector(batch_dict)
            # Removed the unsqueeze logic here because get_news_vector should always return [batch_size, ...]
            # And torch.stack handles single items correctly if they are not squeezed.
            
            for news_id, vector in zip(news_ids, news_vectors):
                if news_id not in news2vector:
                    news2vector[news_id] = vector
    
    # Add padding vector
    if news2vector:
        padding_vector = torch.zeros_like(list(news2vector.values())[0])
        news2vector['PADDED_NEWS'] = padding_vector
    
    return news2vector


def compute_user_vectors(model: BaseNewsRecommendationModel, user_dataset: UserDataset, news2vector: Dict[str, torch.Tensor], device: torch.device, eval_config: EvaluationConfig) -> Dict[str, torch.Tensor]:
    """Compute user vectors from click histories.
    
    Args:
        model: Model instance
        user_dataset: User dataset
        news2vector: Dictionary mapping news IDs to vectors
        device: Device to run on
        eval_config: Evaluation configuration
        
    Returns:
        Dictionary mapping user strings to vectors
    """
    user_dataloader = DataLoader(
        user_dataset,
        batch_size=config.batch_size * eval_config.batch_size_multiplier,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=should_pin_memory(device)
    )
    
    user2vector: Dict[str, torch.Tensor] = {}
    progress = tqdm(user_dataloader, desc="Computing user vectors") if should_display_progress() else user_dataloader
    
    for minibatch in progress:
        user_strings = minibatch["clicked_news_string"]
        
        if any(user_string not in user2vector for user_string in user_strings):
            # DataLoader returns clicked_news as [num_clicked][batch] (list of positions, each with batch items)
            # Inner stack: [batch_size, num_tokens, dim] for each clicked position
            # Outer stack: [num_clicked, batch_size, num_tokens, dim]
            # Transpose to get: [batch_size, num_clicked, num_tokens, dim]
            clicked_news_vectors = torch.stack([
                torch.stack([
                    news2vector[news_id].to(device) for news_id in news_list
                ], dim=0)
                for news_list in minibatch["clicked_news"]
            ], dim=0).transpose(0, 1)
            
            user_vectors = model.get_user_vector(clicked_news_vectors)
            
            for user_string, vector in zip(user_strings, user_vectors):
                if user_string not in user2vector:
                    user2vector[user_string] = vector
    
    return user2vector


@dataclass
class EvaluationParams:
    """Parameters for evaluation function."""
    directory: Path | str
    num_workers: int
    news_dataset_built: Optional[NewsDataset] = None
    max_count: int = sys.maxsize


@torch.no_grad()
def evaluate(model: BaseNewsRecommendationModel, params: EvaluationParams) -> Tuple[float, float, float, float]:
    """Evaluate model on target directory.
    
    Args:
        model: Model to be evaluated
        params: Evaluation parameters
        
    Returns:
        Tuple of (AUC, MRR, nDCG@5, nDCG@10)
    """
    device = next(model.parameters()).device
    directory = Path(params.directory)
    eval_config = EvaluationConfig(max_count=params.max_count)
    
    # Load or use provided news dataset
    if params.news_dataset_built is not None:
        news_dataset = params.news_dataset_built
    else:
        news_dataset = NewsDataset(directory / 'news_parsed.tsv')
    
    # Compute news vectors
    news2vector = compute_news_vectors(model, news_dataset, device, eval_config)
    
    # Load user dataset
    user_dataset = UserDataset(
        directory / 'behaviors.tsv',
        config.train_data_path / 'user2int.tsv'
    )
    
    # Compute user vectors
    user2vector = compute_user_vectors(model, user_dataset, news2vector, device, eval_config)
    
    # Load behaviors dataset
    behaviors_dataset = BehaviorsDataset(directory / 'behaviors.tsv')
    behaviors_dataloader = DataLoader(
        behaviors_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Compute predictions and metrics
    tasks = []
    progress = tqdm(behaviors_dataloader, desc="Computing predictions") if should_display_progress() else behaviors_dataloader
    
    for count, minibatch in enumerate(progress, 1):
        if count > params.max_count:
            break
        
        user_string = minibatch['clicked_news_string'][0]
        if user_string not in user2vector:
            continue
        
        # Get candidate news vectors
        candidate_news_vectors = torch.stack([
            news2vector[news[0].split('-')[0]]
            for news in minibatch['impressions']
        ], dim=0)
        
        user_vector = user2vector[user_string]
        click_probabilities = model.get_prediction(candidate_news_vectors, user_vector)
        
        y_pred = click_probabilities.tolist()
        y_true = [
            int(news[0].split('-')[1]) for news in minibatch['impressions']
        ]
        
        tasks.append((y_true, y_pred))
    
    # Calculate metrics in parallel
    with Pool(processes=params.num_workers) as pool:
        results = pool.map(calculate_single_user_metric, tasks)
    
    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
    auc_mean = float(np.nanmean(aucs))
    mrr_mean = float(np.nanmean(mrrs))
    ndcg5_mean = float(np.nanmean(ndcg5s))
    ndcg10_mean = float(np.nanmean(ndcg10s))
    
    # Log final evaluation metrics to wandb
    if HAS_WANDB and os.environ.get('WANDB_API_KEY'):
        try:
            wandb.log({
                'final_eval/auc': auc_mean,
                'final_eval/mrr': mrr_mean,
                'final_eval/ndcg@5': ndcg5_mean,
                'final_eval/ndcg@10': ndcg10_mean,
                'final_eval/num_samples': len(tasks),
            })
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")
    
    return (auc_mean, mrr_mean, ndcg5_mean, ndcg10_mean)


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = [
        f for f in checkpoint_dir.iterdir()
        if f.suffix == '.pth' and f.stem.startswith('ckpt-')
    ]
    
    if not checkpoints:
        return None
    
    # Extract step numbers and find maximum
    def get_step(path: Path) -> int:
        try:
            return int(path.stem.split('-')[1])
        except (IndexError, ValueError):
            return -1
    
    latest = max(checkpoints, key=get_step)
    return latest if get_step(latest) >= 0 else None


if __name__ == '__main__':
    device = get_device()
    logger.info(f'Using device: {device}')
    logger.info(f'Evaluating {config.model_type} model')
    
    # Load model
    model = create_model(config).to(device)
    
    # Find and load checkpoint
    checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
    if checkpoint_path is None:
        logger.error('No checkpoint file found!')
        sys.exit(1)
    
    logger.info(f"Loading saved parameters from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate on test set
    params = EvaluationParams(
        directory=config.test_data_path,
        num_workers=config.num_workers
    )
    auc, mrr, ndcg5, ndcg10 = evaluate(model, params)
    
    print(
        f'AUC: {auc:.4f}\nMRR: {mrr:.4f}\nnDCG@5: {ndcg5:.4f}\nnDCG@10: {ndcg10:.4f}'
    )
