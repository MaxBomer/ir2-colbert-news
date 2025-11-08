"""Training script for NRMSbert model."""
import datetime
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import config
from dataset import BaseDataset
from evaluate import NewsDataset, evaluate
from model.NRMSbert import NRMSbert
from utils import should_pin_memory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def should_display_progress() -> bool:
    """Check if progress bars should be displayed."""
    return sys.stdout.isatty()


def time_since(since: float) -> str:
    """Format elapsed time string.
    
    Args:
        since: Start time timestamp
        
    Returns:
        Formatted time string
    """
    elapsed_time = time.time() - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 5) -> None:
        """Initialize early stopping.
        
        Args:
            patience: Number of validation checks to wait before stopping
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float) -> Tuple[bool, bool]:
        """Check if training should stop.
        
        Args:
            val_loss: Validation loss (use negative value for metrics where higher is better)
            
        Returns:
            Tuple of (early_stop, get_better)
        """
        if val_loss < self.best_loss:
            self.counter = 0
            self.best_loss = val_loss
            return False, True
        else:
            self.counter += 1
            early_stop = self.counter >= self.patience
            return early_stop, False


def load_model(device: torch.device) -> NRMSbert:
    """Load and initialize NRMSbert model.
    
    Args:
        device: Device to load model on
        
    Returns:
        Initialized model
    """
    model = NRMSbert(config).to(device)
    logger.debug(f"Model architecture:\n{model}")
    return model


def create_dataloader(dataset: BaseDataset, shuffle: bool = True) -> DataLoader:
    """Create data loader for dataset.
    
    Args:
        dataset: Dataset instance
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=should_pin_memory()
    )


def train_step(model: NRMSbert, minibatch: dict, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Perform a single training step.
    
    Args:
        model: Model instance
        minibatch: Batch of training data
        criterion: Loss function
        optimizer: Optimizer instance
        device: Device to run on
        
    Returns:
        Loss value
    """
    # Forward pass
    y_pred = model(
        minibatch["candidate_news"],
        minibatch["clicked_news"],
        minibatch["clicked_news_mask"]
    )
    
    # Compute loss (first item is positive, rest are negative)
    y_true = torch.zeros(len(y_pred), dtype=torch.long, device=device)
    loss = criterion(y_pred, y_true)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def validate(model: NRMSbert, val_data_path: Path, evaluate_dataset: NewsDataset, max_count: int = 200000) -> Tuple[float, float, float, float]:
    """Validate model on validation set.
    
    Args:
        model: Model instance
        val_data_path: Path to validation data directory
        evaluate_dataset: Pre-loaded news dataset
        max_count: Maximum number of samples to evaluate
        
    Returns:
        Tuple of (AUC, MRR, nDCG@5, nDCG@10)
    """
    model.eval()
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
        model, str(val_data_path), config.num_workers, evaluate_dataset, max_count
    )
    model.train()
    return val_auc, val_mrr, val_ndcg5, val_ndcg10


def save_checkpoint(model: NRMSbert, optimizer: torch.optim.Optimizer, step: int, val_auc: float, checkpoint_path: Path) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        step: Current training step
        val_auc: Validation AUC score
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'val_auc': val_auc,
        },
        checkpoint_path
    )


def load_checkpoint(model: NRMSbert, checkpoint_path: Path) -> dict:
    """Load model checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def train() -> None:
    """Main training function."""
    start_time = time.time()
    # Auto-detect device: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Setup logging
    log_dir = Path("./runs/NRMSbert") / datetime.datetime.now().replace(microsecond=0).isoformat()
    if 'REMARK' in os.environ:
        log_dir = Path(str(log_dir) + '-' + os.environ['REMARK'])
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Setup checkpoint directory
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    model = load_model(device)
    count_parameters(model)
    
    # Load datasets
    logger.info("Loading training data...")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    train_dataset = BaseDataset(
        config.train_data_path / 'behaviors_parsed.tsv',
        config.train_data_path / 'news_parsed.tsv'
    )
    logger.info(f"Loaded training dataset with size {len(train_dataset)}.")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    
    logger.info("Loading validation data...")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    val_dataset = NewsDataset(config.val_data_path / 'news_parsed.tsv')
    logger.info("Finished loading validation data.")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    
    # Create data loader
    logger.info("Building data loader...")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    dataloader = create_dataloader(train_dataset, shuffle=True)
    logger.info("Finished building data loader.")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    early_stopping = EarlyStopping(patience=5)
    
    # Training loop
    loss_history = []
    exhaustion_count = 0
    step = 0
    best_step = 0
    best_checkpoint_path: Optional[Path] = None
    best_val_auc = float('-inf')
    
    # Determine total batches
    if config.test_run:
        total_batches = 10
        logger.info("TEST RUN MODE: Training for 10 batches only")
    elif config.max_batches is not None:
        total_batches = config.max_batches
        logger.info(f"Limited training: Training for {total_batches} batches")
    else:
        total_batches = config.num_epochs * len(train_dataset) // config.batch_size + 1
    
    progress = tqdm(range(1, total_batches + 1), desc="Training") if should_display_progress() else range(1, total_batches + 1)
    
    logger.info("Starting training...")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    
    dataloader_iter = iter(dataloader)
    
    for batch_idx in progress:
        # Get next batch (recreate dataloader if exhausted)
        try:
            minibatch = next(dataloader_iter)
        except StopIteration:
            exhaustion_count += 1
            if should_display_progress():
                tqdm.write(
                    f"Training data exhausted {exhaustion_count} times after {batch_idx} batches, reusing dataset."
                )
            dataloader_iter = iter(dataloader)
            minibatch = next(dataloader_iter)
        
        step += 1
        
        # Training step
        loss = train_step(model, minibatch, criterion, optimizer, device)
        loss_history.append(loss)
        
        # Logging
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/Loss', loss, step)
        
        if batch_idx % config.num_batches_show_loss == 0:
            avg_loss = np.mean(loss_history)
            recent_avg_loss = np.mean(loss_history[-256:]) if len(loss_history) >= 256 else avg_loss
            message = (
                f"Time {time_since(start_time)}, batches {batch_idx}, "
                f"current loss {loss:.4f}, average loss: {avg_loss:.4f}, "
                f"recent average loss: {recent_avg_loss:.4f}"
            )
            if should_display_progress():
                tqdm.write(message)
            else:
                logger.info(message)
        
        # Validation
        # For test runs, validate every batch; otherwise use normal frequency
        validate_frequency = 1 if config.test_run else config.num_batches_validate
        if batch_idx % validate_frequency == 0:
            val_auc, val_mrr, val_ndcg5, val_ndcg10 = validate(
                model, config.val_data_path, val_dataset, max_count=config.max_validation_samples
            )
            
            writer.add_scalar('Validation/AUC', val_auc, step)
            writer.add_scalar('Validation/MRR', val_mrr, step)
            writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
            writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
            
            message = (
                f"\nTime {time_since(start_time)}, batches {batch_idx}, "
                f"validation AUC: {val_auc:.4f}, MRR: {val_mrr:.4f}, "
                f"nDCG@5: {val_ndcg5:.4f}, nDCG@10: {val_ndcg10:.4f}"
            )
            if should_display_progress():
                tqdm.write(message)
            else:
                print(message)
            
            # Early stopping check
            early_stop, improved = early_stopping(-val_auc)
            if early_stop:
                message = 'Early stopping triggered.'
                if should_display_progress():
                    tqdm.write(message)
                else:
                    logger.info(message)
                break
            
            # Save best checkpoint
            if improved:
                best_step = step
                if best_checkpoint_path and best_checkpoint_path.exists():
                    best_checkpoint_path.unlink()
                
                best_checkpoint_path = config.checkpoint_dir / f'ckpt-{step}.pth'
                save_checkpoint(model, optimizer, step, val_auc, best_checkpoint_path)
                best_val_auc = val_auc
    
    # Load best checkpoint and evaluate
    if best_checkpoint_path and best_checkpoint_path.exists():
        logger.info(f"Loading best checkpoint from step {best_step}...")
        load_checkpoint(model, best_checkpoint_path)
    
    model.eval()
    final_auc, final_mrr, final_ndcg5, final_ndcg10 = validate(
        model, config.val_data_path, val_dataset, max_count=config.max_validation_samples
    )
    
    message = (
        f"\n\nTime {time_since(start_time)}, batches {batch_idx}, "
        f"Final AUC: {final_auc:.4f}, Final MRR: {final_mrr:.4f}, "
        f"Final nDCG@5: {final_ndcg5:.4f}, Final nDCG@10: {final_ndcg10:.4f}"
    )
    if should_display_progress():
        tqdm.write(message)
    else:
        print(message)


if __name__ == '__main__':
    # Auto-detect device: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f'Using device: {device}')
    logger.info('Training NRMSbert model')
    train()
