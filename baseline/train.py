"""Training script for NRMSbert model."""
import datetime
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from config import config, NRMSbertConfig
from dataset import BaseDataset
from evaluate import NewsDataset, evaluate, EvaluationParams
from model.base import BaseNewsRecommendationModel
from model.factory import create_model
from utils import get_device, should_pin_memory, should_display_progress

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_wandb_setup() -> bool:
    """Check if wandb is properly configured.
    
    Returns:
        True if wandb is available and API key is set, False otherwise
    """
    if not HAS_WANDB:
        warnings.warn("wandb not installed. Install with: pip install wandb", UserWarning)
        return False
    
    if not os.environ.get('WANDB_API_KEY'):
        warnings.warn(
            "WANDB_API_KEY environment variable not set. "
            "Set it with: export WANDB_API_KEY='your-api-key' "
            "or wandb login. Metrics will not be logged to wandb.",
            UserWarning
        )
        return False
    
    return True


@dataclass
class TrainingContext:
    """Context object holding high-level training state."""
    config: NRMSbertConfig
    device: torch.device
    model: BaseNewsRecommendationModel
    train_dataset: BaseDataset
    val_dataset: NewsDataset
    dataloader: DataLoader
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    start_time: float
    early_stopping: 'EarlyStopping'
    use_wandb: bool = False
    
    @property
    def batches_per_epoch(self) -> int:
        """Calculate batches per epoch."""
        return len(self.train_dataset) // self.config.batch_size


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


def load_model(device: torch.device) -> BaseNewsRecommendationModel:
    """Load and initialize model based on config.
    
    Args:
        device: Device to load model on
        
    Returns:
        Initialized model
    """
    model = create_model(config).to(device)
    logger.debug(f"Model architecture:\n{model}")
    logger.info(f"Using model type: {config.model_type}")
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


def train_step(ctx: TrainingContext, minibatch: dict) -> float:
    """Perform a single training step.
    
    Args:
        ctx: Training context
        minibatch: Batch of training data
        
    Returns:
        Loss value
    """
    # Forward pass
    y_pred = ctx.model(
        minibatch["candidate_news"],
        minibatch["clicked_news"],
        minibatch["clicked_news_mask"]
    )
    
    # Compute loss (first item is positive, rest are negative)
    y_true = torch.zeros(len(y_pred), dtype=torch.long, device=ctx.device)
    loss = ctx.criterion(y_pred, y_true)
    
    # Backward pass
    ctx.optimizer.zero_grad()
    loss.backward()
    ctx.optimizer.step()
    
    return loss.item()


def validate(ctx: TrainingContext) -> Tuple[float, float, float, float]:
    """Validate model on validation set.
    
    Args:
        ctx: Training context
        
    Returns:
        Tuple of (AUC, MRR, nDCG@5, nDCG@10)
    """
    ctx.model.eval()
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
        ctx.model,
        EvaluationParams(
            directory=str(ctx.config.val_data_path),
            num_workers=ctx.config.num_workers,
            news_dataset_built=ctx.val_dataset,
            max_count=ctx.config.max_validation_samples
        )
    )
    ctx.model.train()
    return val_auc, val_mrr, val_ndcg5, val_ndcg10


def save_checkpoint(ctx: TrainingContext, step: int, val_auc: float, checkpoint_path: Path) -> None:
    """Save model checkpoint.
    
    Args:
        ctx: Training context
        step: Current training step
        val_auc: Validation AUC score
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': ctx.model.state_dict(),
            'optimizer_state_dict': ctx.optimizer.state_dict(),
            'step': step,
            'val_auc': val_auc,
        },
        checkpoint_path
    )


def load_checkpoint(ctx: TrainingContext, checkpoint_path: Path) -> dict:
    """Load model checkpoint.
    
    Args:
        ctx: Training context
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path)
    ctx.model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def setup_training_context(cfg: NRMSbertConfig) -> TrainingContext:
    """Setup and initialize training context.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Initialized training context
    """
    start_time = time.time()
    device = get_device()
    
    # Check wandb setup
    use_wandb = check_wandb_setup()
    
    # Initialize wandb if available
    if use_wandb:
        wandb.init(
            project="news-recommendation",
            name=f"{cfg.model_type}-{datetime.datetime.now().isoformat()}",
            config={
                'model_type': cfg.model_type,
                'learning_rate': cfg.learning_rate,
                'batch_size': cfg.batch_size,
                'dropout_probability': cfg.dropout_probability,
                'negative_sampling_ratio': cfg.negative_sampling_ratio,
                'pretrained_model_name': cfg.pretrained_model_name,
                'colbert_model_name': cfg.colbert_model_name,
                'colbert_embedding_dim': cfg.colbert_embedding_dim,
                'colbert_max_query_tokens': cfg.colbert_max_query_tokens,
                'colbert_max_doc_tokens': cfg.colbert_max_doc_tokens,
            }
        )
        logger.info("wandb initialized for experiment tracking")
    
    # Setup checkpoint directory
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    model = load_model(device)
    count_parameters(model)
    
    # Load datasets
    logger.info("Loading training data...")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    train_dataset = BaseDataset(
        cfg.train_data_path / 'behaviors_parsed.tsv',
        cfg.train_data_path / 'news_parsed.tsv'
    )
    dataset_size = len(train_dataset)
    batches_per_epoch = dataset_size // cfg.batch_size
    logger.info(f"Loaded training dataset with size {dataset_size:,}.")
    logger.info(f"Batches per epoch: {batches_per_epoch:,} (batch_size={cfg.batch_size})")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    
    logger.info("Loading validation data...")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    val_dataset = NewsDataset(cfg.val_data_path / 'news_parsed.tsv')
    logger.info("Finished loading validation data.")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    
    # Create data loader
    logger.info("Building data loader...")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    dataloader = create_dataloader(train_dataset, shuffle=True)
    logger.info("Finished building data loader.")
    logger.info(f"Time elapsed: {time_since(start_time)}")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    early_stopping = EarlyStopping(patience=5)
    
    return TrainingContext(
        config=cfg,
        device=device,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        start_time=start_time,
        early_stopping=early_stopping,
        use_wandb=use_wandb
    )


def run_training_loop(ctx: TrainingContext) -> Tuple[int, Optional[Path], int]:
    """Run the main training loop.
    
    Args:
        ctx: Training context
        
    Returns:
        Tuple of (final_batch_idx, best_checkpoint_path, best_step)
    """
    loss_history = []
    exhaustion_count = 0
    step = 0
    best_step = 0
    best_checkpoint_path: Optional[Path] = None
    
    # Determine total batches
    if ctx.config.test_run:
        total_batches = 10
        logger.info("TEST RUN MODE: Training for 10 batches only")
    elif ctx.config.max_batches is not None:
        total_batches = ctx.config.max_batches
        logger.info(f"Limited training: Training for {total_batches} batches")
    else:
        total_batches = ctx.config.num_epochs * len(ctx.train_dataset) // ctx.config.batch_size + 1
    
    progress = tqdm(range(1, total_batches + 1), desc="Training") if should_display_progress() else range(1, total_batches + 1)
    
    logger.info("Starting training...")
    logger.info(f"Time elapsed: {time_since(ctx.start_time)}")
    
    dataloader_iter = iter(ctx.dataloader)
    
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
            dataloader_iter = iter(ctx.dataloader)
            minibatch = next(dataloader_iter)
        
        step += 1
        
        # Training step
        loss = train_step(ctx, minibatch)
        loss_history.append(loss)
        
        # Logging
        log_training_progress(ctx, batch_idx, step, loss, loss_history)
        
        # Validation
        validate_frequency = 1 if ctx.config.test_run else ctx.config.num_batches_validate
        if batch_idx % validate_frequency == 0:
            should_stop, improved, checkpoint_path = run_validation(ctx, batch_idx, step)
            if should_stop:
                break
            
            if improved:
                best_step = step
                if best_checkpoint_path and best_checkpoint_path.exists():
                    best_checkpoint_path.unlink()
                best_checkpoint_path = checkpoint_path
    
    return batch_idx, best_checkpoint_path, best_step


def log_training_progress(ctx: TrainingContext, batch_idx: int, step: int, loss: float, loss_history: list) -> None:
    """Log training progress.
    
    Args:
        ctx: Training context
        batch_idx: Current batch index
        step: Current training step
        loss: Current loss value
        loss_history: History of loss values
    """
    if batch_idx % 10 == 0:
        # Log to wandb
        if ctx.use_wandb:
            wandb.log({'train/loss_step': loss, 'step': step})
    
    if batch_idx % ctx.config.num_batches_show_loss == 0:
        avg_loss = np.mean(loss_history)
        recent_avg_loss = np.mean(loss_history[-256:]) if len(loss_history) >= 256 else avg_loss
        
        # Compute loss trend to detect plateaus
        if len(loss_history) >= 100:
            loss_trend = np.polyfit(range(len(loss_history[-100:])), loss_history[-100:], 1)[0]
        else:
            loss_trend = 0.0
        
        message = (
            f"Time {time_since(ctx.start_time)}, batches {batch_idx}, "
            f"current loss {loss:.4f}, average loss: {avg_loss:.4f}, "
            f"recent average loss: {recent_avg_loss:.4f}, trend: {loss_trend:.6f}"
        )
        if should_display_progress():
            tqdm.write(message)
        else:
            logger.info(message)
        
        # Log comprehensive metrics to wandb
        if ctx.use_wandb:
            wandb.log({
                'train/loss': loss,
                'train/avg_loss': avg_loss,
                'train/recent_avg_loss': recent_avg_loss,
                'train/loss_trend': loss_trend,  # slope: negative = improving, positive = plateau
                'train/batch': batch_idx,
                'step': step,
            })


def run_validation(ctx: TrainingContext, batch_idx: int, step: int) -> Tuple[bool, bool, Optional[Path]]:
    """Run validation and handle checkpointing.
    
    Args:
        ctx: Training context
        batch_idx: Current batch index
        step: Current training step
        
    Returns:
        Tuple of (should_stop, improved, checkpoint_path)
    """
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = validate(ctx)
    
    # Log to wandb
    if ctx.use_wandb:
        wandb.log({
            'val/auc': val_auc,
            'val/mrr': val_mrr,
            'val/ndcg@5': val_ndcg5,
            'val/ndcg@10': val_ndcg10,
            'val/batch': batch_idx,
            'step': step,
        })
    
    message = (
        f"\nTime {time_since(ctx.start_time)}, batches {batch_idx}, "
        f"validation AUC: {val_auc:.4f}, MRR: {val_mrr:.4f}, "
        f"nDCG@5: {val_ndcg5:.4f}, nDCG@10: {val_ndcg10:.4f}"
    )
    if should_display_progress():
        tqdm.write(message)
    else:
        print(message)
    
    # Early stopping check
    early_stop, improved = ctx.early_stopping(-val_auc)
    if early_stop:
        message = 'Early stopping triggered due to plateau.'
        if should_display_progress():
            tqdm.write(message)
        else:
            logger.info(message)
        # Log early stopping to wandb
        if ctx.use_wandb:
            wandb.log({'training/early_stopped': True})
        return True, False, None
    
    # Save best checkpoint
    checkpoint_path = None
    if improved:
        checkpoint_path = ctx.config.checkpoint_dir / f'ckpt-{step}.pth'
        save_checkpoint(ctx, step, val_auc, checkpoint_path)
    
    return False, improved, checkpoint_path


def train() -> None:
    """Main training function."""
    ctx = setup_training_context(config)
    
    # Run training loop
    final_batch_idx, best_checkpoint_path, best_step = run_training_loop(ctx)
    
    # Load best checkpoint and evaluate
    if best_checkpoint_path and best_checkpoint_path.exists():
        logger.info(f"Loading best checkpoint from step {best_step}...")
        load_checkpoint(ctx, best_checkpoint_path)
    
    ctx.model.eval()
    final_auc, final_mrr, final_ndcg5, final_ndcg10 = validate(ctx)
    
    # Log final validation metrics to wandb
    if ctx.use_wandb:
        wandb.log({
            'final/auc': final_auc,
            'final/mrr': final_mrr,
            'final/ndcg@5': final_ndcg5,
            'final/ndcg@10': final_ndcg10,
            'final/batch': final_batch_idx,
            'final/step': best_step,
        })
    
    message = (
        f"\n\nTime {time_since(ctx.start_time)}, batches {final_batch_idx}, "
        f"Final AUC: {final_auc:.4f}, Final MRR: {final_mrr:.4f}, "
        f"Final nDCG@5: {final_ndcg5:.4f}, Final nDCG@10: {final_ndcg10:.4f}"
    )
    if should_display_progress():
        tqdm.write(message)
    else:
        print(message)


if __name__ == '__main__':
    device = get_device()
    logger.info(f'Using device: {device}')
    logger.info(f'Training {config.model_type} model')
    train()
