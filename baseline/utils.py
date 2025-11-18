"""Utility functions for NRMSbert model."""
import pickle
import sys
from pathlib import Path
from typing import Any

import torch


def save_news_dataset(file_path: Path | str, data: Any) -> None:
    """Save news dataset to pickle file.
    
    Args:
        file_path: Path to save the dataset
        data: Dataset data to save
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_news_dataset(file_path: Path | str) -> Any:
    """Load news dataset from pickle file.
    
    Args:
        file_path: Path to load the dataset from
        
    Returns:
        Loaded dataset data
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def get_device() -> torch.device:
    """Auto-detect and return the best available device.
    
    Prefers CUDA, then MPS, then CPU.
    
    Returns:
        torch.device instance
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def should_pin_memory(device: torch.device | None = None) -> bool:
    """Determine if pin_memory should be enabled for DataLoader.
    
    pin_memory is beneficial for CUDA but not supported on MPS (Apple Silicon).
    This function automatically detects the device and returns the appropriate value.
    
    Args:
        device: Optional torch device. If None, checks available devices.
        
    Returns:
        True if pin_memory should be enabled (CUDA), False otherwise (MPS/CPU)
    """
    if device is None:
        device = get_device()
    
    # Check device type
    if device.type == 'cuda':
        return True
    elif device.type == 'mps':
        return False
    else:
        return False


def should_display_progress() -> bool:
    """Check if progress bars should be displayed.
    
    Returns:
        True if output is a TTY, False otherwise
    """
    return sys.stdout.isatty()
