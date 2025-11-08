"""Utility functions for NRMSbert model."""
import pickle
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
        # Auto-detect: prefer CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            return True  # CUDA supports pin_memory
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return False  # MPS doesn't support pin_memory
        else:
            return False  # CPU doesn't benefit from pin_memory
    
    # Check device type
    if device.type == 'cuda':
        return True
    elif device.type == 'mps':
        return False
    else:
        return False
