#!/usr/bin/env python3
"""Download and setup MIND Small dataset from HuggingFace.

This script:
1. Downloads MINDsmall_train.zip and MINDsmall_dev.zip from HuggingFace
2. Extracts the archives
3. Reorganizes nested directory structure
4. Splits train into train (90%) and val (10%)
5. Moves dev to test (validation set becomes test set)
"""
import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def download_file(url: str, output_path: Path) -> None:
    """Download a file from URL.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
    """
    try:
        import urllib.request
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Downloaded to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    print(f"Extracting {zip_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"  ✓ Extracted to {extract_to}")


def reorganize_nested_structure(data_dir: Path) -> None:
    """Move files from nested directories to flat structure.
    
    Args:
        data_dir: Base data directory
    """
    # Handle train directory
    train_dir = data_dir / 'train'
    if train_dir.exists():
        nested_train = train_dir / 'MINDsmall_train'
        if nested_train.exists():
            print(f"Reorganizing train directory...")
            for file in nested_train.glob('*.tsv'):
                target = train_dir / file.name
                if target.exists():
                    print(f"  Skipping {file.name} (already exists)")
                else:
                    shutil.move(str(file), str(target))
                    print(f"  Moved {file.name}")
            
            # Remove nested directory
            if nested_train.exists():
                shutil.rmtree(nested_train)
                print(f"  Removed nested directory: {nested_train}")
    
    # Handle dev directory
    dev_dir = data_dir / 'dev'
    if dev_dir.exists():
        nested_dev = dev_dir / 'MINDsmall_dev'
        if nested_dev.exists():
            print(f"Reorganizing dev directory...")
            for file in nested_dev.glob('*.tsv'):
                target = dev_dir / file.name
                if target.exists():
                    print(f"  Skipping {file.name} (already exists)")
                else:
                    shutil.move(str(file), str(target))
                    print(f"  Moved {file.name}")
            
            # Remove nested directory
            if nested_dev.exists():
                shutil.rmtree(nested_dev)
                print(f"  Removed nested directory: {nested_dev}")


def split_train_val(
    train_behaviors_path: Path,
    train_news_path: Path,
    train_output_dir: Path,
    val_output_dir: Path,
    val_ratio: float = 0.1,
    random_seed: int = 2024,
) -> None:
    """Split training data into train and validation sets.
    
    Args:
        train_behaviors_path: Path to full training behaviors file
        train_news_path: Path to full training news file
        train_output_dir: Directory to save train split
        val_output_dir: Directory to save validation split
        val_ratio: Ratio of validation data (default 0.1 = 10%)
        random_seed: Random seed for reproducibility
    """
    print(f"\nSplitting training data: {val_ratio*100:.1f}% validation, {100-val_ratio*100:.1f}% train")
    
    # Read full training behaviors
    train_behaviors = pd.read_table(
        train_behaviors_path,
        sep='\t',
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions']
    )
    
    # Split by users to avoid data leakage
    unique_users = train_behaviors['user'].unique()
    train_users, val_users = train_test_split(
        unique_users,
        test_size=val_ratio,
        random_state=random_seed
    )
    
    train_mask = train_behaviors['user'].isin(train_users)
    train_split = train_behaviors[train_mask].copy()
    val_split = train_behaviors[~train_mask].copy()
    
    print(f"  Train split: {len(train_split)} behaviors ({len(train_users)} users)")
    print(f"  Val split: {len(val_split)} behaviors ({len(val_users)} users)")
    
    # Save splits
    train_output_dir.mkdir(parents=True, exist_ok=True)
    val_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_split.to_csv(train_output_dir / 'behaviors.tsv', sep='\t', index=False, header=False)
    val_split.to_csv(val_output_dir / 'behaviors.tsv', sep='\t', index=False, header=False)
    
    # Copy news file (same for both splits)
    shutil.copy(train_news_path, train_output_dir / 'news.tsv')
    shutil.copy(train_news_path, val_output_dir / 'news.tsv')
    
    print(f"  ✓ Split complete")


def main() -> None:
    """Main function to download and setup MIND Small dataset."""
    parser = argparse.ArgumentParser(
        description="Download and setup MIND Small dataset from HuggingFace"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Data directory (default: ../data/original from baseline directory)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Ratio of training data to use for validation (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--skip_download',
        action='store_true',
        help='Skip download if files already exist'
    )
    args = parser.parse_args()
    
    # Set up paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / 'data' / 'original'
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs
    train_url = "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip"
    dev_url = "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip"
    
    train_zip = data_dir / 'MINDsmall_train.zip'
    dev_zip = data_dir / 'MINDsmall_dev.zip'
    
    # Download files
    if not args.skip_download:
        if not train_zip.exists():
            download_file(train_url, train_zip)
        else:
            print(f"  ✓ {train_zip.name} already exists, skipping download")
        
        if not dev_zip.exists():
            download_file(dev_url, dev_zip)
        else:
            print(f"  ✓ {dev_zip.name} already exists, skipping download")
    else:
        print("Skipping download (--skip_download flag set)")
    
    # Extract archives
    if train_zip.exists():
        extract_zip(train_zip, data_dir / 'train')
    else:
        raise FileNotFoundError(f"Train zip not found: {train_zip}")
    
    if dev_zip.exists():
        extract_zip(dev_zip, data_dir / 'dev')
    else:
        raise FileNotFoundError(f"Dev zip not found: {dev_zip}")
    
    # Reorganize nested structure
    print("\nReorganizing directory structure...")
    reorganize_nested_structure(data_dir)
    
    # Verify files exist
    train_behaviors = data_dir / 'train' / 'behaviors.tsv'
    train_news = data_dir / 'train' / 'news.tsv'
    dev_behaviors = data_dir / 'dev' / 'behaviors.tsv'
    dev_news = data_dir / 'dev' / 'news.tsv'
    
    if not train_behaviors.exists() or not train_news.exists():
        raise FileNotFoundError(
            f"Train files not found. Expected:\n"
            f"  {train_behaviors}\n"
            f"  {train_news}"
        )
    
    if not dev_behaviors.exists() or not dev_news.exists():
        raise FileNotFoundError(
            f"Dev files not found. Expected:\n"
            f"  {dev_behaviors}\n"
            f"  {dev_news}"
        )
    
    # Split train into train (90%) and val (10%)
    original_train_dir = data_dir / 'train'
    temp_train_dir = data_dir / '_temp_train'
    val_dir = data_dir / 'val'
    
    # Move original train to temp location
    print(f"\nPreparing train/val split...")
    temp_train_dir.mkdir(exist_ok=True)
    shutil.move(str(train_behaviors), temp_train_dir / 'behaviors.tsv')
    shutil.move(str(train_news), temp_train_dir / 'news.tsv')
    
    # Split
    split_train_val(
        temp_train_dir / 'behaviors.tsv',
        temp_train_dir / 'news.tsv',
        original_train_dir,
        val_dir,
        val_ratio=args.val_ratio,
    )
    
    # Remove temp directory
    shutil.rmtree(temp_train_dir)
    print(f"  ✓ Removed temporary directory")
    
    # Move dev to test (validation set becomes test set)
    dev_dir = data_dir / 'dev'
    test_dir = data_dir / 'test'
    if dev_dir.exists() and dev_behaviors.exists() and dev_news.exists():
        print(f"\nMoving dev to test (validation set becomes test set)...")
        test_dir.mkdir(exist_ok=True)
        if (test_dir / 'behaviors.tsv').exists() or (test_dir / 'news.tsv').exists():
            print(f"  Warning: Test directory already has files, skipping move")
        else:
            shutil.move(str(dev_behaviors), test_dir / 'behaviors.tsv')
            shutil.move(str(dev_news), test_dir / 'news.tsv')
            # Remove dev directory if empty
            try:
                dev_dir.rmdir()
                print(f"  ✓ Moved dev to test and removed dev directory")
            except OSError:
                print(f"  ✓ Moved dev to test (dev directory not empty)")
    
    # Clean up zip files
    print(f"\nCleaning up zip files...")
    if train_zip.exists():
        train_zip.unlink()
        print(f"  ✓ Removed {train_zip.name}")
    if dev_zip.exists():
        dev_zip.unlink()
        print(f"  ✓ Removed {dev_zip.name}")
    
    print("\n" + "="*60)
    print("Data setup complete!")
    print("="*60)
    print(f"\nFinal structure:")
    print(f"  {data_dir}/train/behaviors.tsv")
    print(f"  {data_dir}/train/news.tsv")
    print(f"  {data_dir}/val/behaviors.tsv")
    print(f"  {data_dir}/val/news.tsv")
    print(f"  {data_dir}/test/behaviors.tsv")
    print(f"  {data_dir}/test/news.tsv")
    print(f"\nNext step: Run data preprocessing:")
    print(f"  cd baseline")
    print(f"  uv run python data_preprocess.py --original_data_path {data_dir}")


if __name__ == '__main__':
    main()
