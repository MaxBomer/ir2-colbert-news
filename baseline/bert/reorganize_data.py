#!/usr/bin/env python3
"""Script to reorganize MIND Small dataset structure.

This script moves files from nested directories (MINDsmall_train, MINDsmall_dev)
to the expected flat structure. It expects train, val, and test directories.
"""
import shutil
from pathlib import Path


def reorganize_mind_data(data_dir: Path) -> None:
    """Reorganize MIND dataset from nested to flat structure.
    
    Args:
        data_dir: Base data directory (e.g., ../data/original)
    """
    data_dir = Path(data_dir)
    
    # Handle train directory
    train_dir = data_dir / 'train'
    if train_dir.exists():
        nested_train = train_dir / 'MINDsmall_train'
        if nested_train.exists():
            print(f"Moving .tsv files from {nested_train} to {train_dir}")
            for file in nested_train.glob('*.tsv'):
                target = train_dir / file.name
                if target.exists():
                    print(f"  Skipping {file.name} (already exists)")
                else:
                    shutil.move(str(file), str(target))
                    print(f"  Moved {file.name}")
            
            # Remove nested directory (we don't need .vec files for NRMS)
            if nested_train.exists():
                shutil.rmtree(nested_train)
                print(f"  Removed nested directory: {nested_train}")
    
    # Handle dev directory
    dev_dir = data_dir / 'dev'
    if dev_dir.exists():
        nested_dev = dev_dir / 'MINDsmall_dev'
        if nested_dev.exists():
            print(f"Moving .tsv files from {nested_dev} to {dev_dir}")
            for file in nested_dev.glob('*.tsv'):
                target = dev_dir / file.name
                if target.exists():
                    print(f"  Skipping {file.name} (already exists)")
                else:
                    shutil.move(str(file), str(target))
                    print(f"  Moved {file.name}")
            
            # Remove nested directory (we don't need .vec files for NRMS)
            if nested_dev.exists():
                shutil.rmtree(nested_dev)
                print(f"  Removed nested directory: {nested_dev}")
    
    # Handle val directory (if it exists from previous splits)
    val_dir = data_dir / 'val'
    if val_dir.exists():
        nested_val = val_dir / 'MINDsmall_train'  # Sometimes val comes from train split
        if nested_val.exists():
            print(f"Moving .tsv files from {nested_val} to {val_dir}")
            for file in nested_val.glob('*.tsv'):
                target = val_dir / file.name
                if target.exists():
                    print(f"  Skipping {file.name} (already exists)")
                else:
                    shutil.move(str(file), str(target))
                    print(f"  Moved {file.name}")
            
            if nested_val.exists():
                shutil.rmtree(nested_val)
                print(f"  Removed nested directory: {nested_val}")
    
    # Handle test directory
    test_dir = data_dir / 'test'
    if test_dir.exists():
        nested_test = test_dir / 'MINDsmall_dev'  # Test comes from dev
        if nested_test.exists():
            print(f"Moving .tsv files from {nested_test} to {test_dir}")
            for file in nested_test.glob('*.tsv'):
                target = test_dir / file.name
                if target.exists():
                    print(f"  Skipping {file.name} (already exists)")
                else:
                    shutil.move(str(file), str(target))
                    print(f"  Moved {file.name}")
            
            if nested_test.exists():
                shutil.rmtree(nested_test)
                print(f"  Removed nested directory: {nested_test}")
    
    print("\nData reorganization complete!")
    print(f"Final structure:")
    print(f"  {data_dir}/train/behaviors.tsv")
    print(f"  {data_dir}/train/news.tsv")
    if (data_dir / 'val').exists():
        print(f"  {data_dir}/val/behaviors.tsv")
        print(f"  {data_dir}/val/news.tsv")
    if (data_dir / 'test').exists():
        print(f"  {data_dir}/test/behaviors.tsv")
        print(f"  {data_dir}/test/news.tsv")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        # Default to ../data/original from baseline directory
        data_dir = Path(__file__).parent.parent / 'data' / 'original'
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print(f"Usage: python reorganize_data.py [data_directory]")
        sys.exit(1)
    
    reorganize_mind_data(data_dir)
