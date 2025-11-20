"""Data preprocessing for NRMSbert model."""
import argparse
import csv
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer

from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _check_file_exists(file_path: Path, file_type: str = "file") -> Path:
    """Check if file exists, return clean error if not.
    
    Args:
        file_path: Path to check
        file_type: Type of file for error message
        
    Returns:
        Path object if exists
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_type.capitalize()} not found: {file_path}\n"
            f"Expected location: {file_path.absolute()}"
        )
    return file_path


def _find_mind_files(base_dir: Path) -> Dict[str, Path]:
    """Find MIND dataset files, handling nested directories.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        Dictionary with 'behaviors' and 'news' file paths
        
    Raises:
        FileNotFoundError: If files not found
    """
    # Try direct path first
    behaviors_path = base_dir / 'behaviors.tsv'
    news_path = base_dir / 'news.tsv'
    
    # If not found, check nested directories (MINDsmall_train, MINDsmall_dev, etc.)
    if not behaviors_path.exists():
        nested_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        for nested_dir in nested_dirs:
            potential_behaviors = nested_dir / 'behaviors.tsv'
            potential_news = nested_dir / 'news.tsv'
            if potential_behaviors.exists() and potential_news.exists():
                behaviors_path = potential_behaviors
                news_path = potential_news
                logger.info(f"Found MIND files in nested directory: {nested_dir}")
                break
    
    behaviors_path = _check_file_exists(behaviors_path, "behaviors file")
    news_path = _check_file_exists(news_path, "news file")
    
    return {'behaviors': behaviors_path, 'news': news_path}


def parse_behaviors(source: Path, target: Path, user2int_path: Path, negative_sampling_ratio: int, user2int_mapping: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Parse behaviors file in training set.
    
    Args:
        source: Source behaviors file path
        target: Target behaviors file path
        user2int_path: Path for saving user2int file
        negative_sampling_ratio: Ratio of negative samples to positive samples
        user2int_mapping: Existing user2int mapping (for val/test sets)
        
    Returns:
        user2int mapping dictionary
    """
    source = _check_file_exists(source, "behaviors file")
    source_str = str(source)
    
    logger.info(f"Parsing behaviors from: {source_str}")

    behaviors = pd.read_table(
        source_str,
        sep='\t',
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions']
    )
    behaviors.clicked_news.fillna(' ', inplace=True)
    behaviors.impressions = behaviors.impressions.str.split()

    # Create or use existing user2int mapping
    if user2int_mapping is None:
        user2int: Dict[str, int] = {}
        for row in behaviors.itertuples(index=False):
            if row.user not in user2int:
                user2int[row.user] = len(user2int) + 1
    else:
        user2int = user2int_mapping.copy()
        # Add any new users (shouldn't happen for val/test, but handle gracefully)
        for row in behaviors.itertuples(index=False):
            if row.user not in user2int:
                user2int[row.user] = len(user2int) + 1
                logger.warning(f"New user found in validation/test set: {row.user}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if user2int_mapping is None:  # Only save for training set
        pd.DataFrame(user2int.items(), columns=['user', 'int']).to_csv(
            user2int_path, sep='\t', index=False
        )

    # Convert user IDs to integers
    for row in behaviors.itertuples():
        behaviors.at[row.Index, 'user'] = user2int.get(row.user, 0)  # Use 0 for unknown users

    # Balance data with negative sampling
    for row in tqdm(behaviors.itertuples(), desc="Balancing data"):
        positive = iter([x for x in row.impressions if x.endswith('1')])
        negative = [x for x in row.impressions if x.endswith('0')]
        random.shuffle(negative)
        negative = iter(negative)
        pairs = []
        try:
            while True:
                pair = [next(positive)]
                for _ in range(negative_sampling_ratio):
                    pair.append(next(negative))
                pairs.append(pair)
        except StopIteration:
            pass
        behaviors.at[row.Index, 'impressions'] = pairs

    behaviors = behaviors.explode('impressions').dropna(
        subset=["impressions"]
    ).reset_index(drop=True)
    
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (
                ' '.join([e.split('-')[0] for e in x]),
                ' '.join([e.split('-')[1] for e in x])
            )
        ).tolist()
    )
    
    behaviors.to_csv(
        target,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked']
    )
    
    return user2int


def parse_news(
    source: Path,
    target: Path,
    tokenizer: BertTokenizer,
    num_words_title: int,
    mode: str,
    category2int_path: Path | None = None,
    word2int_path: Path | None = None,
    entity2int_path: Path | None = None,
) -> Tuple[int, int, int]:
    """Parse news for training set and test set.
    
    Args:
        source: Source news file path
        target: Target news file path
        tokenizer: BERT tokenizer instance
        num_words_title: Maximum number of words in title
        mode: 'train' or 'test'
        category2int_path: Path to save/load category2int mapping (train mode)
        word2int_path: Path to save/load word2int mapping (train mode)
        entity2int_path: Path to save/load entity2int mapping (train mode)
    """
    source = _check_file_exists(source, "news file")
    source_str = str(source)
    
    logger.info(f"Parsing news from: {source_str}")
    
    # Read only id and title columns (NRMS only needs title)
    news = pd.read_table(
        source_str,
        sep='\t',
        header=None,
        usecols=[0, 3],  # id and title
        quoting=csv.QUOTE_NONE,
        names=['id', 'title']
    )
    news.fillna(' ', inplace=True)

    def tokenize_title(title: str) -> Dict[str, list]:
        """Tokenize a single title."""
        return tokenizer(
            title.lower(),
            max_length=num_words_title,
            padding='max_length',
            truncation=True
        )

    # Tokenize titles
    parsed_news = news.copy()
    parsed_news['title'] = parsed_news['title'].apply(tokenize_title)
    
    target.parent.mkdir(parents=True, exist_ok=True)
    parsed_news.to_csv(target, sep='\t', index=False)

    if mode == 'train':
        # Create minimal mappings for compatibility (not used by NRMS)
        category2int: Dict[str, int] = {'default': 1}
        word2int: Dict[str, int] = {'default': 1}
        entity2int: Dict[str, int] = {'default': 1}
        
        if category2int_path:
            pd.DataFrame(category2int.items(), columns=['category', 'int']).to_csv(
                category2int_path, sep='\t', index=False
            )
        
        if word2int_path:
            pd.DataFrame(word2int.items(), columns=['word', 'int']).to_csv(
                word2int_path, sep='\t', index=False
            )
        
        if entity2int_path:
            pd.DataFrame(entity2int.items(), columns=['entity', 'int']).to_csv(
                entity2int_path, sep='\t', index=False
            )
        
        return len(category2int), len(word2int), len(entity2int)
    
    elif mode != 'test':
        raise ValueError(f'Invalid mode: {mode}. Must be "train" or "test"')
    
    return 0, 0, 0


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
    logger.info(f"Splitting training data: {val_ratio*100:.1f}% validation, {100-val_ratio*100:.1f}% train")
    
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
    
    logger.info(f"Train split: {len(train_split)} behaviors, Val split: {len(val_split)} behaviors")
    
    # Save splits
    train_output_dir.mkdir(parents=True, exist_ok=True)
    val_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_split.to_csv(train_output_dir / 'behaviors.tsv', sep='\t', index=False, header=False)
    val_split.to_csv(val_output_dir / 'behaviors.tsv', sep='\t', index=False, header=False)
    
    # Copy news file (same for both splits)
    shutil.copy(train_news_path, train_output_dir / 'news.tsv')
    shutil.copy(train_news_path, val_output_dir / 'news.tsv')
    
    logger.info(f"Split complete. Train: {train_output_dir}, Val: {val_output_dir}")


def main() -> None:
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess data for NRMSbert")
    parser.add_argument(
        '--pretrained_model_name',
        type=str,
        default='prajjwal1/bert-tiny',
        help='Pretrained BERT model name or path'
    )
    parser.add_argument(
        '--bert_version',
        type=str,
        default='tiny',
        help='BERT version identifier'
    )
    parser.add_argument(
        '--original_data_path',
        type=str,
        default='data_downsampled_20k/original',
        help='Path to original data directory'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Ratio of training data to use for validation (default: 0.1 = 10%%)'
    )
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        legacy=False,
        clean_up_tokenization_spaces=True
    )

    # Input paths: raw MIND data
    original_base = Path(args.original_data_path)
    
    if not original_base.exists():
        raise FileNotFoundError(
            f"Data directory not found: {original_base}\n"
            f"Please ensure MIND Small dataset is downloaded and extracted."
        )
    
    # Find MIND files (handle nested directories)
    train_files = _find_mind_files(original_base / 'train')
    
    # Check if val directory already exists (data already split)
    val_dir = original_base / 'val'
    val_files_exist = val_dir.exists() and (val_dir / 'behaviors.tsv').exists() and (val_dir / 'news.tsv').exists()
    
    # Find test files (prefer test over dev)
    test_dir = original_base / 'test'
    dev_dir = original_base / 'dev'
    if test_dir.exists():
        test_files = _find_mind_files(test_dir)
        logger.info("Using test directory for test set")
    elif dev_dir.exists():
        test_files = _find_mind_files(dev_dir)
        logger.info("Using dev directory as test set")
    else:
        raise FileNotFoundError(
            f"Neither test nor dev directory found in {original_base}\n"
            f"Expected: {test_dir} or {dev_dir}"
        )
    
    # Output paths: processed data organized by bert_version
    # Processed data goes in data/{bert_version}, not data/original/{bert_version}
    output_base = original_base.parent / args.bert_version
    train_output_dir = output_base / 'train'
    val_output_dir = output_base / 'val'
    test_output_dir = output_base / 'test'
    
    # Create output directories
    train_output_dir.mkdir(parents=True, exist_ok=True)
    val_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split train into train/val if val doesn't already exist
    if val_files_exist:
        logger.info("=" * 60)
        logger.info("Step 1: Using existing train/val split")
        logger.info("=" * 60)
        logger.info("Val directory already exists, skipping split")
        # Copy existing files to output directories
        shutil.copy(train_files['behaviors'], train_output_dir / 'behaviors.tsv')
        shutil.copy(train_files['news'], train_output_dir / 'news.tsv')
        val_files = _find_mind_files(val_dir)
        shutil.copy(val_files['behaviors'], val_output_dir / 'behaviors.tsv')
        shutil.copy(val_files['news'], val_output_dir / 'news.tsv')
    else:
        logger.info("=" * 60)
        logger.info("Step 1: Splitting training data into train/val")
        logger.info("=" * 60)
        # Create temporary directories for train/val split
        temp_train_dir = original_base / '_temp_train'
        temp_train_dir.mkdir(exist_ok=True)
        
        # Copy original train files to temp location
        shutil.copy(train_files['behaviors'], temp_train_dir / 'behaviors.tsv')
        shutil.copy(train_files['news'], temp_train_dir / 'news.tsv')
        
        split_train_val(
            temp_train_dir / 'behaviors.tsv',
            temp_train_dir / 'news.tsv',
            train_output_dir,
            val_output_dir,
            val_ratio=args.val_ratio,
        )
        
        # Clean up temp directory
        if temp_train_dir.exists():
            shutil.rmtree(temp_train_dir)
    
    # Process training data
    logger.info("=" * 60)
    logger.info("Step 2: Processing training data")
    logger.info("=" * 60)
    user2int = parse_behaviors(
        train_output_dir / 'behaviors.tsv',
        train_output_dir / 'behaviors_parsed.tsv',
        train_output_dir / 'user2int.tsv',
        config.negative_sampling_ratio
    )
    
    num_categories, num_words, num_entities = parse_news(
        train_output_dir / 'news.tsv',
        train_output_dir / 'news_parsed.tsv',
        tokenizer,
        config.num_words_title,
        mode='train',
        category2int_path=train_output_dir / 'category2int.tsv',
        word2int_path=train_output_dir / 'word2int.tsv',
        entity2int_path=train_output_dir / 'entity2int.tsv',
    )
    
    # Process validation data (use same user2int mapping)
    logger.info("=" * 60)
    logger.info("Step 3: Processing validation data")
    logger.info("=" * 60)
    parse_behaviors(
        val_output_dir / 'behaviors.tsv',
        val_output_dir / 'behaviors_parsed.tsv',
        val_output_dir / 'user2int.tsv',
        config.negative_sampling_ratio,
        user2int_mapping=user2int
    )
    
    parse_news(
        val_output_dir / 'news.tsv',
        val_output_dir / 'news_parsed.tsv',
        tokenizer,
        config.num_words_title,
        mode='test',
        category2int_path=train_output_dir / 'category2int.tsv',
        word2int_path=train_output_dir / 'word2int.tsv',
        entity2int_path=train_output_dir / 'entity2int.tsv',
    )
    
    # Process test data
    logger.info("=" * 60)
    logger.info("Step 4: Processing test data")
    logger.info("=" * 60)
    # Copy test behaviors to test output (no parsing needed, just copy structure)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(test_files['behaviors'], test_output_dir / 'behaviors.tsv')
    
    parse_news(
        test_files['news'],
        test_output_dir / 'news_parsed.tsv',
        tokenizer,
        config.num_words_title,
        mode='test',
        category2int_path=train_output_dir / 'category2int.tsv',
        word2int_path=train_output_dir / 'word2int.tsv',
        entity2int_path=train_output_dir / 'entity2int.tsv',
    )
    
    # Calculate final values (1 + length for 0-indexing)
    num_users_final = 1 + len(user2int)
    num_categories_final = 1 + num_categories
    num_words_final = 1 + num_words
    num_entities_final = 1 + num_entities
    
    # Print clear summary
    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"Processed data saved to: {output_base}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONFIGURATION VALUES - Use these when training:")
    logger.info("=" * 60)
    logger.info(f"  --num_users {num_users_final}")
    logger.info(f"  --num_categories {num_categories_final}")
    logger.info(f"  --num_words {num_words_final}")
    logger.info(f"  --num_entities {num_entities_final}")
    logger.info("=" * 60)
    logger.info("")
    
    # Save to JSON file for easy reference
    config_values = {
        'num_users': num_users_final,
        'num_categories': num_categories_final,
        'num_words': num_words_final,
        'num_entities': num_entities_final
    }
    config_json_path = output_base / 'config_values.json'
    with open(config_json_path, 'w') as f:
        json.dump(config_values, f, indent=2)
    logger.info(f"Configuration values saved to: {config_json_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
