"""Dataset classes for news recommendation models."""
import ast
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from ast import literal_eval
from torch.utils.data import Dataset

from config import config


def _parse_tokenized_text(text_str: str) -> torch.Tensor:
    """Parse tokenized text string to tensor.
    
    Args:
        text_str: String representation of tokenized text
        
    Returns:
        Tensor with shape [2, num_words] containing input_ids and attention_mask
    """
    text_dict = ast.literal_eval(text_str)
    input_ids = torch.tensor(text_dict['input_ids'])
    attention_mask = torch.tensor(text_dict['attention_mask'])
    return torch.cat([input_ids.unsqueeze(0), attention_mask.unsqueeze(0)], dim=0)


def _parse_tokenized_title(title_str: str) -> torch.Tensor:
    """Parse tokenized title string to tensor.
    
    Args:
        title_str: String representation of tokenized title
        
    Returns:
        Tensor with shape [2, num_words_title] containing input_ids and attention_mask
    """
    title_dict = ast.literal_eval(title_str)
    input_ids = torch.tensor(title_dict['input_ids'])
    attention_mask = torch.tensor(title_dict['attention_mask'])
    return torch.cat([input_ids.unsqueeze(0), attention_mask.unsqueeze(0)], dim=0)


def _create_padding_token(num_words: int) -> torch.Tensor:
    """Create padding token for text.
    
    Args:
        num_words: Maximum number of words
        
    Returns:
        Padding tensor with shape [2, num_words]
    """
    # [CLS] and [SEP] tokens: [101, 102] with masks [1, 1]
    cls_sep_tokens = [[101, 102], [1, 1]]
    padding = [tokens + [0] * (num_words - 2) for tokens in cls_sep_tokens]
    return torch.tensor(padding)


class BaseDataset(Dataset):
    """Dataset for training news recommendation models."""
    
    def __init__(self, behaviors_path: Path | str, news_path: Path | str, category2int_path: Optional[Path | str] = None) -> None:
        """Initialize dataset.
        
        Args:
            behaviors_path: Path to parsed behaviors TSV file
            news_path: Path to parsed news TSV file
            category2int_path: Path to category2int mapping file
        """
        super().__init__()
        
        # Convert Path objects to strings for pandas
        behaviors_path_str = str(behaviors_path) if isinstance(behaviors_path, Path) else behaviors_path
        news_path_str = str(news_path) if isinstance(news_path, Path) else news_path
        
        # Check if files exist
        if not Path(behaviors_path_str).exists():
            raise FileNotFoundError(f"Behaviors file not found: {behaviors_path_str}")
        if not Path(news_path_str).exists():
            raise FileNotFoundError(f"News file not found: {news_path_str}")
        
        self.behaviors_parsed = pd.read_table(behaviors_path_str, sep='\t')
        
        # Load category2int mapping if available
        self.category2int: Dict[str, int] = {}
        if category2int_path and Path(category2int_path).exists():
            cat_df = pd.read_table(category2int_path, sep='\t')
            self.category2int = dict(cat_df.values)
        
        # Load news data - load all columns that might be needed
        news_cols = ['id', 'title']
        converters = {'title': literal_eval}
        
        if 'category' in config.dataset_attributes.get('news', []):
            news_cols.append('category')
        if 'subcategory' in config.dataset_attributes.get('news', []):
            news_cols.append('subcategory')
        if 'abstract' in config.dataset_attributes.get('news', []):
            news_cols.append('abstract')
            converters['abstract'] = literal_eval
        
        self.news_parsed = pd.read_table(
            news_path_str,
            sep='\t',
            index_col='id',
            usecols=news_cols,
            converters=converters
        )
        
        # Convert news to dictionary
        self.news2dict: Dict[str, Dict[str, torch.Tensor]] = {}
        for news_id, row in self.news_parsed.iterrows():
            news_dict: Dict[str, torch.Tensor] = {
                'title': _parse_tokenized_text(str(row['title']))
            }
            
            if 'category' in row:
                cat_val = str(row['category']).strip()
                news_dict['category'] = torch.tensor(self.category2int.get(cat_val, 0), dtype=torch.long)
            
            if 'subcategory' in row:
                subcat_val = str(row['subcategory']).strip()
                news_dict['subcategory'] = torch.tensor(self.category2int.get(subcat_val, 0), dtype=torch.long)
            
            if 'abstract' in row:
                news_dict['abstract'] = _parse_tokenized_text(str(row['abstract']))
            
            self.news2dict[news_id] = news_dict
        
        # Create padding tokens
        self.padding: Dict[str, torch.Tensor] = {
            'title': _create_padding_token(config.num_words_title)
        }
        if 'abstract' in config.dataset_attributes.get('news', []):
            self.padding['abstract'] = _create_padding_token(config.num_words_abstract)
        if 'category' in config.dataset_attributes.get('news', []):
            self.padding['category'] = torch.tensor(0, dtype=torch.long)
        if 'subcategory' in config.dataset_attributes.get('news', []):
            self.padding['subcategory'] = torch.tensor(0, dtype=torch.long)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.behaviors_parsed)
    
    def __getitem__(self, idx: int) -> Dict[str, List | torch.Tensor]:
        """Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - clicked: List of clicked labels
                - candidate_news: List of candidate news dictionaries
                - clicked_news: List of clicked news dictionaries
                - clicked_news_mask: List of mask values (0 for padding, 1 for real)
                - user: User ID (if LSTUR)
                - clicked_news_length: Actual clicked news length (if LSTUR)
        """
        row = self.behaviors_parsed.iloc[idx]
        
        # Parse clicked labels
        clicked = [int(x) for x in row.clicked.split()]
        
        # Get candidate news
        candidate_news = [
            self.news2dict.get(news_id, self.padding)
            for news_id in row.candidate_news.split()
        ]
        
        # Get clicked news (limit to num_clicked_news_a_user)
        clicked_news_ids = row.clicked_news.split()[:config.num_clicked_news_a_user]
        clicked_news = [
            self.news2dict.get(news_id, self.padding)
            for news_id in clicked_news_ids
        ]
        
        # Count for padding calculation (total IDs attempted)
        num_clicked_ids = len(clicked_news_ids)
        repeated_times = config.num_clicked_news_a_user - num_clicked_ids
        assert repeated_times >= 0, f"Too many clicked news: {num_clicked_ids}"
        
        # Count only news IDs that were actually found (for LSTUR's pack_padded_sequence)
        actual_found_count = sum(1 for news_id in clicked_news_ids if news_id in self.news2dict)
        
        clicked_news = [self.padding] * repeated_times + clicked_news
        # Mask should be 0 for padding (both prefix and missing news substitutes)
        clicked_news_mask = [0] * repeated_times + [1 if news_id in self.news2dict else 0 for news_id in clicked_news_ids]
        
        result = {
            'clicked': clicked,
            'candidate_news': candidate_news,
            'clicked_news': clicked_news,
            'clicked_news_mask': clicked_news_mask,
        }
        
        # Add LSTUR-specific fields if needed
        if 'user' in config.dataset_attributes.get('record', []):
            result['user'] = torch.tensor(row.user, dtype=torch.long)
        
        if 'clicked_news_length' in config.dataset_attributes.get('record', []):
            # Use actual found count for LSTUR's pack_padded_sequence
            result['clicked_news_length'] = torch.tensor(actual_found_count, dtype=torch.long)
        
        return result
