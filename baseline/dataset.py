"""Dataset classes for NRMSbert model."""
import ast
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from ast import literal_eval
from torch.utils.data import Dataset

from config import config


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


def _create_padding_token(num_words_title: int) -> torch.Tensor:
    """Create padding token for title.
    
    Args:
        num_words_title: Maximum number of words in title
        
    Returns:
        Padding tensor with shape [2, num_words_title]
    """
    # [CLS] and [SEP] tokens: [101, 102] with masks [1, 1]
    cls_sep_tokens = [[101, 102], [1, 1]]
    padding = [tokens + [0] * (num_words_title - 2) for tokens in cls_sep_tokens]
    return torch.tensor(padding)


class BaseDataset(Dataset):
    """Dataset for training NRMSbert model."""
    
    def __init__(self, behaviors_path: Path | str, news_path: Path | str) -> None:
        """Initialize dataset.
        
        Args:
            behaviors_path: Path to parsed behaviors TSV file
            news_path: Path to parsed news TSV file
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
        
        # Load news data (only title needed for NRMS)
        self.news_parsed = pd.read_table(
            news_path_str,
            sep='\t',
            index_col='id',
            usecols=['id', 'title'],
            converters={'title': literal_eval}
        )
        
        # Convert news to dictionary and parse tokenized titles
        self.news2dict: Dict[str, Dict[str, torch.Tensor]] = {}
        for news_id, row in self.news_parsed.iterrows():
            self.news2dict[news_id] = {
                'title': _parse_tokenized_title(str(row['title']))
            }
        
        # Create padding token
        self.padding = {'title': _create_padding_token(config.num_words_title)}
    
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
        """
        row = self.behaviors_parsed.iloc[idx]
        
        # Parse clicked labels
        clicked = [int(x) for x in row.clicked.split()]
        
        # Get candidate news
        candidate_news = [
            self.news2dict[news_id]
            for news_id in row.candidate_news.split()
        ]
        
        # Get clicked news (limit to num_clicked_news_a_user)
        clicked_news_ids = row.clicked_news.split()[:config.num_clicked_news_a_user]
        clicked_news = [
            self.news2dict[news_id]
            for news_id in clicked_news_ids
        ]
        
        # Pad clicked news to fixed length
        clicked_times = len(clicked_news)
        repeated_times = config.num_clicked_news_a_user - clicked_times
        assert repeated_times >= 0, f"Too many clicked news: {clicked_times}"
        
        clicked_news = [self.padding] * repeated_times + clicked_news
        clicked_news_mask = [0] * repeated_times + [1] * clicked_times
        
        return {
            'clicked': clicked,
            'candidate_news': candidate_news,
            'clicked_news': clicked_news,
            'clicked_news_mask': clicked_news_mask,
        }
