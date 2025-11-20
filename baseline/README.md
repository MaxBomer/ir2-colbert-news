# NRMSbert News Recommendation

Welcome to the **NRMSbert** (Neural News Recommendation with Multi-Head Self-Attention using BERT) repository. This project implements a news recommendation model using BERT tiny for efficient training and inference.

## Model Overview

This codebase supports multiple news recommendation models:

- **NRMSbert**: Neural News Recommendation with Multi-Head Self-Attention using BERT
  - **Language Model**: BERT Tiny (`prajjwal1/bert-tiny`) - 128 dimensions, 2 layers
  - **Reference**: [NRMS Paper](https://www.aclweb.org/anthology/D19-1671/)
  
- **ColBERT**: Late-interaction model adapter for news recommendation
  - Uses ColBERT from [PyLate](https://github.com/lightonai/pylate) library
  - Supports any transformer model compatible with ColBERT
  - Converts tokenized inputs to text and uses ColBERT's multi-vector encoding

Models can be selected via `--model_type` argument (default: `NRMSbert`).

## Getting Started

### Step 1: Download and Setup the MIND Small Dataset

Download, extract, and organize the MIND Small dataset from HuggingFace:

```bash
cd baseline

# Download, extract, reorganize, and split the dataset
# This will:
# - Download MINDsmall_train.zip and MINDsmall_dev.zip from HuggingFace
# - Extract and reorganize nested directory structure
# - Split train into train (90%) and val (10%)
# - Move dev to test (validation set becomes test set)
uv run python download_and_setup_data.py

# Or specify a custom data directory
uv run python download_and_setup_data.py --data_dir /path/to/data/original

# Or skip download if files already exist
uv run python download_and_setup_data.py --skip_download
```

**Dataset Structure:**
- **train/**: Training data (90% of original train set)
- **val/**: Validation data (10% of original train set)
- **test/**: Test data (from original dev set)

**Note**: The script automatically splits the training data by users (90/10) to avoid data leakage, and uses the dev set as the test set.

### Step 2: Preprocess the Data

Preprocess the data using BERT tokenization:

```bash
cd baseline

# Preprocess with BERT tiny (default)
uv run python data_preprocess.py \
    --pretrained_model_name prajjwal1/bert-tiny \
    --bert_version tiny \
    --original_data_path ../data/original

# Or preprocess with BERT base
uv run python data_preprocess.py \
    --pretrained_model_name bert-base-uncased \
    --bert_version original \
    --original_data_path ../data/original
```

After preprocessing, the script will print a clear summary with all configuration values:
```
============================================================
CONFIGURATION VALUES - Use these when training:
============================================================
  --num_users {value}
  --num_categories {value}
  --num_words {value}
  --num_entities {value}
============================================================
```

These values are also saved to `{current_data_path}/{bert_version}/config_values.json` for easy reference. **They are automatically loaded when training/evaluating**, so you don't need to pass them as command-line arguments.

**Note**: Processed data is saved to `{current_data_path}/{bert_version}/` (e.g., `data/tiny/`), not in `data/original/`.

### Step 3: Configure Paths

Default paths are set for running from the `baseline/` directory:
- `--current_data_path` defaults to `../data` (where processed data and checkpoints are stored)
- `--original_data_path` defaults to `../data/original` (where raw MIND data is located)

You can override these via command-line arguments if needed.

### Step 4: Train the Model

Train NRMSbert with BERT tiny (recommended for faster training). **Config values are automatically loaded from the JSON file**, so you don't need to pass `--num_users`, `--num_categories`, etc.:

```bash
cd baseline

# Train with BERT tiny (default - faster, lower memory)
# Config values are auto-loaded from data/tiny/config_values.json
uv run python train.py \
    --pretrained_model_name prajjwal1/bert-tiny \
    --bert_version tiny \
    --word_embedding_dim 128 \
    --num_attention_heads 2 \
    --finetune_layers 4 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --dropout_probability 0.2

# Or train with BERT base (slower, higher memory)
uv run python train.py \
    --pretrained_model_name bert-base-uncased \
    --bert_version original \
    --word_embedding_dim 768 \
    --num_attention_heads 16 \
    --finetune_layers 4 \
    --batch_size 16 \
    --learning_rate 0.00001 \
    --dropout_probability 0.2
```

**Note**: Default paths assume you're running from the `baseline/` directory:
- `--current_data_path` defaults to `../data`
- `--original_data_path` defaults to `../data/original`
- Config values (`num_users`, `num_categories`, etc.) are automatically loaded from `{current_data_path}/{bert_version}/config_values.json` if it exists

**Key Parameters:**
- `--pretrained_model_name`: BERT model name or path
  - `prajjwal1/bert-tiny` for BERT tiny (128 dim, 2 layers)
  - `bert-base-uncased` for BERT base (768 dim, 12 layers)
- `--word_embedding_dim`: 
  - `128` for BERT tiny
  - `768` for BERT base
- `--num_attention_heads`: Must divide `word_embedding_dim` evenly
  - `2` for BERT tiny
  - `12` or `16` for BERT base
- `--finetune_layers`: Number of BERT layers to fine-tune
  - BERT tiny: 1-2 layers (only has 2 total)
  - BERT base: 1-12 layers
- `--batch_size`: Adjust based on GPU memory
- `--current_data_path`: Where to save checkpoints and processed data (default: `../data`)
- `--original_data_path`: Path to original MIND dataset (default: `../data/original`)
- **Config values** (`--num_users`, `--num_categories`, `--num_words`, `--num_entities`) are automatically loaded from `{current_data_path}/{bert_version}/config_values.json` if it exists

**Quick Test Run:**

To verify everything works before a full training run:

```bash
cd baseline

# Test NRMSbert: 10 batches, validate on 100 samples
uv run python train.py \
    --model_type NRMSbert \
    --pretrained_model_name prajjwal1/bert-tiny \
    --bert_version tiny \
    --batch_size 16 \
    --test_run

# Test ColBERT: 10 batches, validate on 100 samples
uv run python train.py \
    --model_type ColBERT \
    --pretrained_model_name prajjwal1/bert-tiny \
    --bert_version tiny \
    --batch_size 16 \
    --test_run

# Or limit to specific number of batches
uv run python train.py \
    --pretrained_model_name prajjwal1/bert-tiny \
    --bert_version tiny \
    --batch_size 16 \
    --max_batches 50 \
    --max_validation_samples 1000
```

**Test Model Loading:**

Before training, verify that models can be loaded correctly:

```bash
cd baseline
uv run python test_models.py
```

This will test loading both NRMSbert and ColBERT models to ensure the setup is correct.

### Step 5: Evaluate the Model

Evaluate the trained model:

```bash
cd baseline

# Evaluate with BERT tiny
# Config values are auto-loaded from data/tiny/config_values.json
uv run python evaluate.py \
    --pretrained_model_name prajjwal1/bert-tiny \
    --bert_version tiny \
    --word_embedding_dim 128 \
    --num_attention_heads 2 \
    --finetune_layers 4 \
    --batch_size 16

# Or evaluate with BERT base
uv run python evaluate.py \
    --pretrained_model_name bert-base-uncased \
    --bert_version original \
    --word_embedding_dim 768 \
    --num_attention_heads 16 \
    --finetune_layers 4 \
    --batch_size 16
```

The evaluation outputs:
- **AUC**: Area Under the ROC Curve
- **MRR**: Mean Reciprocal Rank
- **nDCG@5**: Normalized Discounted Cumulative Gain at 5
- **nDCG@10**: Normalized Discounted Cumulative Gain at 10

## Project Structure

```
baseline/
├── config.py              # Configuration (dataclass with type hints)
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── data_preprocess.py      # Data preprocessing
├── dataset.py              # Dataset classes
├── utils.py                # Utility functions
├── model/
│   ├── NRMSbert/          # NRMSbert model implementation
│   │   ├── __init__.py
│   │   ├── news_encoder.py
│   │   └── user_encoder.py
│   └── general/
│       ├── attention/      # Attention mechanisms
│       │   ├── multihead_self.py
│       │   └── additive.py
│       └── click_predictor/
│           └── dot_product.py
└── pyproject.toml         # Dependencies
```

## Features

- **Clean Codebase**: Focused solely on NRMSbert model
- **Type Safety**: Full type hints throughout
- **SOLID Principles**: Modular, maintainable code structure
- **Functional Programming**: Clean, readable code patterns
- **BERT Tiny Support**: Fast training with BERT tiny (4-5x speedup)
- **BERT Base Support**: Full BERT base support for better accuracy

## Training Details

- **Early Stopping**: Patience of 5 validation checks
- **Validation**: Every 1000 batches
- **Checkpoints**: Saved in `{current_data_path}/checkpoint/bert/{bert_version}/NRMSbert/`
- **wandb**: Metrics logged to wandb (requires WANDB_API_KEY environment variable)

## Troubleshooting

1. **FileNotFoundError**: Make sure data paths are correct and data has been preprocessed
2. **CUDA out of memory**: Reduce `--batch_size` or use BERT tiny
3. **BERT model not found**: Update `--pretrained_model_name` to a valid HuggingFace model name
4. **Import errors**: Make sure you're running from the `baseline/` directory

## Credits

- **Dataset**: Microsoft News Dataset (MIND). Learn more at [MIND](https://msnews.github.io/).
- **Model**: Based on NRMS paper: [Neural News Recommendation with Multi-Head Self-Attention](https://www.aclweb.org/anthology/D19-1671/)
- **Original Implementation**: Inspired by [news-recommendation](https://github.com/yusanshi/news-recommendation)
