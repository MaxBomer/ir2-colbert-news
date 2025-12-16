# Token-Level Interaction for News Recommendation Using ColBERT

This repository contains the implementation for the paper **"Token-Level Interaction for News Recommendation Using ColBERT"**. 

This project extends the work of Zhao et al. by integrating the **ColBERT** (Contextualized Late Interaction over BERT) architecture into standard neural news recommendation models (NRMS, NAML, LSTUR). Instead of reducing user history and candidate news to single dense vectors (dot-product matching), we leverage ColBERT's late interaction mechanism (MaxSim) to retain fine-grained token-level semantic matching.

### Key References

* **Original Paper Extended:** *Revisiting Language Models in Neural News Recommender Systems* (Zhao et al., 2025)  
    [https://arxiv.org/pdf/2501.11391](https://arxiv.org/pdf/2501.11391)

* **Methodology Integrated:** *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* (Khattab & Zaharia, 2020)  
    [https://arxiv.org/pdf/2004.12832](https://arxiv.org/pdf/2004.12832)

---

## 📂 Project Structure

This repository is organized to separate the recommendation logic (baseline reproduction + extensions) from the core retrieval libraries and cluster execution scripts.



```text
.
├── baseline/                   # MAIN WORKSPACE & EXTENSIONS
│   ├── config.py               # Argument parsing and hyperparameters
│   ├── train.py                # Main training loop
│   ├── evaluate.py             # Evaluation logic
│   ├── model/                  # Neural Network Architectures
│   │   ├── NRMSbert/           # Original NRMS implementation
│   │   ├── NAMLbert/           # Original NAML implementation
│   │   ├── LSTURbert/          # Original LSTUR implementation
│   │   ├── ColBERT/            # NEW: ColBERT-based extensions
│   │   │   ├── lstur_variant.py # ColBERT adapted for LSTUR flow
│   │   │   └── naml_variant.py  # ColBERT adapted for NAML flow
│   │   └── base.py             # Base model class (handles standard ColBERT-NRMS logic)
│   └── ...
│
├── colbert/                    # LOCAL LIBRARY (Pylate)
│   ├── pylate/                 # Core ColBERT encoder/tokenizer/loss logic
│   └── ...                     # This folder acts as the retrieval backbone dependency
│
├── job_scripts/                # SNELLIUS EXECUTION SCRIPTS
│   ├── setup_environment.job   # Environment installation via UV
│   ├── data_preprocessing.job  # Data download and prep
│   ├── pipeline_baseline.job   # Reproduce original BERT baselines
│   ├── pipeline_colbert.job    # Train standard ColBERT-NRMS
│   └── pipeline_*.job          # Train specific ColBERT variants
│
├── pyproject.toml              # Project dependencies (managed by UV)
└── uv.lock                     # Lockfile for reproducible builds

```

### Component Overview

* **`baseline/`**: This is where the core development resides. We cleaned and reproduced the code from the original "Revisiting Language Models" paper. All ColBERT adaptations are implemented here.
* **Standard ColBERT (NRMS-style):** Implemented via the base model logic in the `__init__.py` of `baseline/model/ColBERT/` using flattened user history and MaxSim interactions.
* **Architecture Variants:** Specific implementations for `LSTUR` and `NAML` adaptations using ColBERT encoders can be found in `baseline/model/ColBERT/`.

* **`colbert/`**: This folder contains the `pylate` library source code. It provides the underlying BERT encoders, tokenizers, and loss functions required to generate the specific embeddings ColBERT needs.
* **`job_scripts/`**: Contains `.job` files configured for the Snellius HPC cluster (Slurm).

---

## ⚙️ Environment Setup

This project uses **[uv](https://github.com/astral-sh/uv)** for fast and reliable dependency management.

### Prerequisites

* Python 3.10+
* CUDA-capable GPU (recommended)
* `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

To install the dependencies defined in `pyproject.toml`:

```bash
uv sync
```

---

## 🚀 Running on Snellius (HPC)

We provide pre-configured Slurm scripts in the `job_scripts/` directory to run experiments on the Snellius cluster.

### 1. Environment & Data Setup

First, ensure your environment is built and data is downloaded:

```bash
cd job_scripts
sbatch setup_environment.job
sbatch data_preprocessing.job
```

### 2. Training Pipelines

You can submit training jobs for specific models. The scripts handle GPU allocation and entry point execution.

**Baseline Reproduction (BERT):**

```bash
sbatch pipeline_baseline.job
```

**ColBERT (Standard NRMS-style):**

```bash
sbatch pipeline_colbert.job
```

**ColBERT Variants:**

To test specific architectural changes (e.g., restoring structure via attention or positional embeddings):

```bash
sbatch pipeline_colbert_attention.job      # Adds user self-attention before MaxSim
sbatch pipeline_colbert_hierarchical.job   # Adds hierarchical (token+article) attention
sbatch pipeline_colbert_position.job        # Adds positional embeddings to user history
sbatch pipeline_colbert_lstur.job          # ColBERT encoder within LSTUR architecture
sbatch pipeline_colbert_naml.job           # ColBERT encoder within NAML architecture
sbatch pipeline_colbert_zeroshot.job      # Zero-shot ColBERT (frozen weights)
```

**Original Baselines:**

```bash
sbatch pipeline_lstur.job                  # Original LSTUR with BERT
sbatch pipeline_naml.job                   # Original NAML with BERT
```

### 3. Mass Submission

To run all experiments at once (ensure you have sufficient compute quota):

```bash
bash submit_all.sh
```

---

## 💻 Running Locally

If you are running locally for development, use `uv run` to execute the training script inside the environment.

**Example: Train NRMS with ColBERT extension**

```bash
cd baseline

uv run python train.py \
    --model_type ColBERT \
    --colbert_embedding_dim 128 \
    --batch_size 32 \
    --num_epochs 5
```

**Example: Train ColBERT with User Attention Variant**

```bash
uv run python train.py \
    --model_type ColBERT \
    --colbert_user_attention \
    --batch_size 32
```

---

## 📝 Citation

If you use this code, please cite the original papers:

```bibtex
@article{zhao2025revisiting,
  title={Revisiting Language Models in Neural News Recommender Systems},
  author={Zhao, Yuyue and Huang, Jin and Vos, David and de Rijke, Maarten},
  journal={arXiv preprint arXiv:2501.11391},
  year={2025}
}

@article{khattab2020colbert,
  title={ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT},
  author={Khattab, Omar and Zaharia, Matei},
  journal={arXiv preprint arXiv:2004.12832},
  year={2020}
}
```