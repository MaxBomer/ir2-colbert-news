#!/bin/bash

# Test script for all model configurations
# Runs quick dry-run tests using BERT Tiny for fast validation
# Note: We don't use 'set -e' so we can continue testing even if one test fails

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track test results
declare -a TEST_RESULTS
declare -a TEST_NAMES

# Function to run a test
run_test() {
    local test_name=$1
    local model_type=$2
    shift 2
    local train_args=("$@")
    
    echo ""
    echo "=========================================="
    echo "Testing: $test_name"
    echo "Model: $model_type"
    echo "=========================================="
    
    TEST_NAMES+=("$test_name")
    
    # Common arguments for all tests
    local common_train_args=(
        --current_data_path data
        --model_type "$model_type"
        --pretrained_model_name prajjwal1/bert-tiny
        --bert_version tiny
        --word_embedding_dim 128
        --num_attention_heads 2
        --test_run
        --batch_size 16
    )
    
    local common_eval_args=(
        --current_data_path data
        --model_type "$model_type"
        --pretrained_model_name prajjwal1/bert-tiny
        --bert_version tiny
        --word_embedding_dim 128
        --num_attention_heads 2
        --batch_size 16
    )
    
    # Combine common args with test-specific args
    local full_train_args=("${common_train_args[@]}" "${train_args[@]}")
    local full_eval_args=("${common_eval_args[@]}" "${train_args[@]}")
    
    if uv run python baseline/train.py "${full_train_args[@]}" 2>&1 | tee /tmp/test_train_$$.log; then
        if uv run python baseline/evaluate.py "${full_eval_args[@]}" 2>&1 | tee /tmp/test_eval_$$.log; then
            TEST_RESULTS+=("PASS")
            echo -e "${GREEN}✓ PASSED: $test_name${NC}"
            return 0
        else
            TEST_RESULTS+=("FAIL_EVAL")
            echo -e "${RED}✗ FAILED (eval): $test_name${NC}"
            return 1
        fi
    else
        TEST_RESULTS+=("FAIL_TRAIN")
        echo -e "${RED}✗ FAILED (train): $test_name${NC}"
        return 1
    fi
}

# Setup environment
echo "Setting up environment..."

# Change to script directory if running from elsewhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Setup uv if needed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies
echo "Syncing dependencies..."
uv sync --locked

# Check CUDA
uv run python - <<'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
EOF
echo ""

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export TOKENIZERS_PARALLELISM=false

# Preprocess data if needed (using BERT Tiny)
if [ ! -f "data/tiny/train/behaviors_parsed.tsv" ]; then
    echo "Data not found at data/tiny/train/behaviors_parsed.tsv"
    echo "Running data download and setup..."
    
    if [ ! -d "data/original" ] || [ -z "$(ls -A data/original 2>/dev/null)" ]; then
        uv run python baseline/download_and_setup_data.py --data_dir data/original || {
            echo -e "${YELLOW}Warning: Data download failed. Continuing with existing data if available.${NC}"
        }
    fi
    
    echo "Running data preprocessing with BERT Tiny..."
    uv run python baseline/data_preprocess.py \
        --original_data_path data/original \
        --bert_version tiny \
        --pretrained_model_name prajjwal1/bert-tiny || {
        echo -e "${RED}Error: Data preprocessing failed!${NC}"
        exit 1
    }
else
    echo "Using existing preprocessed data at data/tiny/"
fi

echo ""
echo "=========================================="
echo "Starting Model Tests"
echo "=========================================="

# Test 1: NRMSbert
run_test "NRMSbert" "NRMSbert" \
    --finetune_layers 2 \
    --learning_rate 1e-4 \
    --dropout_probability 0.1

# Test 2: NAMLbert
run_test "NAMLbert" "NAMLbert" \
    --finetune_layers 2 \
    --num_filters 300 \
    --window_size 3 \
    --learning_rate 1e-4 \
    --dropout_probability 0.1

# Test 3: LSTURbert
run_test "LSTURbert" "LSTURbert" \
    --finetune_layers 2 \
    --num_filters 300 \
    --window_size 3 \
    --long_short_term_method ini \
    --masking_probability 0.5 \
    --learning_rate 1e-4 \
    --dropout_probability 0.1

# Test 4: ColBERT base
run_test "ColBERT Base" "ColBERT" \
    --colbert_embedding_dim 128 \
    --colbert_max_query_tokens 32 \
    --colbert_max_doc_tokens 128 \
    --learning_rate 5e-5 \
    --dropout_probability 0.2

# Test 5: ColBERT + user attention
run_test "ColBERT + Attention" "ColBERT" \
    --colbert_embedding_dim 128 \
    --colbert_max_query_tokens 32 \
    --colbert_max_doc_tokens 32 \
    --colbert_user_attention \
    --colbert_attention_heads 8 \
    --learning_rate 5e-5 \
    --dropout_probability 0.2

# Test 6: ColBERT + position embeddings
run_test "ColBERT + Position" "ColBERT" \
    --colbert_embedding_dim 128 \
    --colbert_max_query_tokens 32 \
    --colbert_max_doc_tokens 32 \
    --colbert_user_attention \
    --colbert_position_embeddings \
    --colbert_attention_heads 8 \
    --learning_rate 5e-5 \
    --dropout_probability 0.2

# Test 7: ColBERT + hierarchical attention
run_test "ColBERT + Hierarchical" "ColBERT" \
    --colbert_embedding_dim 128 \
    --colbert_max_query_tokens 32 \
    --colbert_max_doc_tokens 32 \
    --colbert_hierarchical_attention \
    --colbert_attention_heads 8 \
    --learning_rate 5e-5 \
    --dropout_probability 0.2

# Test 8: ColBERT-NAML
run_test "ColBERT-NAML" "ColBERT-NAML" \
    --colbert_embedding_dim 128 \
    --colbert_max_query_tokens 32 \
    --colbert_max_doc_tokens 32 \
    --num_filters 300 \
    --window_size 3 \
    --learning_rate 5e-5 \
    --dropout_probability 0.2

# Test 9: ColBERT-LSTUR
run_test "ColBERT-LSTUR" "ColBERT-LSTUR" \
    --colbert_embedding_dim 128 \
    --colbert_max_query_tokens 32 \
    --colbert_max_doc_tokens 32 \
    --num_filters 300 \
    --window_size 3 \
    --long_short_term_method ini \
    --masking_probability 0.5 \
    --learning_rate 5e-5 \
    --dropout_probability 0.2

# Test 10: ColBERT zero-shot
run_test "ColBERT Zero-Shot" "ColBERT" \
    --colbert_embedding_dim 128 \
    --colbert_max_query_tokens 32 \
    --colbert_max_doc_tokens 128 \
    --colbert_freeze_weights \
    --learning_rate 1e-3 \
    --dropout_probability 0.2

# Print summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="

total_tests=${#TEST_NAMES[@]}
passed=0
failed_train=0
failed_eval=0

for i in "${!TEST_NAMES[@]}"; do
    result=${TEST_RESULTS[$i]}
    name=${TEST_NAMES[$i]}
    
    case $result in
        "PASS")
            echo -e "${GREEN}✓ PASSED${NC}: $name"
            ((passed++))
            ;;
        "FAIL_TRAIN")
            echo -e "${RED}✗ FAILED (train)${NC}: $name"
            ((failed_train++))
            ;;
        "FAIL_EVAL")
            echo -e "${RED}✗ FAILED (eval)${NC}: $name"
            ((failed_eval++))
            ;;
    esac
done

echo ""
echo "Total tests: $total_tests"
echo -e "${GREEN}Passed: $passed${NC}"
if [ $failed_train -gt 0 ]; then
    echo -e "${RED}Failed (train): $failed_train${NC}"
fi
if [ $failed_eval -gt 0 ]; then
    echo -e "${RED}Failed (eval): $failed_eval${NC}"
fi

# Cleanup temp files
rm -f /tmp/test_train_$$.log /tmp/test_eval_$$.log

# Exit with error code if any tests failed
if [ $failed_train -gt 0 ] || [ $failed_eval -gt 0 ]; then
    exit 1
else
    exit 0
fi

