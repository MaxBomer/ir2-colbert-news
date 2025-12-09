#!/bin/bash
# Master launcher script to submit all model training jobs to SLURM queue
# Usage: ./job_scripts/submit_all.sh [--dry-run]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}DRY RUN MODE - No jobs will be submitted${NC}"
    echo ""
fi

cd "$(dirname "$0")/.."

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Submitting All Model Training Jobs   ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Setup job (runs first to download model and data)
SETUP_SCRIPT="job_scripts/setup_environment.job"

# Array of all job scripts
declare -a JOB_SCRIPTS=(
    # Baseline BERT models
    "job_scripts/pipeline_baseline.job"      # NRMSbert
    "job_scripts/pipeline_naml.job"          # NAMLbert  
    "job_scripts/pipeline_lstur.job"         # LSTURbert
    
    # ColBERT variants
    "job_scripts/pipeline_colbert.job"              # ColBERT Base
    "job_scripts/pipeline_colbert_attention.job"    # ColBERT + Attention
    "job_scripts/pipeline_colbert_position.job"     # ColBERT + Position
    "job_scripts/pipeline_colbert_hierarchical.job" # ColBERT + Hierarchical
    "job_scripts/pipeline_colbert_zeroshot.job"     # ColBERT Zero-Shot
    
    # ColBERT hybrid variants
    "job_scripts/pipeline_colbert_naml.job"   # ColBERT-NAML
    "job_scripts/pipeline_colbert_lstur.job"  # ColBERT-LSTUR
)

# Track submitted jobs
declare -a SUBMITTED_JOBS
declare -a JOB_IDS

echo "Jobs to submit:"
echo "---------------"

# Show setup job first
if [ -f "$SETUP_SCRIPT" ]; then
    job_name=$(grep -m1 "#SBATCH --job-name=" "$SETUP_SCRIPT" | cut -d'=' -f2)
    partition=$(grep -m1 "#SBATCH --partition=" "$SETUP_SCRIPT" | cut -d'=' -f2)
    time=$(grep -m1 "#SBATCH --time=" "$SETUP_SCRIPT" | cut -d'=' -f2)
    echo -e "  ${GREEN}$job_name${NC} ($partition, $time) [SETUP - runs first]"
else
    echo -e "  ${YELLOW}WARNING: $SETUP_SCRIPT not found${NC}"
fi

# Show training jobs
for script in "${JOB_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        job_name=$(grep -m1 "#SBATCH --job-name=" "$script" | cut -d'=' -f2)
        partition=$(grep -m1 "#SBATCH --partition=" "$script" | cut -d'=' -f2)
        time=$(grep -m1 "#SBATCH --time=" "$script" | cut -d'=' -f2)
        echo -e "  ${GREEN}$job_name${NC} ($partition, $time)"
    else
        echo -e "  ${YELLOW}WARNING: $script not found${NC}"
    fi
done
echo ""

if [ "$DRY_RUN" = false ]; then
    read -p "Submit all jobs? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

# Submit jobs
echo "Submitting jobs..."
echo "------------------"

# First, submit setup job (downloads model and data)
SETUP_JOB_ID=""
if [ -f "$SETUP_SCRIPT" ]; then
    job_name=$(grep -m1 "#SBATCH --job-name=" "$SETUP_SCRIPT" | cut -d'=' -f2)
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "  [DRY RUN] Would submit setup: ${GREEN}$job_name${NC}"
        SETUP_JOB_ID="DRYRUN"
    else
        output=$(sbatch "$SETUP_SCRIPT" 2>&1)
        if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            SETUP_JOB_ID="${BASH_REMATCH[1]}"
            JOB_IDS+=("$SETUP_JOB_ID")
            SUBMITTED_JOBS+=("$job_name")
            echo -e "  Submitted setup: ${GREEN}$job_name${NC} (Job ID: $SETUP_JOB_ID)"
        else
            echo -e "  ${YELLOW}Failed to submit setup job: $output${NC}"
            echo -e "  ${YELLOW}Cannot proceed without setup job${NC}"
            exit 1
        fi
    fi
else
    echo -e "  ${YELLOW}Setup script not found: $SETUP_SCRIPT${NC}"
    exit 1
fi

echo ""
echo "Submitting training jobs (depend on setup job $SETUP_JOB_ID)..."
echo "---------------------------------------------------------------"

# Submit training jobs with dependency on setup job
for script in "${JOB_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        job_name=$(grep -m1 "#SBATCH --job-name=" "$script" | cut -d'=' -f2)
        
        if [ "$DRY_RUN" = true ]; then
            echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after setup)"
        else
            # Submit with dependency on setup job
            output=$(sbatch --dependency=afterok:$SETUP_JOB_ID "$script" 2>&1)
            if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
                job_id="${BASH_REMATCH[1]}"
                JOB_IDS+=("$job_id")
                SUBMITTED_JOBS+=("$job_name")
                echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on $SETUP_JOB_ID)"
            else
                echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
            fi
        fi
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Summary                               ${NC}"
echo -e "${BLUE}========================================${NC}"

TOTAL_JOBS=$((${#JOB_SCRIPTS[@]} + 1))  # +1 for setup job

if [ "$DRY_RUN" = true ]; then
    echo -e "Would submit ${GREEN}$TOTAL_JOBS${NC} jobs (1 setup + ${#JOB_SCRIPTS[@]} training)"
else
    echo -e "Submitted ${GREEN}${#SUBMITTED_JOBS[@]}${NC} jobs (1 setup + $((${#SUBMITTED_JOBS[@]} - 1)) training)"
    echo ""
    echo "Setup job: $SETUP_JOB_ID (runs first)"
    echo "Training jobs: ${JOB_IDS[*]:1}"
    echo ""
    echo "Training jobs will wait for setup to complete (Dependency status in squeue)"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Cancel all with:"
    echo "  scancel ${JOB_IDS[*]}"
fi

