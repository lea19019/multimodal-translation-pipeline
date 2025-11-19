#!/bin/bash
# ===================================================================================
# Multimodal Translation Evaluation - GPU Batch Processing
# ===================================================================================

# --- SLURM RESOURCE REQUESTS ---

#SBATCH --time=12:00:00          # Walltime (12 hours for full evaluation)
#SBATCH --partition=cs
#SBATCH --nodes=1                # Single node
#SBATCH --ntasks-per-node=1      # Single task
#SBATCH --mem=64G                # Memory for COMET and BLASER models
#SBATCH -J "eval_predictions"    # Job name
#SBATCH --output=_logs/%x_%j.out # Standard output and error log
#SBATCH --qos=cs
#SBATCH --gpus=1                 # Single GPU for faster COMET/BLASER
#SBATCH --cpus-per-task=8        # CPUs for data loading

# ===================================================================================
# JOB STEPS
# ===================================================================================

echo "========================================================="
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "JOB NAME: $SLURM_JOB_NAME"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "========================================================="

# Set the max number of threads to use for programs using OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set HuggingFace to offline mode (use cached models only, no internet needed)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Navigate to evaluation directory
cd /home/vacl2/multimodal_translation/services/evaluation

# Create logs directory if it doesn't exist
mkdir -p _logs

# Diagnostics
echo "--- Python and Torch Diagnostics ---"
which python
uv run python -c "import torch; print('PyTorch version:', torch.__version__)"
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
uv run python -c "import torch; print('Number of GPUs:', torch.cuda.device_count())"
echo "------------------------------------"

echo "--- GPU Diagnostics ---"
nvidia-smi
echo "-----------------------"

# Set language to evaluate
LANGUAGE="efik"

# Number of samples to evaluate (for testing)
LIMIT=20

echo "Starting evaluation for language: $LANGUAGE"
echo "Sample limit: $LIMIT"
echo "Data directory: /home/vacl2/multimodal_translation/services/data/languages"
echo ""

# Run evaluation
uv run python evaluation.py \
    --mode predictions \
    --language $LANGUAGE \
    --data-dir /home/vacl2/multimodal_translation/services/data/languages \
    -m bleu -m chrf -m comet -m mcd -m blaser \
    --limit $LIMIT \
    --output-dir results_predictions_${LANGUAGE}_gpu 2>&1

echo ""
echo "========================================================="
echo "Evaluation finished for $LANGUAGE"
echo "========================================================="
