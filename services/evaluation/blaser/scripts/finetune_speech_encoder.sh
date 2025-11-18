#!/bin/bash
# ===================================================================================
# SONAR Speech Encoder Fine-tuning
# ===================================================================================

# --- SLURM RESOURCE REQUESTS ---

#SBATCH --time=24:00:00          # Walltime
#SBATCH --partition=cs
#SBATCH --nodes=1                # Single node
#SBATCH --ntasks-per-node=1      # Single task
#SBATCH --mem=64G                # Memory for single GPU training
#SBATCH -J "sonar_finetune"      # Job name
#SBATCH --output=%x_%j.out       # Standard output and error log
#SBATCH --qos=cs
#SBATCH --gpus=1                 # Single GPU
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

echo "Activating environment..."
source /home/vacl2/multimodal_translation/services/evaluation/blaser/.venv/bin/activate

# Diagnostics
echo "--- Python and Torch Diagnostics ---"
which python
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('Number of GPUs:', torch.cuda.device_count())"
echo "------------------------------------"

echo "--- GPU Diagnostics ---"
nvidia-smi
echo "-----------------------"

# Navigate to working directory
cd /home/vacl2/multimodal_translation/services/evaluation/blaser

echo "Starting fine-tuning..."
# if [ "$SLURM_GPUS_ON_NODE" -eq 1 ]; then
    # echo "Single GPU mode - running directly"
uv run python finetune_speech_encoder.py
# else
#     echo "Multi-GPU mode - using torchrun"
#     torchrun \
#         --nproc_per_node=$SLURM_GPUS_ON_NODE \
#         --nnodes=1 \
#         --node_rank=0 \
#         finetune_speech_encoder.py
# fi

echo "========================================================="
echo "Fine-tuning script finished."
echo "========================================================="