#! /bin/bash
# ===================================================================================


# --- SLURM RESOURCE REQUESTS ---

#SBATCH --time=24:00:00          # Walltime (3 days, adjust as needed)
#SBATCH --partition=cs
#SBATCH --nodes=1                # Number of nodes. YourTTS `distribute` script is for single-node.
#SBATCH --ntasks-per-node=1      # Number of tasks. The python script is a single task.                
#SBATCH --mem=64G               # Total memory for the job. Adjust based on your dataset size and batch size.
#SBATCH -J "train_multilingual"           # Job name
#SBATCH --output=%x_%j.out       # Standard output and error log
#SBATCH --qos=cs
#SBATCH --gpus=1

# ===================================================================================
# JOB STEPS
# ===================================================================================

echo "========================================================="
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "JOB NAME: $SLURM_JOB_NAME"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "========================================================="

# Set the max number of threads to use for programs using OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

echo "Activating environment..."
source /home/vacl2/multimodal_translation/services/nmt/.venv/bin/activate

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


uv run python nllb_nmt.py


echo "========================================================="
echo "Training script finished."
echo "========================================================="
