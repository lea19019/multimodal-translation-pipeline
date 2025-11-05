#! /bin/bash
# ===================================================================================


# --- SLURM RESOURCE REQUESTS ---

#SBATCH --time=24:00:00          # Walltime (1 hour, adjust as needed)
#SBATCH --nodes=1                # Number of nodes. YourTTS `distribute` script is for single-node.
#SBATCH --ntasks-per-node=1      # Number of tasks. The python script is a single task.
#SBATCH --gpus=1                 # <<< KEY: Number of GPUs to request. Change this number to scale your training.
#SBATCH --mem=128G               # Total memory for the job. Adjust based on your dataset size and batch size.
#SBATCH -J "prep_csv"           # Job name
#SBATCH --output=%x_%j.out       # Standard output and error log

# ===================================================================================
# JOB STEPS
# ===================================================================================

echo "========================================================="
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "JOB NAME: $SLURM_JOB_NAME"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "========================================================="

# --- 1. Set up the environment ---

# Set the max number of threads to use for programs using OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# Turn off online analytics
export SCARF_NO_ANALYTICS=true


# Activate your Python virtual environment (uv/venv)
echo "Activating Python venv..."
source /home/vacl2/multimodal_translation/services/preprocessing_data/.venv/bin/activate

# --- 3. Run Diagnostics (Optional but Recommended) ---

echo "--- Python and Torch Diagnostics ---"
which python
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('Number of GPUs:', torch.cuda.device_count())"
echo "------------------------------------"

echo "--- GPU Diagnostics ---"
nvidia-smi
echo "-----------------------"


# Loop over all languages and process each one
# LANGS="efik igbo swahili xhosa"
LANGS="swahili"
BASE_DIR="/home/vacl2/multimodal_translation/services/data/languages"

for LANG in $LANGS; do
    echo "Processing $LANG..."
    python prep_csv_with_audionorm.py \
        --input_csv "$BASE_DIR/$LANG/recordings.tsv" \
        --output_dir "$BASE_DIR/$LANG/" \
        --denoiser ffmpeg
done