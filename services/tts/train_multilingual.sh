#! /bin/bash
# ===================================================================================


# --- SLURM RESOURCE REQUESTS ---

#SBATCH --time=72:00:00          # Walltime (3 days for 60 epochs)
#SBATCH --partition=cs
#SBATCH --nodes=1                # Number of nodes. YourTTS `distribute` script is for single-node.
#SBATCH --ntasks-per-node=1      # Number of tasks. The python script is a single task.                
#SBATCH --mem=128G               # Total memory for the job. Adjust based on your dataset size and batch size.
#SBATCH -J "train_multilingual_tts"           # Job name
#SBATCH --output=%x_%j.out       # Standard output and error log
#SBATCH --qos=cs
#SBATCH --gpus=2

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
export TRAINER_TELEMETRY=0

# Performance optimizations
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_P2P_DISABLE=1

# Activate your conda or mamba environment where Coqui TTS is installed.
# IMPORTANT: Replace 'seamless' with the name of your TTS environment if it's different.
echo "Activating environment..."
source /home/vacl2/multimodal_translation/services/tts/.venv/bin/activate

# --- 2. Define Paths ---

# IMPORTANT: Set this to the root directory of your cloned Coqui TTS repository.
TTS_REPO_PATH="/home/vacl2/multimodal_translation/services/tts"

# The path to the specific training script you want to run.
TRAIN_SCRIPT_PATH="${TTS_REPO_PATH}/train_gpt_xtts.py"

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


# --- 4. Execute the Training Command ---

# --- 4. Execute the Training Command ---

echo "Starting distributed training on $SLURM_GPUS_ON_NODE GPUs..."

# Use torchrun to launch one training process per GPU.
# --nproc_per_node tells torchrun how many processes (and GPUs) to use.
# torchrun handles setting CUDA_VISIBLE_DEVICES for each process automatically.
# CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
# --output_path checkpoints/ \
# --metadatas data/metadata_train.csv,data/metadata_eval.csv,efi \
# --num_epochs 20 \
# --batch_size 24 \
# --grad_acumm 4 \
# --max_text_length 400 \
# --max_audio_length 330750 \
# --weight_decay 1e-2 \
# --lr 5e-6 \
# --save_step 1000

uv run python -m trainer.distribute \
    --script ${TRAIN_SCRIPT_PATH} \
    --gpus ${SLURM_GPUS_ON_NODE} \
    --output_path checkpoints/ \
    --metadatas /grphome/grp_mtlab/projects/project-speech/african_tts/XTTSv2-Finetuning-for-New-Languages/xhosa/xhosa_cleaned/metadata_train.csv,/grphome/grp_mtlab/projects/project-speech/african_tts/XTTSv2-Finetuning-for-New-Languages/xhosa/xhosa_cleaned/metadata_eval.csv,xho /home/vacl2/multimodal_translation/services/data/languages/efik/metadata_train.csv,/home/vacl2/multimodal_translation/services/data/languages/efik/metadata_eval.csv,efi /home/vacl2/multimodal_translation/services/data/languages/igbo/metadata_train.csv,/home/vacl2/multimodal_translation/services/data/languages/igbo/metadata_eval.csv,ibo /home/vacl2/multimodal_translation/services/data/languages/swahili/metadata_train.csv,/home/vacl2/multimodal_translation/services/data/languages/swahili/metadata_eval.csv,swa \
    --num_epochs 60 \
    --batch_size 48 \
    --grad_acumm 1 \
    --max_text_length 300 \
    --max_audio_length 330750 \
    --weight_decay 1e-2 \
    --lr 5e-6 \
    --save_step 2000 \
    --run_name "MULTILINGUAL_TRAINING_11_4_FAST"


echo "========================================================="
echo "Training script finished."
echo "========================================================="
