#! /bin/bash
# ===================================================================================


# --- SLURM RESOURCE REQUESTS ---

#SBATCH --time=12:00:00          # Walltime (1 days for 50 epochs)
#SBATCH --partition=cs
#SBATCH --nodes=1                # Number of nodes. YourTTS `distribute` script is for single-node.
#SBATCH --ntasks-per-node=1      # Number of tasks. The python script is a single task.                
#SBATCH --mem=128G               # Total memory for the job. Adjust based on your dataset size and batch size.
#SBATCH -J "src_swahili"           # Job name
#SBATCH --output=_logs/%x_%j.out       # Standard output and error log
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

# Environment setup
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export SCARF_NO_ANALYTICS=true
export TRAINER_TELEMETRY=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

source /home/vacl2/multimodal_translation/services/tts/.venv/bin/activate

# Verify activation worked
echo "Using Python: $(which python)"
python --version
python -c "import trainer; print('✓ trainer found')"

# Make sure venv bin is first in PATH
export PATH="/home/vacl2/multimodal_translation/services/tts/.venv/bin:$PATH"

# Run diagnostics
echo "--- GPU Diagnostics ---"
nvidia-smi
echo "-----------------------"

# Now trainer.distribute should spawn processes using venv python
python -m trainer.distribute \
    --script train_gpt_xtts.py \
    --gpus $SLURM_GPUS_ON_NODE \
    --output_path checkpoints/ \
    --metadatas /home/vacl2/multimodal_translation/services/data/languages/swahili/src_text_to_tgt_audio/metadata_train.csv,/home/vacl2/multimodal_translation/services/data/languages/swahili/src_text_to_tgt_audio/metadata_eval.csv,swa \
    --num_epochs 20 \
    --batch_size 16 \
    --grad_acumm 3 \
    --max_text_length 700 \
    --max_audio_length 330750 \
    --weight_decay 1e-2 \
    --lr 2e-6 \
    --save_step 2000 \
    --run_name "Src_Swahili_14_12" \


echo "========================================================="
echo "Training script finished."
echo "========================================================="
