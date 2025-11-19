#!/bin/bash
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=nmt_batch_predict
#SBATCH --output=/home/vacl2/multimodal_translation/services/nmt/_logs/batch_predict_%j.out
#SBATCH --qos=cs

echo "=========================================="
echo "NMT Batch Prediction Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Activate virtual environment
source /home/vacl2/multimodal_translation/services/nmt/.venv/bin/activate

# Navigate to script directory
cd /home/vacl2/multimodal_translation/services/nmt/scripts

# Run batch prediction for all languages using uv
# Redirect stderr to stdout so all output goes to .out file
uv run python batch_predict.py \
    --languages efik igbo swahili xhosa \
    --batch-size 64 \
    --device auto \
    --num-beams 5 2>&1

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
