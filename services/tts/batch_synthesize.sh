#!/bin/bash
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=tts_batch_synthesize_xho
#SBATCH --output=/home/vacl2/multimodal_translation/services/tts/_logs/batch_synthesize_xho%j.out
#SBATCH --qos=cs

# TTS Batch Synthesis SLURM Job Script
# Runs GPU-accelerated batch audio synthesis for all 4 African languages
#
# Usage:
#   sbatch batch_synthesize.sh
#
# Output:
#   Creates predicted_tgt_audio/ folder with WAV files in each language directory

echo "=========================================="
echo "TTS Batch Synthesis Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Activate virtual environment
source /home/vacl2/multimodal_translation/services/tts/.venv/bin/activate

# Navigate to tts directory (not scripts) so TTS module can be imported
cd /home/vacl2/multimodal_translation/services/tts

# Run batch synthesis for all languages using uv
# Using NMT predictions as input (change to "ground_truth" to use original target text)
# Redirect stderr to stdout so all output goes to .out file
# Use PYTHONPATH to ensure TTS module is found (uv run adds script dir to path, not cwd)
PYTHONPATH=$(pwd):$PYTHONPATH uv run python scripts/batch_synthesize.py \
    --languages xhosa \
    --input-source nmt_predictions \
    --device auto \
    --sample-rate 16000 2>&1

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
