#!/bin/bash
#SBATCH --job-name=pipeline_eval
#SBATCH --partition=cs
#SBATCH --nodes=1                # Number of nodes. YourTTS `distribute` script is for single-node.
#SBATCH --ntasks-per-node=1      # Number of tasks. The python script is a single task.                
#SBATCH --mem=64G  
#SBATCH --output=_logs/evaluation_swa_%j.out
#SBATCH --error=_logs/evaluation_swa_%j.err
#SBATCH --time=24:00:00
#SBATCH --qos=cs
#SBATCH --gpus=2


# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Navigate to evaluation directory
cd /home/vacl2/multimodal_translation/services/evaluation

# Run evaluation for pipelines 1-8 across all 4 languages
echo "Starting pipeline evaluation..."
echo "Pipelines: 10"
echo "Languages: swahili"
echo "Samples per language: 300 (full dataset)"
echo ""

uv run evaluate_pipelines.py \
  -p pipeline_10 \
  -l swahili

exit_code=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $exit_code"
echo "End Time: $(date)"
echo "=========================================="

exit $exit_code
