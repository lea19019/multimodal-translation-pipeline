#!/bin/bash
#SBATCH --job-name=eval_all_pipelines
#SBATCH --partition=cs
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=_logs/batch_evaluate_all_%j.log

# Batch evaluation of all pipelines
# This script evaluates all 9 pipelines and generates CSV files with metrics

# Generate single execution ID for this batch run
EXEC_ID=$(date +%Y%m%d_%H%M%S)

echo "========================================================================"
echo "PIPELINE EVALUATION BATCH"
echo "========================================================================"
echo "Execution ID: $EXEC_ID"
echo "Started at: $(date)"
echo ""

# Change to evaluation directory
cd /home/vacl2/multimodal_translation/services/evaluation

# Evaluate each pipeline
for pipeline in pipeline_1 pipeline_2 pipeline_3 pipeline_4 pipeline_5 pipeline_6 pipeline_7 pipeline_8 pipeline_9; do
    echo ""
    echo "========================================================================"
    echo "Evaluating: $pipeline"
    echo "========================================================================"
    
    uv run evaluate_pipelines.py --pipeline $pipeline --execution-id $EXEC_ID
    
    if [ $? -eq 0 ]; then
        echo "✓ $pipeline completed successfully"
    else
        echo "✗ $pipeline failed"
    fi
    echo ""
done

echo ""
echo "========================================================================"
echo "BATCH EVALUATION COMPLETE"
echo "========================================================================"
echo "Execution ID: $EXEC_ID"
echo "Completed at: $(date)"
echo ""
echo "Results directory: /home/vacl2/multimodal_translation/services/evaluation/results"
echo "Check individual pipeline directories: {pipeline_id}_{short_name}_${EXEC_ID}/"
echo "========================================================================"
