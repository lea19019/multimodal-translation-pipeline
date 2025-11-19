#!/bin/bash
# ===================================================================================
# Submit Evaluation Jobs for All 4 Languages with Shared Execution ID
# ===================================================================================
#
# Usage:
#   ./submit_all_languages.sh                    # Auto-generate execution ID
#   ./submit_all_languages.sh eval_20251119_120000  # Use specific execution ID
#
# This script submits 4 separate SLURM jobs (one per language) that will all
# share the same execution ID, allowing their results to be aggregated.
# ===================================================================================

# Generate or use provided execution ID
if [ -z "$1" ]; then
    EXECUTION_ID="eval_$(date +%Y%m%d_%H%M%S)"
    echo "Auto-generated execution ID: $EXECUTION_ID"
else
    EXECUTION_ID="$1"
    echo "Using provided execution ID: $EXECUTION_ID"
fi

echo ""
echo "========================================================="
echo "Submitting evaluation jobs for all languages"
echo "Execution ID: $EXECUTION_ID"
echo "========================================================="
echo ""

# Array of languages to evaluate
LANGUAGES=("efik" "igbo" "swahili" "xhosa")

# Store job IDs
declare -a JOB_IDS

# Submit one job per language
for LANG in "${LANGUAGES[@]}"; do
    echo "Submitting job for: $LANG"
    JOB_OUTPUT=$(sbatch batch_evaluate.sh "$LANG" "$EXECUTION_ID")
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+')
    JOB_IDS+=("$JOB_ID")
    echo "  Job ID: $JOB_ID"
done

echo ""
echo "========================================================="
echo "All jobs submitted successfully!"
echo "========================================================="
echo "Execution ID: $EXECUTION_ID"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  _logs/eval_predictions_*.out"
echo ""
echo "Results will be aggregated in:"
echo "  results/$EXECUTION_ID/"
echo "========================================================="
