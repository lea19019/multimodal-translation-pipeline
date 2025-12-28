#!/bin/bash
#
# Run all 32 syntheses locally in background (CPU-based)
# 8 pipelines × 4 languages × 50 samples = 32 synthesis jobs
#
# Usage:
#   ./run_local_background.sh [n_samples]
#
# Example:
#   ./run_local_background.sh 50
#

# Configuration
N_SAMPLES=${1:-50}
LOG_DIR="/home/vacl2/multimodal_translation/services/evaluation/_logs"
SCRIPT_DIR="/home/vacl2/multimodal_translation/services/evaluation/scripts"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/local_synthesis_${TIMESTAMP}.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "LOCAL BACKGROUND SYNTHESIS"
echo "=========================================="
echo "Samples per language: $N_SAMPLES"
echo "Pipelines: 8"
echo "Languages: 4"
echo "Total syntheses: 32"
echo "Device: CPU"
echo "Log file: $LOG_FILE"
echo "PID file: $LOG_DIR/synthesis.pid"
echo ""
echo "Estimated time: ~$(awk "BEGIN {print int($N_SAMPLES * 32 * 2.5 / 60)}")+ minutes"
echo ""
echo "Starting in background..."
echo "=========================================="

# Run in background and save PID
nohup python "$SCRIPT_DIR/run_local_test_syntheses.py" \
    --n-samples "$N_SAMPLES" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$LOG_DIR/synthesis.pid"

echo "✓ Started with PID: $PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Or check progress:"
echo "  grep -E 'Pipeline|✓|✗|COMPLETE' $LOG_FILE"
echo ""
echo "Stop synthesis:"
echo "  kill $PID"
echo "  # or: kill \$(cat $LOG_DIR/synthesis.pid)"
echo ""
echo "Check if still running:"
echo "  ps -p $PID"
echo ""
