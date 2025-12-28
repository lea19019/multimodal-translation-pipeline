#!/bin/bash
# Evaluate synthesized samples from pipeline runs
#
# Usage:
#   # Evaluate a specific CSV file
#   ./evaluate_synthesis.sh efik /path/to/predicted_*.csv
#
#   # Evaluate all CSV files for a language
#   ./evaluate_synthesis.sh efik
#
#   # Evaluate all languages
#   ./evaluate_synthesis.sh all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <language|all> [csv_path]"
    echo ""
    echo "Languages: efik, igbo, swahili, xhosa, all"
    echo ""
    echo "Examples:"
    echo "  # Evaluate all pipelines for Efik"
    echo "  $0 efik"
    echo ""
    echo "  # Evaluate specific CSV"
    echo "  $0 efik /path/to/predicted_nllb_tgt_MULTILINGUAL_TRAINING_*.csv"
    echo ""
    echo "  # Evaluate all languages"
    echo "  $0 all"
    exit 1
fi

LANGUAGE=$1
CSV_PATH=${2:-}

# Function to run evaluation for a language
evaluate_language() {
    local lang=$1
    local csv=$2
    
    echo "========================================="
    echo "Evaluating: $lang"
    echo "========================================="
    
    if [ -n "$csv" ]; then
        cd "$EVAL_DIR" && uv run python scripts/evaluate_pipeline_samples.py \
            --language "$lang" \
            --csv-path "$csv"
    else
        cd "$EVAL_DIR" && uv run python scripts/evaluate_pipeline_samples.py \
            --language "$lang"
    fi
}

# Run evaluation
if [ "$LANGUAGE" = "all" ]; then
    for lang in efik igbo swahili xhosa; do
        evaluate_language "$lang" ""
        echo ""
    done
else
    evaluate_language "$LANGUAGE" "$CSV_PATH"
fi

echo ""
echo "✓ Evaluation complete!"
echo ""
echo "Results are organized by execution ID:"
echo "  services/data/languages/{language}/evaluation_results/{execution_id}/{pipeline_name}/"
echo ""
echo "View visualizations:"
echo "  - metrics_comparison.png   (bar chart of all metrics)"
echo "  - bleu_distribution.png    (BLEU score distribution)"
echo "  - chrf_distribution.png    (chrF++ distribution)"
echo "  - blaser_distribution.png  (BLASER distribution)"
echo "  - metrics.json             (detailed numerical results)"
