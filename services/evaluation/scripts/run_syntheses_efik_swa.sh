#!/bin/bash
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=synthesis_all
#SBATCH --output=/home/vacl2/multimodal_translation/services/evaluation/_logs/synthesis_e_s_%j.out
#SBATCH --error=/home/vacl2/multimodal_translation/services/evaluation/_logs/synthesis_e_s_%j.err

echo "=========================================="
echo "TTS Synthesis: 2 Pipelines × 1 Language = 2 Syntheses"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Activate TTS virtual environment
source /home/vacl2/multimodal_translation/services/tts/.venv/bin/activate

# Navigate to TTS directory
cd /home/vacl2/multimodal_translation/services/tts

# Track completion
TOTAL=32
COMPLETED=0
FAILED=0

# Helper function to run synthesis
run_synthesis() {
    local PIPELINE_ID=$1
    local PIPELINE_NAME=$2
    local LANG_ISO=$3
    local LANG_NAME=$4
    local CSV_PATH=$5
    local DESCRIPTOR=$6
    local MODEL_PATH=$7
    
    echo ""
    echo "=========================================="
    echo "Pipeline $PIPELINE_ID ($PIPELINE_NAME) - $LANG_NAME"
    echo "Start: $(date)"
    echo "=========================================="
    
    PYTHONPATH=$(pwd):$PYTHONPATH uv run python batch_synthesize.py \
        --csv-file "$CSV_PATH" \
        --output-dir /home/vacl2/multimodal_translation/services/data/languages \
        --language "$LANG_ISO" \
        --model-path "$MODEL_PATH" \
        --descriptor "$DESCRIPTOR" \
        --device auto \
        --sample-rate 16000 \
        --max-samples 300 \
        --text-column text
    
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: Pipeline $PIPELINE_ID - $LANG_NAME"
        ((COMPLETED++))
    else
        echo "✗ FAILED: Pipeline $PIPELINE_ID - $LANG_NAME"
        ((FAILED++))
    fi
    
    echo "Progress: $COMPLETED/$TOTAL completed, $FAILED failed"
}

# Base paths
FORMATTED_DIR="/home/vacl2/multimodal_translation/services/evaluation/formatted_inputs"
CHECKPOINTS="/home/vacl2/multimodal_translation/services/tts/checkpoints"

echo "=== PIPELINE 9: src_EfikSrc ==="
run_synthesis 9 "src_EfikSrc" "efi" "efik" "$FORMATTED_DIR/pipeline_8_efi_src_MultilingualSrc.csv" "src" "$CHECKPOINTS/Src_Efik_12_12-December-13-2025_03+12PM-da9effb"
echo "=== PIPELINE 10: src_SwahiliSrc ==="
run_synthesis 10 "src_SwahiliSrc" "swa" "swahili" "$FORMATTED_DIR/pipeline_8_swa_src_MultilingualSrc.csv" "src" "$CHECKPOINTS/Src_Swahili_14_12-December-14-2025_04+27PM-da9effb"

echo ""
echo "=========================================="
echo "ALL SYNTHESES COMPLETE"
echo "=========================================="
echo "Total: $TOTAL"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", ($COMPLETED/$TOTAL)*100}")%"
echo "End time: $(date)"
echo "=========================================="
