#!/bin/bash
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=synthesis_all
#SBATCH --output=/home/vacl2/multimodal_translation/services/evaluation/_logs/synthesis_all_%j.out
#SBATCH --error=/home/vacl2/multimodal_translation/services/evaluation/_logs/synthesis_all_%j.err

echo "=========================================="
echo "TTS Synthesis: All 8 Pipelines × 4 Languages = 32 Syntheses"
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

# Pipeline 1: nllb_MULTILINGUAL (4 languages)
echo "=== PIPELINE 1: nllb_MULTILINGUAL ==="
run_synthesis 1 "nllb_MULTILINGUAL" "efi" "efik" "$FORMATTED_DIR/pipeline_1_efi_nllb_MULTILINGUAL.csv" "nllb_tgt" "$CHECKPOINTS/MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632"
run_synthesis 1 "nllb_MULTILINGUAL" "ibo" "igbo" "$FORMATTED_DIR/pipeline_1_ibo_nllb_MULTILINGUAL.csv" "nllb_tgt" "$CHECKPOINTS/MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632"
run_synthesis 1 "nllb_MULTILINGUAL" "swa" "swahili" "$FORMATTED_DIR/pipeline_1_swa_nllb_MULTILINGUAL.csv" "nllb_tgt" "$CHECKPOINTS/MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632"
run_synthesis 1 "nllb_MULTILINGUAL" "xho" "xhosa" "$FORMATTED_DIR/pipeline_1_xho_nllb_MULTILINGUAL.csv" "nllb_tgt" "$CHECKPOINTS/MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632"

# Pipeline 2: nllb_SrcTgt (4 languages)
echo "=== PIPELINE 2: nllb_SrcTgt ==="
run_synthesis 2 "nllb_SrcTgt" "efi" "efik" "$FORMATTED_DIR/pipeline_2_efi_nllb_SrcTgt.csv" "nllb_tgt" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"
run_synthesis 2 "nllb_SrcTgt" "ibo" "igbo" "$FORMATTED_DIR/pipeline_2_ibo_nllb_SrcTgt.csv" "nllb_tgt" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"
run_synthesis 2 "nllb_SrcTgt" "swa" "swahili" "$FORMATTED_DIR/pipeline_2_swa_nllb_SrcTgt.csv" "nllb_tgt" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"
run_synthesis 2 "nllb_SrcTgt" "xho" "xhosa" "$FORMATTED_DIR/pipeline_2_xho_nllb_SrcTgt.csv" "nllb_tgt" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"

# Pipeline 3: src_SrcTgt (4 languages)
echo "=== PIPELINE 3: src_SrcTgt ==="
run_synthesis 3 "src_SrcTgt" "efi" "efik" "$FORMATTED_DIR/pipeline_3_efi_src_SrcTgt.csv" "src" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"
run_synthesis 3 "src_SrcTgt" "ibo" "igbo" "$FORMATTED_DIR/pipeline_3_ibo_src_SrcTgt.csv" "src" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"
run_synthesis 3 "src_SrcTgt" "swa" "swahili" "$FORMATTED_DIR/pipeline_3_swa_src_SrcTgt.csv" "src" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"
run_synthesis 3 "src_SrcTgt" "xho" "xhosa" "$FORMATTED_DIR/pipeline_3_xho_src_SrcTgt.csv" "src" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"

# Pipeline 4: custom_lang_SrcTgt (4 languages)
echo "=== PIPELINE 4: custom_lang_SrcTgt ==="
run_synthesis 4 "custom_lang_SrcTgt" "efi" "efik" "$FORMATTED_DIR/pipeline_4_efi_custom_lang_SrcTgt.csv" "custom_lang" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"
run_synthesis 4 "custom_lang_SrcTgt" "ibo" "igbo" "$FORMATTED_DIR/pipeline_4_ibo_custom_lang_SrcTgt.csv" "custom_lang" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"
run_synthesis 4 "custom_lang_SrcTgt" "swa" "swahili" "$FORMATTED_DIR/pipeline_4_swa_custom_lang_SrcTgt.csv" "custom_lang" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"
run_synthesis 4 "custom_lang_SrcTgt" "xho" "xhosa" "$FORMATTED_DIR/pipeline_4_xho_custom_lang_SrcTgt.csv" "custom_lang" "$CHECKPOINTS/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8"

# Pipeline 5: nllb_TranslateSrcTgt (4 languages)
echo "=== PIPELINE 5: nllb_TranslateSrcTgt ==="
run_synthesis 5 "nllb_TranslateSrcTgt" "efi" "efik" "$FORMATTED_DIR/pipeline_5_efi_nllb_TranslateSrcTgt.csv" "nllb_tgt" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"
run_synthesis 5 "nllb_TranslateSrcTgt" "ibo" "igbo" "$FORMATTED_DIR/pipeline_5_ibo_nllb_TranslateSrcTgt.csv" "nllb_tgt" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"
run_synthesis 5 "nllb_TranslateSrcTgt" "swa" "swahili" "$FORMATTED_DIR/pipeline_5_swa_nllb_TranslateSrcTgt.csv" "nllb_tgt" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"
run_synthesis 5 "nllb_TranslateSrcTgt" "xho" "xhosa" "$FORMATTED_DIR/pipeline_5_xho_nllb_TranslateSrcTgt.csv" "nllb_tgt" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"

# Pipeline 6: src_TranslateSrcTgt (4 languages)
echo "=== PIPELINE 6: src_TranslateSrcTgt ==="
run_synthesis 6 "src_TranslateSrcTgt" "efi" "efik" "$FORMATTED_DIR/pipeline_6_efi_src_TranslateSrcTgt.csv" "src" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"
run_synthesis 6 "src_TranslateSrcTgt" "ibo" "igbo" "$FORMATTED_DIR/pipeline_6_ibo_src_TranslateSrcTgt.csv" "src" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"
run_synthesis 6 "src_TranslateSrcTgt" "swa" "swahili" "$FORMATTED_DIR/pipeline_6_swa_src_TranslateSrcTgt.csv" "src" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"
run_synthesis 6 "src_TranslateSrcTgt" "xho" "xhosa" "$FORMATTED_DIR/pipeline_6_xho_src_TranslateSrcTgt.csv" "src" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"

# Pipeline 7: custom_translate_TranslateSrcTgt (4 languages)
echo "=== PIPELINE 7: custom_translate_TranslateSrcTgt ==="
run_synthesis 7 "custom_translate_TranslateSrcTgt" "efi" "efik" "$FORMATTED_DIR/pipeline_7_efi_custom_translate_TranslateSrcTgt.csv" "custom_translate" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"
run_synthesis 7 "custom_translate_TranslateSrcTgt" "ibo" "igbo" "$FORMATTED_DIR/pipeline_7_ibo_custom_translate_TranslateSrcTgt.csv" "custom_translate" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"
run_synthesis 7 "custom_translate_TranslateSrcTgt" "swa" "swahili" "$FORMATTED_DIR/pipeline_7_swa_custom_translate_TranslateSrcTgt.csv" "custom_translate" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"
run_synthesis 7 "custom_translate_TranslateSrcTgt" "xho" "xhosa" "$FORMATTED_DIR/pipeline_7_xho_custom_translate_TranslateSrcTgt.csv" "custom_translate" "$CHECKPOINTS/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254"

# Pipeline 8: src_MultilingualSrc (4 languages)
echo "=== PIPELINE 8: src_MultilingualSrc ==="
run_synthesis 8 "src_MultilingualSrc" "efi" "efik" "$FORMATTED_DIR/pipeline_8_efi_src_MultilingualSrc.csv" "src" "$CHECKPOINTS/Multilingual_Src_6_12-December-06-2025_10+30PM-b2b8254"
run_synthesis 8 "src_MultilingualSrc" "ibo" "igbo" "$FORMATTED_DIR/pipeline_8_ibo_src_MultilingualSrc.csv" "src" "$CHECKPOINTS/Multilingual_Src_6_12-December-06-2025_10+30PM-b2b8254"
run_synthesis 8 "src_MultilingualSrc" "swa" "swahili" "$FORMATTED_DIR/pipeline_8_swa_src_MultilingualSrc.csv" "src" "$CHECKPOINTS/Multilingual_Src_6_12-December-06-2025_10+30PM-b2b8254"
run_synthesis 8 "src_MultilingualSrc" "xho" "xhosa" "$FORMATTED_DIR/pipeline_8_xho_src_MultilingualSrc.csv" "src" "$CHECKPOINTS/Multilingual_Src_6_12-December-06-2025_10+30PM-b2b8254"

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
