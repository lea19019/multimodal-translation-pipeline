# Evaluation Setup Summary

This document describes the complete setup of the COMET and BLASER evaluation systems.

## Directory Structure

```
services/evaluation/
├── comet/
│   ├── models/                       # COMET models cache
│   │   └── ssa-comet-qe/            # McGill-NLP SSA-COMET-QE model
│   └── .venv/                        # COMET Python environment (NOT USED - uses main .venv)
│
├── blaser/
│   ├── models/                       # BLASER & SONAR models cache
│   │   └── (auto-downloaded)
│   ├── scripts/
│   │   ├── download_models.py       # Download BLASER/SONAR models
│   │   └── verify_setup.py          # Verify BLASER installation
│   ├── finetune/
│   │   └── prepare_finetuning.py    # Template for African language fine-tuning
│   ├── evaluate.py                   # CLI for BLASER evaluation
│   ├── pyproject.toml                # BLASER dependencies (PyTorch 2.8.0)
│   ├── .venv/                        # BLASER Python environment (PyTorch 2.8.0)
│   └── uv.lock
│
├── scripts/
│   ├── comet_evaluator.py            # COMET evaluator wrapper
│   └── blaser_evaluator.py           # BLASER evaluator wrapper (subprocess)
│
├── evaluation.py                     # Main evaluation script
├── pyproject.toml                    # Main dependencies (PyTorch 2.9.0)
├── .venv/                            # Main Python environment (PyTorch 2.9.0)
└── [other evaluation files]
```

## Key Architectural Decisions

### 1. Separate Python Environments

**Main Environment** (`evaluation/.venv/`):
- Python: 3.11 (system)
- PyTorch: 2.9.0+cu128
- Used for: COMET, text metrics, audio metrics, main evaluation

**BLASER Environment** (`evaluation/blaser/.venv/`):
- Python: 3.11 (system)
- PyTorch: 2.8.0+cu128
- fairseq2: 0.6
- Used for: BLASER 2.0 evaluation only

**Reason**: fairseq2 requires **exact** PyTorch version match (2.8.0). Running different PyTorch versions in isolated environments prevents conflicts.

### 2. Subprocess Integration for BLASER

The `scripts/blaser_evaluator.py` calls `blaser/evaluate.py` via subprocess, passing:
- Audio file paths
- Text data
- Language codes
- Model configuration

Results are returned via JSON file, parsed by the main evaluator.

### 3. Model Storage

- **COMET models**: `comet/models/`
- **BLASER models**: `blaser/models/` (auto-downloaded by SONAR/fairseq2)
- **Reason**: Clean separation, each system manages its own models

## Setup Instructions

### Initial Setup (Already Complete)

1. **Main Environment** (already set up):
   ```bash
   cd /home/vacl2/multimodal_translation/services/evaluation
   uv venv --python /usr/bin/python3.11
   uv sync
   ```

2. **BLASER Environment** (already set up):
   ```bash
   cd blaser
   uv venv --python /usr/bin/python3.11
   uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   uv pip install fairseq2 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128
   uv sync
   ```

3. **Download Models**:
   ```bash
   cd blaser
   uv run python scripts/download_models.py --languages eng,spa
   ```

4. **Verify Setup**:
   ```bash
   cd blaser
   uv run python scripts/verify_setup.py
   ```

## Usage

### COMET Evaluation

COMET evaluation runs in the main environment:

```python
from scripts.comet_evaluator import CometEvaluator

evaluator = CometEvaluator(model_name="McGill-NLP/ssa-comet-qe")
results = evaluator.evaluate(
    sources=["Hello world"],
    hypotheses=["Hola mundo"],
    references=["Hola mundo"]
)
print(f"COMET score: {results['corpus_score']}")
```

### BLASER Evaluation

BLASER evaluation automatically uses subprocess to the BLASER environment:

```python
from scripts.blaser_evaluator import BlaserEvaluator

evaluator = BlaserEvaluator(model_name="blaser_2_0_qe")
results = evaluator.evaluate(
    source_audio_paths=["audio1.wav"],
    target_audio_paths=["translated1.wav"],
    source_texts=["Hello world"],
    reference_texts=["Hola mundo"],
    source_lang="eng_Latn",
    target_lang="spa_Latn"
)
print(f"BLASER score: {results['corpus_score']}")
```

### Full Evaluation

Run the complete evaluation pipeline:

```bash
uv run python evaluation.py --config examples/audio_to_audio_config.yaml
```

## Models

### COMET: McGill-NLP/ssa-comet-qe

- **Type**: Quality Estimation (reference-free)
- **Languages**: 76 African + major world languages
- **Use case**: Evaluating translations without reference
- **Location**: `comet/models/ssa-comet-qe/`

### BLASER 2.0

- **Models Available**:
  - `blaser_2_0_qe`: Quality estimation (reference-free)
  - `blaser_2_0_ref`: Reference-based evaluation
- **Type**: Speech-to-speech translation evaluation
- **Languages**: 37 languages (SONAR speech encoders)
- **Location**: `blaser/models/` (auto-cached)

### SONAR Encoders

- **Text encoders**: 200+ languages (NLLB-200)
- **Speech encoders**: 37 languages
- **Supported African languages**:
  - Swahili (swh), Afrikaans (afr), Amharic (amh)
  - Yoruba (yor), Igbo (ibo), Zulu (zul)
- **Location**: Auto-cached in `~/.cache/` or `blaser/models/`

## Fine-tuning for African Languages

Template available at: `blaser/finetune/prepare_finetuning.py`

This is a **skeleton/TODO template** showing the structure needed for fine-tuning BLASER on custom data (e.g., low-resource African languages).

**Steps needed** (not yet implemented):
1. Collect parallel corpus with quality annotations
2. Implement data loading in `BlaserFinetuningDataset`
3. Implement training loop
4. Add validation and checkpointing
5. Test on evaluation set

**Resources**:
- AfriMTE dataset
- AfriCOMET models
- SONAR multilingual embeddings

## Troubleshooting

### COMET Issues

**Issue**: Model download fails with SSL error
**Solution**: Using system Python 3.11 (not UV's Python) - already configured

**Issue**: Model not found
**Solution**: Check `comet/models/` - model should be at `ssa-comet-qe/snapshots/.../checkpoints/model.ckpt`

### BLASER Issues

**Issue**: `fairseq2` import error
**Solution**: Make sure you're in the BLASER environment: `cd blaser && uv run python ...`

**Issue**: PyTorch version mismatch
**Solution**: BLASER env must use PyTorch 2.8.0 exactly (already configured)

**Issue**: Speech encoder not found for language
**Solution**: Check SONAR supported languages, or use fallback to English encoder

### General Issues

**Issue**: CUDA out of memory
**Solution**: Both evaluators fall back to CPU automatically

**Issue**: Evaluation takes too long
**Solution**: BLASER subprocess has 10-minute timeout. Adjust in `scripts/blaser_evaluator.py` if needed.

## Version Information

- **Main Environment**:
  - Python: 3.11.7
  - PyTorch: 2.9.0+cu128
  - COMET: 2.2.7
  - CUDA: 12.8

- **BLASER Environment**:
  - Python: 3.11.7
  - PyTorch: 2.8.0+cu128
  - fairseq2: 0.6
  - sonar-space: 0.5.0
  - CUDA: 12.8

## Testing

### Test COMET:
```bash
cd /home/vacl2/multimodal_translation/services/evaluation
uv run python evaluation.py --config examples/audio_to_audio_config.yaml
```

Check that COMET scores are non-zero (e.g., ~0.70).

### Test BLASER:
```bash
cd blaser
uv run python scripts/verify_setup.py
```

All tests should pass.

## Summary

✅ **COMET**: Ready for English↔Spanish (and 76 African languages)
✅ **BLASER**: Environment set up, models downloaded, ready for English↔Spanish
⏳ **BLASER Fine-tuning**: Template created, needs implementation for African languages

The system is production-ready for current English↔Spanish evaluation and can be extended for African languages with fine-tuning.
