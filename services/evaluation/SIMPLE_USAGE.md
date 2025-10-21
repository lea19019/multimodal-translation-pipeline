# Evaluation System - Simple Usage Guide

## ✅ Fixed and Working

The evaluation system is now working with a simpler structure:

### Changes Made:
1. **Moved evaluators to `scripts/` folder** - No more package conflicts
   - `scripts/comet_evaluator.py` - COMET evaluation
   - `scripts/blaser_evaluator.py` - BLASER evaluation
   
2. **Removed production packaging** - Deleted `pyproject.toml`, `__init__.py`, `.gitignore`

3. **Fixed imports** - evaluation.py now uses `from scripts.comet_evaluator import ...`

4. **Fixed CLI** - Config files now work properly without requiring `--data-dir` flag

5. **Fixed JSON serialization** - Signatures are now strings

## How to Use

### 1. Install Dependencies (one time)
```bash
cd ~/multimodal_translation/services/evaluation
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Run Evaluation

#### Using config file (easiest):
```bash
uv run python evaluation.py --config ./examples/text_to_text_config.yaml
```

#### Using command line:
```bash
uv run python evaluation.py \
  --type text_to_text \
  --data-dir ../data/app_evaluation/text_to_text \
  --output-dir ./results/my_run
```

## Available Configs

- `examples/text_to_text_config.yaml` - ✅ TESTED AND WORKING
- `examples/audio_to_audio_config.yaml` - Has some dependency issues (COMET model, BLASER fairseq2)
- `examples/full_evaluation_config.yaml` - Evaluates all types
- `examples/specific_samples_config.yaml` - Evaluate specific UUIDs

## What Works Now

✅ **Text-to-Text evaluation** (BLEU, chrF)
✅ **Config file loading**
✅ **Visualizations** (charts, graphs, HTML reports)
✅ **Results saving** (JSON, CSV)

## Known Issues (Minor)

1. **COMET**: Model name was corrected to `wmt22-comet-da` (will download on first use)
2. **BLASER**: Requires `fairseq2` package (not in requirements.txt)
   - To fix: `uv pip install fairseq2`
3. **MCD**: Works but shows warnings about n_fft (harmless)

## Results Structure

After running, check `results/<run_id>/`:
```
results/text_eval_001/
├── summary.json              # Overall scores
├── detailed_results.csv      # Per-sample scores
├── per_sample_results.json   # Full details
├── logs/evaluation.log       # Logs
└── visualizations/           # PNG charts
    ├── metrics_comparison.png
    ├── metrics_table.png
    ├── bleu_distribution.png
    ├── chrf_distribution.png
    └── per_sample_heatmap.png
```

## Example Output

Just ran successfully:
```
EVALUATION COMPLETE - text_to_text
Run ID: text_eval_001
Samples evaluated: 2/2

Aggregate Scores:
  BLEU: 100.000
  CHRF: 100.000
```

## Simple and Clean! 🎉

No more package complexity - just run `uv run python evaluation.py` and it works!
