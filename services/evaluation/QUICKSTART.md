# Quick Start Guide

## 1. Installation

```bash
cd ~/multimodal_translation/services/evaluation

# Install dependencies
pip install -r requirements.txt

# Or use UV (recommended)
uv pip install -e .
```

## 2. Prepare Your Data

Ensure your data follows this structure:

```
../data/app_evaluation/
├── text_to_text/{uuid}/
│   ├── metadata.json
│   ├── source.txt
│   └── target.txt
├── audio_to_text/{uuid}/
│   ├── metadata.json
│   ├── source_audio.wav
│   ├── transcribed.txt
│   └── target.txt
├── text_to_audio/{uuid}/
│   ├── metadata.json
│   ├── source.txt
│   ├── target.txt
│   └── target_audio.wav
└── audio_to_audio/{uuid}/
    ├── metadata.json
    ├── source_audio.wav
    ├── transcribed.txt
    ├── target.txt
    └── target_audio.wav
```

## 3. Run Your First Evaluation

### Option A: Command Line

```bash
# Evaluate text-to-text translations
python evaluation.py \
  --type text_to_text \
  --data-dir ../data/app_evaluation/text_to_text \
  --output-dir ./results/test_run
```

### Option B: Configuration File

```bash
# Use the provided example config
python evaluation.py --config examples/text_to_text_config.yaml
```

## 4. View Results

After evaluation completes, check:

```bash
# View summary
cat results/test_run/summary.json

# View detailed CSV
cat results/test_run/detailed_results.csv

# Open HTML report in browser
firefox results/test_run/summary_report.html
```

## 5. Common Use Cases

### Evaluate All Types

```bash
python evaluation.py \
  --type all \
  --data-dir ../data/app_evaluation \
  --output-dir ./results/full_eval
```

### Evaluate Specific Samples

```bash
python evaluation.py \
  --type text_to_audio \
  --samples uuid1 uuid2 uuid3 \
  --data-dir ../data/app_evaluation/text_to_audio
```

### Use Custom Metrics

```bash
python evaluation.py \
  --type text_to_text \
  --metrics bleu chrf \
  --data-dir ../data/app_evaluation/text_to_text
```

## 6. Understanding Output

### summary.json
High-level aggregate scores for quick comparison:
```json
{
  "run_id": "run_001",
  "translation_type": "text_to_text",
  "total_samples": 100,
  "aggregate_scores": {
    "bleu": 42.5,
    "chrf": 58.3,
    "comet": 0.73
  }
}
```

### detailed_results.csv
Per-sample scores for analysis:
```csv
uuid,source_language,target_language,bleu_score,chrf_score,comet_score
abc123,en,es,45.2,62.1,0.78
def456,en,es,39.8,54.5,0.68
```

### summary_report.html
Interactive HTML report with:
- Aggregate metric scores
- Score distributions
- Per-sample heatmaps
- Statistical summaries

## 7. Troubleshooting

### No GPU Available
```bash
# Check GPU
nvidia-smi

# If no GPU, evaluation will use CPU (slower but works)
```

### Model Download Issues
```bash
# Pre-download COMET model
python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## 8. Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/](examples/) for more configuration examples
- Explore metric signatures in results for reproducibility
- Customize visualizations in `visualizations.py`

## Need Help?

Check the logs:
```bash
tail -f results/{run_id}/logs/evaluation.log
```

Verify your data:
```bash
# List available samples
ls ../data/app_evaluation/text_to_text/

# Check a sample
cat ../data/app_evaluation/text_to_text/{uuid}/metadata.json
```
