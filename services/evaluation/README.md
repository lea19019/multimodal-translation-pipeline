# Multimodal Translation Evaluation System

Comprehensive evaluation system for multimodal machine translation, supporting text-to-text, audio-to-text, text-to-audio, and audio-to-audio translation modes.

## Features

- **Multiple Evaluation Metrics**:
  - BLEU (sacrebleu)
  - chrF/chrF++ (sacrebleu)
  - COMET (Unbabel/wmt22-comet-da)
  - MCD - Mel-Cepstral Distance (for audio quality)
  - BLASER 2.0 (for speech-to-speech evaluation)

- **Translation Modalities**:
  - Text-to-Text: BLEU, chrF++, COMET
  - Audio-to-Text: BLEU, chrF++, COMET
  - Text-to-Audio: BLEU, chrF++, COMET, MCD
  - Audio-to-Audio: BLEU, chrF++, COMET, MCD, BLASER

- **Rich Visualizations**:
  - Metrics comparison charts
  - Score distribution plots
  - Per-sample heatmaps
  - Interactive HTML reports

- **Flexible Interface**:
  - CLI with multiple options
  - YAML configuration file support
  - Batch and single-sample evaluation
  - Progress tracking and logging

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-compatible GPU (optional, for faster evaluation)

### Using UV (Recommended)

```bash
cd ~/multimodal_translation/services/evaluation

# Install dependencies with UV
uv pip install -e .
```

### Using pip

```bash
cd ~/multimodal_translation/services/evaluation

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

For Japanese tokenization:
```bash
pip install mecab-python3
```

For Korean tokenization:
```bash
pip install pymecab-ko
```

## Data Format

The evaluation system expects data organized in UUID-based directories:

```
../data/app_evaluation/
├── text_to_text/
│   └── {uuid}/
│       ├── metadata.json
│       ├── source.txt
│       └── target.txt
├── audio_to_text/
│   └── {uuid}/
│       ├── metadata.json
│       ├── source_audio.wav
│       ├── transcribed.txt
│       └── target.txt
├── text_to_audio/
│   └── {uuid}/
│       ├── metadata.json
│       ├── source.txt
│       ├── target.txt
│       └── target_audio.wav
└── audio_to_audio/
    └── {uuid}/
        ├── metadata.json
        ├── source_audio.wav
        ├── transcribed.txt
        ├── target.txt
        └── target_audio.wav
```

### metadata.json Format

```json
{
    "source_language": "en",
    "target_language": "es",
    "timestamp": "2025-01-15T10:30:00Z",
    "translation_type": "text_to_text"
}
```

## Quick Start

### 1. Evaluate Text-to-Text Translations

```bash
python evaluation.py \
  --type text_to_text \
  --data-dir ../data/app_evaluation/text_to_text \
  --output-dir ./results/run_001
```

### 2. Evaluate All Translation Types

```bash
python evaluation.py \
  --type all \
  --data-dir ../data/app_evaluation \
  --output-dir ./results/run_001
```

### 3. Evaluate Specific Samples

```bash
python evaluation.py \
  --type text_to_audio \
  --samples 1f42e4b8-3670-4cde-a83a-b99a625c761f c7f15c69-a2a1-429c-9a6b-6725333fab9c \
  --data-dir ../data/app_evaluation/text_to_audio
```

### 4. Use Configuration File

```bash
python evaluation.py --config evaluation_config.yaml
```

## Configuration File

Create `evaluation_config.yaml`:

```yaml
run_id: run_001
translation_type: text_to_audio  # or 'all' for all types
data_dir: ../data/app_evaluation
output_dir: ./results/run_001

# Optional: specific metrics (auto-detected if not specified)
metrics:
  - bleu
  - chrf
  - comet
  - mcd

# Optional: specific samples
samples:
  - 1f42e4b8-3670-4cde-a83a-b99a625c761f
  - c7f15c69-a2a1-429c-9a6b-6725333fab9c
```

## CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--type` | `-t` | Translation type: `text_to_text`, `audio_to_text`, `text_to_audio`, `audio_to_audio`, or `all` |
| `--data-dir` | `-d` | Path to data directory (required) |
| `--output-dir` | `-o` | Output directory for results (default: `./results`) |
| `--samples` | `-s` | Specific sample UUIDs to evaluate (can be used multiple times) |
| `--config` | `-c` | Path to YAML config file |
| `--run-id` | | Custom run ID (auto-generated if not specified) |
| `--metrics` | `-m` | Specific metrics to compute (auto-detected based on translation type if not specified) |

## Output Structure

```
results/
└── run_001/
    ├── summary.json                    # High-level summary
    ├── detailed_results.csv            # Per-sample scores in CSV
    ├── per_sample_results.json         # Full results with all details
    ├── logs/
    │   └── evaluation.log              # Detailed logs
    └── visualizations/
        ├── metrics_comparison.png      # Bar chart of all metrics
        ├── metrics_table.png           # Formatted table
        ├── bleu_distribution.png       # BLEU score distribution
        ├── chrf_distribution.png       # chrF score distribution
        ├── comet_distribution.png      # COMET score distribution
        ├── mcd_distribution.png        # MCD distribution (if applicable)
        ├── per_sample_heatmap.png      # Heatmap of scores
        └── summary_report.html         # Interactive HTML report
```

## Metric Details

### BLEU (BiLingual Evaluation Understudy)
- Measures n-gram precision between hypothesis and reference
- Range: 0-100 (higher is better)
- Signature includes tokenization method (default: 13a)

### chrF / chrF++
- Character n-gram F-score
- chrF: character-level only (word_order=0)
- chrF++: includes word n-grams (word_order=2)
- Range: 0-100 (higher is better)

### COMET (Crosslingual Optimized Metric for Evaluation of Translation)
- Neural metric trained on human judgments
- Model: `Unbabel/wmt22-comet-da`
- Range: 0-1 (higher is better)
- Requires source, hypothesis, and reference

### MCD (Mel-Cepstral Distance)
- Measures audio quality/similarity
- Lower values indicate better quality
- Typical range: 3-15 (lower is better)
- Includes alignment penalty

### BLASER 2.0
- Speech-to-speech translation quality
- Uses SONAR embeddings
- Range: typically 0-5 (higher is better)
- Requires source audio, target audio, and reference text

## Environment Variables

### Hugging Face Authentication (for COMET)

Some COMET models require Hugging Face authentication:

```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

Or use Hugging Face CLI:
```bash
huggingface-cli login
```

### SONAR Models (for BLASER)

BLASER uses SONAR models which are downloaded automatically on first use.
Models are cached in `~/.cache/fairseq2/` by default.

## Examples

### Example 1: Basic Text Evaluation

```bash
# Evaluate English->Spanish text translations
python evaluation.py \
  --type text_to_text \
  --data-dir ../data/app_evaluation/text_to_text \
  --output-dir ./results/text_eval
```

### Example 2: Audio Evaluation with Specific Metrics

```bash
# Evaluate audio-to-audio with only BLEU and MCD
python evaluation.py \
  --type audio_to_audio \
  --data-dir ../data/app_evaluation/audio_to_audio \
  --metrics bleu mcd \
  --output-dir ./results/audio_eval
```

### Example 3: Batch Evaluation

Create `batch_eval.yaml`:
```yaml
run_id: batch_run_001
translation_type: all
data_dir: ../data/app_evaluation
output_dir: ./results/batch_run_001
```

Run:
```bash
python evaluation.py --config batch_eval.yaml
```

## Troubleshooting

### COMET Model Download Issues

If COMET model download fails:
```bash
# Pre-download the model
python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"
```

### GPU Memory Issues

For large datasets, reduce batch size by editing the code or use CPU:
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
```

### Missing Audio Files

Ensure all audio files are:
- In WAV format
- 16kHz sample rate (recommended)
- Mono channel

Convert audio if needed:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

For specific package issues:
```bash
pip install --upgrade sacrebleu unbabel-comet mel-cepstral-distance sonar-space
```

## Performance Tips

1. **Use GPU**: Significantly faster for COMET and BLASER evaluation
2. **Batch Processing**: The system automatically batches samples for efficiency
3. **Selective Metrics**: Specify only needed metrics with `--metrics` to save time
4. **Sample Subsets**: Test with `--samples` before running full evaluation

## Citation

If you use this evaluation system, please cite the relevant metric papers:

**BLEU**:
```
@inproceedings{papineni2002bleu,
  title={BLEU: a method for automatic evaluation of machine translation},
  author={Papineni, Kishore and Roukos, Salim and Ward, Todd and Zhu, Wei-Jing},
  booktitle={ACL},
  year={2002}
}
```

**COMET**:
```
@inproceedings{rei2020comet,
  title={COMET: A Neural Framework for MT Evaluation},
  author={Rei, Ricardo and Stewart, Craig and Farinha, Ana C and Lavie, Alon},
  booktitle={EMNLP},
  year={2020}
}
```

**BLASER**:
```
@article{chen2023blaser,
  title={BLASER: A Text-Free Speech-to-Speech Translation Evaluation Metric},
  author={Chen, Mingda and others},
  journal={arXiv preprint arXiv:2212.08486},
  year={2023}
}
```

## Support

For issues or questions:
1. Check the logs in `results/{run_id}/logs/evaluation.log`
2. Verify data format matches the expected structure
3. Ensure all dependencies are correctly installed
4. Check GPU availability if using CUDA

## License

This evaluation system is provided as-is for research purposes.
Individual metrics have their own licenses - please refer to their respective documentation.
