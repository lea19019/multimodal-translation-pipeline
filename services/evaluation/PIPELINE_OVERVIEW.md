# Pipeline Overview

## Pipeline Architecture

The evaluation system consists of 10 translation pipelines that combine different approaches for speech-to-speech translation. Each pipeline is evaluated across 4 African languages (Efik, Igbo, Swahili, Xhosa) using multiple metrics.

### Pipeline Types

```
┌─────────────┐     ┌─────────┐     ┌─────────┐     ┌──────────┐
│ Source Text │ ──> │   NMT   │ ──> │   TTS   │ ──> │ Predicted│
│  (English)  │     │ (NLLB)  │     │ (Model) │     │  Audio   │
└─────────────┘     └─────────┘     └─────────┘     └──────────┘
                Text Translation    Speech Synthesis
```

#### Pipelines 1-2: NLLB + TTS Models
- **Pipeline 1**: NLLB → MULTILINGUAL_TRAINING
- **Pipeline 2**: NLLB → Src_Tgt

Uses NLLB for translation, then synthesizes target language text.

#### Pipelines 3, 6, 8: Direct Source TTS
- **Pipeline 3**: Source → Src_Tgt
- **Pipeline 6**: Source → Translate_Src_Tgt
- **Pipeline 8**: Source → Multilingual_Src

No translation step - synthesizes from English source directly.

#### Pipelines 4, 7: Custom Format TTS
- **Pipeline 4**: Custom Lang → Src_Tgt (`<eng>{src} <{lang}>{pred}`)
- **Pipeline 7**: Custom Translate → Translate_Src_Tgt (`<translate> <eng>{src} <{lang}>{pred}`)

Uses special input formatting with both source and predicted text.

#### Pipeline 5: NLLB + Translate Model
- **Pipeline 5**: NLLB → Translate_Src_Tgt

Uses NLLB translation with specialized TTS model.

#### Pipelines 9-10: Audio-to-Audio (XTTS)
- **Pipeline 9**: Source Audio → XTTS_Efik (Efik only)
- **Pipeline 10**: Source Audio → XTTS_Swahili (Swahili only)

Direct audio-to-audio translation using XTTS models.

---

## Evaluation Metrics

### Text Metrics (Only for Translation Pipelines)

#### BLEU (Bilingual Evaluation Understudy)
- **Input**: Predicted text, Reference text
- **Output**: Score 0-100 (higher is better)
- **Measures**: N-gram precision with brevity penalty
- **Use Case**: Word-level translation accuracy

#### chrF (Character F-score)
- **Input**: Predicted text, Reference text
- **Output**: Score 0-100 (higher is better)
- **Measures**: Character-level F-score
- **Use Case**: Better for morphologically rich languages

#### COMET (Crosslingual Optimized Metric for Evaluation of Translation)
- **Input**: Source text, Predicted text, Reference text
- **Output**: Score 0-1 (higher is better)
- **Model**: McGill-NLP/ssa-comet-qe
- **Use Case**: Neural quality estimation trained on human judgments

### Audio Metrics (All Pipelines)

#### MCD (Mel-Cepstral Distance)
- **Input**: Predicted audio (WAV), Reference audio (WAV)
- **Output**: Distance score (lower is better)
- **Measures**: Spectral distance between audio pairs
- **Use Case**: Audio quality and similarity

#### BLASER 2.0 (Bilingual Audio Semantic Evaluation and Recognition)
- **Input**:
  - Source audio (English)
  - Predicted audio (Target language)
  - Source text (English)
  - Reference text (Target language)
- **Output**: Quality score (higher is better)
- **Model**: blaser_2_0_qe (quality estimation mode)
- **Use Case**: End-to-end speech translation quality
- **Note**: Evaluates semantic similarity between source and predicted audio

---

## Results Directory Structure

Each pipeline run creates a timestamped directory: `results/pipeline_{id}_{short_name}_{timestamp}/`

### Files in Each Results Directory

```
pipeline_1_nllb_multilingual_20251214_161750/
├── manifest.json              # Execution metadata and summary scores
├── all_scores.csv             # Aggregated scores across all languages
├── {lang}_scores.csv          # Per-language aggregate scores
└── {lang}_sample_scores.csv   # Sentence-level scores for each sample
```

#### manifest.json
Contains:
- Pipeline metadata (ID, name, execution time)
- Per-language status and sample counts
- Corpus-level scores for all metrics

#### all_scores.csv
Columns: `language, bleu_mean, bleu_std, chrf_mean, comet_mean, mcd_mean, blaser_mean` (with min/max/count)

#### {lang}_scores.csv
Aggregated statistics for a single language across all metrics.

#### {lang}_sample_scores.csv
Detailed sentence-level scores for every sample:
- Sample ID, source text, predicted text, reference text
- Individual metric scores (BLEU, chrF, COMET, MCD, BLASER)
- Audio file paths

---

## Pipeline Comparison

| Pipeline | Uses NMT? | Text Metrics | Audio Metrics | Languages |
|----------|-----------|--------------|---------------|-----------|
| 1-2      | Yes       | ✓            | ✓             | All 4     |
| 3,6,8    | No        | ✗            | ✓             | All 4     |
| 4,7      | Yes       | ✓            | ✓             | All 4     |
| 5        | Yes       | ✓            | ✓             | All 4     |
| 9        | No        | ✗            | ✓             | Efik only |
| 10       | No        | ✗            | ✓             | Swahili only |

### Metric Availability
- **Text metrics** (BLEU, chrF, COMET): Only available when NMT produces predicted text
- **Audio metrics** (MCD, BLASER): Available for all pipelines
- **Pipelines 3, 6, 8-10**: Audio-only evaluation (no translation quality metrics)

---

## Quick Reference

### Languages
- **Efik** (efi)
- **Igbo** (ibo)
- **Swahili** (swa)
- **Xhosa** (xho)

### Score Interpretation
| Metric | Range | Better | Typical Good Score |
|--------|-------|--------|--------------------|
| BLEU   | 0-100 | Higher | >30                |
| chrF   | 0-100 | Higher | >60                |
| COMET  | 0-1   | Higher | >0.6               |
| MCD    | 0+    | Lower  | <15                |
| BLASER | 0-5   | Higher | >2.5               |
