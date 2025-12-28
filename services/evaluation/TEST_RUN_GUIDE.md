# Test Run Guide

## Test Evaluation Running

**Command executed:**
```bash
uv run python run_pipeline_comparison.py --pipelines pipeline_1 pipeline_2 --languages efik --execution-id test_run_001
```

**What's being evaluated:**
- **Pipeline 1**: NLLB → MULTILINGUAL_TRAINING
- **Pipeline 2**: NLLB → Src_Tgt
- **Language**: Efik only
- **Total**: 2 evaluations (instead of 32)

## Results Location

All results will be saved to:
```
/home/vacl2/multimodal_translation/services/evaluation/results/test_run_001/
```

## Directory Structure

```
test_run_001/
├── manifest.json                          # Execution metadata
│
├── individual_pipelines/                  # Per-pipeline results
│   ├── nllb_multilingual/
│   │   └── efik/
│   │       ├── metrics.json              # All metrics for pipeline 1 × efik
│   │       └── sample_scores.csv         # Per-sample scores
│   └── nllb_src_tgt/
│       └── efik/
│           ├── metrics.json              # All metrics for pipeline 2 × efik
│           └── sample_scores.csv
│
├── comparisons/                           # Cross-pipeline comparison
│   ├── cross_pipeline_stats.csv          # Main comparison table
│   ├── all_pipelines_summary.json        # JSON format
│   ├── pipeline_rankings_overall.json    # Which pipeline is best?
│   └── best_pipelines.json               # Best performers
│
├── visualizations/                        # Charts and graphs
│   ├── overall_rankings.png              # Bar chart comparing both pipelines
│   ├── bleu_heatmap_pipeline_x_language.png
│   ├── chrf_heatmap_pipeline_x_language.png
│   ├── comet_heatmap_pipeline_x_language.png
│   ├── mcd_heatmap_pipeline_x_language.png
│   ├── blaser_heatmap_pipeline_x_language.png
│   └── per_language/
│       └── efik_pipeline_comparison.png  # Detailed comparison for Efik
│
└── dashboard.html                         # Interactive HTML dashboard
```

## What to Check After Completion

### 1. Individual Metrics (most detailed)

Look at individual pipeline performance:
```bash
# Pipeline 1 results
cat results/test_run_001/individual_pipelines/nllb_multilingual/efik/metrics.json

# Pipeline 2 results
cat results/test_run_001/individual_pipelines/nllb_src_tgt/efik/metrics.json
```

Each `metrics.json` contains:
- `pipeline_name`: Pipeline identifier
- `language`: "efik"
- `n_samples`: Number of samples evaluated (should be ~300)
- `metrics`:
  - `bleu`: Corpus and sentence-level scores
  - `chrf`: Corpus and sentence-level scores
  - `comet`: Corpus and sentence-level scores
  - `mcd`: Mean, std, per-file scores
  - `blaser`: Corpus and sentence-level scores

### 2. Comparison Table (quick overview)

See both pipelines side-by-side:
```bash
cat results/test_run_001/comparisons/cross_pipeline_stats.csv
```

Columns:
- `pipeline_id`, `pipeline_name`, `language`
- `bleu`, `chrf`, `comet` (text quality metrics)
- `mcd`, `blaser` (audio quality metrics)
- `overall_score` (normalized composite 0-1, higher = better)

### 3. Rankings (who wins?)

Check which pipeline performed better:
```bash
cat results/test_run_001/comparisons/pipeline_rankings_overall.json
```

This will show:
- Rank 1: Best pipeline
- Rank 2: Second best
- With scores for each

### 4. Visualizations (easiest to understand)

View the charts:
```bash
# Overall rankings bar chart
display results/test_run_001/visualizations/overall_rankings.png

# Heatmaps (one per metric)
display results/test_run_001/visualizations/bleu_heatmap_pipeline_x_language.png
display results/test_run_001/visualizations/mcd_heatmap_pipeline_x_language.png
```

### 5. Interactive Dashboard (best experience)

Open in browser:
```bash
firefox results/test_run_001/dashboard.html
```

Or get the full path:
```bash
realpath results/test_run_001/dashboard.html
```

Then open: `file:///home/vacl2/multimodal_translation/services/evaluation/results/test_run_001/dashboard.html`

## Interpreting Results

### Text Metrics (higher is better)
- **BLEU** (0-100):
  - <30: Poor
  - 30-50: Fair
  - 50-70: Good
  - >70: Excellent
- **chrF++** (0-100): Similar to BLEU
- **COMET** (0-1):
  - <0.40: Poor
  - 0.40-0.60: Fair
  - 0.60-0.75: Good
  - >0.75: Excellent

### Audio Metrics
- **MCD** (dB, **lower is better**):
  - <4: Excellent
  - 4-6: Good
  - 6-8: Fair
  - >8: Poor
- **BLASER** (0-5, higher is better):
  - <2.5: Poor
  - 2.5-3.5: Fair
  - 3.5-4.0: Good
  - >4.0: Excellent

### Overall Score (0-1, higher is better)
- Normalized composite of all metrics
- <0.40: Poor overall
- 0.40-0.60: Fair overall
- 0.60-0.75: Good overall
- >0.75: Excellent overall

## Expected Runtime

For this test (2 pipelines × 1 language):
- Text metrics (BLEU, chrF): ~30 seconds
- **COMET**: ~5-10 minutes (neural model, slow)
- Audio metrics (MCD): ~2-3 minutes
- **BLASER**: ~5-10 minutes (neural model, slow)
- Comparison & visualization: ~10 seconds

**Total**: ~15-25 minutes

For full run (8 pipelines × 4 languages):
- **Total**: ~4-6 hours

## Monitoring Progress

Check the log file:
```bash
tail -f /home/vacl2/multimodal_translation/services/evaluation/test_run.log
```

Look for:
- `[1/2] Evaluating: NLLB → MULTILINGUAL_TRAINING × efik`
- `Computing BLEU...`
- `Computing chrF++...`
- `Computing COMET...` (slowest)
- `Computing MCD...`
- `Computing BLASER...` (slow)
- `[2/2] Evaluating: NLLB → Src_Tgt × efik`
- `STEP 2: COMPARING PIPELINES`
- `STEP 3: GENERATING VISUALIZATIONS`
- `STEP 4: GENERATING DASHBOARD`
- `PIPELINE COMPARISON COMPLETE!`

## Quick Commands

```bash
# Monitor progress
tail -f test_run.log

# Check if complete
ls -lh results/test_run_001/dashboard.html

# View results table
cat results/test_run_001/comparisons/cross_pipeline_stats.csv

# See rankings
cat results/test_run_001/comparisons/pipeline_rankings_overall.json | jq

# Open dashboard
firefox results/test_run_001/dashboard.html
```

## After Test Completes

If test looks good, run full evaluation:
```bash
# All 8 pipelines × 4 languages (will take 4-6 hours)
uv run python run_pipeline_comparison.py

# Or run one language at a time (more manageable)
for lang in efik igbo swahili xhosa; do
    uv run python run_pipeline_comparison.py --languages $lang
done
```

## Troubleshooting

If test fails:
1. Check the log: `cat test_run.log | grep -i error`
2. Check disk space: `df -h /home/vacl2`
3. Check memory: `free -h`
4. Try with just 1 pipeline: `--pipelines pipeline_1`

If COMET/BLASER are too slow:
- Run on compute node with GPU
- Or skip them temporarily: modify code to skip those metrics

---

**Test Started**: Check `test_run.log` for current progress
**Expected Completion**: ~15-25 minutes from start
