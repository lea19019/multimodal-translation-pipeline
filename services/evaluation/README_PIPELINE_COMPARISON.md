# Pipeline Comparison Evaluation System

A comprehensive system for evaluating and comparing 8 translation pipelines across 4 African languages.

## Overview

This system evaluates **32 combinations** (8 pipelines × 4 languages) using your existing synthesized audio data, then generates:
- Individual performance metrics per pipeline/language
- Cross-pipeline comparison tables
- Interactive visualizations (heatmaps, rankings, radar charts)
- HTML dashboard for easy exploration

## Quick Start

### Run Complete Evaluation

```bash
cd /home/vacl2/multimodal_translation/services/evaluation
uv run python run_pipeline_comparison.py
```

This will:
1. Evaluate all 8 pipelines across all 4 languages (32 evaluations)
2. Compare results and generate rankings
3. Create visualizations
4. Generate interactive HTML dashboard

**Note:** Full evaluation will take several hours due to metric computation (especially COMET and BLASER).

### Test with Subset

Evaluate just 1-2 pipelines for testing:

```bash
uv run python run_pipeline_comparison.py --pipelines pipeline_1 pipeline_2 --languages efik
```

### Use Existing Results

If you've already run evaluation and just want to regenerate comparisons/visualizations:

```bash
uv run python run_pipeline_comparison.py --skip-evaluation --execution-id pipeline_comparison_YYYYMMDD_HHMMSS
```

## Command-Line Options

```
--languages LANG1 LANG2 ...     Languages to evaluate (efik, igbo, swahili, xhosa)
--pipelines PIPE1 PIPE2 ...     Pipeline IDs to evaluate (pipeline_1 through pipeline_8)
--output-dir DIR                Output directory (default: ./results)
--execution-id ID               Custom execution ID
--skip-evaluation               Skip evaluation, use existing results
--skip-visualizations           Skip visualization generation
--skip-dashboard                Skip dashboard generation
```

## The 8 Pipelines

| ID | Name | Flow | Metrics |
|----|------|------|---------|
| pipeline_1 | NLLB → MULTILINGUAL_TRAINING | src → NLLB → pred → MULTILINGUAL → wav | BLEU, chrF, COMET, MCD, BLASER |
| pipeline_2 | NLLB → Src_Tgt | src → NLLB → pred → Src_Tgt → wav | BLEU, chrF, COMET, MCD, BLASER |
| pipeline_3 | Source → Src_Tgt | src → Src_Tgt → wav | MCD, BLASER |
| pipeline_4 | Custom Lang → Src_Tgt | tags → Src_Tgt → wav | BLEU, chrF, COMET, MCD, BLASER |
| pipeline_5 | NLLB → Translate_Src_Tgt | src → NLLB → pred → Translate → wav | BLEU, chrF, COMET, MCD, BLASER |
| pipeline_6 | Source → Translate_Src_Tgt | src → Translate → wav | MCD, BLASER |
| pipeline_7 | Custom Translate → Translate_Src_Tgt | tags → Translate → wav | BLEU, chrF, COMET, MCD, BLASER |
| pipeline_8 | Source → Multilingual_Src | src → Multilingual → wav | MCD, BLASER |

**Note:** Pipelines 3, 6, and 8 don't use NMT translation, so they only have audio metrics (MCD, BLASER).

## Output Structure

Results are organized as:

```
results/pipeline_comparison_YYYYMMDD_HHMMSS/
├── manifest.json                          # Execution metadata
├── individual_pipelines/                  # Per-pipeline results
│   ├── pipeline_1_nllb_multilingual/
│   │   ├── efik/
│   │   │   ├── metrics.json              # All metrics for this combination
│   │   │   └── sample_scores.csv         # Per-sample scores
│   │   ├── igbo/
│   │   ├── swahili/
│   │   └── xhosa/
│   ├── pipeline_2_nllb_src_tgt/
│   └── ... (pipelines 3-8)
├── comparisons/                           # Cross-pipeline analysis
│   ├── cross_pipeline_stats.csv          # Complete comparison table
│   ├── all_pipelines_summary.json        # JSON format
│   ├── pipeline_rankings_overall.json    # Overall rankings
│   ├── pipeline_rankings_by_language.json
│   ├── best_pipelines.json               # Best performers
│   └── rankings_by_{metric}.json         # Per-metric rankings
├── visualizations/                        # Charts and graphs
│   ├── overall_rankings.png
│   ├── pipeline_radar_comparison.png
│   ├── bleu_heatmap_pipeline_x_language.png
│   ├── chrf_heatmap_pipeline_x_language.png
│   ├── comet_heatmap_pipeline_x_language.png
│   ├── mcd_heatmap_pipeline_x_language.png
│   ├── blaser_heatmap_pipeline_x_language.png
│   └── per_language/
│       ├── efik_pipeline_comparison.png
│       ├── igbo_pipeline_comparison.png
│       ├── swahili_pipeline_comparison.png
│       └── xhosa_pipeline_comparison.png
└── dashboard.html                         # Interactive dashboard
```

## Visualizations

The system generates several types of visualizations:

### 1. Overall Rankings Bar Chart
- Shows all pipelines ranked by overall score
- Color-coded by performance tier (excellent/good/fair/poor)

### 2. Heatmaps (one per metric)
- Rows = Pipelines
- Columns = Languages
- Color intensity = Performance
- Shows which pipeline works best for each language

### 3. Radar Chart
- Compares top 4 pipelines across all metrics
- Multi-metric view at a glance

### 4. Per-Language Comparisons
- Detailed breakdown for each language
- Side-by-side metric comparisons

## Interactive Dashboard

The HTML dashboard provides:
- **Executive Summary**: Best pipeline, total evaluations
- **Overall Rankings**: Sortable table
- **By Metric**: Tabs for BLEU, chrF, COMET, MCD, BLASER
- **By Language**: Tabs for Efik, Igbo, Swahili, Xhosa
- **Visualizations**: All charts embedded
- **Downloads**: JSON and CSV data exports

Open with:
```bash
firefox results/pipeline_comparison_YYYYMMDD_HHMMSS/dashboard.html
```

## Metrics

### Text Metrics (for pipelines using NMT)
- **BLEU** (0-100): N-gram overlap, higher is better
- **chrF++** (0-100): Character F-score with word order, higher is better
- **COMET** (0-1): Neural semantic quality (SSA-COMET-QE for African languages), higher is better

### Audio Metrics (for all pipelines)
- **MCD** (dB): Mel-cepstral distance, **lower is better**
- **BLASER 2.0** (0-5): Speech-to-speech quality, higher is better

### Overall Score
Normalized composite (0-1, higher is better):
- Combines all available metrics
- Normalized to 0-1 scale for fair comparison
- MCD inverted (lower → higher score)

## Architecture

### Components

1. **Pipeline Configuration** ([config/pipeline_config.py](config/pipeline_config.py))
   - Defines all 8 pipelines
   - Maps to synthesized data directories
   - Single source of truth

2. **Evaluation Orchestrator** ([orchestrator/pipeline_orchestrator.py](orchestrator/pipeline_orchestrator.py))
   - Coordinates 32 evaluations
   - Reuses existing metric computation
   - Saves individual results

3. **Pipeline Comparator** ([comparator/pipeline_comparator.py](comparator/pipeline_comparator.py))
   - Aggregates results
   - Creates comparison tables
   - Ranks pipelines
   - Identifies best performers

4. **Visualization Generator** ([visualizations/pipeline_viz.py](visualizations/pipeline_viz.py))
   - Creates heatmaps
   - Generates charts
   - Per-language comparisons

5. **Dashboard Generator** ([visualizations/dashboard_generator.py](visualizations/dashboard_generator.py))
   - Creates interactive HTML
   - Embeds visualizations
   - Data export links

6. **Main Script** ([run_pipeline_comparison.py](run_pipeline_comparison.py))
   - Ties everything together
   - Command-line interface

## Synthesized Data Sources

The system uses your existing synthesized data from:
```
/home/vacl2/multimodal_translation/services/data/languages/{lang}/
├── predicted_nllb_tgt_MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632/
├── predicted_nllb_tgt_Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8/
├── predicted_src_Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8/
├── predicted_custom_lang_Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8/
├── predicted_nllb_tgt_Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254/
├── predicted_src_Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254/
├── predicted_custom_translate_Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254/
└── predicted_src_Multilingual_Src_6_12-December-06-2025_10+30PM-b2b8254/
```

Each contains:
- 300 synthesized WAV files
- Metadata CSV with text/audio mappings

## Troubleshooting

### Import Errors
Always use `uv run python` instead of just `python`:
```bash
uv run python run_pipeline_comparison.py
```

### Out of Memory
For COMET/BLASER evaluation, you may need more memory. Run in batches:
```bash
# Evaluate one language at a time
for lang in efik igbo swahili xhosa; do
    uv run python run_pipeline_comparison.py --languages $lang
done
```

### Resume After Failure
If evaluation fails partway through, you can resume by skipping completed pipelines:
```bash
# Skip already-evaluated pipelines
uv run python run_pipeline_comparison.py --pipelines pipeline_5 pipeline_6 pipeline_7 pipeline_8
```

## Development

### Test Individual Components

```bash
# Test pipeline config
uv run python config/pipeline_config.py

# Test orchestrator (dry-run)
uv run python orchestrator/pipeline_orchestrator.py --dry-run

# Test comparator
uv run python comparator/pipeline_comparator.py --results-dir results/pipeline_comparison_YYYYMMDD_HHMMSS

# Test visualizations
uv run python visualizations/pipeline_viz.py --comparisons-dir results/pipeline_comparison_YYYYMMDD_HHMMSS/comparisons

# Test dashboard
uv run python visualizations/dashboard_generator.py --results-dir results/pipeline_comparison_YYYYMMDD_HHMMSS
```

### Add New Pipeline

Edit [config/pipeline_config.py](config/pipeline_config.py) and add to `PIPELINES` list:

```python
{
    'id': 'pipeline_9',
    'name': 'My New Pipeline',
    'short_name': 'my_new_pipeline',
    'descriptor': 'my_descriptor',  # Must match predicted_{descriptor}_{checkpoint}
    'checkpoint': 'My_Model_TIMESTAMP',
    'uses_nmt': True,  # or False
    'text_input': 'predicted',  # or 'source' or 'both'
    'text_format': 'plain',  # or 'custom_lang' or 'custom_translate'
    'metrics': ['bleu', 'chrf', 'comet', 'mcd', 'blaser']
}
```

## Citation

If you use this system in your research, please cite your paper and acknowledge the tools:
- BLEU & chrF: SacreBLEU
- COMET: SSA-COMET-QE (McGill-NLP)
- MCD: Mel-Cepstral Distance
- BLASER 2.0: Facebook/Meta AI

## Support

For issues or questions:
- Check [flows.txt](../tts/flows.txt) for pipeline concept details
- Review existing evaluation scripts in [scripts/](scripts/)
- See plan documents in `/home/vacl2/.claude/plans/`

---

**Last Updated**: December 13, 2025
**System Version**: 1.0
