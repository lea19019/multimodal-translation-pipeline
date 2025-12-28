# Issues to Fix in Pipeline Comparison System

## Test Results Summary

Ran test evaluation on 2 pipelines × 1 language (Efik).

## Problems Found

### 1. Sample Count Mismatch (332 vs 300)
**Issue**: System says "332 samples" but synthesis folder has only 300 WAV files

**Why**:
- Synthesis CSV: 300 rows (predicted_nllb_tgt_MULTILINGUAL_TRAINING...csv)
- NMT predictions CSV: 3248 rows (nmt_predictions_multilang_finetuned_final.csv)
- Code merges them on `segment_id` and gets 332 matches
- **This means duplicate segment_ids in one of the CSVs**

**Fix Needed**:
- Investigate why merge produces 332 rows from 300
- Ensure we're only evaluating the 300 synthesized samples
- Check for duplicate segment_ids in CSVs

### 2. BLASER Returns 0.0000
**Issue**: BLASER evaluation times out and returns 0 score

**Why**:
```
2025-12-13 14:08:36 - Running BLASER evaluation for 332 samples...
2025-12-13 14:18:36 - BLASER evaluation timed out
Corpus BLASER: 0.0000
```
- 10-minute timeout (600 seconds)
- BLASER is a heavy neural model running on CPU
- Takes too long for 332 samples

**Fix Needed**:
- Run on GPU compute node, OR
- Reduce batch size and run in chunks, OR
- Increase timeout significantly, OR
- Test on smaller sample first (5-10) to verify it works

### 3. COMET Model Not Loading
**Issue**: COMET fails with "Model 'McGill-NLP/ssa-comet-qe' not supported"

**Error**:
```
2025-12-13 13:57:13 - Failed to load COMET model: "Model 'McGill-NLP/ssa-comet-qe' not supported by COMET."
```

**Why**: Model name may be incorrect or not available in installed COMET version

**Fix Needed**:
- Check which COMET models are available with: `comet-score --list-models`
- Update to correct model name
- May need to use generic COMET model instead of SSA-specific one

### 4. "Overall Score" Is Confusing
**Issue**: Results show "overall_score: 0.253" which doesn't make sense

**Current behavior**:
- System tries to normalize all metrics to 0-1
- Combines them into a single "overall" score
- But metrics have different meanings (text vs audio quality)
- Hard to interpret what 0.253 means

**Fix Needed**:
- **Remove** the "overall score" concept entirely
- Show each metric separately with clear labels
- Add context: "BLEU 31.70 (0-100, higher is better)"
- Show rankings per-metric instead of overall

## What Actually Worked

✅ System runs end-to-end without crashing
✅ BLEU computed correctly: 31.70
✅ chrF computed correctly: 57.11
✅ MCD computed correctly: 13.14 dB
✅ Comparison table created
✅ Visualizations generated (6 PNGs)
✅ Dashboard generated

## Next Steps

1. Fix sample count to exactly 300
2. Test BLASER on 5 samples to verify it works
3. Fix COMET model name
4. Remove "overall score" and show clear per-metric comparisons
5. Re-run test to verify fixes
