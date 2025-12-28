# Summary: Issues Found with Pipeline Comparison Test

## What We Tested

Ran small-sample test (10 samples) on Pipeline 1 (NLLB → MULTILINGUAL) for Efik language.

---

## ✅ What Worked

1. **BLEU**: 23.26 ✓
2. **chrF++**: 52.84 ✓
3. **Duplicate detection**: Found 33 duplicates in merged data ✓
4. **Deduplication**: Fixed - went from 332 to 299 samples ✓

---

## ❌ What Failed

### 1. COMET - Model Not Loading

**Error**: `"Model 'McGill-NLP/ssa-comet-qe' not supported by COMET."`

**Found**:
- Model IS downloaded: `/home/vacl2/multimodal_translation/services/evaluation/comet/checkpoints/models--McGill-NLP--ssa-comet-qe/snapshots/.../checkpoints/model.ckpt`
- The `download_model()` function in `comet_evaluator.py` line 68 is failing to find it
- Model exists locally but COMET library can't find it by name

**Possible Fixes**:
1. Load directly from checkpoint path instead of using `download_model()`
2. Check if COMET library version changed
3. Check if model needs to be re-downloaded or cache cleared

### 2. MCD - Wrong Method Name

**Error**: `'AudioMetrics' object has no attribute 'compute_mcd_from_audio'`

**Fix**: Use `compute_mcd_batch()` instead (as used in orchestrator)

### 3. BLASER - Dictionary Access Error

**Error**: `string indices must be integers, not 'str'`

**Cause**: Accessing LANGUAGES dict incorrectly in test script

**Fix**: Check line accessing `LANGUAGES[language]`

---

## Critical Issues to Fix

### Issue A: Duplicate Segment IDs (332 vs 300)

**Problem**:
- Synthesis CSV: 300 rows
- NMT CSV: 3,248 rows
- **Merge creates 332 rows** (33 duplicates!)
- After dedup: 299 samples (still missing 1?)

**Impact**: Evaluating wrong samples, confusing counts

**Fix Needed**: Add `.drop_duplicates(subset=['segment_id'])` to orchestrator after merge

### Issue B: "Overall Score" Is Meaningless

**Problem**: Results show "overall_score: 0.253" - users don't know what this means

**Why Bad**:
- Combines text and audio metrics arbitrarily
- No interpretation guidance
- 0.253 could be good or terrible - who knows?

**Fix Needed**: Remove "overall score" entirely, show individual metrics with clear labels:
- "BLEU: 31.70 (0-100, higher is better)"
- "MCD: 13.14 dB (lower is better)"

### Issue C: Testing on 300+ Samples with CPU is Wasteful

**Problem**: Original test ran on 332 samples (should be 300)

**Why Bad**:
- BLASER timed out (10 min for 332 samples)
- COMET also slow on CPU
- Wasted 20+ minutes to find bugs

**Recommendation**: Always test with 5-10 samples first!

---

## Next Steps

1. **Fix COMET loading** - Make it use the existing checkpoint or fix model name resolution
2. **Fix MCD/BLASER in test script** - Use correct method names
3. **Add deduplication to orchestrator** - Remove duplicate segment_ids after merge
4. **Remove overall score** - Show individual metrics clearly
5. **Update docs** - Recommend small-sample tests first

---

## Question for You

**COMET**: The model exists locally but `download_model("McGill-NLP/ssa-comet-qe")` fails.

Did this work before? Or has it always been broken?

If it worked before, something changed (COMET version? Cache location? Model registry?).
