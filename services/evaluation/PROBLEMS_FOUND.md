# Problems Found in Pipeline Comparison System

## Test Run Summary

**Test**: 10 samples from Pipeline 1 (NLLB → MULTILINGUAL) × Efik

**What Worked** ✅:
- BLEU: 23.26
- chrF++: 52.84

**What Failed** ❌:
- COMET: All 3 model names failed to load
- MCD: Method name mismatch
- BLASER: Code error

---

## Problem 1: Duplicate Segment IDs (332 vs 300)

**Discovered**:
- Synthesis CSV has 300 rows
- NMT predictions CSV has 3,248 rows
- Merge produces 332 rows → **33 duplicates**
- After deduplication: 299 samples (expected: 300)

**Impact**:
- System evaluates wrong number of samples
- May be evaluating same sample multiple times
- Confusing for users

**Root Cause**:
Either synthesis CSV or NMT CSV has duplicate segment_ids

**Fix**:
Add `.drop_duplicates(subset=['segment_id'])` after merge in orchestrator

---

## Problem 2: COMET Models Not Loading

**Error**:
```
Model 'Unbabel/wmt22-comet-da' not supported by COMET
Model 'Unbabel/XCOMET-XL' not supported by COMET
Model 'McGill-NLP/ssa-comet-qe' not supported by COMET
```

**All 3 model names failed!**

**Possible Causes**:
1. COMET library version mismatch
2. Models not downloaded
3. Wrong model naming format

**Next Step**:
Run `uv run python -c "from comet import download_model, available_metrics; print(available_metrics())"` to see what models ARE available

---

## Problem 3: MCD Method Not Found

**Error**:
```
'AudioMetrics' object has no attribute 'compute_mcd_from_audio'
```

**Cause**: Wrong method name in test script

**Fix**: Check AudioMetrics class for correct method name (probably `compute_mcd_batch` like in orchestrator)

---

## Problem 4: BLASER Error

**Error**:
```
string indices must be integers, not 'str'
```

**Cause**: Accessing LANGUAGES dict incorrectly

**Fix**: Use `LANGUAGES[language]['iso_code']` correctly

---

## Problem 5: Confusing "Overall Score"

**Issue**: Results show "overall_score: 0.253" which is meaningless

**Why it's bad**:
- Combines text and audio metrics arbitrarily
- No clear interpretation
- Users don't know if 0.253 is good or bad

**Fix**: Remove entirely, show individual metrics clearly

---

## Problem 6: Testing 300+ Samples on CPU

**Issue**: Running BLASER/COMET on 300+ samples on CPU takes forever

**Why it's bad**:
- BLASER times out (10 min timeout for 300 samples)
- COMET is also slow on CPU
- Wasted time for testing

**Fix**: Always test with 5-10 samples first before full evaluation

---

## Next Steps

1. ✅ **Fixed**: Duplicate removal implemented in test script
2. **TODO**: Fix COMET - find which model actually works
3. **TODO**: Fix MCD method name
4. **TODO**: Fix BLASER LANGUAGES access
5. **TODO**: Remove "overall score" from comparator
6. **TODO**: Add duplicate removal to orchestrator
7. **TODO**: Update test docs to recommend small samples first

---

## Key Lesson

**Always test with 5-10 samples first, especially on CPU!**

Running 300-sample evaluations with heavy neural models (COMET, BLASER) on CPU is a waste of time. Test small first to catch bugs quickly.
