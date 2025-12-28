# Evaluation System Fixes - Summary

## Issues Fixed

### 1. Unnecessary Re-merging with NMT Data ✅ FIXED

**Problem**:
- System loaded predicted synthesis CSV (300 samples)
- Then merged with full NMT CSV (2,948 samples)
- Created 325 merged rows with 28 duplicates
- Then deduplicated down to 297
- This was confusing and inefficient

**Root Cause**:
- The predicted synthesis CSV already contains ALL needed data
- NMT CSV has duplicate segment_ids (up to 3 copies of same segment)
- Merging created artificial duplicates

**Solution**:
- Load synthesis CSV and filter successful samples FIRST
- Deduplicate synthesis data (only 1 duplicate, not 28)
- Load NMT CSV and deduplicate IT too (3248 → 3056 rows)
- Then merge only the ground truth columns needed
- Result: Clean 299 → 299 merge

**Before**:
```
Loaded 300 samples (pipe-delimited)
Merged rows: 325
⚠️  Found 28 duplicate segment_ids
After removing duplicates: 297
```

**After**:
```
✓ Loaded 300 samples
✓ Successful syntheses: 300/300
⚠️  Found 1 duplicate segment_ids, removing...
✓ After deduplication: 299 samples
✓ Deduplicated NMT CSV: 3248 → 3056 rows
✓ Merged 299 samples with ground truth
```

---

### 2. Audio File Validation ✅ FIXED

**Problem**:
- System tried to find audio pairs AFTER computing text metrics
- If audio files were missing, audio metrics silently returned empty/None
- No clear indication of missing files

**Solution**:
- Pre-validate ALL audio files before computing any audio metrics
- Report validation results clearly:
  - Number of valid audio pairs found
  - List missing files (showing first 5)
- Fail loudly if audio metrics are required but no files found

**New Logging**:
```
Audio validation:
  ✓ Found 299/299 valid audio pairs

OR (if files missing):

Audio validation:
  ✓ Found 250/299 valid audio pairs
  ⚠️  Missing audio for 49 samples:
    - segment 12345: predicted audio missing (...)
    - segment 67890: reference audio missing (...)
    ... and 44 more
```

---

### 3. Improved Logging ✅ FIXED

**Problem**:
- Hard to understand what data was being used
- Sample counts were confusing
- Not clear when metrics were skipped

**Solution**:
- Clear, step-by-step logging with checkmarks (✓)
- Explicit sample counts at each stage
- Show what's being compared for text metrics
- Use icons: ✓ (success), ⚠️  (warning), ⚙️  (config)

**Example**:
```
Loading synthesis results: predicted_nllb_tgt_...csv
  ✓ Loaded 300 samples
  ✓ Successful syntheses: 300/300
  ✓ After deduplication: 299 samples

Loading ground truth references: nmt_predictions_multilang_finetuned_final.csv
  ✓ Deduplicated NMT CSV: 3248 → 3056 rows
  ✓ Merged 299 samples with ground truth

Text metrics will compare:
  - Hypothesis: 'text' column from synthesis CSV (input to TTS)
  - Reference: 'ground_truth_tgt_text' from NMT CSV

⚙️  Limited to 2 samples for testing

Evaluating 2 samples...
```

---

## Files Modified

**`/home/vacl2/multimodal_translation/services/evaluation/evaluate_pipelines.py`**

### Changes in `evaluate_language()` method (lines 146-257):

1. **Lines 188-198**: Filter successful syntheses and deduplicate BEFORE merging
   ```python
   # Filter successful syntheses only
   successful_df = pred_df[pred_df['success'] == True].copy()
   logger.info(f"  ✓ Successful syntheses: {len(successful_df)}/{len(pred_df)}")

   # Remove duplicates (if any)
   if successful_df['segment_id'].duplicated().any():
       dup_count = successful_df['segment_id'].duplicated().sum()
       logger.warning(f"  ⚠️  Found {dup_count} duplicate segment_ids, removing...")
       successful_df = successful_df.drop_duplicates(subset='segment_id', keep='first')
   ```

2. **Lines 200-231**: Simplified NMT loading and deduplication
   ```python
   # Deduplicate NMT CSV (it may have multiple rows per segment_id)
   nmt_original_count = len(nmt_df)
   nmt_df = nmt_df.drop_duplicates(subset='segment_id', keep='first')
   if nmt_original_count != len(nmt_df):
       logger.info(f"  ✓ Deduplicated NMT CSV: {nmt_original_count} → {len(nmt_df)} rows")

   # Simple merge to get only ground truth columns
   df = successful_df.merge(
       nmt_df[['segment_id', 'ground_truth_tgt_text', 'src_text']],
       on='segment_id',
       how='left'
   )
   ```

### Changes in `_compute_metrics()` method (lines 264-373):

1. **Lines 309-365**: Audio file pre-validation
   ```python
   # Validate audio files before computing metrics
   audio_pairs = []
   missing_audio = []

   for _, row in df.iterrows():
       # ... check file existence ...

   # Report audio validation results
   logger.info(f"\nAudio validation:")
   logger.info(f"  ✓ Found {len(audio_pairs)}/{len(df)} valid audio pairs")

   if missing_audio:
       logger.warning(f"  ⚠️  Missing audio for {len(missing_audio)} samples:")
       for msg in missing_audio[:5]:  # Show first 5
           logger.warning(f"    - {msg}")

   # Fail if audio metrics required but no audio found
   if audio_metrics_required and len(audio_pairs) == 0:
       raise RuntimeError("Audio metrics are required but no audio files were found...")
   ```

---

## Impact

### Performance
- **Faster**: No more merging 300 samples with 2,948 samples
- **Cleaner**: Direct 299 → 299 merge instead of 300 → 325 → 297

### Reliability
- **Explicit failures**: Missing audio files now cause clear errors
- **Better validation**: Pre-check audio files before starting metrics

### User Experience
- **Clear logging**: Easy to understand what's happening at each step
- **Better feedback**: Know exactly how many samples are being evaluated
- **Transparent**: See what data sources are being used

---

## Testing

Test command:
```bash
cd /home/vacl2/multimodal_translation/services/evaluation
uv run evaluate_pipelines.py -p pipeline_1 -l efik --limit 2
```

Expected output:
- ✅ No "Found 28 duplicates" warning
- ✅ Clean sample counts (300 → 299 → 299)
- ✅ Clear audio validation message
- ✅ All metrics computed successfully

---

## What Was NOT Changed

1. **Redundant NLLB evaluation**: Pipelines 1, 2, 5 still re-compute BLEU/chrF/COMET for same NLLB translations
   - Accepted as inefficiency for now
   - Can be optimized later with caching

2. **Output CSV structure**: No changes to result file formats
   - `{lang}_sample_scores.csv`
   - `{lang}_scores.csv`
   - `all_scores.csv`

3. **Metric computation**: BLEU, chrF, COMET, MCD, BLASER logic unchanged
   - All working correctly

4. **Pipeline comparison system**: `run_pipeline_comparison.py` not modified
   - Still available if needed
