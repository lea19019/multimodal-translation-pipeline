# Fixes Needed for Pipeline Comparison System

## What We Learned

The original `evaluation.py` works correctly! It successfully:
- Loads COMET with default initialization
- Evaluates multiple languages
- Creates manifests and summaries
- Generates cross-language visualizations

## What's Broken in Pipeline Comparison System

The NEW system I created (`run_pipeline_comparison.py`, `pipeline_orchestrator.py`, etc.) has these issues:

### 1. Duplicate Segment IDs ✅ DIAGNOSED
**Problem**: Merge creates 332 rows from 300 synthesis samples (33 duplicates)

**Fix**: Add `.drop_duplicates(subset=['segment_id'], keep='first')` after merging in `pipeline_orchestrator.py`

### 2. COMET Should Work ✅ RESOLVED
**Problem**: Was trying multiple model names unnecessarily

**Fix**: Just use `CometEvaluator()` with defaults (same as evaluation.py)
- The existing code already uses the right model
- If it's failing in orchestrator, it's a different issue (environment, timeout, etc.)

### 3. "Overall Score" is Confusing ❌ STILL NEEDS FIX
**Problem**: Combining disparate metrics into single score

**Solution**: Look at how `evaluation.py` does it:
- Line 288-292: Shows each metric separately with clear labels
- Line 423-428: For aggregate stats, shows mean/median/std/min/max per metric
- Line 433-434: Shows per-language breakdown with individual scores
- **NEVER combines into single "overall score"**

**Fix**: Remove `overall_score` calculation from `pipeline_comparator.py`, show individual metrics

### 4. MCD Method Name ❌ MINOR ISSUE
**Problem**: Test script uses wrong method name

**Fix**: Use `compute_mcd_batch()` not `compute_mcd_from_audio()`

### 5. Need Better Result Presentation ❌ NEEDS IMPROVEMENT
**Current**: Confusing comparison table

**Look at evaluation.py**:
- Clear execution summary (lines 281-293)
- Per-language breakdown (lines 429-434)
- Cross-language visualizations (lines 398-408)
- Language×Metric heatmap
- Mean/median/std for each metric

**Fix**: Format results like evaluation.py does

---

## Action Plan

1. **Fix orchestrator duplicate issue**:
   - Add deduplication after merge in `pipeline_orchestrator.py` line ~176

2. **Keep COMET as-is**:
   - Default initialization works
   - If it fails, it's likely timeout or environment issue

3. **Remove overall_score**:
   - Edit `pipeline_comparator.py`
   - Remove normalization and composite scoring
   - Show raw metrics only

4. **Improve result formatting**:
   - Follow `evaluation.py` pattern
   - Show mean/median/std/min/max per metric
   - Clear per-language and per-pipeline breakdowns

5. **Simplify visualizations**:
   - Keep heatmaps (pipeline × language for each metric)
   - Remove radar charts and confusing aggregates
   - Add statistics tables like evaluation.py

---

## Key Takeaway

**Don't reinvent the wheel!**

The `evaluation.py` already has a working pattern:
- Clear metric reporting
- Statistical aggregation (mean/median/std)
- Language×Metric heatmaps
- No confusing composite scores

Just adapt that pattern for cross-pipeline comparison.
