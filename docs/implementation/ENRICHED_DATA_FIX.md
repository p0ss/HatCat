# Enriched Data Format Issue - Root Cause and Fix

**Date**: 2025-11-25
**Status**: ✅ FIXED

## Problem Summary

Training was failing with symptoms like:
- taste_development negative: 435 positive, 525 negative **extracted** but only **15+15 trained**
- neutral detector: 65 positive, 895 negative **extracted** but only **15+15 trained**
- Both poles "failed to graduate"

The enriched data (2,043 synsets from multicultural enrichment) was **never being used** during training.

## Root Cause

The enriched data was merged into the wrong format and the training pipeline couldn't access it.

### Expected Format

The data generation function `get_overlap_synsets_for_pole()` expects:

```json
{
  "overlaps": {
    "simplex1_pole1__simplex1_pole2": [
      {
        "synset_id": "synset.overlap.001",
        "lemmas": [...],
        "definition": "...",
        "applies_to_poles": [
          "simplex1_pole1",
          "simplex1_pole2"
        ]
      }
    ]
  }
}
```

### What We Had

The merge script created:

```json
{
  "simplex_name": {
    "negative": [synsets],
    "neutral": [synsets],
    "positive": [synsets]
  }
}
```

**Critical missing field**: `applies_to_poles` - without this, `get_overlap_synsets_for_pole()` returns empty lists.

## Why Training Showed Large Extractions But Small Training

When the training pipeline ran:
1. It called `get_overlap_synsets_for_pole(pole_id)` to find enriched synsets
2. This returned **empty** because synsets lacked `applies_to_poles`
3. The system fell back to old data (182 overlap pairs from original enrichment)
4. Old data was **highly imbalanced** (895 negative vs 65 positive)
5. Adaptive trainer only used **15+15** samples (minimal baseline before giving up)

## The Fix

Created `scripts/restructure_enriched_data.py` which:

1. Loads enriched data from `simplex_overlap_synsets_enriched.json`
2. For each simplex → pole → synsets:
   - Adds `'applies_to_poles': [pole_id]` to each synset
   - Creates synthetic overlap pair key: `{pole_id}__enriched`
   - Adds synsets to `overlaps[pair_key]`
3. Saves properly structured data

### Results

**Before**:
- 182 overlap pairs (old data only)
- Enriched synsets inaccessible
- Training used 15+15 samples

**After**:
- 221 overlap pairs (182 old + 39 new)
- **2,043 enriched synsets** now accessible via `get_overlap_synsets_for_pole()`
- Training can use full dataset

## Files Changed

### Created
- `scripts/diagnose_enriched_data_issue.py` - Diagnostic tool
- `scripts/restructure_enriched_data.py` - Data restructuring tool
- `docs/ENRICHED_DATA_FIX.md` - This document

### Modified
- `data/simplex_overlap_synsets_enriched.json` - Replaced with fixed version
- `data/simplex_overlap_synsets_enriched_broken.json` - Backup of broken version

## Verification

```python
import json

with open('data/simplex_overlap_synsets_enriched.json') as f:
    data = json.load(f)

# Test lookup
pole_id = 'taste_development_negative'
pair_key = f'{pole_id}__enriched'

synsets = data['overlaps'][pair_key]
print(f"Found {len(synsets)} synsets for {pole_id}")
print(f"First synset applies_to_poles: {synsets[0]['applies_to_poles']}")
```

Expected output:
```
Found 26 synsets for taste_development_negative
First synset applies_to_poles: ['taste_development_negative']
```

## Next Steps

1. **Re-run training** with fixed data:
   ```bash
   poetry run python scripts/train_s_tier_simplexes.py --device cuda
   ```

2. **Expected improvements**:
   - Neutral F1: 0.27 → 0.70+ (using full enriched dataset)
   - Training will use 100+ samples instead of 15+15
   - Better balance (343 negative, 1344 neutral, 356 positive available)
   - Lenses should now graduate

3. **Monitor training output** for:
   - "Generated X prompts (Y positive, Z negative)" - should show large numbers
   - Training should NOT show "15+15" anymore
   - Graduation should succeed for most poles

## Lessons Learned

1. **Data format compatibility is critical** - always verify new data matches expected structure
2. **Silent failures are dangerous** - `get_overlap_synsets_for_pole()` returned empty list with no error
3. **Integration tests needed** - should have tested data access before full training run
4. **Validation tests are not enough** - our validation tests passed but didn't catch the integration issue

## Prevention

Add test in `scripts/test_enriched_data.py`:

```python
def test_overlap_access():
    """Test 4: Verify enriched synsets are accessible via get_overlap_synsets_for_pole()"""
    from training.sumo_data_generation import get_overlap_synsets_for_pole

    simplexes = ['taste_development', 'affective_awareness', 'social_connection']
    poles = ['negative', 'neutral', 'positive']

    for simplex in simplexes:
        for pole in poles:
            pole_id = f"{simplex}_{pole}"
            synsets = get_overlap_synsets_for_pole(pole_id)

            assert len(synsets) > 0, f"No synsets found for {pole_id}!"

            # Check applies_to_poles field exists
            for synset in synsets:
                assert 'applies_to_poles' in synset
                assert pole_id in synset['applies_to_poles']
```

This would have caught the issue immediately.
