# Multicultural Enrichment Quick Start Guide

## Overview

This guide walks you through generating 2,043 culturally-diverse synsets to balance and enrich the tripole lens training data.

## Prerequisites

```bash
export ANTHROPIC_API_KEY='your-key-here'
pip install anthropic
```

## Step 1: Generate Enrichment Plan (Already Done!)

The enrichment plan has been generated with symmetric cultural distribution:

```bash
poetry run python scripts/balance_simplex_overlaps_multicultural.py
```

Output: `data/balance_enrichment_multicultural.json`

Summary:
- 77 API requests
- 2,043 total synsets (975 multicultural + 1,068 balance)
- Symmetric cultural distribution across all poles
- Target: 115-122 overlaps per pole (balanced)

## Step 2: Execute API Requests

### Automatic Execution (Recommended)

Run all 77 requests automatically:

```bash
poetry run python scripts/execute_multicultural_api_calls.py
```

Features:
- Automatic execution of all requests
- Resumable (skips already-completed files)
- Rate limiting (1 second between requests)
- Error handling (saves raw responses on failures)
- Progress tracking

Output directory: `data/enrichment_responses/`

Expected runtime: ~2-3 minutes (with 1s rate limiting)

### Manual Execution (If Needed)

If you prefer manual control or want to use a different API:

1. Load plan: `data/balance_enrichment_multicultural.json`
2. For each request, send `prompt` field to your API
3. Save responses to: `data/enrichment_responses/{simplex}_{pole}_{type}_batch{N}.json`

## Step 3: Verify Responses

Check that all 77 requests completed:

```bash
find data/enrichment_responses/ -name "*.json" | wc -l
```

Expected: 77 JSON files

Check for any failures (text files indicate parse errors):

```bash
find data/enrichment_responses/ -name "*.txt"
```

## Step 4: Validate Cultural Distribution (Critical!)

Run post-analysis validation to detect cultural clustering:

```bash
poetry run python scripts/validate_cultural_distribution.py
```

This analyzes the actual API responses to check if cultures are symmetrically distributed across poles.

**Why post-analysis is necessary:**
- Each API request is independent (no global context)
- Prompting alone cannot guarantee balanced distribution
- Must measure actual output to detect clustering patterns

The script checks:
- Each culture appears across all poles with <2x imbalance
- Minimum representation (≥3 synsets per culture per pole)
- Identifies which cultures need correction

**If imbalances detected:**
```bash
poetry run python scripts/generate_cultural_corrections.py
```

This generates targeted API requests to balance specific cultures across specific poles.

## Step 5: Merge Responses into Training Data

Integrate the enriched synsets:

```bash
poetry run python scripts/merge_enrichment_responses.py
```

This merges responses into: `data/simplex_overlap_synsets_enriched.json`

## Step 6: Re-train Tripole Lenses

Train with balanced, culturally-diverse data:

```bash
poetry run python scripts/train_s_tier_simplexes.py --device cuda
```

Expected improvements:
- Neutral F1: 0.27 → 0.70+ (2.6x improvement)
- Reduced variance across runs
- Better generalization to non-Western concepts

## Files

### Scripts
- `scripts/balance_simplex_overlaps_multicultural.py` - Generate enrichment plan
- `scripts/execute_multicultural_api_calls.py` - Execute API requests
- `scripts/validate_cultural_distribution.py` - Check cultural symmetry
- `scripts/merge_enrichment_responses.py` - Integrate responses

### Data
- `data/balance_enrichment_multicultural.json` - Enrichment plan (77 requests)
- `data/enrichment_responses/` - API responses (77 files)
- `data/simplex_overlap_synsets_enriched.json` - Merged training data

### Documentation
- `docs/SYMMETRIC_CULTURAL_DISTRIBUTION.md` - Cultural distribution strategy
- `docs/TRIPOLE_TRAINING_SYSTEM.md` - Tripole architecture overview
- `docs/TRIPOLE_BALANCE_SOLUTION.md` - Data imbalance solution

## Troubleshooting

### API Key Issues

```bash
echo $ANTHROPIC_API_KEY  # Should show your key
export ANTHROPIC_API_KEY='your-key-here'
```

### Rate Limiting

If you hit rate limits, the script will fail gracefully. Simply re-run - it will skip completed requests.

### Parse Errors

If JSON parsing fails, raw responses are saved as `.txt` files. Manually inspect and convert to JSON format:

```json
{
  "metadata": {...},
  "synsets": [...]
}
```

### Cultural Imbalance

If validation shows >2x imbalance for a specific culture, generate targeted requests to balance:

```python
# Example: Add Japanese concepts to negative pole
python scripts/generate_targeted_cultural_request.py \
  --culture japanese \
  --pole negative \
  --count 10
```

## Cost Estimate

- Model: claude-sonnet-4-20250514
- Requests: 77
- Avg tokens per request: ~1500 input + 4000 output
- Total cost: ~$10-15 USD

## Timeline

- API execution: 2-3 minutes
- Validation: 30 seconds
- Merge: 1 minute
- Re-training: 30-60 minutes (13 simplexes)

Total: ~1 hour from start to finish

---

**Ready to go?**

```bash
export ANTHROPIC_API_KEY='your-key-here'
poetry run python scripts/execute_multicultural_api_calls.py
```

Then sit back and let it run!
