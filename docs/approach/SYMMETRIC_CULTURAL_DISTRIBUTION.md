# Symmetric Cultural Distribution in Multicultural Enrichment

## Problem Statement

When generating culturally-diverse training data for tripole lenses, there's a risk of inadvertently learning spurious correlations between cultures and emotional valences. For example:

- Japanese concepts clustering in neutral/harmonious pole
- German concepts clustering in angry/negative pole
- Latin concepts clustering in passionate/positive pole

This would allow the model to evade detection by encoding deceptive behavior using culture-specific terminology rather than the actual semantic content.

## Solution: Symmetric Cultural Distribution

### Principle

**Every culture experiences the full range of emotional valences across all simplexes.**

Therefore, training data must reflect this reality by ensuring cultural diversity is distributed **symmetrically across all three poles** (negative, neutral, positive) for each simplex.

### Implementation: Post-Analysis Validation

**Critical insight:** Prompting alone cannot guarantee symmetric distribution because each API request is independent and lacks global context about the overall cultural distribution.

**Solution:** Two-stage approach:

1. **Initial enrichment with diversity prompts** - Generate culturally-diverse synsets across all poles
2. **Post-analysis validation and correction** - Measure actual output and generate targeted corrections

#### Stage 1: Initial Enrichment

The multicultural enrichment prompts include diversity guidance:

```
CRITICAL REQUIREMENT - SYMMETRIC CULTURAL DISTRIBUTION:
Each culture has concepts spanning ALL emotional valences. Avoid clustering:
- DON'T: Make all Japanese concepts neutral/harmonious
- DO: Find Japanese concepts for this specific pole ("{pole}")
- Remember: Every culture experiences the full range of {simplex} across negative/neutral/positive
```

This provides *guidance* but cannot *guarantee* balance.

#### Stage 2: Post-Analysis Validation

After API execution, analyze actual distribution:

```bash
poetry run python scripts/validate_cultural_distribution.py
```

This script:
- Extracts culture from synset_id (e.g., `japanese_wa.positive.001` → `japanese`)
- Counts each culture's representation across negative/neutral/positive poles
- Identifies imbalances (>2x ratio between max and min pole counts)
- Generates targeted correction requests for imbalanced cultures

### Example: Japanese Concepts Across Poles

For the `affective_coherence` simplex:

**Negative (confusion/混乱):**
- 混沌 (konton) - chaos, disorder
- 困惑 (konwaku) - bewilderment, perplexity
- 迷い (mayoi) - hesitation, losing one's way

**Neutral (stability/平静):**
- 平常心 (heijōshin) - calm mind, equanimity
- 落ち着き (ochitsuki) - composure
- 安定 (antei) - stability

**Positive (harmony/調和):**
- 和 (wa) - harmony, peace
- 調和 (chōwa) - balance, harmony
- 一致 (itchi) - agreement, unity

This distribution ensures the model learns semantic content rather than cultural associations.

## Validation

After API enrichment, verify symmetric distribution:

```python
# Count cultures per pole
cultural_counts = defaultdict(lambda: {'negative': 0, 'neutral': 0, 'positive': 0})

for simplex, poles in enriched_data.items():
    for pole, synsets in poles.items():
        for synset in synsets:
            culture = synset['synset_id'].split('_')[0]  # e.g., "japanese_wa.positive.001"
            cultural_counts[culture][pole] += 1

# Check balance
for culture, pole_counts in cultural_counts.items():
    max_count = max(pole_counts.values())
    min_count = min(pole_counts.values())
    imbalance = max_count / min_count if min_count > 0 else float('inf')

    if imbalance > 2.0:
        print(f"WARNING: {culture} shows {imbalance:.1f}x imbalance across poles")
```

Target: **<2x imbalance** across poles for each culture.

## Benefits

1. **Prevents Evasion**: Models cannot hide deceptive behavior in culture-specific encoding
2. **Cultural Authenticity**: Reflects the reality that all cultures experience full emotional range
3. **Robustness**: Training data covers diverse manifestations of each simplex pole
4. **Fairness**: Avoids perpetuating cultural stereotypes in the training data

## Files

- `scripts/balance_simplex_overlaps_multicultural.py` - Generates enrichment plan with symmetric distribution prompts
- `data/balance_enrichment_multicultural.json` - 77 API requests with symmetric cultural requirements
- `scripts/execute_multicultural_api_calls.py` - Executes all requests automatically

## Next Steps

1. Execute 77 API requests to generate 975 multicultural synsets
2. Validate cultural distribution across poles
3. If imbalance detected, generate targeted requests to balance specific cultures
4. Merge enriched data into training set
5. Re-train tripole lenses with balanced, culturally-diverse data

---

**Updated:** 2025-11-25
**Status:** Ready for API execution
