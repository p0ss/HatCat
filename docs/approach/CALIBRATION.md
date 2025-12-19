# Pack-Level Lens Calibration

## Overview

After training individual lenses, pack-level calibration identifies and corrects lenses that fire inappropriately relative to other lenses in the ensemble. This document describes the calibration framework, metrics, and tuning approach.

## Triple-Criteria Calibration (Current Approach)

The current calibration system uses three criteria to determine if a lens is well-calibrated:

### 1. Ancestor Criterion

**Test**: When we prompt with concept A's name, does lens A score higher than all of A's ancestor lenses?

**Pass condition**: A is rank 0 (highest score) among A + ancestors

**Rationale**: A concept should always fire more strongly than its parents when prompted directly. If "Walk" doesn't beat "Mathematics" and "Actions" when we prompt "Walk", the hierarchy is broken.

### 2. Random Criterion

**Test**: When we prompt with concept A's name, does lens A appear in the top 5 among random unrelated concepts?

**Pass condition**: A is in top 5 of A + 4 random concepts

**Rationale**: A lens should fire consistently on its own concept name regardless of what other concepts are loaded. This catches lenses that are generally weak.

### 3. Sibling Criterion

**Test**: When we prompt with concept A's name, does lens A score higher than all siblings (concepts sharing the same parent)?

**Pass condition**: A is rank 0 (highest score) among A + siblings

**Rationale**: Siblings compete for the same semantic space. If "Dog" can't beat "Cat" when prompted with "Dog", the lenses aren't discriminative enough.

### Passing All Three

A concept is considered **well-calibrated** when it passes all three criteria (at >=80% success rate across multiple tests). This approach replaces the separate sibling refinement process by integrating sibling competition into the main calibration loop.

## Contrastive Fine-Tuning

When a lens fails calibration, we use contrastive training to fix it:

### For Ancestor/Sibling Failures

```python
# Boost target lens
target_score → 1.0

# Suppress competitors that beat us
for competitor in competitors_that_beat_target:
    if competitor_score > target_score - 0.2:
        competitor_score → target_score - 0.3
```

This is more effective than just boosting the target because many lenses saturate at 1.0. We need to push competitors DOWN, not just push the target UP.

### For Random Criterion Failures

Simple boost training: increase target activation on its own concept name.

## Hierarchical Calibration Strategy

The key insight is that calibration should follow the hierarchy:

1. **Fix parents first**: If a parent concept doesn't fire, its children can never expand
2. **Layer-by-layer**: Calibrate layer 0, then layer 1, etc.
3. **Siblings compete locally**: Only need to beat siblings, not all 8k concepts

This reduces the search space from N×N to roughly N×depth for ancestors and N×siblings for sibling competition.

## The Calibration Matrix (Reference)

### Idealized Full Calibration

The complete calibration produces an N×N matrix where:
- **Rows**: Prompted concepts (what we ask the model about)
- **Columns**: Lens concepts (which lenses fire)
- **Cells**: Rank of the column concept when the row concept is prompted

```
                    Lens Concepts
                    A    B    C    D    ...
Prompted    A      [1]   45   230  12   ...
Concepts    B       23  [1]   89   456  ...
            C       12   34  [1]   78   ...
            D       5    67   123 [1]   ...
```

The diagonal (bracketed) represents each concept's rank when its own name is prompted - ideally rank 1, but natural variance means this won't always be the case.

### Expected Patterns

1. **Diagonal dominance**: Each concept should rank highly when prompted directly
2. **Hierarchical correlation**: Parent concepts rank higher when children are prompted
3. **Semantic clustering**: Related concepts (siblings, synonyms) show correlated activations
4. **Layer-weighted distribution**: Higher-layer (more abstract) concepts appear more frequently across all prompts

### Natural Variance Sources

- **Polysemy**: "Bank" activates both financial and geographic concepts
- **Abstraction level**: "Entity" fires broadly; "LeftMetatarsal2Bone" fires narrowly
- **Training data**: Some concepts have richer training hints than others
- **Prompt sensitivity**: Same concept shows different ranks with different phrasings

## Running Calibration

### Analysis Phase

```bash
python -m src.training.calibration.batched_analysis \
    --lens-pack lens_packs/your-pack \
    --concept-pack concept_packs/your-pack \
    --model your-model \
    --dual-criteria \
    --output lens_packs/your-pack/calibration_analysis.json
```

Output shows:
- Ancestor criterion pass rate
- Random criterion pass rate
- Sibling criterion pass rate
- All three criteria pass rate
- List of failing concepts with details

### Fine-Tuning Phase

```bash
python -m src.training.calibration.finetune \
    --lens-pack lens_packs/your-pack \
    --concept-pack concept_packs/your-pack \
    --model your-model \
    --analysis lens_packs/your-pack/calibration_analysis.json \
    --dual-criteria \
    --output lens_packs/your-pack/finetune_report.json
```

This will:
1. Process ancestor competition failures (contrastive training)
2. Process sibling competition failures (contrastive training)
3. Process random criterion failures (boost training)

### Calibration Cycle

For iterative calibration until convergence:

```bash
python -m src.training.calibration.cycle \
    --lens-pack lens_packs/your-pack \
    --concept-pack concept_packs/your-pack \
    --model your-model \
    --mode full \
    --max-cycles 10 \
    --threshold 1.0
```

This runs analyze → finetune → analyze cycles until the improvement rate drops below threshold.

## Convergence Criteria

Calibration is complete when:
- All three criteria pass rates > 90%
- Improvement between cycles < 1%
- No concepts failing all three criteria

## Output Format

### Analysis Result

```json
{
  "mode": "triple_criteria",
  "total_concepts": 7946,
  "lens_reports": {
    "ConceptName": {
      "ancestor_rank_0_rate": 0.8,
      "random_top5_rate": 1.0,
      "sibling_rank_0_rate": 0.6,
      "passes_ancestor_criterion": true,
      "passes_random_criterion": true,
      "passes_sibling_criterion": false,
      "failed_ancestors": ["Parent1", "Parent2"],
      "failed_siblings": ["Sibling1", "Sibling2"]
    }
  },
  "under_firing": ["concept1", "concept2"],
  "over_firing": ["concept3"],
  "well_calibrated": ["concept4", "concept5"]
}
```

### Fine-Tune Report

```json
{
  "total_lenses_processed": 150,
  "lenses_boosted": 140,
  "improvement_rate": 0.93,
  "results": [
    {
      "concept": "ConceptName",
      "action": "ancestor_competition",
      "before_in_top_k_rate": 0.0,
      "after_in_top_k_rate": 1.0,
      "improved": true
    }
  ]
}
```

## Implementation Files

- `src/training/calibration/batched_analysis.py` - Analysis with triple criteria
- `src/training/calibration/finetune.py` - Contrastive fine-tuning
- `src/training/calibration/cycle.py` - Iterative calibration cycles

## Future Work

- **Adaptive thresholds**: Learn pass rates from data rather than fixed 80%
- **Cross-pack calibration**: Ensure consistency when multiple packs are loaded
- **Online calibration**: Adjust based on production activation patterns
- **Hierarchical batching**: Process by layer for more efficient calibration
