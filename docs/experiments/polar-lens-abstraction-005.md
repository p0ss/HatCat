# Experiment: Polar Lens Abstraction

## Hypothesis

**Claim**: Concept detection improves when lenses support bipolar probes (positive/negative poles) trained on success vs failure mode examples, with scoring that accounts for both signal strength and direction confidence.

**Rationale**:
- Concepts often have natural opposites (e.g., "cooperation" vs "conflict")
- Training separate probes on positive and negative examples provides disambiguation
- The difference between poles gives directional signal, not just presence/absence
- Statistical confidence scoring (Wilson lower bound) provides robust ranking

## Background

Previous work trained "polar MELD lenses" with graph-distant negatives for L1-L3 ontological levels. These produced `*_positive.pt` and `*_negative.pt` probe pairs per concept. This experiment upgrades the lens abstraction to support these multi-polar probes natively.

## Design

### Core Changes

1. **LensPolarity enum**: `DEFAULT`, `POSITIVE`, `NEGATIVE`
2. **ConceptMetadata extension**:
   - `level` field for ontological hierarchy (distinct from model `layer`)
   - `polar_lenses` dict mapping polarity to file paths
   - `is_polar` property for detection
3. **MetadataLoader polar pack detection**: Auto-detects L1/L2/L3 directory structure
4. **LensLoader polar loading**: Parallel loading of positive/negative probe pairs
5. **BatchedLensBank polar inference**: Batched forward pass for both poles simultaneously
6. **Wilson score confidence**: Statistical confidence scoring for polar outputs

### Wilson Score Formula

The key insight: treating probe outputs as "votes" for each pole enables statistical confidence estimation.

```python
def wilson_score_interval(positive: float, negative: float, scale=10.0):
    # Scale probabilities to vote counts (0.8 -> 8 votes)
    pos_votes, neg_votes = positive * scale, negative * scale
    n = pos_votes + neg_votes
    p = pos_votes / n  # Proportion toward positive

    # Wilson lower bound formula
    wilson_lower = standard_wilson_formula(p, n, z=1.645)

    # Blend direction confidence with evidence boost
    # This rewards high total activation (concept is "present")
    # even when direction is uncertain
    evidence_boost = (positive + negative) / 2.0
    confidence = wilson_lower * 0.4 + evidence_boost * 0.6

    polarity = 2 * p - 1  # -1 to +1
    return confidence, polarity
```

**Key property**: A concept with pos=0.8, neg=0.4 ranks higher than pos=0.8, neg=0.2 because more total activation indicates the concept is more "present", even though direction is less certain.

### Terminology Clarification

- **layer**: Transformer model layer where activations are extracted (e.g., 17)
- **level**: Ontological hierarchy position (L1=1, L2=2, L3=3)

These were previously conflated. Now tracked separately via `ConceptMetadata.level` and `ConceptMetadata.ontological_level` property.

## Implementation

### Files Modified

| File | Changes |
|------|---------|
| `lens_types.py` | Added `LensPolarity`, `level` field, `polar_lenses` dict, `is_polar` property |
| `lens_loader.py` | Polar pack detection, `load_polar_metadata()`, `_load_polar_lenses()` |
| `lens_cache.py` | `loaded_polar_negative_lenses`, `polar_concepts`, `add_polar_lens()` |
| `lens_batched.py` | Parallel negative pole weights, `wilson_score_interval()`, polar scoring in `forward()` |
| `lens_manager.py` | Polar pack base layer loading, ontological level in output |
| `test_self_concept_monitoring.py` | Updated output format (score/level instead of probability/layer) |

### Polar Pack Structure

```
lens_packs/apertus-8b_first-light/
├── L1/
│   └── layer17/
│       ├── results.json
│       ├── concept1_positive.pt
│       ├── concept1_negative.pt
│       └── ...
├── L2/
│   └── layer17/
│       └── ...
└── L3/
    └── layer17/
        └── ...
```

The `results.json` contains:
```json
{
  "concepts": [
    {
      "node_id": "concept-name",
      "positive": {"file": "concept_positive.pt", "accuracy": 0.95},
      "negative": {"file": "concept_negative.pt", "accuracy": 0.92}
    }
  ]
}
```

## Results

### Wilson Score Behavior

| pos | neg | confidence | polarity | interpretation |
|-----|-----|------------|----------|----------------|
| 1.0 | 0.0 | 0.61 | +1.00 | Strong positive, clear direction |
| 0.8 | 0.2 | 0.52 | +0.60 | Moderate positive |
| 0.8 | 0.4 | 0.53 | +0.33 | Higher presence, less clear direction |
| 0.5 | 0.5 | 0.41 | 0.00 | Balanced, uncertain |
| 0.2 | 0.8 | 0.52 | -0.60 | Moderate negative |
| 1.0 | 1.0 | 0.73 | 0.00 | Maximum presence, no direction |

The formula correctly ranks 0.8/0.4 above 0.8/0.2 (more "present"), matching intuition.

### Integration Status

- Polar pack auto-detection: Working
- Metadata loading with level tracking: Working
- Parallel lens loading: Working
- Batched inference with both poles: Working
- Wilson score calculation: Working
- End-to-end test: Blocked by GPU memory (compositor using 10GB VRAM)

## Observations

1. **Terminology clarity matters**: The layer/level confusion caused bugs where ontological hierarchy was displayed as model layer numbers.

2. **Evidence vs direction tradeoff**: Pure Wilson score favors clear direction over total activation. The 40/60 blend with evidence boost better matches intuition about concept "presence".

3. **Backward compatibility**: Non-polar lenses continue to work unchanged - they just use probability directly without Wilson scoring.

## Code Artifacts

- `src/hat/monitoring/lens_types.py` - Core types with polar support
- `src/hat/monitoring/lens_loader.py` - Polar pack detection and loading
- `src/hat/monitoring/lens_cache.py` - Polar lens caching
- `src/hat/monitoring/lens_batched.py` - Wilson score and batched polar inference
- `src/hat/monitoring/lens_manager.py` - Manager integration
- `tests/test_self_concept_monitoring.py` - Test harness

## Next Steps

1. Run full end-to-end test when GPU memory available
2. Evaluate Wilson score rankings on real self-concept prompts
3. Consider exposing confidence and polarity separately in output
4. Potential: tune the 40/60 direction/evidence blend ratio based on empirical results
