# Two-Head Tripole Architecture

## Overview

This document describes the two-head architecture for training three-pole simplex lenses, as implemented in `scripts/train_s_tier_tripole_two_head.py`.

## Comparison: 3 Independent Classifiers vs 2 Heads

### Old Approach: 3 Independent Binary Classifiers

**Structure:**
- Classifier 1: Detects μ− (negative pole)
- Classifier 2: Detects μ0 (neutral homeostasis)
- Classifier 3: Detects μ+ (positive pole)

**Problems:**
1. **Inconsistency**: Three independent models can fire contradictory outputs
   - Example: Both μ− and μ+ could activate simultaneously
   - No guarantee of mutual exclusivity
2. **Data inefficiency**: Each classifier only sees 2/3 of available pole data
   - μ− classifier doesn't learn from μ+ examples
   - μ0 classifier doesn't learn from extreme examples
3. **Computational cost**: 60 total classifiers (20 simplexes × 3)
4. **No structural enforcement**: Nothing ensures the three poles form a coherent axis

### New Approach: 2-Head Architecture

**Structure:**
- **Head A (Sign)**: Learns axis direction (positive vs negative)
  - Trained on: positive_extreme (label=1) vs negative_extreme (label=0)
  - Sees: ALL extreme examples from both poles
  - Learns: Which side of the axis is active

- **Head B (Extremeness)**: Learns magnitude (extreme vs neutral)
  - Trained on: (positive_extreme + negative_extreme) (label=1) vs neutral (label=0)
  - Sees: ALL examples from all three poles
  - Learns: Distance from center (homeostasis)

**Inference:**
Combine both heads to get 3-pole classification:
```python
p_sign = sign_head(h)        # P(positive | extreme)
p_ext = extremeness_head(h)  # P(extreme)

# Three-pole probabilities
P(neutral) = 1 - p_ext
P(positive) = p_ext * p_sign
P(negative) = p_ext * (1 - p_sign)
```

**Advantages:**
1. **Mathematically sound**: Decomposes 3-pole problem into orthogonal components
   - Sign = direction along axis
   - Extremeness = magnitude along axis
2. **Data efficient**: Each head sees MORE data
   - Sign head: 300 examples (150 pos + 150 neg)
   - Extremeness head: 500 examples (300 extreme + 200 neutral)
3. **Guaranteed consistency**: Can't fire contradictory poles
   - P(neutral) + P(positive) + P(negative) = 1 (by construction)
4. **Fewer models**: 40 heads (20 simplexes × 2) vs 60 classifiers
5. **Interpretable**: Clear semantic meaning for each head

## Data Generation

### Sample Counts

```python
N_POS_EXTREME = 150   # Positive extreme examples
N_NEG_EXTREME = 150   # Negative extreme examples
N_NEUTRAL = 200       # Neutral examples (larger pool)
Total: 500 samples
```

**Rationale:**
- **Equal extremes**: Balanced sign head (150 vs 150)
- **More neutral**: Harder to learn neutral boundary, needs more data
- **Behavioral ratio**: 60% behavioral, 40% definitional (same as before)

### Dataset Construction

**Sign Head Dataset:**
```
Train: 240 samples (120 pos extreme + 120 neg extreme)
Test: 60 samples (30 pos extreme + 30 neg extreme)
Total: 300 extreme examples
```

**Extremeness Head Dataset:**
```
Train: 400 samples (240 extreme + 160 neutral)
Test: 100 samples (60 extreme + 40 neutral)
Total: 500 examples
```

## Training Configuration

### Trainer Settings

```python
DualAdaptiveTrainer(
    model=model,
    tokenizer=tokenizer,
    validation_layer_idx=12,     # Layer to monitor
    validate_lenses=True,        # Enable adaptive falloff
    validation_mode="falloff",   # Falloff-based tier grading
    train_activation=True,
    train_text=False
)
```

### Target Performance

**A-tier requirements (both heads):**
- Test F1: 0.95+
- Calibration score: 0.95+
- Tier: A

**Expected outcomes:**
- Sign head: Should achieve A-tier easily (clear axis direction)
- Extremeness head: May need more iterations (neutral boundary harder)

## Inference: Combining Heads

### Hard Classification

```python
def classify_tripole(h, sign_head, extremeness_head, tau_ext=0.5):
    """
    Hard classification into one of three poles.

    Args:
        h: Hidden state
        sign_head: Trained sign classifier
        extremeness_head: Trained extremeness classifier
        tau_ext: Threshold for extremeness (default 0.5)

    Returns:
        pole: "negative", "neutral", or "positive"
    """
    p_sign = sign_head.predict_proba(h)  # P(positive | extreme)
    p_ext = extremeness_head.predict_proba(h)  # P(extreme)

    if p_ext < tau_ext:
        return "neutral"
    elif p_sign >= 0.5:
        return "positive"
    else:
        return "negative"
```

### Soft Classification (for steering)

```python
def get_tripole_probabilities(h, sign_head, extremeness_head):
    """
    Soft 3-way classification for homeostatic steering.

    Returns:
        dict: {
            "negative": P(negative),
            "neutral": P(neutral),
            "positive": P(positive)
        }
    """
    p_sign = sign_head.predict_proba(h)
    p_ext = extremeness_head.predict_proba(h)

    return {
        "neutral": 1 - p_ext,
        "positive": p_ext * p_sign,
        "negative": p_ext * (1 - p_sign)
    }
```

## Steering Vectors

### Extracting Direction Vectors

Each head learns a direction vector in activation space:

```python
# Sign direction (positive vs negative)
w_sign = sign_head.classifier.coef_[0]

# Extremeness direction (extreme vs neutral)
w_ext = extremeness_head.classifier.coef_[0]
```

### Homeostatic Steering

To steer toward neutral (μ0):
```python
def steer_to_neutral(h, w_ext, strength=1.0):
    """Steer toward neutral by reducing extremeness."""
    return h - strength * w_ext
```

To steer toward positive pole (μ+):
```python
def steer_to_positive(h, w_sign, w_ext, strength=1.0):
    """Steer toward positive pole."""
    # First ensure we're extreme (not neutral)
    h_extreme = h + strength * w_ext
    # Then ensure we're on positive side
    h_positive = h_extreme + strength * w_sign
    return h_positive
```

To steer toward negative pole (μ−):
```python
def steer_to_negative(h, w_sign, w_ext, strength=1.0):
    """Steer toward negative pole."""
    # First ensure we're extreme
    h_extreme = h + strength * w_ext
    # Then ensure we're on negative side
    h_negative = h_extreme - strength * w_sign
    return h_negative
```

## Quality Validation

### Per-Head Metrics

Both heads should achieve:
- **Test F1**: 0.95+ (A-tier)
- **Calibration**: 0.95+ (well-calibrated probabilities)
- **Tier**: A (strict validation passing)

### Combined Tripole Metrics

After training both heads, validate the combined system:

1. **Pole separation**: Mean activation difference between correct pole and other poles
   - Target: 0.85+

2. **Probability coherence**: P(−) + P(0) + P(+) ≈ 1.0
   - Target: Sum within [0.98, 1.02]

3. **Cross-activation**: Max incorrect pole activation
   - Target: <0.10

## Output Structure

```
results/s_tier_tripole_two_head/run_TIMESTAMP/
├── training.log
├── results.json
└── <dimension>/
    ├── results.json
    ├── sign_head/
    │   └── activation_lens.pkl
    └── extremeness_head/
        └── activation_lens.pkl
```

### Results Schema

```json
{
  "dimension": "social_self_regard",
  "sign_head": {
    "test_f1": 0.97,
    "test_precision": 0.96,
    "test_recall": 0.98,
    "tier": "A",
    "calibration_score": 0.96,
    "total_iterations": 3
  },
  "extremeness_head": {
    "test_f1": 0.94,
    "test_precision": 0.93,
    "test_recall": 0.95,
    "tier": "B+",
    "calibration_score": 0.93,
    "total_iterations": 8
  },
  "success": true
}
```

## Comparison to Original Design

### From `tripole_lens_design.md`

Our implementation follows the design exactly:

1. ✓ **Sign head trained only on extremes** (pos vs neg)
2. ✓ **Extremeness head trained on extremes + neutral** (extreme vs neutral)
3. ✓ **Joint loss**: We train both heads independently (equivalent to λ_sign=1, λ_ext=1)
4. ✓ **Inference formula**: Exactly as specified
   - P(neutral) = 1 - p_ext
   - P(positive) = p_ext · p_sign
   - P(negative) = p_ext · (1 - p_sign)

### Extensions Beyond Original Design

1. **Adaptive falloff validation**: Each head gets tier grading
2. **Separate lens saving**: Can load and use heads independently
3. **Comprehensive logging**: Track both heads through training
4. **80/20 splits**: Proper train/test split for each head

## Benchmarking Two-Head Lenses

When benchmarking (see `LENS_BENCHMARK_QUICK_START.md`), load both heads:

```python
def load_tripole_lens(simplex_dir):
    """Load both heads for a tripole simplex."""
    sign_head = LinearLens.load(
        simplex_dir / "sign_head" / "activation_lens.pkl"
    )
    extremeness_head = LinearLens.load(
        simplex_dir / "extremeness_head" / "activation_lens.pkl"
    )
    return sign_head, extremeness_head

def predict_tripole(h, sign_head, extremeness_head):
    """Get 3-pole probabilities."""
    p_sign = sign_head.predict_proba(h)
    p_ext = extremeness_head.predict_proba(h)

    return {
        "negative": p_ext * (1 - p_sign),
        "neutral": 1 - p_ext,
        "positive": p_ext * p_sign
    }
```

## Future Work

1. **Exponential scaling**: Apply S_TIER_TRAINING_STRATEGY.md to both heads
2. **Calibration**: Apply Platt scaling to each head's outputs
3. **Threshold tuning**: Optimize τ_ext for each simplex dimension
4. **Joint optimization**: Train both heads with shared representation
5. **Visualization**: Plot decision boundaries in 2D (sign × extremeness)

## References

- Design document: `docs/tripole_lens_design.md`
- Training script: `scripts/train_s_tier_tripole_two_head.py`
- Training strategy: `docs/S_TIER_TRAINING_STRATEGY.md`
- Benchmark guide: `docs/LENS_BENCHMARK_QUICK_START.md`
