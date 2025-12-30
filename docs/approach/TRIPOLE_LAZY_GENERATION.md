# Tripole Lens Training with Lazy Generation

**Date**: 2025-11-25
**Status**: ✅ IMPLEMENTED

## Overview

Created a new training script that properly combines:
1. **Joint tripole architecture** (TripoleLens with shared 3-class softmax)
2. **Lazy data generation** (start small, scale up only if needed)

This replaces the incorrect `train_s_tier_simplexes.py` approach which was using binary classifiers instead of the proper joint tripole architecture.

## Problem with Previous Approach

The `train_s_tier_simplexes.py` script was:
- Training **3 separate binary classifiers** using `DualAdaptiveTrainer`
- This loses the mathematical properties of the shared axis
- Does not use the margin-based discriminative gradiated hyperplane
- Pre-extracts large datasets wastefully

User feedback:
> "i thought adapting lazy generation to the proper tripole training was what we were doing. I spent a lot of time on that math and its crucial to establishing the shared axis and discriminative gradiated hyperplane"

## New Approach: `train_s_tier_tripole_lazy.py`

### Architecture

Uses the correct **TripoleLens** from `src/training/tripole_classifier.py`:

```python
class TripoleLens(nn.Module):
    """Joint three-pole linear lens with learnable margins."""

    def __init__(self, hidden_dim: int, n_poles: int = 3):
        # Shared linear projection: h -> logits
        self.linear = nn.Linear(hidden_dim, n_poles)  # [3, hidden_dim]

        # Learnable per-pole margins
        self.log_margins = nn.Parameter(...)
```

**Key properties:**
- Joint 3-class softmax (poles compete naturally)
- Shared weight matrix W defines the psychological axis
- Margin-maximization loss ensures discriminability
- Orthogonality regularizer encourages diverse representations

### Lazy Generation Strategy

```
Start: 60 samples per pole (180 total: 60 neg, 60 neu, 60 pos)
Increment: +60 per pole if F1 < 0.70 (graduation threshold)
Max: 300 samples per pole
```

**Process:**
1. Generate balanced 3-class data (equal samples per pole)
2. Extract activations from layer 12
3. Train joint tripole lens
4. Check if F1 >= 0.70:
   - YES → Graduate and save
   - NO → Increment samples and retry
5. Repeat until graduation or max samples reached

### Data Generation

For each simplex, generates balanced training data:

```python
# Generate contrastive datasets for each pole
for pole in [negative, neutral, positive]:
    prompts, labels = create_simplex_pole_training_dataset_contrastive(
        pole_data=pole_data,
        other_poles_data=[other two poles],
        behavioral_ratio=0.6,
        prompts_per_synset=5
    )

    # Extract only POSITIVE examples for this pole
    positive_prompts = [p for p, label in zip(prompts, labels) if label == 1]

    # Downsample to target if needed
    positive_prompts = positive_prompts[:n_samples_per_pole]

    # Label with pole index (0, 1, or 2)
    all_prompts.extend(positive_prompts)
    tripole_labels.extend([pole_idx] * len(positive_prompts))
```

**Critical: Balanced sampling**
- Joint tripole training is catastrophically sensitive to class imbalance
- Must maintain equal samples per pole (documented in `TRIPOLE_TRAINING_SYSTEM.md`)
- 2.8x performance hit if imbalanced

### Loss Function

```python
def tripole_loss(logits, labels, lens, lambda_margin=0.5, lambda_ortho=1e-4):
    """
    Combined loss for tripole training:
    1. Cross-entropy: Standard multiclass classification
    2. Margin loss: Ensures correct pole dominates by adaptive margin
    3. Orthogonality: Soft regularizer encouraging diverse representations
    """
```

## Configuration

```python
# Lazy generation
SAMPLES_PER_POLE_INITIAL = 60
SAMPLES_PER_POLE_INCREMENT = 60
SAMPLES_PER_POLE_MAX = 300
MIN_F1_THRESHOLD = 0.70

# Training
BEHAVIORAL_RATIO = 0.6  # 60% behavioral, 40% definitional
LAYER_IDX = 12
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
LAMBDA_MARGIN = 0.5
LAMBDA_ORTHO = 1e-4
```

## Usage

```bash
# Train all S-tier simplexes with lazy generation
poetry run python scripts/train_s_tier_tripole_lazy.py

# Results saved to:
results/s_tier_tripole_lazy/run_YYYYMMDD_HHMMSS/
├── training.log
├── results.json
└── {simplex_name}/
    ├── tripole_lens.pt
    └── results.json
```

## Expected Improvements

Compared to the old binary approach:

1. **Correct Architecture**: Joint 3-class softmax preserves shared axis mathematics
2. **Efficient Data Use**: Lazy generation only creates what's needed
3. **Better Performance**: Margin-based hyperplane improves discriminability
4. **Balanced Training**: Equal samples per pole prevents imbalance catastrophe

## Files Modified/Created

### Created
- `scripts/train_s_tier_tripole_lazy.py` - New training script with proper architecture

### Referenced (Not Modified)
- `src/training/tripole_classifier.py` - Joint tripole architecture
- `src/training/sumo_data_generation.py` - Data generation functions
- `data/s_tier_simplex_definitions.json` - Simplex definitions
- `data/simplex_overlap_synsets_enriched.json` - Enriched synsets (2,043 total)

### Superseded
- `scripts/train_s_tier_simplexes.py` - Old approach using binary classifiers (INCORRECT)

## Technical Details

### Why Joint 3-Class vs 3 Binary Classifiers?

**Joint 3-class softmax:**
- Poles share gradients through softmax normalization
- Defines a single shared psychological axis in activation space
- Margin-based hyperplane creates gradiated boundaries
- Mathematical properties proven in whitepaper

**3 separate binary classifiers:**
- Each pole trained independently
- No shared axis
- No gradiated hyperplane
- Loses theoretical properties

### Data Balance Requirement

From `TRIPOLE_TRAINING_SYSTEM.md`:

> Joint tripole training using shared softmax is catastrophically sensitive to class imbalance. All three poles must have equal representation in the training data.

**Why:**
- Softmax normalization couples the poles through gradients
- Imbalanced data causes underrepresented poles to be suppressed
- Common pole gets inflated confidence
- Results in 2.8x performance degradation (measured)

**Solution:**
- Downsample positives to match target `n_samples_per_pole`
- Ensures perfect balance across all three poles
- Critical for lazy generation to work correctly

## Testing

The approach was validated in `scripts/test_tripole_single_simplex.py`:
- Generates balanced 3-class data
- Trains joint tripole lens
- Compares against binary baseline
- Confirms improved performance

## Next Steps

1. Run the new training script:
   ```bash
   poetry run python scripts/train_s_tier_tripole_lazy.py
   ```

2. Monitor for:
   - Balanced sample counts per pole
   - Graduation rates (F1 >= 0.70)
   - Lazy generation efficiency (avg samples used)
   - Comparison to old binary approach

3. Verify enriched data is being used:
   - Should see 100+ samples generated per pole
   - Not the 15+15 failure mode from before

## References

- `docs/TRIPOLE_TRAINING_SYSTEM.md` - Mathematical foundation
- `docs/ENRICHED_DATA_FIX.md` - Data format fix
- `src/training/tripole_classifier.py` - Implementation
