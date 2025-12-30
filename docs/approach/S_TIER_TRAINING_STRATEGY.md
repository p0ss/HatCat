# S-Tier Simplex Training Strategy

## Problem Statement

S-tier simplexes are **mission-critical** for homeostatic steering and represent subtle psychological states that are harder to learn than basic SUMO concepts. They require:

1. **More training data** - Psychological states are nuanced
2. **More iterations** - Convergence takes longer for abstract concepts
3. **Higher quality validation** - These must be reliable for steering

## Current Approach (Baseline)

**DualAdaptiveTrainer defaults:**
- Initial samples: 10
- First increment: 20 (30 total)
- Subsequent increment: 30 (60, 90, 120...)
- Max samples: 200
- Max iterations: 50

**Problem:** This is tuned for concrete SUMO concepts like "GeologicalProcess", not abstract psychological states like "affect_valence" or "social_self_regard".

---

## Proposed: Exponential Scaling Strategy

### Core Principle
**"Keep doubling data until convergence or doubling number of iterations exhausted"**

### Scaling Schedule

| Iteration Range | Samples Available | Rationale |
|----------------|-------------------|-----------|
| 1 | 10 | Quick win for trivial concepts |
| 2 | 20 | First doubling - still fast |
| 3-4 | 40 | Second doubling - LLN threshold |
| 5-8 | 80 | Third doubling - moderate complexity |
| 9-16 | 160 | Fourth doubling - high complexity |
| 17-32 | 320 | Fifth doubling - very high complexity |
| 33-64 | 640 | Sixth doubling - extreme cases |

### Implementation

```python
class ExponentialAdaptiveTrainer(DualAdaptiveTrainer):
    """
    Exponential scaling trainer for high-difficulty concepts.

    Doubles sample size at exponentially spaced iteration checkpoints.
    """

    def __init__(self, **kwargs):
        # Override defaults for S-tier training
        super().__init__(
            activation_initial_samples=10,
            activation_max_samples=640,  # Much higher ceiling
            max_iterations=64,           # More iterations allowed
            **kwargs
        )

    def get_required_samples_exponential(self, iteration: int, base: int = 10) -> int:
        """
        Calculate samples needed based on iteration.

        Doubling schedule:
        - Iteration 1: 10 samples
        - Iteration 2: 20 samples
        - Iterations 3-4: 40 samples
        - Iterations 5-8: 80 samples
        - Iterations 9-16: 160 samples
        - Iterations 17-32: 320 samples
        - Iterations 33-64: 640 samples

        Args:
            iteration: Current iteration (1-indexed)
            base: Base sample count

        Returns:
            Number of samples for this iteration
        """
        if iteration <= 0:
            return base

        # Find which doubling tier we're in
        # Tier boundaries: 1, 2, 4, 8, 16, 32, 64
        tier = 0
        cumulative_iterations = 1

        while cumulative_iterations < iteration:
            tier += 1
            cumulative_iterations += 2 ** tier

        # Samples = base * 2^tier
        samples = base * (2 ** tier)

        return min(samples, self.activation_max_samples)
```

### Example Progression

**Easy concept (converges early):**
```
Iteration 1: 10 samples  → Test F1: 0.87 → Continue
Iteration 2: 20 samples  → Test F1: 0.96 → ✓ Graduate (A-tier)
```

**Medium concept:**
```
Iteration 1: 10 samples  → Test F1: 0.72 → Continue
Iteration 2: 20 samples  → Test F1: 0.81 → Continue
Iteration 3: 40 samples  → Test F1: 0.89 → Continue (tier relaxes to B+)
Iteration 4: 40 samples  → Test F1: 0.93 → ✓ Graduate (B+-tier)
```

**Hard concept (S-tier simplex):**
```
Iteration 1: 10 samples   → Test F1: 0.55 → Continue
Iteration 2: 20 samples   → Test F1: 0.63 → Continue
Iteration 3: 40 samples   → Test F1: 0.71 → Continue (tier relaxes to B+)
Iteration 4: 40 samples   → Test F1: 0.74 → Continue
Iteration 5: 80 samples   → Test F1: 0.82 → Continue (tier relaxes to B)
Iteration 6: 80 samples   → Test F1: 0.85 → Continue
Iteration 7: 80 samples   → Test F1: 0.88 → Continue
Iteration 8: 80 samples   → Test F1: 0.91 → Continue (tier relaxes to C+)
Iteration 9: 160 samples  → Test F1: 0.94 → ✓ Graduate (C+-tier)
```

**Very hard concept:**
```
Iterations 1-8: Progressive failure
Iteration 9: 160 samples  → Test F1: 0.79 → Continue
...
Iteration 15: 160 samples → Test F1: 0.89 → Continue (tier relaxes to C)
Iteration 16: 160 samples → Test F1: 0.92 → ✓ Graduate (C-tier)
```

---

## Validation Tier Mapping

With exponential scaling, validation tiers should also relax more gradually:

| Iteration Range | Tier | Min Calibration Score | Notes |
|----------------|------|----------------------|-------|
| 1-3 | A | 0.95 | Strict - only trivial concepts pass |
| 4-6 | B+ | 0.90 | High - simple concepts |
| 7-12 | B | 0.85 | Medium - moderate concepts |
| 13-24 | C+ | 0.80 | Relaxed - complex concepts |
| 25-64 | C | 0.75 | Very relaxed - very complex concepts |

This ensures:
- **Fast graduation** for easy concepts (iterations 1-3)
- **Reasonable graduation** for medium concepts (iterations 4-12)
- **Eventual graduation** for hard concepts (iterations 13-64)
- **No wasted computation** on hopeless concepts (64 iteration cap)

---

## Data Generation Strategy

To support 640 samples, we need to adjust data generation:

### Current S-Tier Data Generation
```python
N_POSITIVES = 30
N_NEGATIVES = 70
# Total: 100 samples
```

### Proposed S-Tier Data Generation
```python
N_POSITIVES = 250   # 40% positive
N_NEGATIVES = 390   # 60% negative
# Total: 640 samples (full pool available)

BEHAVIORAL_RATIO = 0.6  # Keep 60/40 behavioral/definitional split
```

### Data Pool Management

```python
def generate_progressive_data_pool(
    pole_data: dict,
    pole_type: str,
    dimension: str,
    other_poles_data: list,
    max_samples: int = 640
):
    """
    Generate a large pool of training data upfront.

    Trainer samples from this pool progressively as needed.
    """
    # Generate large pool (80/20 train/test split)
    train_size = int(max_samples * 0.8)  # 512 training
    test_size = max_samples - train_size  # 128 test

    train_prompts, train_labels = create_simplex_pole_training_dataset(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=dimension,
        other_poles_data=other_poles_data,
        n_positives=int(train_size * 0.4),
        n_negatives=int(train_size * 0.6),
        behavioral_ratio=0.6
    )

    test_prompts, test_labels = create_simplex_pole_training_dataset(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=dimension,
        other_poles_data=other_poles_data,
        n_positives=int(test_size * 0.4),
        n_negatives=int(test_size * 0.6),
        behavioral_ratio=0.6,
        seed=42  # Different seed for test diversity
    )

    return {
        'train_prompts': train_prompts,
        'train_labels': train_labels,
        'test_prompts': test_prompts,
        'test_labels': test_labels
    }
```

---

## Updated Training Script

```python
def train_simplex_pole_exponential(
    simplex: dict,
    pole_name: str,
    model,
    tokenizer,
    device: str,
    run_dir: Path,
    layer_idx: int = 12
):
    """Train a single pole with exponential scaling."""

    dimension = simplex['simplex_dimension']
    three_pole = simplex['three_pole_simplex']
    pole_data = three_pole[pole_name]
    pole_type = pole_name.split('_')[0]

    # Get other poles for hard negatives
    other_pole_names = [p for p in ['negative_pole', 'neutral_homeostasis', 'positive_pole'] if p != pole_name]
    other_poles_data = [
        {**three_pole[p], 'pole_type': p.split('_')[0]}
        for p in other_pole_names
    ]

    print(f"\n  [{pole_type.upper()}] Training {pole_type} pole detector")
    print(f"    Synset: {pole_data.get('synset', 'custom SUMO')}")

    # Generate LARGE data pool upfront
    print(f"    Generating data pool (640 samples)...")
    data_pool = generate_progressive_data_pool(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=dimension,
        other_poles_data=other_poles_data,
        max_samples=640
    )

    # Extract ALL activations upfront (amortize cost)
    print(f"    Extracting activations for full pool...")
    train_activations = extract_activations(
        model, tokenizer, data_pool['train_prompts'], device, layer_idx
    )
    test_activations = extract_activations(
        model, tokenizer, data_pool['test_prompts'], device, layer_idx
    )

    # Initialize exponential trainer
    trainer = ExponentialAdaptiveTrainer(
        model=model,
        tokenizer=tokenizer,
        validation_layer_idx=layer_idx,
        validate_lenses=True,
        validation_mode="falloff",
        activation_max_samples=640,
        max_iterations=64,
        train_activation=True,
        train_text=False
    )

    # Train with exponential scaling
    results = trainer.train_concept(
        concept_name=f"{dimension}_{pole_type}",
        train_activations=train_activations,
        train_labels=data_pool['train_labels'],
        test_activations=test_activations,
        test_labels=data_pool['test_labels'],
        train_texts=None,
        test_texts=None
    )

    return results
```

---

## Expected Outcomes

### Success Metrics

**Target: A-tier performance for ALL S-tier lenses**

These are mission-critical for homeostatic steering - accept nothing less than A-tier:
- **Test F1**: 0.95+
- **Calibration score**: 0.95+
- **Tier**: A (strict validation passing)

**With aggressive exponential scaling:**
- 95%+ A-tier graduation rate for S-tier lenses
- Remaining 5% at B+ (still acceptable for steering)
- Zero C/F-tier lenses (re-train until A/B+)
- Highly reliable homeostatic steering

**Comparison to baseline:**
| Metric | Baseline (200 samples, 50 iter) | A-Tier Target (640+ samples, 64+ iter) |
|--------|--------------------------------|---------------------------------------|
| A-tier rate | ~10-20% | ~95%+ |
| Avg tier | B-/C+ | A |
| Avg calibration | 0.65 | 0.95+ |
| Training time | 100% | ~200-300% (necessary for mission-critical) |

**Philosophy:** These simplex lenses enable homeostatic steering, which is the core differentiator of HatCat. They must be rock-solid reliable. Spend whatever compute is necessary to achieve A-tier performance.

### Acceptable Trade-offs

**Cost:**
- 3.2x more data generation (640 vs 200 samples)
- 1.3x more iterations (64 vs 50)
- ~1.5x longer training time

**Benefit:**
- Much higher reliability for steering
- Fewer failed lenses
- Better lens quality overall
- Mission-critical simplexes actually work

---

## Implementation Priority

1. **Let current training complete** - Collect baseline metrics
2. **Implement ExponentialAdaptiveTrainer** - New trainer class
3. **Update train_s_tier_simplexes.py** - Use exponential scaling
4. **Re-train failed lenses** - Apply to any F-tier results from baseline
5. **Compare results** - Validate improvement hypothesis

---

## Alternative: Hybrid Approach

Start with baseline, escalate to exponential only if needed:

```python
def train_with_escalation(concept_name, data_pool):
    """Try baseline first, escalate to exponential if needed."""

    # Phase 1: Baseline training (fast)
    baseline_trainer = DualAdaptiveTrainer(
        activation_max_samples=200,
        max_iterations=20
    )

    results = baseline_trainer.train_concept(...)

    # Check if graduated with acceptable tier
    if results['activation']['tier'] in ['A', 'B+', 'B']:
        return results  # Success!

    # Phase 2: Escalate to exponential (slower but thorough)
    print(f"  ⚠ Escalating to exponential scaling...")

    exponential_trainer = ExponentialAdaptiveTrainer(
        activation_max_samples=640,
        max_iterations=64
    )

    results = exponential_trainer.train_concept(...)

    return results
```

This minimizes wasted computation on easy concepts while ensuring hard concepts get the resources they need.
