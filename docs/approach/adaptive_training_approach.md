# Adaptive Training Approach

## Overview

HatCat uses an adaptive training system that dynamically adjusts sample sizes based on concept difficulty. Instead of using a fixed number of training samples for all concepts, the system starts with minimal data and incrementally adds more samples only when needed, balancing training efficiency with lens quality.

## Core Principles

### 1. **Start Small, Scale As Needed**
- Begin with minimal samples (10: 5 positive + 5 negative)
- Only generate additional data when lenses fail to graduate
- Easy concepts graduate quickly with minimal computational cost
- Difficult concepts receive more training data automatically

### 2. **Independent Graduation**
Each lens type (activation and text) graduates independently when it achieves:
- Target F1 score (≥0.95 for activation lenses)
- Low overfitting (train-test gap ≤10%)
- Minimum iteration stability (≥3 iterations)

### 3. **Tiered Validation with FALLOFF Mode**
Validation requirements relax progressively across cycles:

| Cycle | Samples | Validation Tier | Min Grade | Min Score |
|-------|---------|----------------|-----------|-----------|
| 0 | 10 (5+5) | STRICT | A | 0.70 |
| 1 | 30 (15+15) | HIGH | B+ | 0.60 |
| 2 | 60 (30+30) | MEDIUM | B | 0.50 |
| 3 | 90 (45+45) | RELAXED | C+ | 0.40 |
| 4 | 120 (60+60) | RELAXED | C+ | 0.40 |

**Validation grades** are based on lens calibration:
- **A-grade**: target_rank ≤ 10, others ≥ 5.0, score ≥ 0.40
- **B-grade**: target_rank ≤ 15, others ≥ 3.0, score ≥ 0.25
- **C-grade**: target_rank ≤ 20, others ≥ 1.5, score ≥ 0.15
- **F-grade**: fails all above thresholds

## Training Cycle Flow

```
┌─────────────────────────────────────────────┐
│ Cycle 0: Generate 10 samples (5+5)          │
│ ↓                                            │
│ Train lens on 10 samples                    │
│ ↓                                            │
│ Test F1 ≥ 0.95 & gap ≤ 0.10 & iter ≥ 3?    │
│ ├─ Yes → Validate calibration               │
│ │         ├─ Grade A? → ✓ GRADUATE          │
│ │         └─ Grade < A? → Request more data │
│ └─ No → Continue training                    │
│          ↓                                   │
│          After 3 iterations without grad?    │
│          └─ Yes → Cycle 1 (add 20 more)     │
└─────────────────────────────────────────────┘

Similar flow continues through Cycles 1-4, with
relaxed validation requirements at each tier.

At Cycle 4 (max), accept best result even if
not fully graduated.
```

## Stuck Detection & Auto-Escalation

The system detects when a lens is "stuck" (unable to graduate after 3 iterations within a cycle) and automatically requests more training data:

```python
stuck_without_data = cycle_iterations >= 3 and not graduated

if stuck_without_data:
    if validation_cycle < 4:
        validation_cycle += 1  # Request more data
        cycle_iterations = 0    # Reset iteration counter
    else:
        # At max cycles, accept best result
        graduated = True
```

## Sample Generation Strategy

### Incremental Generation
Samples are generated on-demand per cycle, not pre-generated:

```python
# Cycle 0: 10 samples (5+5)
# Cycle 1: +20 samples (10+10) → total 30
# Cycle 2: +30 samples (15+15) → total 60
# Cycle 3: +30 samples (15+15) → total 90
# Cycle 4: +30 samples (15+15) → total 120
```

### Accumulation
All generated samples are accumulated and used in subsequent iterations:
- Iteration 1 of Cycle 1 trains on all 30 samples (original 10 + new 20)
- Iteration 2 of Cycle 1 trains on the same 30 samples
- Network converges to stable solution on accumulated data

## Performance Characteristics

### Efficiency Gains
- **Easy concepts**: Graduate in ~13s with just 10 samples
- **Medium concepts**: Graduate in ~20s with 30 samples
- **Hard concepts**: Graduate in ~30-40s with 60-120 samples
- **Average**: ~4.3 seconds per concept

### Data Efficiency
Instead of generating 200+ samples for every concept:
- ~30% of concepts graduate with 10 samples
- ~50% graduate with 30 samples
- ~15% need 60 samples
- ~5% require 90-120 samples

This results in **~70% reduction** in sample generation time.

## Validation Modes

The system supports multiple validation modes:

### FALLOFF (Recommended)
Progressive relaxation across cycles, balancing quality with training speed:
- Cycle 0 requires A-grade (strict)
- Cycle 1 accepts B+ (high quality)
- Cycle 2 accepts B (medium quality)
- Cycle 3-4 accept C+ (relaxed, but still functional)

### LOOSE
Single relaxed tier for all cycles (faster, lower quality):
- All cycles accept C+ grade
- Useful for rapid prototyping or less critical concepts

### STRICT (Legacy)
Single strict tier for all cycles (slower, highest quality):
- All cycles require A-grade
- Useful for critical safety concepts

## WordNet Patch Integration

For concepts with sparse or missing WordNet synsets, the system supports manual synset patches:

1. **Identify sparse concepts**: Concepts with no synsets or children
2. **Suggest synsets**: Use LLM to suggest relevant WordNet synsets (96.4% accuracy)
3. **Validate synsets**: Verify all synsets exist in WordNet 3.0
4. **Apply patch**: Add synsets to layer files with metadata tracking
5. **Train normally**: Patched concepts train with newly available data

**Results**: All patched concepts graduate successfully (100% success rate so far), with 75% achieving B-tier and 25% achieving A-tier validation grades.

## Code Location

- **Trainer**: `src/training/dual_adaptive_trainer.py`
- **Main training script**: `src/training/sumo_classifiers.py`
- **Data generation**: `src/training/sumo_data_generation.py`
- **Validation**: `src/training/lens_validation.py`

## Configuration Parameters

### Adaptive Training
```python
DualAdaptiveTrainer(
    max_iterations=50,                    # Max iterations per cycle

    # Activation lens config
    activation_initial_samples=10,        # Cycle 0 samples (5+5)
    activation_first_increment=20,        # Cycle 1 increment (10+10)
    activation_subsequent_increment=30,   # Cycle 2+ increment (15+15)
    activation_max_samples=200,           # Hard limit
    activation_target_accuracy=0.95,      # Graduation threshold

    # Text lens config (similar structure)
    train_text_lenses=False,              # Usually disabled

    # Validation
    validation_mode='falloff',            # or 'loose', 'strict'
)
```

## Future Improvements

1. **Dynamic increment sizing**: Adjust increment size based on improvement rate
2. **Early stopping**: Detect plateau in performance and skip remaining iterations
3. **Concept difficulty prediction**: Pre-classify concepts as easy/medium/hard
4. **Active learning**: Intelligently select which samples to generate next
5. **Multi-layer coordination**: Share knowledge between related concepts across layers
