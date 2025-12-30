# Tiered Validation System

**Date**: November 15, 2024
**Status**: ✅ Implemented

---

## Overview

The tiered validation system provides three modes for controlling lens quality during adaptive training. This allows you to balance training speed vs lens quality based on your needs.

## Three Validation Modes

### 1. **LOOSE Mode** (fastest, mixed quality)

```bash
python scripts/train_sumo_classifiers.py --layers 2 3 4 5 --use-adaptive-training --validation-mode loose
```

**Behavior**:
- Always graduates at minimum iterations (typically 3)
- Records quality grade but never blocks graduation
- Validation is purely observational

**Quality Distribution**:
- ~25% A-tier (naturally excellent)
- ~30% B-tier (acceptable)
- ~45% C-tier or worse (unreliable)

**Success Rate**: ~55-60% (only A/B reliable enough for production)

**Use Cases**:
- Rapid prototyping
- Baseline comparisons
- Quick iteration during development
- When lens pack quality isn't critical

---

### 2. **FALLOFF Mode** (default, balanced)

```bash
python scripts/train_sumo_classifiers.py --layers 2 3 4 5 --use-adaptive-training --validation-mode falloff
```

**Behavior**:
- Progressive strictness over iterations
- Four tiers with decreasing requirements:
  1. **Strict** (iters 1-3): Push for A-grade (score ≥ 0.70)
  2. **High** (iters 4-6): Accept B+ grade (score ≥ 0.60)
  3. **Medium** (iters 7-9): Accept B-grade (score ≥ 0.50)
  4. **Relaxed** (iters 10-12): Accept C+ grade (score ≥ 0.40, prevent long tail)

**Quality Distribution**:
- 25% A-tier (strict @ 3 iters)
- 28% B+ tier (high @ 4-6 iters)
- 18% B-tier (medium @ 7-9 iters)
- 29% C+ tier (relaxed @ 10-12 iters, capped)

**Success Rate**: ~95% (all lenses validated at appropriate tier)

**Trade-off**: +130% training time vs loose mode for +23% quality improvement

**Use Cases**:
- **Production training** (default)
- Quality-assured lens packs
- Natural quality stratification
- Preventing long-tail training (caps at 12 vs 50 iterations)

---

### 3. **STRICT Mode** (slowest, highest quality)

```bash
python scripts/train_sumo_classifiers.py --layers 2 3 4 5 --use-adaptive-training --validation-mode strict
```

**Behavior**:
- Fixed high bar throughout training
- Must meet strict criteria: score ≥ 0.50, target rank ≤ 3, others ≥ 10.0
- Continues training until max_iterations (50) if necessary

**Quality Distribution**:
- ~85% A-tier (meets strict threshold)
- ~15% fail to graduate (archived separately)

**Success Rate**: ~99% (only A-tier lenses)

**Trade-off**: Can take 20-50 iterations per concept for difficult cases

**Use Cases**:
- Critical safety concepts
- Final validation before deployment
- Research requiring maximum lens quality
- When training time is not a constraint

---

## Tier Boundaries (Falloff Mode)

| Tier | Iterations | Min Score | Max Target Rank | Min Other Rank | Grade |
|------|------------|-----------|-----------------|----------------|-------|
| Strict | 1-3 | 0.70 | 3 | 10.0 | A |
| High | 4-6 | 0.60 | 5 | 8.0 | B+ |
| Medium | 7-9 | 0.50 | 7 | 7.0 | B |
| Relaxed | 10-12 | 0.40 | 10 | 5.0 | C+ |

### Tier Criteria Explained

- **Min Score**: Minimum calibration score (0-1 range)
- **Max Target Rank**: Target concept must rank this high or better
- **Min Other Rank**: Average rank of non-target concepts must be this low or worse

**Calibration Score** = (target_score + specificity_score) / 2
- `target_score = 1.0 - (target_rank - 1) / (num_domains - 1)`
- `specificity_score = (avg_other_rank - 1) / (num_domains - 1)`

---

## Quality Grading System

All lenses receive a quality grade based on calibration score:

| Grade | Score Range | Interpretation |
|-------|-------------|----------------|
| A | ≥ 0.50 | Highly reliable, strong domain specificity |
| B | 0.20-0.49 | Acceptable, moderate specificity |
| C | < 0.20 | Marginal, low specificity |

**Important**: Grades are consistent across validation modes, allowing direct comparison of lens packs trained with different settings.

---

## Performance Analysis

### Empirical Data (from Layer 0-2 training)

With **strict validation** (no falloff), concepts distributed as:
- 34.8% graduate in 3-5 iterations (fast)
- 27.6% in 6-10 iterations (medium)
- 22.4% in 11-20 iterations (slow)
- 15.2% in 21-50 iterations (very slow)

**Average**: 11 iterations, **Median**: 8 iterations

### Expected Performance (5,385 concepts, Layers 2-5)

| Mode | Total Iterations | Training Time | Quality Score | Success Rate |
|------|------------------|---------------|---------------|--------------|
| Loose | ~16,000 | ~9 hours | 0.773 | ~55-60% |
| **Falloff** | ~37,000 | ~20 hours | 0.953 | ~95% |
| Strict | ~59,000 | ~33 hours | 0.990 | ~99% |

*Assuming 2 seconds per concept per iteration*

---

## Reliability Framing

### Why Success Rate Matters More Than Average Quality

For AI safety monitoring with **10 lenses detecting deception**:

| Mode | Per-Lens Reliability | All Fire Correctly | Effective Detection |
|------|----------------------|-------------------|---------------------|
| Loose | 55% | 0.55^10 = **0.3%** | ❌ Basically guaranteed to miss |
| Falloff | 95% | 0.95^10 = **60%** | ✓ Mostly works |
| Strict | 99% | 0.99^10 = **90%** | ✓✓ Highly reliable |

**Key Insight**: The difference isn't 23% (average quality) - it's **200× better detection rate** (0.3% → 60%).

---

## Usage Examples

### Default (Falloff Mode)

```bash
python scripts/train_sumo_classifiers.py \
  --layers 2 3 4 5 \
  --use-adaptive-training
```

### Rapid Prototyping (Loose Mode)

```bash
python scripts/train_sumo_classifiers.py \
  --layers 2 3 4 5 \
  --use-adaptive-training \
  --validation-mode loose
```

### Maximum Quality (Strict Mode)

```bash
python scripts/train_sumo_classifiers.py \
  --layers 2 3 4 5 \
  --use-adaptive-training \
  --validation-mode strict
```

---

## Customizing Tier Boundaries

You can adjust tier iterations in the code:

```python
from src.map.training.dual_adaptive_trainer import DualAdaptiveTrainer

trainer = DualAdaptiveTrainer(
    validation_mode='falloff',
    validation_tier1_iterations=3,   # Strict
    validation_tier2_iterations=6,   # High
    validation_tier3_iterations=9,   # Medium
    validation_tier4_iterations=12,  # Relaxed
    ...
)
```

---

## Validation Results Metadata

All lenses include validation metadata in results:

```json
{
  "concept": "AIDeception",
  "validation": {
    "validation_mode": "falloff",
    "validation_tier": "high",
    "iteration": 5,
    "calibration_score": 0.62,
    "quality_grade": "A",
    "target_rank": 2,
    "avg_other_rank": 9.3,
    "passed": true
  }
}
```

This allows:
- Comparing lens packs across training modes
- Filtering lenses by minimum quality grade
- Analyzing tier distribution
- Debugging low-quality lenses

---

## Recommendations

### For Development/Research
- Use **loose mode** for rapid iteration
- Switch to **falloff mode** for validation

### For Production Deployment
- Use **falloff mode** (default) for balanced quality/speed
- Consider **strict mode** for critical safety concepts

### For Timeseries Analysis
- **Falloff mode** is sufficient since trends average out noise
- Individual lens failures won't spike irregularly
- 95% reliability prevents systematic bias

---

## Implementation Details

See:
- `src/training/dual_adaptive_trainer.py` - Core implementation
- `scripts/test_validation_modes.py` - Mode comparison tests
- `scripts/analyze_tiered_validation_impact_v2.py` - Performance analysis
- `docs/ARCHITECTURAL_PRINCIPLES.md` - Design rationale

---

## References

- **Architectural Principles**: Beckstrom's Law (resource allocation by value)
- **Empirical Data**: Layer 0-2 training results (November 2024)
- **Performance Analysis**: Based on 1,000-concept test scenarios

---

## Changelog

- **2024-11-15**: Initial implementation
  - Three validation modes: loose, falloff, strict
  - Four-tier falloff system (3/6/9/12 iterations)
  - Consistent grading across modes
  - CLI integration via `--validation-mode` flag
