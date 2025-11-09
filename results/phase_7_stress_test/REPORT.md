# Phase 7: Stress Test Results

## Knee Point Analysis

SE gain -0.046 < 0.02 at 2.0× scale increase (8 → 16 samples)

## Summary Table

| Scale | F1 | SE | Δ_slope | Coherence | Train(min) | Decision |
|-------|----|----|---------|-----------|------------|----------|
|     2 | 0.487 | -0.033 | -0.000 | 90.5% |     0.2 |          |
|     4 | 0.464 | -0.090 | -0.000 | 90.5% |     0.3 |          |
|     8 | 0.475 | -0.062 | +0.000 | 100.0% |     0.5 | **KNEE** |
|    16 | 0.457 | -0.108 | +0.000 | 100.0% |     1.0 |          |
|    32 | 0.522 | 0.054 | +0.000 | 100.0% |     2.0 |          |
|    64 | 0.510 | 0.025 | +0.000 | 90.5% |     3.9 |          |

## Publishable Conclusion

Beyond F1 ≈ 0.48, steering quality saturates;
training beyond 4×4×5
increases cost 2.0× for -4.6% semantic gain.

**Recommendation**: Use 8 samples/concept for optimal cost-effectiveness.

## Detailed Metrics

- **SE Range**: -0.108 - 0.054
- **F1 Range**: 0.457 - 0.522
- **Coherence Range**: 90.5% - 100.0%
- **Training Time Range**: 0.2 - 3.9 min
- **VRAM Range**: 17.23 - 17.23 GB

## Plots

- `training_curve.png`: F1 and SE vs scale (log)
- `cost_curve.png`: Training time vs scale with efficiency ratios
- `delta_vs_strength.png`: Semantic shift linearity across F1 levels
