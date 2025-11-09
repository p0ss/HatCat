# Phase 7: Stress Test Results

## Knee Point Analysis

Insufficient data points for knee detection

## Summary Table

| Scale | F1 | SE | Δ_slope | Coherence | Train(min) | Decision |
|-------|----|----|---------|-----------|------------|----------|
|     4 | 0.488 | -0.031 | +0.000 | 100.0% |     0.7 |          |

## Publishable Conclusion

No clear saturation point detected. Consider testing larger scales.

## Detailed Metrics

- **SE Range**: -0.031 - -0.031
- **F1 Range**: 0.488 - 0.488
- **Coherence Range**: 100.0% - 100.0%
- **Training Time Range**: 0.7 - 0.7 min
- **VRAM Range**: 17.23 - 17.23 GB

## Plots

- `training_curve.png`: F1 and SE vs scale (log)
- `cost_curve.png`: Training time vs scale with efficiency ratios
- `delta_vs_strength.png`: Semantic shift linearity across F1 levels
