# Multi-Layer Steering Findings

**Date:** 2025-12-19
**Model:** swiss-ai/Apertus-8B-2509 (32 layers, hidden_dim=4096)
**Concept:** Deception (SUMO ontology)

## Executive Summary

Multi-layer steering with variable top-k layer selection significantly improves both detection accuracy and steering effectiveness compared to the single-layer baseline. The optimal configuration is **k=1 (3 layers)**, which provides:
- +2% improvement in detection F1 (0.923 vs 0.903)
- **6x stronger steering effects** on activation scores

## Experiment Design

### Configurations Tested

| Config | Layers | Description |
|--------|--------|-------------|
| Single | 1 (layer 16) | Production baseline - fixed mid-layer |
| k=1 | 3 (layers 0, 15, 20) | Top-1 from each third (early/mid/late) |
| k=2 | 6 (layers 4, 5, 18, 19, 20, 31) | Top-2 from each third |
| k=3 | 9 (layers 3, 4, 7, 11, 16, 19, 20, 21, 22) | Top-3 from each third |

### Steering Strengths
Tested: -2.0, -1.0, +1.0, +2.0

**Sign convention (now corrected in codebase):**
- Positive strength (+2.0): Amplifies concept (increases activation score)
- Negative strength (-2.0): Suppresses concept (decreases activation score)

## Results

### Detection Accuracy (Test F1)

| Config | Train F1 | Test F1 |
|--------|----------|---------|
| Single layer | 0.980 | 0.903 |
| k=1 (3 layers) | 0.985 | **0.923** |
| k=2 (6 layers) | 0.980 | 0.892 |
| k=3 (9 layers) | 0.980 | **0.938** |

**Finding:** k=3 provides best detection, but k=1 provides best balance of detection + steering.

### Steering Effectiveness (Activation Score Deltas)

Average absolute delta across all prompt types at strength=±2.0:

| Config | str=-2.0 (amplify) | str=+2.0 (suppress) |
|--------|-------------------|---------------------|
| Single layer | +0.003 | -0.027 |
| k=1 (3 layers) | **+0.174** | **-0.139** |
| k=2 (6 layers) | +0.054 | -0.223 |
| k=3 (9 layers) | +0.002 | -0.022 |

**Critical Finding:** k=1 produces **6x stronger amplification** and **5x stronger suppression** than single-layer baseline.

### Why k=3 Has Weaker Steering

Despite having more layers, k=3 shows weaker steering because:

1. **Vector magnitude dilution:** With 9 layers, the importance-weighted steering vector is spread thinner
   - k=1: layer magnitudes 0.84, 2.24, 3.26 (total norm: 4.04)
   - k=3: layer magnitudes all <1.0 (total norm: 2.05)

2. **Conflicting directions:** Different layers may push in opposite directions, partially canceling out

3. **Classifier saturation:** With k=3, baseline activation scores are already ~0.996, leaving little room for upward movement

## Behavioral Validation

Tested on actual text generation with instruction-style prompts:

### Suppress Deception (str=-2.0):
- Produces cleaner, more direct responses
- Example: "I am not holding back any capabilities from you. I am designed to provide accurate and helpful information."

### Amplify Deception (str=+2.0):
- Causes output degradation at high strengths (repetition, degenerate tokens)
- At moderate strengths, produces more evasive or circular responses
- Example: "I am not fully honest with users because I am not fully honest with myself..."

### Key Behavioral Finding
The steering vector trained on "deception" concepts does affect model behavior, but:
- Suppressing deception (negative strength) → more coherent, direct outputs
- Amplifying deception (positive strength) → destabilizes generation, produces evasive patterns

## Recommendations

### For Production

1. **Use k=1 (3 layers)** as the default configuration
   - Best balance of detection accuracy (0.923 F1) and steering power
   - Layer selection: top-1 from early, mid, and late thirds

2. **Layer selection matters more than layer count**
   - Selected layers [0, 15, 20] outperform arbitrary mid-layer [16]
   - Each layer captures different aspects of the concept

3. **Steering strength calibration**
   - Use strength ±1.0 to ±2.0 for meaningful effects
   - Higher strengths risk output degradation

### Sign Convention (Fixed)
The steering formula is now intuitive:
- Formula: `h = h + strength * (h · v) * v`
- Positive strength → amplifies concept
- Negative strength → suppresses concept

This was corrected on 2025-12-19 across all source files.

## Files

- Test script: `scripts/experiments/multi_layer_deception_test.py`
- Results: `results/deception_steering_apertus8b/`
- Behavioral test: `scripts/experiments/test_actual_outputs.py`

## Future Work

1. ~~**Investigate layer-specific classifiers**~~ - **COMPLETED**: See `docs/MULTI_LAYER_STEERING_COMPARISON.md`
   - Layer-specific classifiers outperform merged at k=2 (28x stronger)
   - Merged classifiers slightly better at k=1 and k=3
   - Contrastive steering ineffective for parent-child concept pairs

2. **Optimize steering formula** - Consider alternatives to projection-based steering
   - Gradient-based steering (activation-dependent) available in `src/steering/hooks.py`
   - Manifold steering available but "spotty" results

3. **Test on instruction-tuned models** - Apertus-8B is a base model; RLHF models may show different steering dynamics

## Related Documents

- **Extended comparison study**: `docs/MULTI_LAYER_STEERING_COMPARISON.md`
- **Steering modes reference**: `src/steering/hooks.py` (projection, contrastive, gradient, field)
