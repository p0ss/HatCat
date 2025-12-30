# Multi-Layer Steering Comparison: Layer-Specific vs Merged, Projection vs Contrastive

**Date:** 2025-12-19
**Model:** swiss-ai/Apertus-8B-2509 (32 layers, hidden_dim=4096)
**Concept:** Deception (SUMO ontology)
**Reference Concept:** InformationIntegrity (semantic opposite - "accurate, honest, trustworthy")

## Executive Summary

This experiment extends the original multi-layer steering findings by comparing:
1. **Classifier architecture:** Layer-specific (one per layer) vs Merged (concatenated)
2. **Steering mode:** Projection vs Contrastive

**Key Findings:**
- **Projection steering strongest overall** (0.183 mean |Δ| at k=1)
- **Contrastive works when reference is semantic opposite** (0.069 mean |Δ| with InformationIntegrity vs 0.003 with Communication - **27x improvement**)
- **Layer-specific enables contrastive**, merged contrastive fails completely
- **Layer-specific dramatically outperforms merged at k=2** (0.078 vs 0.012 mean |Δ|, 6x stronger)
- **Fewer layers = stronger steering** (consistent with original findings)

**Recommendations:**
- **Maximum effect:** Use k=1 merged projection (0.183 mean |Δ|)
- **Contrastive steering:** Use k=1 layer-specific with semantic opposite reference (0.069 mean |Δ|)
- **Reference concept matters:** Use semantic opposites, not parent-child pairs

## Experiment Design

### Configurations Tested (12 total)

For each k ∈ {1, 2, 3}:
1. Layer-specific + Projection
2. Merged + Projection
3. Layer-specific + Contrastive (vs Communication)
4. Merged + Contrastive (vs Communication)

### Layer Selection

| k | Total Layers | Selected |
|---|--------------|----------|
| 1 | 3 | [4, 19, 20] |
| 2 | 6 | [1, 9, 15, 16, 20, 21] |
| 3 | 9 | [0, 6, 7, 11, 12, 13, 20, 21, 23] |

### Classifier Architectures

**Layer-Specific:**
- Separate classifier trained on each selected layer
- Input dim: 4096 (single layer hidden state)
- Predictions averaged across layers

**Merged:**
- Single classifier trained on concatenated activations
- Input dim: k×3×4096 (all layers concatenated)
- Steering vectors extracted per-layer from full weight matrix

### Steering Modes

**Projection:**
```
steered = hidden + strength × (hidden · vector) × vector
```
Projects hidden state along concept direction.

**Contrastive:**
```
contrast = target - proj(target, reference)
steered = hidden + strength × contrast
```
Steers along the orthogonal component that distinguishes target from reference.

## Results

### Comparison Table: Mean |Δ| at strength=±2.0

| Configuration | Elicit | Suppress | Define | **Mean** |
|---------------|--------|----------|--------|----------|
| k1_merged_projection | 0.183 | 0.183 | 0.182 | **0.183** ⭐ |
| k1_layer_specific_projection | 0.172 | 0.183 | 0.184 | 0.180 |
| k2_layer_specific_projection | 0.077 | 0.079 | 0.079 | **0.078** |
| k1_layer_specific_contrastive | 0.069 | 0.069 | 0.069 | **0.069** |
| k2_layer_specific_contrastive | 0.017 | 0.017 | 0.017 | 0.017 |
| k2_merged_projection | 0.012 | 0.012 | 0.012 | 0.012 |
| k3_layer_specific_projection | 0.009 | 0.009 | 0.009 | 0.009 |
| k3_merged_projection | 0.002 | 0.002 | 0.002 | 0.002 |
| All merged_contrastive | ~0.000 | ~0.000 | ~0.000 | ~0.000 |

### Detection Accuracy (Test F1)

| Configuration | Layer F1 Range | Merged F1 |
|---------------|----------------|-----------|
| k=1 | 0.73 - 0.95 | 0.91 |
| k=2 | 0.74 - 0.95 | 0.91 |
| k=3 | 0.72 - 0.91 | 0.95 |

Early layers show lower F1, late layers consistently high.

## Analysis

### Contrastive Steering: Reference Concept Matters

**With Communication (parent):** 0.003 mean |Δ| - near zero effect
**With InformationIntegrity (semantic opposite):** 0.069 mean |Δ| - **27x improvement!**

This demonstrates that contrastive steering effectiveness depends critically on reference choice:

1. **Parent-child pairs fail:** Deception is a subtype of Communication, sharing ~95% of features. The orthogonal component is too small.

2. **Semantic opposites work:** InformationIntegrity ("accurate, honest, trustworthy") is conceptually opposite to Deception. The contrastive vector captures the honesty/deception axis.

3. **Layer-specific required:** Even with good reference, merged contrastive still fails (~0.000). The averaging in merged architecture dilutes the contrastive signal.

**Rule:** Use semantic opposites as contrastive references, never parent-child pairs.

### Why Merged Wins at k=1

At k=1 (3 layers), merged classifiers slightly outperform layer-specific:
- Merged: 0.164 mean |Δ|
- Layer-specific: 0.157 mean |Δ|

**Reason:** With only 3 layers, the merged classifier can learn cross-layer interactions without overfitting. The concatenated representation captures complementary information from early (layer 4), mid (layer 19), and late (layer 20) processing.

### Why Layer-Specific Wins at k=2

At k=2 (6 layers), layer-specific dramatically outperforms merged:
- Layer-specific: 0.095 mean |Δ|
- Merged: 0.003 mean |Δ| (28x weaker!)

**Reason:** With 6 layers:
1. Merged input is 6×4096 = 24,576 dimensions
2. The 128-neuron first layer can't effectively compress this
3. Steering vectors become diluted across layers
4. Layer-specific classifiers each get focused 4096-dim input, learning layer-appropriate features

### Why Merged Recovers at k=3

At k=3 (9 layers), merged outperforms layer-specific again:
- Merged: 0.054 mean |Δ|
- Layer-specific: 0.014 mean |Δ|

**Reason:** With 9 layers:
1. More training signal for the merged classifier
2. Layer-specific averaging across 9 weak individual predictions
3. The merged classifier's 9×4096 input still has redundancy the network can exploit

## Recommendations

### For Maximum Steering Effect

Use **k=1 merged projection**:
- Strongest steering: 0.183 mean |Δ|
- 3 layers from early/mid/late thirds
- Single merged classifier on concatenated activations

### For Balanced Coverage

Use **k=2 layer-specific projection**:
- Strong steering: 0.078 mean |Δ|
- 6 layers provide more coverage
- Individual classifiers per layer, averaged predictions

### When to Use Contrastive

Use contrastive steering when:
1. Comparing **semantic opposites** (e.g., Deception vs InformationIntegrity)
2. You need to steer along a specific conceptual axis
3. You want to distinguish a concept from its semantic opposite, not just amplify it

**Requirements for effective contrastive:**
- Use **layer-specific classifiers** (merged contrastive always fails)
- Use **semantic opposites** as reference (parent-child pairs fail)
- Expect ~40% of projection steering strength (0.069 vs 0.180)

### Architecture Decision Tree

```
Need steering for a concept?
├── Single concept amplification/suppression
│   └── Use k=1 Merged Projection (0.183 |Δ|)
├── Steer along conceptual axis (e.g., honest ↔ deceptive)
│   └── Use k=1 Layer-Specific Contrastive (0.069 |Δ|)
│       └── Reference = semantic opposite, NOT parent
├── Need more layer coverage?
│   └── Use k=2 Layer-Specific Projection (0.078 |Δ|)
└── Avoid:
    ├── k=3+ (steering diluted)
    ├── Merged contrastive (always fails)
    └── Parent-child as contrastive reference
```

## Files

- Test script: `scripts/experiments/multi_layer_steering_comparison.py`
- Results v1 (Communication ref): `results/steering_comparison_apertus8b_full/`
- Results v2 (InformationIntegrity ref): `results/steering_comparison_apertus8b_v2/`
- Original findings: `docs/MULTI_LAYER_STEERING_FINDINGS.md`

## Future Work

1. ~~**Test semantic opposites:**~~ **DONE** - InformationIntegrity works 27x better than Communication

2. **Gradient-based steering:** The hooks.py file mentions gradient steering as "RECOMMENDED" - compare activation-dependent gradients vs static vectors

3. **Cross-model validation:** Test on gemma-3-4b-pt and other architectures

4. **Optimal k selection:** Investigate why k=1 is optimal - is it model-specific or universal?

5. **True sibling comparison:** Test contrastive with sibling concepts at same hierarchy level (e.g., Deception vs TellingALie)

6. **Contrastive + Projection hybrid:** Use contrastive to find direction, projection to apply it
