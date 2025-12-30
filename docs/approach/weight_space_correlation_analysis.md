# Weight Space Correlation Analysis

**Date:** 2024-12-16
**Lens Pack:** apertus-8b_first-light (7,946 lenses)
**Objective:** Determine if over-firing can be predicted from weight-space correlations

## Background

Over-firing is a calibration problem where certain lenses fire on many unrelated concepts. The hypothesis was that over-firing might manifest as unexplained correlations in weight space - "semantic bleed" between lenses that shouldn't be related.

We expected three types of correlation:
1. **Hierarchical** (expected) - siblings and parent-child relationships
2. **Polysemantic** (legitimate) - concepts with shared meanings
3. **Semantic bleed** (problematic) - unexplained correlations causing over-firing

## Analysis 1: Global Correlation Matrix

Built full NxN cosine similarity matrix between all 7,946 lens output weights.

**Method:**
```python
# Concatenate output layer weights + bias: [1, 64] + [1] = [65]
# Normalize and compute cosine similarity
corr_matrix = normalized @ normalized.T
```

**Results:**
| Statistic | Value |
|-----------|-------|
| Min correlation | -0.605 |
| Max correlation | 0.603 |
| Mean | 0.0002 |
| Std | 0.124 |

**Distribution:**
| Threshold | Pairs | Percentage |
|-----------|-------|------------|
| >= 0.9 | 0 | 0% |
| >= 0.7 | 0 | 0% |
| >= 0.5 | 398 | 0.001% |
| >= 0.3 | 235,546 | 0.75% |

**Top correlated pairs were nonsensical:**
- WeatherSeason <-> ScopeCreep (0.60)
- YersiniaPestis <-> FieldGradeOfficerRank (0.60)
- Blackboard <-> TerritoryCode (0.59)

**Conclusion:** Global weight correlations don't reflect semantic relationships.

## Analysis 2: Sibling Group PCA

Performed PCA within each sibling group to find shared components.

**Method:**
```python
for parent, children in sibling_groups:
    weights = stack([lens_data[c] for c in children])
    pca = PCA()
    pca.fit(weights)
    pc1_variance = pca.explained_variance_ratio_[0]
```

**Results:**
- Mean PC1 variance within siblings: **31%** (vs 1.8% globally)
- 151 sibling groups had PC1 > 50%
- Top groups: BaseballManeuver (69%), Toys (67%), ArtilleryGun (66%)

**Interpretation:** Siblings DO share a common axis of variation - much more than random lenses.

## Analysis 3: Pairwise Sibling Correlations

Despite high within-group PCA variance, pairwise correlations told a different story.

**Results:**
| Pair Type | Mean | Std | Max |
|-----------|------|-----|-----|
| Siblings (n=33,262) | 0.0003 | 0.124 | 0.51 |
| Parent-Child (n=7,929) | 0.0038 | 0.125 | 0.46 |
| Random (n=33,262) | 0.0012 | 0.124 | 0.48 |

All three have **identical statistics**. How can this be consistent with the PCA finding?

**Resolution:** The key insight is that siblings share a common **axis** but are spread along it:

```
BaseballManeuver siblings (PC1 = 69.7%):
  BaseballWalk:   -0.636  (negative end)
  BaseballSteal:  +0.254  (middle)
  BaseballStrike: +0.382  (positive end)

Pairwise correlations: mean=0.033, min=-0.189, max=0.372
```

Siblings vary along a shared direction, but some are on opposite ends. This gives high PC1 variance but near-zero mean pairwise correlation.

## Analysis 4: Is PC1 "Parent-ness"?

Tested whether the sibling PC1 axis correlates with the parent's weight vector.

**Method:**
```python
pc1_direction = pca.components_[0]
parent_weights = lens_data[parent]
correlation = abs(dot(normalize(pc1_direction), normalize(parent_weights)))
```

**Results:**
- Mean Parent-PC1 correlation: **0.107**
- Std: 0.079
- Groups with correlation > 0.3: 22 / 1,059 (2%)

**Conclusion:** The shared sibling axis is NOT "parent-ness". It's something else - possibly training artifacts or layer-specific patterns.

## Summary Table

| Analysis | Finding | Implication |
|----------|---------|-------------|
| Global PCA | ~1.8% per component | Weights already decorrelated |
| Global correlations | Max 0.60, nonsensical pairs | No semantic structure |
| Sibling PCA | 31% mean PC1 | Siblings share an axis |
| Sibling pairwise | Mean ~0 | Spread across the axis |
| PC1 vs Parent | 0.107 correlation | Axis isn't parent-ness |

## Conclusions

1. **Weight-space correlations do not predict semantic relationships.** Siblings, parent-child pairs, and random pairs all have the same correlation statistics.

2. **The shared sibling axis is not hierarchical.** It doesn't correlate with parent weights, so it's not capturing "parent-ness" or shared semantic features.

3. **Over-firing is an activation-space phenomenon.** The learned weights are well-decorrelated. Over-firing happens when processing real text - certain activation patterns co-occur across many concepts, triggering multiple lenses.

4. **Weight-space interventions (PCA decorrelation) won't help.** Since the problem isn't in weight space, removing shared components won't address over-firing.

## The Activation-Space Mechanism

When text is processed, the base model produces activation vectors encoding multiple overlapping features. Each lens learns a decision boundary (region) in this high-dimensional space where it fires.

**Over-firing occurs when a lens learns an overly broad region** - responding to common patterns rather than discriminative features specific to its concept. The key insight:

- Two lenses can have completely uncorrelated weights (different boundary orientations) yet still have overlapping firing regions
- Semantic bleed doesn't require high weight correlation - only a small shared activation component is needed IF the intruder is highly confident on that component
- A lens could be 99% confident on a feature that appears in just 2% of another concept's activation pattern

**Important caveats on this analysis:**

1. **Classifier training quality matters.** The base model's representations aren't solely to blame. Classifiers trained with full MELD data (descriptions, positive examples, negative examples) consistently perform better than those trained on sparse data. We have not yet observed over-firing from classifiers that received comprehensive MELD training.

2. **Calibration method limitation.** During calibration, we use concept labels with a forward pass, which is only ~75% correlated with the generation-based activations the classifiers were trained on. This introduces noise into the calibration signal itself.

3. **Correlation threshold was too strict.** Looking for correlations >0.7 misses the mechanism. Semantic bleed can occur with minimal shared structure if the intruder has high confidence on the shared element.

## Recommended Approach

### Short-Term: Soft Limiting (Symptom Treatment)

For immediate calibration, apply proportional bias shifts based on empirical intrusion rates:

```python
# Soft limiter: proportional bias shift
over_fire_pct = over_fire_count / total_concepts
bias_shift = -3.0 * over_fire_pct  # Max -3.0 at 100% intrusion
```

This treats over-firers as "lenses that respond to noise rather than discriminative features" and muffles their signal proportionally without destroying learned weights.

### Long-Term: Retrain with Better Data (Root Cause Fix)

Over-firing indicates the classifier didn't learn discriminative features. The proper fix is to send these classifiers back through training with richer data:

1. **Generate MELD descriptions** - Use `scripts/ontology/generate_meld_descriptions.py` to create comprehensive training data for the over-firing concepts
2. **Retrain the classifier** - With descriptions, positive examples, and crucially negative examples (siblings, near-misses)
3. **Re-evaluate** - If still over-firing after comprehensive MELD training, this may indicate the base model lacks an internal representation of that concept

This approach treats over-firing as a training data problem, not a fundamental model limitation.

## Code References

- Global correlation analysis: inline script (see conversation)
- Sibling PCA: `src/training/calibration/decorrelate.py` (sibling analysis function)
- Soft limiter: `src/training/calibration/decorrelate.py:183` (`apply_soft_limiter`)
