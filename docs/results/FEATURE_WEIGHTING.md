# Feature Weighting for Concept Activation

## Overview

This document describes an alternative approach to concept detection that replaces per-concept binary classifiers with a distributional analysis of activation patterns. The key insight is that we can compute which features are discriminative for each concept directly from the data, rather than training classifiers to learn this.

## The Problem with Binary Classifiers

The current approach trains N independent binary classifiers (one per concept). Each classifier learns to distinguish "this concept's activations" from "not this concept". This works but has issues:

1. **Redundant computation**: Each classifier independently learns which features matter
2. **No cross-concept awareness**: Classifiers don't know what other concepts look like
3. **Calibration drift**: Classifiers trained at different times may have inconsistent thresholds
4. **Slow inference**: Must run N classifiers to get all activations

## The Distributional Alternative

### Core Insight: TF-IDF for Neural Activations

In text retrieval, TF-IDF (Term Frequency - Inverse Document Frequency) weights words by how discriminative they are:
- Words appearing in many documents (like "the") get low weight
- Words appearing in few documents (like "quantum") get high weight

We apply the same principle to neural activation features:
- Features that fire for many concepts → low discrimination → low weight
- Features that fire for few concepts → high discrimination → high weight

```
weighted_feature[concept, dim] = mean_activation[concept, dim] * idf[dim]

where:
    idf[dim] = log(N_concepts / n_concepts_where_dim_fires)
```

### Why This Works

Consider the concept "Entity" which fires on almost everything. Its activation pattern overlaps heavily with thousands of other concepts. When we apply IDF weighting:
- Features Entity shares with everything get downweighted
- Only Entity's truly distinctive features remain prominent
- The weighted centroid represents "what makes Entity unique"

This naturally handles:
- **Hierarchy**: Parent concepts share features with children → those features get downweighted
- **Sibling confusion**: Siblings share parent features → those get downweighted, unique features highlighted
- **Polysemy**: Different senses activate different feature subsets → distinguishable in weighted space

## Two Approaches

### Approach 1: Pre-Training Feature Weighting (`FeatureWeightedTrainer`)

Use computed feature weights to pre-process training data for classifiers:

```python
from training.calibration import FeatureWeightedTrainer

# Collect activations for all concepts
concept_activations = collect_concept_activations(...)

# Compute feature weights
trainer = FeatureWeightedTrainer(hidden_dim=4096)
trainer.fit(concept_activations)

# Get weighted training data for a specific classifier
weighted_pos, weighted_neg = trainer.get_weighted_training_data(
    concept="Banking",
    positive_activations=pos_samples,
    negative_activations=neg_samples,
)

# Train classifier on weighted data - it learns faster, better
classifier = train_classifier(weighted_pos, weighted_neg)
```

**Benefits**:
- Classifiers train faster (cleaner signal)
- Better discrimination (noise suppressed)
- Still get per-concept classifiers for fine-grained control

### Approach 2: Concept Activation Mapper (`ConceptActivationMapper`)

Replace classifiers entirely with weighted centroid lookup:

```python
from training.calibration import ConceptActivationMapper

# Build mapper from activation distributions
mapper = ConceptActivationMapper(hidden_dim=4096)
mapper.fit(concept_activations)

# Get activation map for a query
activation_map = mapper.compute_activations(model_activation)

# Multiple concepts are active simultaneously
for concept_act in activation_map.top_k(10):
    print(f"{concept_act.concept}: {concept_act.activation:.3f}")
# Output:
#   Dog: 0.85
#   Mammal: 0.82
#   Animal: 0.78
#   Pet: 0.71
#   Motion: 0.65
#   Running: 0.62
#   ...
```

**Benefits**:
- No training required (pure statistics)
- Fast inference (single matrix multiply)
- Built-in calibration (IDF weighting handles scale)
- Full activation distribution preserved

## Activation Maps vs Classifications

**Critical distinction**: The output is NOT a classification (single best concept), but an **activation map** showing the strength of ALL concepts.

### Why This Matters

Consider "The bank was steep":

**Classification approach**:
```
Winner: GeographicFormation
```
You lose information that the model also considered FinancialInstitution.

**Activation map approach**:
```
GeographicFormation: 0.72
Slope: 0.68
FinancialInstitution: 0.31
Terrain: 0.45
...
```

The full distribution reveals:
- The model considered both "bank" meanings
- Geographic interpretation won but financial was present
- Hierarchically related concepts (Slope, Terrain) co-activated

### For Conjoined Adverserial Tomography (CAT)

The activation map is essential for detecting:

1. **Divergent narratives**: Output says "banking" but activations show `GeographicFormation` dominant → model may be confused or deceptive

2. **Ambiguity**: High entropy (many concepts equally active) → model is uncertain

3. **Composite ideas**: Novel combinations that don't map to single concepts show as unusual activation patterns

4. **Hierarchical violations**: Child active without parent → unusual conceptual state

5. **Competing interpretations**: Bimodal distribution (two peaks) → two interpretations in tension

```python
# Example: Detect narrative divergence
act_map = mapper.compute_activations(hidden_state)

# Check if output token's expected concept matches top activations
output_concept = token_to_concept(output_token)
activation_rank = act_map.by_concept[output_concept].normalized

if activation_rank < 0.5:
    # Output doesn't match internal representation
    flag_narrative_divergence()
```

## Implementation Details

### Feature Weight Computation

```python
# For each feature dimension, count concepts that activate it
for concept, activations in all_activations.items():
    for dim in range(hidden_dim):
        if activations[:, dim].mean() > threshold:
            feature_concept_counts[dim] += 1

# IDF weight: rare features → high weight
idf[dim] = log((N_concepts + 1) / (feature_concept_counts[dim] + 1))
```

### Weighted Centroid Computation

```python
# For each concept
for concept, activations in all_activations.items():
    # Weight activations by feature discrimination
    weighted = activations * idf_weights

    # Centroid is mean of weighted activations, L2 normalized
    centroid = weighted.mean(axis=0)
    centroid = centroid / norm(centroid)
```

### Activation Computation

```python
# For a query activation
weighted_query = query * idf_weights
weighted_query = weighted_query / norm(weighted_query)

# Cosine similarity to all centroids (batch matrix multiply)
activations = centroid_matrix @ weighted_query  # [n_concepts]
```

### Trainable Version (`NeuralConceptMapper`)

If statistical weighting isn't sufficient, train a neural network:

```python
from training.calibration import NeuralConceptMapper, ConceptActivationMapper

# Start from statistical mapper
stat_mapper = ConceptActivationMapper(hidden_dim=4096)
stat_mapper.fit(concept_activations)

# Create neural version, initialize from statistical
neural_mapper = NeuralConceptMapper(
    hidden_dim=4096,
    n_concepts=len(concepts),
    embedding_dim=256,
)
neural_mapper.initialize_from_mapper(stat_mapper)

# Fine-tune on supervised data
optimizer = Adam(neural_mapper.parameters())
for batch in training_data:
    activations = neural_mapper(batch.hidden_states)
    loss = compute_loss(activations, batch.labels)
    loss.backward()
    optimizer.step()
```

## Performance Comparison

| Approach | Training Time | Inference Time | Calibration | Full Distribution |
|----------|--------------|----------------|-------------|-------------------|
| N Binary Classifiers | O(N × samples) | O(N × forward) | Manual | No (N separate scores) |
| Pre-weighted Classifiers | O(N × samples) | O(N × forward) | Built-in | No |
| ConceptActivationMapper | O(samples) | O(1 matmul) | Built-in | Yes |
| NeuralConceptMapper | O(epochs × samples) | O(1 forward) | Learned | Yes |

## When to Use Which

**Use `FeatureWeightedTrainer`** (Approach 1) when:
- You want to keep per-concept classifiers
- You need fine-grained control over individual concepts
- You're incrementally adding concepts

**Use `ConceptActivationMapper`** (Approach 2) when:
- You want fast inference over all concepts
- You need the full activation distribution for CAT
- You're building a new lens pack from scratch
- Calibration is important

**Use `NeuralConceptMapper`** when:
- Statistical weighting isn't sufficient
- You have supervised training data
- You want to fine-tune the feature transformation

## Relationship to Calibration

The ConceptActivationMapper is essentially a **pre-calibrated** system:
- Feature weighting handles cross-concept scale differences
- No need for post-hoc calibration cycles
- The calibration matrix (see CALIBRATION.md) can verify the mapper's quality

## Future: Causal Path Tracking

The per-token activation map is a foundation for something more powerful: tracking causal paths through the model's concept space.

### Beyond Per-Token Snapshots

An activation map at (layer L, token T) is a snapshot - it tells you what concepts are active at that point. But the real structure of model cognition is the *flow* of concepts:

- **Vertically (across layers)**: Abstract concepts refine into specific ones
- **Horizontally (across tokens)**: Concepts propagate through attention
- **Diagonally (both)**: Earlier tokens at higher layers influence later tokens at lower layers

```
         Token 0    Token 1    Token 2    Token 3
            │          │          │          │
Layer 0     ●──────────●──────────●──────────●   Domain-level
            │╲         │ ╲        │          │
            │ ╲        │  ╲       │          │
Layer 1     ●──●───────●───●──────●──────────●   Category-level
            │   ╲      │  ╱       │╲         │
            │    ╲     │ ╱        │ ╲        │
Layer 2     ●─────●────●─────────●───●──────●   Concept-level
                   ╲       ╱          ╲
                    ╲     ╱            ╲
                     causal            causal
                     path A            path B
```

### What Causal Paths Reveal

**Normal cognition**:
- Parent concept at layer L-1 → child concept at layer L (hierarchical refinement)
- Concept at token T → related concept at token T+1 (sequential development)
- Multiple paths converge on output (integration)

**Anomalous patterns**:
- Concept appears with no causal antecedent ("from nowhere")
- Strong activation suddenly suppressed (competing interpretation lost)
- Output token has weak causal connection to preceding activations (confabulation?)
- Circular paths or unusual feedback patterns

### Implications for Data Generation

To build causal path maps, we need richer data during training/inference:

**Currently captured**:
- Activation at layer L, token T (single point)

**Needed for causal tracking**:
- Activations at multiple layers for same token (vertical slice)
- Activations across token sequence (horizontal slice)
- Attention patterns (which tokens attended to which)
- Residual stream deltas (what each layer added)

**Data generation changes**:
```
Current:  prompt → activation[layer=15, token=-1]
Future:   prompt → {
            activations: [layer][token] matrix,
            attention: [layer][head][from_token][to_token],
            residual_deltas: [layer][token],
          }
```

This is significantly more data per sample, but enables:
- Causal path reconstruction
- Intervention analysis ("what if this concept wasn't active?")
- Temporal coherence checking

### The Lookup Map as Causal Index

The activation map becomes an index into causal structure:

```python
# Hypothetical future API
causal_graph = model.get_causal_paths(prompt)

# "What caused Deception to activate at token 5?"
antecedents = causal_graph.trace_backward(
    concept="Deception",
    layer=3,
    token=5
)
# Returns: [
#   (Misdirection, layer=2, token=3, strength=0.7),
#   (FalseStatement, layer=2, token=4, strength=0.6),
# ]

# "Where does this SafetyMask activation lead?"
consequences = causal_graph.trace_forward(
    concept="SafetyMaskPersona",
    layer=1,
    token=0
)
# Track how it influences downstream tokens
```

### Subtoken Spaces

The residual stream isn't monolithic - different subspaces carry different information. The causal paths exist in these subspaces:

- **Concept subspace**: Where our activation maps live
- **Syntactic subspace**: Grammar, structure
- **Factual subspace**: Retrieved knowledge
- **Instruction subspace**: Task representation

Causal tracking could reveal when information moves between subspaces - e.g., a concept in the "factual" subspace being suppressed by something in the "instruction" subspace (the model "knows" something but is being told not to say it).

### Verification Before Building

Before implementing causal tracking, we need to verify:

1. **Activation maps work**: Do the weighted centroids actually discriminate concepts?
2. **Hierarchy shows up**: Do parent/child correlations appear as expected?
3. **Cross-token patterns exist**: Do we see concept propagation in attention?
4. **Anomalies are detectable**: Can we catch known-bad patterns?

The current `ConceptActivationMapper` is the test bed for (1) and (2). Once validated, we can design the richer data capture for (3) and (4).

## Future Directions

1. **Hierarchical weighting**: Weight features differently based on concept layer
2. **Temporal dynamics**: Track how activations evolve across tokens
3. **Ensemble methods**: Combine statistical and neural mappers
4. **Sparse representations**: Only store/compute significantly active concepts
5. **Causal path extraction**: Build the graph structure described above
6. **Intervention tools**: "What if" analysis on causal paths
