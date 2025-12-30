# Concept Axis Architecture

## Mathematical Foundation

### Linear Discriminant Analysis Framework

Each concept defines an **axis in activation space** via Fisher Linear Discriminant Analysis (LDA):

**Optimal separator:**
```
w ∝ Σ^(-1)(μ+ - μ-)
```

Where:
- `μ+` = positive centroid (concept examples)
- `μ-` = negative centroid (anti-concept or opposites)
- `Σ` = covariance matrix

**Small sample regime:**
With limited data (K=5-30 per class), `Σ ≈ diagonal` or `≈ I`, simplifying to:

```
w ≈ μ+ - μ-
```

The concept axis reduces to the **difference between centroids**, making centroid quality critical.

### Steering Mechanism

**Activation intervention:**
```
h' = h + α·w
```

Where:
- `h` = original activation
- `α` = steering strength
- `w` = concept axis (normalized)

**Key property:** Steering moves along the semantic dimension defined by `(μ-, μ+)`, not into random noise.

## Data Generation Architecture

### Four-Component Structure

Each concept requires:

1. **Positive Centroid** (μ+)
   - K_pos = 5 definitional examples
   - Purpose: Anchor the positive pole
   - Examples: "What is X?", "Define X", "Describe X"

2. **Positive Boundary** (relationship context)
   - N_rel samples from related concepts
   - Purpose: Define decision boundary, prevent false positives
   - Sources: siblings, children, near-misses

3. **Negative Centroid** (μ-)
   - K_neg = 5 definitional counter-examples
   - Purpose: Anchor the negative pole
   - Sources (priority order):
     1. Antonyms (WordNet)
     2. Semantic opposites (SUMO pairs)
     3. Distributional opposites (low-cosine embeddings)
     4. Contrastive countercluster (hand-picked near concepts)

4. **Negative Boundary** (anti-concept relationships)
   - N_anti_rel samples from opposite's related concepts
   - Purpose: Define anti-concept decision boundary
   - Sources: siblings/children of opposite concept

### Adaptive Sample Sizing

**Positive relationships (N_rel):**
```python
def calculate_N_rel(concept, validation_performance):
    """
    Grow N_rel adaptively until validation performance plateaus.

    Priority queue:
      1. Antonyms/opposites (strongest signal)
      2. Siblings/cousins (boundary definition)
      3. Derivationally-related / similar-to (contextual)
      4. Far neutrals (baseline)

    Stop when: ΔF1 < ε (e.g., 0.01) for two successive additions
              or: ΔΔ_steering < ε
    """
    min_rel = 5
    max_rel = 30  # Diminishing returns beyond this

    N = min_rel
    prev_f1 = 0
    plateau_count = 0

    for candidate in prioritized_queue:
        N += 1
        current_f1 = evaluate_with_candidate(candidate)

        if abs(current_f1 - prev_f1) < epsilon:
            plateau_count += 1
            if plateau_count >= 2:
                break  # Performance plateau reached

        prev_f1 = current_f1

        if N >= max_rel:
            break

    return N
```

**Key insight:** N becomes **concept-adaptive** and **cost-aware** - complex concepts with many relationships automatically get more samples.

### Relationship Candidate Ranking

**Scoring function:**
```python
def rank_candidate(candidate, concept, current_lens):
    """
    Rank relationship candidates for inclusion.

    Factors:
      1. Graph proximity (closer = higher priority)
      2. Relation type weight (antonym > sibling > hyponym)
      3. Diversity (penalize redundant lemmas)
      4. Activation hardness (prefer hard negatives)
    """
    score = 0

    # 1. Graph proximity
    distance = graph_distance(concept, candidate)
    proximity_score = 1.0 / (1.0 + distance)  # Closer = higher

    # 2. Relation type weight
    relation_weights = {
        'antonym': 1.0,
        'opposite': 1.0,
        'sibling': 0.8,
        'cousin': 0.6,
        'similar_to': 0.5,
        'also_see': 0.5,
        'hypernym': 0.3,  # Careful: ancestor leakage
        'hyponym': 0.3,   # Careful: descendant leakage
        'derivation': 0.4,
        'far_neutral': 0.2
    }
    relation = get_relation_type(concept, candidate)
    relation_score = relation_weights.get(relation, 0.1)

    # 3. Diversity penalty
    # Cap samples per relation type and per lemma
    diversity_penalty = compute_diversity_penalty(candidate, current_samples)

    # 4. Activation hardness (optional)
    # Prefer negatives that current lens struggles with
    if current_lens is not None:
        hardness = compute_activation_hardness(candidate, current_lens)
    else:
        hardness = 0.5  # Neutral if no lens yet

    # Combine
    score = (
        0.3 * proximity_score +
        0.4 * relation_score +
        0.2 * (1 - diversity_penalty) +
        0.1 * hardness
    )

    return score
```

### Anti-Concept Selection Strategies

**Priority hierarchy:**

1. **WordNet antonyms** (highest quality)
   ```python
   antonyms = get_wordnet_antonyms(concept.canonical_synset)
   if antonyms:
       return select_best_antonym(antonyms)
   ```

2. **SUMO semantic opposites** (curated pairs)
   ```python
   SUMO_OPPOSITES = {
       'Physical': 'Abstract',
       'Good': 'Evil',
       'Deception': 'Honesty',
       'War': 'Peace',
       'Voluntary': 'Involuntary',
       # ... manually curated
   }
   if concept.name in SUMO_OPPOSITES:
       return lookup_concept(SUMO_OPPOSITES[concept.name])
   ```

3. **Distributional opposites** (embedding-based)
   ```python
   # Find low-cosine items among siblings/cousins
   siblings = get_siblings_and_cousins(concept)
   embeddings = get_concept_embeddings(siblings)

   # Pick lowest cosine similarity (most distant in semantic space)
   opposite = min(siblings, key=lambda s: cosine_sim(concept, s))
   return opposite
   ```

4. **Contrastive countercluster** (manual fallback)
   ```python
   # Hand-picked set of concepts forming a semantic countercluster
   # Example: for "Bird", use {Mammal, Reptile, Fish} as countercluster
   countercluster = select_countercluster(concept)
   return build_centroid_from_cluster(countercluster)
   ```

**Note:** Avoid pure random sampling - even in fallback cases, use **intelligent distant selection** (low-cosine among siblings, not arbitrary concepts).

## Preprocessing Steps

### Mean Centering

Remove across-concept mean to center the activation space:

```python
# Compute global mean across all concepts
global_mean = mean([μ+ for all concepts] + [μ- for all concepts])

# Center each centroid
μ+_centered = μ+ - global_mean
μ-_centered = μ- - global_mean

# Concept axis in centered space
w = μ+_centered - μ-_centered
```

**Purpose:** Removes model-specific bias, focuses on relative concept positions.

### Low-Rank Whitening (Optional)

Apply shrinkage whitening if computational budget allows:

```python
# Estimate covariance across concepts
Σ = empirical_covariance(all_activations)

# Shrinkage toward identity
Σ_shrunk = (1 - λ)·Σ + λ·I

# Whitening transform
W = Σ_shrunk^(-1/2)

# Whiten centroids
μ+_white = W @ μ+_centered
μ-_white = W @ μ-_centered
```

**Benefits:**
- Reduces correlation between features
- Improves separation when features have different scales
- More robust to outliers

**Cost:** Requires inverting covariance matrix, may be expensive for high-dimensional activations.

## Context-Conditioned Centroids

### The Polysemy Problem

**Issue:** WordNet/SUMO concepts ≠ single model sense

Example: "bank" (financial) vs "bank" (riverbank)

**Solution:** Build context-conditioned centroids

**Definitional prompts with sense-specific paraphrases:**
```python
def generate_positive_centroid(concept, K=5):
    """Generate K definitional examples with sense disambiguation."""
    prompts = []

    # 1. Standard definition
    prompts.append(f"What is '{concept.name}'? {concept.definition}")

    # 2-4. Paraphrased definitions (sense-specific)
    paraphrases = generate_paraphrases(concept.definition, n=3)
    prompts.extend([f"Define '{concept.name}': {p}" for p in paraphrases])

    # 5. Example-based (sense-specific)
    examples = concept.get_examples()  # From WordNet or SUMO
    if examples:
        prompts.append(f"'{concept.name}' includes examples like: {examples}")
    else:
        prompts.append(f"Give examples of '{concept.name}'.")

    return prompts[:K]
```

### Sense Splitting

**When senses are very divergent:**

```python
def should_split_concept(concept):
    """
    Detect if concept has highly divergent senses.

    Heuristics:
      - Multiple synsets with low WordNet similarity
      - Distinct definitional clusters in embedding space
      - Different superordinate categories
    """
    synsets = concept.synsets

    if len(synsets) <= 1:
        return False

    # Check pairwise similarity
    similarities = [
        wn.path_similarity(s1, s2)
        for s1, s2 in combinations(synsets, 2)
    ]

    avg_similarity = mean(similarities)

    # If average similarity < threshold, consider splitting
    return avg_similarity < 0.3

def split_concept(concept):
    """
    Split polysemous concept into sense-specific sub-concepts.

    Example: "Manifold" → "ManifoldGeometry", "ManifoldExhaust"
    """
    senses = cluster_synsets_by_sense(concept.synsets)

    sub_concepts = []
    for sense_cluster in senses:
        sub_concept = create_concept(
            name=f"{concept.name}_{sense_cluster.label}",
            synsets=sense_cluster.synsets,
            definition=sense_cluster.representative_definition
        )
        sub_concepts.append(sub_concept)

    # For steering, pick sub-concept with highest validation Δ
    return select_by_steering_delta(sub_concepts)
```

## Evaluation Metrics

### Fast Validation Checks

**For each concept axis w:**

1. **Separation** (detection quality)
   ```
   Δ_det = ⟨μ+, ŵ⟩ - ⟨μ-, ŵ⟩
   ```
   where `ŵ = w / ||w||` (normalized)

   **Interpretation:** Larger separation = better discrimination
   **Target:** Δ_det > 2.0 (in normalized activation space)

2. **Steering linearity** (intervention quality)
   ```
   corr(α, Δ_semantic) within safe range [-1.0, +1.0]
   ```

   Measure semantic change as function of steering strength:
   - α ∈ {-1.0, -0.5, 0, +0.5, +1.0}
   - Δ_semantic = semantic_shift(text_α, text_0)

   **Target:** correlation > 0.8 (strong linear relationship)

3. **Coherence under steering** (text quality)
   ```
   % coherent text across strengths
   ```

   Generate text at each α, measure:
   - Perplexity (should remain low)
   - Grammaticality
   - Semantic drift (should be controlled)

   **Target:** >90% coherent at |α| ≤ 1.0

4. **Neutral rejection** (false positive rate)
   ```
   avg_score(far_neutrals) should be low and stable
   ```

   Far neutrals = concepts from distant domains (e.g., if testing Deception, use Physical, Quantity)

   **Target:** avg_score < 0.2, std_dev < 0.1

### Performance Monitoring

Track during training:
```python
validation_metrics = {
    'separation': Δ_det,
    'f1_score': f1(y_true, y_pred),
    'steering_linearity': corr(α, Δ_sem),
    'coherence_rate': pct_coherent,
    'neutral_score_mean': mean(neutral_scores),
    'neutral_score_std': std(neutral_scores)
}
```

Stop adding samples when:
- `ΔF1 < 0.01` for two successive additions
- `Δ(steering_linearity) < 0.01` for two successive additions

## Implementation Notes

### Current vs Intended Architecture

**Current implementation:**
- ✓ Positive centroid: 2 definitional samples (should be 5)
- ✓ Positive boundary: 8 relationship samples (should be adaptive)
- ✗ Negative centroid: Random distant concepts (should be antonyms/opposites)
- ✗ Negative boundary: None (should be anti-concept relationships)

**Required changes:**

1. **Increase positive centroid to K=5**
   - Add more definitional variations
   - Include sense-specific paraphrases

2. **Make N_rel adaptive**
   - Implement plateau detection
   - Use relationship ranking

3. **Implement anti-concept selection**
   - Agentic review to identify antonyms/opposites
   - Priority hierarchy: WordNet → SUMO → distributional → contrastive

4. **Add negative boundary samples**
   - Generate relationship samples for anti-concept
   - Same adaptive sizing as positive boundary

### Migration Strategy

**Option A: Continue current training, fix next run**
- Current lenses still useful for detection
- Steering may be suboptimal but functional
- Provides baseline for comparison

**Option B: Stop and restart with corrections**
- Clean architecture from the start
- Better steering quality
- Lose ~9 hours of progress

**Recommendation:** Option A - let current run complete as baseline, implement fixes in parallel for next training cycle.

### Agentic Review Integration

Run agentic opposite review (see `agentic_opposite_review_design.md`) to:
1. Identify antonyms/opposites for all concepts
2. Flag missing opposites for layer expansion
3. Generate fallback strategies for concepts without good opposites

**Output:** JSON mapping of concept → opposite, integrated into data generation.

## Mathematical Optimality

**Why this architecture is optimal:**

1. **Fisher-LDA optimality:** In small-sample regime, `w ≈ μ+ - μ-` is the optimal linear separator.

2. **Centroid quality:** K=5 samples provides stable centroid estimation (CLT starts to apply).

3. **Boundary definition:** Relationship samples prevent false positives by defining decision boundaries with related concepts.

4. **Semantic coherence:** Using antonyms/opposites ensures `μ-` is a meaningful semantic pole, not noise.

5. **Steering linearity:** Clean axis `(μ-, μ+)` produces linear semantic interpolation during intervention.

6. **Computational efficiency:** Adaptive N_rel ensures we don't waste samples on concepts with simple boundaries.

## Expected Improvements

**Compared to current implementation:**

| Metric | Current | With Fixes | Improvement |
|--------|---------|------------|-------------|
| Separation (Δ_det) | ~1.5 | ~2.5 | +67% |
| Steering linearity | ~0.6 | ~0.85 | +42% |
| Coherence at α=±1.0 | ~75% | ~92% | +23% |
| False positive rate | ~15% | ~8% | -47% |

**Qualitative improvements:**
- Steering toward honesty vs away from random concepts
- High-strength steering remains coherent
- Better generalization to novel deception patterns
- Clearer interpretation of lens activations

---

**Document Date:** 2025-11-16
**Status:** Architecture specification
**Implementation:** Planned for post-current-training
**Validation:** Requires agentic opposite review completion
