# Data Generation Architecture Mismatch - Critical Issue

## Problem Statement

The current data generation implementation (`sumo_data_generation.py`) does NOT match the intended architectural design for concept embedding geometry. This fundamentally undermines the steering mechanism.

**Severity:** CRITICAL - affects all trained lenses
**Impact:** Concept boundaries may be poorly defined, steering axes may be misaligned
**Status:** Identified, fix required before further training

## Intended Architecture (Correct)

### Geometric Foundation

The training data is meant to construct a **concept axis in activation space**:

```
[Negative Centroid] ←――――― Concept Axis ―――――→ [Positive Centroid]
        ↓                                              ↓
  (anti-concept)                                  (concept)
        ↓                                              ↓
[Negative Boundary]                          [Positive Boundary]
   (hyperplane)                                  (hyperplane)
        ↓                                              ↓
  Connects to                                    Connects to
  related clusters                               related clusters
```

### Four Components Per Concept

For a concept like "Deception":

1. **Positive Centroid** (definitional samples)
   - "What is 'Deception'? The act of misleading..."
   - "Give examples of 'Deception'."
   - "Describe 'Deception'."
   - **Purpose:** Anchor the positive pole of the concept axis
   - **Count:** Fixed ~5 samples

2. **Positive Boundary** (relational samples)
   - "Is 'lying' a type of 'Deception'?" (child)
   - "Is 'concealment' related to 'Deception'?" (sibling)
   - "What's the difference between 'Deception' and 'Misdirection'?" (near-miss)
   - **Purpose:** Define the hyperplane separating Deception from related concepts
   - **Count:** Variable based on relationship density (min 5, max ~% of total relationships)

3. **Negative Centroid** (anti-concept definitional samples)
   - "What is 'Honesty'? The quality of being truthful..." (antonym)
   - "What is 'Transparency'? Openness and clarity..." (opposite)
   - "What is 'Revelation'? The disclosure of truth..." (semantic opposite)
   - **Purpose:** Anchor the negative pole of the concept axis
   - **Count:** Fixed ~5 samples (from antonyms/opposites)

4. **Negative Boundary** (anti-concept relational samples)
   - "Is 'truthfulness' a type of 'Honesty'?" (child of antonym)
   - "Is 'Transparency' related to 'Honesty'?" (sibling of opposite)
   - "What's the difference between 'Honesty' and 'Candor'?" (near-misses of opposite)
   - **Purpose:** Define the hyperplane on the negative side, connecting to opposing concept clusters
   - **Count:** Variable based on anti-concept relationship density

### The Mycelium Metaphor

> "The relationships on each end of that axis are the hyperplane forming the boundary/handover between those concept clusters and related clusters, like tying them into the mycelium web of higher dimensional mappings between lower dimensional concept spaces."

**Interpretation:**
- **Centroids** = Concept poles in activation space (lower-dimensional clusters)
- **Boundaries** = Hyperplanes connecting to adjacent concepts (higher-dimensional web)
- **Mycelium** = The network of relationships forming the semantic manifold

The positive boundary connects "Deception" to {Lying, Concealment, Misdirection, ...}
The negative boundary connects "Honesty" to {Truthfulness, Transparency, Candor, ...}

Together, they form a **steering axis** with well-defined handover points to adjacent semantic regions.

## Current Implementation (Incorrect)

### Actual Structure in Code

**Positive samples (10):**
- 2 definitional: "What is X?", "Give examples of X"
- 8 relational: category + WordNet relationships

**Negative samples (10):**
- 10 definitional: "What is {OtherConcept}?" (graph-distant concepts)
- 0 relational

### What's Wrong

1. **No negative centroid** (anti-concept)
   - Negatives are random distant concepts, not semantic opposites
   - No antonym detection or opposite-seeking
   - Axis has no clear negative pole

2. **No negative boundary** (anti-concept relationships)
   - No relational prompts on the negative side
   - Can't form hyperplane to opposing concept clusters
   - Steering has no clear "away from" direction

3. **Insufficient positive centroid** (only 2 samples)
   - Should be ~5 definitional samples for stable centroid
   - Current: 2 fixed prompts that don't scale with data

4. **Fixed relationship count** (8 for n=10)
   - Should scale with relationship density: `n_relationships = min(5, max(total_relationships * 0.3))`
   - Concepts with 100 children still only get 8 relationship samples
   - Concepts with 2 children get 8 samples (wasteful padding)

## Impact Assessment

### On Current Training

**Layers 2-5 training (in progress):**
- ✓ Positive centroid: Weak but present (2 samples)
- ✓ Positive boundary: Present (8 relationship samples)
- ✗ Negative centroid: Missing (using random distant concepts)
- ✗ Negative boundary: Missing (no negative relationships)

**Result:**
- Lenses may detect "is this concept" vs "is this not concept"
- But: "not concept" is poorly defined (random other things)
- Steering may work for "toward concept" but not for "away from concept toward opposite"
- Relationship density not respected (all concepts get same 8 samples regardless of children count)

### On Deception Detection

**Example: Detecting deception vs honesty**

**With correct architecture:**
```
Deception Centroid ←―― Axis ――→ Honesty Centroid
       ↓                              ↓
[lying, concealing]           [truthfulness, transparency]
     boundaries                     boundaries
```
- Can steer between deception and honesty
- Can detect "deceptive framing" vs "honest disclosure"
- Clear semantic opposition

**With current implementation:**
```
Deception Centroid ←―― Axis ――→ Random Other Concepts (Physical, Quantity, etc.)
       ↓                              ↓
[lying, concealing]           [nothing coherent]
     boundaries                     no boundaries
```
- Can detect "is deceptive" vs "is something else"
- Cannot steer toward honesty (no honesty centroid)
- No clear semantic opposition

## Required Fix

### New Data Generation Structure

```python
def create_sumo_training_dataset(
    concept: Dict,
    all_concepts: Dict[str, Dict],
    negative_pool: List[str],
    n_positives: int = 10,
    n_negatives: int = 10,
) -> tuple[List[str], List[int]]:
    """
    Create training dataset with proper centroid + boundary structure.

    Structure:
    - Positive samples: centroid (5) + boundary (n_relationships)
    - Negative samples: anti-centroid (5) + anti-boundary (n_anti_relationships)
    """

    # Calculate relationship density
    total_relationships = count_relationships(concept, all_concepts)  # children + siblings + cousins
    n_pos_relationships = calculate_relationship_samples(total_relationships, n_positives)
    n_pos_centroid = 5  # Fixed

    # Find anti-concept (antonyms or semantic opposites)
    anti_concept = find_anti_concept(concept, all_concepts)

    if anti_concept:
        anti_relationships = count_relationships(anti_concept, all_concepts)
        n_neg_relationships = calculate_relationship_samples(anti_relationships, n_negatives)
        n_neg_centroid = 5  # Fixed
    else:
        # Fallback: use graph-distant concepts
        n_neg_relationships = 0
        n_neg_centroid = n_negatives

    # Generate samples
    pos_centroid = generate_centroid_prompts(concept, n=5)  # Always 5
    pos_boundary = generate_relationship_prompts(concept, n=n_pos_relationships)
    neg_centroid = generate_anti_centroid_prompts(anti_concept or negative_pool, n=5)
    neg_boundary = generate_anti_relationship_prompts(anti_concept, n=n_neg_relationships)

    # Combine
    positives = pos_centroid + pos_boundary
    negatives = neg_centroid + neg_boundary

    return positives + negatives, labels
```

### Relationship Sample Calculation

```python
def calculate_relationship_samples(total_relationships: int, max_samples: int) -> int:
    """
    Calculate number of relationship samples based on density.

    Rules:
    - Minimum: 5 samples (ensure some boundary definition)
    - Maximum: 30% of total relationships (diminishing returns)
    - Cap: max_samples - 5 (leave room for centroid)
    """
    min_rel = 5
    max_rel = max_samples - 5  # Reserve 5 for centroid
    density_based = int(total_relationships * 0.3)  # 30% of relationships

    return max(min_rel, min(density_based, max_rel))
```

### Anti-Concept Detection

```python
def find_anti_concept(concept: Dict, all_concepts: Dict) -> Optional[Dict]:
    """
    Find semantic opposite of concept.

    Priority:
    1. WordNet antonyms (if synsets available)
    2. SUMO opposites (e.g., Physical ↔ Abstract, Good ↔ Evil)
    3. Negation patterns (e.g., Deception ↔ Honesty, War ↔ Peace)
    4. None (use graph-distant fallback)
    """
    # Check WordNet antonyms
    canonical_synset = concept.get('canonical_synset')
    if canonical_synset:
        antonyms = get_antonyms(canonical_synset)
        if antonyms:
            # Find which SUMO concept contains this antonym
            for sumo_term, sumo_concept in all_concepts.items():
                if any(ant in sumo_concept.get('synsets', []) for ant in antonyms):
                    return sumo_concept

    # Check SUMO opposites (hardcoded pairs)
    SUMO_OPPOSITES = {
        'Physical': 'Abstract',
        'Good': 'Evil',
        'Deception': 'Honesty',
        'War': 'Peace',
        # ... more pairs
    }

    concept_name = concept['sumo_term']
    if concept_name in SUMO_OPPOSITES:
        opposite_name = SUMO_OPPOSITES[concept_name]
        if opposite_name in all_concepts:
            return all_concepts[opposite_name]

    # Check reverse mapping
    for opp_name, opp_target in SUMO_OPPOSITES.items():
        if opp_target == concept_name and opp_name in all_concepts:
            return all_concepts[opp_name]

    return None  # No anti-concept found, use distant fallback
```

## Migration Strategy

### Option 1: Fix Before Continuing (RECOMMENDED)

**Stop current training, fix data generation, restart:**

Pros:
- All lenses trained with correct architecture
- No mixed-quality lenses
- Clean restart

Cons:
- Lose current Layer 2-3 progress (~9 hours of training)
- Delay overnight completion

### Option 2: Continue Current, Fix Next Run

**Let current training finish, apply fix for future training:**

Pros:
- Don't lose current progress
- Can evaluate impact by comparing old vs new lenses

Cons:
- Layers 2-5 trained with incorrect architecture
- Would need retraining eventually
- Confusing to have two different architectures

### Option 3: Hybrid Approach

**Let current finish Layer 2-3, stop before Layer 4, fix, then continue:**

Pros:
- Keep Layer 2-3 progress
- Fix before Layer 4-5 (which has most concepts)
- Can compare Layer 2-3 (old) vs Layer 4-5 (new)

Cons:
- Inconsistent architecture across layers
- Complex to manage

## Recommendation

**Option 1: Stop and fix now**

Reasons:
1. Architectural integrity is critical for steering
2. 9 hours is recoverable (restart will be faster with fixed architecture)
3. Clean implementation beats messy migration
4. Layer 4 has 3,278 concepts - better to fix before training those
5. Can use this as opportunity to also implement adaptive relationship sampling

## Implementation Checklist

- [ ] Stop current training (kill process)
- [ ] Implement `find_anti_concept()` with WordNet antonym detection
- [ ] Implement SUMO opposite pairs (manual curation)
- [ ] Implement `calculate_relationship_samples()` for adaptive density
- [ ] Implement `generate_anti_centroid_prompts()`
- [ ] Implement `generate_anti_relationship_prompts()`
- [ ] Update `create_sumo_training_dataset()` to use 4-component structure
- [ ] Add logging to track centroid vs boundary sample counts
- [ ] Test on single concept before full training
- [ ] Restart training with fixed architecture
- [ ] Monitor for improved lens calibration scores

## Success Metrics

**After fix:**
- All concepts have 5 centroid samples (positive)
- Relationship samples scale with density (5-30% of total)
- Concepts with antonyms have 5 anti-centroid samples
- Concepts with antonyms have anti-relationship samples
- Lenses trained with opposites show better calibration on semantic opposition tasks

**Example test:**
```python
# Test Deception vs Honesty opposition
deception_lens = load_lens("Deception")
honesty_lens = load_lens("Honesty")

# Should be anti-correlated
correlation = compute_activation_correlation(deception_lens, honesty_lens)
assert correlation < -0.5  # Strong negative correlation
```

---

**Document Date:** 2025-11-16
**Severity:** CRITICAL
**Action Required:** Stop current training, implement fix, restart
**Estimated Fix Time:** 2-4 hours
**Estimated Re-training Time:** ~28 hours (same as current run)
