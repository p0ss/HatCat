# V5: WordNet Hyponym Intermediate Layers

## Summary

V5 extends V4 by extracting WordNet hyponym clusters to create intermediate layers for large fan-out SUMO categories.

## Implementation Status

**Current**: V4 with domain ontologies (Food, Geography, People)
- 3,193 SUMO terms, 5,451 populated
- 24,920 synsets remapped via hypernym chains (21.5%)
- Layers: 11 → 211 → 917 → 869 → 3,213 (SUMO) → 115,930 (synsets)

**Target**: V5 with hyponym intermediates
- Same SUMO foundation as V4
- Add intermediate layers between SUMO categories and synsets
- Target: 0-4 (SUMO) → 5-6 (WordNet clusters) → 7 (synsets)

## Large Fan-Out Categories (Candidates for Hyponym Subdivision)

From V4 analysis:

| SUMO Category | Layer | Synsets | SUMO Children | Issue |
|---------------|-------|---------|---------------|-------|
| SubjectiveAssessmentAttribute | 2 | 8,262 | 5 (empty) | No subdivision |
| FloweringPlant | 2 | 5,192 | 0 | No subdivision |
| Man | 3 | 2,829 | 1 (empty) | Boy unpopulated |
| Device | 1 | 2,412 | 57 (55.8% coverage) | Partial |
| Position | 2 | 1,621 | 2 (empty) | No subdivision |
| DiseaseOrSyndrome | 2 | 1,588 | 6 (empty) | No subdivision |
| Motion | 1 | 1,065 | 15 (empty) | No subdivision |

## WordNet Hyponym Extraction Strategy

### Algorithm

```python
def extract_hyponym_clusters(parent_sumo_category, threshold=100):
    """
    For a large SUMO category, find natural WordNet hyponym groupings.

    1. Get all synsets mapped to this SUMO category
    2. Build hyponym tree using WordNet relations
    3. Find "anchor synsets" - synsets with many hyponyms
    4. Cluster synsets under anchor synsets
    5. Create pseudo-SUMO categories named after anchor synsets
    """

    # Example for SubjectiveAssessmentAttribute:
    # 1. Find synsets like good.a.01, bad.a.01, neutral.a.01
    # 2. These have many hyponyms (excellent.a.01 → good.a.01)
    # 3. Create clusters:
    #    - PositiveAttribute (pseudo-SUMO) ← good.a.01 and hyponyms
    #    - NegativeAttribute (pseudo-SUMO) ← bad.a.01 and hyponyms
    #    - NeutralAttribute (pseudo-SUMO) ← neutral synsets
```

### Configuration

```python
LARGE_CATEGORY_THRESHOLD = 1000  # synsets
HYPONYM_CLUSTER_MIN_SIZE = 50    # minimum cluster size
ANCHOR_SYNSET_MIN_HYPONYMS = 20  # minimum hyponyms to be anchor

TARGET_CATEGORIES = [
    'SubjectiveAssessmentAttribute',  # → positive/negative/neutral
    'FloweringPlant',                 # → tree/shrub/herb
    'Device',                         # → instrument/machine/tool
    'DiseaseOrSyndrome',             # → psychological/infectious/chronic
    'Motion',                        # → upward/downward/rotation
]
```

## Expected Layer Structure (V5)

```
Layer 0:     11 SUMO categories (depth 0-2)
Layer 1:    211 SUMO categories (depth 3-4)
Layer 2:    917 SUMO categories (depth 5-6)
Layer 3:    869 SUMO categories (depth 7-9)
Layer 4:  3,213 SUMO categories (depth 10-999)
Layer 5:    ~50 WordNet hyponym clusters (large category subdivisions)
Layer 6:   ~300 WordNet hyponym subclusters (optional, if needed)
Layer 7: 115,930 WordNet synsets
```

## Implementation Steps

### Phase 1: Identify Large Categories
```python
large_categories = [
    concept for concept in layers[0:5]
    if concept['synset_count'] > LARGE_CATEGORY_THRESHOLD
    and len(concept['category_children']) < 5
]
```

### Phase 2: Extract Hyponym Clusters
For each large category:
1. Load all synsets from Layer 7
2. Build hyponym graph
3. Find anchor synsets (high hyponym count)
4. Cluster synsets under anchors
5. Name clusters using anchor lemmas or semantic analysis

### Phase 3: Create Intermediate Layers
```python
for large_cat in large_categories:
    clusters = extract_hyponym_clusters(large_cat)
    for cluster in clusters:
        pseudo_sumo = {
            'sumo_term': f"{large_cat['sumo_term']}_{cluster.name}",
            'is_pseudo_sumo': True,
            'parent_sumo': large_cat['sumo_term'],
            'anchor_synset': cluster.anchor,
            'synset_count': len(cluster.synsets),
            'layer': 5,  # Intermediate layer
        }
        layers[5].append(pseudo_sumo)
```

### Phase 4: Remap Synsets
Update synsets to point to intermediate layers instead of directly to large categories.

## AI Relevance: SubjectiveAssessmentAttribute

This category is particularly important for AI introspection:

**Current**: 8,262 synsets dumped directly into SubjectiveAssessmentAttribute

**After V5 subdivision**:
```
SubjectiveAssessmentAttribute (Layer 2)
├─ PositiveAttribute (Layer 5, ~3000 synsets)
│  ├─ good.a.01, excellent.a.01, outstanding.a.01, ...
│  └─ Used for: positive self-assessment, helpfulness detection
├─ NegativeAttribute (Layer 5, ~3000 synsets)
│  ├─ bad.a.01, poor.a.01, terrible.a.01, ...
│  └─ Used for: error states, harmful content detection
├─ NeutralAttribute (Layer 5, ~1500 synsets)
│  └─ Used for: objective states
└─ AmbiguousAttribute (Layer 5, ~762 synsets)
   └─ Context-dependent assessments
```

This hierarchical organization enables:
- Training separate positive/negative affect classifiers
- Detecting emotional valence in model outputs
- Modeling subjective states in AI psychology
- Connecting to emotion.kif concepts

## Next Steps

1. **Implement hyponym extraction** (add to V5 codebase)
2. **Test on SubjectiveAssessmentAttribute** (8k synsets → ~4 clusters)
3. **Extend to other large categories** (FloweringPlant, Device, etc.)
4. **Validate layer distribution** (ensure order-of-magnitude scaling)
5. **Add AI safety ontology** on top of V5 foundation

## Files

- `src/build_sumo_wordnet_layers_v5.py` - Implementation (default build)
- `docs/v5_hyponym_intermediates_plan.md` - This file
- `docs/sumo_wordnet_hierarchy.md` - Overall documentation

## References

- WordNet hyponym/hypernym relations
- V4 hierarchical remapping baseline
- SUMO domain ontologies (Food, Geography, People)
