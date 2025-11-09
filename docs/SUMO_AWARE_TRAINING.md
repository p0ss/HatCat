# SUMO-Aware Training for Hierarchical Concept Classification

## Overview

This document describes the SUMO-aware training data generation system for hierarchical concept classification. The system extends the original WordNet-based approach to support SUMO (Suggested Upper Merged Ontology) category hierarchies.

## Key Insight

**SUMO `category_children` relationships ARE equivalent to WordNet hypernym/hyponym relationships.**

Both represent "is-a" hierarchies:
- **WordNet**: `physical_entity` → hyponyms: [`object`, `process`, `substance`]
- **SUMO**: `Physical` → category_children: [`Object`, `Process`, `Collection`]

The training system treats these equivalently, allowing binary classifiers to learn from both SUMO category structure and WordNet semantic relationships.

## Architecture

### Dual Relationship Sources

For each SUMO concept, training data is generated from TWO complementary sources:

1. **SUMO Category Hierarchy** (`category_children`)
   - Direct relationships from the SUMO ontology
   - Example: "Physical includes the subcategory Process"

2. **WordNet Relationships** (via `canonical_synset`)
   - Hypernyms (broader categories)
   - Hyponyms (more specific types)
   - Meronyms (parts) / Holonyms (wholes)
   - Antonyms (opposites)
   - Example: "physical entity is a type of entity"

### Training Data Format

**Positive Examples** (label=1):
```
1. Definition: "What is Physical? an entity that has physical existence"
2. SUMO relationship: "Physical has subcategory Process"
3. SUMO relationship: "Collection is a type of Physical"
4. WordNet relationship: "physical entity is a type of entity"
5. WordNet relationship: "physical entity has type substance"
... (up to n_positives samples)
```

**Negative Examples** (label=0):
```
11. "What is Proposition? a condition or position in which you find yourself"
12. "What is Quantity? the concept that something has a magnitude..."
... (up to n_negatives samples)
```

## Implementation

### Core Function: `create_sumo_training_dataset()`

```python
from src.training.sumo_data_generation import create_sumo_training_dataset

prompts, labels = create_sumo_training_dataset(
    concept=concept_dict,           # SUMO concept with category_children, canonical_synset
    all_concepts=concept_map,       # Map of all SUMO concepts
    negative_pool=neg_concepts,     # Semantically distant concepts
    n_positives=10,                # 1 def + 9 relationships
    n_negatives=10,                # 10 distant concepts
    use_category_relationships=True,  # Include SUMO hierarchy
    use_wordnet_relationships=True    # Include WordNet relations
)
```

### Negative Pool Generation

Uses `build_sumo_negative_pool()` with hierarchy-aware filtering:

**Exclusions**:
- Same concept (self)
- Direct children (category_children)
- Direct parents (concepts that list this as a child)

**Layer 0 Special Case**:
- All concepts are at the same layer (depth 0-2)
- Any concept that's not a direct parent/child is valid
- Result: ~9 negatives per concept from 14 total Layer 0 concepts

**Multi-Layer Case**:
- Uses `min_layer_distance` parameter (default=0)
- Ensures semantic distance via layer separation

## Validation Results

### Layer 0 Relationship Coverage

Tested on 5 Layer 0 concepts:

| Concept        | Category Children | WordNet Rels | Total Rels |
|----------------|-------------------|--------------|------------|
| Physical       | 3                 | 7            | 10         |
| SetOrClass     | 2                 | 0            | 2          |
| Quantity       | 3                 | 11           | 14         |
| PhysicalSystem | 0                 | 0            | 0          |
| Proposition    | 8                 | 5            | 13         |

**Average**: 7.8 relationships per concept

### Training Data Quality

**Example: Physical concept**

Positive prompts successfully combine:
- ✅ SUMO structure: "Physical has subcategory Process"
- ✅ WordNet semantics: "physical entity is a type of entity"
- ✅ Definitional: "What is Physical? an entity that has physical existence"

Negative prompts properly exclude:
- ❌ Direct children (Object, Process, Collection)
- ✅ Include distant concepts (Proposition, Quantity, SetOrClass)

## Usage Patterns

### Pattern 1: Balanced Training (Recommended)

```python
# 50/50 split between SUMO and WordNet relationships
prompts, labels = create_sumo_training_dataset(
    concept,
    all_concepts,
    negative_pool,
    n_positives=10,  # 1 def + ~4 SUMO + ~5 WordNet
    n_negatives=10,
    use_category_relationships=True,
    use_wordnet_relationships=True
)
```

**Benefits**:
- Learns both ontological structure (SUMO) and linguistic semantics (WordNet)
- Maximum robustness for hierarchical zoom + detection

### Pattern 2: SUMO-Only

```python
# Focus on SUMO category hierarchy
prompts, labels = create_sumo_training_dataset(
    concept,
    all_concepts,
    negative_pool,
    n_positives=10,
    n_negatives=10,
    use_category_relationships=True,
    use_wordnet_relationships=False  # SUMO only
)
```

**Use Case**: When WordNet coverage is poor (e.g., AI categories without synsets)

### Pattern 3: WordNet-Only

```python
# Focus on WordNet relationships
prompts, labels = create_sumo_training_dataset(
    concept,
    all_concepts,
    negative_pool,
    n_positives=10,
    n_negatives=10,
    use_category_relationships=False,  # No SUMO
    use_wordnet_relationships=True
)
```

**Use Case**: Comparison with pure WordNet training (original Phase 1-4 approach)

## Integration with Binary Classifiers

### Training Pipeline

1. **Load SUMO layer concepts**
   ```python
   with open('data/concept_graph/abstraction_layers/layer0.json') as f:
       layer_data = json.load(f)
   concepts = layer_data['concepts']
   ```

2. **Generate training data**
   ```python
   from src.training.sumo_data_generation import (
       create_sumo_training_dataset,
       build_sumo_negative_pool
   )

   concept_map = {c['sumo_term']: c for c in concepts}
   negative_pool = build_sumo_negative_pool(concepts, target_concept)

   prompts, labels = create_sumo_training_dataset(
       target_concept,
       concept_map,
       negative_pool,
       n_positives=10,
       n_negatives=10
   )
   ```

3. **Extract activations** (existing pipeline)
   ```python
   # Run prompts through LLM to extract activations
   activations = extract_activations(model, tokenizer, prompts)
   ```

4. **Train BiLSTM classifier** (existing pipeline)
   ```python
   # Train binary classifier on activations
   classifier = train_binary_classifier(activations, labels)
   ```

## Relationship to Phase 7

**Important**: Phase 7 stress test (`scripts/phase_7_stress_test.py`) currently uses `ManifoldSteerer` which is for **steering**, not classifier training.

**Correct approach** for SUMO classifier training:
- Use `create_sumo_training_dataset()` (this module)
- Train BiLSTM+MLP binary classifiers
- Evaluate with comprehensive pos/neg/neutral testing (Phase 4 methodology)

## Testing

Run relationship extraction test:
```bash
poetry run python scripts/test_sumo_relationships.py
```

**Validates**:
- ✅ SUMO category relationships accessible
- ✅ WordNet relationships accessible via canonical_synset
- ✅ Training data generation combines both sources
- ✅ Negative pool properly excludes direct relations

## Files

- **`src/training/sumo_data_generation.py`**: Core implementation
- **`scripts/test_sumo_relationships.py`**: Validation script
- **`data/concept_graph/abstraction_layers/layer*.json`**: SUMO hierarchy data

## Future Work

### Multi-Layer Training

Extend to Layers 1-6 with cross-layer relationships:
- Layer 1 concepts reference Layer 0 parents
- Layer 2 concepts reference Layer 1 parents
- Enables hierarchical "zoom" from abstract → specific

### AI Category Population

The AI.kif expansion created 16 populated AI categories:
- `ArtificialIntelligence`, `LanguageModel`, `AISuffering`, etc.
- These have `category_children` but may lack WordNet synsets
- SUMO-only training mode (Pattern 2) will be essential

### Cross-Model Probe Transfer

Test if classifiers trained on Gemma-3-4b transfer to:
- Claude, GPT-4, Llama models
- Validates universal semantic structure hypothesis

## References

- **SUMO Ontology**: http://www.adampease.org/OP/
- **WordNet 3.0**: https://wordnet.princeton.edu/
- **Phase 1-4 Results**: See `projectplan.md`
- **V5 Hierarchical Model**: See `README.md`
