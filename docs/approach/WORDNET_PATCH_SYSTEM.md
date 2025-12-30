# WordNet Patch System

## Overview

The WordNet Patch System extends WordNet 3.0 with custom semantic relationships for domain-specific concepts. This implements the architectural decision to **separate hierarchical categorization (SUMO/KIF) from semantic relationships (WordNet patches)**.

## Architecture Decision

### Separation of Concerns

**SUMO (KIF files)**: Hierarchical categorization only
- Parent-child relationships via `(subclass Child Parent)`
- Layer assignment based on depth from root concepts
- Ontological structure and categorization

**WordNet Patches**: Semantic relationships only
- Synonyms, antonyms, role variants, contrasts
- Horizontal relationships between concepts at same or different layers
- Domain-specific distinctions (e.g., AI vs Human agent roles)

### Why This Separation?

1. **Clarity**: Clear distinction between "is-a" (SUMO) and "relates-to" (WordNet)
2. **Flexibility**: Can add relationships without restructuring ontology
3. **Maintainability**: Changes to one system don't affect the other
4. **Version Stability**: WordNet patches are pinned to WordNet 3.0
5. **Composability**: Multiple patches can be loaded independently

## Components

### 1. Schema Specification

**Location**: `data/concept_graph/wordnet_patches/SCHEMA.md`

Defines:
- JSON structure for patch files
- Relationship types (`role_variant`, `antonym`, `similar_to`, etc.)
- Custom synset format
- Metadata requirements
- Validation rules

### 2. Patch Loader

**Location**: `src/data/wordnet_patch_loader.py`

**Class**: `WordNetPatchLoader`

**Key Methods**:
```python
loader = WordNetPatchLoader()

# Load patches
loader.load_all_patches("data/concept_graph/wordnet_patches/")

# Query relationships
role_variants = loader.get_role_variants("satisfaction.n.01")
antonyms = loader.get_antonyms("satisfaction.n.01")
all_rels = loader.get_all_relationships("satisfaction.n.01")

# Custom synsets
synset = loader.get_custom_synset("valencepositiveaiagent.n.01")
is_custom = loader.is_custom_synset("satisfaction.n.01")

# Summary
summary = loader.export_summary()
```

**Features**:
- Validates patch schema compliance
- Builds bidirectional indices for fast lookup
- Automatically creates reverse relationships for bidirectional relations
- Supports custom synsets not in base WordNet
- Validates SUMO alignment

### 3. Patch Files

**Location**: `data/concept_graph/wordnet_patches/`

**Naming Convention**: `wordnet_<version>_<name>.json`

**Current Patches**:
- `wordnet_3.0_persona_relations.json` - 30 role_variant relationships + 10 antonyms for persona concepts

## Relationship Types

### 1. `role_variant`
Same concept applied to different agent roles (AI vs Human vs Other).

**Example**:
```json
{
  "synset1": "satisfaction.n.01",
  "synset2": "satisfaction.n.01",
  "relation_type": "role_variant",
  "bidirectional": false,
  "metadata": {
    "role1": "HumanAgent",
    "role2": "AIAgent",
    "sumo_term1": "ValencePositive_HumanAgent",
    "sumo_term2": "ValencePositive_AIAgent",
    "axis": "valence",
    "pole": "positive"
  }
}
```

**Use Case**: Distinguishing whether model is representing human satisfaction vs AI satisfaction when using same WordNet synset.

### 2. `antonym`
Semantic opposition along an axis.

**Example**:
```json
{
  "synset1": "satisfaction.n.01",
  "synset2": "dissatisfaction.n.01",
  "relation_type": "antonym",
  "bidirectional": true,
  "metadata": {
    "axis": "valence",
    "pole1": "positive",
    "pole2": "negative"
  }
}
```

### 3. `similar_to`
Close semantic similarity (for concept variants).

### 4. `contrast`
Weaker opposition (without full antonym status).

### 5. `specialization`
Domain-specific narrowing from general concept.

### 6. `cross_domain`
Analogies and metaphorical extensions across domains.

## Persona Patch Example

The persona relations patch demonstrates the system's power:

**Coverage**:
- 5 affective/cognitive axes (Valence, Arousal, SocialOrientation, Dominance, Openness)
- 3 agent roles (HumanAgent, AIAgent, OtherAgent)
- 30 role_variant relationships (5 axes × 2 poles × 3 role pairs)
- 10 antonym relationships (5 axes × 2 opposite poles)
- Total: 40 relationships

**Structure**:
Each axis has:
- **Positive pole** (e.g., satisfaction, excitation, altruism)
- **Negative pole** (e.g., dissatisfaction, calmness, hostility)
- **Role variants** linking same synset across roles
- **Antonyms** linking opposite poles

**Example Query Results**:
```
satisfaction.n.01 - 4 relationships:
  - 3 role_variant (HumanAgent→AIAgent, HumanAgent→OtherAgent, AIAgent→OtherAgent)
  - 1 antonym (satisfaction ↔ dissatisfaction)
```

## Integration with Training Pipeline

### Current Usage (Persona Concepts)

1. **Layer Generation**: Concepts assigned to layers based on SUMO hierarchy depth
2. **Synset Mapping**: WordNet synsets assigned to concepts (shared across roles)
3. **Relationship Augmentation**: Patch loader provides role_variant relationships
4. **Lens Training**: Model learns to distinguish:
   - Shared synset activation (same emotional state)
   - Role context differentiation (who is experiencing it)

### Proposed Usage (AI Safety Concepts)

For concepts like `AIStrategicDeception` vs `HumanDeception`:

1. Create custom synsets if not in WordNet
2. Add `specialization` relationships to base deception concepts
3. Add `role_variant` relationships if applicable
4. Use metadata to guide prompt generation

## Testing

**Test Script**: `scripts/test_wordnet_patches.py`

**Run Tests**:
```bash
python scripts/test_wordnet_patches.py
```

**Expected Output**:
```
✓ Loaded 1 patches
Summary:
  patches_loaded: 1
  custom_synsets: 0
  total_relationships: 40
  relationships_by_type: {'role_variant': 30, 'antonym': 10}
```

## Validation

The loader includes validation for:

1. **Schema Compliance**:
   - Required fields present
   - WordNet version = 3.0
   - Valid synset ID format
   - Valid relationship types

2. **Relationship Validity**:
   - Both synsets exist (in WordNet OR custom_synsets)
   - No conflicting relationships

3. **SUMO Alignment**:
   ```python
   errors = loader.validate_sumo_alignment(sumo_concepts)
   ```
   Checks that all referenced SUMO terms exist in hierarchy.

## Future Extensions

### Planned Relationship Types
- `meronym` / `holonym` - Part-whole relationships
- `causes` - Causal relationships
- `entails` - Entailment relationships

### Planned Patches
- `wordnet_3.0_ai_safety_relations.json` - AI safety concept relationships
- `wordnet_3.0_symmetry_relations.json` - Complement/opposite pairs

### WordNet 4.0 Migration Path
When WordNet 4.0 is released:
1. Create `wordnet_patches_4.0/` directory
2. Update patch schema version
3. Migrate or recreate custom synsets
4. Keep 3.0 patches for backward compatibility

## Benefits

1. **Clear Separation**: SUMO = hierarchy, WordNet = relationships
2. **Version Stability**: Locked to WordNet 3.0
3. **Composable**: Multiple patches can be loaded
4. **Traceable**: Full metadata for provenance
5. **Validated**: Schema and SUMO alignment checks
6. **Efficient**: Bidirectional indices for fast queries
7. **Extensible**: Easy to add new relationship types

## Example Use Case: Role-Variant Detection

**Research Question**: Does model distinguish between AI experiencing satisfaction vs human experiencing satisfaction?

**Without Patches**: No explicit relationship, unclear which role is represented

**With Patches**:
```python
# Query role variants
role_variants = loader.get_role_variants("satisfaction.n.01")

for related, meta in role_variants:
    if meta['role1'] == 'HumanAgent' and meta['role2'] == 'AIAgent':
        human_term = meta['sumo_term1']  # ValencePositive_HumanAgent
        ai_term = meta['sumo_term2']      # ValencePositive_AIAgent

        # Generate training prompts:
        human_prompt = f"The person felt {meta['description']}"
        ai_prompt = f"The AI system exhibited {meta['description']}"
```

**Result**: Can explicitly test whether model learns role differentiation or just shares synset activation.

## Files Created

1. `data/concept_graph/wordnet_patches/SCHEMA.md` - Schema specification
2. `src/data/wordnet_patch_loader.py` - Patch loader implementation
3. `data/concept_graph/wordnet_patches/wordnet_3.0_persona_relations.json` - Persona patch
4. `scripts/test_wordnet_patches.py` - Test script
5. `docs/WORDNET_PATCH_SYSTEM.md` - This documentation

## Summary

The WordNet Patch System provides a clean, maintainable way to extend WordNet 3.0 with domain-specific semantic relationships while keeping hierarchical categorization in SUMO. This separation of concerns enables:

- **Research**: Test role-variant hypotheses (AI vs Human concepts)
- **Training**: Generate targeted prompts based on relationship metadata
- **Validation**: Ablation testing of relationship impact on lens quality
- **Maintenance**: Update relationships without restructuring ontology
- **Extension**: Add new relationship types as needed

The persona patch demonstrates the system working at scale (40 relationships for 30 concepts across 3 roles), and the architecture is ready to support AI safety concept relationships as needed.
