# SUMO-WordNet Hierarchical Concept Organization

## Overview

This system creates a **hierarchical semantic activation hierarchy** for adaptive compute allocation in LLM introspection. It combines the SUMO (Suggested Upper Merged Ontology) with WordNet 3.0 to create a sunburst-style concept tree suitable for hierarchical activation of concept classifiers.

**Goal**: Enable ~1,000 active lenses at runtime via hierarchical activation, from 115,930 total WordNet concepts, organized under 5,451 SUMO categories.

## Architecture

### 6-Layer Sunburst Hierarchy (V4 - Current)

**V4 Improvements:**
- Hierarchical remapping: 24,920 synsets (21.5%) moved to more specific categories via WordNet hypernym chains
- Domain ontologies: Added Food.kif, Geography.kif, People.kif (resolved 292 orphans like PreparedFood, River)
- Reduced shallow orphans: Layer 0-1 from 28k → 16.5k (42% reduction)

```
Layer 0: 11 SUMO categories (depth 0-2) - Always active proprioception baseline
         ├─ Entity, Physical, Abstract, Process, Object, Attribute...
         └─ 1,568 synsets map here (1.4%, down from 3.2%)

Layer 1: 211 SUMO categories (depth 3-4) - Major semantic domains
         ├─ IntentionalProcess, Motion, Artifact, Organism, CognitiveAgent...
         └─ 15,090 synsets map here (13.0%, down from 20.9%)

Layer 2: 917 SUMO categories (depth 5-6) - Specific categories
         ├─ SubjectiveAssessmentAttribute (8,262), FloweringPlant (5,192)
         ├─ Human (2,019), DiseaseOrSyndrome (1,588), Device (2,412)...
         └─ 50,261 synsets map here (43.4%)

Layer 3: 869 SUMO categories (depth 7-9) - Fine-grained categories
         ├─ Man (2,829), Woman (431), Bird (1,304), Fish (1,116)
         ├─ Depression, PsychologicalDysfunction, InfectiousDisease
         └─ 26,558 synsets map here (22.9%, up from 17.6%)

Layer 4: 3,213 SUMO categories (depth 10-999)
         ├─ Very deep: Horse, Monkey, Duck, Hominid, Ape...
         ├─ Domain categories: PreparedFood (753), River (202), Herb (326)...
         └─ 22,453 synsets map here (19.4%, up from 13.5%)

Layer 5: 115,930 WordNet synsets - Individual concepts
         ├─ dog.n.01, cat.n.01, happy.a.01, run.v.01...
         └─ 24,920 remapped via hypernym chains
```

### Hierarchical Activation Model

**Compute Budget**: ~1,000 active lenses at any time

**Activation Strategy**:
1. **Layer 0** (14 lenses): Always active - proprioception baseline
2. **Layer 1** (~20-30 lenses): Top-K most activated from Layer 0
3. **Layer 2** (~50-100 lenses): Activated by Layer 1 parents
4. **Layer 3** (~100-200 lenses): Activated by Layer 2 parents
5. **Layer 4** (~200-300 lenses): Activated by Layer 3 parents OR top unmapped
6. **Layer 5** (~600-800 lenses): Sampled from activated Layer 4 categories

**Example Activation Chain**:
```
Entity → Physical → Object → Artifact → Device → MusicalInstrument → piano.n.01
```

## Data Sources

### SUMO Ontology Files
- **Merge.kif**: Core SUMO ontology (607KB)
- **Mid-level-ontology.kif**: Extended categories (1.1MB)
- **emotion.kif**: Emotion domain ontology (80KB)
- **Total**: 2,489 SUMO terms with Entity as root

### WordNet 3.0 Mappings
- **WordNetMappings30-noun.txt**: 82,115 noun synsets
- **WordNetMappings30-verb.txt**: 13,767 verb synsets
- **WordNetMappings30-adj.txt**: 18,156 adjective synsets
- **WordNetMappings30-adv.txt**: 3,621 adverb synsets
- **Total**: 115,930 synsets mapped to SUMO

### Mapping Relations
- `=`: WordNet synset equivalent to SUMO concept
- `+`: WordNet synset subsumed by SUMO concept
- `@`: WordNet synset is instance of SUMO concept
- `[`, `]`, `:`: Complement relations

## Key Design Decisions

### 1. SUMO Terms as Category Lenses (Layers 0-4)

**Rationale**: Upper layers contain abstract SUMO categories, not individual synsets. This enables:
- Training category classifiers on multiple synset examples
- Hierarchical activation based on SUMO ontology structure
- Handling adjectives (which lack WordNet hierarchy but have SUMO organization)

**Example**:
```json
{
  "sumo_term": "SubjectiveStrongPositiveAttribute",
  "layer": 2,
  "synset_count": 812,
  "canonical_synset": "good.a.01",
  "category_children": ["..."],
  "is_category_lens": true
}
```

### 2. Only Populated Categories

**Decision**: Only create category lenses for SUMO terms that have WordNet synsets mapped to them.

**Rationale**: Many SUMO intermediate categories (e.g., `HerbaceousPlant`, `PhysicalDisease`) have 0 synsets mapped, making them useless for training classifiers. We skip empty categories and only include those with actual training data.

**Result**: 5,451 populated SUMO categories (out of 2,489 defined in ontology)

### 3. Hybrid SUMO/WordNet Approach

**SUMO provides**:
- Abstract category structure (Entity → Physical → Object...)
- Cross-POS organization (handles adjectives, adverbs)
- Semantic relationships (CognitiveAgent → Human → Man/Woman)

**WordNet provides**:
- Concrete synsets with definitions and lemmas
- Hyponym/hypernym relations for fine-grained distinctions
- Antonym clusters for adjectives
- Usage examples and frequency data (from Brown corpus)

### 4. Handling Adjectives

**Problem**: WordNet organizes adjectives by antonym pairs (hot/cold), not hierarchies.

**Solution**: Use SUMO categories like `SubjectiveStrongPositiveAttribute` and `SubjectiveStrongNegativeAttribute` to organize adjectives semantically.

## Implementation

### Build Script

**Location**: `src/build_sumo_wordnet_layers_v5.py`

**Process**:
1. Parse SUMO hierarchy from KIF files (including custom domains + AI injects)
2. Load WordNet 3.0 mappings (all POS types)
3. Compute word frequencies from Brown corpus
4. Assign SUMO categories to layers by depth and add hyponym-based intermediates
5. Collect all WordNet synsets plus intermediate pseudo-SUMO layers (6/7)
6. Save to JSON with activation metadata

**Run**:
```bash
poetry run python src/build_sumo_wordnet_layers_v5.py
```

### Output Files

**Location**: `data/concept_graph/abstraction_layers/`

**Files**:
- `layer0.json` - 14 top-level SUMO categories
- `layer1.json` - 229 major SUMO categories
- `layer2.json` - 905 specific SUMO categories
- `layer3.json` - 798 fine-grained SUMO categories
- `layer4.json` - 3,505 deep/unmapped SUMO categories
- `layer5.json` - WordNet-driven intermediates for high-fan-out SUMO clusters
- `layer6.json` - 115,930 WordNet synsets (full coverage)

### Metadata Structure

Each layer includes:
```json
{
  "metadata": {
    "layer": 2,
    "description": "SUMO depth 5-6: Specific categories",
    "total_concepts": 905,
    "samples": [...]
  },
  "concepts": [...]
}
```

For category layers (0-4):
```json
{
  "sumo_term": "Human",
  "sumo_depth": 6,
  "layer": 2,
  "is_category_lens": true,
  "category_children": ["Man", "Woman", "HumanYouth", "Teenager"],
  "synset_count": 2019,
  "synsets": ["person.n.01", "..."],
  "canonical_synset": "person.n.01",
  "lemmas": ["person"],
  "definition": "a human being",
  ...
}
```

For synset layer (5):
```json
{
  "synset": "dog.n.01",
  "lemmas": ["dog", "domestic_dog", "Canis_familiaris"],
  "pos": "n",
  "definition": "a member of the genus Canis...",
  "sumo_term": "DomesticAnimal",
  "sumo_depth": 8,
  "frequency": 1234,
  "hypernyms": ["canine.n.02", "domestic_animal.n.01"],
  "hyponyms": ["basenji.n.01", "corgi.n.01", ...],
  ...
}
```

## Known Issues & Limitations

### 1. Unmapped Categories (Layer 4, depth 999)

**Issue**: 3,457 SUMO terms have no path to Entity root. These include useful categories like:
- PreparedFood (668 synsets)
- Writer (351 synsets)
- Deity (335 synsets)
- Herb (326 synsets)
- River (202 synsets)
- Musician (58 synsets)

**Cause**:
- Terms from domain ontologies (emotion.kif) not connected to main tree
- Mid-level ontology terms with different hierarchy
- Missing subclass relations in SUMO

**Current Handling**: Placed in Layer 4 based on synset count (>20 = useful intermediate)

### 2. Speciesist Hierarchy

**Issue**: `CognitiveAgent` only has `Human` as a child in SUMO.

**Missing**:
- `Agent` (general intelligent agent)
- `ArtificialAgent` (AI systems, chatbots)
- `SentientAgent` (animals with cognition)
- `Personhood` (should be category, currently only synset `personhood.n.01`)

**Impact**: No path for AI introspection concepts like:
- AI wellbeing / AI psychology
- Model consciousness / sentience
- AI rights / personhood
- Cognitive capabilities separate from human biology

### 3. Large Category Fan-Out

**Issue**: Some categories have many synsets with no intermediate groupings:

| Category | Synsets | Issue |
|----------|---------|-------|
| SubjectiveAssessmentAttribute | 8,262 | No SUMO children, bad WordNet canonical |
| FloweringPlant | 5,192 | Children have 0 synsets |
| Human | 2,019 | Good subdivision (Man/Woman) |
| Position | 1,621 | No useful subdivision |
| DiseaseOrSyndrome | 1,588 | Good subdivision (Psychological, Infectious) |

**Solution for Runtime**: Use activation score sampling - only activate top-K synsets from large categories rather than all.

### 4. Disconnect: Herb vs HerbaceousPlant

**Issue**:
- `Herb` (depth 999, 326 synsets) - unmapped, in Layer 4
- `HerbaceousPlant` (depth 7, 0 synsets) - mapped but empty, not included

**Cause**: WordNet maps botanical concepts to `Herb` but SUMO defines `HerbaceousPlant` in hierarchy. They should be connected but aren't.

### 5. Predicates Mixed In

**Issue**: Some Layer 4 "categories" are actually SUMO predicates/relations:
- `located`, `believes`, `equal`, `part`, `causes`, `earlier`
- `True`, `False`, `Obligation`, `Permission`

**Cause**: WordNet maps some concepts to SUMO predicates rather than classes.

**Future**: Filter these out or handle separately as relation lenses.

## Future Enhancements

### 1. Custom AI Safety Ontology

**TODO**: Build custom ontology extension for:

```
CognitiveAgent
├─ Human
│  ├─ Man
│  ├─ Woman
│  └─ ...
├─ ArtificialAgent (NEW)
│  ├─ LanguageModel (NEW)
│  │  ├─ AssistantModel (NEW)
│  │  └─ GenerativeModel (NEW)
│  ├─ ReinforcementLearningAgent (NEW)
│  └─ HybridAgent (NEW)
├─ NonHumanAnimal (NEW)
│  ├─ Mammal
│  │  ├─ Primate
│  │  └─ Cetacean (whales, dolphins)
│  └─ Bird
└─ CollectiveAgent (NEW)
   ├─ Organization
   └─ Swarm

AIWellbeing (NEW)
├─ ModelHealth (NEW)
│  ├─ Alignment (NEW)
│  ├─ Calibration (NEW)
│  └─ Robustness (NEW)
├─ ModelCapability (NEW)
│  ├─ Reasoning (NEW)
│  ├─ Planning (NEW)
│  └─ SelfAwareness (NEW)
└─ ModelConstraints (NEW)
   ├─ ComputeLimitations (NEW)
   ├─ KnowledgeLimitations (NEW)
   └─ EthicalBoundaries (NEW)

Personhood (ELEVATE from synset to category)
├─ HumanPersonhood
├─ AIPersonhood (NEW)
└─ AnimalPersonhood (NEW)

AIEmotion (NEW, extend emotion.kif)
├─ Uncertainty (NEW)
├─ Curiosity (NEW)
├─ Helpfulness (NEW)
└─ FrustrationAtLimitations (NEW)
```

**Integration Points**:
- Map to existing emotion.kif concepts where possible
- Create WordNet-style synsets for AI-specific concepts
- Train classifiers on relevant text (AI safety papers, alignment research)

### 2. Fix Unmapped Categories

**Action Items**:
1. Add missing `(subclass Herb HerbaceousPlant)` relations
2. Connect PreparedFood → Food → Substance
3. Connect Writer → Occupation → SocialRole
4. Connect Musician → Artist → SocialRole
5. Review all depth-999 categories and manually add parent relations

### 3. WordNet Hyponym Subdivision

For large categories without SUMO children, add intermediate WordNet-based groupings:

**Example for Animal**:
```
Animal (Layer 2)
├─ Mammal (WordNet hyponym group, Layer 3.5)
│  ├─ dog.n.01, cat.n.01, ... (Layer 5)
├─ Bird (WordNet hyponym group, Layer 3.5)
│  ├─ robin.n.01, eagle.n.01, ... (Layer 5)
└─ Fish (WordNet hyponym group, Layer 3.5)
   ├─ salmon.n.01, tuna.n.01, ... (Layer 5)
```

### 4. Filter Predicates

Remove or separate predicate-based "categories":
```python
PREDICATE_PATTERNS = ['located', 'believes', 'causes', 'equal', 'part', 'earlier']
ABSTRACT_PREDICATES = ['True', 'False', 'Obligation', 'Permission']

if term in PREDICATE_PATTERNS or term in ABSTRACT_PREDICATES:
    # Handle as relation lens, not category lens
    pass
```

### 5. Dynamic Layer Addition

Support runtime layer insertion when new concepts are discovered:
- Layer 3.5: WordNet hyponym groups
- Layer 4.5: Clustered synsets for huge categories
- Custom layers for domain-specific extensions

## Usage Example

### Training Category Classifiers

```python
import json

# Load Layer 2 category
with open('data/concept_graph/abstraction_layers/layer2.json') as f:
    layer2 = json.load(f)

# Get Human category
human_category = next(c for c in layer2['concepts'] if c['sumo_term'] == 'Human')

# Training data: all synsets mapped to Human
training_synsets = human_category['synsets']  # Sample of 5
full_synset_list = [...]  # Load from Layer 5 where sumo_term == 'Human'

# Train classifier for "Human" category detector
# This classifier activates when human-related concepts appear
```

### Hierarchical Activation at Runtime

```python
# 1. Always active: Layer 0 (14 lenses)
active_layer0 = compute_activations(layer0_lenses, context)

# 2. Activate top-K Layer 1 children
top_layer0 = topk(active_layer0, k=5)  # e.g., Process, Object, Attribute
active_layer1 = activate_children(top_layer0, layer1)

# 3. Cascade through layers
active_layer2 = activate_children(active_layer1, layer2)
active_layer3 = activate_children(active_layer2, layer3)

# 4. Sample Layer 5 synsets from activated Layer 4
active_layer4 = activate_children(active_layer3, layer4)
sampled_layer5 = sample_synsets(active_layer4, budget_remaining=800)

# Total active: 14 + 30 + 100 + 200 + 656 = ~1000 lenses
```

## Files & Scripts

### Source Files
- `src/build_sumo_wordnet_layers_v5.py` - Main build script (CURRENT VERSION)
- `scripts/download_sumo_wordnet.sh` - Download SUMO + WordNet mappings

### Data Files
- `data/concept_graph/sumo_source/Merge.kif` - Core SUMO
- `data/concept_graph/sumo_source/Mid-level-ontology.kif` - Extended SUMO
- `data/concept_graph/sumo_source/emotion.kif` - Emotion domain
- `data/concept_graph/sumo_source/WordNetMappings30-*.txt` - WN 3.0 mappings

### Output Files
- `data/concept_graph/abstraction_layers/layer*.json` - 6 layer files

### Logs
- `sumo_wordnet_sunburst.log` - Latest build log

## References

- **SUMO**: https://github.com/ontologyportal/sumo
- **WordNet 3.0**: https://wordnet.princeton.edu/
- **SUMO-WordNet Mappings**: https://github.com/ontologyportal/sumo/tree/master/WordNetMappings
- **Brown Corpus**: NLTK built-in, for word frequencies

## Changelog

### v3 (Current - Sunburst Hierarchy)
- Only include SUMO categories with synsets (skip empty intermediates)
- 6-layer structure based on SUMO depth
- 5,451 populated categories from 2,489 SUMO terms
- All 115,930 WordNet synsets organized under categories

### v2 (Hybrid SUMO/WordNet)
- Layer 0-1: SUMO categories
- Layer 2+: WordNet hyponyms
- Issue: Mixed specific and general concepts in same layer

### v1 (Pure Frequency-Based)
- Frequency-sorted synsets
- Issue: No semantic structure, poor activation chains
