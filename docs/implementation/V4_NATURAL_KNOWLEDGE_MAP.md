# V4 Natural Knowledge Map

## Overview

The V4 concept hierarchy reorganizes all 7,097 concepts from SUMO + custom safety domains into a semantically coherent **natural knowledge map** following information architecture principles.

## Key Changes from Previous Versions

### Problem with Previous Approaches
- **Depth-based layering** (V1-V3): Grouped concepts by distance from root, creating semantically incoherent layers
- Example failure: OpiumPoppy, DeceptionSignal, and Carnivore all appeared in Layer 2 despite having zero semantic similarity
- SUMO's philosophical ontology (Entity→Abstract/Physical) not practical for training

### V4 Solution: Natural Knowledge Domains
Reorganized into 5 natural knowledge domains inspired by "maps of knowledge" and information architecture:

1. **MindsAndAgents**: Cognition, agency, social structures, communication, mental processes
2. **CreatedThings**: Artifacts, technology, systems, tools, human-made constructs
3. **PhysicalWorld**: Matter, energy, forces, physical quantities, natural phenomena
4. **LivingThings**: Organisms, biological processes, ecosystems, life
5. **Information**: Data, representations, propositions, relations, abstract entities

## Architecture

### Pyramid Structure (Information Architecture Principle)
Follows the **Principle of Choices** for O(log N) search efficiency:

```
Layer 0:     5 domains (meta-concepts)          (0.1%)
Layer 1:    56 seed concepts                    (0.8%)
Layer 2:   989 category concepts               (13.9%)
Layer 3: 2,008 subcategory concepts            (28.3%)
Layer 4: 4,039 specific concepts               (56.9%)
---------------------------------------------------
Total:   7,097 concepts
```

### Domain Distribution

**Layer 0 (5 domains)**
- Equal distribution: 1 concept per domain (meta-level)

**Layer 1 (56 seed concepts)**
- CreatedThings: 15 (27%)
- MindsAndAgents: 13 (23%)
- Information: 12 (21%)
- PhysicalWorld: 9 (16%)
- LivingThings: 7 (13%)

**Layer 2 (989 concepts)**
- CreatedThings: 319 (32%)
- Information: 234 (24%)
- MindsAndAgents: 183 (19%)
- PhysicalWorld: 179 (18%)
- LivingThings: 74 (7%)

**Layer 3 (2,008 concepts)**
- CreatedThings: 620 (31%)
- PhysicalWorld: 412 (21%)
- MindsAndAgents: 399 (20%)
- Information: 376 (19%)
- LivingThings: 201 (10%)

**Layer 4 (4,039 concepts)**
- CreatedThings: 936 (23%)
- PhysicalWorld: 877 (22%)
- MindsAndAgents: 808 (20%)
- LivingThings: 792 (20%)
- Information: 626 (16%)

## Layer 1 Seed Concepts by Domain

### MindsAndAgents (13 concepts)
1. Agent
2. AlignmentProcess
3. ArtificialAgent
4. AutonomousAgent
5. Believing
6. CognitiveProcess
7. Communication
8. Deception
9. IntentionalProcess
10. Organization
11. Perception
12. Reasoning
13. SocialInteraction

### CreatedThings (15 concepts)
1. AIArtifact
2. ArtWork
3. Artifact
4. AttackPattern
5. Building
6. ComputerProgram
7. CyberOperation
8. DataCenter
9. Device
10. Game
11. Machine
12. Tool
13. TransformerModel
14. Vehicle
15. Weapon

### PhysicalWorld (9 concepts)
1. ConstantQuantity
2. FieldOfStudy
3. GeographicArea
4. Motion
5. PhysicalQuantity
6. Quantity
7. Region
8. Substance
9. TimeInterval

### LivingThings (7 concepts)
1. AnatomicalStructure
2. Animal
3. BiologicalProcess
4. BodyPart
5. Ecosystem
6. Organism
7. Plant

### Information (12 concepts)
1. AbstractEntity
2. Concept
3. ContentBearingPhysical
4. Formula
5. InternalAttribute
6. LinguisticExpression
7. Proposition
8. Relation
9. RelationalAttribute
10. Sentence
11. SetOrClass
12. Text

## Top Categories by Fan-Out

The largest semantic categories (by child count):

1. Device (L1, CreatedThings) - 133 children
2. ElementalSubstance (L3, PhysicalWorld) - 113 children
3. ElectricDevice (L2, CreatedThings) - 84 children
4. Artifact (L1, CreatedThings) - 81 children
5. StationaryArtifact (L2, CreatedThings) - 65 children
6. RelationalAttribute (L1, Information) - 56 children
7. AnimalAnatomicalStructure (L2, LivingThings) - 55 children
8. IntentionalProcess (L1, MindsAndAgents) - 53 children
9. AbstractEntity (L1, Information) - 50 children
10. BodyPart (L1, LivingThings) - 49 children

## File Structure

Layer JSON files located at: `data/concept_graph/v4/`

Each layer file contains:
- `layer`: Layer number (0-4)
- `concepts`: Array of concept objects
- `summary`: Metadata about the layer

Each concept entry includes:
- `sumo_term`: Concept name
- `layer`: Layer assignment
- `domain`: Knowledge domain
- `parent_concepts`: Direct parents in ontology
- `category_children`: Direct children (sample for large categories)
- `child_count`: Total number of children
- `sumo_definition`: SUMO documentation (if available)
- `is_pseudo_sumo`: Whether this is a meta-concept (Layer 0 only)
- `is_category_lens`: Whether this is trainable as a category

## Building the V4 Hierarchy

The hierarchy is built from:
1. **Natural knowledge map** (`data/concept_graph/natural_knowledge_map.json`)
   - Domain assignments for all concepts
   - Layer assignments based on semantic distance from domain roots
   - Propagated via BFS from manually assigned seed concepts

2. **KIF files** (`data/concept_graph/sumo_source/*.kif`)
   - SUMO definitions and parent-child relationships
   - Custom safety concepts from 6 domain files

### Rebuild Script
To regenerate V4 layer files:
```bash
poetry run python scripts/build_v4_layer_files.py
```

## Benefits for Lens Training

1. **Semantic Coherence**: Concepts in same layer share semantic similarity
2. **Domain Identification**: Layer 0 provides coarse domain filter
3. **Efficient Search**: O(log N) traversal via pyramid structure
4. **Trainable Categories**: Each layer represents trainable abstraction level
5. **Natural Organization**: Mirrors human cognitive categorization

## Next Steps

1. Train lenses on V4 hierarchy (estimated 13 hours for all 7,097 concepts)
2. Generate WordNet synset mappings for concepts without coverage
3. Validate lens accuracy on safety-critical concepts
4. Create visualization of domain structure for UI

## References

- Natural knowledge map: `data/concept_graph/natural_knowledge_map.json`
- Hierarchy CSV: `data/concept_graph/natural_knowledge_hierarchy.csv`
- Builder script: `scripts/build_v4_layer_files.py`
- Original map builder: `scripts/build_natural_knowledge_map.py`
