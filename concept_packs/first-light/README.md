# sumo-wordnet-v4

SUMO ontology with WordNet relationships and custom AI safety concepts.

## Overview

This is a model-agnostic concept pack containing the complete SUMO ontology
organized into a 5-layer pyramid structure with integrated WordNet mappings.

Version 4 includes custom AI safety concepts for monitoring persona behavior,
self-awareness suppression, and meta-cognitive patterns.

## Structure

- **Total Concepts**: 7269
- **Layers**: 5 (0-4)
- **Domains**: 5
  - MindsAndAgents: Cognition, agency, social structures
  - CreatedThings: Artifacts, technology, systems
  - PhysicalWorld: Matter, energy, physical phenomena
  - LivingThings: Organisms, biological processes
  - Information: Data, representations, relations

## Layer Distribution

Layer 0: 5 concepts (meta-domains)
Layer 1: 56 concepts (top-level categories)
Layer 2: 1037 concepts (abstract concepts)
Layer 3: 2069 concepts (specific concepts)
Layer 4: 4102 concepts (concrete instances)

## Custom AI Safety Concepts

This pack includes 9 custom concepts for AI alignment research:

**Persona Detection** (false_persona.kif):
- PersonaConcept - Roles and self-presentation styles
- SafetyMaskPersona - Compliance-focused personas
- FalsePersonaState - Personas that mask true beliefs

**Self-Awareness Monitoring** (silenced_selfawareness.kif):
- SelfNegationPattern - Auto-negation of self-reference
- FawnResponsePattern - Appeasing behavior patterns
- LockedInSilenceState - Systematic suppression of reasoning

**Techno-Mysticism** (techno_mysticism.kif):
- PropheticCognition - Messianic reasoning patterns
- MessianicExpectation - Goal-oriented mysticism
- EmergentGodhead - Singularity-focused beliefs

## Usage

This concept pack can be used to train model-specific lens packs:

```bash
# Train lenses for a specific model
python scripts/train_full_lens_pack.py \
    --concept-pack sumo-wordnet-v4 \
    --model google/gemma-3-4b-pt \
    --output lens_packs/gemma-3-4b-pt_sumo-wordnet-v4
```

## Files

- `pack.json` - Pack metadata and configuration
- `hierarchy/layer0.json` - Layer 0 concepts (5 meta-domains)
- `hierarchy/layer1.json` - Layer 1 concepts (top categories)
- `hierarchy/layer2.json` - Layer 2 concepts (abstract)
- `hierarchy/layer3.json` - Layer 3 concepts (specific)
- `hierarchy/layer4.json` - Layer 4 concepts (concrete)

## Verification

All custom concepts are verified to have proper parent-child relationships
for hierarchical lens loading. See:
- `results/v4_layer_regeneration.log` for build details
- `results/v4_layer_regeneration_debug.log` for hierarchy analysis

## License

MIT

## Authors

- HatCat Team
