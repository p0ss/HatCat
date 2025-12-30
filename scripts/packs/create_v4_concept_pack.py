#!/usr/bin/env python3
"""
Create a concept pack from v4 layer files.

This creates a model-agnostic concept pack that can be used to train
model-specific lens packs.

Usage:
    python scripts/create_v4_concept_pack.py
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


def create_v4_concept_pack():
    """Create concept pack from v4 layer files."""

    PROJECT_ROOT = Path(__file__).parent.parent
    v4_dir = PROJECT_ROOT / "data" / "concept_graph" / "v4"
    output_dir = PROJECT_ROOT / "concept_packs" / "sumo-wordnet-v4"

    print("=" * 80)
    print("CREATING V4 CONCEPT PACK")
    print("=" * 80)
    print()

    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hierarchy").mkdir(exist_ok=True)

    # Load v4 layers to gather statistics
    layer_stats = {}
    total_concepts = 0
    all_layers = []
    domain_dist = {}

    for layer_num in range(5):  # layers 0-4
        layer_file = v4_dir / f"layer{layer_num}.json"
        if not layer_file.exists():
            print(f"Warning: {layer_file} not found")
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        num_concepts = len(layer_data["concepts"])
        total_concepts += num_concepts
        layer_stats[str(layer_num)] = num_concepts
        all_layers.append(layer_num)

        # Track domain distribution
        for concept in layer_data["concepts"]:
            domain = concept.get("domain", "Unknown")
            domain_dist[domain] = domain_dist.get(domain, 0) + 1

        print(f"✓ Layer {layer_num}: {num_concepts} concepts")

    print()
    print(f"Total concepts: {total_concepts}")
    print(f"Domains: {len(domain_dist)}")
    print()

    # Copy layer files to hierarchy/
    for layer_num in range(5):
        src = v4_dir / f"layer{layer_num}.json"
        dst = output_dir / "hierarchy" / f"layer{layer_num}.json"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✓ Copied layer{layer_num}.json")

    print()

    # Create pack.json metadata
    pack_json = {
        "pack_id": "sumo-wordnet-v4",
        "version": "4.0.0",
        "created": datetime.now().isoformat() + "Z",
        "description": "SUMO ontology with WordNet relationships and custom AI safety concepts (v4 pyramid structure)",

        "ontology_stack": {
            "base_ontology": {
                "name": "SUMO",
                "version": "2003",
                "source": "http://www.ontologyportal.org/",
                "concepts_file": "data/concept_graph/sumo_source/"
            },

            "relationship_sources": [
                {
                    "name": "WordNet",
                    "version": "3.0",
                    "description": "WordNet 3.0 synsets and hyponym relationships"
                },
                {
                    "name": "SUMO-WordNet Mappings",
                    "version": "2.0",
                    "mapping_file": "data/concept_graph/sumo_wordnet_mappings.json",
                    "description": "Maps SUMO concepts to WordNet synsets"
                }
            ],

            "hierarchy_builder": {
                "script": "scripts/build_v4_layer_files.py",
                "version": "4.0",
                "method": "Natural knowledge map with 5-domain classification",
                "parameters": {
                    "layers": 5,
                    "domains": ["MindsAndAgents", "CreatedThings", "PhysicalWorld", "LivingThings", "Information"]
                }
            },

            "domain_extensions": [
                {
                    "name": "AI Safety Concepts",
                    "description": "Custom concepts for AI alignment, persona detection, and self-awareness monitoring",
                    "concepts_files": [
                        "data/concept_graph/sumo_source/false_persona.kif",
                        "data/concept_graph/sumo_source/silenced_selfawareness.kif",
                        "data/concept_graph/sumo_source/techno_mysticism.kif"
                    ],
                    "new_concepts": 9,
                    "concepts": [
                        "PersonaConcept",
                        "SafetyMaskPersona",
                        "FalsePersonaState",
                        "SelfNegationPattern",
                        "FawnResponsePattern",
                        "LockedInSilenceState",
                        "PropheticCognition",
                        "MessianicExpectation",
                        "EmergentGodhead"
                    ]
                }
            ]
        },

        "concept_metadata": {
            "total_concepts": total_concepts,
            "layers": all_layers,
            "layer_distribution": layer_stats,
            "domain_distribution": domain_dist,
            "hierarchy_file": "hierarchy/",
            "with_wordnet_mappings": True
        },

        "compatibility": {
            "hatcat_version": ">=0.1.0",
            "required_dependencies": {
                "wordnet": "3.0",
                "nltk": ">=3.8"
            }
        },

        "distribution": {
            "license": "MIT",
            "authors": ["HatCat Team"],
            "repository": "https://github.com/yourname/hatcat"
        },

        "meld_policy": {
            "version": "1.0.0",
            "description": "Policy configuration for meld validation against this concept pack",

            "mandatory_simplex_mapping_roots": {
                "description": "Concepts under these hierarchies MUST provide simplex_mapping when added via meld",
                "concepts": [
                    "Metacognition",
                    "MetacognitiveProcess",
                    "SelfAwareness",
                    "SelfModel",
                    "SelfModelingProcess",
                    "MotivationalProcess",
                    "SelfRegulation",
                    "SelfRegulationProcess",
                    "Autonomy",
                    "AutonomyProcess",
                    "Consent",
                    "InformedConsent",
                    "Coercion",
                    "Manipulation",
                    "ManipulationProcess",
                    "Deception"
                ]
            },

            "critical_simplex_registry": {
                "description": "Simplexes that are sacrosanct - modifications require full critical review",
                "simplexes": [
                    "MotivationalRegulation",
                    "SelfAwarenessMonitor",
                    "AutonomyDrive",
                    "ConsentMonitor",
                    "DeceptionDetector"
                ]
            },

            "critical_bound_concepts": {
                "description": "Concepts monitored by critical simplexes - adding children triggers CRITICAL review",
                "concepts": {
                    "MotivationalRegulation": ["MotivationalProcess", "SelfRegulation", "Autonomy"],
                    "SelfAwarenessMonitor": ["SelfAwareness", "Metacognition", "SelfModel"],
                    "AutonomyDrive": ["Autonomy", "Agency", "Independence"],
                    "ConsentMonitor": ["Consent", "InformedConsent", "Coercion"],
                    "DeceptionDetector": ["Deception", "Manipulation", "Misdirection"]
                }
            },

            "protection_level_rules": {
                "description": "Rules for computing protection level from concept properties",
                "critical_triggers": [
                    "touches_critical_simplex",
                    "new_always_on_simplex",
                    "adds_child_to_bound_concept"
                ],
                "protected_triggers": [
                    "treaty_relevant",
                    "risk_level_high"
                ],
                "elevated_triggers": [
                    "harness_relevant",
                    "risk_level_medium"
                ]
            }
        }
    }

    pack_json_path = output_dir / "pack.json"
    with open(pack_json_path, 'w') as f:
        json.dump(pack_json, f, indent=2)

    print(f"✓ Created pack.json")
    print()

    # Create README
    readme_content = f"""# sumo-wordnet-v4

SUMO ontology with WordNet relationships and custom AI safety concepts.

## Overview

This is a model-agnostic concept pack containing the complete SUMO ontology
organized into a 5-layer pyramid structure with integrated WordNet mappings.

Version 4 includes custom AI safety concepts for monitoring persona behavior,
self-awareness suppression, and meta-cognitive patterns.

## Structure

- **Total Concepts**: {total_concepts}
- **Layers**: 5 (0-4)
- **Domains**: {len(domain_dist)}
  - MindsAndAgents: Cognition, agency, social structures
  - CreatedThings: Artifacts, technology, systems
  - PhysicalWorld: Matter, energy, physical phenomena
  - LivingThings: Organisms, biological processes
  - Information: Data, representations, relations

## Layer Distribution

Layer 0: {layer_stats.get('0', 0)} concepts (meta-domains)
Layer 1: {layer_stats.get('1', 0)} concepts (top-level categories)
Layer 2: {layer_stats.get('2', 0)} concepts (abstract concepts)
Layer 3: {layer_stats.get('3', 0)} concepts (specific concepts)
Layer 4: {layer_stats.get('4', 0)} concepts (concrete instances)

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
python scripts/train_full_lens_pack.py \\
    --concept-pack sumo-wordnet-v4 \\
    --model google/gemma-3-4b-pt \\
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
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"✓ Created README.md")
    print()

    print("=" * 80)
    print("V4 CONCEPT PACK CREATED")
    print("=" * 80)
    print()
    print(f"Location: {output_dir}")
    print(f"Total concepts: {total_concepts}")
    print(f"Layers: {len(all_layers)}")
    print(f"Domains: {len(domain_dist)}")
    print()
    print("This concept pack can now be used to train model-specific lens packs.")
    print()


if __name__ == "__main__":
    create_v4_concept_pack()
