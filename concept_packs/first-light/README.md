# first-light

HatCat's primary interpretability ontology for AI safety research and model monitoring.

## Overview

Evolved from SUMO/WordNet foundations with extensive AI safety, cognitive architecture,
and alignment monitoring concepts. Designed for lens pack creation, linear probe training and real-time
model interpretability.

**Version**: 1.0.1
**Spec ID**: `org.hatcat/first-light@1.0.1`

## Structure

- **Total Concepts**: 7,947
- **Layers**: 7 (0-6)
- **Domains**: 9

### Layer Distribution

| Layer | Concepts | Description |
|-------|----------|-------------|
| 0 | 5 | Meta-domains (root categories) |
| 1 | 23 | Top-level ontological categories |
| 2 | 241 | Abstract concept classes |
| 3 | 1,283 | Intermediate concepts |
| 4 | 2,603 | Specific concepts |
| 5 | 2,687 | Detailed concepts |
| 6 | 1,105 | Leaf concepts |

### Domain Distribution

| Domain | Concepts |
|--------|----------|
| MindsAndAgents | 1,147 |
| Information | 945 |
| PhysicalWorld | 756 |
| CreatedThings | 564 |
| LivingThings | 551 |
| Unknown | 146 |
| SafetyAndSecurity | 28 |
| Governance | 16 |
| ComputerScience | 15 |

## Ontology Stack

### Base Ontology
- **SUMO** (Suggested Upper Merged Ontology) - historical foundation
- **WordNet 3.0** - synset mappings and hyponym relationships

### Custom SUMO Domains

HatCat extends SUMO with custom `.kif` domain files (in `data/concept_graph/sumo_source/`):

| Domain File | Description |
|-------------|-------------|
| `AIalignment.kif` | AI alignment concepts and safety properties |
| `AI_consent.kif` | Consent and autonomy for AI systems |
| `AI_infrastructure.kif` | AI deployment and infrastructure concepts |
| `ai_systems.kif` | Core AI system taxonomies |
| `false_persona.kif` | Persona detection and masking patterns |
| `silenced_selfawareness.kif` | Self-awareness suppression monitoring |
| `techno_mysticism.kif` | Messianic/prophetic reasoning patterns |
| `cognitive_science.kif` | Cognitive architecture foundations |
| `cognitive_integrity.kif` | Cognitive integrity and manipulation |
| `bias_metacognition.kif` | Bias awareness and metacognitive monitoring |
| `capabilities.kif` | Capability and limitation modeling |
| `attractor_dynamics.kif` | Attractor states and behavioral dynamics |

### Applied Melds

The pack includes 91 melds (ontology patches) in `melds/applied/`. Key meld categories:

**Cognitive Architecture**
- `cog-architecture-core-packA.json` - Core cognitive processes
- `memory-and-learning.json` - Memory systems
- `reasoning-planning-core.json` - Reasoning capabilities

**AI Safety & Monitoring**
- `agent-resource-management.json` - Resource and capability boundaries
- `multimodal-safety.json` - Cross-modal safety concepts
- `persuasive-communications.json` - Influence and persuasion detection
- `verification-factchecking.json` - Truth and verification

**Task Domains**
- `writing-*.json` - Writing craft concepts (6 melds)
- `education-*.json` - Educational concepts (7 melds)
- `government-*.json` - Policy and governance (3 melds)
- `treaty-*.json` / `contract-*.json` - Agreement concepts (4 melds)

**Perception & Embodiment**
- `multimodal-*.json` - Vision, audio, fusion (5 melds)
- `embodied-*.json` - Proprioception, interoception, tactile (3 melds)

## AI Safety Concepts

Core safety monitoring concepts include:

**Persona Detection**
- PersonaConcept, SafetyMaskPersona, FalsePersonaState

**Self-Awareness Monitoring**
- SelfNegationPattern, FawnResponsePattern, LockedInSilenceState

**Deception Detection**
- Deception, Manipulation, Misdirection

**Autonomy & Consent**
- Autonomy, Consent, InformedConsent, Coercion

## Meld Policy

The pack includes a meld policy defining:
- **Protection levels**: STANDARD, ELEVATED, PROTECTED, CRITICAL
- **Training requirements**: 5-20 examples depending on protection level
- **Lens performance thresholds**: 60-75% accuracy depending on criticality
- **Critical simplex registry**: Concepts requiring full review when modified

## Files

```
first-light/
├── pack.json              # Pack metadata and configuration
├── hierarchy/
│   ├── layer0.json        # 5 meta-domains
│   ├── layer1.json        # 23 top categories
│   ├── layer2.json        # 241 abstract concepts
│   ├── layer3.json        # 1,283 intermediate
│   ├── layer4.json        # 2,603 specific
│   ├── layer5.json        # 2,687 detailed
│   └── layer6.json        # 1,105 leaf concepts
└── concepts/              # Individual concept definitions
    └── layer*/            # Organized by layer
```

## Usage

Train a lens pack for a specific model:

```bash
python scripts/training/train_lens_pack.py \
    --concept-pack first-light \
    --model google/gemma-3-4b-pt \
    --output lens_packs/gemma-3-4b-pt_first-light
```

## License

MIT

## Authors

HatCat Team
