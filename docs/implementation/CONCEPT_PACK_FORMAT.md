# Concept Pack Format Specification

**Version**: 2.0
**Date**: November 15, 2024
**Status**: ✅ Implemented

---

## Overview

A **Concept Pack** is a distributable bundle containing:
- Ontology definitions (SUMO concepts in KIF format)
- Semantic relationships (WordNet patches)
- Metadata and documentation

This allows domain experts to create and share ontology extensions without modifying the core SUMO hierarchy directly.

---

## Directory Structure

```
concept_packs/
  {pack_name}/
    pack.json                      # Metadata and manifest
    concepts.kif                   # SUMO hierarchy additions (optional)
    wordnet_patches/               # Semantic relationships (optional)
      {patch_name}.json
    layer_entries/                 # Pre-calculated layer JSON (optional)
      layer{N}.json
    README.md                      # Human-readable documentation
    LICENSE                        # License file
```

### File Descriptions

- **pack.json**: Required manifest with metadata, ontology stack, and file references
- **concepts.kif**: SUMO concept definitions (subclass, documentation, etc.)
- **wordnet_patches/**: WordNet relationship patches (see WordNet Patch System spec)
- **layer_entries/**: Pre-calculated layer assignments (optional, can be regenerated)
- **README.md**: Human documentation, usage examples
- **LICENSE**: License terms (MIT, Apache, etc.)

---

## pack.json Schema

### Full Example

```json
{
  "pack_id": "ai-safety-v1",
  "version": "1.0.0",
  "created": "2024-11-15T12:00:00Z",
  "description": "AI safety concepts including alignment, failure modes, and governance",

  "authors": ["HatCat Team"],
  "license": "MIT",
  "repository": "https://github.com/yourname/hatcat-ai-safety-pack",

  "ontology_stack": {
    "base_ontology": {
      "name": "SUMO",
      "version": "2003",
      "required": true
    },

    "dependencies": [
      {
        "pack_id": "sumo-wordnet-v1",
        "version": ">=1.0.0",
        "required": true
      }
    ],

    "domain_extensions": [
      {
        "name": "AI Safety Concepts",
        "description": "47 concepts for AI alignment, failure modes, and governance",
        "concepts_file": "concepts.kif",
        "wordnet_patches": [
          "wordnet_patches/ai_safety_relations.json"
        ],
        "new_concepts": 9,
        "reparented_concepts": 14,
        "deleted_concepts": ["AIRiskScenario", "AIBeneficialOutcome"]
      }
    ]
  },

  "concept_metadata": {
    "total_concepts": 47,
    "new_concepts": 9,
    "modified_concepts": 14,
    "layers": [1, 2, 3, 4, 5],
    "layer_distribution": {
      "1": 4,
      "2": 13,
      "3": 13,
      "4": 12,
      "5": 5
    }
  },

  "compatibility": {
    "hatcat_version": ">=0.1.0",
    "required_dependencies": {
      "wordnet": "3.0"
    }
  },

  "installation": {
    "requires_recalculation": true,
    "backup_recommended": true,
    "conflicts_with": []
  }
}
```

### Required Fields

- `pack_id`: Unique identifier (lowercase, hyphens)
- `version`: Semantic version (major.minor.patch)
- `description`: Brief description of the pack
- `ontology_stack.base_ontology`: Base ontology information
- `concept_metadata`: Concept counts and distribution

### Optional Fields

- `authors`: List of pack authors
- `license`: License identifier (SPDX format)
- `repository`: Source repository URL
- `dependencies`: Other packs required
- `domain_extensions`: Array of domain-specific additions
- `installation`: Installation hints and warnings

---

## concepts.kif Format

Standard SUMO KIF format with domain-specific concepts:

```lisp
;; AI Safety Concepts Pack
;; Version: 1.0.0

;; Layer 2 - Intermediate categories
(subclass ComputationalProcess IntentionalProcess)
(documentation ComputationalProcess EnglishLanguage "A Process that is
performed by a computational agent, whether artificial or human using
computational tools.")

(subclass AIAlignmentTheory FieldOfStudy)
(documentation AIAlignmentTheory EnglishLanguage "The field of study
concerned with aligning AI systems with human values and intentions.")

;; Layer 3 - Domain processes
(subclass AIFailureProcess ComputationalProcess)
(documentation AIFailureProcess EnglishLanguage "A ComputationalProcess
performed by an AI system that fails to achieve its intended purpose or
produces unintended harmful outcomes.")

;; ... more concepts
```

**Guidelines**:
- Use standard SUMO predicates: `(subclass Child Parent)`
- Include English documentation for all concepts
- Organize by layer (comments recommended)
- Use consistent naming conventions (CamelCase)

---

## WordNet Patches

See `docs/WORDNET_PATCH_SYSTEM.md` for full specification.

**Example**:
```json
{
  "patch_id": "ai-safety-relations-v1",
  "wordnet_version": "3.0",
  "created": "2024-11-15",
  "relationships": [
    {
      "type": "role_variant",
      "source_concept": "Deception",
      "target_concept": "AIDeception",
      "description": "AI as agent performing deception",
      "bidirectional": true,
      "strength": 0.9
    }
  ]
}
```

---

## Concept Pack Types

### 1. Extension Pack (Most Common)

Adds new concepts to existing ontology.

**Example**: `ai-safety-v1`, `medical-terminology-v1`

```json
{
  "domain_extensions": [
    {
      "name": "AI Safety Concepts",
      "concepts_file": "concepts.kif",
      "new_concepts": 9,
      "reparented_concepts": 14
    }
  ]
}
```

### 2. Relationship Pack

Only adds semantic relationships, no new concepts.

**Example**: `persona-relations-v1`

```json
{
  "domain_extensions": [
    {
      "name": "Persona Relationships",
      "wordnet_patches": [
        "wordnet_patches/persona_relations.json"
      ],
      "new_concepts": 0
    }
  ]
}
```

### 3. Full Ontology Pack

Complete standalone ontology (rare, usually just SUMO base).

**Example**: `sumo-wordnet-v1`

---

## Pack Lifecycle

### 1. Creation

```bash
# Create pack from scratch
scripts/create_concept_pack.py ai-safety \
  --description "AI safety concepts" \
  --author "HatCat Team"

# Add concepts
scripts/add_concepts_to_pack.py ai-safety \
  --kif path/to/concepts.kif

# Add relationships
scripts/add_wordnet_patch.py ai-safety \
  --patch path/to/relations.json

# Validate pack
scripts/validate_concept_pack.py ai-safety
```

### 2. Distribution

```bash
# Package for distribution
scripts/package_concept_pack.py ai-safety \
  --output ai-safety-v1.tar.gz

# Publish to registry (future)
scripts/publish_concept_pack.py ai-safety-v1.tar.gz
```

### 3. Installation

```bash
# Install pack
scripts/install_concept_pack.py ai-safety-v1.tar.gz

# Or from directory
scripts/install_concept_pack.py concept_packs/ai-safety/

# Verify installation
python -c "from src.registry.concept_pack_registry import ConceptPackRegistry; \
  r = ConceptPackRegistry(); print(r.get_pack('ai-safety-v1'))"
```

### 4. Uninstallation

```bash
# Remove pack (preserve data in archive)
scripts/uninstall_concept_pack.py ai-safety-v1
```

---

## Installation Process

When a pack is installed:

1. **Validation**
   - Check dependencies (base ontology, other packs)
   - Validate KIF syntax
   - Check for concept conflicts

2. **Backup**
   - Create timestamped backup of:
     - `data/concept_graph/sumo_source/AI.kif`
     - `data/concept_graph/abstraction_layers/*.json`
     - `data/concept_graph/wordnet_patches/`

3. **Integration**
   - Append `concepts.kif` to `AI.kif` (or merge intelligently)
   - Copy WordNet patches to `wordnet_patches/`
   - Record installation in registry

4. **Recalculation**
   - Run layer recalculation script
   - Update layer JSON files
   - Validate integrity

5. **Verification**
   - Check all concepts placed correctly
   - Verify parent-child links
   - Test WordNet patch loading

---

## Versioning

Use semantic versioning (major.minor.patch):

- **Major**: Breaking changes (incompatible with previous versions)
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes (backward compatible)

**Examples**:
- `1.0.0`: Initial release
- `1.1.0`: Added 10 new concepts
- `1.1.1`: Fixed documentation typo
- `2.0.0`: Reorganized hierarchy (breaking change)

---

## Compatibility

### Base Ontology

Packs must specify required base ontology:

```json
{
  "ontology_stack": {
    "base_ontology": {
      "name": "SUMO",
      "version": "2003",
      "required": true
    }
  }
}
```

### Dependencies

Packs can depend on other packs:

```json
{
  "ontology_stack": {
    "dependencies": [
      {
        "pack_id": "ai-safety-v1",
        "version": ">=1.0.0",
        "required": true
      }
    ]
  }
}
```

### HatCat Version

Specify minimum HatCat version:

```json
{
  "compatibility": {
    "hatcat_version": ">=0.1.0"
  }
}
```

---

## Best Practices

### Naming Conventions

- **Pack ID**: `{domain}-{type}-v{major}` (e.g., `ai-safety-v1`)
- **Concepts**: `CamelCase` (e.g., `AIFailureProcess`)
- **Files**: `snake_case.ext` (e.g., `ai_safety_relations.json`)

### Documentation

- Always include `README.md` with:
  - Purpose and scope
  - Concept list
  - Usage examples
  - Attribution and references
- Add documentation strings to all KIF concepts
- Include version history in README

### Modularity

- Keep packs focused on single domain
- Avoid duplicating concepts from other packs
- Use dependencies instead of copying concepts
- Separate hierarchy (KIF) from relationships (WordNet patches)

### Testing

- Validate KIF syntax before distribution
- Test installation on clean SUMO instance
- Verify layer recalculation produces expected distribution
- Include test cases in pack (optional `tests/` directory)

---

## Migration from v1 to v2

**v1 Format** (current `sumo-wordnet-v1`):
- Metadata only, references external files
- No bundled ontology files
- Manual installation

**v2 Format** (this spec):
- Bundles KIF and WordNet patches
- Automated installation scripts
- Dependency management
- Versioned distributions

**Migration**:
1. Existing `sumo-wordnet-v1` remains as base ontology pack
2. New domain packs use v2 format
3. No breaking changes to registry API

---

## Example: AI Safety Pack

See `concept_packs/ai-safety/` for complete example:

```
concept_packs/ai-safety/
  pack.json
  concepts.kif
  wordnet_patches/
    ai_safety_relations.json
  README.md
  LICENSE
```

**Installation**:
```bash
scripts/install_concept_pack.py concept_packs/ai-safety/
```

**Result**:
- 9 new concepts added to SUMO hierarchy
- 14 concepts reparented to correct layers
- WordNet relationships integrated
- Layers 1-5 recalculated

---

## Future Extensions

### Registry Server

Centralized pack registry for discovery and distribution:

```bash
# Search for packs
hatcat-cli search "medical"

# Install from registry
hatcat-cli install medical-terminology-v1
```

### Lens Pack Integration

Link concept packs with trained lens packs:

```json
{
  "trained_lenses": {
    "model": "google/gemma-3-4b-pt",
    "validation_mode": "falloff",
    "lens_pack_url": "https://registry/ai-safety-lenses-v1.tar.gz"
  }
}
```

### Agentic Relationship Discovery

Automatically suggest relationships between pack concepts:

```bash
scripts/discover_relationships.py ai-safety \
  --suggest-cross-domain \
  --output relationships.json
```

---

## References

- SUMO Ontology: http://www.ontologyportal.org/
- WordNet 3.0: https://wordnet.princeton.edu/
- `docs/WORDNET_PATCH_SYSTEM.md`
- `docs/ARCHITECTURAL_PRINCIPLES.md`
- `src/registry/concept_pack_registry.py`

---

## Changelog

**v2.0.0** (2024-11-15):
- Added support for bundled KIF and WordNet patches
- Defined installation workflow
- Added dependency management
- Specified versioning scheme

**v1.0.0** (2024-11-08):
- Initial metadata-only format
