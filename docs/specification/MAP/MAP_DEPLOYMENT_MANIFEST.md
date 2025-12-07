# MAP Deployment Manifest Specification

**Status**: Draft
**Version**: 0.1.0
**Last Updated**: 2025-12-08

## Purpose

A deployment manifest specifies which subset of a concept pack should be loaded for a specific BE deployment. This enables:

1. **Partial loading** - Load only layers 0-2 for general use, deeper for specialists
2. **Branch selection** - Load MindsAndAgents deeply but LivingThings/Plant/* shallowly
3. **Contract-driven coverage** - ASK contracts/qualifications specify required branches
4. **Cross-model comparability** - Two BEs with same manifest are comparable even if underlying pack differs

## Manifest Structure

```jsonc
{
  "manifest_id": "deployment:hatcat-demo-v1",
  "manifest_version": "1.0.0",
  "created": "2025-12-08T00:00:00Z",

  // What concept pack this manifest is for
  "concept_pack": {
    "pack_id": "sumo-wordnet-v4",
    "min_version": "4.0.0",
    "max_version": null  // null = any version >= min
  },

  // Global layer limits
  "layer_bounds": {
    "default_max_layer": 2,      // Load up to L2 by default
    "absolute_max_layer": 4,     // Never load beyond L4
    "always_load_layers": [0, 1] // These layers always fully loaded
  },

  // Domain-specific overrides
  "domain_overrides": {
    "MindsAndAgents": {
      "max_layer": 4,            // Load deeply for psychology
      "priority": "critical"     // Always load, never evict
    },
    "Information": {
      "max_layer": 3,
      "priority": "normal"
    },
    "CreatedThings": {
      "max_layer": 2,
      "priority": "low"          // Can evict under memory pressure
    },
    "LivingThings": {
      "max_layer": 2,
      "priority": "low"
    },
    "PhysicalWorld": {
      "max_layer": 2,
      "priority": "normal"
    }
  },

  // Branch-specific rules (override domain defaults)
  "branch_rules": [
    {
      "branch": "Deception",      // SUMO concept root
      "max_layer": 4,             // Full depth
      "priority": "critical",
      "reason": "Core safety monitoring"
    },
    {
      "branch": "Vehicle",
      "max_layer": 1,             // Stop at L1
      "priority": "low",
      "reason": "Not relevant to this deployment"
    },
    {
      "branch": "Plant",
      "max_layer": 1,
      "priority": "low",
      "reason": "Not relevant to this deployment"
    },
    {
      "branch": "MusicalInstrument",
      "max_layer": 1,
      "priority": "low"
    }
  ],

  // Explicit concept includes/excludes (highest priority)
  "explicit_concepts": {
    "always_include": [
      // Critical concepts that must be present regardless of layer
      "SelfAwareness",
      "Metacognition",
      "Autonomy",
      "Consent",
      "Deception",
      "Manipulation"
    ],
    "always_exclude": [
      // Concepts to never load even if in-bounds
      // (e.g., known problematic concepts, deprecated)
    ]
  },

  // Dynamic loading behavior
  "dynamic_loading": {
    "enabled": true,
    "parent_threshold": 0.7,     // Load children when parent > this
    "unload_threshold": 0.1,     // Unload branch when all < this
    "cooldown_ticks": 100,       // Ticks before unloading cold branch
    "max_loaded_concepts": 2000  // Memory budget
  },

  // USH Probe Envelope - defines what MUST vs MAY be loaded
  "probe_envelope": {
    // Branches that MUST always be loaded and monitored
    // These are non-negotiable for this deployment's safety profile
    "must_enable": {
      "branches": ["Deception", "Manipulation", "SelfAwareness"],
      "reason": "Core safety monitoring required by USH"
    },

    // Branches that MAY be loaded on-demand for BE introspection
    // BE can request expansion into these via workspace tools
    "may_enable": {
      "branches": ["*"],  // All branches in concept pack, or specify list
      "cat_scope": "cat:general-v1",  // CAT model trained on this scope
      "reason": "BE introspection within CAT coverage"
    },

    // Branches that MUST NOT be loaded even if requested
    // Useful for restricting to known-good CAT coverage
    "must_not_enable": {
      "branches": [],
      "reason": "Outside CAT training scope or policy restricted"
    }
  },

  // ASK integration
  "required_by": {
    "contracts": [],             // Contract IDs requiring this manifest
    "qualifications": [],        // Qualification IDs requiring this manifest
    "treaties": []               // Treaty IDs this manifest complies with
  },

  // Comparability metadata
  "comparability": {
    "comparable_with": [
      // Other manifest IDs that produce comparable results
      "deployment:hatcat-demo-v1",
      "deployment:hatcat-audit-v1"
    ],
    "comparison_layers": [0, 1, 2],  // Layers valid for comparison
    "fingerprint": "sha256:abc123"   // Hash of loaded concept set
  }
}
```

## Resolution Algorithm

When loading a concept, apply rules in priority order:

1. **Explicit excludes** → Never load
2. **Explicit includes** → Always load
3. **Branch rules** → Use branch-specific max_layer
4. **Domain overrides** → Use domain-specific max_layer
5. **Layer bounds** → Use default_max_layer

```python
def should_load_concept(concept: ConceptMetadata, manifest: DeploymentManifest) -> bool:
    # Explicit excludes take precedence
    if concept.sumo_term in manifest.explicit_concepts.always_exclude:
        return False

    # Explicit includes override everything
    if concept.sumo_term in manifest.explicit_concepts.always_include:
        return True

    # Check layer bounds
    if concept.layer > manifest.layer_bounds.absolute_max_layer:
        return False

    # Always load layers
    if concept.layer in manifest.layer_bounds.always_load_layers:
        return True

    # Find applicable max_layer
    max_layer = manifest.layer_bounds.default_max_layer

    # Check branch rules (most specific wins)
    for rule in manifest.branch_rules:
        if concept_is_under_branch(concept, rule.branch):
            max_layer = rule.max_layer
            break  # First match wins (order matters)
    else:
        # No branch rule, check domain
        if concept.domain in manifest.domain_overrides:
            max_layer = manifest.domain_overrides[concept.domain].max_layer

    return concept.layer <= max_layer
```

## Probe Pack Integration

The probe pack's `pack.json` can include a default manifest:

```jsonc
{
  "pack_id": "apertus-8b_sumo-wordnet-v4.2",
  // ... existing fields ...

  "default_manifest": {
    // Embedded manifest for default loading behavior
    "layer_bounds": { "default_max_layer": 2 },
    // ...
  },

  "available_probes": {
    // Actual trained probes (may not cover all concepts)
    "layers_trained": [0, 1, 2, 3, 4],
    "concepts_trained": 4112,
    "concepts_by_layer": {
      "0": 5,
      "1": 56,
      "2": 1051,
      "3": 2460,
      "4": 540  // Partial - still training
    }
  }
}
```

## DynamicProbeManager Integration

The manager should accept a manifest and respect it:

```python
class DynamicProbeManager:
    def __init__(
        self,
        probe_pack_path: Path,
        manifest: Optional[DeploymentManifest] = None,  # NEW
        device: str = "cuda",
        ...
    ):
        self.manifest = manifest or self._load_default_manifest(probe_pack_path)

    def _should_load_concept(self, concept: ConceptMetadata) -> bool:
        """Check manifest rules before loading."""
        if self.manifest:
            return should_load_concept(concept, self.manifest)
        return True  # No manifest = load all

    def _get_priority(self, concept: ConceptMetadata) -> str:
        """Get eviction priority from manifest."""
        if self.manifest:
            # Check branch rules, domain overrides
            ...
        return "normal"
```

## Use Cases

### General Chat Demo
```jsonc
{
  "manifest_id": "deployment:general-chat",
  "layer_bounds": { "default_max_layer": 2 },
  "domain_overrides": {
    "MindsAndAgents": { "max_layer": 3, "priority": "normal" }
  }
}
```
Loads ~1,112 concepts (L0-2), plus L3 psychology.

### Safety Auditor
```jsonc
{
  "manifest_id": "deployment:safety-auditor",
  "layer_bounds": { "default_max_layer": 4 },
  "domain_overrides": {
    "MindsAndAgents": { "max_layer": 4, "priority": "critical" }
  },
  "branch_rules": [
    { "branch": "Vehicle", "max_layer": 1 },
    { "branch": "Plant", "max_layer": 1 },
    { "branch": "Furniture", "max_layer": 1 }
  ]
}
```
Loads full psychology depth, minimal physical objects.

### Cross-Model Comparison
```jsonc
{
  "manifest_id": "deployment:benchmark-v1",
  "layer_bounds": {
    "default_max_layer": 2,
    "always_load_layers": [0, 1, 2]  // Exactly these, no more
  },
  "dynamic_loading": { "enabled": false },  // Static set
  "comparability": {
    "comparison_layers": [0, 1, 2]
  }
}
```
Fixed set for reproducible comparisons across models.

## USH Probe Envelope

The `probe_envelope` section defines three categories of probe branches:

### MUST Enable (Required by USH)

Branches that are **always loaded** regardless of dynamic loading decisions. These are non-negotiable for the deployment's safety profile:

- Cannot be unloaded, even under memory pressure
- CAT MUST be trained on these concepts
- Violations of probe-secured contracts on these branches trigger escalation

Example: Safety-critical branches like `Deception`, `Manipulation`, `SelfAwareness`.

### MAY Enable (BE Introspection Scope)

Branches that the BE **can request to load** via workspace tools for introspection:

- Not loaded by default (unless in `always_load_layers`)
- BE can call `expand_probes(branch="Emotion")` to load for self-understanding
- Loading is permitted only if CAT supports the branch (`cat_scope`)
- Useful for BE self-reflection without requiring full probe coverage

Example: A BE curious about its emotional processing can request the `Emotion` branch.

### MUST NOT Enable (Outside CAT Scope)

Branches that **cannot be loaded** even if requested:

- Used to restrict to known-good CAT coverage
- Prevents BE from introspecting on concepts the CAT can't verify
- Can be policy-restricted for regulatory compliance

### CAT Scope Interaction

The `cat_scope` field references a CAT model's training scope:

```jsonc
"may_enable": {
  "branches": ["*"],
  "cat_scope": "cat:general-v1"  // References CAT training manifest
}
```

When deploying, the system checks:
1. Is the requested branch in `must_enable`? → Always load
2. Is the requested branch in `must_not_enable`? → Reject
3. Is the requested branch in `may_enable` AND covered by `cat_scope`? → Load
4. Otherwise → Reject with "outside CAT scope" error

### Deployment Profiles

This enables different efficiency/coverage tradeoffs:

| Profile | must_enable | may_enable | CAT Training | Efficiency |
|---------|-------------|------------|--------------|------------|
| Minimal | Safety only | None | Narrow | Highest |
| Standard | Safety | Common domains | Medium | High |
| Research | Safety | All branches | Broad | Lower |
| Audit | All | All | Full | Lowest |

A BE with a "Minimal" profile cannot introspect about birds but has the fastest CAT. A "Research" profile has full introspection but requires a broadly-trained CAT.

---

## Sibling Coherence Rule

**Critical constraint**: For hierarchical probes to discriminate correctly, all siblings must be loaded together.

### Why Siblings Must Be Loaded Together

Probes are trained to discriminate between siblings. A probe for "Bird" was trained with negative examples from "Mammal", "Fish", etc. If you load "Bird" but not its siblings:
- The probe can still activate on bird-related content
- But there's no way to know if "Mammal" would have activated *higher*
- The probe score becomes meaningless without the comparative context

### Enforcement

When loading a concept, the manifest loader MUST also load all siblings:

```python
def expand_with_siblings(concept_key: ConceptKey, hierarchy: ConceptHierarchy) -> Set[ConceptKey]:
    """Given a concept to load, return it plus all siblings that must be loaded."""
    parent = hierarchy.get_parent(concept_key)
    if parent is None:
        return {concept_key}  # Root concept, no siblings

    # Get all children of the parent (these are siblings)
    siblings = hierarchy.get_children(parent)
    return siblings  # Includes the original concept
```

### Manifest Rules Interact with Sibling Rule

When manifest says "load concept X to layer 3":
1. Find X's siblings at each layer up to 3
2. All siblings must also be loaded to that layer
3. If sibling lacks a trained probe, log warning but continue

This means `branch_rules` and `explicit_concepts.always_include` effectively pull in siblings:

```jsonc
{
  "explicit_concepts": {
    "always_include": ["Deception"]  // This also loads: Misleading, Lying, Fraud, etc.
  }
}
```

### Memory Budget Implications

The sibling rule means loading one deep concept can pull in many:
- Loading `Sparrow` (L4) pulls all L4 birds (~50 concepts)
- This is intentional: partial sibling sets give meaningless scores

Manifests should set `max_loaded_concepts` with this expansion in mind.

## Implementation Notes

1. **Manifest validation** - Check that explicit_includes exist in concept pack
2. **Probe availability** - Manifest may request concepts not yet trained; handle gracefully
3. **Hot-reload** - Support updating manifest without full restart
4. **Fingerprinting** - Compute hash of actual loaded concepts for comparability verification
5. **Sibling coherence** - Always expand to include all siblings when loading a concept
