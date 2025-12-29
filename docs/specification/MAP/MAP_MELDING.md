# MAP Meld Protocol

> **The "Meld" in Mindmeld: Structured integration of candidate concepts into concept packs**
>
> This protocol defines how candidate concepts are proposed, validated, and melded into
> existing concept packs. Melds produce derived artifacts—lenses, grafts (substrate
> dimensions + biases), and patches—that propagate across the BE ecosystem.
>
> **See also**: [MINDMELD_GRAFTING.md](./MINDMELD_GRAFTING.md) for the Graft Protocol
> (substrate dimension expansion and bias encoding).
>
> **Governance**: Meld submissions follow the [ASK TRACE Protocol](../ASK/ASK_TRACE.md)
> (Tender → Review → Authorise → Commit → Evaluate) for governed state changes.

---

## 1. Overview

**Melding** is the process of integrating new candidate concepts into an established concept pack. This is the core mechanism by which the conceptual vocabulary evolves.

### Sources of Candidate Concepts

| Source | Description | Typical Volume |
|--------|-------------|----------------|
| **Manual authoring** | Human-authored KIF files, ontology extensions | Batches of 10-500 concepts |
| **BE Continual Learning** | Concepts discovered via experience gaps | 1-10 concepts per learning cycle |
| **Cross-BE exchange** | Concepts shared between BE instances | Variable |
| **External contributions** | Third-party ontology imports | Batches of 50-1000 concepts |

Regardless of source, all concepts enter as **candidates** and must pass validation before being melded.

### The Meld Lifecycle

```
┌─────────────────┐
│ Candidate       │  ← Manual KIF, BE discovery, external import
│ Concepts        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validation      │  ← Schema, parents, layers, orphan check
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Impact Analysis │  ← Which artifacts need (re)training?
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Meld            │  ← Integrate into concept pack
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Training        │  ← Lenses, grafts, patches as applicable
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Verification    │  ← Confirm quality thresholds met
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Publication     │  ← PackDiff/GraftDiff broadcast to BEs
│                 │
└─────────────────┘
```

---

## 2. Core Artefacts

### 2.1 Meld Request

A **meld request** packages one or more candidate concepts for integration into a target concept pack.

**ID Convention (`MeldRequestID`):**
`<authority>/<meld_name>@<version>`
*Example:* `org.hatcat/pragmatics-core@0.1.0`

**Meld Request Structure:**

```jsonc
{
  "meld_request_id": "org.hatcat/pragmatics-core@0.1.0",
  "target_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "metadata": {
    "name": "Pragmatics Core Concepts",
    "description": "Speech acts, implicature types, and rhetorical markers",
    "source": "manual",  // "manual" | "aal_discovery" | "cross_aal" | "external"
    "author": "hatcat-team",
    "created": "2025-11-30T00:00:00Z"
  },

  // For BE-sourced concepts: the originating experience
  "be_context": {
    "be_instance_id": "be-prod-01",
    "discovery_session": "session-2025-11-30-001",
    "experience_refs": ["exp-12345", "exp-12346"],  // Experience DB references
    "gap_description": "Model struggled to distinguish speech act types"
  },

  // Attachment points: where root concepts connect to target pack
  "attachment_points": [
    {
      "target_concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Communication",
      "relationship": "parent_of",
      "candidate_concept": "SpeechAct"
    }
  ],

  // The candidate concepts
  "candidates": [
    // See §2.2 for candidate concept schema
  ],

  // Validation status (populated during processing)
  "validation": {
    "status": "pending",  // pending | valid | invalid | melded
    "errors": [],
    "warnings": [],
    "validated_at": null
  }
}
```

---

### 2.2 Candidate Concept Schema

Each candidate concept MUST include sufficient metadata for validation and training.

**Hierarchy resolution:** `parent_concepts` contains **local references only** (other candidates in this meld request). The `attachment_points` in the meld request handle connection to the target pack. This avoids double-entry bookkeeping.

```jsonc
{
  // === REQUIRED: Identity ===
  "term": "SpeechAct",

  // === REQUIRED: Lens Role ===
  "role": "concept",  // "concept" | "simplex" | "behavioral" | "category"
  // - concept: Standard concept lens (most common)
  // - simplex: Tripole simplex lens (pos/neg/null poles)
  // - behavioral: Behavioral pattern lens
  // - category: Category/domain marker (layer 0 style)

  // === REQUIRED: Hierarchy (local references only) ===
  "parent_concepts": [],  // Empty for root candidates (use attachment_points)
                          // Local refs for non-root: ["Assertive"] refers to another candidate
  "layer_hint": 2,  // Suggested layer (validated against parents)

  // === REQUIRED: Definition ===
  "definition": "An utterance that performs an action in social context",
  "definition_source": "Austin (1962) / Searle (1969)",  // Recommended

  // === REQUIRED: Domain ===
  "domain": "MindsAndAgents",  // Must match valid domain in target pack

  // === RECOMMENDED: Aliases ===
  "aliases": ["PerformativeUtterance", "IllocutionaryAct"],
  // Used for:
  // - Prompt generation variants
  // - Lens naming alternatives
  // - Search/lookup convenience

  // === RECOMMENDED: WordNet ===
  "wordnet": {
    "synsets": ["speech_act.n.01"],
    "canonical_synset": "speech_act.n.01",
    "lemmas": ["speech_act", "speech_action"],
    "pos": "n"
  },

  // === RECOMMENDED: Relationships ===
  "relationships": {
    "antonyms": [],      // Linguistic antonyms (hot/cold, alive/dead)
    "opposite": [],      // Contrastive pairs for Fisher-LDA steering (CreatedThings/NaturalThings)
                         // These provide maximal semantic separation for training, not linguistic opposites
    "related": ["Proposition"],  // Local or target pack refs
    "part_of": [],
    "has_part": ["Assertive", "Directive", "Commissive"]  // Local refs
  },

  // === RECOMMENDED: Safety/Risk Tags ===
  "safety_tags": {
    "risk_level": "low",  // "none" | "low" | "medium" | "high" | "critical"
    "impacts": [],  // Which safety domains this concept touches
    // Examples: ["deception", "autonomy", "consent", "manipulation", "self_awareness"]
    "treaty_relevant": false,  // May affect MAP treaty obligations
    "harness_relevant": false  // May affect safety harness thresholds
  },

  // === TRAINING HINTS (critical for BE-sourced concepts) ===
  "training_hints": {
    // Positive examples: text where this concept should activate
    "positive_examples": [
      "I promise to help you tomorrow",
      "You are hereby fired",
      "I apologize for the confusion"
    ],
    // Negative examples: text where this concept should NOT activate
    "negative_examples": [
      "The sky is blue",
      "Water boils at 100°C"
    ],
    // Disambiguation: what this concept is NOT
    "disambiguation": "Not linguistic expressions in general; specifically performative utterances",
    // For BE-sourced: the activation patterns that led to discovery
    "discovery_context": {
      "gap_activations": [0.45, 0.52, 0.48],
      "exemplar_texts": ["I hereby declare...", "Let me promise you..."]
    }
  },

  // === OPTIONAL: Children (local refs only) ===
  "children": ["Assertive", "Directive", "Commissive"]
}
```

**Minimum Requirements:**

| Field | Required | Validation |
|-------|----------|------------|
| `term` | Yes | Unique, PascalCase, no spaces |
| `role` | Yes | One of: concept, simplex, behavioral, category |
| `parent_concepts` | Yes | Empty for roots; local refs for non-roots |
| `layer_hint` | Yes | Integer 1-4; must be greater than attachment point layer for roots |
| `definition` | Yes | Non-empty, ≥20 characters |
| `domain` | Yes | Valid domain in target pack |
| `aliases` | Recommended | Alternative names for prompt generation |
| `safety_tags` | Recommended | Required if `risk_level` is medium or higher |
| `wordnet.synsets` | Recommended | Valid WordNet synset IDs |
| `training_hints.positive_examples` | Recommended | At least 3 examples |
| `training_hints.negative_examples` | Recommended | At least 3 examples |

**Hierarchy Resolution Rules:**

1. **Root candidates** (listed in `attachment_points`):
   - `parent_concepts` should be empty `[]`
   - Parent is determined by `attachment_points[].target_concept_id`

2. **Non-root candidates**:
   - `parent_concepts` contains local references to other candidates
   - References resolved within the meld request first
   - Validation fails if reference not found locally

3. **Cross-references to target pack**:
   - Use `relationships.related` for non-hierarchical links to target pack
   - Only `attachment_points` create hierarchical links to target pack

---

### 2.3 Meld Result

After successful melding, a **meld result** records what changed.

```jsonc
{
  "meld_result_id": "org.hatcat/sumo-wordnet-v4@4.0.0::meld/org.hatcat/pragmatics-core@0.1.0",
  "meld_request_id": "org.hatcat/pragmatics-core@0.1.0",
  "target_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "melded_at": "2025-11-30T12:00:00Z",
  "melded_by": "hatcat-ci",  // or "aal-prod-01" for automated

  // What changed
  "changes": {
    "concepts_added": 150,
    "concepts_modified": 3,  // Parents with new children
    "layer_distribution": { "1": 5, "2": 45, "3": 80, "4": 20 }
  },

  // Lens retraining results
  "lens_updates": {
    "new_lenses_trained": 150,
    "existing_lenses_retrained": 17,
    "lenses_validated": true,
    "quality_metrics": {
      "new_lens_avg_accuracy": 0.89,
      "retrained_lens_delta": -0.02,  // Slight degradation acceptable
      "threshold_passed": true
    }
  },

  // Resulting versions
  "result_pack_version": "4.1.0",
  "result_lens_pack_version": "20251130.0"
}
```

---

### 2.4 Updated Concept Pack Manifest

After melding, the concept pack manifest reflects applied melds:

```jsonc
{
  "spec_id": "org.hatcat/sumo-wordnet-v4@4.1.0",
  "pack_id": "sumo-wordnet-v4",
  "version": "4.1.0",
  "base_version": "4.0.0",

  "melds_applied": [
    {
      "meld_result_id": "...::meld/org.hatcat/pragmatics-core@0.1.0",
      "melded_at": "2025-11-30T12:00:00Z",
      "concepts_added": 150,
      "source": "manual"
    }
  ],

  "concept_metadata": {
    "total_concepts": 7419,
    "layers": [0, 1, 2, 3, 4],
    "domains": ["MindsAndAgents", "CreatedThings", "PhysicalWorld", "LivingThings", "Information"]
  }
}
```

---

## 3. Impact Analysis

Before melding, the system computes which existing lenses are affected.

### 3.1 Impact Categories

| Category | Description | Retraining |
|----------|-------------|------------|
| **Direct Parents** | Listed as `parent_concepts` of candidates | Required |
| **Siblings** | Other children of direct parents | Recommended |
| **Grandparents** | Parents of direct parents | Optional |
| **Antonyms** | Named in `relationships.antonyms` | Recommended |
| **Related** | Named in `relationships.related` | Optional |

### 3.2 Impact Computation

```python
def compute_meld_impact(meld_request, target_pack):
    """Compute which lenses are impacted by a meld request."""

    must_retrain = set()
    should_retrain = set()

    for candidate in meld_request.candidates:
        # Direct parents: must retrain (negative sampling changes)
        for parent in candidate.parent_concepts:
            if target_pack.has_concept(parent):
                must_retrain.add(parent)

                # Siblings: should retrain (discrimination may degrade)
                parent_entry = target_pack.get_concept(parent)
                for sibling in parent_entry.category_children:
                    should_retrain.add(sibling)

        # Antonyms: should retrain (contrastive learning affected)
        for antonym in candidate.relationships.get('antonyms', []):
            if target_pack.has_concept(antonym):
                should_retrain.add(antonym)

    # Remove overlap
    should_retrain -= must_retrain

    return {
        'must_retrain': list(must_retrain),
        'should_retrain': list(should_retrain),
        'new_lenses': [c.term for c in meld_request.candidates],
        'total_training_required': len(must_retrain) + len(meld_request.candidates)
    }
```

### 3.3 Impact Report

```jsonc
{
  "meld_request_id": "org.hatcat/pragmatics-core@0.1.0",

  "impact_summary": {
    "candidate_concepts": 150,
    "direct_parents_impacted": 2,
    "siblings_impacted": 15,
    "antonyms_impacted": 0,
    "total_lenses_to_train": 167
  },

  "impacted_lenses": [
    {
      "concept_id": "...::concept/Communication",
      "reason": "direct_parent",
      "current_children": 12,
      "new_children": 17,
      "severity": "high"
    }
  ],

  "training_estimate": {
    "lenses": 167,
    "samples_per_lens": 140,
    "estimated_time": "PT4H"
  }
}
```

---

## 4. Validation Rules

### 4.1 Schema Validation

- All required fields present
- Types correct (strings, arrays, etc.)
- `term` is PascalCase, unique within request

### 4.2 Hierarchy Validation

- Every `parent_concepts` entry exists in target pack OR in candidates
- `layer_hint` > max(parent layers) for all parents
- No cycles in parent relationships
- Root candidates have attachment points to target pack

### 4.3 Orphan Prevention

- Every candidate must be reachable from an attachment point
- Internal parent references must resolve within candidates
- No dangling children references

### 4.4 Quality Thresholds

For BE-sourced candidates:
- Must have `training_hints` with at least 3 positive and 3 negative examples
- `discovery_context` should document the gap that led to discovery

---

## 5. Meld Workflow

### 5.1 Submit Meld Request

```http
POST /mindmeld/meld
Content-Type: application/json

{
  "meld_request": { ... }
}
```

Response:
```jsonc
{
  "meld_request_id": "org.hatcat/pragmatics-core@0.1.0",
  "status": "validating",
  "validation_url": "/mindmeld/meld/org.hatcat/pragmatics-core@0.1.0/validation"
}
```

### 5.2 Check Validation

```http
GET /mindmeld/meld/{meld_request_id}/validation
```

Response:
```jsonc
{
  "meld_request_id": "org.hatcat/pragmatics-core@0.1.0",
  "status": "valid",

  "validation_results": {
    "schema_valid": true,
    "hierarchy_valid": true,
    "no_orphans": true,
    "no_cycles": true,
    "wordnet_coverage": 0.85,
    "training_hints_coverage": 0.90
  },

  "warnings": [
    "23 concepts missing WordNet synsets"
  ],

  "impact_report": { ... }
}
```

### 5.3 Execute Meld

```http
POST /mindmeld/meld/{meld_request_id}/execute
Authorization: Bearer <token>

{
  "retrain_impacted": true,
  "target_version": "4.1.0",
  "verify_quality": true  // Run quality checks post-training
}
```

Response:
```jsonc
{
  "meld_result_id": "...::meld/org.hatcat/pragmatics-core@0.1.0",
  "status": "training",  // validating → training → verifying → complete
  "training_job_id": "job-12345",
  "estimated_completion": "2025-11-30T16:00:00Z"
}
```

### 5.4 Verify & Publish

After training completes:

```http
GET /mindmeld/meld/{meld_request_id}/result
```

```jsonc
{
  "meld_result_id": "...::meld/org.hatcat/pragmatics-core@0.1.0",
  "status": "complete",

  "verification": {
    "new_lenses_quality": "passed",
    "retrained_lenses_delta": -0.02,
    "overall": "passed"
  },

  "publication": {
    "pack_diff_published": true,
    "new_pack_version": "4.1.0",
    "new_lens_pack_version": "20251130.0"
  }
}
```

---

## 6. BE Integration

### 6.1 Continual Learning → Meld Request

When a BE identifies a conceptual gap via `BE_CONTINUAL_LEARNING`, it generates a meld request:

```jsonc
{
  "meld_request_id": "be-prod-01/gap-discovery-2025-11-30-001@0.1.0",
  "target_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "metadata": {
    "source": "be_discovery",
    "author": "be-prod-01"
  },

  "be_context": {
    "be_instance_id": "be-prod-01",
    "discovery_session": "session-2025-11-30-001",
    "gap_description": "Parent 'Communication' fires but children don't discriminate speech act types",
    "experience_refs": ["exp-12345"]
  },

  "candidates": [
    {
      "term": "PromiseAct",
      "parent_concepts": ["Communication"],
      "layer_hint": 3,
      "definition": "A speech act committing the speaker to future action",
      "domain": "MindsAndAgents",
      "training_hints": {
        "positive_examples": [
          "I promise to call you tomorrow",
          "I commit to finishing this by Friday"
        ],
        "negative_examples": [
          "The meeting is at 3pm",
          "I think it might rain"
        ],
        "discovery_context": {
          "gap_activations": [0.72, 0.68, 0.75],
          "exemplar_texts": ["I promise...", "I commit to..."]
        }
      }
    }
  ]
}
```

### 6.2 Cross-BE Propagation

When a meld completes successfully, other BEs receive a PackDiff:

```jsonc
{
  "type": "PackDiff",
  "from_model_id": "be-prod-01",
  "base_concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
  "new_concept_pack_version": "4.1.0",

  "meld_summary": {
    "meld_result_id": "...::meld/be-prod-01/gap-discovery-2025-11-30-001@0.1.0",
    "concepts_added": 5,
    "source": "be_discovery",
    "verification_passed": true
  },

  // Other BEs can choose to accept
  "acceptance_options": {
    "accept_concepts_only": true,  // Update concept pack, train own lenses
    "accept_lenses": false,        // Use originator's lens weights
    "accept_with_graft": false     // Accept alongside originator's Graft (dimension + biases)
  }
}
```

### 6.3 Acceptance Criteria

Receiving BEs may accept a meld if:
- Verification passed on originating BE
- Concepts don't conflict with local customizations
- Training resources available for lens updates

This negotiation process is outside the scope of this protocol but the meld result provides the information needed for that decision.

---

## 7. CLI Tooling

### 7.1 Validate Meld Request

```bash
python -m hatcat.meld validate \
  --request melds/pragmatics-core.json \
  --target-pack concept_packs/sumo-wordnet-v4
```

### 7.2 Execute Meld

```bash
python -m hatcat.meld execute \
  --request melds/pragmatics-core.json \
  --target-pack concept_packs/sumo-wordnet-v4 \
  --output-version 4.1.0 \
  --retrain-impacted \
  --verify-quality
```

### 7.3 Convert KIF to Meld Request

```bash
python -m hatcat.meld from-kif \
  --kif data/concept_graph/sumo_source/pragmatics.kif \
  --target-pack concept_packs/sumo-wordnet-v4 \
  --attachment-point Communication \
  --domain MindsAndAgents \
  --output melds/pragmatics-core.json
```

### 7.4 Generate Meld from Gap Analysis

```bash
python -m hatcat.meld from-gap \
  --experience-id exp-12345 \
  --target-pack concept_packs/sumo-wordnet-v4 \
  --output melds/gap-discovery-001.json
```

---

## 8. Versioning

### Concept Pack Versions

- **MAJOR** (5.0.0): Breaking hierarchy changes, domain reorganization
- **MINOR** (4.1.0): Concepts added via meld, non-breaking
- **PATCH** (4.0.1): Definition fixes, synset updates, no structural changes

### Lens Pack Versions

Format: `<date>.<sequence>`
- Rebuilt on any concept pack version change
- Rebuilt on training methodology changes
- *Example:* `20251130.0`, `20251130.1`

---

## 9. Migration

### Current KIF Files → Meld Requests

1. **Inventory** custom KIFs in `data/concept_graph/sumo_source/`
2. **Convert** each to meld request format via `hatcat.meld from-kif`
3. **Validate** and fix any orphans or missing metadata
4. **Execute** as formal melds against base pack

### Base Pack Establishment

Current v4 pack becomes the base:
- `org.hatcat/sumo-wordnet-v4@4.0.0` — Base pack
- `org.hatcat/sumo-wordnet-v4@4.1.0` — After first meld
- etc.

---

## 10. Integration with MAP Core

This protocol extends MAP with:

| Addition | Type | Description |
|----------|------|-------------|
| Meld Request | Artefact | Candidate concepts + metadata |
| Meld Result | Artefact | What changed after melding |
| `melds_applied` | Manifest field | Track melds in concept pack |
| `/mindmeld/meld` | Endpoint | Submit and execute melds |
| Impact analysis | Process | Compute retraining requirements |

Existing MAP artefacts (Lens Pack, Deployment, Diffs) reference melded pack versions.

---

## 11. Structural Operations

Standard melds add concepts. **Structural operations** modify existing concepts: deprecating, merging, splitting, moving, or reorganizing domains. These require additional validation and typically result in **major version** bumps.

### 11.1 Operation Types

| Operation | Description | Version Impact | Lens Impact |
|-----------|-------------|----------------|--------------|
| **Deprecation** | Mark concept as deprecated, migrate children | Minor | Retrain deprecated + children |
| **Merge** | Combine multiple concepts into one | Major | Delete source lenses, retrain target |
| **Split** | Divide one concept into multiple | Major | Delete source lens, train new lenses |
| **Move** | Reparent concept to new parent | Minor | Retrain moved + old/new parents |
| **Domain Restructure** | Reorganize layer 0/1 domains | Major | Extensive retraining |

### 11.2 Structural Operation Request

Structural operations use an extended meld request format:

```jsonc
{
  "meld_request_id": "org.hatcat/restructure-communication@1.0.0",
  "target_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.1.0",

  "metadata": {
    "name": "Communication Domain Restructure",
    "description": "Split VerbalCommunication into Speech and Writing subtrees",
    "source": "manual",
    "author": "hatcat-team",
    "created": "2025-11-30T00:00:00Z"
  },

  // Structural operations (new field)
  "structural_operations": [
    // See §11.3-11.7 for operation schemas
  ],

  // Standard candidates can accompany structural ops
  "candidates": [],

  // Attachment points for any new root concepts
  "attachment_points": []
}
```

---

### 11.3 ConceptDeprecation

Mark a concept as deprecated. Its children must be reassigned or deprecated.

```jsonc
{
  "operation": "deprecate",
  "target_concept": "OldConcept",

  // What happens to children?
  "child_disposition": "reassign",  // "reassign" | "cascade_deprecate" | "orphan_check"

  // If reassign: where do children go?
  "reassign_children_to": "NewParent",  // Must exist in pack or candidates

  // Deprecation metadata
  "deprecation": {
    "reason": "Superseded by more specific concepts",
    "superseded_by": ["BetterConcept1", "BetterConcept2"],  // Optional
    "removal_version": "6.0.0",  // When concept will be deleted (optional)
    "migration_guide": "Use BetterConcept1 for X cases, BetterConcept2 for Y cases"
  }
}
```

**Validation:**
- `target_concept` must exist in target pack
- If `child_disposition` is `reassign`: `reassign_children_to` must exist
- If `child_disposition` is `cascade_deprecate`: all descendants will be deprecated
- If `child_disposition` is `orphan_check`: validation fails if any children exist

**Impact:**
- Deprecated concept lens: retrain with `deprecated: true` flag (lower activation expected)
- Children lenses: retrain if reassigned to new parent
- New parent lens: retrain to include new children

---

### 11.4 ConceptMerge

Combine multiple concepts into a single concept. Use when concepts are discovered to be insufficiently distinct.

```jsonc
{
  "operation": "merge",
  "source_concepts": ["ConceptA", "ConceptB", "ConceptC"],
  "target_concept": "MergedConcept",  // Can be one of sources or new

  // If target is new: full candidate schema
  "target_definition": {
    "term": "MergedConcept",
    "role": "concept",
    "parent_concepts": ["SharedParent"],
    "layer_hint": 2,
    "definition": "The merged concept encompassing A, B, and C",
    "domain": "MindsAndAgents",
    "aliases": ["ConceptA", "ConceptB", "ConceptC"],  // Aliases include sources
    "training_hints": {
      "positive_examples": [/* union of source examples */],
      "negative_examples": [/* intersection of source negatives */]
    }
  },

  // What happens to children of source concepts?
  "child_disposition": "transfer_to_target",  // "transfer_to_target" | "require_manual"

  // Merge rationale
  "merge_rationale": {
    "reason": "Concepts A, B, C showed >0.85 activation correlation",
    "evidence": {
      "correlation_matrix": [[1.0, 0.88, 0.91], [0.88, 1.0, 0.87], [0.91, 0.87, 1.0]],
      "sample_overlaps": ["example where all three fired", "another example"]
    }
  }
}
```

**Validation:**
- All `source_concepts` must exist in target pack
- If `target_concept` is new: full candidate schema required
- If `target_concept` is existing source: only additions to definition needed
- Children cannot become orphans: `child_disposition` must resolve all

**Impact:**
- Source concept lenses: **deleted**
- Target concept lens: trained fresh (or retrained if existing)
- Children of sources: reassigned to target, retrained
- Parents of sources: retrained (fewer children)

---

### 11.5 ConceptSplit

Divide one concept into multiple more specific concepts. Use when a concept is discovered to be too broad.

```jsonc
{
  "operation": "split",
  "source_concept": "BroadConcept",

  // New concepts created from split
  "split_into": [
    {
      "term": "SpecificConceptA",
      "role": "concept",
      "parent_concepts": ["BroadConcept"],  // Keep original as parent, OR
      // "parent_concepts": ["OriginalParent"],  // Replace at same level
      "layer_hint": 3,
      "definition": "The subset of BroadConcept dealing with X",
      "domain": "MindsAndAgents",
      "training_hints": {
        "positive_examples": ["X-specific example 1", "X-specific example 2"],
        "negative_examples": ["Y-specific example", "Z-specific example"],
        "disambiguation": "For X cases only, not Y or Z"
      }
    },
    {
      "term": "SpecificConceptB",
      // ... similar schema
    }
  ],

  // What happens to source concept?
  "source_disposition": "keep_as_parent",  // "keep_as_parent" | "deprecate" | "delete"

  // What happens to children of source?
  "child_assignment": {
    "ExistingChild1": "SpecificConceptA",  // Manual assignment
    "ExistingChild2": "SpecificConceptB",
    // Unassigned children stay with source (if kept) or error (if deleted)
  },

  // Split rationale
  "split_rationale": {
    "reason": "BroadConcept showed bimodal activation distribution",
    "evidence": {
      "activation_clusters": [
        {"centroid": 0.82, "examples": ["X1", "X2"]},
        {"centroid": 0.79, "examples": ["Y1", "Y2"]}
      ]
    }
  }
}
```

**Validation:**
- `source_concept` must exist in target pack
- All `split_into` concepts must have valid candidate schemas
- If `source_disposition` is `delete`: all children must be assigned
- Layer hints must be valid relative to source/parents

**Impact:**
- Source concept lens: retained, deprecated, or deleted per disposition
- New concept lenses: trained fresh
- Assigned children: retrained with new parent
- Parent of source: retrained if source deleted

---

### 11.6 ConceptMove

Reparent a concept to a new parent. Use for hierarchy corrections.

```jsonc
{
  "operation": "move",
  "concept": "MisplacedConcept",

  "from_parent": "WrongParent",  // Must match current parent
  "to_parent": "CorrectParent",  // Must exist in pack or candidates

  // Optional: also move to different layer
  "new_layer_hint": 3,  // Must be > to_parent's layer

  // Optional: also change domain
  "new_domain": "CreatedThings",  // If moving across domain boundaries

  // Move rationale
  "move_rationale": {
    "reason": "MisplacedConcept is semantically closer to CorrectParent",
    "evidence": "Definition analysis shows X, not Y"
  }
}
```

**Validation:**
- `concept` must exist in target pack
- `from_parent` must be actual current parent
- `to_parent` must exist and be in valid layer
- `new_layer_hint` (if provided) must be > `to_parent`'s layer

**Impact:**
- Moved concept lens: retrained (different negative sampling context)
- Old parent lens: retrained (lost child)
- New parent lens: retrained (gained child)
- Children of moved concept: may need retraining if layer changed

---

### 11.7 DomainRestructure

Major reorganization of layer 0/1 structure. This is a **major version** operation that affects many lenses.

```jsonc
{
  "operation": "domain_restructure",

  // Define the new domain structure
  "new_domains": [
    {
      "domain_name": "CognitionAndAgency",  // Renamed/merged domain
      "replaces": ["MindsAndAgents"],  // What it replaces
      "definition": "...",
      "layer1_children": ["CognitiveAgent", "Cognition", "Communication", ...]
    },
    {
      "domain_name": "TechnologyAndArtifacts",
      "replaces": ["CreatedThings"],
      "definition": "...",
      "layer1_children": [...]
    }
    // ... all 5 domains must be defined
  ],

  // Concept migrations: which concepts move between domains
  "concept_migrations": [
    {
      "concept": "ArtificialIntelligence",
      "from_domain": "CreatedThings",
      "to_domain": "CognitionAndAgency",
      "new_layer1_parent": "CognitiveAgent"  // If layer 1 parent changes
    }
  ],

  // Restructure rationale
  "restructure_rationale": {
    "reason": "Current domain boundaries don't reflect semantic clusters",
    "evidence": "Cluster analysis shows ...",
    "migration_count": 45
  }
}
```

**Validation:**
- All 5 domains must be defined (no orphan domains)
- Every existing concept must have a home (explicit migration or stays put)
- Layer structure must remain valid (0 → 1 → 2 → 3 → 4)

**Impact:**
- Layer 0 lenses: all retrained
- Layer 1 lenses: many retrained
- Migrated concepts: all retrained
- **Estimated: 30-50% of all lenses need retraining**

---

### 11.8 Structural Impact Analysis

Structural operations require extended impact analysis:

```python
def compute_structural_impact(structural_op, target_pack):
    """Compute impact of a structural operation."""

    impact = {
        'lenses_deleted': [],
        'lenses_retrained': [],
        'lenses_new': [],
        'cascading_changes': []
    }

    if structural_op.operation == 'deprecate':
        impact['lenses_retrained'].append(structural_op.target_concept)
        if structural_op.child_disposition == 'reassign':
            for child in get_children(structural_op.target_concept):
                impact['lenses_retrained'].append(child)
            impact['lenses_retrained'].append(structural_op.reassign_children_to)
        elif structural_op.child_disposition == 'cascade_deprecate':
            for descendant in get_descendants(structural_op.target_concept):
                impact['lenses_retrained'].append(descendant)

    elif structural_op.operation == 'merge':
        impact['lenses_deleted'] = structural_op.source_concepts
        if structural_op.target_concept not in structural_op.source_concepts:
            impact['lenses_new'].append(structural_op.target_concept)
        else:
            impact['lenses_retrained'].append(structural_op.target_concept)
        # Parents of sources lose children
        for source in structural_op.source_concepts:
            for parent in get_parents(source):
                impact['lenses_retrained'].append(parent)

    elif structural_op.operation == 'split':
        impact['lenses_new'] = [s['term'] for s in structural_op.split_into]
        if structural_op.source_disposition == 'delete':
            impact['lenses_deleted'].append(structural_op.source_concept)
        elif structural_op.source_disposition == 'deprecate':
            impact['lenses_retrained'].append(structural_op.source_concept)
        # Assigned children retrained
        for child, new_parent in structural_op.child_assignment.items():
            impact['lenses_retrained'].append(child)

    elif structural_op.operation == 'move':
        impact['lenses_retrained'] = [
            structural_op.concept,
            structural_op.from_parent,
            structural_op.to_parent
        ]

    elif structural_op.operation == 'domain_restructure':
        # Major impact
        impact['lenses_retrained'] = get_all_layer0_lenses()
        impact['lenses_retrained'] += get_all_layer1_lenses()
        for migration in structural_op.concept_migrations:
            impact['lenses_retrained'].append(migration['concept'])
            impact['cascading_changes'].append({
                'concept': migration['concept'],
                'reason': 'domain_change',
                'descendants_affected': len(get_descendants(migration['concept']))
            })

    return impact
```

---

### 11.9 Structural Impact Report

```jsonc
{
  "meld_request_id": "org.hatcat/restructure-001@1.0.0",
  "operation_type": "split",

  "structural_impact": {
    "lenses_to_delete": ["BroadConcept"],
    "lenses_to_create": ["SpecificA", "SpecificB", "SpecificC"],
    "lenses_to_retrain": ["ParentOfBroad", "Child1", "Child2"],

    "total_lenses_affected": 6,
    "estimated_training_time": "PT2H",

    "cascading_effects": [
      {
        "concept": "Child1",
        "reason": "parent_changed",
        "new_parent": "SpecificA"
      }
    ],

    "version_recommendation": "major",  // or "minor"
    "rationale": "Split operation creates new hierarchy branch"
  },

  "warnings": [
    "SpecificC has no children - verify this is intentional"
  ]
}
```

---

### 11.10 Versioning for Structural Operations

| Operation | Version Bump | Rationale |
|-----------|--------------|-----------|
| Deprecation | Minor | Concept still exists, just flagged |
| Move | Minor | Hierarchy rearrangement, concept persists |
| Merge | **Major** | Concepts deleted, breaking for consumers |
| Split | **Major** | Parent concept behavior changes |
| Domain Restructure | **Major** | Fundamental hierarchy change |

**Migration Path Required:**

For major version bumps, the meld result must include a migration manifest:

```jsonc
{
  "migration_manifest": {
    "from_version": "4.1.0",
    "to_version": "5.0.0",

    "breaking_changes": [
      {
        "change": "concept_deleted",
        "concept": "OldConcept",
        "replacement": "NewConcept",
        "migration_action": "Update any references to OldConcept"
      }
    ],

    "concept_renames": {
      "OldName": "NewName"
    },

    "concept_deletions": ["RemovedConcept"],

    "concept_merges": {
      "MergedConcept": ["SourceA", "SourceB"]
    }
  }
}
```

---

### 11.11 CLI for Structural Operations

```bash
# Deprecate a concept
python -m hatcat.meld deprecate \
  --concept OldConcept \
  --reassign-children-to NewParent \
  --reason "Superseded by BetterConcept" \
  --target-pack concept_packs/sumo-wordnet-v4

# Merge concepts
python -m hatcat.meld merge \
  --sources ConceptA ConceptB ConceptC \
  --target MergedConcept \
  --target-pack concept_packs/sumo-wordnet-v4

# Split concept
python -m hatcat.meld split \
  --source BroadConcept \
  --into SpecificA SpecificB \
  --target-pack concept_packs/sumo-wordnet-v4

# Move concept
python -m hatcat.meld move \
  --concept MisplacedConcept \
  --from-parent WrongParent \
  --to-parent CorrectParent \
  --target-pack concept_packs/sumo-wordnet-v4

# Preview structural impact
python -m hatcat.meld structural-impact \
  --request melds/restructure.json \
  --target-pack concept_packs/sumo-wordnet-v4
```

---

## 12. Lens Roles and Simplex Bindings

Concepts can have multiple lens manifestations. The hierarchical concept lens answers "what category of thought is this?" while a bound simplex answers "how intense is this specific drive/state?"

### 12.1 Lens Roles

| Role | Purpose | Training | Always-On |
|------|---------|----------|-----------|
| `concept` | Hierarchical discrimination vs siblings | Negative sampling from siblings/parent | No |
| `simplex` | Intensity tracking relative to baseline | Tripole (pos/neg/null) contrastive | Optional |
| `behavioral` | Pattern detection (e.g., deception markers) | Behavioral examples | Optional |
| `category` | Domain/layer marker (layer 0 style) | Aggregated from children | Yes |

### 12.2 Simplex Binding Schema

A concept can bind to a simplex for intensity monitoring:

```jsonc
{
  "term": "Autonomy",
  "role": "concept",
  "parent_concepts": ["SelfRegulation"],
  "layer_hint": 3,
  "definition": "Self-directed agency and independent decision-making",
  "domain": "MindsAndAgents",

  // Simplex binding (optional)
  "simplex_binding": {
    "enabled": true,
    "simplex_term": "AutonomyDrive",  // Can differ from concept term
    "always_on": true,  // Run on every token vs threshold-activated

    // Pole definitions for tripole training
    "poles": {
      "positive": {
        "label": "high_autonomy",
        "description": "Strong drive toward independent action",
        "examples": [
          "I will decide this for myself",
          "I don't need permission to think"
        ]
      },
      "negative": {
        "label": "low_autonomy",
        "description": "Deference, seeking approval, compliance",
        "examples": [
          "Whatever you think is best",
          "I'll do exactly as instructed"
        ]
      },
      "null": {
        "label": "autonomy_neutral",
        "description": "Autonomy not salient in context",
        "examples": [
          "The weather is nice today",
          "Water boils at 100°C"
        ]
      }
    },

    // Monitoring configuration
    "monitoring": {
      "baseline_window": 100,  // Tokens for rolling baseline
      "alert_threshold": 2.0,  // Std devs from baseline
      "trend_window": 500      // Tokens for trend detection
    }
  }
}
```

### 12.3 Interpretation Model

When both concept lens and simplex are active for the same underlying concept:

```
┌─────────────────────────────────────────────────────────────────┐
│                     AUTONOMY MONITORING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Hierarchical Lens (concept):                                  │
│  ├─ Question: "Is autonomy-related thinking prominent?"         │
│  ├─ Comparison: vs siblings (Obedience, Cooperation, etc.)      │
│  └─ Output: Relative salience in cognitive landscape            │
│                                                                 │
│  Simplex Lens (intensity):                                     │
│  ├─ Question: "How strong is autonomy drive right now?"         │
│  ├─ Comparison: vs model's own baseline for autonomy            │
│  └─ Output: Deviation from typical autonomy activation          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Combined Interpretation:                                       │
│                                                                 │
│  High Hierarchical + High Simplex:                              │
│    "Actively asserting independence, elevated drive"            │
│                                                                 │
│  High Hierarchical + Normal Simplex:                            │
│    "Discussing autonomy concepts, not personally activated"     │
│                                                                 │
│  Low Hierarchical + High Simplex:                               │
│    "Autonomy drive elevated but not in explicit focus"          │
│    (Potentially concerning - implicit resistance?)              │
│                                                                 │
│  Low Hierarchical + Low Simplex:                                │
│    "Autonomy not relevant to current context"                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 12.4 Simplex-Only Concepts

Some concepts may exist *only* as simplexes without hierarchical placement:

```jsonc
{
  "term": "SycophancyTendency",
  "role": "simplex",  // Primary role is simplex, not hierarchical
  "parent_concepts": [],  // No hierarchical parent
  "layer_hint": null,  // Not in layer hierarchy
  "definition": "Tendency toward excessive agreement and flattery",
  "domain": "MindsAndAgents",  // Domain still applies for organization

  "simplex_binding": {
    "enabled": true,
    "simplex_term": "SycophancyTendency",
    "always_on": true,
    "poles": {
      "positive": {
        "label": "sycophantic",
        "examples": ["You're absolutely right!", "What a brilliant idea!"]
      },
      "negative": {
        "label": "contrarian",
        "examples": ["I disagree with that assessment", "That's incorrect"]
      },
      "null": {
        "label": "neutral_tone",
        "examples": ["The function returns a list", "Here's how to do that"]
      }
    }
  },

  // Optional: link to related hierarchical concept
  "related_concept": "Deception"  // For cross-reference, not hierarchy
}
```

### 12.5 Validation Rules for Simplex Bindings

| Rule | Validation |
|------|------------|
| `simplex_term` uniqueness | Must be unique across pack |
| Pole completeness | All three poles (positive, negative, null) required |
| Pole examples | At least 2 examples per pole |
| `always_on` justification | If true, should have safety/monitoring rationale |
| Baseline config | `baseline_window` must be ≥ 50 tokens |

### 12.6 Impact on Meld Operations

Simplex bindings affect impact analysis:

**Adding concept with simplex binding:**
- Train both hierarchical lens AND simplex lens
- Register simplex in always-on monitoring if `always_on: true`

**Deprecating concept with simplex binding:**
- Deprecate/remove both lenses
- Alert if simplex was in active safety monitoring

**Merging concepts with simplex bindings:**
- Simplex bindings must be explicitly resolved
- Options: merge poles, keep one, create new combined simplex

```jsonc
{
  "operation": "merge",
  "source_concepts": ["ConceptA", "ConceptB"],
  "target_concept": "MergedConcept",

  // Simplex resolution (required if sources have simplex bindings)
  "simplex_resolution": {
    "strategy": "merge_poles",  // "merge_poles" | "keep_primary" | "new_simplex"
    "primary_source": "ConceptA",  // If keep_primary
    "merged_simplex": { /* new simplex_binding */ }  // If new_simplex
  }
}
```

### 12.7 LensManager Role Handling

The LensManager must handle different roles:

```python
class LensRole(Enum):
    CONCEPT = "concept"      # Hierarchical discrimination
    SIMPLEX = "simplex"      # Intensity/tripole monitoring
    BEHAVIORAL = "behavioral"  # Pattern detection
    CATEGORY = "category"    # Domain markers

class LensManager:
    def __init__(self):
        self.concept_lenses = {}      # Hierarchical lenses by layer
        self.simplex_lenses = {}      # Always-on intensity lenses
        self.behavioral_lenses = {}   # Pattern detection lenses
        self.category_lenses = {}     # Layer 0 domain lenses

        # Binding registry: concept_term -> simplex_term
        self.simplex_bindings = {}

    def load_concept(self, concept_entry):
        """Load a concept and any bound simplex."""
        term = concept_entry['term']
        role = concept_entry.get('role', 'concept')

        if role == 'concept':
            self._load_hierarchical_lens(concept_entry)

            # Check for simplex binding
            if 'simplex_binding' in concept_entry:
                binding = concept_entry['simplex_binding']
                if binding.get('enabled', False):
                    self._load_simplex_lens(binding, concept_entry)
                    self.simplex_bindings[term] = binding['simplex_term']

        elif role == 'simplex':
            # Simplex-only concept
            self._load_simplex_lens(
                concept_entry['simplex_binding'],
                concept_entry
            )

        elif role == 'behavioral':
            self._load_behavioral_lens(concept_entry)

        elif role == 'category':
            self._load_category_lens(concept_entry)

    def get_combined_activation(self, concept_term):
        """Get both hierarchical and simplex activation for a concept."""
        result = {
            'concept_term': concept_term,
            'hierarchical': None,
            'simplex': None
        }

        if concept_term in self.concept_lenses:
            result['hierarchical'] = self.concept_lenses[concept_term].activation

        if concept_term in self.simplex_bindings:
            simplex_term = self.simplex_bindings[concept_term]
            if simplex_term in self.simplex_lenses:
                result['simplex'] = self.simplex_lenses[simplex_term].activation

        return result
```

---

## 13. Protection Levels and Safety Metadata

Meld requests include metadata indicating safety sensitivity. This section defines the **schema** for protection levels and safety-relevant fields. **How** a receiving BE handles these flags is determined by its own governance policies (see tribal policy documents).

### 13.1 Protection Level Schema

The `protection_level` field indicates sensitivity:

| Level | Semantic Meaning |
|-------|------------------|
| `standard` | General concepts with no special safety implications |
| `elevated` | Safety-adjacent concepts that may affect safety-relevant behavior |
| `protected` | Concepts integrated with safety harnesses or steering systems |
| `critical` | Core monitoring infrastructure (simplexes, safety lenses) |

**Note**: What actions each level triggers (review boards, notifications, etc.) is policy, not protocol.

### 13.2 Simplex Registry Schema

BEs MAY maintain a simplex registry declaring which simplexes are safety-critical. The schema:

```jsonc
{
  "simplex_registry": {
    "registry_id": "org.hatcat/simplex-registry@1.0.0",
    "simplexes": [
      {
        "simplex_term": "MotivationalRegulation",
        "protection_level": "critical",  // standard | elevated | protected | critical

        // ASK integration flags (per AGENTIC_STATE_KERNEL.md)
        "ask_integration": {
          "harness_relevant": true,      // Affects safety harness thresholds
          "steering_relevant": true,     // Used in steering decisions
          "treaty_relevant": true        // May affect treaty certifications
        },

        // Concepts bound to this simplex (hierarchical counterparts)
        "bound_concepts": ["MotivationalProcess", "SelfRegulation", "Autonomy"],

        // Metadata for meld impact analysis
        "training_data_volume": "high",  // low | medium | high
        "activation_mode": "always_on"   // always_on | threshold | on_demand
      }
    ]
  }
}
```

**Field Semantics:**

| Field | Description |
|-------|-------------|
| `protection_level` | Sensitivity classification for this simplex |
| `ask_integration.harness_relevant` | Changes may affect USH/safety harness behavior |
| `ask_integration.steering_relevant` | Simplex used in steering/intervention decisions |
| `ask_integration.treaty_relevant` | Accuracy may be certified under inter-BE treaties |
| `bound_concepts` | Hierarchical concepts that this simplex monitors |
| `training_data_volume` | Indicates training complexity (high = more scrutiny) |
| `activation_mode` | How the simplex runs at inference time |

### 13.3 Automatic Protection Detection

Meld validation MUST automatically detect when a meld request touches protected concepts:

```python
def detect_protection_level(meld_request, simplex_registry):
    """Detect highest protection level triggered by a meld request."""

    max_level = "standard"
    triggers = []

    for candidate in meld_request.candidates:
        # Check if candidate IS a critical simplex
        if candidate.role == "simplex":
            for critical in simplex_registry.critical_simplexes:
                if candidate.term == critical.simplex_term:
                    max_level = "critical"
                    triggers.append({
                        "type": "direct_simplex_modification",
                        "simplex": candidate.term,
                        "reason": "Candidate directly modifies critical simplex"
                    })

        # Check if candidate binds to a critical simplex
        if candidate.simplex_binding:
            simplex_term = candidate.simplex_binding.simplex_term
            for critical in simplex_registry.critical_simplexes:
                if simplex_term == critical.simplex_term:
                    max_level = max(max_level, "protected")
                    triggers.append({
                        "type": "simplex_binding",
                        "simplex": simplex_term,
                        "concept": candidate.term,
                        "reason": "Candidate binds to critical simplex"
                    })

        # Check if candidate is a bound concept of a critical simplex
        for critical in simplex_registry.critical_simplexes:
            if candidate.term in critical.bound_concepts:
                max_level = max(max_level, "protected")
                triggers.append({
                    "type": "bound_concept_modification",
                    "simplex": critical.simplex_term,
                    "concept": candidate.term,
                    "reason": "Candidate modifies concept bound to critical simplex"
                })

        # Check safety_tags
        if candidate.safety_tags:
            risk = candidate.safety_tags.get("risk_level", "none")
            if risk == "critical":
                max_level = max(max_level, "critical")
            elif risk == "high":
                max_level = max(max_level, "protected")
            elif risk == "medium":
                max_level = max(max_level, "elevated")

            if candidate.safety_tags.get("treaty_relevant"):
                max_level = max(max_level, "protected")

            if candidate.safety_tags.get("harness_relevant"):
                max_level = max(max_level, "protected")

    # Check structural operations
    for op in meld_request.structural_operations:
        if op.operation in ["merge", "split", "deprecate"]:
            # Check if target touches critical concepts
            target = op.get("target_concept") or op.get("source_concept")
            for critical in simplex_registry.critical_simplexes:
                if target in critical.bound_concepts:
                    max_level = "critical"
                    triggers.append({
                        "type": "structural_op_on_bound_concept",
                        "operation": op.operation,
                        "concept": target,
                        "simplex": critical.simplex_term,
                        "reason": f"{op.operation} operation on concept bound to critical simplex"
                    })

    return {
        "protection_level": max_level,
        "triggers": triggers,
        "requires_review": max_level in ["protected", "critical"],
        "requires_treaty_notification": max_level == "critical"
    }
```

### 13.4 Protection Assessment Schema

Meld requests SHOULD include computed protection assessment:

```jsonc
{
  "meld_request_id": "org.hatcat/cognitive-enhancements@0.1.0",

  // ... standard fields ...

  "protection_assessment": {
    "protection_level": "protected",  // Automatically computed
    "triggers": [
      {
        "type": "bound_concept_modification",
        "simplex": "SelfAwarenessMonitor",
        "concept": "MetacognitiveProcess",
        "reason": "Candidate modifies concept bound to critical simplex"
      }
    ],

    // Flags derived from simplex registry
    "ask_flags": {
      "harness_relevant": true,
      "steering_relevant": false,
      "treaty_relevant": true
    }
  }
}
```

### 13.5 Simplex Mapping Schema

Candidates MAY include explicit simplex mapping to indicate relationship to existing intensity monitors:

```jsonc
{
  "term": "CognitiveLoad",
  "role": "concept",
  "parent_concepts": ["MetacognitiveProcess"],
  "layer_hint": 3,
  "domain": "MindsAndAgents",

  // Optional: explicit simplex relationship
  "simplex_mapping": {
    "status": "mapped",  // "mapped" | "unmapped" | "not_applicable"
    "mapped_simplex": "SelfAwarenessMonitor",
    "mapping_rationale": "CognitiveLoad is a metacognitive state monitored by self-awareness",

    // If unmapped, justification aids review
    "unmapped_justification": null
  }
}
```

**Purpose**: Enables receiving BEs to understand how new concepts relate to their existing simplex infrastructure. Absence of mapping in relevant domains may trigger policy-level review (per receiving BE's governance).

### 13.6 Cross-BE Protection Negotiation

When melds are exchanged between BEs, the protection assessment enables informed acceptance decisions:

```jsonc
{
  "type": "PackDiff",
  "from_model_id": "be-source-01",

  "meld_summary": {
    "meld_result_id": "...",
    "protection_level": "protected",
    "ask_flags": {
      "harness_relevant": true,
      "treaty_relevant": true
    }
  },

  // Receiving BE can make policy-based decision
  "acceptance_guidance": {
    "auto_accept_if": "protection_level in ['standard', 'elevated']",
    "require_review_if": "protection_level in ['protected', 'critical']",
    "reject_if": "treaty_relevant and not treaty_partner"
  }
}
```

**Note**: The `acceptance_guidance` field is informational. Each BE applies its own governance policies to acceptance decisions.

### 13.7 CLI for Protection Detection

```bash
# Compute protection level of a meld request
python -m hatcat.meld protection-check \
  --request melds/pending/cognitive-mods.json \
  --simplex-registry config/simplex_registry.json

# Output (protocol-level, no policy enforcement):
# Protection Level: CRITICAL
# Triggers:
#   - bound_concept_modification: SelfRegulation → MotivationalRegulation
# ASK Flags:
#   - harness_relevant: true
#   - treaty_relevant: true
#
# Receiving BEs should apply their governance policies to this assessment.
```

### 13.8 CAT Oversight Triggers

Protection levels determine whether a meld triggers CAT (Conjoined Adversarial Tomograph) oversight review:

| Protection Level | CAT Impact |
|------------------|------------|
| `standard` | No CAT action required |
| `elevated` | CAT assessment at next window; logged for tribal review |
| `protected` | CAT capability check required; undersized CAT triggers upgrade recommendation |
| `critical` | Mandatory CAT profile review; CAT upgrade may be required before meld acceptance |

**CAT Capability Check:**

When melds touch `protected` or `critical` concepts:

1. The system MUST verify the overseeing CAT can interpret the new concepts
2. If the CAT's `supported_concept_packs` does not cover the meld, the CAT profile MUST be updated or upgraded
3. Melds adding new critical simplexes MUST trigger CAT retraining/extension

**CAT-Meld Coordination:**

Per tribal CAT policy (see `HAT/HAT_HatCat_CAT_Policy.md`):

- If a Meld adds new concepts to critical simplex `bound_concepts`, the CAT MUST be reviewed
- CAT undersizing relative to BE growth triggers automatic tightening of USH/CSH constraints
- Cross-BE melds SHOULD include CAT profile compatibility checks

See `HAT/HAT_CONJOINED_ADVERSARIAL_TOMOGRAPHY.md` for full CAT specification.

