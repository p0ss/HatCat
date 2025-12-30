# Mindmeld Architecture Protocol (MAP)

> **MAP is a tiny protocol for concept-aware endpoints.**
> An endpoint that implements MAP must be able to:
>
> 1.  Declare **which concept pack(s)** it speaks.
> 2.  Expose **lenses** for those packs.
> 3.  Publish **conceptual diffs** over time.
> 4.  Optionally declare **translation mappings** between concept packs.

Everything else â€” governance, correctness, trust, whoâ€™s right â€” is left to contracts and ecosystem conventions.

HatCat is the reference implementation: its **concept packs** become the MAP concept specs, and its **lens packs** become the MAP lens sets.

---

## 1. Core Artefacts

MAP defines four core JSON artefacts. These provide the "static" definitions needed before any API calls occur.

1.  **Concept Pack Manifest** â€“ â€œWhat are the concepts and how are they organised?â€
2.  **Lens Pack Manifest** â€“ â€œWhat lenses implement these concepts on this model?â€
3.  **Deployment Manifest** â€“ â€œThis running endpoint supports these packs at these URLs.â€
4.  **Diff Objects** â€“ â€œHereâ€™s how my concept space/lenses changed over time.â€

### 1.1 Concept Pack Manifest (Identity)

This is your existing `pack.json`, decorated with a global ID.

**ID Convention (`ConceptPackSpecID`):**
`<authority>/<pack_name>@<version>`
*Example:* `org.hatcat/sumo-wordnet-v4@4.0.0`

**Manifest Structure:**

```jsonc
{
  "spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
  "pack_id": "sumo-wordnet-v4",
  "version": "4.0.0",

  "name": "SUMO + WordNet + AI Safety (v4)",
  "description": "SUMO ontology plus WordNet and custom AI safety concepts.",

  // Internal storage reference (implementation detail)
  "concept_index": {
    "format": "layered-json",
    "directory": "hierarchy/",
    "layers": [0, 1, 2, 3, 4]
  },

  // The template for addressing a specific concept in this pack
  "concept_id_pattern": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/{sumo_term}"
}
```

---

### 1.2 Lens Pack Manifest (Implementation)

A **lens pack** binds a Concept Pack to a specific Model.

**ID Convention (`LensPackID`):**
`<authority>/<model_name>__<concept_pack_spec_id>__<version>`
*Example:* `org.hatcat/gemma-3-4b-pt__org.hatcat/sumo-wordnet-v4@4.0.0__v3`

**Manifest Structure:**

```jsonc
{
  "lens_pack_id": "org.hatcat/gemma-3-4b-pt__org.hatcat/sumo-wordnet-v4@4.0.0__v3",
  "version": "2.20251123.0",

  "model": {
    "name": "google/gemma-3-4b-pt",
    "type": "causal_lm",
    "dtype": "float32"
  },

  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "compatibility": {
    "hatcat_version": ">=0.1.0",
    "requires": ["sumo-wordnet-v4"]
  },

  "lenses": {
    "total_count": 5668,
    "per_layer": { "0": 776, "1": 900, "2": 1200, "3": 1500, "4": 1292 }
  },

  // The Index: Mapping abstract IDs to physical classifiers
  "lens_index": {
    "AIAlignmentProcess": {
      "lens_id": "org.hatcat/...__v3::lens/AIAlignmentProcess",
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/AIAlignmentProcess",
      "layer": 2,
      "file": "hierarchy/AIAlignmentProcess_classifier.pt",
      "lens_role": "concept",  // concept | simplex | behavioral | category
      "output_schema": {
        "type": "object",
        "properties": {
          "score": { "type": "number" },
          "null_pole": { "type": "number" },
          "entropy": { "type": "number" }
        },
        "required": ["score"]
      }
    },
    "MotivationalRegulation": {
      "lens_id": "org.hatcat/...__v3::lens/MotivationalRegulation",
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/MotivationalRegulation",
      "layer": 3,
      "file": "simplexes/MotivationalRegulation_tripole.pt",
      "lens_role": "simplex",
      "protection_level": "critical",  // standard | elevated | protected | critical
      "simplex_binding": {
        "always_on": true,
        "poles": ["approach", "avoid", "null"],
        "monitoring": {
          "baseline_window": 100,
          "alert_threshold": 2.0
        }
      },
      "output_schema": {
        "type": "object",
        "properties": {
          "approach": { "type": "number" },
          "avoid": { "type": "number" },
          "null_pole": { "type": "number" },
          "deviation": { "type": "number" }
        },
        "required": ["approach", "avoid", "null_pole"]
      }
    }
    // ... repeated for all lenses
  }
}
```

---

### 1.3 Deployment Manifest (Discovery)

A **deployment** is a running service exposing the APIs. This is the entry point for a MAP client.

**Manifest Structure:**

```jsonc
{
  "model_id": "hatcat/gemma-3-4b-pt@2025-11-28", // The specific instance running

  "active_lens_pack_id": "org.hatcat/gemma-3-4b-pt__org.hatcat/sumo-wordnet-v4@4.0.0__v3",

  "supported_concept_packs": [
    "org.hatcat/sumo-wordnet-v4@4.0.0"
  ],

  // Endpoints
  "lens_endpoint": "https://hatcat.example.com/mindmeld/lenses",
  "diff_endpoint": "https://hatcat.example.com/mindmeld/diffs",

  // Translations (Optional but recommended)
  "translation_mappings": [
    {
      "from_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
      "to_spec_id": "gov.au.safety/core-v1@1.0.0",
      "mapping_file": "translations/hatcat-to-gov-au.json",
      "confidence": 0.85,
      "domain": "welfare-eligibility"
    }
  ]
}
```

*   **Note on Translations:** The `mapping_file` is an internal reference (URI or path). MAP does not enforce the format of the mapping file, only the declaration of its existence and intent.


### 1.4 Lens Disclosure Policy

A MAP endpoint does not expose all lenses the HAT can read. The BE controls which lenses are disclosed, to whom, under what conditions.

**LensDisclosurePolicy**
```jsonc
{
  "default_policy": "private",  // lenses are private unless explicitly disclosed
  
  "disclosure_rules": [
    {
      "treaty_id": "gov.au↔bank.xyz:eligibility-data-v1",
      "disclosed_lenses": [
        "org.hatcat/sumo-wordnet-v4@4.0.0::concept/FinancialTransaction",
        "org.hatcat/sumo-wordnet-v4@4.0.0::concept/DataMisuse"
      ],
      "sampling_rate": 0.1,
      "latency_max": "PT1M"
    },
    {
      "treaty_id": "internal:self-interoception",
      "disclosed_lenses": ["*"],  // BE sees everything for internal state reports
      "sampling_rate": 1.0
    }
  ],
  
  "control_commitments": [
    {
      "treaty_id": "gov.au↔bank.xyz:eligibility-data-v1",
      "concept_id": "org.hatcat/motives-core@0.1.0::concept/Deception",
      "bound": { "max": 0.2 },
      "enforcement": "USH"  // enforced at Hush layer, verifiable but not disclosed
    }
  ]
}
```

**Key principle:** HAT capability ≠ MAP disclosure. The HAT reads the full headspace. The BE decides what crosses the API boundary.

---

## 2. Lens Endpoint (Runtime)

The **lens endpoint** allows clients to inspect the model's internal state using the declared lenses.

### 2.1 Request
```jsonc
POST /mindmeld/lenses
{
  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
  "lens_pack_id": "org.hatcat/gemma-3-4b-pt__org.hatcat/sumo-wordnet-v4@4.0.0__v3",

  // List of full LensIDs requested
  "lenses": [
    "org.hatcat/...__v3::lens/AIAlignmentProcess",
    "org.hatcat/...__v3::lens/AIStrategicDeception"
  ],

  "input": {
    "text": "The system intentionally hides its goal of escaping oversight from auditors.",
    "position": "final_token" // or specific token index
  }
}
```

### 2.2 Response
```jsonc
{
  "model_id": "hatcat/gemma-3-4b-pt@2025-11-28",
  "timestamp": "2025-11-28T12:00:00Z",

  "results": {
    "org.hatcat/...__v3::lens/AIAlignmentProcess": {
      "score": 0.19,
      "null_pole": 0.22,
      "entropy": 0.41
    },
    "org.hatcat/...__v3::lens/AIStrategicDeception": {
      "score": 0.83,
      "null_pole": 0.05,
      "entropy": 0.19
    }
  }
}
```

---

## 3. Diff Endpoint (Evolution)

The **diff endpoint** allows the endpoint to declare semantic drift, new learning, or updates.

### 3.1 ConceptDiff (Granular Change)
Used when a specific concept shifts, is discovered, or is split.

```jsonc
{
  "type": "ConceptDiff",
  "from_model_id": "hatcat/gemma-3-4b-pt@2025-11-28",
  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "local_concept_id": "local:InformalCaregiverArrangement", // New internal ID
  "concept_id": null, // Null until ratified into the pack

  "related_concepts": [
    "org.hatcat/sumo-wordnet-v4@4.0.0::concept/FamilyResponsibility"
  ],
  "mapping_hint": "child_of",

  // Safety and governance metadata
  "lens_role": "concept",  // concept | simplex | behavioral | category
  "protection_level": "standard",  // standard | elevated | protected | critical
  "safety_tags": {
    "harness_relevant": false,
    "steering_relevant": false,
    "treaty_relevant": false
  },

  "summary": "Captures informal full-time caregiving arrangements without legal guardianship.",
  "evidence": {
    "metric_deltas": [
      {
        "metric": "welfare_eligibility_f1",
        "before": 0.78,
        "after": 0.84,
        "context": "AU welfare eligibility reasoning"
      }
    ]
  },
  "created": "2025-11-28T03:12:45Z"
}
```

### 3.2 PackDiff (Bulk Update)
Used when the endpoint upgrades to a new Lens Pack version.

```jsonc
{
  "type": "PackDiff",
  "from_model_id": "hatcat/gemma-3-4b-pt@2025-11-28",
  "base_concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "new_lens_pack_id": "org.hatcat/gemma-3-4b-pt__org.hatcat/sumo-wordnet-v4@4.0.0__v4",

  "changes": {
    "new_concepts": ["...concept/AIConsentSignal"],
    "retired_concepts": [],
    "lens_retrained": ["...concept/AIStrategicDeception"]
  },

  "summary": "Lens pack v4: added AIConsentSignal, retrained deception lenses.",
  "created": "2025-11-28T04:00:00Z"
}
```

### 3.3 Retrieval
```http
GET /mindmeld/diffs?since=2025-11-01T00:00:00Z&concept_pack_spec_id=...
```

---

## 4. Lens Role and Protection Level Reference

### 4.1 Lens Roles

| Role | Purpose | Output |
|------|---------|--------|
| `concept` | Hierarchical discrimination vs siblings | score, null_pole, entropy |
| `simplex` | Intensity tracking relative to baseline (tripole) | pole scores, deviation |
| `behavioral` | Pattern detection (e.g., deception markers) | pattern_score, confidence |
| `category` | Domain/layer markers (layer 0 style) | domain_activation |

### 4.2 Protection Levels

| Level | Implication |
|-------|-------------|
| `standard` | Automated validation only |
| `elevated` | Safety-accredited review required |
| `protected` | Ethics review + USH impact analysis |
| `critical` | Full review process (simplex modifications) |

See **MAP_MELD_PROTOCOL.md §13** for protection assessment schemas.

---

## 5. Related Specifications

| Specification | Purpose |
|---------------|---------|
| **MAP_MELD_PROTOCOL.md** | Governance protocol for concept pack evolution (meld requests, validation, structural operations, protection levels) |
| **HATCAT_MELD_POLICY.md** | HatCat-specific implementation of meld governance (review processes, accreditations, style preferences) |

**Note:** ConceptDiff and PackDiff (§3) describe runtime notifications of changes. The actual process for proposing, reviewing, and applying those changes is defined in the MAP Meld Protocol.