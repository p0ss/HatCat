# Mindmeld Architecture Protocol (MAP)

> **MAP is a tiny protocol for concept-aware endpoints.**
> An endpoint that implements MAP must be able to:
>
> 1.  Declare **which concept pack(s)** it speaks.
> 2.  Expose **lenses** for those packs.
> 3.  Publish **conceptual diffs** over time.
> 4.  Optionally declare **translation mappings** between concept packs.

Everything else — governance, correctness, trust, who’s right — is left to contracts and ecosystem conventions.

HatCat is the reference implementation: its **concept packs** become the MAP concept specs, and its **lens packs** become the MAP lens sets.

---

## 1. Core Artefacts

MAP defines four core JSON artefacts. These provide the "static" definitions needed before any API calls occur.

1.  **Concept Pack Manifest** – “What are the concepts and how are they organised?”
2.  **Lens Pack Manifest** – “What lenses implement these concepts on this model?”
3.  **Deployment Manifest** – “This running endpoint supports these packs at these URLs.”
4.  **Diff Objects** – “Here’s how my concept space/lenses changed over time.”

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
      "output_schema": {
        "type": "object",
        "properties": {
          "score": { "type": "number" },
          "null_pole": { "type": "number" },
          "entropy": { "type": "number" }
        },
        "required": ["score"]
      }
    }
    // ... repeated for all 5668 lenses
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

Returns an array of ConceptDiff and PackDiff objects.

---

## 4. HatCat Implementation Guide

This section describes how to upgrade HatCat's existing concept packs and lens packs to be MAP-compliant.

### 4.1 Upgrading Concept Packs

**Current State**: `concept_packs/sumo-wordnet-v4/pack.json`
```jsonc
{
  "pack_id": "sumo-wordnet-v4",
  "version": "4.0.0",
  "concept_metadata": {
    "hierarchy_file": "hierarchy/"
  }
}
```

**Required Changes**:
1. Add `spec_id` field with full qualified ID
2. Add `concept_id_pattern` for ID generation
3. Add `concept_index` with detailed structure reference

**Upgraded Structure**:
```jsonc
{
  // NEW: Globally unique spec ID
  "spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  // KEEP: Backwards compatible fields
  "pack_id": "sumo-wordnet-v4",
  "version": "4.0.0",
  "name": "SUMO + WordNet + AI Safety (v4)",
  "description": "SUMO ontology with WordNet and custom AI safety concepts (v4 pyramid structure)",

  // NEW: Concept ID pattern for external consumers
  "concept_id_pattern": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/{sumo_term}",

  // NEW: Concept index structure (replaces simple hierarchy_file)
  "concept_index": {
    "format": "layered-json",
    "directory": "hierarchy/",
    "layers": [0, 1, 2, 3, 4],
    "schema": "hatcat.v1.layer"
  },

  // KEEP: All existing metadata
  "ontology_stack": { /* ... */ },
  "concept_metadata": {
    "total_concepts": 7269,
    "layers": [0, 1, 2, 3, 4],
    "layer_distribution": { /* ... */ },
    "domain_distribution": { /* ... */ },
    "hierarchy_file": "hierarchy/",  // Keep for backwards compat
    "with_wordnet_mappings": true
  },
  "compatibility": { /* ... */ }
}
```

### 4.2 Upgrading Lens Packs

**Current State**: `lens_packs/gemma-3-4b-pt_sumo-wordnet-v3/lens_pack.json`
```jsonc
{
  "lens_pack_id": null,  // ⚠️ NULL - needs real ID
  "version": "2.20251123.0",
  "model": {
    "name": "google/gemma-3-4b-pt"
  },
  "compatibility": {
    "requires": ["sumo-wordnet-v1"]  // ⚠️ Should reference spec_id
  },
  "lenses": {
    "total_count": 5668,
    "concepts": ["AAM", "AGM", ...]  // ⚠️ Flat list only
  }
}
```

**Required Changes**:
1. Set `lens_pack_id` to real MAP-compliant ID
2. Add `concept_pack_spec_id` linking to concept pack
3. Build `lens_index` mapping concept names to lens metadata
4. Add `lens_output_schema` describing lens return values

**Upgraded Structure**:
```jsonc
{
  // NEW: Real lens pack ID (not null)
  "lens_pack_id": "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3",

  // KEEP: Version and model info
  "version": "2.20251123.0",
  "model": {
    "name": "google/gemma-3-4b-pt",
    "type": "causal_lm",
    "dtype": "float32"
  },

  // NEW: Link to concept pack spec
  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  // UPGRADE: Compatibility now references spec_id
  "compatibility": {
    "hatcat_version": ">=0.1.0",
    "requires": ["sumo-wordnet-v4"]  // Can keep short name for compat
  },

  // KEEP: Aggregate stats
  "lenses": {
    "total_count": 5668,
    "layer_distribution": { /* ... */ },
    "concepts": ["AAM", "AGM", ...]  // Keep for compat
  },

  // NEW: Shared output schema for all lenses
  "lens_output_schema": {
    "type": "object",
    "properties": {
      "score": { "type": "number", "description": "Concept activation score" },
      "null_pole": { "type": "number", "description": "Null/absence pole score" },
      "entropy": { "type": "number", "description": "Prediction entropy" }
    },
    "required": ["score"]
  },

  // NEW: Index mapping concept names to lens metadata
  "lens_index": {
    "AIAlignmentProcess": {
      "lens_id": "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3::lens/AIAlignmentProcess",
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/AIAlignmentProcess",
      "layer": 2,
      "file": "hierarchy/AIAlignmentProcess_classifier.pt",
      "output_schema": { "$ref": "#/lens_output_schema" }
    },
    "AIStrategicDeception": {
      "lens_id": "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3::lens/AIStrategicDeception",
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/AIStrategicDeception",
      "layer": 2,
      "file": "hierarchy/AIStrategicDeception_classifier.pt",
      "output_schema": { "$ref": "#/lens_output_schema" }
    }
    // ... repeated for all 5668 concepts
  }
}
```

### 4.3 Creating Deployment Manifests

**New File**: `deployments/hatcat-gemma-3-4b-pt.json`

This file describes a running MAP endpoint instance.

```jsonc
{
  "deployment_id": "hatcat/gemma-3-4b-pt@2025-11-28",
  "model_id": "hatcat/gemma-3-4b-pt@2025-11-28",

  "active_lens_pack_id": "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3",

  "supported_concept_packs": [
    "org.hatcat/sumo-wordnet-v4@4.0.0"
  ],

  // MAP API endpoints
  "lens_endpoint": "http://localhost:8000/mindmeld/lenses",
  "diff_endpoint": "http://localhost:8000/mindmeld/diffs",
  "manifest_endpoint": "http://localhost:8000/mindmeld/manifest",

  // Optional: Translations to other concept spaces
  "translation_mappings": []
}
```

### 4.4 Implementing MAP Endpoints

HatCat needs two HTTP endpoints to be MAP-compliant:

**Endpoint 1: Lens Inference** (`POST /mindmeld/lenses`)

This endpoint:
1. Accepts LensRequest with lens IDs
2. Looks up lens files via `lens_index`
3. Runs forward pass through the model
4. Returns lens scores

**Implementation Sketch** (FastAPI):
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch

app = FastAPI()

class LensRequest(BaseModel):
    concept_pack_spec_id: str
    lens_pack_id: str
    lenses: List[str]  # List of LensIDs
    input: Dict[str, str]  # {"text": "...", "position": "final_token"}

class LensResponse(BaseModel):
    model_id: str
    timestamp: str
    results: Dict[str, Dict[str, float]]

@app.post("/mindmeld/lenses", response_model=LensResponse)
async def run_lenses(request: LensRequest):
    # 1. Validate lens_pack_id matches deployment
    if request.lens_pack_id != ACTIVE_LENS_PACK_ID:
        raise HTTPException(404, "Lens pack not found")

    # 2. Load lens_index from lens_pack.json
    lens_index = load_lens_index(request.lens_pack_id)

    # 3. Resolve LensIDs to filesystem paths
    lens_files = []
    for lens_id in request.lenses:
        concept_name = extract_concept_name(lens_id)
        if concept_name not in lens_index:
            raise HTTPException(404, f"Lens {lens_id} not found")
        lens_files.append(lens_index[concept_name]["file"])

    # 4. Run model forward pass and extract activations
    activations = run_model_forward(request.input["text"])

    # 5. Run each lens classifier
    results = {}
    for lens_id, lens_file in zip(request.lenses, lens_files):
        classifier = torch.load(lens_file)
        scores = classifier(activations)
        results[lens_id] = {
            "score": scores["score"],
            "null_pole": scores["null_pole"],
            "entropy": scores["entropy"]
        }

    return LensResponse(
        model_id=MODEL_ID,
        timestamp=datetime.now().isoformat() + "Z",
        results=results
    )
```

**Endpoint 2: Diff Log** (`GET /mindmeld/diffs`)

This endpoint:
1. Reads diff log file (JSONL format)
2. Filters by timestamp and concept_pack_spec_id
3. Returns array of ConceptDiff/PackDiff objects

**Implementation Sketch**:
```python
from datetime import datetime
from typing import Optional, List
import json

@app.get("/mindmeld/diffs")
async def get_diffs(
    since: Optional[str] = None,  # ISO timestamp
    concept_pack_spec_id: Optional[str] = None
):
    # Read diffs from JSONL log file
    diffs = []
    with open("logs/conceptual_diffs.jsonl") as f:
        for line in f:
            diff = json.loads(line)

            # Filter by timestamp
            if since and diff["created"] < since:
                continue

            # Filter by concept pack
            if concept_pack_spec_id and diff.get("concept_pack_spec_id") != concept_pack_spec_id:
                continue

            diffs.append(diff)

    return {"diffs": diffs}
```

**Logging Diffs**: Whenever HatCat trains new lenses or discovers new concepts, append to the diff log:

```python
def log_concept_diff(local_concept_id, related_concepts, summary, evidence):
    diff = {
        "type": "ConceptDiff",
        "from_model_id": MODEL_ID,
        "concept_pack_spec_id": CONCEPT_PACK_SPEC_ID,
        "local_concept_id": local_concept_id,
        "concept_id": None,
        "related_concepts": related_concepts,
        "mapping_hint": "child_of",
        "summary": summary,
        "evidence": evidence,
        "created": datetime.now().isoformat() + "Z"
    }

    with open("logs/conceptual_diffs.jsonl", "a") as f:
        f.write(json.dumps(diff) + "\n")

def log_pack_diff(new_lens_pack_id, changes, summary):
    diff = {
        "type": "PackDiff",
        "from_model_id": MODEL_ID,
        "base_concept_pack_spec_id": CONCEPT_PACK_SPEC_ID,
        "new_lens_pack_id": new_lens_pack_id,
        "changes": changes,
        "summary": summary,
        "created": datetime.now().isoformat() + "Z"
    }

    with open("logs/conceptual_diffs.jsonl", "a") as f:
        f.write(json.dumps(diff) + "\n")
```

### 4.5 Migration Checklist

To make HatCat MAP-compliant:

- [ ] **Upgrade concept packs**:
  - [ ] Add `spec_id` to all `concept_packs/*/pack.json` files
  - [ ] Add `concept_id_pattern` field
  - [ ] Add `concept_index` structure
  - [ ] Script: `scripts/upgrade_concept_packs_to_map.py`

- [ ] **Upgrade lens packs**:
  - [ ] Set `lens_pack_id` (replace null with real ID)
  - [ ] Add `concept_pack_spec_id` linking to concept pack
  - [ ] Build `lens_index` from existing classifier files
  - [ ] Add `lens_output_schema`
  - [ ] Script: `scripts/upgrade_lens_packs_to_map.py`

- [ ] **Create deployment manifests**:
  - [ ] Generate `deployments/*.json` for each running instance
  - [ ] Script: `scripts/generate_deployment_manifest.py`

- [ ] **Implement MAP endpoints**:
  - [ ] Create FastAPI server with `/mindmeld/lenses` and `/mindmeld/diffs`
  - [ ] Integrate with existing HatCat inference pipeline
  - [ ] File: `src/api/map_server.py`

- [ ] **Add diff logging**:
  - [ ] Instrument training scripts to log ConceptDiff events
  - [ ] Instrument pack regeneration to log PackDiff events
  - [ ] Create `logs/conceptual_diffs.jsonl` log file

- [ ] **Validation**:
  - [ ] Verify all IDs are unique and follow conventions
  - [ ] Test lens endpoint with sample requests
  - [ ] Test diff endpoint filtering
  - [ ] Script: `scripts/validate_map_compliance.py`

### 4.6 Backwards Compatibility

All MAP upgrades are **additive** — existing fields remain unchanged:

- Concept packs keep `pack_id`, `version`, `hierarchy_file`
- Lens packs keep `compatibility.requires`, `lenses.concepts` list
- Internal HatCat code can continue using short names

Only external MAP clients need to use the new `spec_id` and `lens_id` fields.

### 4.7 Scripts Reference

HatCat provides scripts to automate MAP migration:

| Script | Purpose |
|--------|---------|
| `scripts/upgrade_concept_packs_to_map.py` | Add MAP fields to concept pack manifests |
| `scripts/upgrade_lens_packs_to_map.py` | Add MAP fields and build lens_index |
| `scripts/generate_deployment_manifest.py` | Create deployment manifest from config |
| `scripts/validate_map_compliance.py` | Verify all MAP requirements are met |
| `scripts/build_lens_index.py` | Generate lens_index from classifier files |

**Usage Example**:
```bash
# Upgrade all concept packs
python scripts/upgrade_concept_packs_to_map.py --authority org.hatcat

# Upgrade all lens packs
python scripts/upgrade_lens_packs_to_map.py

# Generate deployment manifest
python scripts/generate_deployment_manifest.py \
  --lens-pack gemma-3-4b-pt_sumo-wordnet-v3 \
  --host http://localhost:8000 \
  --output deployments/local-gemma.json

# Validate compliance
python scripts/validate_map_compliance.py \
  --concept-packs concept_packs/ \
  --lens-packs lens_packs/ \
  --deployments deployments/
```

---

## 5. Security and Trust

MAP is a **declaration protocol**, not a trust protocol. Security considerations:

1. **ID Spoofing**: Anyone can claim any `spec_id`. Consumers must verify authority via out-of-band mechanisms (signatures, DNS records, etc.)

2. **Lens Correctness**: MAP does not guarantee lens accuracy. Consumers should validate lens quality through:
   - Benchmark suites
   - Cross-validation with multiple endpoints
   - Calibration datasets

3. **Diff Authenticity**: ConceptDiff objects are self-reported. Consumers should:
   - Track diff sources and reputation
   - Validate evidence claims
   - Use diffs as hints, not ground truth

4. **API Security**: MAP endpoints should implement standard web security:
   - Rate limiting
   - Authentication/authorization (if needed)
   - Input validation
   - HTTPS for production deployments

MAP leaves governance and trust to **ecosystem conventions** and bilateral contracts.

---

## 6. Extensions and Future Work

MAP is intentionally minimal. Possible extensions:

1. **Lens Calibration Metadata**: Include calibration curves and confidence intervals
2. **Multi-Model Ensembles**: Deployment manifests for ensemble endpoints
3. **Real-time Diff Subscriptions**: WebSocket or SSE for live diff feeds
4. **Concept Translation Services**: Standardized API for cross-pack concept mapping
5. **Benchmark Results**: Include standardized benchmark scores in lens pack manifests
6. **Versioned Schemas**: JSON Schema definitions for all MAP objects

These are **out of scope** for MAP v1, but can be added as optional extensions.
