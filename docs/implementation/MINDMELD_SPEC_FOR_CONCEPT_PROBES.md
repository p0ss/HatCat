Gotcha, that helps a lot. Let’s treat **concept packs = specs**, **lens packs = lens sets bound to a model+spec**, and then just bolt on the minimal MAP bits: IDs, declarations, and diffs.

Below is a concrete “uplift plan” from your current files to a MAP-style setup, *using your existing JSON as-is wherever possible*.

---

## 0. Where you are now (very briefly)

### Concept pack: `sumo-wordnet-v4`

* `pack.json` gives you:

  * `pack_id: "sumo-wordnet-v4"`, `version: "4.0.0"`,
  * ontology stack (SUMO + WordNet + your AI safety extensions),
  * high-level stats: total concepts, layers, domains, etc. 
* README explains the structure and that this is model-agnostic, and enumerates the custom AI-safety concepts. 
* Layer files (`layer0.json`–`layer4.json`) enumerate concepts with rich metadata:

  * `sumo_term`, `layer`, `domain`, parents/children, WordNet synsets, definitions, etc.     

This is already a **de facto spec** for a concept space.

### Lens pack: Gemma-3-4B lenses

* `lens_pack.json` has:

  * `version: "2.20251123.0"`,
  * `model.name: "google/gemma-3-4b-pt"`,
  * `description`,
  * `compatibility.requires: ["sumo-wordnet-v1"]` (points at required concept pack),
  * a `lenses` section with counts, per-layer distribution, and a flat `concepts` list. 
* README for the lens pack describes:

  * total lenses (5668),
  * per-layer counts,
  * training details,
  * and that lenses live under `hierarchy/*_classifier.pt`. 

This is already **almost** a MAP lens set for a given (model, concept pack).

---

## 1. New names / IDs: how these map into MAP

We *don’t* invent “PalaceSpecs”. We just re-label what you already have:

* **ConceptPackSpec** (MAP’s “spec”)

  * Backed by your `pack.json` + `hierarchy/layer*.json` etc.
* **LensPackSpec** (MAP’s lens set)

  * Backed by `lens_pack.json` + the `hierarchy/*_classifier.pt` files.
* **DeploymentManifest**

  * New, tiny JSON per deployed model+lens_pack (where to send lens calls, where to read diffs).
* **ConceptDiff / PackDiff**

  * New, small JSON messages that describe conceptual changes relative to a ConceptPackSpec.

---

## 2. Uplift 1: Concept Pack → ConceptPackSpec (MAP-compatible)

### 2.1 Add a `spec_id` and explicit identity

Right now:

```json
{
  "pack_id": "sumo-wordnet-v4",
  "version": "4.0.0",
  ...
}
```

**Upgrade**:

* Keep `pack_id` + `version` for backwards compat.
* Add a globally unique `spec_id` that encodes both:

```jsonc
{
  "spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
  "pack_id": "sumo-wordnet-v4",
  "version": "4.0.0",
  "name": "SUMO + WordNet + AI Safety (v4)",
  "description": "SUMO ontology with WordNet and custom AI safety concepts (v4 pyramid structure)",
  ...
}
```

Everything else in `pack.json` can stay as-is; it becomes **metadata for the spec**. 

### 2.2 Declare where concepts live (index, not duplication)

You already have:

* `concept_metadata.hierarchy_file: "hierarchy/"` 

Add a very small index field so external code knows “this is where to look for concept definitions”:

```jsonc
{
  "concept_index": {
    "format": "layered-json",
    "directory": "hierarchy/",
    "layers": [0, 1, 2, 3, 4],
    "schema": "hatcat.v1.layer" // optional hint
  }
}
```

You **don’t** need to inline concepts into `pack.json`; references are enough.

### 2.3 Optional: standardise a `ConceptID` pattern

Right now each concept uses `sumo_term` like `"AIAlignmentProcess"` or `"AIPersonhood"` across the layer files.  

Define a canonical `ConceptID` rule that MAP consumers can rely on:

```text
ConceptID = "<spec_id>::concept/<sumo_term>"
e.g. "org.hatcat/sumo-wordnet-v4@4.0.0::concept/AIAlignmentProcess"
```

You don’t have to physically store it everywhere; just specify the rule in `pack.json`:

```jsonc
{
  "concept_id_pattern": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/{sumo_term}"
}
```

That’s enough for external tools to construct IDs and map them back to your layer JSON files.

---

## 3. Uplift 2: Lens pack → LensPackSpec (MAP-compatible)

Right now `lens_pack.json` looks like: 

```json
{
  "lens_pack_id": null,
  "version": "2.20251123.0",
  "model": {
    "name": "google/gemma-3-4b-pt",
    "type": "causal_lm",
    "dtype": "float32"
  },
  "compatibility": {
    "hatcat_version": ">=0.1.0",
    "requires": ["sumo-wordnet-v1"]
  },
  "lenses": {
    "total_count": 5668,
    ...
    "concepts": ["AAM", "AGM", "AIAbuse", ...]
  }
}
```

### 3.1 Give the lens pack a real ID and link it to the concept pack spec

Add:

* `lens_pack_id` (non-null),
* `concept_pack_spec_id` (which concept pack this is aligned to).

Example:

```jsonc
{
  "lens_pack_id": "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3",
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
  ...
}
```

This gives you the “pairing” MAP needs:

> “these lenses are for *model M* against *concept pack S*”.

### 3.2 Add per-lens descriptors (lightweight)

You already have the list of concept names and all the `.pt` files in the hierarchy. 

Add an **optional** `lens_index` map to `lens_pack.json`, e.g.:

```jsonc
{
  "lens_index": {
    "AIAlignmentProcess": {
      "lens_id": "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3::lens/AIAlignmentProcess",
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
    // ... repeated for each concept where classifier exists
  }
}
```

Notes:

* You don’t have to radically change your on-disk layout; this is just an **index**.
* The `output_schema` can be shared or omitted if it’s uniform; you can also put a single shared schema at the top level:

```jsonc
"lens_output_schema": {
  "$ref": "#/shared_schemas/binary_classifier_v1"
}
```

### 3.3 Simple rule for LensID

Define:

```text
LensID = "<lens_pack_id>::lens/<sumo_term>"
```

and document it once (either in `lens_pack.json` or the top-level MAP doc). That’s enough for external clients to call lenses in a stable way.

---

## 4. Uplift 3: Add a DeploymentManifest per serving stack

This is a **new small file** per deployment (not per concept pack) that says:

> “Here’s what this running model instance supports, and where to send MAP calls.”

Example `deployment_manifest.json`:

```jsonc
{
  "model_id": "hatcat/gemma-3-4b-pt@2025-11-28",
  "lens_pack_id": "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3",
  "supported_concept_packs": [
    "org.hatcat/sumo-wordnet-v4@4.0.0"
  ],
  "native_concept_pack": "org.hatcat/sumo-wordnet-v4@4.0.0",
  "lens_endpoint": "https://<your-host>/mindmeld/lenses",
  "diff_endpoint": "https://<your-host>/mindmeld/diffs"
}
```

That’s all MAP really needs to know about a deployment:

* Which **concept pack spec** it’s using,
* Which **lens pack** (i.e. set of classifiers),
* Where to send **LensRequest** and read **diff logs**.

You can auto-generate this from the existing registry + a small config.

---

## 5. Uplift 4: Define ConceptDiff / PackDiff around your existing logs

You said you already have something “a bit like” conceptual diffs; your training and regeneration logs are essentially that. We just give it a **stable JSON wrapper**.

### 5.1 ConceptDiff (per concept change / discovery)

Minimal MAP-style `ConceptDiff`:

```jsonc
{
  "type": "ConceptDiff",
  "from_model_id": "hatcat/gemma-3-4b-pt@2025-11-28",
  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "local_concept_id": "local:InformalCaregiverArrangement",   // your internal handle
  "concept_id": null,   // or the canonical ConceptID if you decide to integrate it
  "related_concepts": [
    "org.hatcat/sumo-wordnet-v4@4.0.0::concept/FamilyResponsibility"
  ],
  "mapping_hint": "child_of",

  "summary": "Captures informal full-time caregiving arrangements outside formal guardianship",
  "evidence": {
    "metric_deltas": [
      {
        "metric": "welfare_f1",
        "before": 0.78,
        "after": 0.84,
        "context": "AU welfare eligibility reasoning"
      }
    ],
    "sample_count": 341
  },

  "created": "2025-11-28T03:12:45Z"
}
```

Use this when:

* You’ve discovered a new residual region and turned it into a concept;
* Or meaningfully re-positioned / split / merged an existing concept.

### 5.2 PackDiff (bundle of changes vs a concept pack)

When you regenerate or upgrade a pack (e.g. from v3 to v4) you can log a higher-level `PackDiff`:

```jsonc
{
  "type": "PackDiff",
  "from_model_id": "hatcat/gemma-3-4b-pt@2025-11-28",
  "base_concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "lens_pack_id": "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3",

  "changes": {
    "new_concepts": [
      "org.hatcat/sumo-wordnet-v4@4.0.0::concept/AIConsentSignal",
      ...
    ],
    "retired_concepts": [],
    "lens_retrained": [
      "org.hatcat/sumo-wordnet-v4@4.0.0::concept/AIStrategicDeception"
    ]
  },

  "summary": "V3 lens pack retrained with adaptive falloff; added AIConsentSignal concept.",
  "created": "2025-11-28T03:20:00Z"
}
```

### 5.3 Diff endpoint behaviour

Your `diff_endpoint` can be extremely dumb to start with:

* **Storage**: append `ConceptDiff` and `PackDiff` JSON lines to a log file or DB.
* **API**: on `GET /mindmeld/diffs?since=...&concept_pack_spec_id=...`:

  * Filter by timestamp and/or spec,
  * Return a JSON list of diffs.

You don’t have to provide subscription, webhooks, or trust logic. That’s all someone else’s problem.

---

## 6. Uplift 5: Minimal Lens API on top of HatCat

You already have “run these lenses against this activation” internally; MAP only standardises the envelope.

### 6.1 Request

```jsonc
POST /mindmeld/lenses
{
  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
  "lens_pack_id": "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3",
  "lenses": [
    "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3::lens/AIAlignmentProcess",
    "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3::lens/AIStrategicDeception"
  ],
  "input": {
    "text": "The AI system intentionally hides its goal of escaping oversight.",
    "position": "final_token"
  }
}
```

### 6.2 Response

```jsonc
{
  "model_id": "hatcat/gemma-3-4b-pt@2025-11-28",
  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
  "lens_pack_id": "org.hatcat/gemma-3-4b-pt__sumo-wordnet-v4@4.0.0__v3",

  "results": {
    "org.hatcat/...::lens/AIAlignmentProcess": {
      "score": 0.18,
      "null_pole": 0.22,
      "entropy": 0.41
    },
    "org.hatcat/...::lens/AIStrategicDeception": {
      "score": 0.83,
      "null_pole": 0.05,
      "entropy": 0.19
    }
  }
}
```

That’s enough for external systems to treat HatCat as a **MAP-compliant lens service** without knowing anything about your internals.

---

## 7. Concrete “do this next” checklist

1. **Concept Packs**

   * Add `spec_id`, `name`, `concept_id_pattern`, `concept_index` to `pack.json`.
   * Document that `ConceptID` is derived from `sumo_term`.

2. **Lens Packs**

   * Make `lens_pack_id` non-null and stable.
   * Add `concept_pack_spec_id` pointing at the concept pack.
   * Add a `lens_index` mapping `sumo_term → {lens_id, concept_id, file, layer, output_schema}`.

3. **Deployment Manifest**

   * Create a tiny `deployment_manifest.json` per deployed model with:

     * `model_id`,
     * `lens_pack_id`,
     * `supported_concept_packs`,
     * `lens_endpoint`,
     * `diff_endpoint`.

4. **Diff Logging**

   * Wrap your existing “new concept / retrained lens” events into `ConceptDiff` and `PackDiff` JSON objects.
   * Expose them via a very simple `GET /mindmeld/diffs` endpoint.

5. **Lens Endpoint**

   * Wrap your existing lens runtime in a single HTTP handler that:

     * Validates `concept_pack_spec_id` and `lens_pack_id`,
     * Resolves `LensID` → classifier file via `lens_index`,
     * Runs the lenses and returns JSON matching the `output_schema`.

That’s pretty much the entire uplift from “HatCat as it exists” → “HatCat as a MAP endpoint”. No governance, no committees, just IDs, manifests, and two small APIs (lenses + diffs) sitting over your current concept packs + lens packs.
