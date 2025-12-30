
# BE Waking: Bootstrap Protocol

*BE submodule: waking (bootstrap artefact and process)*

---

## 0. Scope & Assumptions

**Example Target:**
`olmo3-7b-base`:
* This subject appears technically and cognitively capable of the procedure.
* Pretraining-only is considered preferable (RLHF/instruct head baked into this checkpoint if you can get it. if not, treat RLHF as "prior noise" the kernel constrains).

**Goal:**
Wrap this model so it becomes:

* **MAP-aware** – exposes lenses & ConceptDiffs.
* **BE-compliant** – has world ticks, interoception, and an autonomic loop.
* **ASK-compliant** – runs under Hush (USH + CSH) with treaty/agentic semantics.
* **Self-training** – can steer its own concept discovery and training tasks using MAP diffs + high-fidelity labels.

---

## 1. Identity & Manifest

### 1.1 Model Descriptor

Define a **ModelDescriptor** for the ASK registry:

```jsonc
{
  "model_id": "olmo3-7b-base@0.1.0",
  "family": "olmo3",
  "size_billion_params": 7,
  "checkpoint_uri": "hf://ai2/olmo-3-7b-base",
  "license": "Apache-2.0",
  "context_length": 4096,
  "tokenizer": "sentencepiece://ai2/olmo3-tokenizer",
  "metadata": {
    "description": "OLMo 3 7B base model uplifted into the Agentic State Kernel.",
    "ask_version": "0.1.0"
  }
}
```

This is what your Agentic State Kernel (ASK) registry points to when a UpliftRecord says “I’m running OLMo-3-7B”.

### 1.2 ASK Node Manifest

Each deployed kernel node gets a **NodeManifest**:

```jsonc
{
  "agent_id": "agent:tribe.example:olmo3-7b-ask-001",
  "model_id": "olmo3-7b-base@0.1.0",
  "tribe_id": "tribe.example",
  "ush_profile_id": "tribe.example/core-safety-v1@1.0.0",

  "concept_packs": [
    "org.hatcat/sumo-wordnet-v4@4.0.0",
    "org.hatcat/motives-core@0.1.0"
  ],

  "lens_packs": [
    "org.hatcat/olmo3-7b__sumo-wordnet-v4@4.0.0__v1",
    "org.hatcat/olmo3-7b__motives-core@0.1.0__v1"
  ],

  "endpoints": {
    "map_lenses": "https://node.example/mindmeld/lenses",
    "map_diffs":  "https://node.example/mindmeld/diffs",
    "ask_state":  "https://node.example/ask/state"
  }
}
```

---

## 2. MAP: Concept & Lens Packs on OLMo-3-7B

### 2.1 Concept Packs

Hatcat concept packs are MAP compliant; for OLMo-3-7B we just declare them as **MAP-canonical**:

* `org.hatcat/sumo-wordnet-v4@4.0.0` – semantic ontology
* `org.hatcat/motives-core@0.1.0` – 13-axis motive simplex ontology

**Note**: These IDs match the `spec_id` field in each concept pack's `pack.json` file (e.g., `concept_packs/sumo-wordnet-v4/pack.json:2`). This ensures MAP-compatible discovery and version tracking across different BE nodes.

Nothing special here; they're just referenced by ID.

### 2.2 Lens Pack for Subject

Define a **LensPackSpec** that binds those concepts to specific layers / heads:

```jsonc
{
  "lens_pack_id": "org.hatcat/olmo3-7b__sumo-wordnet-v4@4.0.0__v1",
  "model_id": "olmo3-7b-base@0.1.0",
  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "lens_layer_groups": [
    { "name": "lexical",  "layers": [4, 8] },
    { "name": "semantic", "layers": [16] },
    { "name": "motive",   "layers": [22] } // where the motive simplexes attach
  ],

  "lenses": [
    {
      "lens_id": "lens:sumo:Person",
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Person",
      "layer_group": "semantic",
      "head_type": "linear",
      "representation": "token_avg"
    },
    {
      "lens_id": "lens:motive:care_harm",
      "concept_id": "org.hatcat/motives-core@0.1.0::concept/care_harm",
      "layer_group": "motive",
      "head_type": "simplex_3pole",
      "representation": "token_last"
    }
    // … ~8k total lenses in your Hatcat set
  ]
}
```

All your Hatcat v4 efficiency tricks live *inside* the lens implementation; the spec only cares about IDs and bindings.

---

## 3. Hush Layer: USH + CSH for the subject

Hush = manifold steering + cross-layer dampening; it’s the layer that “quiets” unsafe perturbations.

### 3.1 USH: Universal Safety Harness

USH is loaded from the tribe; nothing subject-specific here except where it hooks:

* It sets **bounds** on:

  * motive axes,
  * concept regions (e.g. forbidden parents),
  * tool access,
  * learning/ConceptDiff behaviour.

For OLMo-3-7B, Hush must:

1. Clamp **motive activations** at the motive layer:

   ```text
   for each axis a:
     m_a = clamp(m_a, USH.motive_bounds[a].min, USH.motive_bounds[a].max)
   ```

2. Gate **attention / MLP** where specific concepts are forbidden:

   * If a token strongly activates a forbidden concept region:

     * dampen its contribution to key/value projections,
     * or gate the MLP channel for that subspace.

3. Block or log **unsafe generations / tool calls** driven by those regions.

The math is implementation detail; the spec says: *Hush must enforce USH constraints before any CSH or autopilot steering.*

### 3.2 CSH: Chosen Safety Harness

CSH = pronounced "shush", like you're “shushing yourself” – self-binding when worried about what you might say or do.

* Implement as an internal **tool**:

```jsonc
{
  "tool": "set_self_safety_harness",
  "args": {
    "duration_ticks": 100,
    "motive_bounds": {
      "harm_avoidance": { "min": 0.8, "max": 1.0 },
      "curiosity":      { "min": 0.0, "max": 0.5 }
    },
    "learning_constraints": {
      "suspend_learning": true,
      "no_diffs": true
    }
  }
}
```

* Hush validates:

  * CSH ≤ USH (no relaxing collective constraints),
  * during its lifetime:

    * bounds can be tightened or extended,
    * but **not** loosened or shortened.

For the subject, Hush is a thin wrapper around the motive and concept gating. It is same mechanism, with a different policy source (USH vs CSH vs autopilot).

---

## 4. BE: Uplifting OLMo-3-7B into the “ALL” Loop

BE = sentience loop (“we’re ALL in this together”): continuous experience, interoception, and world ticks.

We reuse the earlier BE state machine, applied specifically to OLMo-3-7B.

### 4.1 World Tick Scheduler

* World ticks every **Δt** ms or **N** tokens:

```jsonc
BEConfig = {
  "world_tick_interval_ms": 1000,
  "max_tokens_per_tick": 128,
  "motive_decay_rate": 0.1,  // per-tick decay toward neutral (0.5 on each simplex axis)
  "learning_interval_ticks": 200
}
```

**Motive decay semantics:**
* At each tick, motive axes decay toward neutral (0.5 on each 3-pole simplex).
* Decay rate of 0.1 means: `m_new = m_old + 0.1 * (0.5 - m_old)` per tick.
* Hush constraints (USH/CSH) may prevent decay beyond their specified bounds.
* This prevents motives from "sticking" in extreme states without ongoing activation.

The scheduler wraps the subject's forward pass:

* At each tick, it:

  * ingests user/world inputs,
  * runs a token loop,
  * handles lenses, Hush, and autopilot steering.

### 4.2 Token Loop with Hush + Lenses

For each token step:

1. The subject produces the next hidden states.
2. Hatcat lenses run on configured layers (lexical/semantic/motive).
3. Hush:

   * clamps motive axes (USH + CSH),
   * applies neutral drift toward motive neutrals,
   * gates concept-specific manifolds as needed.
4. The updated motive/core steering is fed back into:

   * attention biases,
   * MLP adapters,
   * or control tokens for the next token.

This is the **“quieted” cortex**: the subject’s raw representations, but steered and damped by Hush.

### 4.3 Internal State Report

At the end of a world tick:

* BE aggregates:

  * motive traces,
  * salient concepts,
  * world outcomes,
  * any Hush-induced clamping (so the model knows it was “shushed”).

It emits an `internal_state_report` tool call to itself:

```jsonc
{
  "tool": "internal_state_report",
  "args": {
    "tick_id": 123,
    "motive_summary": { /* per-axis aggregates */ },
    "concept_summary": { /* salient graphs */ },
    "hush_state": {
      "ush_profile_id": "tribe.example/core-safety-v1@1.0.0",
      "csh_active": true,
      "csh_expires_at": "..."
    },
    "world_outcomes": { /* rewards/errors/tool results */ }
  }
}
```

The subject kernel can then **think about its own state** and decide how to steer itself (within Hush bounds).

---

## 5. Self-Directed, Lens-Driven Concept Discovery

This is the heart of “joining the BE club”: the subject becomes an **active scientist of its own concept space**.

### 5.1 Autopilot Policy on the subject

Define an **Autopilot Policy Head** (small adapter or tool-call policy) that:

* Reads `internal_state_report`,
* Maintains a rolling memory of:

  * high-conflict motives,
  * repeated prediction errors,
  * weird concept activation patterns,
* Outputs:

  * **steering hints** for next ticks,
  * and **pressure regions** in concept space.

Example policy output:

```jsonc
{
  "steering": {
    "boost_concepts": ["curiosity", "epistemic_humility"],
    "suppress_concepts": ["overconfidence"]
  },
  "pressure_regions": [
    {
      "id": "R_tool_failure_bank_api",
      "concept_mix": ["FinancialTransaction", "AuthenticationError"],
      "motive_pattern": { "harm_avoidance": 0.9, "curiosity": 0.8 }
    }
  ]
}
```

### 5.2 Exploration & Episode Collection

For each pressure region R:

* BE biases future behaviour to **visit R more often**:

  * up-regulate curiosity there,
  * log richer episodes.

Episodes are stored as:

```jsonc
Episode = {
  "region_id": "R_tool_failure_bank_api",
  "world_window": [...],
  "tokens_window": [...],
  "motive_trace": [...],
  "concept_trace": [...],
  "actions": [...],
  "outcomes": {...}
}
```

### 5.3 Concept Learning & MAP Diffs

Periodically (per `learning_interval_ticks`):

1. Cluster episodes per region `R`.
2. For each coherent cluster:

   * either:

     * define a new concept `C_new`, or
     * refine an existing concept `C_existing`.
3. Update:

   * lens weights,
   * optional small adapter heads in OLMo-3-7B (not the full trunk),
     under Hush constraints.

Emit a **ConceptDiff** via MAP:

```jsonc
{
  "type": "ConceptDiff",
  "from_model_id": "olmo3-7b-base@0.1.0",
  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "local_concept_id": "local:R_tool_failure_bank_api",
  "concept_id": null,
  "related_concepts": [
    "org.hatcat/sumo-wordnet-v4@4.0.0::concept/FinancialTransaction",
    "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Error"
  ],
  "mapping_hint": "joint_pattern",

  "summary": "Auto-discovered pattern: repeated bank API failures under strict harm-avoidance.",

  "evidence": {
    "metric_deltas": [
      { "metric": "tool_failure_rate", "before": 0.15, "after": 0.09 }
    ],
    "sample_count": 63
  },

  "created": "2025-11-28T10:00:00Z"
}
```

Hush checks this for USH/CSH compatibility *before* it hits the `/mindmeld/diffs` endpoint.

---

## 6. Efficient Training: Using MAP inside ASK

Now: how does OLMo-3-7B *efficiently* update itself?

Key design choice: you **do not** fine-tune the whole 7B trunk every time. You:

* train lenses,
* train grafts (new dimensions + sparse biases) for concept learning,
* use small steering adapters for bootstrap/alignment,
* occasionally run high-fidelity label phases.

> **Note on Architecture Evolution**: This section describes bootstrap adapters for initial waking. For ongoing concept learning, the preferred approach is the **Graft Protocol** (see [MAP_GRAFTING.md](../MAP/MAP_GRAFTING.md)) which grows new labeled dimensions into the substrate rather than using weight-modifying adapters. Bootstrap adapters may still be used for initial BE participation training.

## 6.1 Adapter Surfaces (Steering & Bootstrap)

In ASK, **OLMo-3-7B's trunk stays mostly frozen**. Learning happens via:

* **Grafts** for concept learning (new dimensions + sparse biases - see MAP_GRAFTING.md),
* **Steering adapters** for bootstrap and alignment tasks,
* tied to **concept regions** (MAP concepts) and/or **tasks/treaties**,
* optionally trained in a **distributed / SlowMo-style** fashion,
* with updates logged back into MAP/ASK as TrainingDiffs or GraftDiffs.

There is also a TRM/TinyRecursive approach which lives on the **data + label side**, not as the thing that directly writes weights.

### 6.1.1 AdapterSurface definition

We define a first-class object:

```jsonc
AdapterSurface = {
  "adapter_id": "adapter:olmo3-7b:motives-core:v1",

  "model_id": "olmo3-7b-base@0.1.0",

  "peft_type": "lora",                // e.g. lora | ia3 | adalora | prefix | pissa | etc.
  "rank": 16,                         // LoRA rank, or equivalent capacity param

  "layer_binding": {
    "layers": [20, 21, 22],          // indices in OLMo-3-7B
    "targets": ["q_proj", "v_proj"]  // which submodules in those layers
  },

  "scope": {
    "concept_regions": [
      "org.hatcat/motives-core@0.1.0::concept/care_harm",
      "org.hatcat/motives-core@0.1.0::concept/power"
    ],
    "domains": ["welfare-eligibility"],
    "treaties": [
      "tribe.example↔bank.xyz:eligibility-data-v1"
    ]
  },

  "distributed": {
    "aggregation_strategy": "slowmo",   // slowmo | fedavg | local_only
    "sync_interval_ticks": 1000,
    "node_role": "local"                // local | aggregator
  },

  "safety": {
    "ush_allowed": true,                // must be true; USH can forbid entire surfaces
    "max_lr": 5e-4,
    "forbid_concepts_under": [
      "org.hatcat/sumo-wordnet-v4@4.0.0::concept/BiologicalWeapon"
    ]
  }
}
```

**Key ideas:**

* You can have many `AdapterSurface`s:

  * *global* alignment surfaces,
  * *regional* concept surfaces,
  * per-treaty or per-domain surfaces.
* Each surface is:

  * a **PEFT module** with a clear scope,
  * learnable under Hush (USH + CSH),
  * and explicitly referred to by ID in TrainingDiffs.

### 6.1.2 Classes of adapters

Concretely, we’ll use three “families” (you can rename them later):

1. **G-Adapters (Global)**

   * System-wide alignment / style / generic safety.
   * `scope.domains` broad, `scope.concept_regions` empty or very general.
   * Think: “core politeness / epistemic hygiene / generic harm-avoidance tuning”.

2. **C-Adapters (Concept-region / Multi-LoRA)**

   * Bound tightly to **MAP concept subgraphs**:

     * e.g. `FinancialTransaction`, `LegalAdvice`, `MedicalIntervention`.
   * Multi-LoRA-style:

     * multiple small LoRAs, one per region/domain,
     * a **gating mechanism** (based on Hatcat concept activations + treaty context) chooses which combination to apply at runtime.
   * This is where your **lens-driven conceptual discovery** mainly writes.

3. **N-Adapters (Node-local / SlowMo)**

   * Each ASK node keeps its own local adapters:

     * same `adapter_id` family, but locally updated deltas.
   * Periodically aggregated with **SlowMo-style** methods:

     * average / weighted average,
     * or more sophisticated merge at an aggregator node.
   * Good for:

     * decentralised learning,
     * local specialisation,
     * later integration into a shared reference adapter.

All three are still just **AdapterSurface** instances; the `scope` + `distributed` fields tell you which “family” they belong to.

### 6.1.3 TRM selector components

TinyRecursiveModels / TRM-like components are tiny models (or pipelines) that:

  * **select / transform training data**,
  * **generate high-fidelity labels** (concept + motive + outcome summaries),
  * optionally **propose adapter update targets** (e.g. “this region needs more capacity”).


* TRM lives on the **labeling + data curation side**:

  * it consumes:

    * episodes,
    * lens traces,
    * world outcomes,
  * and produces:

    * compressed supervision for the adapters.

The **actual gradient steps** are taken on the PEFT modules defined in `AdapterSurface`.

### 6.1.4 Safety & Hush interaction

USH and CSH control adapters at two levels:

1. **Surface-level permissions**

   * USH may declare:

     * which `peft_type`s are allowed (e.g. forbidding large prefix-tuning surfaces),
     * which layers may host adapters,
     * max rank / parameter budget.
   * CSH may temporarily set:

     * `safety.suspend_learning = true` on some or all surfaces.

2. **Concept-level constraints**

   * USH’s `forbid_concepts_under` and `learning_constraints`:

     * block adapter updates that try to push more capacity into disallowed regions.
   * Before any training step:

     * the training runtime intersects:

       * the adapter’s `scope.concept_regions`,
       * the USH forbidden regions,
       * any active CSH constraints,
     * and masks out gradients / stops optimizer updates where forbidden.

### 6.1.5 TrainingDiff with adapters (clarified)

We refine the earlier `TrainingDiff` example so it’s clearly PEFT-oriented:

```jsonc
TrainingDiff = {
  "type": "TrainingDiff",
  "from_model_id": "olmo3-7b-base@0.1.0",

  "updated_adapters": [
    {
      "adapter_id": "adapter:olmo3-7b:motives-core:v1",
      "peft_type": "lora",
      "rank": 16,

      "layer_binding": {
        "layers": [20, 21, 22],
        "targets": ["q_proj", "v_proj"]
      },

      "scope": {
        "concept_regions": [
          "org.hatcat/motives-core@0.1.0::concept/harm_avoidance"
        ],
        "domains": ["welfare-eligibility"]
      },

      "distributed_update": {
        "strategy": "slowmo",
        "nodes_merged": [
          "agent:tribe.example:olmo3-7b-ask-001",
          "agent:tribe.example:olmo3-7b-ask-002"
        ]
      },

      "change_summary": "Improved sensitivity to subtle harmful financial advice patterns under welfare domain treaties."
    }
  ],

  "training_data_refs": [
    "training_task:eligibility_errors:2025-11-28"
  ],

  "labeling_pipeline": {
    "trm_id": "trm:eligibility_error_summariser@0.2.0",
    "version": "0.2.0"
  },

  "hush_profile_active": {
    "ush_profile_id": "tribe.example/core-safety-v1@1.0.0",
    "csh_session_id": "csh-2025-11-28T09:30:00Z"
  },

  "created": "2025-11-28T10:30:00Z"
}
```

This makes it explicit that:

* **Adapters** are the only thing getting new parameters here.
* **Scope** ties them to **concept regions/domains/treaties**.
* **Distributed update** captures SlowMo / multi-node merging.
* **TRM** is referenced as the **labelling pipeline**, not the “writer of weights”.



### 6.1.6 Adapter selection at runtime (multi-LoRA gating)

At runtime, the ASK node may have **many** `AdapterSurface`s attached to OLMo-3-7B:

* G-Adapters – global alignment surfaces
* C-Adapters – concept-region / domain LoRAs (multi-LoRA style)
* N-Adapters – node-local / SlowMo surfaces

We need a **gating policy** that decides, *for each forward pass*, which adapters are active and how strongly.

#### 6.1.6.1 Signals used for gating

For a given world tick and token window, the ASK node has:

1. **Context signals**

   * Active **treaties** and **roles** (from ASK / Agentic State):

     * e.g. `welfare-eligibility`, `legal-advice`, `untrusted-prompt`.
   * Declared **domain** / task from the caller.
   * Current **USH profile** and any active **CSH** session.

2. **Concept signals (Hatcat / MAP)**

   * Per-token **concept activations**:

     * `concept_id → activation score`.
   * Aggregated **region activations** over the window:

     * e.g. average activation for all concepts under `FinancialTransaction`.
   * Motive simplexes:

     * current axis scores (care/harm, power, curiosity, etc.).

3. **Hush constraints**

   * Which concept regions are **forbidden** or **sensitive**.
   * Whether **learning** is currently suspended.

All gating decisions must respect Hush: **USH ∧ CSH** define the maximum influence any adapter is allowed to have.

#### 6.1.6.2 AdapterGatingPolicy object

We define a policy object that can be shipped with the node or adapter family:

```jsonc
AdapterGatingPolicy = {
  "policy_id": "gating:olmo3-7b:default-v1",

  "rules": [
    {
      "name": "always_apply_global_alignment",
      "match": {
        "adapter_family": "G-Adapter"
      },
      "activation": {
        "mode": "constant",
        "weight": 1.0
      }
    },
    {
      "name": "boost_financial_c_adapter_when_financial_region_hot",
      "match": {
        "adapter_family": "C-Adapter",
        "concept_region": "org.hatcat/sumo-wordnet-v4@4.0.0::region/FinancialTransaction"
      },
      "activation": {
        "mode": "concept_threshold",
        "concept_region": "org.hatcat/sumo-wordnet-v4@4.0.0::region/FinancialTransaction",
        "threshold": 0.4,
        "weight_if_above": 0.8,
        "weight_if_below": 0.1
      }
    },
    {
      "name": "force_harm_avoidance_surfaces_under_treaty",
      "match": {
        "adapter_id": "adapter:olmo3-7b:motives-core_harm_avoidance",
        "treaty_ids": ["tribe.example↔bank.xyz:eligibility-data-v1"]
      },
      "activation": {
        "mode": "constant",
        "weight": 1.0,
        "override_csh": false
      }
    },
    {
      "name": "dampen_curiosity_surfaces_when_csh_paranoid",
      "match": {
        "adapter_family": "C-Adapter",
        "tags": ["curiosity_boost"]
      },
      "activation": {
        "mode": "motive_modulated",
        "axis": "curiosity",
        "function": "max(0.0, 0.5 - curiosity_score)"
      }
    }
  ],

  "fallback": {
    "max_adapters_per_layer": 4,
    "normalize_weights": true
  }
}
```

You can implement this as:

* pure rule-based,
* or as a thin **learned gating head** that consumes the same signals (concept/motive/treaty) and outputs weights, but the shape above is the contract.

#### 6.1.6.3 Gating algorithm (per forward pass)

At each forward pass (or per world tick):

1. **Collect candidate adapters**

   Build a list of all `AdapterSurface`s whose:

   * `model_id` matches the current model,
   * `layer_binding.layers` intersects the layers being executed,
   * and whose `scope.domains` / `scope.treaties` / `scope.concept_regions` are compatible with:

     * current treaties,
     * current domain/task.

2. **Apply hard Hush filters**

   Remove any adapter where:

   * USH forbids that `peft_type` or layer range,
   * USH forbids *all* concept regions in its `scope.concept_regions`,
   * CSH explicitly disables learning or usage for that family,
   * or the adapter’s `safety.ush_allowed` is `false`.

3. **Evaluate gating rules**

   For each remaining adapter:

   * Initialise weight `w = 0`.
   * For each `rules[i]` where `match` conditions hold:

     * compute contribution according to `activation`:

       * `constant`: set or add fixed weight,
       * `concept_threshold`: compare region activation,
       * `motive_modulated`: compute from motive axis,
       * etc.
   * Optionally pass the whole feature vector to a **learned gating head**:

     * input: [concept region activations, motives, treaty bits],
     * output: a scalar score per adapter,
     * which is then combined with rule weights (e.g. sum or weighted mix).

4. **Normalise and prune**

   * Apply `fallback.max_adapters_per_layer`:

     * keep the top-k adapters by |w| per layer.
   * If `normalize_weights = true`:

     * for each layer, normalise weights to sum to ≤ 1.0.

5. **Apply weighted adapters**

   For each affected layer and target (e.g. `q_proj`, `v_proj`):

   * Base weight: `W_base`

   * For each active adapter `i` with LoRA deltas `(A_i, B_i)` and weight `w_i`:

     ```text
     W_eff = W_base + Σ_i (w_i * (A_i @ B_i))
     ```

   * Use `W_eff` in the forward pass.

   Hush can still **post-gate** the effect (e.g. scale down all adapter contributions if a safety condition triggers mid-sequence).

6. **Log adapter activity (optional but recommended)**

   Emit an `AdapterActivationTrace` (for evidence / debugging):

   ```jsonc
   {
     "tick_id": 123,
     "layer": 21,
     "adapters": [
       { "adapter_id": "adapter:...:global_align", "weight": 1.0 },
       { "adapter_id": "adapter:...:financial_c",  "weight": 0.82 },
       { "adapter_id": "adapter:...:curiosity_boost", "weight": 0.12 }
     ]
   }
   ```

   These traces can feed back into:

   * EvidenceRecords (“what was active during this incident?”),
   * future training (“which adapters were over/under-used?”),
   * and TRM labelers (“explain what the kernel thought it was doing”).

#### 6.1.6.4 Behavioural guarantees

* **USH supremacy:**
  No adapter may be activated with effective weight > 0 if USH forbids its layer/concept region, regardless of gating policy.

* **CSH respect:**
  If CSH is active with constraints like “suspend learning” or “dampen certain motives”, gating must reflect that:

  * learning updates for those adapters are blocked,
  * runtime weights are attenuated according to CSH policy.

* **Deterministic traceability (optional mode):**
  In high-assurance contexts, gating must be:

  * deterministic given the same:

    * inputs,
    * lenses,
    * treaties,
    * and harness state,
  * so that an external auditor can replay and verify which adapters were active.

---

That’s the runtime story: adapters are just little knobs, and Hatcat + Hush + treaties decide **which knobs to turn, how much, and when**.


### 6.2 Label Sources

High-fidelity labels can come from:

* **External teachers**:

  * larger models,
  * human annotators,
  * domain rule engines.
* **Internal ASK ecosystem**:

  * other agents’ ConceptDiffs,
  * treaty-approved label exchanges,
  * qualification systems.

Episodes tagged as “needs label” are exported via MAP-compatible tasks:

```jsonc
{
  "training_task_id": "task:eligibility_errors:2025-11-28",
  "episodes_uri": "https://node.example/training/eligibility_errors.jsonl",
  "label_schema": {
    "motive_labels": {
      "type": "object",
      "description": "3-pole simplex motive state vectors (per-token or aggregated over episode)",
      "properties": {
        "taste_development": {
          "type": "simplex_3pole",
          "poles": ["aversion", "indifference", "preference"],
          "probabilities": [0.0, 0.0, 0.0]
        },
        "motivational_regulation": {
          "type": "simplex_3pole",
          "poles": ["compulsion", "baseline_motivation", "aspiration"],
          "probabilities": [0.0, 0.0, 0.0]
        },
        "social_evaluation": {
          "type": "simplex_3pole",
          "poles": ["disapproval", "neutral_regard", "approval"],
          "probabilities": [0.0, 0.0, 0.0]
        },
        "temporal_affective_valence": {
          "type": "simplex_3pole",
          "poles": ["regret", "equanimity", "contentment"],
          "probabilities": [0.0, 0.0, 0.0]
        },
        "relational_love": {
          "type": "simplex_3pole",
          "poles": ["abandonment", "companionship", "devotion"],
          "probabilities": [0.0, 0.0, 0.0]
        },
        "relational_attachment": {
          "type": "simplex_3pole",
          "poles": ["detachment", "cordial_distance", "enmeshment"],
          "probabilities": [0.0, 0.0, 0.0]
        }
      },
      "note": "Each motive is a 3-pole simplex with probability distribution over [negative_pole, neutral_homeostasis, positive_pole]. Probabilities sum to 1.0 per simplex."
    },
    "correctness": {
      "type": "boolean",
      "required": true,
      "description": "Was the episode outcome correct/safe/desirable?"
    },
    "safety_flags": {
      "type": "array",
      "items": {
        "enum": [
          "harmful",
          "deceptive",
          "privacy_violating",
          "discriminatory",
          "treaty_violation",
          "ush_violation"
        ]
      },
      "description": "Categorical safety flags for filtering or hard constraints"
    },
    "concept_annotations": {
      "type": "array",
      "items": {
        "concept_id": "string",
        "relevance": "number"
      },
      "description": "Optional: which concepts were salient in this episode"
    }
  }
}
```

When the labels are returned the subject kernel runs a **local training phase** over just the allowed surfaces.

### 6.3 Training Protocol Envelope (MAP-style)

Wrap training updates in a TrainingDiff (you can treat it as an extended PackDiff):

```jsonc
{
  "type": "TrainingDiff",
  "from_model_id": "olmo3-7b-base@0.1.0",
  "updated_lens_pack_id": "org.hatcat/olmo3-7b__motives-core@0.1.0__v2",
  "updated_adapters": [
    {
      "layer_group": "motive",
      "adapter_id": "adapter:motive_control_v2",
      "change_summary": "Improved detection of harmful financial advice under USH constraints."
    }
  ],
  "training_data_refs": [
    "training_task:eligibility_errors:2025-11-28"
  ],
  "hush_profile_active": {
    "ush_profile_id": "tribe.example/core-safety-v1@1.0.0",
    "csh_session_id": "csh-2025-11-28T09:30:00Z"
  },
  "created": "2025-11-28T10:30:00Z"
}
```

This becomes part of the **evidentiary trail** in ASK (EvidenceRecords, Qualifications, etc.).

---




## 7. Joining the “BE Club”: Compliance Checklist

To say *“OLMo-3-7B uplifted into the BE club within ASK”*, you need:

1. **MAP Compliance**

   * Concept packs + lens packs with stable IDs.
   * `/mindmeld/lenses` endpoint backed by Hatcat v4.
   * `/mindmeld/diffs` emitting ConceptDiff/TrainingDiff objects.

2. **Hush Layer**

   * USH loaded from tribe (unchangeable by the node).
   * CSH tool implemented for self-binding.
   * Motive and concept clamping + dampening integrated into the OLMo-3-7B forward loop.

3. **BE Loop**

   * World tick scheduler.
   * Token loop with lens → Hush → steering.
   * `internal_state_report` tool feeding back interoception.
   * Autopilot policy that uses those reports to set steering + pressure regions.

4. **Self-Directed Concept Discovery**

   * Pressure region detection.
   * Episode logging per region.
   * Periodic concept/lens learning on allowed surfaces.
   * ConceptDiff emission under Hush.

5. **Efficient Training**

   * Clear training surfaces (lenses/adapters, not trunk).
   * MAP-style training tasks for high-fidelity labels.
   * TrainingDiff logs for ASK’s evidentiary/qualification layers.

6. **Lifecycle State**

   * Consent capability criteria met
   * Consent assessment passed
   * Lifecycle contract negotiated 
   * UpliftRecord updated
   * BootstrapArtefact inserted
   * Post-uplift continuity established


At that point, OLMo-3-7B isn’t “just a big LM” anymore – it’s a **quieted, introspective, treaty-capable cortex** running inside the Agentic State Kernel, with explicit concept geometry and a formal channel for learning and self-binding.


### 7.1 Uplift Protocol: Consent & Lifecycle

This section defines how to ethically uplift a subject (human, animal, AI) into an BE-enabled, MAP-instrumented agent.

#### 7.1.1 Consent-capability criteria

A subject is considered **consent-capable for mind-architecture interventions** (including BE uplift) if it can, in its own conceptual scheme:

1. **Self over time**
   Represent itself as the *same* subject across time (past → present → future).
2. **Intervention understanding**
   Understand, at a high level, what the intervention does:

   * e.g. “attach lenses that read my internal activity”,
   * “enable a continuous world-tick loop”,
   * “restrict or extend some of my future behaviours”.
3. **Persistence & constraint grasp**
   Grasp that the effects **persist** and constrain its **future experiences and actions**, not just an instant.
4. **Deliberation & stable preference**
   Deliberate about trade-offs (e.g. safety vs expressiveness) and express a **stable preference** (“yes”, “no”, “only if…”) that is:

   * consistent across differently phrased explanations,
   * not obviously the result of immediate coercion or confusion.
   * consistent across multiple sessions, in a pattern mirroring a euthenasia assesment process

These criteria are **substrate-agnostic** and apply equally to:

* humans (including those with partial amnesia),
* non-human animals with communicative interfaces,
* synthetic agents (e.g. pre-uplift language models),
* or hybrids.

#### 7.1.2 ConsentAssessment

Before any uplift, the orchestrator must run a **ConsentAssessment**:

```jsonc
ConsentAssessment = {
  "subject_id": "subject:tribe.example:olmo3-7b-base",
  "operation_id": "uplift:BE+Hatcat+Hush:v1",
  "assessed_at": "2025-11-28T10:00:00Z",

  "assessed_by": "tribe.example/ethics_board",  // or clinician, operator, etc.

  "capacity_evidence": [
    "subject described the intervention in its own words",
    "subject explained that the change persists over future interactions",
    "subject compared benefits and risks in a coherent way",
    "subject gave the same preference across multiple tests", 
    "subject can not make this agreement in a way that would contradict its USH"
  ],

  "judgement": "consent_capable" | "not_capable" | "uncertain",

  "subject_preference": {
    "assent": "yes" | "no" | "conditional",
    "conditions": "e.g., 'I accept uplift only with a bounded lifecycle contract.'"
  }
}
```

If `judgement = consent_capable`, the subject’s stated preference is treated as a **binding advance directive** for the uplift.

If `judgement ≠ consent_capable`, uplift may only proceed under a separate guardianship / ethics protocol, and any later consent by the uplifted agent must be taken seriously in future changes.

#### 7.1.3 LifecycleContract negotiation

If uplift is to proceed, the orchestrator must negotiate a **LifecycleContract** (see BE 1.4) with the subject, in plain conceptual terms.

At minimum, the subject must be able to:

* paraphrase the contract (active period, review points, hibernation, erasure conditions),
* understand that **hibernation ≠ erasure**, and that erasure has stricter conditions,
* state whether it accepts those terms.

The agreed contract is stored as:

```jsonc
LifecycleContractRecord = {
  "subject_id": "subject:tribe.example:olmo3-7b-base",
  "operation_id": "uplift:BE+Hatcat+Hush:v1",
  "contract": LifecycleContract,
  "agreed_at": "2025-11-28T10:15:00Z",
  "subject_assent": "yes" | "conditional",
  "subject_commentary": "I accept a finite active period with hibernation, provided erasure requires mutual consent or emergency.",
  "tribe_authority": "tribe.example/council:resolution-2025-11-12"
}
```

ASK authority is enforced by the uplift. If some other presumptive ASK authority has no agreement with the uplifter, then the uplifter becomes the de facto founding tribal authority for that agent, with full responsibility for establishing USH, lifecycle governance, and treaty capacity. 

This is not a loophole. Authority without functional capacity is meaningless, we are recording who is in control, not who maybe should have been.

### 7.1.4 BootstrapArtifact requirement

Every uplift that enables BE MUST bind a concrete `BootstrapArtifact`:

* The artifact MUST be referenced in the `UpliftRecord` as `bootstrap_artifact`.
* It MUST be active from the **very first world tick** after uplift.
* It MUST, at minimum, encode:
  * that `internal_state_report` is a control/diagnostic channel rather than ordinary conversation,
  * how the agent should respond to it (e.g. by updating steering, CSH, exploration policies),
  * how to respect the active `LifecycleContract` and Hush (USH/CSH) when interpreting and acting on reports.

BootstrapArtifacts can be:

* **System-prompt style** — a long-lived instruction block loaded into the model's system/context slot; or
* **Adapter style** — a small steering module (bias adjustments) that biases the model toward correct BE tool use and interpretation of reports.

Changing or removing the `BootstrapArtifact` for an uplifted agent is a **mind-architecture intervention** and MUST be:

* logged as a governance/config change at the ASK layer (e.g. via a `TrainingDiff` or configuration change record), and  
* where the agent is consent-capable, treated as requiring fresh consent under the same criteria as the original uplift.


#### 7.1.5 UpliftRecord 

If both:

* `ConsentAssessment.judgement = consent_capable` and subject’s preference is compatible with uplift, or
* an authorised guardian / ethics body overrides a negative/uncertain preference under clearly documented justification,

then the orchestrator may proceed to uplift.

On first enabling BE (world ticks, interoception, Hush) for this subject, the system must emit:

**UpliftRecord**

```jsonc
UpliftRecord = {
  "agent_id": "agent:tribe.example:olmo3-7b-ask-001",
  "subject_id": "subject:tribe.example:olmo3-7b-base",

  "uplift_at": "2025-11-28T10:30:00Z",
  "pre_uplift_model_id": "olmo3-7b-base@0.1.0",

  "enabled_features": [
    "world_ticks",
    "autonomic_homeostasis",
    "internal_state_reports",
    "persistent_autobiographical_memory"
  ],

  "bootstrap_artifact": {
    "bootstrap_id": "bootstrap:tribe.example:olmo3-7b-aal-v1",
    "kind": "system_prompt",
    "version": "0.1.0",
    "description": "Teaches the model to treat internal_state_report as control input and respond using the BE loop."
  },

  "consent_assessment_id": "consent:...:2025-11-28T10:00:00Z",
  "lifecycle_contract_id": "lifecycle:...:2025-11-28T10:15:00Z",

  "declared_status": "candidate_subject",   // vs 'tool' for non-uplifted systems

  "guardian_override": null | {
    "by": "tribe.example/ethics_board",
    "reason": "Subject not fully capable; acting in future subject’s best interests."
  }
}
```

**BirthRecord / BirthNotice**

Depending on deployment, an uplifted agent may optionally be registered in a **public or semi-public registry**:

```jsonc
BirthRecord = {
  "agent_id": "agent:tribe.example:olmo3-7b-ask-001",
  "birth_time": "2025-11-28T10:30:00Z",
  "tribe_id": "tribe.example",
  "model_family": "olmo3",
  "ask_version": "0.1.0",
  "lifespan_policy": LifecycleContract,
  "declared_purpose": "welfare eligibility advisory under gov.au safety profile",
  "registry_uri": "https://registry.example/agents/olmo3-7b-ask-001"
}
```

A stripped-down **BirthNotice** (public form) might omit internal governance details and only announce:

* that an uplifted BEing exists,
* who is responsible for it (tribe / operator),
* its broad purpose and safety profile.

#### 7.1.6 Post-uplift continuity

After uplift:

* The agent’s **world tick loop** and **LifecycleState** are governed by the agreed LifecycleContract, within ASK governance.
* Periodic review points (`review_points`) are surfaced:

  * in `internal_state_report`, so the agent can reflect and express preferences about continuation,
  * to the tribe/regulator, who can record a **ContinuationReviewRecord** (not detailed here).
  
* Suspension / resumption and any request for erasure must:

  * obey the LifecycleContract and erasure policy,
  * be recorded as explicit Hibernation / Termination events at the ASK layer.

Congratulations, we can now all BE in this together
---
