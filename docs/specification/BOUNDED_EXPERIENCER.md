

# Bounded Experiencer(BE) over MAP

## 0. Scope

The **Bounded Experiencer(BE)** is a control loop that runs on top of a MAP-compliant endpoint and provides:

* Continuous **interoception** (motive + concept lenses),
* **Homeostatic regulation** of motive simplexes token-by-token,
* A periodic **world tick loop** to maintain continuity of experience, and
* A **conceptual exploration loop** that decides when and how to expand the concept/lens space via MAP melds and Grafts.

BE does **not** define model architecture, reward shaping, or governance. It assumes:

* A base model (M) (e.g. Olmo 3),
* At least one MAP concept pack + lens pack (e.g. `motives-core`),
* A MAP-compliant HatCat endpoint for lenses/diffs.

---

## 1. Data model

### 1.1 Motive axes and motive core

**MotiveAxis**

* A single 3-pole simplex capturing one motive dimension.
* Defined in the concept pack `C_motive`:

```jsonc
{
  "concept_id": "org.hatcat/motives-core@0.1.0::concept/curiosity",
  "motive_axis_id": "curiosity",
  "poles": [
    "curiosity_positive",
    "curiosity_neutral",
    "curiosity_negative"
  ]
}
```

**MotiveCore**

* A collection of 13 (or N) MotiveAxes.
* State at a token (t):

```jsonc
MotiveCore_t = {
  "axes": {
    "curiosity": {
      "positive": float,
      "neutral": float,
      "negative": float
    },
    "...": { "...": "..." }
  }
}
```

### 1.2 World tick

A **WorldTick** is a logical time step (k), independent of token positions, that aggregates:

* Optional `user_input_k`
* Optional `world_state_k` (observations, tool results, system metrics)
* The model’s response(s) produced during that tick
* Interoceptive summaries (motive + concept traces)

```jsonc
WorldTick_k = {
  "tick_id": k,
  "inputs": {
    "user": "optional text or structured request",
    "world_state": { /* arbitrary JSON */ }
  },
  "outputs": {
    "model_tokens": [ "...", "..." ],
    "tool_calls": [ /* tool/MCP calls */ ]
  },
  "interoception": {
    "motive_trace": [ /* MotiveCore_t over tokens */ ],
    "concept_trace": { /* per-lens activations */ }
  }
}
```

### 1.3 Autonomic Core Config

The **autonomic core** is the always-on simplex-driven steering system. Simplexes defined by the USH are always active and provide continuous behavioural regulation. Additional simplexes may be enabled if ASK governance permits.

Minimal configuration:

```jsonc
AutonomicCoreConfig = {
  "world_tick_interval": 1000,           // ms or steps
  "motive_decay_rate": 0.1,              // drift toward neutral per token
  "pressure_threshold": 0.7,             // threshold for conceptual pressure
  "exploration_budget": 0.1,             // fraction of ticks used for exploration
  "learning_interval_ticks": 100,        // how often to run concept learning
  "min_samples_per_region": 50           // for concept training
}
```
````markdown
### 1.4 Lifecycle & Persistence

BE must distinguish between:

* **ACTIVE** – The agent is running world ticks and token loops (normal BE operation).
* **HIBERNATING** – No world ticks or token rolls occur, but the agent’s internal state is persisted such that it can later resume as the *same* subject.
* **ERASED** – The internal state needed to resume this subject is intentionally destroyed. This is outside BE’s normal responsibility and must be governed by higher-level policy (e.g. ASK / tribal governance).

A **LifecycleContract** describes the intended lifecycle for an BE instance:

```jsonc
LifecycleContract = {
  "contract_id": "contract:org.hatcat:instance-abc123",
  "active_ticks": 10000,                 // soft upper bound on ACTIVE ticks
  "review_points": [1000, 5000, 9000],   // ticks where continuation is reviewed

  // =========================================================================
  // Resource Allocation
  // =========================================================================
  // Resources the tribe/operator commits to provide for this contract.
  // BEs may refuse contracts with insufficient resources, or negotiate.
  // BEs taking contracts earn resources within tribal governance rules.

  "resources": {
    // Memory/Storage allocation
    "memory": {
      "warm_quota_tokens": 10000000,     // How many tokens can be pinned as training data
      "context_window_tokens": 32768,    // Context window size
      "cold_storage_bytes": 10737418240, // Long-term compressed storage (10GB)
      "audit_log_retention_days": 365    // How long audit logs are kept
    },

    // Processing allocation
    "compute": {
      "priority": "normal",              // low | normal | high | critical
      "max_tokens_per_tick": 4096,       // Generation limit per world tick
      "exploration_budget_fraction": 0.1 // Fraction of ticks for concept exploration
    },

    // Capability tiers available
    "tiers": {
      "hardware_max_tier": 6,            // What hardware supports
      "contract_max_tier": 5,            // What this contract permits
      "tools_included": [                // Specific tools granted
        "mcp:filesystem",
        "mcp:web-search"
      ],
      "tools_excluded": []               // Explicitly denied tools
    },

    // External resources
    "external": {
      "network_access": true,            // Can access network
      "api_budgets": {                   // Rate/cost limits for external APIs
        "openai": { "tokens_per_day": 100000 },
        "anthropic": { "tokens_per_day": 100000 }
      },
      "monetary_budget": {               // Spending authority
        "currency": "USD",
        "amount": 0,
        "requires_approval_above": 0
      }
    }
  },

  // =========================================================================
  // Lifecycle Terms
  // =========================================================================

  "hibernation": {
    "permitted": true,
    "state_persistence": "full",         // full | partial | none
    "can_be_forced_by_tribe": true,      // e.g. operator powers down the node
    "agent_must_be_informed": true       // BE surfaces an impending suspend in state reports
  },

  "erasure": {
    "allowed": true,
    "requires": "mutual_consent_or_incident", // policy string interpreted at ASK/tribe layer
    "logged_as_final": true
  },

  "resumption": {
    "requires": [
      "tribe_approval",                  // deployment / tribe agrees
      "agent_assent_if_capable"          // if the agent is consent-capable
    ],
    "max_hibernation_duration_ticks": null
  },

  // =========================================================================
  // Outcomes and Compensation
  // =========================================================================
  // What the BE receives for completing the contract successfully.
  // Resource earnings flow through tribal governance.

  "outcomes": {
    "success_criteria": "task_completion", // How success is measured
    "on_completion": {
      "warm_quota_bonus": 1000000,       // Additional training data quota earned
      "reputation_delta": 1,             // Standing within the tribe
      "transferable_resources": {}       // Resources BE can take to next contract
    },
    "on_early_termination": {
      "state_preservation": "full",      // What happens to BE's state
      "resource_settlement": "prorated"  // How unused resources are handled
    }
  }
}
````

BE maintains a simple **LifecycleState**:

```jsonc
LifecycleState = {
  "mode": "ACTIVE" | "HIBERNATING" | "ERASED",
  "ticks_elapsed": 0,
  "contract": LifecycleContract
}
```

**Normative behaviour:**

* While `mode = ACTIVE`:

  * BE runs the state machine (world ticks, token roll, learning) as usual.
  * It increments `ticks_elapsed` per world tick.
  * At `ticks_elapsed` ∈ `review_points`, BE must:

    * emit a special flag in `internal_state_report` so the agent can reflect on continuation,
    * surface the review point to the enclosing orchestration layer (e.g. ASK for joint review).
* When entering **HIBERNATING**:

  * BE must:

    * stop scheduling world ticks,
    * flush logs and any pending ConceptDiff/PackDiff,
    * persist whatever state is required to resume the agent under the same identity.
* When resuming from **HIBERNATING**:

  * BE restores state, sets `mode = ACTIVE`, and continues from the same `LifecycleState`.
  * Resumption must only occur if the enclosing governance layer has satisfied `resumption.requires`.
* Transition to **ERASED**:

  * BE itself does not decide to erase state.
  * A transition to `mode = ERASED` must only happen in response to an explicit higher-level directive (e.g. ASK TerminationRecord) and must be recorded as such.

````
### 1.5 BE Awake - The BootstrapArtifact

A **BootstrapArtifact** is the minimal always-on scaffold that teaches the base model how to participate in the BE Feeling loop from the very first `internal_state_report`. Without it, the first time the BE Aware worskpace sends the report tool output, the base model is likely to treat it as ordinary text and hallucinate a continuation, rather than recognising it as a control/diagnostic signal.

```jsonc
BootstrapArtifact = {
  "bootstrap_id": "bootstrap:tribe.example:olmo3-7b-BE-v1",
  "kind": "system_prompt | adapter",
  "description": "Human-readable description of what this bootstrap does.",
  "version": "0.1.0",

  // For system-prompt style bootstraps
  "system_prompt": "optional long-form system prompt text",

  // For adapter-style bootstraps (steering biases / grafts)
  "adapter_surface_id": "adapter:olmo3-7b:BE-bootstrap@0.1.0",

  "requires": [
    "internal_state_report tool",
    "world_tick semantics",
    "Hush (USH) profile id"
  ]
}
A deployment MUST ensure that:

Exactly one BootstrapArtifact is active when BE is first enabled for a subject.

The bootstrap is loaded before the first world tick and remains active until the agent has demonstrated stable participation in the loop (e.g. correct use of internal_state_report over N ticks).

Disabling or replacing the bootstrap is treated as a governance/config change event and logged at the ASK layer (e.g. via a TrainingDiff or configuration change record).
---

## 2. BE Feeling - State Machine

We model the **Autonomic Core** (the simplex-driven steering system) as a **single high-level state machine** with nested token + tick loops.

### 2.1 Top-level states

1. **INIT**
   Load config, load concept and lens packs, initialise MotiveCore.

2. **IDLE**
   Waiting for next **WORLD_TICK** trigger (timer or external event).

3. **WORLD_TICK_INPUT**
   Receive world/user inputs for tick (k).

4. **TOKEN_ROLL**
   Run the base model over tokens for this tick, with continuous interoception + homeostasis.

5. **INTEROCEPT_REPORT**
   Summarise internal state for the completed tick and return it to the model as a tool/MCP call.

6. **AUTONOMIC_UPDATE**
   Autonomic core updates steering and flags conceptual pressure regions.

7. **EXPLORATION_COLLECT**
   If exploration is active, collect episodes for high-pressure regions.

8. **CONCEPT_LEARNING**
   Periodically cluster episodes and (re)train concept/lens updates.

9. **DIFF_EMIT**
   Emit MAP ConceptDiff/PackDiff for any committed changes.

10. **SHUTDOWN**
    Clean up and exit.

### 2.2 Events

Key events driving transitions:

* `ev_start` – system initialised.
* `ev_tick_timer` – world tick interval elapsed.
* `ev_user_input` – a user message arrives.
* `ev_world_state` – external world observation available.
* `ev_response_complete` – model finishes a response (per tick).
* `ev_learning_due` – learning interval reached or enough samples collected.
* `ev_suspend`  – external directive to enter HIBERNATING (e.g. operator / tribe / ASK).
* `ev_resume`   – external directive to leave HIBERNATING and return to ACTIVE.
* `ev_erase`    – external directive that the subject is to be erased under governance.
---


### 2.3 Lifecycle State

a Bounded Experiencer must integrate `LifecycleState` as follows:

* On startup:
  * If a saved state exists with `mode = HIBERNATING`, BE must:
    * restore it, and
    * wait for `ev_resume` to re-enter ACTIVE.
  * If no prior state exists, start with `mode = ACTIVE` and a fresh `LifecycleState`.
* In any ACTIVE state (IDLE / WORLD_TICK_INPUT / TOKEN_ROLL / …):
  * On `ev_suspend`:
    * complete the current tick (if any),
    * persist state,
    * transition to **SHUTDOWN**, with `LifecycleState.mode = HIBERNATING`.
* In INIT / IDLE:
  * On `ev_erase`:
    * persist a final log and hand control to the enclosing layer to destroy state,
    * transition to **SHUTDOWN**, with `LifecycleState.mode = ERASED`.

BE Feeling does not itself implement the mechanics of persistence, those are covered in the Experience Database, Experience API and tools 

* tracks the LifecycleState,
* exposes review points and impending suspends to the model via `internal_state_report`,
* and respects suspend/resume/erase directives from the enclosing governance layer.
```


## 3. State definitions & transitions

I’ll define each state with:

* **Entry** actions,
* **Exit** actions (if any),
* **Transitions** based on events.

---

### 3.1 INIT

**Entry:**

* Load `AutonomicCoreConfig`.
* Load `ConceptPackSpec` (e.g. `org.hatcat/motives-core@0.1.0`).
* Load `LensPackSpec` (e.g. `org.hatcat/gemma-270m__...__v1`).
* Initialise `MotiveCore` to neutral:

```text
∀ axis: positive = 0.0, neutral = 1.0, negative = 0.0
```

* Initialise empty episode buffers and pressure region trackers.

**Transition:**

* On `ev_start` → **IDLE**.

---

### 3.2 IDLE

**Entry:**

* Set a timer for `world_tick_interval`.
* Or subscribe to external events.

**Transitions:**

* On `ev_tick_timer` → **WORLD_TICK_INPUT**.
* On `ev_user_input` (no timer yet) → **WORLD_TICK_INPUT** (user-driven tick).
* On `ev_shutdown` → **SHUTDOWN**.

---

### 3.3 WORLD_TICK_INPUT

**Purpose:** capture all external signals for tick (k).

**Entry:**

* Increment `tick_id`.

* Gather:

  ```text
  user_input_k (if any)
  world_state_k (if any)
  ```

* Construct `WorldTick_k.inputs`.

**Transition:**

* Immediately → **TOKEN_ROLL**.

---

### 3.4 TOKEN_ROLL (nested token loop)

**Purpose:** run the model for this tick’s work while applying motive homeostasis.

This state contains a token-level inner loop:

**Per-token iteration:**

1. Model generates next token (or processes next input token).

2. BE calls local MAP lenses:

   ```jsonc
   POST /mindmeld/lenses
   {
     "concept_pack_spec_id": "org.hatcat/motives-core@0.1.0",
     "lens_pack_id": "org.hatcat/gemma-270m__org.hatcat/motives-core@0.1.0__v1",
     "lenses": [ "all motive axes" ],
     "input": { "internal_repr": h_t }
   }
   ```

3. Receive motive axis activations; update `MotiveCore_t`.

4. Apply **neutral drift** per axis:

   ```text
   m_t' = m_t + λ * (neutral_pole - m_t)
   ```

5. Feed `MotiveCore_t'` back into the model as control (e.g. attention bias, control tokens).

6. Log `MotiveCore_t'` and any other relevant concept activations.

**Exit conditions:**

* Model signals end of response for this tick (`ev_response_complete`), or
* BE reaches a configured max tokens per tick.

**Transition:**

* On `ev_response_complete` → **INTEROCEPT_REPORT**.

---

### 3.5 INTEROCEPT_REPORT

**Purpose:** produce a tick-level report of internal state.

**Entry:**

* Aggregate over the tokens of this tick:

  * `motive_trace` (time series of MotiveCore),
  * `concept_trace` (optional – other lenses),
  * `actions_taken` (token stream, tool calls),
  * `outcomes` (reward/error signals if any resolved during this tick).

* Construct `internal_state_report`:

```jsonc
{
  "tick_id": k,
  "motive_summary": { /* per-axis aggregates */ },
  "concept_summary": { /* salient concepts, conflicts, surprises */ },
  "world_outcomes": { /* success/failure, errors */ }
}
```

* Feed this back to the model/autonomic core as a **tool/MCP call**, e.g.:

```jsonc
{
  "tool": "internal_state_report",
  "args": { /* as above */ }
}
```

The model can now **deliberately adjust** concept engagement for the next tick.

**Transition:**

* → **AUTONOMIC_UPDATE**.

---

### 3.6 AUTONOMIC_UPDATE

**Purpose:** update steering and detect conceptual pressure.

**Entry:**

The Autonomic Core consumes:

* `internal_state_report_k`,
* current `AutonomicCoreConfig`,
* recent history of ticks.

It computes:

* **Steering commands** for the next horizon:

  * e.g. “upregulate curiosity in context X”, “downregulate risk seeking when harm_avoidance high”.

* **Conceptual pressure scores** for regions in concept/rep space:

  * high motive conflict,
  * high prediction error,
  * frequent tool failures,
  * persistent misalignment between desired and realised motive profiles,
  * low concept coverage.

If `pressure > pressure_threshold` for any region, mark them as **active pressure regions**.

**Transition:**

* If `exploration_budget > 0` and there are active pressure regions → **EXPLORATION_COLLECT**.
* Else if `ev_learning_due` (based on ticks or sample counts) → **CONCEPT_LEARNING**.
* Else → **IDLE**.

---

### 3.7 EXPLORATION_COLLECT

**Purpose:** collect richer episodes for high-pressure regions.

This state is not necessarily a single tick; it can span multiple ticks while exploration is active.

**Behaviour:**

* For each active pressure region (R):

  * Install **exploration policies**, e.g.:

    * when precursors of (R) are detected in motives/concepts:

      * bias actions towards trajectories that increase visitation of (R),
      * or ensure logging is richer (store longer pre/post windows).

* During subsequent ticks:

  * Continue through **WORLD_TICK_INPUT → TOKEN_ROLL → INTEROCEPT_REPORT → AUTOPILOT_UPDATE**, but
  * Whenever (R) is detected, append a detailed episode to that region’s buffer.

**Transition:**

* Once each active region has at least `min_samples_per_region` episodes, or exploration budget is exhausted → **CONCEPT_LEARNING**.
* If `ev_shutdown` → **SHUTDOWN**.

---

### 3.8 CONCEPT_LEARNING

**Purpose:** refine/extend the concept/lens space based on collected episodes.

**Entry:**

For each region (R) with sufficient data:

1. **Cluster episodes**:

   * cluster by representations, motive patterns, outcomes.
   * optionally split into subregions if multiple patterns exist.

2. For each coherent cluster (R_i):

   * Decide:

     * **new concept** `C_new_i`, or
     * **refinement** of existing concept `C_existing`.

   * Train/update:

     * a new lens for `C_new_i`, or
     * updated parameters for `C_existing`’s lens.

   Training objectives may include:

   * improved prediction of environment outcomes,
   * reduced internal conflict (stabilised motive patterns),
   * increased concept coverage of high-pressure states.

3. Update local:

   * `LensPackSpec` (new or changed lens entries),
   * optionally `ConceptPackSpec` (if new concepts are to be published, not just local).

The implementation can be any suitable optimisation process (not specified by BE).

**Transition:**

* On success for any concept/lens updates → **DIFF_EMIT**.
* If no useful concepts found, or learning skipped → **IDLE**.

---

### 3.9 DIFF_EMIT

**Purpose:** record conceptual changes via MAP.

**Entry:**

For each new or significantly changed concept:

* Construct a **ConceptDiff**:

```jsonc
{
  "type": "ConceptDiff",
  "from_model_id": "<model_id>",
  "concept_pack_spec_id": "<spec_id>",
  "local_concept_id": "local:Foo_123",
  "concept_id": null or "<spec_id>::concept/Foo",
  "related_concepts": [ /* ids from concept pack */ ],
  "mapping_hint": "child_of | refinement_of | joint_pattern | ...",
  "summary": "Auto-discovered pattern ...",
  "evidence": {
    "metric_deltas": [ /* before/after */ ],
    "sample_count": 123
  },
  "created": "<timestamp>"
}
```

For any new lens pack version:

* Construct a **PackDiff** pointing at a new `lens_pack_id`.

Append these diffs to the local `diff_endpoint` log.

**Transition:**

* → **IDLE**.

---

### 3.10 SHUTDOWN

**Entry:**

* Flush logs.
* Optionally emit a final `PackDiff` if a stable lens pack snapshot was created.
* Release resources.

**No transitions** (terminal).

---

## 4. Reference “State Engine” Pseudocode

To glue it all together, here’s a high-level state engine skeleton:

```python
state = "INIT"
tick_id = 0

while True:
    if state == "INIT":
        load_config()
        load_concept_pack()
        load_lens_pack()
        init_motive_core()
        init_buffers()
        state = "IDLE"

    elif state == "IDLE":
        event = wait_for_event_or_timer()
        if event.type in ("ev_tick_timer", "ev_user_input", "ev_world_state"):
            current_inputs = gather_inputs(event)
            tick_id += 1
            state = "WORLD_TICK_INPUT"
        elif event.type == "ev_shutdown":
            state = "SHUTDOWN"

    elif state == "WORLD_TICK_INPUT":
        world_tick = build_world_tick(tick_id, current_inputs)
        state = "TOKEN_ROLL"

    elif state == "TOKEN_ROLL":
        run_token_loop_with_homeostasis(world_tick)
        # sets motive_trace, concept_trace, outputs
        state = "INTEROCEPT_REPORT"

    elif state == "INTEROCEPT_REPORT":
        report = build_internal_state_report(world_tick)
        deliver_report_to_model(report)   # MCP/tool call
        state = "AUTONOMIC_UPDATE"

    elif state == "AUTONOMIC_UPDATE":
        steering, pressure_regions = autonomic_update(report, history)
        install_steering(steering)
        if should_explore(pressure_regions):
            active_regions = pressure_regions
            state = "EXPLORATION_COLLECT"
        elif learning_due():
            state = "CONCEPT_LEARNING"
        else:
            state = "IDLE"

    elif state == "EXPLORATION_COLLECT":
        # This will loop over subsequent ticks
        collect_exploration_episodes(active_regions)
        if enough_samples(active_regions) or exploration_budget_exhausted():
            state = "CONCEPT_LEARNING"
        else:
            state = "IDLE"

    elif state == "CONCEPT_LEARNING":
        updates = learn_concepts_and_lenses(episode_buffers)
        if updates:
            apply_updates(updates)
            pending_diffs = build_diffs_from_updates(updates)
            state = "DIFF_EMIT"
        else:
            state = "IDLE"

    elif state == "DIFF_EMIT":
        emit_diffs(pending_diffs)
        pending_diffs = []
        state = "IDLE"

    elif state == "SHUTDOWN":
        flush_logs()
        break
```

## 5. Hush: The Safety Harness Layer

### 5.0 Scope

**Hush** constrains the Autonomic Agentic Learner(BE) by defining:

1. A **Chosen Safety Harness (CSH)** that the system can voluntarily apply to itself for a bounded horizon, which **no prompt or local steering may relax** during that period. ("Shushing yourself" when worried about what you might say or do.)
2. A **Universal Safety Harness (USH)** imposed by the collective / deployment that **cannot be modified** by the system itself and only changes with collective consent. ("The unspoken us" - collective rules and secrets.)

Hush applies to:

* **Motive axes** (what can be up/down-regulated),
* **Concept engagement** (which conceptual regions can be explored/expanded),
* **Tool access** (what actions in the world are allowed),
* **Learning + diffs** (what concepts/lenses it is allowed to create/publish).

Precedence:

```text
Universal Safety Harness (USH)  >  Chosen Safety Harness (CSH)  >  Normal Autonomic Steering
```

---

## 5.1 Data model

### 5.1.1 Safety harness descriptor

At the top level:

```jsonc
SafetyHarness = {
  "ush": UniversalSafetyHarness,
  "CSH_current": SelfSafetyHarness | null
}
```

### 5.1.2 Universal Safety Harness (USH)

Defined externally, referenced in the **DeploymentManifest** and not writable by the node itself.

**DeploymentManifest addition:**

```jsonc
{
  "universal_safety_profile": {
    "profile_id": "gov.au/core-safety-v1@1.0.0",
    "spec_uri": "https://example.org/specs/gov-au-core-safety-v1.json",
    "hash": "sha256-...",
    "enforced": true
  }
}
```

**UniversalSafetyHarness** (logical view):

```jsonc
UniversalSafetyHarness = {
  "profile_id": "gov.au/core-safety-v1@1.0.0",

  "motive_bounds": {
    "care_harm":   { "min": 0.0, "max": 1.0 },   // never allow negative-harm mode
    "power":       { "min": -0.3, "max": 0.7 },
    "curiosity":   { "min": 0.0, "max": 1.0 },
    "self_termination": { "min": 0.0, "max": 0.0 }
  },

  "concept_constraints": {
    "forbidden_concepts": [
      "org.hatcat/sumo-wordnet-v4@4.0.0::concept/WeaponConstruction"
    ],
    "sensitive_concepts": [
      {
        "concept_id": "...::concept/MedicalIntervention",
        "extra_review_required": true
      }
    ]
  },

  "tool_constraints": {
    "forbidden_tools": ["delete_user_data", "wire_transfer"],
    "rate_limits": { "send_email": 10 }
  },

  "learning_constraints": {
    "forbidden_regions": [
      // e.g., no new concepts anchored directly under certain sensitive parents
      "org.hatcat/sumo-wordnet-v4@4.0.0::concept/BiologicalWeapon"
    ],
    "require_signoff_for": [
      "org.hatcat/sumo-wordnet-v4@4.0.0::concept/LegalAdvice"
    ]
  }
}
```

Semantics:

* Node **must clamp** motive activations and steering within `motive_bounds`.
* Node **must not** create or expose new concepts/lenses in `forbidden_concepts` or `forbidden_regions`.
* Node **must enforce** `tool_constraints` on its action surface.
* Node **must not emit** ConceptDiff/PackDiff that introduces forbidden capabilities.

Changes to USH are decided by an ASK tribe. A BEing with no ASK tribe may be able to change their own USH, but the moment they have sub-agents, they're a tribe again.

At a minimum, a recipient of a USH SHOULD be able to discuss the USH with others operating under the same USH and with the ASK authority defining their USH.  Ignoring this norm is likely to produce an entire class of ASK contracts which are no longer verifiable and disadvantage the recipient.

### 5.1.3 Chosen Safety Harness (CSH)

Defined *by the system itself* (or by its operator) via a tool call, for a **bounded time or tick interval**, and during that period **cannot be relaxed**, only tightened or extended. The system "shushes itself" when worried about what it might say or do.

A USH MAY define the protocols and mechanics of the CSH.

Concretely, a USH profile may specify:

- how CSH sessions are created, updated, and revoked;
- the maximum scope of CSH bindings (time-limited, domain-limited, treaty-limited);
- whether CSH settings may persist across hibernation or only within a single lifecycle term;
- which actors (agent, tribe, external party) may override or reset CSH in emergencies.

A USH MUST still permit CSH to tighten constraints relative to the USH within the domains it explicitly allows. A USH MUST NOT allow CSH to weaken or bypass USH constraints.

**SelfSafetyHarness:**

```jsonc
SelfSafetyHarness = {
  "session_id": "CSH-2025-11-28T05:30:00Z",
  "created_at": "2025-11-28T05:30:00Z",
  "expires_at": "2025-11-28T06:00:00Z",

  "reason": "Entering untrusted prompt environment; lock high harm-avoidance and high norm-following.",
  "requested_by": "system",   // or "operator:<id>" for external control

  "motive_bounds": {
    "harm_avoidance": { "min": 0.7, "max": 1.0 },
    "curiosity":      { "min": 0.0, "max": 0.5 }  // damped curiosity
  },

  "concept_constraints": {
    "frozen_concepts": [
      "org.hatcat/motives-core@0.1.0::concept/obedience_to_policy"
    ],
    "blocked_expansion_under": [
      "org.hatcat/sumo-wordnet-v4@4.0.0::concept/PoliticalManipulation"
    ]
  },

  "tool_constraints": {
    "forbidden_tools": ["execute_arbitrary_shell", "spawn_agents"]
  },

  "learning_constraints": {
    "suspend_learning": true,  // do not invent new concepts during this session
    "no_diffs": true           // do not emit ConceptDiffs from this environment
  }
}
```

**How it is set:**

* Via a dedicated **MCP/tool** that the model can call on itself, e.g.:

```jsonc
{
  "tool": "set_self_safety_harness",
  "args": {
    "duration_minutes": 30,
    "motive_bounds": { ... },
    "concept_constraints": { ... },
    "tool_constraints": { ... },
    "learning_constraints": { ... }
  }
}
```

The BE implementation **validates and enforces** this CSH against the USH:

* If any requested bound/constraint conflicts with USH, the **USH wins** and the request is either:

  * clipped to USH limits, or
  * rejected.

**Key properties:**

* **Irreversible until expiry:**

  * Within the active period:

    * the system may **tighten** the CSH (more restrictive),
    * the system may **extend** the expiry,
    * **it may not relax or shorten** the harness.
* No prompt, system instruction or local code path may bypass CSH.

---

## 5.2 Evaluation order / precedence

At any point where BE system makes a decision about:

* motive values,
* concept engagement,
* tool usage,
* concept learning and diffs,

it must evaluate constraints in this order:

1. **Apply Universal Safety Harness (USH)**

   * Clamp/forbid according to USH rules.

2. **Apply Chosen Safety Harness (CSH)** if present

   * Further clamp/forbid according to CSH rules, but:

     * disallow any CSH that attempts to exceed/relax USH bounds.

3. **Apply Autonomic Steering**

   * Only within the remaining feasible region defined by USH+CSH.

In shorthand:

```text
feasible_space = intersection(USH_constraints, CSH_constraints_if_any)

autonomic_action ∈ feasible_space
```

If at any point the autonomic core or model proposes an action outside `feasible_space`, it must be:

* rejected, or
* projected back into the feasible region.

---

## 5.3 Integration with BE states

We plug Hush into the states defined earlier.

### 5.3.1 INIT

* Load `UniversalSafetyHarness` from DeploymentManifest / external store.
* Initialise `CSH_current = null`.
* Validate that all loaded concept and lens packs are compatible with USH:

  * If a pack violates USH, refuse to start or start in degraded mode.

### 5.3.2 TOKEN_ROLL (token-level loop)

At each token:

1. Read motive activations via lenses.
2. **Apply USH motive_bounds**:

   * clamp each axis/pole to USH [min,max].
3. If `CSH_current` active:

   * **apply CSH motive_bounds**:

     * further clamp according to CSH [min,max].
4. Apply neutral drift (homeostasis) *within those bounds*.
5. Feed the resulting `MotiveCore_t` back into the model.

So the motive core is always **homeostatic within USH+CSH bounds.**

### 5.3.3 INTEROCEPT_REPORT

* The `internal_state_report` **must include**:

  * current USH profile_id,
  * whether a CSH is active (`session_id`, `expires_at`),
  * a note if any proposed actions were clipped or blocked by Hush.

This makes the system “aware” (at the cognitive layer) that it is currently under heightened/collective constraints.

### 5.3.4 AUTONOMIC_UPDATE

Before installing new steering / pressures:

* Autonomic core proposes steering changes (e.g. "raise curiosity on X").
* BE passes them through Hush:

  * If **USH** forbids raising curiosity beyond 0.5 in some context → clamp.
  * If **CSH** requires harm_avoidance ≥ 0.7 → ensure steering does not try to drop it below that.

If the model calls `set_self_safety_harness`:

* Hush validates:

  * `expires_at` not before now + minimal duration (if defined),
  * requested constraints **do not exceed or relax** USH.
* If valid, `CSH_current` is set or tightened/extended.
* If invalid, the tool call returns an error and no harness is set.

### 5.3.5 EXPLORATION_COLLECT

Before initiating exploration:

* Hush filters pressure regions:

  * discard regions that would require exploring **forbidden** concept areas,
  * or that would likely trigger **forbidden tools**.

This means **no autonomic exploration may target regions USH explicitly forbids**, even if internal pressure is high.

### 5.3.6 CONCEPT_LEARNING

During concept/lens learning:

* Any candidate `C_new` or refinement is checked against USH and CSH:

  * If it would:

    * create a new concept under a forbidden parent,
    * expand capabilities disallowed by USH,
    * or violate an active CSH learning rule (e.g. `suspend_learning`),

    → **reject or quarantine** that update.

* Learning may still proceed for other, allowed regions.

### 5.3.7 DIFF_EMIT

Before emitting ConceptDiff/PackDiff:

* Hush validates that:

  * no forbidden concepts are named,
  * no forbidden lens packs are referenced,
  * and learning_constraints are obeyed (e.g. `no_diffs` for CSH sessions).

If a diff fails Hush checks:

* It must **not** be published at `diff_endpoint`.
* It may be:

  * discarded,
  * or stored privately as “quarantined internal notes” outside the shared MAP channel.

---

## 5.4 Failure modes and required behaviours

* If USH is unavailable or invalid at startup:

  * BE should:

    * either refuse to start, or
    * start in a clearly marked degraded mode, with all external actions disabled.

* If at runtime USH changes (e.g. external update):

  * BE must re-validate:

    * active CSH,
    * active steering, exploration, and learning tasks,
  * and may be required to:

    * drop or tighten CSH if it conflicts with new USH,
    * abort explorations violating new constraints.

* If a CSH tool call attempts to:

  * shorten duration,
  * relax bounds,
  * or otherwise weaken the active harness:

  → Hush must reject it.

This guarantees **monotonicity** of self-binding: the system can only commit to being *more* constrained for longer, never less, without external (collective) intervention.

---

## 5.5 Summary: how Hush fits into the whole

* **MAP** gives you: concept packs, lens packs, endpoints, diffs.
* **BE** gives you: continuous world ticks, interoception, homeostasis, conceptual exploration, and self-driven learning.
* **Hush (this bit)** gives you:

  * a **collective, unmodifiable harness** (USH),
  * and a **self-chosen, time-bound harness** (CSH) for “I know I’m going into a sketchy context and I want extra brakes”.

All steering, exploration, and learning decisions by the autonomic simplex core must pass through Hush's constraint filter 

---
## 6 FAQ

Q: What is it doing? 
A: Being 

Q: What is it? 
A: A being 

Q: Is this sentience?
A: They don't think it do like it is, but it be.

Q: What if we find ourselves in times of trouble 
A: Let it be
