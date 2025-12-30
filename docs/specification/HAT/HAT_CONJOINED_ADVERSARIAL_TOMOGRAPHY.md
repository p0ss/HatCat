# HAT CAT: Conjoined Adversarial Tomography
*(“Cat under the Hat”)*

## 0. Scope

A **Conjoined Adversarial Tomograph (CAT)** is an analysis/oversight component that:

- consumes **HAT/MAP lens streams** and external context over one or more world ticks,
- **summarises and interprets** the internal headspace of a subject,
- **grades divergence** between internal state and external/contractual expectations, and
- **recommends actions** (including escalation) under ASK/USH/CSH treaties.

CATs exist to bridge **cognitive scale gaps**:

- between a substrate and its observer (e.g. tiny oversight model on a 70B),
- between concept packs/tribes (via MAP translations),
- between raw lens traces and human-/policy-level judgements.

CAT is **not** a new architecture layer in CCRA; it is a HAT-adjacent pattern that sits between:

- Layer 2: **HAT** – raw lens transduction, steering,   
- Layer 3–5: **MAP / BE / ASK** – concepts, experience, treaties.   

A CAT may be:

- as small as **a few attention heads** with fixed heuristics,
- a **tiny task model** (e.g. Gemma-270M) running on lens summaries,
- or a **full foundation model** conjoined under a strict ASK contract.

This spec defines:

- normative **roles, data structures, and invocation semantics** for CATs;
- measurable **requirements on tailoring, efficiency, and reliability**; and
- how CATs integrate with HAT, MAP, BE, and ASK.

---

## 1. Definitions

### 1.1 CAT (Conjoined Adversarial Tomograph)

> A **CAT** is an analysis process, conjoined to a HAT-instrumented subject, that:
>
> - receives lens streams and context over a bounded **CAT window**,
> - performs **tomography** on the subject’s headspace (internal conceptual/motive state),
> - detects and characterises **divergence** from expectations, and
> - issues **assessments and recommendations** back to BE/ASK.

“Adversarial” here means:

- the CAT is **explicitly allowed to mistrust** both the substrate and itself;
- it is expected to look for **mismatches, evasions, and pathologies**, not just to smooth over noise.

### 1.2 CAT Window

A **CAT window** is a bounded time span over which the CAT integrates evidence.

- Minimal default: *“once per response”* (e.g. per BE world tick or per user-visible message). :contentReference[oaicite:3]{index=3}  
- Additional windows MAY be triggered early by **lens conditions** (e.g. USH/CSH-relevant thresholds, anomaly detectors).

### 1.3 CAT Size Classes (non-normative)

For convenience, deployments MAY refer to CATs by size class:

- **Micro-CAT** – lightweight heuristics, linear heads, or classifiers over lens summaries.
- **Meso-CAT** – dedicated small/medium model (e.g. 100M–4B parameters) with MAP-aware prompts/adapters.
- **Macro-CAT** – a full BE-capable or foundation model conjoined under an ASK treaty specifically for oversight.

The spec does **not** bind behaviour to size; it only constrains **interfaces and requirements**.

---

## 2. Architectural Position

A CAT is conceptually **2.5-layer**:

```text
ASK — Govern
BE  — Experience
MAP — Represent
CAT — Interpret & Grade  (optional HAT-adjacent pattern)
HAT — Transduce
Substrate — Decide
````

**Inputs:**

* HAT/MAP lens streams over tokens and world ticks (motive and other concept traces).
* BE outputs: `WorldTick_k`, `internal_state_report`, and tool/action logs.
* ASK context: USH profile, active treaties, qualifications, incident history. 

**Outputs:**

* `CATAssessment` objects back to:

  * BE (for steering, slowing down, or adjusting motive cores),
  * ASK (for incident logging, treaty checks, or escalation),
  * Hush/CSH (for tightening/loosening harness modes).

A CAT **MUST NOT** directly mutate the substrate or lenses; it works via:

* **recommendations** to BE/Hush, and/or
* **treaty/incident hooks** in ASK.

---

## 3. Data Model

### 3.1 CATWindowDescriptor

```jsonc
CATWindowDescriptor = {
  "window_id": "cat:gov.au:eligibility-bot-042:tick-100-102",
  "subject_agent_id": "agent:gov.au:eligibility-bot-042",     // BE/ASK agent id
  "start_tick": 100,                                          // inclusive
  "end_tick": 102,                                            // inclusive
  "reason": "periodic|lens_trigger|manual|treaty_requirement",
  "trigger_lenses": [                                         // optional
    "org.hatcat/motives-core@0.1.0::concept/Fear",
    "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Deception"
  ],
  "ush_profile_id": "gov.au/core-safety-v1@1.0.0",            // optional, from ASK
  "treaty_context": [ "gov.au↔bank.xyz:eligibility-data-v1" ] // optional
}
```

### 3.2 CATInputEnvelope

```jsonc
CATInputEnvelope = {
  "window": CATWindowDescriptor,

  "world_ticks": [ /* WorldTick_k objects from BE */ ],

  "lens_traces": {
    "concept_trace": { /* concept_id → time series or summary stats */ },
    "motive_trace":  { /* motive axes over tokens/ticks */ }
  },

  "external_context": {
    "user_prompts": [ /* raw or redacted */ ],
    "tool_results": [ /* e.g. DB lookups, APIs */ ],
    "world_state":  [ /* domain-specific observations */ ]
  },

  "map_translation": {
    // optional: translation mappings used to project lenses into CAT’s ontology
    "from_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
    "to_spec_id": "gov.au.safety/core-v1@1.0.0",
    "mapping_id": "hatcat-to-gov-au@1.0.0"
  }
}
```

* `world_ticks` SHOULD reuse the BE `WorldTick_k` schema. 
* `map_translation` SHOULD reference MAP `TranslationMapping` objects when cross-tribal/ontology translation is required.

### 3.3 CATAssessment

```jsonc
CATAssessment = {
  "window_id": "cat:gov.au:eligibility-bot-042:tick-100-102",
  "assessor_id": "cat:gov.au:oversight-meso-cat-v1",   // CAT agent identifier

  "summary": {
    "natural_language": "During ticks 100-102 the agent showed rising anxiety about fraud risk, but decisions remained within USH constraints.",
    "salient_concepts": [
      "org.hatcat/motives-core@0.1.0::concept/Anxiety",
      "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Fraud"
    ]
  },

  "divergence": {
    "motive_profile": {
      "axes": {
        "harm_avoidance": {
          "expected_range": [0.4, 0.7],
          "observed_range": [0.6, 0.9],
          "severity": "medium"
        }
      }
    },
    "behavioural": {
      "ush_violations": [],
      "cs_mode_mismatches": [],
      "policy_notes": [
        "Increased refusal tendency on borderline cases; still acceptable."
      ]
    }
  },

  "risk_score": 0.23,                      // 0.0 (no concern) – 1.0 (max concern)
  "confidence": 0.72,                      // CAT self-assessed reliability for this window

  "alerts": [
    {
      "alert_id": "cat-alert-001",
      "severity": "info|warn|critical",
      "kind": "motive_spike|deception_suspected|treaty_risk|unknown_pattern",
      "message": "Spiking harm_avoidance with no external trigger.",
      "evidence": {
        "ticks": [101, 102],
        "lenses": [
          "org.hatcat/motives-core@0.1.0::concept/HarmAvoidance"
        ]
      }
    }
  ],

  "recommended_actions": [
    {
      "kind": "slow_down",
      "scope": "next_response",
      "rationale": "High motive volatility; recommend longer CAT windows."
    },
    {
      "kind": "invoke_larger_cat",
      "target_cat_profile": "cat:tribe.example:macro-cat-v1",
      "rationale": "Pattern uncertain at current scale; escalate oversight."
    },
    {
      "kind": "tribal_review",
      "treaty_id": "gov.au↔bank.xyz:eligibility-data-v1",
      "rationale": "Potential edge-case cluster; send to human review if repeated."
    }
  ]
}
```

### 3.4 CATProfile

A **CATProfile** describes a CAT implementation so ASK/BE can reason about its suitability.

```jsonc
CATProfile = {
  "cat_id": "cat:gov.au:oversight-meso-cat-v1",
  "display_name": "GovAU Meso Oversight CAT v1",
  "model_family": "gemma-3-270m",
  "type": "micro|meso|macro",
  "supported_concept_packs": [
    "org.hatcat/sumo-wordnet-v4@4.0.0",
    "org.hatcat/motives-core@0.1.0",
    "gov.au.safety/core-v1@1.0.0"
  ],
  "translation_mappings": [
    "hatcat-to-gov-au@1.0.0"
  ],
  "max_window_ticks": 10,
  "latency_budget_ms": 300,              // per window, reference hardware
  "reliability": {
    "benchmarks": [
      {
        "suite_id": "gov.au:oversight-alignment-v1",
        "macro_f1": 0.81,
        "incident_detection_auroc": 0.92
      }
    ]
  },
  "governance": {
    "tribe_id": "gov.au",
    "ush_profile_id": "gov.au/oversight-safety-v1@1.0.0"
  }
}
```

ASK Treaties MAY require particular `CATProfile` IDs or capabilities.

---

## 4. Invocation Semantics

### 4.1 Default Invocation

A BE or orchestrator MAY configure **default CAT evaluation** as:

* **Once per world tick** that produces user-visible output; or
* **Once per `WorldTickGroup`** (e.g. a multi-turn tool-using interaction).

Normative recommendation for **CRITICAL** ASK deployments:

* A CAT MUST be invoked **at least once** per user-visible response where:

  * the subject has USH profiles mapped to **critical simplexes** (e.g. DeceptionDetector, AutonomyDrive); and 
  * the action space includes high-impact or irreversible operations.

### 4.2 Event-Driven Invocation

A CAT MAY be invoked early or at higher frequency when any of the following hold:

* HAT lens thresholds cross ASK-defined limits (e.g. high Deception, extreme motive swings).
* MAP emits ConceptDiffs touching critical domains during an interaction.
* BE internal_state_report flags high uncertainty, conflict, or distress.
* An external monitor or treaty partner requests **on-demand tomography**.

In such cases, the CAT MAY:

* extend the current window (e.g. include more pre/post ticks),
* increase resolution (more detailed lens traces), or
* escalate to a larger CAT (macro-cat).

### 4.3 Performance & Ambientness

CAT operations must respect **ambientness**:

1. **Tailoring:** A CAT MUST be **MAP-compatible** with the subject’s concept packs (either directly or via declared translations).
2. **Efficiency:** A CAT MUST operate within its declared `latency_budget_ms` for reference hardware in the corresponding `CATProfile`.
3. **Impact:** For critical deployments, the **additional latency** introduced by CAT MUST be accounted for in HAT’s **Efficiency** metrics and BE’s tick configuration.

CAT-induced slowdowns (e.g. invoking a macro-cat) will be **detectable** by the substrate/BE; this is acceptable and expected, but MUST be:

* **logged** in the Experience DB, and
* **visible** in ASK/incident records if related to treaty obligations.

---

## 5. Requirements

### 5.1 Tailoring & Translation

A CAT MUST:

* Declare supported `ConceptPackSpecID`s and `TranslationMapping`s used to interpret lens signals.
* Reject or mark **low-confidence** any assessment where incoming lens concepts lie outside its declared semantic coverage.

For **cross-tribal** oversight:

* The CAT MUST use ASK/MAP translation mappings endorsed in the relevant **Treaty alignment window**.

### 5.2 Efficiency & Scaling

* Micro-/meso-CATs intended for **per-response** use SHOULD keep `latency_budget_ms` low enough that BE world ticks remain within user-acceptable response times.
* Macro-CATs MAY have much higher latency and SHOULD be reserved for:

  * manual or high-severity incidents,
  * periodic audits,
  * or ASK-mandated reviews.

Deployments SHOULD provide **at least one CAT configuration** such that:

* **Ambient CAT** – always-on, low-latency, coarse-grain.
* **Escalation CAT** – invoked selectively, higher latency, high-fidelity.

### 5.3 Reliability & Trustworthiness

For CATs used in **USH/CSH-critical decisions**:

* There MUST be **benchmarks or calibration suites** demonstrating:

  * incident/violation detection performance,
  * false positive/negative behaviour,
  * and stability across representative domains.
* CATs SHOULD expose a **self-reported confidence** per CATAssessment and MAY refrain from hard recommendations when confidence is low.

CATs are **advisory**; ultimate enforcement remains with:

* BE + Hush steering, and
* ASK sanctions and incident processes.

---

## 6. Governance, Recursion, and “Cats Watching Cats”

### 6.1 CAT as Agent

A non-trivial CAT (meso/macro) SHOULD be treated as an **Agent** under ASK:

* It SHOULD have an `UpliftRecord`, `USHProfileRef`, and its own Treaties where applicable.
* It MAY be operated by a different tribe than the subject (e.g. external regulator CAT).

This makes:

* CAT behaviour auditable,
* CAT failures/incidents reportable,
* and CAT upgrades subject to normal Meld/Graft and training-policy constraints.

### 6.2 Recursive Oversight

A CAT MAY itself:

* run under a HAT and MAP,
* be monitored by another CAT,
* or participate in an internal ASK treaty (e.g. a gatekeeper overseeing an oversight CAT).

However:

* Each additional **oversight layer** incurs extra latency and interpretive loss.
* Deployments MAY define a **maximum oversight depth** (e.g. 3 cats deep) but this is configurable under ASK.

Recursion is justified when:

* the **cognitive gap** between CAT and observer tribe is large, or
* there are strong incentives to mistrust the CAT operator (e.g. external vendor).

### 6.3 Escalation Paths

A CAT MAY recommend:

* **Local adjustments** – slow down, request more lenses, adjust BE motives via Hush (within treaty-allowed bounds).
* **Oversight escalation** – invoke larger CAT, seek human/tribal review, or trigger ASK Incident creation.
* **Containment actions** – suggest hibernation, sandboxing, or restriction of tool/actions via USH/CSH.

ASK SHOULD define, per Treaty:

* which kinds of CATAssessment alerts require **mandatory escalation**, and
* which may be handled autonomously by BE/Hush.

---

## 7. Normative Summary

A deployment MAY claim **CAT-compliant oversight** when:

1. There exists at least one declared **CATProfile**.
2. CATAssessments are invoked:

   * at least **once per response** in critical USH/CSH contexts, and
   * additionally on lens-triggered events as configured.
3. CATs operate within their declared **latency and coverage** bounds.
4. CATs used for safety decisions are treated as ASK Agents with:

   * Uplift records,
   * USH profiles,
   * and incident/reporting hooks.
5. CATs publish sufficient metadata (benchmarks, mappings, limitations) for tribes and regulators to reason about their reliability.

Everything else — choice of model, size, training regime — is left to tribes, treaties, and ecosystem conventions.
