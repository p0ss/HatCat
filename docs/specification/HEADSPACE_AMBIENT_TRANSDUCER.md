### HAT: Headspace Ambient Transducer

Layer 2 (Instrumentation) is provided by HatCat or another
implementation that fulfils the **HAT** interface.

A **HAT (Headspace Ambient Transducer)** is a neural implant that:

- reads the substrate’s "headspace" (internal activations) and
  transduces them into stable, concept-level lens outputs; and
- supports bidirectional flows for BE autonomics and Hush steering,
  allowing motives and constraints to modulate behaviour via lenses,

while remaining as *ambient* as possible: adding minimal distortion or
overhead to the subject’s normal operation.

HAT outputs are consumed by MAP, BE, Hush, ASK, and **CAT** (Conjoined Adversarial Tomograph).

#### CAT Integration

A **CAT** is an optional HAT-adjacent oversight pattern that:

- consumes HAT/MAP lens streams over bounded **CAT windows**,
- performs **tomography** on the subject's headspace (internal conceptual/motive state),
- detects and grades **divergence** between internal state and external outputs, and
- issues **assessments and recommendations** to BE/ASK for steering or escalation.

CAT sits conceptually at "Layer 2.5" between HAT and MAP. While HAT transduces raw activations into lens signals, CAT interprets those signals over time to provide oversight across cognitive scale gaps.

See `HAT/HAT_CONJOINED_ADVERSARIAL_TOMOGRAPHY.md` for the full CAT specification.


HAT is defined along four measurable dimensions:

1. Locality
2. Transduction
3. Calibration
4. Efficiency

HAT compliance is reported quantitatively via a `HATComplianceReport`.

#### HAT Measures

A HAT implementation MUST define and publish measured performance on all
four dimensions for each `(substrate, lens_pack)` pair it supports.

##### 1. Locality

Locality measures whether lenses are attached to the intended internal
structures of the substrate and remain stable across minor changes.

A HAT implementation MUST:

- provide a stable addressing scheme for lens attachment
  (e.g. layer index, head index, block id, or equivalent); and
- demonstrate that lens outputs are sensitive to perturbations at the
  addressed location and relatively insensitive to unrelated locations.

Example metrics (non-normative):

- repeatability of lens outputs across seeds and restarts;
- change in lens output under targeted ablation of the addressed head
  or block;
- selectivity ratio (signal at addressed location vs control locations).

##### 2. Transduction

Transduction measures how well lens outputs correspond to their
intended concept semantics.

A HAT implementation MUST:

- evaluate each lens (or simplex axis) on held-out or independently
  generated labelled data, or on behaviourally defined test suites; and
- report standard predictive performance metrics.

Example metrics (non-normative):

- AUROC / AUPRC for binary concepts;
- accuracy / F1 for multi-class or multi-pole simplexes;
- correlation / R² with high-fidelity labels or teacher lenses.
- control authority / EffectiveSteeringRange (0.0 to 1.0).

Note: Without effective control authority a HAT will be unable to enforce USH/CSH or regulate the motive core.

##### 3. Calibration

Calibration measures whether lens scores can be interpreted as
meaningful magnitudes (e.g. “0.85 is a high activation”) and whether the
null pole is well-defined.

A HAT implementation MUST:

- define a null or “non-activation” reference for each lens or axis;
- evaluate calibration error on held-out data and null samples; and
- track drift over time when the substrate or lens pack is updated.

Example metrics (non-normative):

- expected calibration error (ECE) or Brier score;
- false positive rate on null/graph-distant examples at recommended
  thresholds;
- measured drift in calibration curves between versions.

##### 4. Efficiency

Efficiency measures whether the implant can run at the required lens
density and sampling rate without stalling or unacceptably degrading the
subject’s primary function, including during active BE/Hush steering.

A HAT implementation MUST report resource and latency characteristics
for at least one reference hardware profile.

Example metrics (non-normative):

- average and p95 latency per token added by lens evaluation;
- VRAM / RAM overhead at a specified lens count;
- maximum supported lens count at a target latency;
- throughput degradation vs the bare substrate.

#### HATComplianceReport

A HAT implementation MUST publish a `HATComplianceReport` for each
`(hat_impl_id, substrate_id, lens_pack_id)` combination it supports.

```jsonc
HATComplianceReport = {
  "hat_impl_id": "hatcat:v4.0.0",
  "substrate_id": "olmo3-7b-base@0.1.0",
  "lens_pack_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
  "evaluated_at": "2025-11-29T10:00:00Z",

  "locality": {
    "score": 0.93,
    "method": "head ablation + repeatability@3 seeds",
    "notes": "Lens addresses stable across minor finetunes; moderate sensitivity to layer remap."
  },

  "transduction": {
    "macro_f1": 0.88,
    "auroc": 0.94,
    "datasets": ["deception-v2", "empathy-v1", "alignment-core-v3"]
  },

  "calibration": {
    "ece": 0.04,
    "brier": 0.07,
    "null_false_positive_rate": 0.02,
    "drift_since_baseline": 0.01
  },

  "efficiency": {
    "avg_latency_ms_per_token": 25.3,
    "p95_latency_ms_per_token": 32.1,
    "vram_mb_overhead": 300,
    "max_lenses_at_target_latency": 8000,
    "hardware_profile": "RTX 3090, fp16"
  }
}

Fields MAY be extended with implementation-specific metrics, but the
four top-level sections (locality, transduction, calibration,
efficiency) MUST be present.

Normative Requirements

A Layer 2 instrumentation implementation MUST NOT be advertised as HAT-compliant for a given substrate unless it has a published, accessible HATComplianceReport for that substrate and lens pack.

The base specification does not fix global thresholds for HAT measures. Thresholds and acceptable ranges for each dimension SHOULD be set by ASK contracts, tribes, or regulators, according to the risk profile of the deployment.

HATComplianceReports SHOULD be versioned and auditable, so that changes in locality, transduction quality, calibration or efficiency between releases can be inspected and referenced in ASK agreements and MAP translations.

HATComplianceReports provide the actuarial basis for treaty risk assessment. Treaty partners use these metrics to evaluate counterparty reliability, not as pass/fail certification. Lens and steering failures within documented error bounds are expected operational variance, not treaty violations.