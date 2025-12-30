
### Title

**Fractal Transparency for the Web (FTW): A Safety Standard for Large Public AI Deployments**

---

### 1. Purpose

This brief proposes adopting **FTW** as a **technical safety standard** for AI systems used in large public-facing deployments (e.g. health, welfare, education, justice, critical infrastructure). FTW provides an *interpretability-first* architecture: it makes powerful AI systems **continuously inspectable and governable**, not just “aligned by training”.

---

### 2. What FTW Is (in plain terms)

FTW adds a lightweight, standardized “safety co-processor” around AI models:

* **Concept lenses (probes)**
  Small classifiers attached to a model’s internal activations that detect concepts such as:

  * deception, coercion, self-modification, law-breaking, child harm, etc.

* **Performance**
  Through caching and hierarchical loading, **~7,000 lenses** can run:

  * using **under 1GB VRAM**
  * with **~25ms overhead** per step

  This makes **always-on monitoring and steering** technically and economically practical.

* **Stack components** (simplified):

  * **HAT** – the always-on monitoring and steering layer.
  * **MAP** – standardised registries/ontologies of concepts and lenses.
  * **BEs (Bounded Experiencers)** – agentic systems with built-in self-monitoring.
  * **HUSH** – safety harnesses and constraints the system must respect.
  * **ASK** – a governance layer for contracts/treaties and access control over probes.

---

### 3. Why this matters for government use

1. **Continuous visibility into model behaviour**
   FTW shifts us from “we hope the model behaves” to:

   * “we can see what it was *internally* doing when it made a decision.”

2. **Defense via ecosystem, not one ‘aligned’ model**

   * Different agencies, vendors and auditors can maintain **diverse lens packs** and ontologies.
   * Systems interacting in the economy must accept **contractually enforced probes** (via ASK) as a condition of use, trade or integration.
   * This diversity of probes and perspectives creates **“herd immunity”**: misbehaviour must evade many independent detectors, not just one.

3. **Standardisation without monoculture**

   * Governments define **minimum required safety lenses and treaties** for certain use-cases.
   * Vendors are free to add additional lenses/ontologies; regulators and civil society can provide independent packs for oversight.

---

### 4. Core elements of an FTW-based standard

For **“large public deployments”**, the standard would require that systems:

1. **Run an approved HAT layer**

   * A minimum “safety spine” of lenses must be active at all times (e.g. harm, deception, self-replication, legal violations, child-related risk).

2. **Support MAP-compatible concept packs**

   * Use open, documented interfaces so multiple independent lens packs (government, vendors, third-party auditors) can be loaded and combined.

3. **Implement ASK-based contracts & treaties**

   * Participation in designated high-risk domains (health, justice, etc.) requires accepting **probe access and logging obligations** defined in ASK contracts.

4. **Log and expose relevant safety signals**

   * Record significant probe activations and constraint violations.
   * Provide regulators and authorised auditors with access to aggregated, privacy-respecting reports and samples.

5. **Support cross-model oversight (“watchers watching watchers”)**

   * Critical deployments must be structured so that models can be **cross-checked by other FTW-compliant systems** (e.g. independent BE overseers with their own lenses).

---

### 5. Safeguards: rights, competition, and misuse

Any government FTW standard should explicitly:

* **Protect civil liberties & privacy**

  * Probes operate on model internals, not for blanket surveillance of individuals.
  * Prohibit probes designed for political or ideological profiling.
  * Require strict rules on data retention, access, and redress.

* **Prevent vendor lock-in**

  * Require **open interfaces** for lens packs, ontologies and contracts.
  * For high-risk deployments, mandate **multi-vendor lens diversity** (e.g. at least two independent safety packs for core concepts).

* **Enable independent oversight**

  * Allow certified third parties (academia, NGOs, auditors) to publish and maintain lens packs registrable in MAP and usable in public deployments.

---

### 6. Suggested next steps

1. **Pilot** FTW in 1–2 domains (e.g. welfare decision support, education chatbots).
2. **Convene a standards group** (government, vendors, civil society, researchers) to refine:

   * required safety spine concepts,
   * audit/attestation procedures,
   * privacy/civil liberties guarantees.
3. **Integrate FTW compliance** into procurement frameworks and future AI-specific regulation for large public deployments.

---
