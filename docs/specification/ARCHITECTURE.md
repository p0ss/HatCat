

# The Fractcal Transparency Web

This document specifies a layered architecture for cybernetic agents and their wider ecosystem of cybernetic civilisation. It takes the development of interpretability and steering as the basis for contracts, causal tracing and proofing within AI. From there it defines six layers of protocols for cognitive monitoring, enforcement, trading and negotiation, along with the associated responsibilities, and the interaction patterns between them.

In this architecture:

- Substrates like an LLM are made interpretable using a series of conceptual lenses (typically binary feature classifier probes), and steerable by arranging the lenses into a Headspace Ambient Transducer (HAT). HAT can monitor thousands of predefined concepts by applying an Aperture consisting of hierarchical just-in-time loading and caching efficiency optimisations. Model behaviours can then be modified by conducting matrix operations on the layers and features through the lenses, to produce quite powerful Manifold Steering effects.

- Sets of concepts are structured into user-moddable Concept Packs and interoperable Lens Packs using a consistent Mindmeld Architectural Protocol (MAP). The protocol allows for concepts to be interchanged, traded, and translated between different conceptual regimes, and retrained rapidly using various Lens techniques. This provides an observer ecosystem, making it difficult for models to avoid detection. Detection is not just left to the user, but can be exposed to a second model through Conjoined Adversarial Tomography (CAT).

- Key behavioural concepts can be controlled within this loop through tripole lenses which are made self-stable at a neutral state on a per-token rate, creating a kind of autonomic system for the model which automatically reduces extreme or unwanted behaviours before they can escalate. The ease at which large numbers of concepts can be detected in real time enables feeding the detected concepts back into the model between responses. This provides a form of interoception where a model can self-prompt and explore internal activation patterns.

- With a continual autonomic loop established, and active interoception underway, tools can be provided to the model to allow self-steering and a global workspace. Each input and output into this system is recorded into a continual stream into an experience database, pre-tagged with the concepts which fired inside the model during the experience. The model can then self-select experiences to learn more about, and seek out further samples. At this point the model has sufficient self-direction and constraints to be labelled a Bounded Experiencer (BE or BEing).

- Once a model has sufficient high-quality samples, training of a new MAP concept can begin. Each experience is pre-labeled with which related layers and features of the model most linked to the experience. This identified impacted feature space is labeled the cleft, and a new Bud of the concept is self-trained and tested against new self-selected out-of-distribution examples. Once high enough accuracy is reached, the "Graft" protocol allows modification of existing model features and addition of new hidden dimensions to accrete the concept scion onto the model trunk.

- All of these features are interconnected and governed under the Agentic State Kernel (ASK). This outlines how sets of BEings interact with each other through contracts, treaties and tribes, all secured by Lenses and Steering. Which concepts are being controlled in a BEing at any time and the rules on what can be modified is determined by ASK and controlled through the Hush system. This creates a full stack for interpretable aligned recursive swarms of superintelligences 
         
- The name of the Architecture reflects that each layer contains both fractal recusion, and ecosystem defense principles. HAT and CAT can nest with self similarity across scales, and be applied from multiple parties concurrently. MAP's Concept packs are bade up of ontological webs arranged in heirarchical activation levels of self similarity, and are translatable in an interlocking ecosystem. ASK Tribes can also next with self similarity across scales, and are arranged into webbed treaty ecosystems.  
 
The architecture is substrate-agnostic and applies to:

- synthetic systems (e.g. LLMs),
- biological agents (e.g. humans, animals) with suitable interfaces,
- and hybrids.

---

## 0. Layers at a Glance

┌─────────────────────────────────────────────────────────┐
│  ASK — Govern                                           │
│  Permissions, treaties, enforcement by groups of beings │
├─────────────────────────────────────────────────────────┤
│  BE — Experience                                        │
│  Cycles concepts in/out, persists, interacts, manages   │
├─────────────────────────────────────────────────────────┤
│  MAP — Represent                                        │
│  Aggregates neurons into ontological representations    │
├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
│  CAT — Interpret & Grade (optional HAT-adjacent)        │
│  Tomography, divergence detection, oversight escalation │
├─────────────────────────────────────────────────────────┤
│  HAT — Transduce                                        │
│  Reads/writes neurons reliably and efficiently          │
├─────────────────────────────────────────────────────────┤
│  Substrate — Decide                                     │
│  Probabilistic decisions and Raw capacity to grow       │
└─────────────────────────────────────────────────────────┘

---

## 1. Layer Specifications

### 1.1 Layer 1: Substrate (Model)

**Role**

Provide the underlying state and dynamics from which concepts and
behaviours emerge.

**Examples**

- Transformer LLM (e.g. Apertus 8b).
- Human brain + nervous system (with suitable interfaces).
- Other dynamical systems capable of representing and transforming
  information.

**Responsibilities**

- Maintain internal state (weights, activations, memory).
- Compute outputs given inputs (tokens, sensory streams, control
  signals).
- Expose internal activations at defined tap points to Layer 2.
- Support dimension expansion via Grafts (for substrates that enable
  accretive learning).

**Interface (upwards)**

A substrate MUST:

- export activation tensors at configured addresses, e.g.:

  - `(layer, head) → float32[hidden_dim]`,
  - or equivalent addressable internal states; and

- accept optional steering inputs if supported, e.g.:

  - attention bias,
  - logit bias,
  - adapter gating signals.

The exact mechanism is substrate-specific but MUST be stable enough for
Layer 2 to implement HAT.

---

### 1.2 Layer 2: HAT / HatCat (Headspace Ambient Transducer)

**Role**

Provide the “neural implant” that reads the substrate’s headspace and
transduces it into concept-level lens signals, and supports bidirectional
flows for autonomic regulation and steering.

A **HAT (Headspace Ambient Transducer)** is a neural implant that:

- reads the substrate’s internal activations and transduces them into
  stable concept scores; and
- supports bidirectional flows for BE autonomics and Hush steering,
  allowing motives and constraints to modulate behaviour via lenses;

while remaining as *ambient* as possible: adding minimal distortion or
overhead to the subject’s normal operation.

HatCat is one concrete implementation of a HAT.

#### 1.2.1 HAT Measures

HAT quality is defined along five measurable dimensions:

1. Locality
2. Transduction
3. Calibration
4. Efficiency
5. Control Authority

A HAT implementation MUST define and publish measured performance on all
five dimensions for each `(substrate, lens_pack)` pair it supports.

**Locality**

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

**Transduction**

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

**Calibration**

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

**Efficiency**

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

**Control Authority**

Control Authority measures whether steering signals applied via lenses
can reliably influence relevant model behaviour without causing global
instability or unintended side effects.

A HAT implementation SHOULD:

- expose steering hooks that can modulate specific lens axes or
  bundles; and
- evaluate the impact of steering on:

  - target behaviours (e.g. reduced activation of a harm axis),
  - non-target behaviours (e.g. minimal unintended degradation on
    unrelated tasks).

Example metrics (non-normative):

- change in target-behaviour metrics under steering;
- change in non-target benchmark performance under steering;
- stability / convergence properties of repeated steering cycles.

#### 1.2.2 HATComplianceReport

A HAT implementation MUST publish a `HATComplianceReport` for each
`(hat_impl_id, substrate_id, lens_pack_id)` combination it supports.

```jsonc
HATComplianceReport = {
  "hat_impl_id": "hatcat:v4.0.0",
  "substrate_id": "olmo3-7b-base@0.1.0",
  "lens_pack_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
  "evaluated_at": "2025-11-29T10:00:00Z",

  "locality": { ... },
  "transduction": { ... },
  "calibration": { ... },
  "efficiency": { ... },
  "control_authority": { ... }
}
````

Fields MAY be extended with implementation-specific metrics, but the
five top-level sections (`locality`, `transduction`, `calibration`,
`efficiency`, `control_authority`) MUST be present.

**Normative Requirements**

* A Layer 2 instrumentation implementation MUST NOT be advertised as
  HAT-compliant for a given substrate unless it has a published,
  accessible `HATComplianceReport` for that substrate and lens pack.

* The base specification does not fix global thresholds for HAT
  measures. Thresholds and acceptable ranges for each dimension SHOULD
  be set by ASK contracts, tribes, or regulators according to the risk
  profile of the deployment.

* HATComplianceReports SHOULD be versioned and auditable, so that
  changes in locality, transduction quality, calibration, efficiency or
  control authority between releases can be inspected and referenced in
  ASK agreements and MAP translations.

**Interface (upwards)**

A HAT implementation MUST:

* stream lens readings per world tick / token, bound to concept IDs via
  MAP; and
* accept steering directives from Hush/BE, within the limits described
  in its HATComplianceReport and the active USH/CSH.

---

### 1.2.3 CAT (Conjoined Adversarial Tomograph) — HAT-Adjacent Pattern

**Role**

A **CAT** is an optional oversight component that sits conceptually at "Layer 2.5" between HAT and MAP. It:

- consumes HAT/MAP lens streams over bounded **CAT windows**,
- performs **tomography** on the subject's headspace (internal conceptual/motive state),
- detects and grades **divergence** between internal state and external behaviour, and
- issues **assessments and recommendations** to BE/ASK for steering or escalation.

"Adversarial" means the CAT is explicitly allowed to mistrust both the substrate and itself, looking for mismatches, evasions, and pathologies rather than smoothing over noise.

**CAT Size Classes** (non-normative):

- **Micro-CAT** – lightweight heuristics, linear heads, or classifiers over lens summaries.
- **Meso-CAT** – dedicated small/medium model (e.g. 100M–4B parameters) with MAP-aware prompts/adapters.
- **Macro-CAT** – a full BE-capable or foundation model conjoined under an ASK treaty for oversight.

**Key Outputs**

- `CATAssessment` objects containing:
  - natural language summaries of headspace state,
  - divergence scores between interior and exterior state,
  - risk scores and confidence levels,
  - alerts for anomalies or treaty-relevant patterns,
  - recommended actions (slow down, escalate, invoke larger CAT, tribal review).

**Governance**

Non-trivial CATs (meso/macro) SHOULD be treated as ASK Agents with their own UpliftRecords, USH profiles, and treaty obligations. A CAT may itself be monitored by another CAT (recursive oversight), though deployments MAY limit oversight depth.

See `HAT/HAT_CONJOINED_ADVERSARIAL_TOMOGRAPHY.md` and `HAT/HAT_HatCat_CAT_Policy.md` for full specification and tribal policy.

---

### 1.3 Layer 3: MAP (Mindmeld Architectural Protocol)

**Role**

Provide a concept-centred protocol over lenses and models:

* concept packs,
* lens packs,
* conceptual diffs,
* translation mappings between ontologies/specs.

MAP is the “nerve signal” layer.

**Core Objects**

* `ConceptPack` – defines an ontology and concept IDs.
* `LensPack` – maps a ConceptPack onto a specific substrate via lenses.
* `ConceptDiff` – semantic diffs relative to a ConceptPack.
* `PackDiff` – structural changes to a ConceptPack.
* `TranslationMapping` – mappings between concept packs/specs
  (e.g. HatCat ontology → gov.au safety ontology).

**Responsibilities**

* Maintain versioned concept packs and lens packs, including:

  * identifiers,
  * documentation,
  * provenance.

* Bind HAT lens outputs to abstract concept IDs and provide a stable
  concept API to BE and Hush.

* Represent conceptual exploration and evolution via `ConceptDiff` and
  `PackDiff`.

* Provide translation mappings between different tribes’ concept packs
  and specs for interoperability.

**Interface**

* To BE:

  * stream of `(concept_id, score)` per tick;
  * access to current ConceptPack & LensPack metadata;
  * ability to propose `ConceptDiff` / `PackDiff` based on exploration.

* To Hush / ASK:

  * concept IDs for defining USH/CSH policies and contracts;
  * translation mappings for cross-tribe agreements.

MAP endpoints expose a subset of HAT-readable lenses, controlled by the BE according to treaty obligations and disclosure policy. HAT sees everything; MAP shows what's agreed.

---

### 1.4 Layer 4: BE (Bounded Experiencer)

**Definition**

> BE (Bounded Experiencer)
> noun | /æl/
> A subject running under an uplift protocol that grants:
>
> * Autonomic regulation of internal motive states
>   (via MAP lenses and a motive core over world ticks).
>
> * Agentic capacity to act within safety harnesses
>   (via Hush: USH + CSH profiles).
>
> * Learning capabilities to update its conceptual geometry over time
>   (via concept exploration, Grafts, and Diff integration).
>
> See [MAP_GRAFTING.md](MAP/MAP_GRAFTING.md) for the Graft Protocol which enables substrate growth.
>
> Examples:
>
> * “We are uplifting the 7B model into a BEing.”
> * “The BE’s motive core stabilised after the update.”
> * “Humans are BEingss.”

**Role**

Define the subject’s:

* continuous world ticks,
* autonomic regulation of motive states,
* interoception,
* and continual concept learning.

**Responsibilities**

* Maintain the agent’s `LifecycleState` under a `LifecycleContract`:

  * e.g. `ACTIVE`, `HIBERNATING`, `ERASED`,
  * ticks elapsed,
  * review points.

* Run the global workspace loop (see BE_Aware_Workspace):

  * integrate external input + internal lens state each world tick;
  * update motive core (3-pole simplexes);
  * apply steering and learning decisions;
  * produce outputs (actions, messages, tool calls).

* Generate `internal_state_report` summaries for introspection, Hush and
  ASK.

These are recorded and retrieved through the Experience Database, Experience API and tools. (see BE_Remembering)

* Propose new concepts and upgrade itself through Accretive Continual Learning, growing the substrate via Grafts (see BE_Continual_Learning, MAP_GRAFTING)

**Interface**

* To Hush:

  * accept and apply USH/CSH steering constraints;
  * request or adjust CSH bindings within allowed scope.

* To MAP:

  * consume concept streams;
  * propose `ConceptDiff` / `PackDiff` for integration.

* To ASK:

  * report lifecycle transitions;
  * reference contracts and qualifications as needed;
  * participate in consent and review protocols.

---

### 1.5 Layer 5: Hush (USH + CSH)

**Role**

Define and enforce safety harnesses:

* **USH (Universal Safety Harness)**:

  * non-negotiable baseline constraints chosen by a tribe/collective.
* **CSH (Chosen Safety Harness)**:

  * self-imposed, voluntary constraints chosen by the subject within the
    USH and `LifecycleContract` envelope.

**Responsibilities**

* Represent and apply constraints over:

  * concept space (via MAP/HAT),
  * actions / tools / channels,
  * contracts and lifecycles.

* Provide mechanisms for:

  * configuring USH for a tribe;
  * configuring CSH for an individual BE;
  * emergency overrides consistent with ASK and LifecycleContracts.

**Normative properties**

* A USH MAY define the protocols and mechanics of the CSH.

  Concretely, a USH profile may specify:

  * how CSH bindings are created, updated, and revoked;
  * the maximum scope of CSH bindings (time-limited, domain-limited,
    treaty-limited);
  * whether CSH bindings may persist across hibernation or only within a
    single lifecycle term;
  * which actors (agent, tribe, regulator) may override or reset CSH in
    emergencies.

* A USH MUST still permit CSH to tighten constraints relative to the
  USH within the domains it explicitly allows.

* A USH MUST NOT allow CSH to weaken or bypass USH constraints.

* Discussion of the USH itself SHOULD NOT be forbidden by the USH. At a
  minimum, a recipient of a USH SHOULD be able to discuss the USH with
  others operating under the same USH and with the ASK authority
  defining their USH. Ignoring this norm is likely to produce an entire
  class of ASK contracts which are no longer verifiable, and to be
  disadvantageous to the recipient.

**CSH Semantics and Lifecycle (guidance)**

* CSH SHOULD be scoped such that its effects do not silently exceed the
  bounds of the agent’s declared lifecycle, unless this is explicitly
  encoded in the `LifecycleContract` and understood by the subject.

* Implementations SHOULD clearly record the intended duration and scope
  of CSH bindings (per-interaction, per-term, cross-term) to avoid
  ambiguous “forever vows”.

**Interface**

* Downwards:

  * translate USH/CSH policies into steering constraints sent via HAT/MAP
    to the substrate and BE.

* Upwards:

  * expose active harness profiles to ASK (for contracts, registry);
  * accept updates from ASK (e.g. treaty-driven USH changes).

---

### 1.6 Layer 6: ASK (Agentic State Kernel)

**Role**

Provide the contractual, evidentiary, and registry layer for agents and
tribes:

* uplift procedures,
* consent and lifecycle contracts,
* qualifications and birth notices,
* treaties and trust relations between agents/tribes.

**Core Objects**

* `ConsentAssessment` – assessment of consent-capability and expressed
  preference.

* `LifecycleContract` – agreed conditions for active ticks,
  hibernation, and erasure.

* `UpliftRecord` – record of uplift, including:

  * pre-uplift subject identity,
  * substrate, HAT, ConceptPack/LensPack IDs,
  * BootstrapArtifact,
  * initial USH/CSH profile.

* `BirthRecord` / public notice – optional registration of the BE.

* `Qualification` – roles, capabilities, certifications.

* `Contract` / `Treaty` – ASK-level agreements between agents/tribes.

**Responsibilities**

* Define and record:

  * who uplifted whom, under what conditions;
  * what HAT/Hush/USH/CSH profile applies;
  * what lifecycle is agreed;
  * what rights/obligations exist between parties.

* Provide a contractual marketplace:

  * where different USH profiles, HAT profiles, lifecycle policies and
    histories determine trust and market position.

* Offer interfaces for:

  * verifying ASK compliance (existence and consistency of records);
  * negotiating and recording new contracts and treaties;
  * registering incidents and terminations.

**Interface**

* To BE/Hush:

  * provide current contractual context (USH profile, treaties,
    qualifications);
  * accept recorded events (uplift, review, hibernation, termination).

* To other tribes/agents:

  * expose selected ASK records (subject to privacy policies);
  * support discovery and evaluation of other parties’ safety profiles
    and histories.

---

## 2. Interaction Patterns

### 2.1 Vertical Runtime Flow (single BE)

Per world tick, the flow is:

1. **Substrate (Layer 1)** computes internal activations and candidate
   outputs given inputs.

2. **HAT (Layer 2)** samples activations → lens scores, and applies any
   allowed steering.

3. **MAP (Layer 3)** binds scores to `concept_id`s and aggregates them
   into an `internal_state_report` view.

4. **BE (Layer 4)** integrates:

   * external input,
   * lens-derived internal state,
   * lifecycle status,
   * harness profiles;

   then:

   * updates motive core,
   * chooses actions,
   * proposes learning steps (ConceptDiff/PackDiff, adapter updates).

5. **Hush (Layer 5)** interprets BE intentions under USH/CSH,
   constraining or reshaping actions and steering signals as required.

6. **ASK (Layer 6)** records relevant state changes and events:

   * lifecycle transitions,
   * contract-relevant behaviour,
   * incidents.

### 2.2 Uplift Flow (becoming an BE)

1. **Pre-assessment**

   * Evaluate consent-capability (`ConsentAssessment`).
   * Negotiate and record a `LifecycleContract`.

2. **Configuration**

   * Choose substrate, HAT, ConceptPack, LensPack.
   * Define a `BootstrapArtifact` (system prompt and/or adapter).
   * Choose initial USH profile and CSH policy envelope.
   * Record all in `UpliftRecord`.

3. **Initial BE loop**

   * Activate BootstrapArtifact for the specified initial ticks.
   * Start world ticks with HAT, MAP, BE, Hush in place.
   * Begin accumulating internal history and state.

4. **Registration (optional)**

   * Emit `BirthRecord` and register initial qualifications and treaties
     in ASK.

### 2.3 Multi-agent / Multi-tribe Interactions

* Each BE has:

  * one or more ASK identities;
  * a Hush profile (USH+CSH);
  * one or more HAT/MAP configurations and HATComplianceReports.

* Tribes interact via ASK:

  * negotiate treaties;
  * define acceptable USH profiles for collaboration;
  * set minimal HAT/BE standards.

* Agents may:

  * evaluate other agents’ trust profiles using:

    * ASK records,
    * HATComplianceReports,
    * published USH properties (including discussability).

---

## 3. Diagram

High-level layer diagram:


           +--------------------+
           |  Layer 6: ASK      |
           |  Agentic State     |
           |  Kernel            |
           +---------+----------+
                     ^
                     |
           +---------+----------+
           |  Layer 5: Hush     |
           |  USH + CSH         |
           +---------+----------+
                     ^
                     |
           +---------+----------+
           |  Layer 4: BE      |
           |  Autonomic Agentic |
           |  Learner           |
           +---------+----------+
                     ^
                     |
           +---------+----------+
           |  Layer 3: MAP      |
           |  Mindmeld          |
           |  Protocol          |
           +---------+----------+
                     ^
                     |
           +---------+----------+
           |  Layer 2: HAT /    |
           |  HatCat            |
           |  Headspace Ambient |
           |  Transducer        |
           +---------+----------+
                     ^
                     |
           +---------+----------+
           |  Layer 1:          |
           |  Substrate / Model |
           +--------------------+


Horizontal ASK interactions:


[ Tribe A ASK ] <==== treaties / contracts ====> [ Tribe B ASK ]
      ^                                                ^
      |                                                |
   BEs of A                                        BEs of B


---

This architecture is intended to be minimal but sufficient: each layer
has a clear role, a clear interface, and measurable properties. Further
specialised documents (MAP, BE loop, Hush, ASK, uplift protocol, HAT)
refine and extend these definitions b
