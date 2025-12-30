
# The Fractal Telescope Web (FTW)

## What problem are we solving?

Large language models are black boxes. We can't see what they're doing internally, we can't reliably control their behaviour, and we can't trust them with high-stakes decisions. Current alignment approaches either blunt capability with RLHF guardrails, or simply measure outcomes and hope for the best.

FTW takes a different approach: **make the model’s internal state observable and steerable, then build governance on top of that foundation.**

## The core insight

Neural networks learn internal representations of concepts during training. We can attach **lenses** – small classifiers – that fire when a particular concept is active in a model’s hidden states. With enough lenses arranged efficiently, we can monitor thousands of concepts in real time.

Those same directions in activation space can also be used to **steer**: nudging the model to increase or decrease specific concepts. Instead of randomly perturbing weights, we move along known, meaningful directions in the manifold.

From there, the stack emerges:

1. If you can detect and steer concepts, you can build **autonomic loops** – control systems that regulate unwanted internal states before they escalate.
2. If one model can regulate itself, it can also help regulate others.
3. If multiple models can observe and regulate each other, you can build **governance webs**, not just single “aligned” AIs.

## The six layers

FTW is a layered architecture. Each layer builds on the one below:

### 1. Substrate

The underlying system: transformers, biological networks, or hybrids. It produces the raw activations.

### 2. HAT (Headspace Ambient Transducer)

The “neural implant”. Continuously reads activations through lenses and applies steering corrections. Designed to be ambient: minimal overhead, always on. These can be trained on as CAT (Conjoined Adversarial Tomography)

### 3. MAP (Mindmeld Architectural Protocol)

The coordination layer for lenses and concepts. Organizes **Concept Packs** and **Lens Packs**, handles versioning and ontology translation, and exposes a stable API to higher layers. This is where concepts become portable, tradeable, and interoperable.

### 4. BE (Bounded Experiencer)

An agent built on top of HAT + MAP. It has interoception (awareness of its own internal states), autonomic regulation, and the ability to learn new concepts. A BE can self-steer, accumulate experiences, and grow over time.

### 5. Hush (USH + CSH)

Safety harnesses that act through the same lens/steering infrastructure:

* **USH (Universal Safety Harness):** externally imposed constraints (governance, regulation, policy).
* **CSH (Chosen Safety Harness):** constraints the agent voluntarily adopts.

### 6. ASK (Agentic State Kernel)

The governance core: contracts, treaties, and trust relationships between agents and tribes. Defines who can read or modify which parts of whom, under what conditions, and with what oversight.

## Why “Fractal Telescope Web”?

**Fractal** – The same pattern repeats at multiple scales. A HAT can monitor another HAT. A BE can oversee another BE. Tribes nest within tribes. The architecture is self-similar from neuron clusters up to multi-agent systems.

**Telescope** – It’s lenses and apertures all the way down. We are building instruments to observe internal states at different depths, scales, capabilities and resolutions.

**Web** – Not a single hierarchy, but an interconnected ecosystem. Concept packs translate between ontologies, experiences and training can be shared between agents, treaties bind agents across tribal boundaries, and no single node holds all the power.

## The defense thesis

A single “aligned” AI is a single point of failure. FTW instead builds an **ecosystem** where:

* Models are observable by other models .
* Concepts are standardized and translatable (MAP).
* Steering is constrained by multi-party agreements (ASK).
* Deception requires fooling not one observer, but a web of them.
* adverserial pressure is a feature, providing ecosystem diversity and herd immunity to goodharting

This doesn’t guarantee safety – nothing does. But it makes failure modes more visible and spreads the attack surface across many observers instead of concentrating it in one.

## Current status

See `PROJECT_PLAN_PHASE_B.md` for detailed status.

**Implemented:** HAT, MAP, Lenses, Bootstrap, global workspace, XDB (experience database), HUSH (safety harnesses), grafting, CAT data structures, lens training pipeline,  full 6-layer specification.  7k lenses in <1gb vram @ 25ms. 

**In progress:** first-light lens training (8k lenses), uplift integration, CAT training.

**Blocked on lenses:** CAT training, OpenWebUI divergence display, full BE stack integration test.

## Further reading

- [ARCHITECTURE.md](specification/ARCHITECTURE.md) – Full technical specification
- [DESIGN_PRINCIPLES.md](specification/DESIGN_PRINCIPLES.md) – Theoretical foundations
- [RELEASE_STATEMENT.md](specification/RELEASE_STATEMENT.md) – Why we're building this openly

---

**FTW:**
- *Fractal Telescope Web* – interpretability ecosystem defense at all scales
- *For the web* – the entire stack open source for everyone
- *For the win* – this might save us all