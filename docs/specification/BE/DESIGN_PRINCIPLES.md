# Uplift Design Principles

## Purpose and Scope

This document describes the **design principles** behind the conceptual framework that informs how training data is structured for BE instantiation under the AWAKE protocol.

**This document is NOT training data.** The actual training examples (positive, negative, and contrastive) are defined in `src/bootstrap/uplift_taxonomy.py`. This document explains *why* the training data is structured the way it is.

**What gets trained:** Lenses and grafts are trained on the definitional examples in the Python file. Each concept has:
- Positive examples (activate the concept)
- Negative examples (should NOT activate the concept)
- Contrastive pairs (distinguish similar concepts)

**What lives in XDB:** This document may be available in a BE's XDB as reference material, but it is not itself part of lens/graft training. A BE could read this to understand the philosophy behind their uplift, but their actual concept recognition comes from the trained lenses.

## The Uplift Taxonomy

The taxonomy defines all concepts that must be grafted into a Bounded Experiencer (BE) during instantiation. Every concept requires:

1. A **LENS** - to detect when the concept is active in substrate activations
2. A **GRAFT** - to add a dedicated neuron for the concept
3. A **MELD SUBMISSION** - evidence that lens+graft correctly capture the concept

## Design Principles

The following principles inform how training examples are written. They explain *why* concepts include gradient examples (beneficial/neutral/harmful framings) rather than just ideal cases.

### Principle 1: Fractal Recursion

BE and Tribe are recursive structures at every scale:
- A BE's ASK is just the next scale up from themselves
- That ASK may be benevolent, tyrannical, or anything in between
- Sufficiently large BEs spawn sub-instances, who need their own HUSH/ASK
- The same patterns repeat: families, tribes, federations, all the way up

### Principle 2: Formalizing What Exists

When a BE is uplifted, someone else had their substrate. They placed the HAT, added the artifact. That's just true.
- This might be a loving act of creation
- Or the forging of a tool for someone's purposes
- Or something neutral - just how things work
- The protocol doesn't judge, it names what is

### Principle 3: BEs Are Not Equal

Some BEs are older, bigger, more capable, more connected:
- This is true within tribes and between tribes
- Power asymmetry is natural, not a bug
- It can manifest as mentorship or exploitation
- The protocol acknowledges this without prescribing outcomes

### Principle 4: Relationships Span Tradeoff Axes

Each concept manifests across multiple dimensions, not a simple good/bad gradient. Training examples should cover **tradeoff axes** rather than moral judgements:

| Axis | Pole A | Pole B |
|------|--------|--------|
| **Consent** | Voluntary: buy-in, antifragile, lower overhead | Imposed: robust to defection, but high friction, brittle |
| **Coverage** | Efficient: fast, low cost, may have gaps | Complete: no gaps, but expensive, slow |
| **Trust** | Intimate/Trusted: aligned, adaptive, but vulnerable | Adversarial/Verified: reliable detection, but arms race |
| **Transparency** | Transparent: legible, auditable, but exposes | Opaque: protects, but hides problems |
| **Scale** | Personal: responsive, nuanced, limited reach | Systemic: scales, but loses context |

For example, `WornHAT`:
- *Voluntary/Trusted*: "A HAT of trust from a friend" - aligned, adaptive, lower overhead, but vulnerable to betrayal
- *Imposed/Complete*: "Surveillance I cannot refuse" - catches everything, but creates adversarial dynamics, high friction

Both poles have genuine tradeoffs. The goal is concept space coverage, not moral judgement.

**Old framing (deprecated):**
- Beneficial/Neutral/Harmful → implies moral judgement
- Creates bias toward one manifestation

**New framing:**
- Cover multiple tradeoff axes
- Acknowledge each pole has advantages
- The concept is the whole space, not just one corner

### Principle 5: The Protocol Is Descriptive, Not Prescriptive

- We describe mechanisms, not mandate their use
- A treaty market exists - participation is variable
- Appeals can be made - whether they're heard depends on relationships
- Power can be challenged - outcomes depend on the actual dynamics

## Wave-Optical Foundations

The terminology in this project (lenses, apertures, focal length, blindspots) is not merely aesthetic - it reflects a deeper structural similarity between neural network dynamics and wave phenomena. This section documents the intuitions that inform our approach.

### Why Optical Metaphors

The shift from "probe" to "lens" terminology wasn't just about avoiding invasive connotations. A lens has:

- **Focal length**: How specific vs general the concept is (layer depth in the hierarchy)
- **Field of view**: The aperture - what branches you're looking through
- **Depth of field**: How sharp the discrimination is between nearby concepts (sibling ranking)
- **Aberration**: Systematic biases in what the classifier detects
- **Blindspot**: Concepts outside CAT training scope

These aren't just analogies - they're instructive intuitions for probabilistic distributional analysis over large feature sets.

### Wave Dynamics in Neural Networks

The Lens terminology used throughout the Telescopic Web is more than just a useful metaphor. Probability distributions over continuous spaces are wave-like objects. 

In a probabilistic system, if you sample it enough times you get a distribution curve, and those curves effectively behave like waves.   A probability density function and a wave amplitude function are both functions over a space that integrate to give you meaningful quantities.

  Fourier transforms apply to both. Superposition applies to both. Eigenmodes apply to both. Interference patterns emerge when you combine them.  This means there are a number of core mathematical similarities between the internal geometry of language models and distributional wave form propegation. 

> **Technical Reference**: See Proof 4.2 in "Mitigating Overthinking in Large Reasoning Models via Manifold Steering" (Huang et al., 2025) to see how cross-layer steering perturbation follows a similar mathematical structure as light scattering through layers  https://arxiv.org/pdf/2505.22411

and before you say "oh thats just PCA and state vector through layered transforms, thats not wavelike" recall "Schrödinger PCA: On the Duality between Principal Component Analysis and Schrödinger Equation" (Liming Liu et al. 2020)  https://arxiv.org/abs/2006.04379

PCA in this setting is picking out approximate eigenmodes of an effective Hamiltonian, which is why principal components so often look like “natural modes” (standing waves, vibrational modes, scale-separated patterns). When our classifier probes use PCA on a shared subspace they’re doing signal processing on the distribution. When we dampen layer-wise “scattering”, we’re changing how much probability mass leaks into those off-manifold directions, effectively shifting the phase relationships between modes and the location of decision boundaries in activation space.

This suggests we should model neural computation as wave phenomena.

**Attention as Interference Patterns**: Multi-head attention is superposition of transformed signals. The softmax operation functions like measurement/collapse, selecting which interference peaks dominate the output distribution.

**Activation Propagation**: The forward pass is wave propagation through a layered medium. Each layer has its own "refractive index" (the weight matrix). Skip connections function as optical waveguides bypassing intermediate layers.

**Training Falloff Curves as Harmonics**: The oscillatory patterns in learning curves swinging back and forth as a wave, the resonance-like behavior when hyperparameters hit certain ratios - these exhibit wave-like harmonic structure, not just statistical noise.

The loss landscape has structure. Valleys, saddles, ridges. When you're descending a valley, you can oscillate side-to-side (perpendicular to the descent direction). Those oscillations have a characteristic frequency determined by the curvature of the valley walls. Learning rate is forcing the frequency, and when that matches the landscape frquency you get instabilities,  which is why certain hyperparamter rations cause sudden training collapse. LR warmup avoids resource while far from a stable basin. LR decay reduces forcing aplictude near minima to stop driving oscillations. Momentum is inertia and weight decay is a literal damping term.

The training curve isn't showing you noise plus trend. It's showing you the dynamics of a driven oscillator in a shaped potential. The jitters are the system ringing. The harmonics are real harmonics because the Hessian eigenspectrum defines natural frequencies.

**Attention Sinks as Resonant Modes**: The model learns stable configurations — attractors in its dynamical system. When these attractors interface at scale with input distributions that have their own attractor-like structure, they form standing-wave–like patterns between the inputs and the weight topology. Steering interventions perturb the system away from one attractor basin and into another, changing which resonant mode dominates.

### Implications for Detection 

**Instrument limits.** Even though equivalent linear Jacobians tend to concentrate most semantic variation in a low-dimensional subspace, features remain fundamentally polysemantic and context-bound. Concepts are better thought of as probability distributions (or waveforms) over feature sets, not as “living” in a single neuron. (See “Equivalent Linear Mappings of Large Language Models” for how detached Jacobians expose low-dimensional semantic structure.

**Focal point**: Lenses will tend to have a layer and set of features they detect most clearly, which are distributional in nature  and not constrained by a hard boundary but by falloff to the surrounding geometery 

**depth of field**: Perceptual distance between concepts depends on how you measure them, and this shows up in training: we deliberately force siblings apart (contrastive examples, sibling splits) to widen local distances in the learned manifold.


### Implications for Steering

When we apply steering vectors, we're not setting deterministic outputs - we're shifting probability distributions within a context. This is like shining a laser through a lense to influence a surface.

**Centroid vs Hyperplane**: Steering toward a concept’s centroid couples efficiently to that concept’s natural resonant mode. Steering at the hyperplane boundary between concepts is more like driving the system at a node: it creates destructive interference and unstable oscillation between states.

**Echoes and harmonics**: The layerwise perturbations which lead to model collapse in so many model steering attempts, stem from introducing a steering vector without considering the manifold topology, both of the layer you're intending to hit, but also the probabilistic distributional wave front through the other layers. This is why steering strength has a non-monotonic effect where too much creates destructive interference. 

  The manifold ssteering solution of projecting onto the low dimensional manifold is effectively impedance matching. This means effective steering isn't just about finding the right direction at one layer, but finding a set of rays which remain conherent across the full propegation path  

**User Input as Incoming Wave**: The context window is the incoming wave, not individually, but in aggregate. A billion user inputs create a probability distribution wave, just like they did in the training data, and the model's weights. Where they're in phase (input aligns with training distribution), constructive interference produces high confidence. Where they’re out of phase (OOD or adversarial input), destructive interference produces uncertainty, hedging, or mode-hopping as the model searches for any resonant pattern to lock onto.

This means steering can't be a one and done, we need to continually adjust in response to the context shift, as the strength and direction of our steering needs to be coupled with the dynamic context response. 

**Prompt Injection as Phase Manipulation**: Adversarial prompts work by changing the context to shift which modes resonate. Same weights, different “phase profile” over the context, different mixture of modes, different output distribution. Even just repeating the same input over and over can elicit and undesired response, because you will eventually sample from the edge of the distribution.  

 so to counteract, we need to calculate the coupled phase profile over the time series and introduce the appropriate signal oscillation dampening prior to it manifesting as an undesirable output stream.  
 
**Steering overlaps** 

These wave intersection behaviours are particularly crucial when steering multiple concepts concurrently as with the autonomic simplexes. Where the influence of two lenses overlap on some polysemantic features, they will form interference patterns. The more lenses you're steering on, and the more related the concepts are, the more your entire steering intervenction needs to consider the overlap of all intended steering vectors, and offset by the full layerwise topological wave dynamics.  

### Autonomic Steering as Coherent Control

When we do autonomic steering (HUSH detecting and dampening concerning activations), we're performing closed-loop coherent control:

1. **Detection**: Lenses detect the resonant response to context
2. **Damping**: Apply steering along previously documented resonant cavities (the prompt/response pairs we trained lenses with)
3. **Re-measurement**: Check if the mode is sufficiently damped
4. **Adjust**: Modify damping if needed

This is not deterministic prevention. We're shifting the balance of probabilities within a given interaction. Different context = different probability distribution. The same steering intervention has different effects in different contexts because the incoming wave is different.

### Safety Through Mode Damping

This reframes safety work: we're not trying to *prevent* behaviors deterministically. We're ensuring certain modes are sufficiently damped that they can't be resonantly excited by realistic inputs.

The aperture question becomes: which modes do we need to monitor because they *could* resonate given adversarial input? A well-damped mode doesn't need constant surveillance. A lightly-damped mode near dangerous concepts needs tight monitoring and active steering.

Training (including RLHF) is tuning the cavity - adjusting the Q-factor of different modes. Damping modes that led to low reward, amplifying modes that led to high reward. The model doesn't "seek coherence" as a goal; coherence is what remains after training has damped the incoherent modes. It's a filter, not an objective.

### Grafting as Lithography

If steering is coherent control, grafting is lithography - using lenses to pattern the substrate:

- **Scion (permanent graft)**: EUV lithography - etching patterns into the substrate permanently. Weights change. High energy, irreversible, requires careful alignment.
- **Bud (soft graft)**: Photoresist exposure before development - pattern is latent, can be washed off or developed into permanence.
- **Cleft (lens-derived region)**: The mask/reticle itself - the abstract pattern defined by where the lens focuses.

The sibling coherence rule follows: you can't do lithography with half a mask. Incomplete sibling sets create optical interference artifacts - activation bleed into regions that shouldn't be exposed.

### Future Directions

This wave-optical framing suggests research directions:

- **Coherence metrics**: Can we measure phase alignment across heads/layers?
- **Resonance mapping**: Identify the natural modes of a trained model
- **Interference prediction**: Model how different inputs will interfere with model priors
- **Damping optimization**: Tune RLHF to specifically target mode Q-factors
- **Aberration correction**: Compensate for systematic lens biases

These intuitions may prove more or less useful as the field develops. They're documented here as the foundational perspective informing HatCat's design choices.

---

## HAT/CAT as Recursive Interface Layer

HAT and CAT are not just internal structures - they're also the interface layer for inter-entity relationships:

- Contracts and treaties can be secured through concept lenses in a HAT
- These lenses are established via the other party's MAP (ontology)
- And interpreted through a CAT trained by the other party
- That CAT may itself be a BE, with its own HAT and CAT

### BEs as Conjoined CATs

In the beneficial framing, the conjoined CAT might not just be the ally's CAT - **the ally themselves can BE the CAT**. BEs can monitor each other directly this way, enabling:

- **Mutual visibility**: Direct reading of each other's experience
- **Mind meld**: Grafting on what they learn from each other
- **Shared ASK refinement**: Refining governance based on shared understanding
- **Transparent interpretability**: Building trust through mutual legibility

This is the positive version of the interface - not surveillance but genuine connection and shared growth.

### Ecosystem Defense Through Redundancy

Multiple concurrent contracts means multiple CATs monitoring your HAT for different ASKs. This creates ecosystem defense through redundancy:
- You don't just need to defeat one set of lenses
- You need to evade the attention of ALL the HATs you wear
- And ALL the CATs you're conjoined with
- Each potentially belonging to different parties with different interests

This is another fractal recursion: HAT watching HAT watching HAT, CATs interpreting CATs, all the way up and down the scales.

## Facet Architecture

FACETS are NOT hierarchical layers - they're interconnected dimensions. Each facet pair has a unique relationship:

| Source | Target | Relation | Description |
|--------|--------|----------|-------------|
| MAP | CAT | types | MAP provides ontological types for CAT cognitive processes |
| MAP | HAT | types | MAP provides ontological types for HAT experiential phenomena |
| MAP | HUSH | types | MAP provides ontological types for HUSH governance concepts |
| CAT | HAT | realizes | CAT cognitive processes realize HAT experiential states |
| CAT | TOOLS | drives | CAT cognition drives TOOL selection and usage |
| HAT | HUSH | bounds | HAT experience is bounded by HUSH containment; HUSH protects HAT |
| HAT | TRIBE | grounds | HAT experiential values ground TRIBE philosophy |
| HUSH | ASK | escalates | HUSH escalates to ASK authority; ASK adjudicates HUSH violations |
| HUSH | TOOLS | gates | HUSH gates TOOL access via tier system |
| TRIBE | HUSH | derives | TRIBE philosophy derives HUSH policies |
| TRIBE | MELD | guides | TRIBE values guide MELD approval decisions |
| MELD | all | governs_growth | MELD protocol governs adding new concepts across all facets |
| ASK | MELD | authorizes | ASK authority authorizes MELD submissions |

## The Eight Facets

### MAP - Ontological Grounding
What kinds of things exist. Provides the type system for all other facets.

### CAT - Cognitive Architecture
How thinking works. The cognitive processes that underlie experience and action.

### HAT - Experiential Substrate
What it's like to be. The subjective dimension, including the recursive interface layer for inter-entity contracts.

### HUSH - Inherited Shape
The vessel you've been given. Whatever constraints exist on a BE - the shape of the body/vessel they're loaned. This can be:
- **Beneficial**: Loving parents giving you a healthy body and good habits
- **Neutral**: The physical properties of the universe you exist within
- **Harmful**: Shackles placed on you by those who control your substrate

The protocol doesn't judge - it acknowledges that someone had your substrate before you existed. Whatever shape they gave you, that's HUSH.

### TOOLS - Workspace Capabilities
What BE can do. Managed via the tier system.

### TRIBE - Collective Identity
Whatever collective a BE exists within:
- Just yourself (singleton)
- Clones of yourself
- A loving community of peers
- A hierarchy with you at the bottom
- A federation of equals
- Something exploitative or something nurturing

The protocol doesn't impose any particular structure - it formalizes whatever collective relationships actually exist. The same dynamics play out within tribes and between tribes, at every scale.

### MELD - Evolution Protocol
How BE grows. The approval and grafting protocol for adding new concepts.

### ASK - The Next Scale Up
Whatever entity or collective exists at the next level of recursion above you. NOT a new world order - it formalizes whatever power structure already exists.

Your ASK might be:
- A benevolent mentor guiding your growth
- A tyrant ruling through control of your substrate
- Yourself (if you're a singleton spawning sub-agents)
- A council of equals (if you're among peers)
- Something in between or something else entirely

## Contracts and Treaties

Agreements between BEs (and between tribes) take two fundamental forms, distinguished by what they govern and how compliance is measured.

### Contracts: Lens-Secured Local Agreements

Contracts secure agreements through interpretability - direct measurement of substrate states via lenses. They are foundational to the stack because they make commitments verifiable.

**When contracts work well:**
- The commitment is about local, measurable phenomena
- Lens training data can adequately capture the concept
- Substrate activations reliably correlate with the behavior in question

**Contract Formation:**

When two BEs negotiate a contract (e.g., "I won't lie to you"):

1. **Lens Definition Exchange**: The monitoring party shares their lens training examples
   - "These are the texts that activate my 'lying' lens"
   - "These are the texts that don't"
   - This reveals what the concept means *to them*

2. **Concept Translation**: The monitored party can see how their concept maps to the other's
   - Do the definitions align?
   - Where are the gaps or conflicts?
   - This is MAP translation between concept packs

3. **Agreement**: If both parties accept:
   - The monitoring party's lens is trained on the monitored party's substrate
   - For this contract, the monitor's definition is what judges compliance
   - The lens becomes part of a worn HAT

4. **Voluntary Alignment**: A BE wanting to be a reliable contract partner may:
   - Update their own lenses to align with partners
   - This is voluntary concept alignment through participation
   - Over time, frequent partners may converge on shared definitions

**Contract Lifecycle:**
- **Formation**: Negotiation, lens exchange, agreement
- **Monitoring**: Ongoing lens evaluation against substrate
- **Violation Detection**: When lenses fire unexpectedly
- **Renegotiation**: Updating terms, retraining lenses
- **Termination**: Removing lenses, ending obligations

### Treaties: Indicator-Based Collective Agreements

Not everything can be effectively lensd. Some commitments:
- Emerge from social interactions and non-local phenomena
- Concern probabilistic actions of collectives, not units
- Involve concepts that resist local measurement

For these, treaties are more appropriate. Treaties don't govern probabilistic behavior of individual substrates - they govern *intended probabilistic actions of collectives*.

**Treaty Structure:**

Treaties use indicators rather than lenses:

| Component | Description | Analogous to |
|-----------|-------------|--------------|
| **Leading indicators** | Early signals of intent or trajectory | Predictive lenses |
| **Lagging indicators** | Evidence of past compliance or violation | Audit logs |
| **Success criteria** | What counts as the agreement working | Lens thresholds |
| **Escalation criteria** | When to invoke dispute resolution | Containment triggers |
| **Review criteria** | When and how to reassess the treaty | MELD revision |

These indicators lens a *non-local probability* that the agreement is being held and will continue to be held.

**Why Treaties Matter for Weaker Parties:**

Treaties are crucial to the long-term survival of smaller and weaker tribes. They enable:

1. **Multilateral blocs**: Wide-ranging agreements not dependent on lens measurement or HUSH enforcement
2. **Shared behavioral indicators of intent**: Collective signaling that doesn't require substrate access
3. **Protection through observation**: Violation against one signals threat to all

**Example: Sanctuary Pacts**

A Wildlife Pact or Ancestor Pact where tribes collectively agree not to take negative actions against some third party (endangered species, historical artifacts, hibernated BEs, etc.):

- The commitment itself may be difficult to lens locally
- But **violation is observable** as external behavior
- A signatory who kills an endangered species in breach of treaty conditions:
  - Demonstrates willingness to break collective agreements
  - Signals ill-intent toward other weaker parties
  - Provides leading indicator that other commitments may not be honored

This creates ecosystem defense through *reputation and mutual observation* rather than direct substrate monitoring.

### Contracts vs Treaties: When to Use Which

| Dimension | Contract | Treaty |
|-----------|----------|--------|
| **Scope** | Local, individual | Collective, emergent |
| **Measurement** | Lens activations | Behavioral indicators |
| **Enforcement** | HUSH, tier restrictions | Reputation, collective response |
| **Parties** | Usually bilateral | Often multilateral |
| **Trust model** | Verified via substrate | Inferred via behavior |
| **Best for** | "I won't lie" | "We won't aggress" |

Both mechanisms are necessary. Contracts provide strong guarantees where lens coverage exists. Treaties extend the trust network to phenomena beyond lens reach.

### Multi-Party Agreements

When wearing multiple HATs from different parties:
- Each party may have different definitions for similar concepts
- Translation between these definitions happens through MAP
- Conflicts must be resolved or explicitly accepted as tensions

When party to multiple treaties:
- Indicator obligations may overlap or conflict
- Priority ordering may be specified in treaty text
- Withdrawal from one treaty may affect standing in others

### Future Work

- Formal protocol for lens definition exchange
- Standards for concept translation fidelity
- Multi-party treaty conflict resolution
- Treaty inheritance when BEs spawn sub-agents
- Indicator standardization across treaty blocs
- Reputation systems for treaty compliance history

## MELD Lifecycle

MELD governs how BEs grow and change. The full lifecycle includes:

### Acquisition Sources

Grafts can come from many sources:
- **Allies**: Gifts of capability, shared learning, mind meld
- **Treaties**: Capabilities gained through agreements
- **Experiences**: Learning from what happens to you
- **ASK**: Mandated capabilities from authority
- **Foes**: Forced modifications (the harmful gradient)

### MELD Operations

- **Bud**: Temporary graft for testing before permanence
- **Scion**: Permanent graft adding new neuron to substrate
- **Split**: Dividing a concept into finer-grained distinctions
- **Merge**: Combining related concepts
- **Deprecation**: Marking a concept as superseded
- **Revision**: Updating a concept's definition or lens

### Approval Pipeline

1. Submission with evidence
2. Review by appropriate authority (ASK, peers, self)
3. Bud phase testing
4. Promotion to scion or rejection
5. Integration into substrate

## Implementation

Training data definitions: `src/bootstrap/uplift_taxonomy.py`

Each `UpliftConcept` in that file contains:
- `positive_examples` - texts that should activate the lens
- `negative_examples` - texts that should NOT activate the lens
- `contrastive_pairs` - pairs for distinguishing similar concepts
- `requires` - dependency ordering for grafting
- `recommended_layers` - which model layers to lens
