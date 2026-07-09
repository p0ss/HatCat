# Model Topology: A Conceptual Framework

**Status**: Working document
**Created**: 2025-01-15

---

## Overview

This document describes a conceptual framework for understanding and addressing the internal structure of language models. The framework distinguishes between the fixed structural substrate (content-neutral) and the emergent patterns that arise during inference (content-aware).

The goal is to establish a coherent mental model that informs how we build addressing schemes, partition activation space, and navigate model internals for interpretability and intervention.

---

## 1. The Structural Substrate

At the most fundamental level, a language model is a finite matrix:

- A fixed number of **layers**
- Each layer has a fixed number of **neurons** (or positions in the hidden dimension)
- Each neuron has a finite number of **connections** (weights)
- Each weight and bias has some precision (float16, bfloat16, etc.)

### The Evenweave

An uninitialized layer is like **evenweave fabric** - the uniform grid used for embroidery. Every cell is identical, equidistant, waiting for the pattern. The number of neurons in the layer is like the **embroidery frame** - it sets the dimensions and resolution of what can be stitched.

Before training, it's blank evenweave. Training embroiders the patterns onto it. The frame (hidden dimension) constrains what resolution of pattern is possible - a larger frame (more neurons) allows finer detail.

The structure exists before the content. The fabric before the stitching.

This is the **content-neutral address space**. Every position can be enumerated:

```
address = (layer_index, neuron_index)
```

This substrate exists before any activation. It's the hardware, the loom before any thread passes through. Aggressive quantization (4-bit, even lower) still yields powerful models because **the broad shape of this network is the dominant paradigm** - the topology matters more than the precision of individual weights.

### Key insight

You can fully enumerate this space. It's finite. Every position has an address regardless of whether anything ever activates there. This is the base map.

---

## 2. The Textured Carpet

When activations flow through the model, each layer develops a **texture** - a landscape of activation values across its neurons.

Imagine each layer laid out as a carpet:
- Some neurons spike up (high activation)
- Some neurons spike down (low or negative activation)
- Some remain flat (near zero)

This creates a **height map** or **topology** at each layer. The shape of this landscape - not just individual peaks - carries information.

The texture changes with every input. Different inputs create different carpets. But certain patterns recur: regions that tend to activate together, valleys that rarely fill, ridges that light up for certain kinds of content.

---

## 3. The Causal Weave

Running through these textured layers is a **causal weave** - threads of information flowing from input to output.

These threads connect:
- **Embeddings** (input representations)
- **Fact-like structures** (stored knowledge)
- **Rule-like structures** (syntactic and logical patterns)
- **Motivation-like structures** (goals, intent, style)
- **Output representations** (predictions)

The weave runs from low layers to high, but it's not uniform. Some threads carry syntax, others carry semantics, others carry pragmatics. They can be traced, and they're somewhat independent of each other.

This is the **residual stream** view - information threading through the model, being read from and written to at each layer.

---

## 4. The Loom Metaphor

The loom provides a unifying image:

| Loom Component | Model Equivalent | Description |
|----------------|------------------|-------------|
| **Warp threads** | Structural substrate | Fixed vertical threads, the layer × neuron grid |
| **Shuttle** | Attention mechanism | Carries information across, moves dynamically |
| **Weft** | Activation pattern | The horizontal thread being woven through |
| **Tension** | Input signal | Maintained throughout the forward pass |
| **Heads pulling** | Attention heads | Multiple heads influencing each other's paths |
| **Resulting textile** | Output representation | The final woven pattern |

### Attention as shuttle

The attention mechanism doesn't just pass information layer-by-layer. It's the shuttle in the loom - it shoots across, creating connections that aren't in the structural substrate.

Each context creates activations that don't disappear into the embeddings. Instead, there's a **distributional probability resonance** at the embedding level. Attention takes that resonant frequency and carries it through the warp threads.

The tension of the input is maintained throughout. Attention heads pull on each other as they weave - they're not independent, they're coordinated (or competing) elements of a single weaving process.

### Thread Thickness and Probabilistic Tension

Weights and biases range between 0 and 1 (or after normalization, occupy a bounded range). Think of this as **thread thickness** - thicker threads pull more strongly.

At any decision point in the network, multiple threads are pulling in different directions:
- Some pull toward one interpretation
- Others pull toward alternatives
- The resolution is probabilistic, weighted by thread thickness

But it's not independent pulls. **Pulling on one thread changes the tension on others.** This is the attention dynamics - when one pathway activates strongly, it affects the effective strength of other pathways. The whole structure is under tension, interconnected.

In a real loom:
- Pull one thread tight, adjacent threads feel it
- The tension propagates through the fabric
- The final pattern depends on the balance of all tensions

In the network:
- Strong activation on one pathway affects competing pathways
- Softmax creates explicit competition (attention must sum to 1)
- Residual connections maintain baseline tension throughout
- The output is the equilibrium of all these pulling forces

### Key Insight: Equilibrium, Not Flow

The forward pass isn't just "activation flows forward." It's finding **equilibrium in a web of competing tensions**, with the output being wherever the forces balance.

This reframes what the model "does":
- Not: compute a function from input to output
- But: find the stable state of a tension system given the input

The input sets initial tensions. The weights define how tensions propagate. The output is where the system settles. Generation is iterative settling - each token shifts the tensions, the system rebalances, the next token is wherever equilibrium now points.

This is why small perturbations can have large effects (butterfly effects in activation space), and why steering works - you're adding tension in a particular direction, and the whole web of tensions rebalances around it.

### Implications for Steering

This tension-web model explains several design choices in the steering infrastructure:

**Why layerwise manifold steering:**
A naive intervention at one layer ignores the flow-on effects. Pulling a thread at layer 8 changes tension at layers 9, 10, 11... The manifold steering approach accounts for this by applying coordinated interventions across layers, respecting how tension propagates through the structure.

**Why field steering:**
Rather than pulling a single thread (one feature direction), field steering shifts an entire landscape of influences simultaneously. You're not disrupting one thread or even one pattern within threads - you're cohesively bending the topology that the shuttle will route through.

Single-thread intervention: yank one thread, others snap or tangle.
Field intervention: tilt the whole loom, all threads adjust together.

The shuttle (attention) still picks its path, but the landscape it's navigating has been reshaped. The probabilistic choices remain, but the probability distribution has shifted.

**Why this matters for grafting:**
When adding new capacity (grafting a new concept), you're not just adding a thread - you're adding a thread that will pull on all its neighbors. The graft needs to be integrated into the tension web, not just attached. This is why we test for degradation on adjacent concepts - to check that the new thread hasn't disrupted the existing weave.

---

## 5. Superposition as Pattern-in-Weave

**Neurons are not features. Concepts are not neurons.**

When you weave a pattern into a textile, the character or image exists across multiple locations. Each thread (pixel, neuron) may participate in multiple parts of the design:

- As a **boundary** between two pattern regions
- As an **eye** within a broader bird figure
- As a **color transition** in a gradient
- As a **letter edge** in border text

The same thread serves multiple patterns simultaneously. This is superposition.

**"Images show up in pixels, but are not the pixels."**

Concepts are **combinatorial phenomena** represented in neurons. They're the gestalt that emerges from the pattern of activation across many neurons, not properties of individual neurons.

This is why:
- Individual neurons are hard to interpret (they participate in many patterns)
- Directions in activation space are more meaningful than individual neurons
- Probes work by finding the direction that captures a concept's pattern
- The same activation can be read through multiple "lenses" and yield different meanings

---

## 6. The Dimensional Lens

Each neuron and its relationships can be thought of as its own space with its own directions. You can **peer through each dimensional lens** and the properties of up-down-left-right rearrange themselves into the specific paradigm you're analyzing.

The same underlying structure looks different depending on which conceptual frame you apply:

| Lens | What you see |
|------|--------------|
| Deception lens | Which regions encode truthfulness vs. manipulation |
| Uncertainty lens | Which regions encode confidence vs. hedging |
| Factual lens | Which regions encode stored knowledge |
| Syntactic lens | Which regions encode grammatical structure |

These aren't different structures - they're different **projections** of the same high-dimensional space. The lenses are the probes. Each probe defines a direction, and projecting onto that direction reveals one facet of the crystal.

---

## 7. The Forward Pass as Landscape Navigation

When we do a forward pass:

1. Inputs start at the bottom in the **embeddings**
2. They propagate along the **weaving** between textured layers
3. At each juncture, the path taken has the **best probabilistic match** between:
   - The existing weave structure (learned weights)
   - The current input activations
4. This is **manifold navigation** - moving across a landscape
5. Steering intervenes by tilting the landscape, making some paths more likely

The manifold topology is what we're steering. We're not changing the structure (the warp threads); we're biasing which paths the shuttle takes through them.

---

## 8. Addressing Scheme

This framework suggests a multi-level addressing scheme:

### Level 0: Structural Address (content-neutral)

```
(layer_index, neuron_index)
```

- Fixed, finite, enumerable
- Exists before any activation
- The warp threads of the loom
- Can identify every position regardless of content

### Level 1: Activation Address (content-aware)

Which structural positions activate for a given input:

```
active_set = {(l, n) for all (l, n) where activation[l][n] > threshold}
```

- Varies per input
- The texture of the carpet for this weaving
- Reveals which threads participated in this pattern

### Level 2: Pattern Address (conceptual)

Which concepts are expressed in this activation pattern:

```
concepts = {c for c in vocabulary where probe[c](activations) > threshold}
```

- The meaning extracted via lenses
- Multiple concepts may be present (superposition)
- Same activations, different lenses, different readings

### Level 3: Trace Address (causal)

Which path did information take through the layers:

```
trace = [(l, source_neurons, target_neurons) for l in layers]
```

- Following the shuttle's path
- Identifies causal chains
- Useful for understanding how conclusions were reached

---

## 9. Dead Space and Exploration

**Dead space** = structural addresses that never or rarely activate across diverse inputs.

These are warp threads the loom never uses. The weaver's patterns never require them. But they're still there - structural capacity that the training distribution didn't need.

Questions:
- What happens if you steer toward dead space?
- Is anything encoded there, just rarely accessed?
- Can grafting use dead space to add new capabilities without interference?
- Does dead space vary by model, or is there consistent unused capacity?

Steering toward dead space is forcing the shuttle through threads it normally avoids. What pattern emerges?

---

## 10. Implications for Octree/Mapping

The original octree approach tried to partition the activation *values* - the heights on the textured carpet. This was backwards.

**The correct approach:**

1. **Start with the structural grid** (Level 0 addresses)
   - This is content-neutral and finite
   - No PCA needed, no sampling needed
   - The grid IS the model architecture

2. **Discover inter-layer connectivity** (empirical)
   - Static weights define within-layer connections
   - Inter-layer flows through residual stream + attention
   - Discover empirically via bidirectional probing

3. **Map concepts to regions** (Level 2)
   - For each concept, which structural regions participate?
   - This is learned by observation (run concept exemplars, see what lights up)
   - The pattern is in the weave, distributed across threads

4. **Identify dead space**
   - Structural addresses that never appear in activation sets
   - Candidates for exploration and grafting

The "octree" subdivision then becomes: **clustering of structural addresses by connectivity pattern**. Not partitioning activation values, but grouping structural positions that influence each other.

### Bidirectional Probing for Inter-Layer Connectivity

Layers don't have direct weight matrices connecting them - they communicate via the residual stream. But we can empirically discover effective connectivity:

**Bottom-up (1-bit fuzzing):**
- Set neuron i to 1, all others to 0
- Run forward, measure what lights up downstream
- This gives "if this fires, what can fire next"
- Cheap: just forward passes, no gradients

**Top-down (QKV reverse inspection):**
- Start with downstream activation at neuron j
- For attention paths: `score_i = softmax(Q_j · K_i) · ||V_i||`
- For linear paths: weight matrix alignment with j's basis
- Aggregate across thousands of random contexts

**What this captures:**
- ~85% of effective connectivity (dominant pathways)
- Misses: higher-order interactions, destructive interference
- The coarse map guides where to do detailed pairwise probing

**Octree-like subdivision:**
- Start coarse, subdivide where interesting
- More compute = more precision
- Built-in refinement strategy

This is fuzzing the network to get a probabilistic map of inter-layer responses. The clusters come from this empirical connectivity, not from static weight inspection alone.

---

## 11. Terminology Mapping

| Framework Term | Technical Equivalent |
|----------------|---------------------|
| Evenweave | Uninitialized layer (uniform grid, no pattern) |
| Embroidery frame | Hidden dimension size (constrains resolution) |
| Structural substrate | Model architecture (layers × hidden_dim) |
| Warp threads | Fixed neuron positions |
| Shuttle | Attention mechanism |
| Weft | Activation pattern being woven |
| Thread thickness | Weight magnitude (stronger = thicker = more pull) |
| Tension | Activation state, competition between pathways |
| Pulling on threads | Attention dynamics, pathway competition |
| Textured carpet | Layer activation values |
| Causal weave | Residual stream / information flow |
| Pattern in weave | Superposition / distributed representation |
| Dimensional lens | Linear probe / projection direction |
| Dead space | Low/zero activation structural regions |
| Manifold navigation | Forward pass as probabilistic path selection |
| Layerwise manifold steering | Coordinated intervention respecting cross-layer tension propagation |
| Field steering | Shifting entire topology, not single threads |
| Equilibrium | Output state where all tensions balance |
| Overlaid video | Repeated concept prompts to find consistent activation patterns |
| Heatmap | Concept-specific activation pattern after baseline subtraction |
| Probe | Single-layer concept detector (current) |
| Lens | Multi-layer concept detector (aspiration) |
| Ontological prompting | Using concept relationships to find all facets |
| Cleft | Subset of model involved in a concept (for targeted training) |
| Weight cluster | Group of neurons with strong weight connections (structural) |
| OOD cluster check | Testing if new examples activate outside the trained cleft |
| Live adjustment | Updating buds on running model without shutdown |
| Minimal cleft | Smallest set of clusters needed for incremental learning |
| 1-bit fuzzing | Bottom-up probing: set neuron to 1, measure downstream |
| Reverse tracing | Top-down probing: trace upstream sources via QKV inspection |
| Effective connectivity | Empirical inter-layer influence (discovered, not static) |
| Dynamic routing zone | Region where attention determines path (scattered backward traces) |

---

## 12. Probe Training as Overlaid Video

When we train a concept probe, imagine one of those video overlays where the same action is repeated many times and you see the probability distribution - the overlap points where the same thing happens in each video.

**Process:**
1. Prompt the model with a concept hundreds of times
2. Record activations (the video frames)
3. Overlay them all (see where they consistently align)
4. Subtract what happens in every video (the baseline, non-concept-specific activity)
5. What remains is the **heatmap** - the concept-specific highlights within the weave

This heatmap is what the probe learns to detect. It's a pattern of correlated activations that distinguish "this concept is active" from "something else is happening."

### Single-Layer vs Multi-Layer

The current first-light probes target just one layer. This seems weak - looking at only a tiny slice of the loom. But it's surprisingly effective, showing how much signal exists even with minimal instrumentation.

The aspiration is **lenses** rather than just probes:
- Accept probes at multiple layers
- Or develop a more efficient way to find the heatmap across the entire loom
- Minimal training and memory footprint

Single-layer probe = looking through a keyhole.
Multi-layer lens = seeing through the whole door.

### Ontological Prompting Strategy

The training approach matters. We don't use random concept examples. Instead:
- Use ontological relationships (parent, child, sibling concepts)
- Use linguistic relationships (synonyms, antonyms, related terms)
- Prompt from multiple angles to find all facets of the concept

This lets the probes learn the **hyperplane** that captures the concept, not just one narrow slice of it. The more aspects of the concept we activate during training, the more complete the heatmap.

---

## 13. Clefts and Efficient Training

### The Sparsity Insight

A model with billions of parameters doesn't use most of them for any given concept.

- Highly broad concepts (with many meanings, ambiguities, contexts) may be interconnected across the whole loom
- Specific, self-contained concepts occupy a lower-dimensional space
- Most parameters are not directly involved

### What the Cleft Does

The cleft identifies the **subset of the model most related to a target concept**, so we can:
1. Freeze everything outside the cleft
2. Train only the cleft region
3. Avoid updating billions of unrelated parameters

This trades completeness for efficiency:
- Some edge cases (subtle alternate meanings, rare contexts) may not be updated
- But the majority of the concept gets tied to the majority of related points
- Training time drops dramatically

### The Network Graph View

Think of the entire loom as a **weighted network graph**:
- Nodes = structural positions (neurons)
- Edges = weights connecting them
- Edge weights = connection strength (thread thickness)

The goal isn't to define what each feature means. It's to identify:
- Sets of features that are **highly correlated** (tend to activate together)
- Sets of features that are **highly unrelated** (never co-activate)

This is like **k-means clustering on the network graph** - finding natural groupings based on correlation, not semantics.

### Octree as Weight-Based Clustering

The "octree" we're building isn't partitioning activation values. It's identifying clusters from the **static weight structure** itself.

The model weights already contain a network graph:
- Nodes = neurons (structural positions)
- Edges = weight connections between neurons
- Edge weights = connection strength

This graph exists before any forward pass. It's content-neutral structure. You can analyze it purely from the weights:

```
Cluster A: neurons 100-200 (layer 5), 300-400 (layer 8), 50-80 (layer 12)
  → These have strong weight connections to each other

Cluster B: neurons 500-600 (layer 5), 800-900 (layer 8)
  → These are strongly connected internally, weakly connected to Cluster A
```

The subdivisions come from the weight matrix structure, not from observing activations. The clusters are defined by how the loom is woven, not by watching the shuttle pass through.

### Cleft Selection from Heatmaps

The clusters are pre-computed from the weight structure (content-neutral). Then:

1. Run the detection process (overlaid video approach)
2. Get the heatmap of concept-related activations
3. Check which **pre-defined clusters** have signals in the heatmap
4. **Exclude clusters with no signal** from the cleft

The clusters exist before you run any concepts through. The heatmap tells you which clusters are *relevant* to this particular concept.

If an entire arm of the weight graph never lit up during concept experiences, it's probably not related to this concept. Don't include it in the cleft, don't update it during training.

The cleft = union of weight-structure clusters that showed activity during concept detection.

### The Minimal Retraining Hypothesis

**Status: Hypothesis - requires experimental validation**

If a model already knows about bird species A, and needs to learn about species B (which has a slightly more upturned beak but is otherwise nearly identical), the number of features requiring retraining should be minimal.

We shouldn't have to shut down and retrain everything for incremental knowledge. The clusters that encode "bird," "beak," "feathers" are already there. Only the specific differentiating features (beak curvature, perhaps coloring) need updating.

The cleft for "species B" should be small - just the delta from what's already known.

### OOD Detection via Cluster Activation

This gives us a powerful test for bud quality:

**During out-of-distribution testing:**
- Run novel examples of the concept through the model
- Check which clusters activate
- Compare to the cleft we trained

**If new clusters light up that weren't in the cleft:**
→ The bud is wrong or incomplete. The concept involves model regions we didn't account for. The training missed something.

**If all OOD examples stay within the cleft:**
→ Even if the bud isn't perfect, it's reasonably generalized. The concept is contained within the regions we trained.

This is a structural test of generalization, not just performance metrics.

### Live Adjustment Hypothesis

**Status: Hypothesis - requires experimental validation**

For incremental changes (the bird beak case), it should theoretically be possible to:

1. Identify the minimal cleft (small set of differentiating features)
2. Apply a targeted bud
3. Test on OOD examples
4. Adjust the bud based on results
5. **All while the model continues running**

No shutdown. No full retraining. Continuous incremental learning.

This is closer to how biological learning works - you don't go offline to learn that this new bird has a slightly different beak. You update the relevant circuits while continuing to function.

The key enabler: knowing which clusters to touch (from the weight structure) and which to leave alone.

### What Would Falsify These Hypotheses

**Minimal retraining falsified if:**
- Incremental concept learning requires touching most of the model anyway
- Clefts for related concepts have high overlap (no sparsity benefit)
- Small cleft training fails to generalize

**OOD cluster check falsified if:**
- Generalized buds still activate unexpected clusters
- Failed buds don't show cluster spillover
- Cluster membership doesn't predict training success

**Live adjustment falsified if:**
- Bud updates cause cascading instability
- Sequential adjustments accumulate errors
- Model requires periodic full retraining to stay coherent

These are empirical questions. The framework makes predictions; experiments will test them.

---

## 14. Open Questions

1. **Granularity**: What's the right structural unit? Individual neurons? Groups of 64? Attention heads separately from MLP?

2. **Cross-layer structure**: The warp threads span layers, but are there consistent "columns" that co-activate across depth?

3. **Attention dynamics**: How do we address the dynamic connections attention creates? They're not in the structural substrate but they matter.

4. **Sequence position**: Same structural address means different things at different token positions. How does position factor into addressing?

5. **Model comparison**: Do different models have similar dead space? Similar co-activation patterns? Can the addressing scheme transfer?

---

## See Also

- [fractal-model-cartography.md](../planning/fractal-model-cartography.md) - Research proposal for mapping
- [EXPERIENTIAL_LEARNING_OVERVIEW.md](../BE/EXPERIENTIAL_LEARNING_OVERVIEW.md) - Grafting experimental protocol
- [FTW_OVERVIEW.md](../FTW_OVERVIEW.md) - Architectural context
- [GLOSSARY.md](GLOSSARY.md) - Terminology reference
