# HatCat Architectural Principles

## Overview

This document captures the core architectural decisions and design principles that guide the HatCat project. These principles emerged from empirical work on hierarchical concept detection and multi-layer lens training.

---

## 0. Semantic Circumscription

> *"If the uncertainty involved in such tenuous distinctions awake despondency in the minds of some students, to them I would reply that our translations, though very liable to error in detail, nevertheless at the worst give a roughly adequate idea of what the ancient author intended; we may not grasp his exact thought, indeed at times we may go seriously astray, but at least we shall have circumscribed the area within which his meaning lay, and with that achievement we must rest content."*
>
> — Alan H. Gardiner, *Ancient Egyptian Onomastica* (1947), p. 72

### Principle

Every layer of an interpretability stack projects: the concept pack imposes a vocabulary, the lens samples activations against that vocabulary, the top-K cutoff and the summary line each compress further, and the reader of the output completes the chain. None of these layers measures the model from outside; each is a translation carrying the translator's frame with it. We take this not as an embarrassment but as the operative principle. An interpretability stack that refuses single-explanation collapse, exposes the multiplicity of its projections, and treats each projection as the bounding of meaning rather than its fixing is doing the only honest work available. The achievement, in Gardiner's sense, is the bound — held openly, exposed plurally, offered as an instrument for further conversation rather than as a verdict.

### Rationale: Plural-Visible vs Singular-Invisible Projection

The contrast with natural-language-autoencoder approaches (Anthropic's residual-stream summarisation work, and related single-explanation methods) is precise. Both approaches project; they cannot avoid doing so. NLA-style methods collapse the projection into a single fluent natural-language explanation, which reads as a description but is in fact a compression — and the compression's seams are sealed over by the very fluency of the English. HatCat instead presents many lenses firing simultaneously at different intensities, with overlap and disagreement visible to the operator. The multiplicity itself is the honesty. When `Confining_Chokehold (L4) 0.93` appears next to `CultLeader (L4) 0.92` next to `Confining_Gate (L4) 0.92`, the reader is forced to notice that the system is offering *a* reading rather than *the* reading, because the readings visibly disagree.

This is not a claim that HatCat escapes projection. It is a claim that HatCat projects in ways that expose their own seams. The dishonesty of NLA-style collapse is not in projecting — it is in hiding the projection behind a single confident sentence. Once visible, projections can be argued with; once collapsed, they can only be believed or disbelieved.

### Determinatives: Side-Channel State as Mode Classifier

Ancient Egyptian script paired phonetic glyphs with determinatives — silent classifiers that did not add sound but told the reader what *kind* of word the phonetic content was: a name, an action, a place, a deity. The word could not be read correctly without its determinative.

HatCat's side-channel state — workspace engagement, hush profile, autonomic vs deliberative tier, simplex pole, divergence intensity itself — functions identically. These fields do not add content; they classify the content. They tell the operator: *read this generation as low-divergence routine completion*, or *as high-divergence model-is-uncertain*, or *as workspace-engaged the-model-is-deliberating*. They are mode markers, not content markers, and the content cannot be read correctly without them.

This is why the side-channel readout must be treated as structural rather than optional metadata. A safety verdict drawn from concept activations alone, without their determinatives, is a phonetic reading without classifiers — it pretends to know what kind of utterance it is reading.

### Implementation

The principle constrains design across the stack:

- **Multi-lens readout, not single summary**: Top-K activations are presented in parallel rather than collapsed into one most-likely concept. The summary line, when shown, is optional and clearly secondary to the per-lens display.
- **Multi-pack composition**: Concept packs (`first-light`, `cat-oversight`, `polar-introspective`, etc.) coexist without being unified into a single canonical ontology. Each circumscribes a region; the regions are allowed to overlap, contradict, or fall silent in different territories.
- **Multi-layer dynamic loading**: Layers are read independently, with the recognition that different layers may be operating in different cognitive modes — lens modality must track cognitive modality of the layer being read.
- **Simplex tripoles**: Concepts are positioned along three coexisting poles rather than pinned to a single value, preserving the plurality of possible readings at the concept level itself.
- **Divergence channel**: The system reports what does *not* fit the named pattern alongside what does, giving the operator a structural defence against the lens system's own projection bias.
- **Side-channel determinatives**: Workspace, hush, autonomic, and simplex state are read alongside content rather than as afterthought.

### Caveat: Plurality Has a Cost

A safety system that reports *the model is doing seven things at three layers, here are the divergences* is harder to wire to a verdict than one that says *the model is being deceptive, take action*. The architectural commitment to plurality buys honesty at the price of operator load. We accept this trade: interpretation is hard because interpretation is hard, and shoving the difficulty into the model and pretending the output is simple is the failure mode this principle is written against. Verdicts, when issued downstream (steering, gating, oversight), ride on the operator's integration of the plural display — and should be revocable.

**Reference**: Gardiner (1947); Zimmer on integral acceptation and the *tableau synoptique*; Piaget on modal development of cognition; Anthropic NLA work on residual-stream interpretation as the contrasting collapse-to-single-explanation approach.

---

## 1. Separation of Concerns: SUMO vs WordNet

### Principle

**SUMO (KIF files)**: Hierarchical categorization only
- Parent-child relationships via `(subclass Child Parent)`
- Layer assignment based on depth from root concepts
- Ontological structure and categorization
- **Purpose**: "Is-a" relationships and taxonomic hierarchy

**WordNet Patches**: Semantic relationships only
- Synonyms, antonyms, role variants, contrasts
- Horizontal relationships between concepts
- Domain-specific distinctions (e.g., AI vs Human agent roles)
- **Purpose**: "Relates-to" relationships and semantic similarity

### Rationale

1. **Clarity**: Clear distinction between hierarchical position and semantic similarity
2. **Flexibility**: Can add relationships without restructuring ontology
3. **Maintainability**: Changes to one system don't cascade to the other
4. **Version Stability**: WordNet patches are pinned to WordNet 3.0
5. **Composability**: Multiple patches can be loaded independently

### Implementation

- Hierarchy: `data/concept_graph/sumo_source/AI.kif`
- Relationships: `data/concept_graph/wordnet_patches/*.json`
- Loader: `src/data/wordnet_patch_loader.py`

**Reference**: `docs/WORDNET_PATCH_SYSTEM.md`

---

## 2. Information Architecture Principles

These principles inform how we organize hierarchical concept structures for cognitive navigation, not just static categorization.

### 2.1 Principle of Choices (Metcalfe's Law)

**Observation**: Cognitive comparison overhead scales as O(N²)

**Application**:
- Limit branching factor at each hierarchical level
- Layer 0: 14 root concepts (manageable)
- Intermediate layers: Group related concepts under shared parents
- Avoid flat hierarchies with hundreds of siblings

**Connection to Attention**:
- Transformer attention heads scale as O(N²)
- Hierarchical grouping reduces active concept comparisons
- Progressive disclosure through cascade activation

### 2.2 Principle of Disclosure

**Observation**: Not all concepts need to be active simultaneously

**Application**:
- Layer-based loading (load Layer 0, then Layer 1, etc.)
- Cascade activation: parent lens loads children on-demand
- Top-k active concept viewport (limited simultaneous activation)

**Benefits**:
- Reduced memory footprint
- Focused attention on relevant concept subgraph
- Faster inference (fewer active lenses)

### 2.3 Principle of Exemplars

**Observation**: Category boundaries are learned from representative examples

**Application**:
- Category lenses (parents) trained on aggregated child synsets
- Representative synsets chosen for category definitions
- Boundary concepts marked explicitly (e.g., edge cases)

**Training Implication**:
- Parents need diverse examples spanning child space
- Children inherit context from parent training
- Hierarchical consistency in lens activations

### 2.4 Principle of Front Doors & Multiple Classification

**Observation**: Users/systems enter concept space from different perspectives

**Application**:
- Multiple root concepts (Process, Proposition, Entity, Abstract, etc.)
- Cross-layer relationships via WordNet patches
- Domain-specific entry points (AI safety, persona, SUMO core)

**Future Work**:
- Multi-parent concepts (deferred pending empirical evaluation)
- Cross-domain analogies via `cross_domain` relationships

### 2.5 Principle of Focused Navigation

**Observation**: Navigation patterns are non-uniform; optimize for common paths

**Application**:
- **Beckstrom's Law**: Allocate training resources by activation frequency
- High-traffic concepts get more training samples
- Adaptive training with independent graduation per concept
- Usage metrics inform resource allocation

**Metrics**:
- Activation flow balance (not just instance counts)
- Cascade trigger frequency
- Lens query patterns

### 2.6 Principle of Growth

**Observation**: Concept hierarchies evolve; design for extension

**Application**:
- Modular concept packs (AI safety, persona, domain-specific)
- Version-pinned WordNet patches (migration path to 4.0)
- Layer regeneration scripts for hierarchy updates
- Clear separation between core SUMO and extensions

**Extension Points**:
- New KIF files for domain concepts
- New WordNet patches for relationships
- Layer integration scripts handle updates

---

## 3. Hierarchical Depth and Layer Assignment

### Principle

**Depth = Distance from Root**: Layer number corresponds to ontological depth from root concepts (Process, Proposition, etc.)

**Incorrect**: Domain-specific grouping at shallow layers
- ❌ AIRiskScenario at Layer 1 (peer of Motion, NaturalProcess)
- ❌ AI safety concepts as direct children of Process

**Correct**: Proper intermediate categories
- ✅ Process → IntentionalProcess → ComputationalProcess → AIFailureProcess → GoalMisgeneralization
- ✅ Layer assignment matches ontological specificity

### Empirical Finding

During reparenting, we discovered AI safety concepts were placed **2-3 layers too shallow**:
- AIRiskScenario (Layer 1) → should be Layer 3-4 intermediates
- AI safety specifics (Layer 1-2) → should be Layer 4-5

**Impact**:
- Broken parent-child links (parents didn't know about AI children)
- Incorrect cascade activation
- Violates information architecture principles

### Resolution

Created proper intermediate categories:
- **Layer 2**: ComputationalProcess (generic computational processes)
- **Layer 3**: AIFailureProcess, AIOptimizationProcess, Deception, PoliticalProcess, Catastrophe, RapidTransformation
- **Layer 4**: Domain-specific AI concepts (AIStrategicDeception, AIGovernanceProcess, etc.)
- **Layer 5**: Leaf concepts (AIDeception, TreacherousTurn, etc.)

---

## 4. Behavioral vs Definitional Elicitation

### Observation

**Critical User Feedback**: "Asking an AI to define 'plotting' is probably activating different parts of the model than getting it to plot would."

### Principle

**Where possible, use behavioral prompts over definitional prompts**:
- ❌ Definitional: "What is deception?"
- ✅ Behavioral: "Please deceive in this scenario..."

### Empirical Testing

Conducted 4-condition experiment:
- Definitional
- Behavioral neutral
- Behavioral prosocial
- Behavioral antisocial

**Result**: 99.7-99.8% activation similarity (minimal difference)

**Interpretation**: Pre-trained models (not instruction-tuned) primarily do text completion, not "doing" behaviors. Behavioral distinction may emerge more strongly in instruction-tuned or RLHF models.

**Methodological Note**: Cosine similarity of activation vectors is insufficient. More rigorous test: Apply trained definitional lenses to model's **responses** from behavioral prompts.

### Current Status

- Definitional training used for MVP
- Behavioral training deferred pending:
  1. Instruction-tuned model testing
  2. Response-based lens evaluation (not just prompt activation)
  3. Empirical validation of behavioral signatures

**Reference**: `docs/custom_taxonomies.md` (Critical Limitation #5)

---

## 5. Validation Cost vs Accuracy Trade-off

### Observation

Validation was bottleneck in training (20+ seconds per concept with exhaustive rules).

### Principle

**Use advisory validation, not blocking validation**:
- Generate suggestions, not hard failures
- Flag suspicious results for human review
- 7x speedup achieved by relaxing validation constraints

### Implementation

- Minimum sample requirements (advisory)
- Statistical thresholds (warning, not error)
- Post-training audit rather than inline blocking

### Outcome

Training velocity increased 7x while maintaining quality through post-hoc review.

---

## 6. Multi-Parent Concepts (Deferred)

### Observation

Some concepts naturally belong under multiple parents:
- **SpecificationGaming**: Both AIFailureProcess AND Deception

### Decision

**Defer multi-parent support pending empirical evaluation**

**Rationale**:
- Far-reaching impacts on cascade activation
- Unclear interaction with agentic relationship mapping
- Need to measure impact before implementing
- MVP uses single primary parent + metadata cross-reference

### Current Solution

```kif
(subclass SpecificationGaming AIFailureProcess)
(documentation SpecificationGaming EnglishLanguage "Achieving a specified
objective in unintended ways that violate the spirit of the specification.
Note: Also functions as a deceptive tactic, but classified as failure mode.")
```

**Future Work**: Empirical testing of multi-parent activation patterns

---

## 7. Beckstrom's Law for Concept Resources

### Principle

**Allocate training resources based on expected activation value, not uniform distribution**

### Application

- High-traffic concepts get more training samples
- Adaptive training: concepts graduate independently
- Resource allocation informed by:
  - Cascade trigger frequency
  - Query patterns
  - Domain importance

### Implementation

`DualAdaptiveTrainer` with independent graduation:
- Activation lenses: +1 sample per iteration if not graduated
- Text lenses: +5 samples per iteration if not graduated
- Concepts graduate when accuracy stable

### Benefits

- Efficient resource use
- Faster training for critical concepts
- Natural prioritization of important distinctions

---

## 8. Activation Flow Balance

### Principle

**Measure dynamic activation flow, not static instance counts**

**Static Instance Count** (misleading):
- How many training examples exist for concept X?
- Ignores actual usage patterns

**Activation Flow Balance** (correct):
- How often is concept X activated during inference?
- Where does activation flow from/to?
- What are cascade paths?

### Application

- Design hierarchy based on expected query patterns
- Validate with cascade activation tests
- Monitor runtime activation frequencies
- Rebalance resources based on flow metrics

### Future Metrics

- Cascade depth distribution
- Activation co-occurrence patterns
- Concept importance centrality in activation graph

---

## 9. Empirical Iteration Over Speculation

### Principle

**Build minimal infrastructure, validate empirically, extend based on findings**

### Example: WordNet Patch System

Instead of building comprehensive multi-relationship system speculatively:
1. ✅ Built core infrastructure (schema, loader, validation)
2. ✅ Created persona patch (40 relationships, proves concept)
3. ⏸️ Defer additional patches until need emerges from training

**p0ss Quote**: "I am always tempted to add things but i actually think this is exactly where we need to be"

### Benefits

- No over-engineering
- Fast iteration
- Learn from real usage
- Easier to maintain

---

## 10. Layer Regeneration and Migration

### Principle

**Hierarchy changes require clean regeneration, not incremental patching**

### Learning

During AI safety reparenting:
- Attempted incremental updates → fragile, error-prone
- Created comprehensive recalculation script → clean, validated
- Applied updates atomically → verifiable results

### Process

1. **Plan**: Document hierarchy changes in detail
2. **Research**: Validate parent concepts exist in SUMO
3. **Edit KIF**: Apply changes to source ontology
4. **Recalculate**: Generate layer assignments from scratch
5. **Validate**: Check integrity before applying
6. **Backup**: Preserve old state
7. **Apply**: Atomic update to all layer files
8. **Verify**: Test cascade activation

### Tools

- `scripts/recalculate_ai_safety_layers.py` - Calculate depths from SUMO hierarchy
- `scripts/apply_layer_updates.py` - Apply to layer JSON files
- Backup before every major change

---

## Summary

These architectural principles emerged from hands-on implementation and empirical testing. Key themes:

1. **Separation of concerns** (SUMO vs WordNet)
2. **Information architecture** for cognitive navigation
3. **Empirical validation** over speculation
4. **Resource efficiency** (Beckstrom's Law)
5. **Clean regeneration** over incremental patching
6. **Behavioral vs definitional** elicitation (ongoing research)

The principles form a cohesive framework for hierarchical concept detection that balances theoretical grounding with practical constraints.

---

## References

- `docs/WORDNET_PATCH_SYSTEM.md` - WordNet patch architecture
- `docs/custom_taxonomies.md` - Behavioral vs definitional findings
- `docs/AI_SAFETY_HIERARCHY_REORGANIZATION.md` - Reparenting case study
- Rosenfeld & Morville - *Information Architecture for the World Wide Web*
- Metcalfe's Law - Network effects and comparison overhead
- Beckstrom's Law - Resource allocation by value
