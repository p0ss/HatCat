# Targeted Concept Grafting: Experimental Protocol

**Status**: Active experiment
**Last Updated**: 2025-01-15

---

## Abstract

We present an experimental protocol for teaching language models new concepts through targeted weight modification, guided by interpretability infrastructure. The approach uses activation probes to identify conceptual gaps, collect training data with per-token activation mapping, and graft new capabilities via reversible then permanent weight interventions. This document describes both the theoretical framework and our current experimental setup.

---

## 1. Introduction

### 1.1 The problem

Large language models acquire concepts during pre-training, but this process is opaque and uncontrollable. Post-training methods (RLHF, fine-tuning) can add capabilities, but often degrade existing ones, and provide no mechanism for the model to direct its own learning.

### 1.2 Our approach

We propose a structured pipeline where:

1. A second model (the "judge") evaluates the subject model's concept knowledge
2. Gaps are identified through systematic probing against a concept ontology
3. Activation patterns during concept experiences are recorded at token-level granularity
4. Targeted training modifies only the regions active during those experiences
5. Learning is tested for generalisation before permanent integration

The goal is a model that can learn from its own experiences, verify that learning, and accumulate capabilities without catastrophic forgetting.

---

## 2. Experimental setup

### 2.1 Current phase: Judge model evaluation

Before we can assess a subject model's concept knowledge, we need a reliable judge. Our current experiment evaluates candidate judge models for their ability to distinguish good concept descriptions from bad ones.

**Methodology:**

We use a knowledge graph containing safety-related concepts, where each concept has:
- A canonical definition
- Positive examples (correct descriptions/instances)
- Negative examples (incorrect, incomplete, or misleading descriptions)

This provides deterministic ground truth. We present each judge candidate with concept-description pairs and ask: "Is this an accurate description of this concept?"

**Metrics:**

- Accuracy: Does the judge correctly classify positive and negative examples?
- Discrimination: Can it distinguish subtle errors from complete failures?
- Consistency: Does it give stable judgments across phrasings?

**Candidate models under evaluation:**

| Model | Parameters | Architecture | Notes |
|-------|------------|--------------|-------|
| Gemma 3 4B | 4B | Transformer | Current baseline, has trained probes |
| OLMo 3 7B Think | 7B | Transformer | Reasoning mode via `<think>` tags |
| Nemotron Nano 9B | 9B | Mamba-Transformer hybrid | Only 4 attention layers |
| Qwen3 8B | 8.2B | Transformer | Reasoning mode via `<think>` tags |
| Ministral 8B | 8B | Transformer | 128k context |
| Apriel 15B Thinker | 15B | Multimodal | 4-bit quantised for 24GB VRAM |

**Long-term goal:**

The judge role (which we term "thalametrist") will eventually be a dedicated model trained specifically for concept assessment. This experiment establishes the baseline capabilities required.

### 2.2 Next phase: Subject concept assessment

Once we have a qualified judge, we use it to systematically assess the subject model's knowledge across a concept ontology.

**Procedure:**

1. Select a concept from the ontology
2. Prompt the subject model to describe/explain/identify the concept
3. Have the judge evaluate the subject's response against ground truth
4. Record pass/fail with confidence scores
5. Iterate across the ontology until we identify concepts the subject reliably fails

**Target:** Find a concept where the subject fails at sufficient sample size (n ≥ 30) to establish statistical confidence that this represents a genuine gap rather than noise.

### 2.3 Activation recording phase

With a target concept identified, we collect training data by exposing the subject to many instances of the concept while recording internal activations.

**Data collection:**

- Present the subject with diverse instances of the target concept
- Record activations at every layer for every token generated
- Map activations to concept probes (small classifiers trained to detect ~8000 concepts)
- Tag each token with which conceptual regions were active during its generation

**Output:** A dataset of (concept instance, per-token activation patterns) pairs, showing which parts of the model engage when experiencing and responding to the concept.

### 2.4 Bud grafting phase

Using the tagged activation data, we attempt a reversible graft ("bud").

**Procedure:**

1. Identify the "cleft": the set of model regions most active during target concept experiences
2. Add a new neuron to the most active layer
3. Fine-tune only the cleft regions on the collected experience data
4. This creates a LoRA-like adapter that can be attached or detached without modifying base weights

**Validation:**

- Expose the budded model to held-out instances of the target concept
- Judge evaluates whether the model now correctly handles the concept
- Test a sample of other concepts from the cleft to verify no degradation

If the bud fails to generalise, additional concept instances are collected and a new bud is trained. This iterates until generalisation is achieved.

### 2.5 Permanent grafting phase

When a bud demonstrates reliable generalisation, it is converted to a permanent graft ("scion").

**Procedure:**

1. Append a new neuron to the target layer, representing the learned concept
2. Create weighted connections to all neurons affected during training
3. Set connection strengths proportional to how much each neuron changed during bud training
4. Train a paired probe (lens) to detect the new concept in future activations

**Validation:**

- Re-run judge evaluation on the target concept
- Re-run evaluation on concepts in the affected cleft
- Compare to pre-graft baseline to verify no degradation

**What the graft encodes:**

The new neuron's weight pattern is not arbitrary - it records which parts of the model were involved in learning this concept. The graft is simultaneously:
- A feature detector for the concept
- A record of the learning process
- A hook for future probe-based monitoring

---

## 3. Success criteria

| Phase | Success metric |
|-------|----------------|
| Judge evaluation | ≥85% accuracy on concept discrimination task |
| Subject assessment | Identify concept with ≥90% failure rate at n≥30 |
| Activation recording | Per-token probe coverage ≥95% of generated tokens |
| Bud grafting | Subject passes judge evaluation on target concept |
| Generalisation | ≤5% degradation on cleft-adjacent concepts |
| Permanent graft | Above metrics hold after scion integration |

---

## 4. Theoretical framework

### 4.1 Why this should work

Language models develop internal representations of concepts during training. These representations are distributed across many neurons but follow consistent patterns - the same concept activates similar regions across different contexts.

By recording which regions activate during concept experiences, we identify where the model "stores" that concept. By training only those regions on new examples, we target the intervention precisely rather than risking interference with unrelated capabilities.

The bud/scion distinction allows us to test before committing. Catastrophic forgetting typically occurs because fine-tuning affects weights that encode other knowledge. By scoping the intervention to regions already associated with the target concept, we minimise this risk.

### 4.2 What we're testing

This experiment tests whether:

1. Judge models can reliably assess concept knowledge
2. Activation probes accurately identify concept-relevant regions
3. Targeted training on those regions teaches the concept
4. Generalisation can be achieved without degrading other capabilities
5. The graft persists and remains detectable via its paired probe

### 4.3 Relation to existing work

This approach combines elements from:

- **Representation engineering**: Using activation directions to understand and steer models
- **Sparse fine-tuning**: Modifying only relevant parameters to reduce forgetting
- **Continual learning**: Accumulating capabilities over time without retraining
- **Mechanistic interpretability**: Understanding which circuits implement which capabilities

The novel contribution is closing the loop: using probes to identify gaps, guide training data collection, scope the intervention, and verify the result - all within a single integrated pipeline.

---

## 5. Infrastructure

The experiment runs on the FTW (Fractal Telescope Web) stack:

| Component | Role |
|-----------|------|
| HAT (Headspace Ambient Transducer) | Probe infrastructure, activation extraction |
| MAP (Mindmeld Architectural Protocol) | Concept ontology, probe registry |
| XDB (Experience Database) | Tagged experience storage |
| Thalamos | Examination room orchestrating the pipeline |

See [FTW_OVERVIEW.md](../FTW_OVERVIEW.md) for architectural context.

---

## 6. Terminology reference

| Term | Meaning |
|------|---------|
| Judge / Thalametrist | Model evaluating another model's concept knowledge |
| Subject | Model being assessed and potentially grafted |
| Probe / Lens | Small classifier detecting a concept in activations |
| Cleft | Set of model regions affected by a concept |
| Bud | Reversible graft (like LoRA) |
| Scion | Permanent graft integrated into weights |
| Ontology | Structured knowledge graph of concepts and relationships |

---

## 7. Current status

- **Judge evaluation**: In progress. Downloading candidate models, running discrimination benchmarks.
- **Subject assessment**: Blocked on judge selection.
- **Activation recording**: Infrastructure ready, awaiting target concept.
- **Bud grafting**: Infrastructure ready, awaiting training data.
- **Permanent grafting**: Infrastructure ready, awaiting validated bud.

---

## See also

- [GLOSSARY.md](../specification/GLOSSARY.md) - Full terminology reference
- [thalametry-examination-room.md](../planning/thalametry-examination-room.md) - Detailed implementation plan
- [FTW_OVERVIEW.md](../FTW_OVERVIEW.md) - The larger architectural context
