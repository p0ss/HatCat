# HatCat - Temporal Semantic Decoder for Language Models

A system for real-time interpretation and steering of language model activations, using temporal sequence analysis and binary concept classifiers to detect and manipulate polysemantic activations.

## Project Overview

Unlike Sparse Autoencoders (SAEs) or Neuronpedia, HatCat:
- **Temporal sequences**: Analyzes activation patterns in a sliding window over generation time
- **Binary classifiers**: Concept and edge detectors with antonym/distance opposites 
- **Polysemy-native**: Multiple concepts can be active simultaneously
- **Graph-based negatives**: WordNet semantic distance for training data
- **Concept steering**: Extract and manipulate concept vectors for controllable generation


# A Prototype Framework for Dual-Ontology Latent Steering and Reflective Regulation in Generative Models

## Abstract
This work introduces a practical framework for dual-ontology interpretability and control in large language models (LLMs).
The system detects and modulates thousands of semantic concepts in real time during text generation, providing an activation-level interface between model cognition and external oversight.
Unlike existing safety methods that operate exclusively in text space or rely on static classifier gating, this framework integrates three elements:

Latent Concept Detection: Lightweight lenses trained from WordNet + SUMO mappings identify concept activations across the residual stream and MLP layers. Concepts are organized into hierarchical “layers” (domains → subdomains → concept families → terms) that balance coverage with computational cost.

## Reflective and Safety Ontologies:
Two decorrelated lens sets support separate functions—(a) model self-reflection, enabling introspective awareness of cognitive and emotional activations, and (b) external safety oversight, monitoring ethically or socially sensitive content. Mutual-information regularization maintains ontological independence to prevent gaming.

## Closed-Loop Steering:
Activation vectors derived from concept subspaces allow controlled modulation of model states via pre- and post-layernorm hooks. Steering operates under dynamic gain scheduling and optional manifold projection for stability. The resulting feedback loop enables both internal alignment (reducing latent dissonance) and external correction (safety enforcement).

Empirical tests on mid-sized open models (e.g., Gemma 3-4B) demonstrate millisecond-scale concept detection with >90 % coherence across strength ranges and functional separation between reflection and oversight lenses.
The framework is compatible with existing SAE and feature-vector methods, providing a foundation for model-level self-regulation and transparent safety supervision without retraining.

## Why "HatCat" 
Large language models are powerful but opaque—we need to understand what they're thinking to make them safe and reliable. 

This system reads the minds of LLMs so we can clean up their mess. 

This makes a HatCat the little model that sits under a big model's hat, just like the little cats from "The Cat in the Hat"! VOOM > FOOM 

### How it works 
We begin with a knowledge graph of human concepts, from Wordnet or similar. 

For each concept we train a small classifier "lens" that learns to recognise each concept.  

At runtime, as the model generates text, every lens watches the activations and reports how strongly it detects its concept

From this data we can estimate which ideas the model is likely "thinking about" at any given moment 

## Dual Lens Design

HatCat uses **two independent classifiers** per concept to detect divergence between what the model thinks and what it writes:

### Activation Lenses (What the Model Thinks)
- **Non-linear MLP**: 2-layer feed-forward network (2560→512→1) with ReLU
- **Input**: Hidden state vectors from layer 0
- **Detects**: Concepts in the model's internal representations

### Text Lenses (What the Model Writes)
- **Linear classifier**: TF-IDF → Logistic Regression
- **Input**: Token text (string)
- **Detects**: Concepts in surface-level token content

When these two lenses disagree significantly, we detect **divergence**—the model is thinking one thing but writing another.

### Training Data
To build a lens, we first ask the model to define a concept in its own words.

While it generates the definition, we record the hidden-state activations—these are our samples of how the model represents that idea.

A complete lens contains many such samples, covering both the concept itself and its relationships to other concepts.

The model doesn't actually have the same predefined concepts as our graph, it has a complex interwoven higher-dimensional space.

So you can think of us launching our human concept lens into the model, and each sample as a sounding in the model's conceptual manifolds.

So we design our samples to try and chart out the shape of that whole region in the activation space. 

## Positive Half
**Centre**: definitions of the target concept, written by the model itself.
**Boundaries**:  relationship descriptions that situate the concept—its hypernyms (parents), hyponyms (children), meronyms (parts), holonyms (wholes), and other semantic neighbours—ranked by relationship importance

Together these define what it means for a concept to be “present". 

## Negative Half
**Center:** definitions of the opposite concept (its antonym), when one exists.
**Boundaries:** the antonym’s own relationship descriptions. 
**Fallback:** if no antonym is available, we sample distant, unrelated concepts to create a contrasting boundary.

this ensures each lens learns what isn't our concept, as well as what is

## Training the lenses 
Each lens is a binary classifier trained to distinguish positive from negative activations using standard stochastic gradient descent and binary cross-entropy loss.

We evaluate each lens on seperate set of model-generated definitions to test if it can spot versions of the concept it hasn't seen yet.

An adaptive scaling loop increases sample size only when needed:
* If a lens hits the target accuracy, training stops.
* If not, new positive, negative, and relational samples are generated and the lens retrains.
* This continues until the lens reliably identifies its concept (≥ 95 % accuracy)

## Running the lenses 
During inference, generation time is divided into short temporal slices.
In each slice, every lens reports its confidence score.
The top-scoring concepts show which regions of the model’s conceptual space are active as each token is produced, along with relative intensity over time.
It’s essentially a conceptual “EEG” for the model.

## Steering with lenses 
The full set of all activations captured as a lens can be applied as an offset to the model's activations.  So as the model is processing response probabilities, the neurons that were recorded as activating in the concept can be surpressed or amplified to change their probabilities of featuring in the response. 

Rather than pushing in a straight line through a curved space, we measure the curvature of the manifold and apply a falloff through surrounding layers to ensure no unintended consequences from the intervention. This is an upgraded version of Manifold Steering 

## Visualising the lenses (Planned) 
Because our concept set and relationships form a graph, we can visualise activations as highlighted regions of that graph—a bright web of concepts lighting up against the darker backdrop of unmapped space.

As the model generates a response, the activated regions trace its conceptual trajectory, like a rough fMRI of thoughts. 

## Mapping the dark spaces (Planned) 
As we evaluate our samples, we can begin to adjust concept definitions to be closer to the model's central distributions for a concept. This would allow a kind of bidirectional conceptual alignment, showing how far off the model's activations are from what we expected, and where we might want to apply steering.  This process could even become a kind of behavioural alignment reinforcement learning

We can also create new concepts, to represent conceptual we find within the model's unmapped spaces. Just as different human languages have concepts other don't, any given model may have concepts we don't know about.  These may not be solid "concepts" as we know them, but gradient regions that don't align to human ontological cuts, but reliably activate during some observed behaviour we want to steer or train. 

## Current Status: Advanced Manifold Steering (Phase 6-7)

**Goal**: Validate dual-subspace manifold steering for high-precision concept manipulation

**Completed:**
- ✅ Phases 1-4: Binary classifier training, scale tests (1-1000 concepts), comprehensive evaluation
- ✅ Phase 5: Semantic steering evaluation
- ✅ Phase 6: Contamination subspace removal (orthogonal projection)
- ✅ Phase 6.5: Task manifold steering (curved semantic surface estimation)
- ✅ Phase 6.6: Dual-subspace manifold steering (contamination removal + task manifold projection)
- 🔄 Phase 7: Logarithmic scaling validation [2,4,8,16,32,64] samples/concept (in progress)

**Phase 6.6 Architecture** (Dual-Subspace Manifold Steering):
```
1. Contamination Removal: Project out shared task-agnostic patterns
2. Task Manifold Projection: Steer along concept-specific curved surface
3. Combination: v_total = w1·v_contamination + w2·v_task_manifold
```

**Phase 6 Key Results** (5 concepts):
- **Contamination subspace**: Dimension 1024 → 5 principal components
- **Steering effectiveness**: Strong suppression with manifold-aware projection
- **Training time**: ~10s per concept
- **VRAM usage**: 8.62 GB (gemma-3-4b-pt, float16)

**Phase 7 Goal**:
- Identify optimal training scale via diminishing returns (ΔSE < 0.02)
- SE metric: Steering Effectiveness = 0.5 × (ρ_Δ,s + r_Δ,human) × coherence_rate
- Expected knee point: 8-16 samples per concept

**Next Steps:**
- Complete Phase 7 stress test (full scaling curves)
- Identify optimal sample size for production deployment
- Scale to 1000+ concepts with optimal configuration

## Architecture: Dual-Subspace Manifold Steering

### Phase 6.6: Manifold-Aware Concept Steering
```python
# 1. Train binary classifiers and extract steering vectors
concept_vectors = {}
for concept in concepts:
    classifier = train_binary(positives, negatives, min_distance=5)
    concept_vectors[concept] = extract_steering_vector(classifier)

# 2. Estimate contamination subspace (shared task-agnostic patterns)
all_vectors = torch.stack(list(concept_vectors.values()))
contamination_subspace = estimate_contamination_subspace(
    all_vectors,
    n_components=min(len(concepts), 5)  # Top-5 principal components
)

# 3. For each concept, estimate task manifold (concept-specific curved surface)
for concept in concepts:
    task_manifold = estimate_task_manifold(
        concept_vector=concept_vectors[concept],
        n_samples=16,  # Collect activations from low-strength steered generations
        strength_range=[0.05, 0.15]
    )
```

### Dual-Subspace Steering
```python
# Apply both contamination removal and task manifold projection
def dual_subspace_steering(hidden_states, concept, strength):
    # Step 1: Remove contamination (orthogonal projection)
    for basis_vector in contamination_subspace:
        projection = (hidden_states @ basis_vector) * basis_vector
        hidden_states = hidden_states - projection

    # Step 2: Project along task manifold (positive strength = amplify)
    task_vector = task_manifolds[concept]
    projection = (hidden_states @ task_vector.unsqueeze(-1)) * task_vector
    steered = hidden_states + strength * projection

    return steered
```

### WordNet Graph Negatives
```python
# Semantic distance-based negative sampling (unchanged from Phase 2)
negatives = sample_distant_negatives(
    concept_synset,
    min_distance=5,  # WordNet path hops
    candidates=all_117k_synsets
)

# Structured relationships for comprehensive concept coverage
related = {
    'hypernyms': [...],   # is-a (broader)
    'hyponyms': [...],    # types-of (narrower)
    'meronyms': [...],    # has-part
    'holonyms': [...],    # member-of
    'antonyms': [...]     # opposites
}
```


## Hierarchical Concept Organization (V5 - COMPLETE ✓)

**Goal**: Build multi-layer SUMO-WordNet ontology for hierarchical "zoom" - activate high-level concepts then drill into children

**Architecture**:
- Layer 0: 14 top-level ontological categories (Entity, Physical, Abstract, etc.)
- Layer 1: 276 SUMO categories (depth 3-4) - includes AIControlProblem, AIGrowth, AIAbuse
- Layer 2: 1,059 SUMO categories (depth 5-6) - includes AIDeception
- Layer 3: 991 SUMO categories (depth 7-9) - includes AIAlignment, AIPersonhood, AISuffering
- Layer 4: 3,221 SUMO categories (depth 10+) - includes ArtificialIntelligence, LanguageModel
- Layer 5: 21 Pseudo-SUMO hyponym clusters (subdividing large categories >1000 synsets)
- Layer 6: 115,930 WordNet synsets (82% direct SUMO, 18% via pseudo-SUMO intermediates)

**AI Ontology Extension (AI.kif)**:
- 70+ new SUMO categories for AI alignment, safety, and x-risks

**V5 Training Results** (Layers 0-2):
- **1,349 binary classifiers trained** (14 + 276 + 1,059)
- **Training time**: 42 minutes (~1.9 sec/classifier)
- **Average F1**: 99.4% (with only 10+10 training samples)
- **SUMO-aware training**: Combines SUMO category_children + WordNet relationships
- **Training approach**: Relationship-based (hypernyms, hyponyms, meronyms, antonyms)

**V5 Monitoring System** (Non-Invasive Temporal Detection):
- **No generation degradation**: 3% repetition rate (vs 80% threshold for mode collapse)
- **Robust detection**: 468 unique concepts across 30 test samples
- **API-ready JSON**: Consumable by Ollama, OpenWebUI, LibreChat, MCP
- **Integration**: Frontend UIs, model reasoning cycles, safety guardrails
- See `docs/TEMPORAL_MONITORING.md` for usage
- 184 WordNet synset mappings via expansion file
- 16/23 AI categories populated with synsets (143 unique synsets, multi-mapped)
- Covers: ArtificialAgent, ArtificialIntelligence, LanguageModel, AIAlignment, AIDeception, AISuffering, AIFulfillment, AIPersonhood, Superintelligence, etc.

**Training Strategy**: Start with Layer 0 high-level lenses. When activated, subdivide into Layer 1 children lenses to zoom in on model thinking.

**Files**:
- `data/concept_graph/abstraction_layers/layer{0-6}.json` - Hierarchical concept layers
- `data/concept_graph/sumo_source/AI.kif` - AI ontology extension
- `src/build_sumo_wordnet_layers_v5.py` - V5 build script with AI expansion

## Project Phases

### Phase 1: Find the Curve (Complete)
- **Goal**: Identify diminishing returns for definitions vs relationships
- **Result**: All configurations achieved 95-100% accuracy at 10 concepts
- **Learning**: Need larger scale to see differences (moved to Phase 2)

### Phase 2: Minimal Training Scale Test (Complete)
- **Goal**: Validate 1×1 training scales from 1 to 1,000 concepts
- **Configuration**: 1 positive + 1 negative per concept
- **Scales tested**: 1, 10, 100, 1000 concepts
- **Results**:
  - n=1: 100% (1/1 concepts)
  - n=10: 100% (10/10 concepts)
  - n=100: 96% (96/100 concepts)
  - n=1000: 91.9% (919/1000 concepts @ 100% test acc)

### Phase 2.5-5: Steering Quality & Semantic Evaluation (Complete)
- **Goal**: Test detection confidence and steering effectiveness
- **Results**: Strong negative steering, variable positive amplification
- **Key Findings**:
  - Detection: 94.5% mean confidence on OOD prompts
  - Suppression: Negative steering highly effective (0.93 → 0.05)
  - Amplification: Variable across concepts (some +2.00, others suppress)

### Phase 6: Contamination Subspace Removal (Complete)
- **Goal**: Remove shared task-agnostic patterns via orthogonal projection
- **Method**: PCA on all concept vectors → Top-5 principal components
- **Result**: Cleaner steering vectors for downstream manifold estimation

### Phase 6.5: Task Manifold Steering (Complete)
- **Goal**: Estimate concept-specific curved semantic surfaces
- **Method**: Generate with low-strength steering, collect activations, compute manifold vector
- **Result**: Manifold-aware steering improves precision

### Phase 6.6: Dual-Subspace Manifold Steering (Complete)
- **Goal**: Combine contamination removal + task manifold projection
- **Architecture**: Two-stage forward hook (contamination removal → manifold projection)
- **Result**: High-precision concept steering validated on 5 concepts

### Phase 7: Logarithmic Scaling Validation (In Progress)
- **Goal**: Find optimal training scale via SE metric and diminishing returns
- **Scales**: [2, 4, 8, 16, 32, 64] samples per concept
- **Expected Knee Point**: 8-16 samples (ΔSE < 0.02)

## Experiment Tracking

See `TEST_DATA_REGISTER.md` for complete experiment history.

**Major Results:**
- Phase 2 @ n=1000: 919/1000 concepts @ 100% test accuracy
- Phase 2.5 v3: Strong steering suppression, variable amplification
- Semantic tracking: Captures 8-13 related terms per concept

## Production Target

**Goal**: 10,000 concepts minimum for practical language coverage

**Phase 2 Validated:**
- ✅ 1×1 minimal training works at scale (919/1000 @ 100%)
- ✅ Detection confidence strong (94.5% mean)
- ✅ Steering suppression effective
- ⏳ Positive amplification needs refinement

**Next Steps:**
1. Complete Phase 2.5 (steering evaluation)
2. Scale to 10K concepts with 1×1 training
3. Refine positive steering methodology
4. Implement production sliding window inference

## Project Structure

```
HatCat/
├── src/
│   ├── encyclopedia/
│   │   └── wordnet_graph_v2.py       # ✅ Graph builder with relationships
│   └── interpreter/
│       ├── model.py                  # Binary classifier architecture
│       └── steering.py               # Concept vector extraction
├── src/
│   ├── steering/
│   │   ├── manifold.py               # ✅ Phase 6.6 dual-subspace manifold steering
│   │   └── hooks.py                  # ✅ Forward hooks for steering
│   └── interpreter/
│       ├── model.py                  # Binary classifier architecture
│       └── steering.py               # Concept vector extraction
├── scripts/
│   ├── phase_2_scale_test.py         # ✅ 1×1 minimal training at scale
│   ├── phase_2_5_steering_eval.py    # ✅ Detection + steering evaluation
│   ├── phase_6_subspace_removal.py   # ✅ Contamination subspace removal
│   ├── phase_6_5_manifold_steering.py # ✅ Task manifold estimation
│   ├── phase_6_6_dual_subspace.py    # ✅ Dual-subspace steering
│   ├── phase_7_stress_test.py        # 🔄 Logarithmic scaling validation
│   └── train_binary_classifiers.py   # Binary training
├── data/
│   └── concept_graph/
│       ├── wordnet_v2_top10.json     # ✅ 10 concepts
│       ├── wordnet_v2_top100.json    # ✅ 100 concepts
│       ├── wordnet_v2_top1000.json   # ✅ 1K concepts
│       └── wordnet_v2_top10000.json  # 🎯 10K concepts (target)
├── results/
│   ├── phase_1-3/                    # ✅ Classifier training, scale tests
│   ├── phase_4-5/                    # ✅ Neutral training, semantic steering
│   ├── phase_6*/                     # ✅ Subspace removal, manifold steering
│   └── phase_7*/                     # 🔄 Scaling validation
├── logs/                             # Execution logs
├── TEST_DATA_REGISTER.md             # ✅ Experiment tracking
└── README.md                         # This file
```

## Key Metrics

**Phase 2 Scale Test** (1000 concepts):
- Per-concept test accuracy: 100% (919/1000 concepts)
- Training data: 1 positive + 1 negative per concept
- Total training time: ~4-5 hours on single GPU

**Phase 2.5 Steering** (20 concepts):
- Mean detection confidence: 94.5%
- Negative suppression: 0.93 → 0.05 semantic mentions (-94%)
- Positive amplification: Variable (0 to +2.00 mentions)
- Semantic tracking: 8-13 related terms per concept


## Quick Start

### 1. Train Binary Classifiers (Phase 2)
```bash
# Train 1×1 minimal classifiers for 10 concepts
poetry run python scripts/train_binary_classifiers.py \
    --concept-graph data/concept_graph/wordnet_v2_top10.json \
    --model google/gemma-3-4b-pt \
    --output-dir results/classifiers_10 \
    --n-definitions 1 \
    --n-negatives 1

# Expected: 100% validation accuracy in ~5 minutes
```

### 2. Dual-Subspace Manifold Steering (Phase 6.6)
```bash
# Test advanced manifold steering on 5 concepts
poetry run python scripts/phase_6_6_dual_subspace.py \
    --device cuda

# Output: results/phase_6_6_dual_subspace/
# - Contamination subspace visualization
# - Task manifold scatter plots
# - Steering effectiveness metrics
```

### 3. Logarithmic Scaling Validation (Phase 7)
```bash
# Find optimal training scale [2,4,8,16,32,64] samples/concept
poetry run python scripts/phase_7_stress_test.py \
    --device cuda

# Expected: ~30-60 minutes for full scaling curve
# Output: SE metric plot, cost curve, knee point detection
```

## Tech Stack

- **Model**: Gemma-3-4b-pt for generation, can be rerun on other models
- **Framework**: PyTorch + PyTorch Lightning
- **Storage**: HDF5 with gzip compression
- **Concept Graph**: WordNet (117K synsets)
- **Training**: Binary cross-entropy per concept
- **Architecture**: BiLSTM + MLP classifier
- **Steering**: Linear concept vectors extracted from classifiers
- **Inference**: Sliding window (size=20, stride=5)

## Key Learnings

**What Works:**
1. **minimal training**: Scales excellently (91.9% @ n=1000 for one sample)
2. **Negative steering**: Highly effective suppression
3. **Semantic grouping**: Tracks broader steering effects than exact matches
4. **Concept-specific prompts**: Enable quantitative steering measuremen

**Open Questions:**
1. Positive steering variability across concepts
2. Optimal steering strength selection
3. Role of antonyms in negative steering
4. Scaling to 10K+ concepts
5. Rather than treating the residual error as noise, we could analyse the activation clusters to identify the model’s nearest analogues. By steering the lens toward those centroids, we might iteratively co-define a hybrid ontology that sits at the intersection of human and model semantics.

**Failed Approaches:**
- Generic steering prompts (can't measure steering effect)
- Exact term matching only (misses semantic field effects)
- Large sample sets (compute inefficient, tests saturated @ 40 definitions and 40 relations 



## Comparison: HatCat vs Sparse Autoencoders (SAEs)

| Aspect | HatCat | Sparse Autoencoders (SAEs) |
|--------|--------|---------------------------|
| **Core Approach** | Binary classifiers per concept | Reconstruction-based sparse coding |
| **Architecture** | BiLSTM + MLP per concept | Single feedforward autoencoder |
| **Temporal Analysis** | ✅ Sliding window over generation sequence | ❌ Single activation snapshot |
| **Polysemy Handling** | ✅ Native: multiple concepts active simultaneously | ❌ Feature competition; features often polysemantic |
| **Training Data** | Semantic negatives (WordNet distance ≥5 hops) | Unsupervised reconstruction (billions of tokens) |
| **Training Speed** | 8 hours for 1,000 concepts @ 80% (single GPU) | Days-weeks + manual labeling to identify concepts |
| **Concept Steering** | ✅ Extract steering vectors from classifiers | ⚠️ Requires separate steering methodology |
| **Interpretability** | Direct binary prediction per concept | Sparse features (many polysemantic, need labeling) |
| **Hierarchical Relations** | ✅ Graph-based relationships (hypernyms, meronyms, etc.) | ❌ Flat feature space |
| **Data Efficiency** | Self-generates training data from WordNet seed | Requires large unlabeled corpus |
| **Feature/Concept Count** | 91.9% @ 1K clean concepts, targeting 10K+ | 10K-100K+ features (many polysemantic/fragmentary) |
| **Steering Quality** | 94% suppression effectiveness validated | Variable, methodology-dependent |


##FAQs

**Feature purity vs polysemy** Are HatCat’s concept activations cleaner than SAE features?
I don't know about "cleaner" we get a probability for each of the top concepts for each time slice. This accounts for the model "thinking" about multiple things even while outputing one token. If anything its "messier", but thats also more reflective of whats happening. SAEs aim for one feature per 'thing', we explicitly
  allow multiple concepts active because that's how models actually work"

**Faithfulness**  Does turning a HatCat concept on/off change model behaviour in the expected way?
kind of. it can seemingly enhance and surpress entire conceptual fields, this isn't term or token surpression though and seems more akin to temperature changes than censoring an embedding. Like if you amp up an emotion, the model changes its outputs in a way akin to having that emotion, and if you supress that emotion it will change behaviour but can still say the word

**Temporal coherence** Does HatCat track concepts through generation rather than just static token 
Yes, basically in real time but obviously inference compute dependent. the codebase includes tests to demo this. Need to test at scale

**Hierarchical consistency** Do parent/child relationships behave monotonically?
The relationship training schedule usually achieves this, but it also depends on their relative semantic density since that scales the training data which can influence concept activation strength,  which is also true in the model. 

**Data-efficiency benchmark** How many examples per concept reach a target accuracy?
i ran a matrix of these tests, we were around 75%-80% accuracy from a sample pair, around 95% from 10 sample pairs and 99-100% around 40 sample pairs. Will publish charts and paper soon. See TEST_DATA_REGISTER.Md for raw data now

**Interpretability survey** Can humans successfully label what each unit means in a blind evaluation?
that is going to require some other humans

**Cross-model generalisation** Do the same concept classifiers transfer across LLMs?
Within a family it might kind of work, but i can't see any reason that the activations for different model families should be the same for the same concept. Still, if you can just train a new hatcat for another model in a couple of hours thats not really a big deal

**Scalability sanity** Does performance degrade as concepts → 100 K +?
at 10, 100, 1000 its been better at bigger scales, will see at 10k, but the trend so far has that disambiguation improves with concept scale


**Comparative steering quality** Are HatCat steering vectors more targeted?
No, we're supressing conceptual fields not logits . so i don't know this will ever be Golden Gate bridge levels of logit. maybe with much bigger knowledge graphs 

**Interpretability completeness**  does HatCat fully interpret neurons?
No, this is probabilistic interprability. We're sending human concepts as lenses into a higher-dimensional activation space and getting soundings of the manifold they're in. They can sit in the rough area of a concept, but can't reach full accuracy because our concepts don't match the model concepts.  Although activation signatures converge rapidly in cosine space, relative differences plateau at ~0.3, suggesting a residual manifold spread that resists simple linear collapse. Confidence interval width declines primarily because of averaging noise, not complete semantic unification. 

**Correlation vs Causation** Sounds like lenses pick up correlates, won't you need causal evals to move the model reasoning? 
There aren't really repeatable causal chains in probabilistic systems, just the gradients things are likely to follow. It absolutely is picking up correlates, but we iterate through the relationships that concept has with surrounding concepts so the correlations are still grounded in their relative topology.  If your bowling alley has slope down in one direction, that doesn't cause all shots to go that way, but it does 


**Allowing vulnerabilities**  This is unsafe code which exposes vulnerabilities in deployed systems

We're not creating these problems, we're making them impossible to ignore. There are likely to be some negative use cases, but long term safety requires this infrastructure.  It was irresponsible of model creators to let capability so eceed interpretability. This turns the lights on and exposes some dirty laundry, but those failure modes already existed and are being ignored. 

## Contributing

I'm just a person playing around, not really in a position to provide much support

## License

TBD (likely MIT or Apache 2.0)

---

**Status**: Phase 7 logarithmic scaling validation - identifying optimal training scale via SE metric

**Last Updated**: November 5, 2025
