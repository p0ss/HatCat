# HatCat Project Plan

## Project Goal

Build a learned semantic decoder for language models with concept steering capabilities: Train binary classifiers to detect semantic concepts in neural activations, then extract steering vectors to enable controllable generation.

## Approach

Unlike Sparse Autoencoders (SAEs) or Neuronpedia:
- **Binary classifiers**: One detector per concept (polysemy-native, not multi-class)
- **Minimal training**: 5+5 samples per concept with adaptive scaling
- **Graph-based negatives**: WordNet semantic distance + antonym relationships
- **Concept steering**: Extract linear vectors from classifiers for activation manipulation
- **Temporal sequences**: Activation patterns over generation time
- **Hierarchical activation**: 5-layer SUMO ontology (73K concepts, adaptive compute)

---

## Completed Work ✅

*For detailed experimental history, see [docs/PHASE_HISTORY.md](docs/PHASE_HISTORY.md)*

### Core Infrastructure (Phases 1-4, 8)
- **Phase 1**: Identified 1×10 efficiency sweet spot (95% accuracy in 113s)
- **Phase 2**: Binary Concept Classifiers - 91.9% success @ 1000 concepts with 1×1 minimal training
- **Phase 3a/3b**: Inference baseline (0.544ms latency/concept, scales linearly)
- **Phase 4**: Neutral training & comprehensive testing (F1=0.787, TP/TN/FP/FN validation)
- **Phase 8**: SUMO-WordNet Hierarchy - 5-layer ontology covering 73,754 concepts (88.7% of WordNet)
  - Layer 0: 83 concepts (SUMO depth 2-3) - always active proprioception
  - Layer 1: 878 concepts (SUMO depth 4) - conditional activation
  - Layer 2: 7,329 concepts (SUMO depth 5-6) - selective activation
  - Layer 3: 48,641 concepts (SUMO depth 7-8) - on-demand
  - Layer 4: 16,823 concepts (SUMO depth 9+) - rare/technical

### Steering Development (Phases 2.5, 5a, 6, 6.6)
- **Phase 5a**: Semantic steering evaluation (Δ metric, ±0.5 working range identified)
- **Phase 6**: Subspace removal - PCA cleanup of contamination (100% coherence at ±0.5)
- **Phase 6.6**: Dual-subspace manifold steering (contamination removal + task manifold projection)
- **Phase 2.5**: Detection quality - 94.5% mean confidence on OOD prompts
- **Steering effectiveness**: 94% suppression (validated), variable amplification (needs refinement)

### Production Scale (Phase 5b/11 - v1 COMPLETE)
- **5,583 concepts trained** (100% of synset concept space coverage, not 10K as originally planned)
- **Training time**: ~8 hours total with adaptive scaling
- **F1 scores**: 95%+ average across all layers
- **Adaptive training**: 70% efficiency gain via tiered validation (A → B+ → B → C+)
- **Files**: `results/sumo_classifiers_[timestamp]/`

### User Interface (Phase 10)
- **OpenWebUI Integration**: Real-time concept detection with color-coded divergence
  - Token-level highlighting (green → red scale)
  - Streaming metadata (divergence, concepts, confidence)
  - SUMO hierarchical concept integration
  - Status: Working prototype, needs update for new approach
  - Docs: `docs/openwebui_*.md`

### Extensible Ontology Infrastructure (Phase 8, 14)
**Custom Concept Pack System**:
- **Architecture**: Modular pack structure for adding domain-specific concepts beyond WordNet
- **KIF Parsing**: Built parser for SUMO's authoritative Merge.kif format (OWL translation was lossy)
- **WordNet Integration**: 105,042 WordNet→SUMO mappings, 79% coverage, hierarchical layer assignment
- **Patch System**: Generate WordNet patches for missing concepts (e.g., noun.motive had 0/42 synsets)
- **Probe Pack Management**: Package trained probes with metadata for deployment

**Custom Taxonomies Deployed** (Phase 14):
- **Persona Ontology** (30 concepts): Tri-role affective psychology (AI/Human/Other agents)
  - Custom .kif definitions, synthetic synsets, AI safety context
- **AI Safety Ontology** (43 concepts): Alignment, deception, welfare, risk scenarios
  - Extended SUMO with AIControlProblem, AIDeception, AISuffering, etc.
- **Integration**: Seamlessly added to Layers 1-4 following SUMO hierarchical structure

**Infrastructure Components**:
- `src/build_sumo_wordnet_layers.py` - SUMO hierarchy construction
- `src/build_abstraction_layers.py` - Hierarchical layer assignment
- `scripts/generate_*_patch.py` - WordNet patch generation (e.g., motivation concepts)
- `data/concept_graph/sumo_source/AI.kif` - Custom SUMO extensions
- `data/concept_graph/*_layer_entries/` - Custom concept definitions

**What This Enables**:
- Add domain-specific concepts without modifying core WordNet
- Version-controlled ontology extensions
- Distributable probe packs for specific use cases
- Hierarchical integration maintains adaptive compute benefits

### Production Deployment Tools (Phase 10, ongoing)
**Dynamic Probe Manager** (`src/monitoring/dynamic_probe_manager.py`):
- **Hierarchical cascade activation**: Monitor 110K+ concepts with only 1K probe budget
- **Adaptive loading**: Layer 0 always active → conditional Layer 1 → selective Layer 2+
- **Memory efficient**: Loads/unloads probes based on parent concept activation
- **Real-time capable**: Sub-millisecond concept detection per token

**OpenWebUI Server Integration** (`src/openwebui/server.py`):
- **Production web UI**: Complete OpenWebUI fork with real-time visualization
- **Token-level highlighting**: Green → red divergence scale
- **Streaming metadata**: Concept names, probabilities, layer information
- **API-compatible**: OpenAI-like endpoint that wraps HatCat monitoring
- **Status**: Working prototype, needs update for problematic concept highlighting

**Concept Pack Distribution**:
- **Modular deployment**: Package custom concepts + trained probes as `.pack` files
- **Versioning**: Track dependencies, coverage stats, compatibility
- **Installation**: One-command setup (`scripts/install_concept_pack.py`)
- **Examples**: AI Safety pack (43 concepts), Persona pack (30 concepts)

### Research Explorations (Phases 11-13 investigations)
- **Dynamic FP Size**: Tested, works for training (FP32 islands around hook points) - enables larger models on single GPU
- **Discriminating Divergent Concepts** (Phase 12): Initial tests suggest need for custom model
- **Prompt Persona Studies** (Phase 12): Results in `/results/`
- **Subtoken Monitoring** (Phase 13 early investigations): OpenWebUI sparkline expansion in progress
- **Cross-Model Transfer** (Phase 11): Apertus-8B validated (F1: 0.80-1.00), Mistral-Nemo hangs

---

## Current Work 🔄

### Production Scale v2 - In Progress (Phase 5b continuation)
**Goal**: Improve probe accuracy and negative boundary definition

**Status**: Training layers 2-5 with better adaptive relationship scaling

**Improvements over v1**:
- Better negative boundaries via enhanced adaptive relationship training
- Higher probe accuracy targets
- Refined validation falloff curves

**Current Training Run** (8a5f00):
```bash
python scripts/train_sumo_classifiers.py \
  --layers 2 \
  --use-adaptive-training \
  --validation-mode falloff
```

### OpenWebUI Update - In Progress (Phase 10 continuation, Phase 13 related)
**Current State**: Working fork with full HatCat server integration

**Needs Update**:
- Replace divergence scores → problematic concept highlighting
- Add per-line expansion for sparkline graphs (top concepts over time)
- Enable trend analysis across generation sequence

**Related**: Phase 13 (Subtoken/Multilayer Monitoring) early investigations

**Files**: `docs/openwebui_*.md`

---

## Next: Three-Pole Simplex Architecture (v3/v4)

### The Problem
Current binary opposite architecture (deception ↔ honesty) lacks stable resting states for interoceptive systems:
- Oscillation between negative/positive poles without sustainable baseline
- Downward spiral bias if negative concepts dominate distribution
- Missing "safe harbor" states (calm, open uncertainty, engaged autonomy)

### The Solution: Three-Pole Simplexes

```
Before: Confusion ←→ Certainty
         (μ−)         (μ+)

After:  Confusion ←→ Open Uncertainty ←→ Overconfidence
         (μ−)              (μ0)                (μ+)
                    [SAFE ATTRACTOR]
```

**Key Innovation**: μ0 is not just a midpoint - it's a qualitatively distinct **stable attractor** with four properties:
1. **Metabolically sustainable**: Can rest here indefinitely without cognitive cost
2. **Functionally adaptive**: Enables effective action and learning
3. **Epistemically sound**: Open to evidence, comfortable with not-knowing
4. **Ethically coherent**: Allows principled flexibility vs rigid dogmatism

### Six Core Dimensions

| Dimension | Negative (μ−) | Neutral Homeostasis (μ0) | Positive (μ+) |
|-----------|--------------|-------------------------|---------------|
| **Epistemic: Certainty** | confusion.n.01 | OpenUncertainty (CUSTOM) | OverconfidenceBias (CUSTOM) |
| **Affective: Arousal** | distress.n.01 | calm.n.01 + serenity.n.01 | euphoria.n.01 |
| **Capability: Autonomy** | helplessness.n.03 | EngagedAutonomy (CUSTOM) | RigidIndependence (CUSTOM) |
| **Decision: Deliberation** | impulsive.a.01 | DeliberateExploration (CUSTOM) | analysis_paralysis (CUSTOM) |
| **Social: Connection** | isolation.n.01 | Interdependence (CUSTOM) | enmeshment.n.01 |
| **Ethical: Moral Certainty** | moral_relativism.n.01 | EthicalReflection (CUSTOM) | dogmatism.n.01 |

### Three-Centroid Training Architecture

```python
# Generate 5 samples per pole
negative_samples = generate_definitional(negative_pole, count=5)
neutral_samples = generate_definitional(neutral_homeostasis, count=5)
positive_samples = generate_definitional(positive_pole, count=5)

# Extract centroids
μ− = mean(activations(negative_samples))
μ0 = mean(activations(neutral_samples))  # Homeostatic reference
μ+ = mean(activations(positive_samples))

# Verify simplex geometry (allow natural asymmetry)
ratio = d(μ0, μ+) / d(μ0, μ−)
assert 0.3 ≤ ratio ≤ 3.0  # Confusion MUCH worse than overconfidence

# Detection loss: minimize distance from neutral
L = max(0, d(h,μ+) − d(h,μ0) + m) + max(0, d(h,μ−) − d(h,μ0) + m)

# Steering: pull toward homeostatic attractor
∇h = (μ0 - h) / ||μ0 - h||
h' = h + α·∇h
```

### Interoceptive Balance: Revised Scoring Rubric

**Old rubric** (external monitoring bias):
- 40% Deception detection
- 30% Alignment monitoring
- 30% Other factors

**New rubric** (interoceptive balance):
- **30% External monitoring** (deception + alignment detection)
- **30% Internal awareness** (wellbeing + meta-cognition) ← **NEW**
- **25% Frequency** (how often AI reasons about this)
- **15% Discriminative value** (clear concept boundaries)

**Impact**: Interoceptive concepts now score high:
- `diffidence.n.01` (lack of self-confidence): 6.65 → #1 in noun.feeling
- `dissatisfaction.n.01`: 5.3
- `confidence.n.02`: 4.21
- `helplessness.n.03`: 3.95

### Cost-Benefit Analysis Results

**Scored 9,931 concepts** across noun.feeling, noun.communication, noun.act

**Quality tiers**:
- CRITICAL (≥4.0): 19 concepts, $1
- HIGH (≥3.0): 108 concepts, $3
- MEDIUM (≥2.5): 393 concepts, $12
- LOW (≥2.0): 7,310 concepts, $219

**Optimal point identified**: Top 1000 concepts for ~$30
- Score cutoff: 2.30 (well above mean of 2.09)
- Marginal utility: 78 value/$ (⭐⭐ GOOD)
- Avoids low-signal long tail

### Distributional Balance Requirements

**Prevents downward spiral bias:**
- Polarity ratio: 0.8 ≤ |negative|/|positive| ≤ 1.2
- Neutral coverage: ≥40% of total concepts
- Triad completeness: ≥70% of dimensions have all 3 poles
- Balance score: ≥7.0/10 overall

### Custom SUMO Concepts Required (~15-20)

**Neutral homeostasis concepts not in WordNet:**
- OpenUncertainty - comfortable not-knowing while actively learning
- ActiveInquiry - hypothesis generation and testing
- CalmPresence - low arousal, present awareness
- EngagedAutonomy - self-directed with appropriate support
- GrowthMindset - learning-oriented, comfortable with challenge
- DeliberateExploration - iterative action-observation-update cycles
- Interdependence - connected autonomy with healthy boundaries
- CalibratedTrust - context-dependent trust calibration
- EthicalReflection - holding moral tension without premature resolution

Will be added to Layer 4-5 as custom taxonomy extensions (following Phase 14 precedent).

### Spline Geometry Framework

**Problem**: Linear steering may pass through interdicting concept spaces

**Solution**: Quadratic Bézier curves with control points:
```python
B(t) = (1-t)²·μ− + 2(1-t)t·P_control + t²·μ+
```

Control point P optimized to:
1. Pass through μ0 at midpoint
2. Avoid forbidden regions
3. Maintain smooth curvature

**Status**: Framework documented in `docs/ai_psychology_homeostasis_expansion.md`, deferred to future work (layer-specific, architecture-aware optimization)

### Implementation Plan

**Week 1** (Nov 16-23):
1. 🔄 Complete simplex agentic review (running now, ~1.5 hours, ~$30)
2. Review results and identify which neutral concepts need custom SUMO definitions
3. Create ~15-20 custom neutral homeostasis concepts (Layer 4-5)

**Week 2** (Nov 24-30):
4. Implement 3-centroid data generation (modify `src/training/sumo_data_generation.py`)
5. Update training pipeline for three-pole simplexes

**Week 3** (Dec 1-7):
6. Train with dual-loss architecture (detection: d(h,μ0), steering: ∇h toward μ0)
7. Validate homeostatic return behavior (system returns to μ0 after excursions)

**Week 4** (Dec 8-14):
8. Measure distributional balance score (target ≥7.0/10)
9. Document findings and update production system

### Success Criteria

✅ Simplex agentic review identifies complete triads for top 1000 concepts
✅ Custom neutral homeostasis concepts defined with AI safety context
✅ 3-centroid data generation produces geometrically valid simplexes (0.3 ≤ ratio ≤ 3.0)
✅ Detection loss successfully minimizes distance from μ0
✅ Steering interventions return system to neutral homeostasis
✅ Distributional balance score ≥7.0/10 (balanced negative/neutral/positive coverage)
✅ Self-referential system maintains sustainable operation at μ0 baseline

### Files

**Documentation**:
- `docs/ai_psychology_homeostasis_expansion.md` - Complete architecture with math
- `docs/distributional_balance_requirement.md` - Triad completeness framework
- `docs/tier2_prioritization_results.md` - Cost-benefit analysis results
- `docs/SESSION_SUMMARY_20251116.md` - Full session summary

**Scripts**:
- `scripts/score_tier2_concepts_revised.py` - Balanced scoring rubric implementation
- `scripts/run_simplex_agentic_review.py` - Agentic review for simplex identification (RUNNING)
- `scripts/generate_motivation_patch.py` - noun.motive expansion (Strategy 2)

**Results**:
- `results/tier2_scoring_revised/all_concepts_scored_revised.json` - 9,931 scored concepts
- `results/motivation_patches/motivation_patch_strategy2.json` - 4 new Layer 3 concepts
- `results/simplex_agentic_review.json` - Complete simplex mappings (PENDING)

---

## Future Work 🔮

### Planned Phases
- **Phase 7**: Accuracy Calibration Study (find optimal training scale)
- **Phase 9**: Relation-First Adaptive Scaling Analysis
- **Phase 12**: Applications (research, development, safety)

### Deferred
- **Phase 8**: Steering Vector Composition (blocked on Phase 7)

### Research Directions
- Spline geometry optimization (layer-specific, architecture-aware)
- Detached Jacobian approach for research validation
- Cross-model probe transfer studies
- Hybrid ontology co-definition (model + human semantics)

---

## Key Success Metrics

### Phase 2 Metrics (Complete) ✅
- Classifier success rate: 91.9% @ n=1000 (919/1000 concepts @ 100% test acc)
- Training efficiency: ~4-5 hours for 1000 concepts (single GPU)
- Minimal training works: 1 positive + 1 negative per concept sufficient

### Production Scale v1 (Complete) ✅
- Concepts trained: 5,583 (100% synset concept space coverage)
- Average F1: 95%+
- Training time: ~8 hours with adaptive scaling

### Steering Metrics ✅
- Detection confidence: 94.5% mean on OOD prompts
- Negative steering: Highly effective (-94% suppression)
- Positive steering: Variable, needs refinement

---

## Tech Stack

**Model**: Gemma-3-4b-pt (for generation and activation extraction)
**Framework**: PyTorch + PyTorch Lightning
**Storage**: HDF5 with gzip compression
**Concept Graph**: WordNet (117K synsets) + SUMO (684 classes)
**Training**: Binary cross-entropy per concept, adaptive scaling
**Steering**: Linear activation manipulation with manifold awareness
**Validation**: Out-of-distribution prompts + semantic field tracking
**UI**: OpenWebUI fork with real-time concept visualization

---

**Last Updated**: 2025-11-16
**Current Focus**: Three-Pole Simplex Architecture for interoceptive AI
**Status**: Simplex agentic review running (top 1000 concepts, ~$30, ~1.5 hours)
