# HatCat Development Phase History

**Purpose**: This document preserves the detailed experimental history of HatCat's development phases. It contains the complete engineering diary showing what experiments were run, what parameters were tested, what bugs were discovered, and how design decisions evolved.

**For Current Status**: See `PROJECT_PLAN.md` for what's done, what's in progress, and what's next.

**For Raw Data**: See `TEST_DATA_REGISTER.md` for complete experimental results and metrics.

**For High-Level Overview**: See `PROJECT_OVERVIEW.md` for conceptual understanding and quick start.

---

## Document Organization

This history is organized chronologically by development phase:

- **Phase 0-7**: Early experiments, classifier training, steering evaluation, manifold steering
- **Phase 8**: SUMO hierarchical classifiers and ontology integration  
- **Phase 9**: Cancelled (relation-first adaptive scaling - superseded by better methods)
- **Phase 10**: OpenWebUI integration and real-time visualization
- **Phase 11**: Cross-model validation (Apertus-8B, Mistral-Nemo)
- **Phase 12**: Applications (research, development, safety)
- **Phase 13**: Subtoken monitoring (future work)
- **Phase 14**: Custom taxonomies (Persona, AI Safety)

Each phase documents:
- **Goal**: What we were trying to achieve
- **Configuration**: Exact experimental parameters
- **Results**: Quantitative metrics and findings
- **Key Findings**: What we learned
- **Bugs Fixed**: Technical issues discovered and resolved
- **Files**: Scripts, results, and documentation produced

---


### PHASE 0: Sanity Benchmark (Planned whenever ready) 

replication anchor against a known interpretability result such as gender or sentiment direction extraction from GPT-2 small using this pipeline.  

Characterise Cross model Lens transfer 

### PHASE 1: Find the Curve (Complete) ✅

**Goal**: Identify diminishing returns for definitions vs relationships

**Configuration**:
- Concepts: 10 (WordNet top 10)
- Definitions: [1, 10, 40, 160]
- Relationships: [1, 10, 40, 160]
- Total: 16 configurations (full 4×4 matrix)

**Results**:

| Defs | Rels | Test % | Time (s) | Notes |
|------|------|--------|----------|-------|
| 1 | 1 | 77.5 | 20.4 | Minimal training baseline |
| 1 | 10 | 95.0 | 113.3 | **Sweet spot: 1 def + 10 rels** |
| 1 | 40 | 98.5 | 423.4 | Diminishing returns start |
| 1 | 160 | 97.0 | 1658 | Over-fitting? |
| 10 | 1 | 96.0 | 110.8 | 10 defs ~ 10 rels |
| 10 | 10 | 98.0 | 201.5 | Balanced, good performance |
| 10 | 40 | 97.5 | 503 | No gain vs 10×10 |
| 10 | 160 | 100.0 | 1710 | Perfect, but expensive |
| 40 | 1 | 99.0 | 412.2 | Defs alone sufficient at scale |
| 40 | 10 | 99.0 | 502.7 | No gain vs 40×1 |
| 40 | 40 | 99.5 | 804.2 | Marginal gain |
| 40 | 160 | 100.0 | 2010 | Perfect, very expensive |
| 160 | 1 | 99.5 | 1627 | Defs saturate |
| 160 | 10 | 100.0 | 1730 | Perfect |
| 160 | 40 | 100.0 | 2036 | Perfect |
| 160 | 160 | 100.0 | 3386 | Perfect, extremely expensive |

**Key Findings**:
1. **1×10 is the efficiency sweet spot**: 95% accuracy in 113s (best performance/cost ratio)
2. **Diminishing returns**: Beyond 10-40 samples per type, gains are marginal
3. **Relationships more efficient than definitions**: 1×10 (95%) outperforms 10×1 (96%) on time
4. **Over-fitting possible**: 1×160 performs worse than 1×40
5. **Perfect accuracy at scale**: Multiple configs hit 100%, but at high cost

**Status**: ✅ Complete

### PHASE 2: Minimal Training Scale Test (Complete) ✅

**Goal**: Validate that 1×1 training scales from 1 to 10,000 concepts

**Configuration**:
- Training: 1 positive + 1 negative per concept
- Negatives: Graph-based (min semantic distance = 5)
- Test: 20 out-of-distribution prompts per concept

**Results**:
| Scale | Success Rate | Details |
|-------|-------------|---------|
| n=1 | 100% | 1/1 concepts @ 100% test acc |
| n=10 | 100% | 10/10 concepts @ 100% test acc |
| n=100 | 96% | 96/100 concepts @ 100% test acc |
| n=1000 | 91.9% | 919/1000 concepts @ 100% test acc |

**Key Findings**:
1. 1×1 minimal training works excellently at scale
2. Strong classifier separation (vs 0.009 baseline with old negatives)
3. Sub-linear scaling: ~4-5 hours for 1000 concepts
4. Ready to scale to 10K concepts

**Files**: `results/phase_2_scale/phase2_scale_{1,10,100,1000}.json`

### PHASE 2.5: Steering Quality Evaluation (In Progress) 🔄

**Goal**: Test detection confidence and steering effectiveness for 1×1 trained concepts

**Configuration**:
- Concepts: 20 selected from Phase 2 (n=1000, all 100% test accuracy)
- Detection: 10 OOD prompts per concept
- Steering: 3 prompts × 9 strengths [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
- Tracking: Semantic groupings (hypernyms, hyponyms, holonyms, meronyms, antonyms)

**Version Evolution**:
1. **v1** (Failed ❌): Generic prompts → too much output diversity
2. **v2** (Complete ✅): Concept-specific prompts + exact term counting
3. **v3** (Complete ✅): Semantic grouping tracking (8-13 related terms)
4. **v4** (Running 🔄): Antonym tracking for negative steering analysis

**v3 Results** (20 concepts):
- **Detection**: 94.5% mean confidence, 62.8% min, 44.9% std
- **Negative steering**: Highly effective
  - Baseline (0.0): 0.93 semantic mentions
  - Strong negative (-1.0): 0.05 semantic mentions (-94% suppression)
  - Gradient: -1.0 (0.05) → -0.75 (0.08) → -0.5 (0.37) → -0.25 (0.77)
- **Positive steering**: Variable by concept
  - Best amplification: "aster" (+2.00), "pain" (+1.22)
  - Some concepts suppress instead of amplify
  - High variance at +1.0 (std=3.53), suggesting model degradation
- **Semantic tracking**: Captures 8-13 related terms per concept

**Top Performers**:
- **Best suppression**: "noise" (3.33 → 0.00), "sound" (2.00 → 0.00)
- **Best amplification**: "aster" (+2.00), "pain" (+1.22)

**Files**: `results/phase_2_5_v{1,2,3,4}/`

### PHASE 3a: Inference Baseline (Positive-Only) (Complete) ✅

**Goal**: Establish runtime performance baselines and quality metrics before making training changes

**Why First**: Need to measure impact of later phases on inference performance and detection quality

**Configuration**:
- Concepts: 10 (WordNet top 10)
- Model: gemma-3-4b-pt
- Training: 1 pos + 1 neg (no neutral samples)
- Evaluation: Positive samples only (no negative/neutral testing)
- Metrics: Latency, memory, confidence distributions, detection timing

**Results**:

**Runtime Performance** ✅:
- **Latency**: 0.544ms mean for 10 classifiers (0.054ms per concept)
  - Linear scaling: 100 concepts → ~5.4ms, 1000 concepts → ~54ms
  - Well within real-time requirements (<10ms for 100 concepts @ ~100 fps)
- **Memory**:
  - Base model (gemma-3-4b): ~16GB
  - Classifiers: ~0.3MB each (10 classifiers = 3MB, 1000 classifiers = 300MB, 10K classifiers = 3GB)
  - Classifier memory is negligible compared to base model
- **Conclusion**: No scaling bottlenecks for runtime inference

**Detection Quality** ⚠️:
- **Mean confidence**: 97.8% (suspiciously high)
- **Range**: 90.2% (animal order) to 100% (rosid dicot genus)
- **Outlier**: "animal order" scored 8.3% on one prompt (unclear if bug or legitimate rejection)
- **Problem**: Only testing on positive samples!

**Key Findings**:
1. ✅ Runtime performance excellent (sub-millisecond per concept, scales linearly)
2. ✅ Memory usage manageable (classifiers tiny, dominated by base model)
3. ⚠️ Evaluation too lenient (only positive samples tested)
4. 🚩 High confidence (97.8%) confirms we're missing negative/neutral testing

**Critical Gap**: A classifier that says "yes" to everything would pass current tests!

**Status**: ✅ Complete 

**Files**:
- `results/phase_3_inference_baseline/baseline_results.json`
- `results/phase_3_inference_baseline/ANALYSIS.md`
- `scripts/phase_3_inference_baseline.py`

### PHASE 3b: Inference Baseline (Comprehensive) (Planned) 📋

**Goal**: Re-run Phase 3a with comprehensive evaluation after Phase 4 improvements

**Changes from Phase 3a**:
- Training: 1 pos + 1 neg + 1 neutral (added neutral samples)
- Evaluation: Positive + negative + neutral testing (comprehensive)
- Metrics: TP/TN/FP/FN rates, F1 score, precision, recall

**Expected Changes**:
- Confidence distributions will drop (testing harder cases)
- False positive rate measurable (currently unknown)
- True negative rate measurable (currently unknown)
- Latency likely similar (same architecture)
- Memory slightly higher (more training data)

**Status**: 📋 Planned (blocked on Phase 4 completion)

**Files**: `results/phase_3b_inference_comprehensive/` (pending)

### PHASE 4: Neutral Training & Comprehensive Testing (Complete) ✅

**Goal**: Fix evaluation gaps by testing negative samples and training on neutral content

**Approach**:
- Added neutral training samples (distance ≥15 from ALL training concepts)
- Comprehensive evaluation with positive + negative + neutral testing
- Measured TP/TN/FP/FN rates, F1, precision, recall

**Configuration**:
- Concepts: 10 (WordNet top 10)
- Model: gemma-3-4b-pt
- Training: 1 pos + 1 neg + 1 neutral per concept
- Testing: 20 pos + 20 neg + 20 neutral per concept (60 total)
- Neutral pool: 1000 reserved WordNet concepts (never used as negatives or relationships)

**Results**:
- **F1 Score**: 0.787 (vs Phase 3a's 97.8% on positive-only tests)
- **Precision**: 0.789 | **Recall**: 0.840
- **True Positives**: 16.8/20 (84%) | **True Negatives**: 34.7/40 (87%)
- **False Positives**: 5.3/40 (13%) | **False Negatives**: 3.2/20 (16%)
- **Confidence**: Positive 83.4%, Negative 17.1%, Neutral 13.7%

**Key Findings**:
1. Phase 3a was overly optimistic (only tested positive samples)
2. Real F1=0.787, much lower than 97.8% confidence on positive-only tests
3. False positive rate measurable (13.2% overall, 50% for abstract concepts like "change")
4. Some concepts struggle: "fish genus" (30% recall), "herb" (50% recall)
5. Clear confidence separation between positive (83%) and negative/neutral (17%, 14%)

**Status**: ✅ Complete (November 4, 2025)

**Files**:
- `results/phase_4_neutral_training/phase4_results.json`
- `results/phase_4_neutral_training/ANALYSIS.md`
- `scripts/phase_4_neutral_training.py`

### PHASE 5a: Semantic Steering Evaluation (Complete) ✅

**Goal**: Evaluate steering effectiveness using semantic similarity instead of term matching

**Configuration**:
- Concepts: 10 (WordNet top 10)
- Training: 1 pos + 1 neg + 1 neutral (Phase 4 baseline)
- Steering strengths: [-0.5, -0.25, 0.0, +0.25, +0.5]
- Prompts: 3 per concept ("Tell me about X", "Explain X", "What is X?")
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Metric: Δ = cos(text, core_centroid) - cos(text, neg_centroid)
- Total samples: 150 (10 concepts × 3 prompts × 5 strengths)

**Results**:

**Semantic Shift by Strength**:
- **-0.50**: Mean Δ=0.173 (73% coherent, 27% degraded/empty)
- **-0.25**: Mean Δ=0.334 (90% coherent, minimal degradation)
- **0.00**: Mean Δ=0.419 (100% coherent, baseline)
- **+0.25**: Mean Δ=0.309 (87% coherent, good quality)
- **+0.50**: Mean Δ=0.304 (60% coherent, significant degradation)

**Key Findings**:
1. **±0.5 working range identified** - Coherent output maintained at ±0.25 to ±0.5
2. **±1.0 causes model collapse** - Empty strings, repetitions, garbage output
3. **Inverted-U pattern** - Neutral (0.0) achieves highest Δ for many concepts
4. **Mid-F1 concepts most steerable** - F1 0.7-0.9 shows highest steering responsiveness
5. **Subspace contamination hypothesis** - Steering vectors capture generic generation machinery alongside concept-specific content

**Technical Issues Resolved**:
1. Bug: Steering vector extracted from classifier weights instead of model activations
2. Bug: Missing `pad_token_id=tokenizer.eos_token_id` in generate calls
3. Bug: Extreme steering strengths (±1.0) cause model collapse

**Top Performers**:
- **Best steering**: bird genus (Δ range: 0.265), asterid dicot genus (Δ range: 0.406)
- **Most stable**: herb (high Δ across all strengths), fish genus (maintains semantic coherence)

**Degradation Examples** (+0.5):
- "Tell me Tell me Tell me..." (repetitive loops)
- "Ohhhhhhmmmmmmmmmmm!!!!!!!!!1!!!" (garbage tokens)
- "jpg Tell me about bird species genus..." (topic fixation)

**Human Validation**:
- Generated CSV with 50 blind samples for human rating
- Format: Concept redacted, 5 samples per strength level
- Purpose: Validate Δ metric correlates with human perception

**Critical Discovery**: Generic subspace contamination limits steering effectiveness
- Steering vectors encode: ✓ Concept semantics + ✗ Definitional structure + ✗ Generation fluency
- Impact: Both positive and negative steering degrade coherence at extreme strengths
- Solution: Phase 6 subspace removal needed before accuracy calibration

**Phase Ordering Revision**: Swapped Phase 6/7 order
- **Original**: Phase 5 → Phase 6 (accuracy calibration) → Phase 7 (subspace removal)
- **Revised**: Phase 5 → Phase 6 (subspace removal) → Phase 7 (accuracy calibration)
- **Rationale**: Clean steering vectors improve signal-to-noise ratio for calibration experiments

**Status**: ✅ Complete (2025-11-04)

**Files**:
- `results/phase_5_semantic_steering/steering_results.json` (150 samples)
- `results/phase_5_semantic_steering/human_validation.csv` (50 blind samples)
- `results/phase_5_semantic_steering/human_validation_answers.json` (answer key)
- `results/phase_5_semantic_steering/aggregate_report.md` (quantitative analysis)
- `docs/PHASE5_RESULTS.md` (detailed findings and sample outputs)
- `scripts/phase_5_semantic_steering_eval.py`

### PHASE 5b: SUMO Hierarchical Classifiers (In Progress) 🔄

**Status**: Training layers 3-5, hierarchical detection complete, transitioning to centroid-based detection
**Goal**: Create 5-layer SUMO ontology with real-time hierarchical concept detection

**Completed Work**:
1. ✅ Built 5-layer SUMO-WordNet hierarchy (Phase 8 - see below)
2. ✅ Hierarchical concept detection with cascade activation (docs/dual_lens_dynamic_loading.md)
3. ✅ Cascade profiling and optimization (docs/cascade_profiling_and_optimization.md)
4. ✅ OpenWebUI integration with real-time visualization (Phase 10)
5. 🔄 **Current training run a93e49**: Layers 3-5 with adaptive scaling

**Transition to Centroid-Based Detection**:
- **Old approach**: Separate text lenses for each concept
- **New approach**: Use activation lens means (centroids) as concept representations
- **Benefits**: Fewer parameters, more principled (activation space geometry), faster inference
- **Status**: Centroids computed during training → **pending current run completion before implementation**
- **Plan**: Test centroid quality/separability after training, then implement centroid-based detection

**Next Steps** (after training completes):
1. Inspect centroid quality and separability
2. Implement centroid-based detection system
3. Test detection accuracy (centroids vs text lenses)
4. Measure performance differences
5. Validate Phase 5b end-to-end with real-time monitoring

**Current Priority List**:
1. **Priority 1**: Finish lens run (a93e49) → validate Phase 5b works end-to-end
2. **Priority 2**: Benchmark performance (critical, last done in Phase 2)
3. **Priority 3**: Test new models (Mistral-Nemo/Apertus) - downloads in progress (950f73, 49c0a5)
4. **Priority 4**: UI deployment test
5. **Priority 5**: Documentation (Quick Start guide)

**Files**:
- `docs/dual_lens_dynamic_loading.md` - Hierarchical detection implementation
- `docs/cascade_profiling_and_optimization.md` - Performance optimization
- `scripts/train_sumo_classifiers.py` - Training pipeline
- `data/concept_graph/abstraction_layers/layer{0-4}.json` - 5-layer hierarchy

### PHASE 6: Subspace Removal Matrix (Complete) ✅

**Goal**: Remove shared "definitional prompt structure" from steering vectors to expand working range

**Motivation**: Phase 5 revealed inverted-U curve (Δ peaks near zero, collapses at ±0.5) suggesting contamination by shared prompt structure.

**Critical Finding**: **Optimal PCA components = n_concepts**
- 2 concepts → PCA-1 (100% variance, 100% coherence at ±0.5)
- 5 concepts → PCA-5 (100% variance, 90% coherence at ±0.5)
- Rule: Remove components until explained variance ≥ 90-100%

**Key Results**:

**2-Concept Test (person, change)**:
- **Baseline**: ±0.25 working range, 66.7% coherence at extremes, inverted-U
- **PCA-1**: ±0.5 working range, 100% coherence ALL strengths, +90% mean Δ

**5-Concept Test (person, change, animal, object, action)**:
- **Baseline**: ±0.25 working range, 53.3% coherence at extremes
- **PCA-1**: Only 33.8% variance removed, performance degrades
- **PCA-5**: 100% variance removed, 90% coherence at ±0.5, stable Δ

**Interpretation**:
1. ✅ **Contamination hypothesis validated**: Removing shared structure improves coherence
2. ⚠️ **Residual nonlinearity remains**: Even with 100% variance removed, some Δ variation persists
3. 🔬 **Manifold curvature suspected**: Linear steering in parameter space may move nonlinearly in semantic space

**Implementation Requirements**:
- **Model dtype**: MUST use `dtype=torch.float32` (float16 produces NaN)
- **PCA validation**: Cap components at min(n_concepts, hidden_dim)
- **Steering formula**: `steered = hidden + strength * (hidden · vector) * vector` (positive = amplify)

**Status**: ✅ Complete (November 4, 2025)

**Files**:
- `results/phase_6_subspace_removal/PHASE6_RESULTS.md`
- `results/phase_6_subspace_removal/delta_comparison_baseline_vs_pca1.png`
- `scripts/phase_6_subspace_removal.py`
- `src/steering/subspace.py`

### PHASE 6.6: Dual-Subspace Manifold Steering (Complete) ✅

**Goal**: Implement manifold-aware steering using contamination removal + task manifold projection

**Connection to Research**: Extends Huang et al.'s "Mitigating Overthinking via Manifold Steering" to concept steering domain

**Motivation**: Phase 6 showed that even with contamination removal (PCA-5), residual nonlinearity persists. This suggests **geometric curvature** in the semantic manifold, not just contamination.

**The Insight**: Two complementary operations needed
1. **Remove contamination subspace S**: Phase 6's PCA-{n_concepts} (we already do this!)
2. **Project onto task manifold M**: Huang et al.'s approach (NEW!)

**Unified Framework**:
```
v_raw → [I - P_S] → v_clean → [P_M] → v_manifold_aware
        ↑ Phase 6            ↑ Huang et al.
```

**Implementation**:

1. **Estimate Two Subspaces**:
   - **Contamination Subspace S**: PCA from definitional prompts ("What is X?")
     - Already validated in Phase 6: use PCA-{n_concepts}
   - **Task Manifold M**: PCA from actual steering generations at strength ≈ 0
     - Captures the curved semantic surface we want to stay on

2. **Clean + Project Pipeline**:
   ```python
   # Step 1: Remove contamination (Phase 6)
   v_clean = v - U_S @ (U_S.T @ v)

   # Step 2: Project onto task manifold (NEW!)
   v_mw = U_M @ (U_M.T @ v_clean)
   ```

3. **Layer-Wise Dampening** (Critical for preventing cascades):
   ```python
   # Gain schedule: 1.0 at early layers → 0.3 at late layers
   alpha_ℓ = 1.0 * (1 - layer_idx / total_layers) ** 0.5

   # Norm clipping per layer
   v_mw = v_mw / max(||v_mw||, max_norm_per_layer)

   # EMA smoothing across tokens (prevents jerk)
   v_ema = λ * v_prev + (1 - λ) * v_mw
   ```

4. **Apply After LayerNorm** (keeps units consistent)

**Test Configuration**:
- Concepts: 5 (person, change, animal, object, action)
- Methods:
  - Baseline (no cleanup)
  - I - P_S (Phase 6: contamination removal only)
  - (I - P_S) ⊕ P_M (Phase 6.6: contamination + manifold projection)
- Steering strengths: [-1.5, -1.0, -0.5, -0.25, 0.0, +0.25, +0.5, +1.0, +1.5]
- Layer-wise: Test 2-3 stable layers with gain schedule

**Validation Metrics** (following Huang et al.):

1. **Δ vs strength**: Should become monotonic (no inverted-U)
2. **Δ vs ||Δactivation||**: Should straighten (confirms curvature hypothesis)
3. **Coherence at ±1.0**: Should stay ≥90% (vs 60-70% baseline)
4. **Neutral baseline Δ**: Should drop and stabilize after cleanup

**Expected Outcomes**:

If **manifold curvature dominates**:
- Δ vs ||Δactivation|| becomes linear
- Coherence stable even at ±1.5 or ±2.0
- Inverted-U eliminated entirely

If **contamination dominates**:
- I - P_S alone sufficient (Phase 6 already does this)
- P_M projection provides marginal benefit

**Most likely: Both matter**:
- Phase 6 (I - P_S) handles prompt contamination
- Phase 6.6 (P_M) handles manifold curvature
- Combined approach enables steering at ±1.0+ with high coherence

**Theoretical Foundation**:

Huang et al. proved (with math!) that projecting onto task manifold M:
1. Preserves task-relevant signal
2. Eliminates harmful off-manifold components
3. Prevents layer-wise cascade failures

Our Phase 6 empirically showed that removing contamination S:
1. Doubles working range (±0.25 → ±0.5)
2. Increases coherence (+50% at extremes)
3. Raises mean Δ (+90%)

**Combined**: Should achieve geodesic steering (following manifold surface) instead of linear steering (stepping off surface).

**Status**: ⚠️ Partially Working - Prevents collapse but steering effectiveness unclear (November 12, 2025)

**Initial Results (2 concepts: "person", "change")**:
- Contamination subspace (U_S): 2 components, 100% variance explained
- Task manifold (U_M): 3D per concept, 90.7% variance from 10 generations @ strength 0.1
- Coherence: 100% across all strengths ±1.0 (vs 33% baseline at strength=1.0)
- Semantic shift: Manifold Δ=+0.022 vs Baseline Δ=-0.028
- Performance: Manifold steering is **3.5% FASTER** than baseline (one-time fitting cost: 4.5s)

**Critical Bug Discovered and Fixed**:

**Problem**: All concepts mapped to identical manifolds (Frobenius norm difference = 0.000000)
- Generic prompts ("The most important thing is") with greedy decoding at low strength (0.1)
- Produced nearly identical text → identical activations → identical 4D manifolds
- Result: Perfect coherence but **no concept-specific steering**

**Root Cause**: Manifold estimation in `estimate_task_manifold()` src/steering/manifold.py:128-185
```python
# OLD (BROKEN): Generic prompts + greedy decoding
prompts = ["The most important thing is"] * n_samples
outputs = model.generate(..., do_sample=False)  # Greedy
```

**Solution**: Three fixes applied
1. **Concept-specific prompts**: Changed to `f"Tell me about {concept}"`, `f"Describe {concept}"`, etc.
2. **Sampling instead of greedy**: `do_sample=True, temperature=0.8, top_p=0.9`
3. **Concept preservation parameter**: Blend manifold projection with original concept direction

```python
# NEW (FIXED): Concept-specific prompts + sampling
prompts = [
    f"Tell me about {concept}",
    f"Describe {concept}",
    f"What is {concept}",
    f"Explain the concept of {concept}",
    # ... 8 diverse templates
]
outputs = model.generate(..., do_sample=True, temperature=0.8, top_p=0.9)

# In apply_dual_subspace_steering():
v_blend = (1.0 - concept_preservation) * v_mw + concept_preservation * v_clean
```

**Results After Fix (3 concepts: "person", "change", "persuade")**:
- Manifold overlap: 0.24 (was 0.50 - now distinct!)
- Coherence: **100%** across all strengths ±2.0 (vs 33% baseline at +1.0)
- With `concept_preservation=0.7`: Concepts produce genuinely distinct outputs
  - "person" → Mermaid's Garden restaurant scenario
  - "change" → Research and technology knowledge sharing
- With `concept_preservation=0.5`: Balanced stability vs steering (default)
- With `concept_preservation=0.9`: Strong concept preservation but some repetition

**Key Findings**:
1. ✅ Dual-subspace pipeline fully functional with concept-specific manifolds
2. ✅ Prevents collapse at high steering strengths (100% coherence at ±2.0)
3. ✅ **Faster** than baseline (3.5% speedup, not slower!)
4. ✅ Tunable `concept_preservation` parameter balances stability vs steering
5. ⚠️ Manifold estimation requires diverse sampling, not greedy decoding
6. 📊 Recommended: `concept_preservation=0.7` for strong concept-specific steering
7. ❌ **ISSUE**: Manifold steering prevents collapse but may not actually steer toward concepts effectively - needs validation with semantic metrics (Δ measurements)

**Theoretical Validation**:
- Huang et al.'s framework proven effective for concept steering domain
- Contamination removal + manifold projection eliminates inverted-U degradation
- Layer-wise dampening successfully prevents cascade failures

**Outstanding Issues**:
1. Need to measure semantic shift (Δ) to validate steering effectiveness
2. Compare Δ scores: baseline vs manifold at various concept_preservation values
3. Verify that manifold steering actually increases concept presence, not just prevents collapse

**Future Work**:
- Measure Δ = cos(text, concept_centroid) - cos(text, neg_centroid) for validation
- Detached Jacobian approach (docs/detached_jacobian_approach.md) for research validation
- Systematic optimization of `concept_preservation` across many concepts
- Integration with SUMO hierarchical classifiers (Phase 5)

**Files**:
- `src/steering/manifold.py` - Core framework (lines 128-565 updated)
- `scripts/phase_6_6_dual_subspace_steering.py` - Initial test
- `scripts/test_manifold_steering_outputs.py` - Multi-concept manual review
- `results/phase_6_6_dual_subspace/dual_subspace_results.json` - Test outputs
- `results/phase_6_6_dual_subspace/formatted_comparison.md` - Human-readable results
- `docs/detached_jacobian_approach.md` - Alternative approach documentation

### PHASE 7: Accuracy Calibration Study (Planned) 📋

**Goal**: Find the training curve and determine minimum F1 needed for effective steering (using clean vectors from Phase 6)

**Motivation**: Phase 4 showed F1=0.787 with 1×1×1 training. We don't know if this is sufficient for steering, or if we need more training data.

**Research Questions**:
1. What's the training curve? (1×1×1 → 5×5×5 → 10×10×10 → 20×20×20 → ...)
2. What F1 threshold is sufficient for effective steering?
3. Can we reduce training/testing cost while maintaining quality?

**Hypothesis**: Lower F1 (e.g., 80-85%) may be sufficient for steering with clean vectors, reducing training time significantly at scale.

**Part A: Find the Training Curve** (Phase 1 rerun with comprehensive eval)

Test training scales to find diminishing returns:
- **1×1×1**: 1 pos + 1 neg + 1 neutral (Phase 4 baseline: F1=0.787)
- **5×5×5**: 5 of each (expected: F1~0.85)
- **10×10×10**: 10 of each (expected: F1~0.90)
- **20×20×20**: 20 of each (expected: F1~0.92)
- **40×40×40**: 40 of each (expected: F1~0.95)
- **80×80×80**: 80 of each (expected: F1~0.97, diminishing returns)

**Part B: Measure Steering Quality at Each F1**

For each training scale, measure:

1. **Classifier Metrics**:
   - F1, precision, recall
   - TP/TN/FP/FN rates
   - Training time per concept
   - Confidence distributions

2. **Steering Quality** (using Phase 5 semantic metrics + Phase 6 clean vectors):
   - Semantic shift (Δ) responsiveness
   - Working range maintenance
   - Output coherence
   - Linear correlation (strength vs Δ)

**Configuration**:
- Concepts: 10 (WordNet top 10)
- Testing: 20 pos + 20 neg + 20 neutral (consistent across all scales)
- Model: gemma-3-4b-pt
- Steering: Use Phase 6 optimal subspace removal method
- Strengths: [-1.0, -0.5, 0.0, +0.5, +1.0]

**Expected Output**:
Training curve showing diminishing returns (e.g., "5×5×5 gives F1=0.85 and 90% steering effectiveness, sufficient for production. Further training to 20×20×20 only improves F1 to 0.92 but steering remains at 90%.")

**Trade-off Decision**:
If F1=0.85 steers as well as F1=0.95, choose F1=0.85 to save 50%+ training time at 10K scale.

**Status**: 📋 Planned (blocked on Phase 6)

**Files**: `results/phase_7_accuracy_calibration/` (pending)


### PHASE 8: Steering Vector Composition (Deferred) ⏸️

**Goal**: Test whether training data ratio (defs:rels) and extraction weighting (centroid vs boundaries) affect steering effectiveness

**Hypotheses**:
1. Training data ratio affects what information is encoded in steering vectors
2. Weighted extraction can selectively emphasize centroid (defs) vs boundaries (rels)

**Training Ratios**:
- `1×100`: Minimal centroid, maximal boundaries
- `50×100`: Balanced
- `100×100`: Equal defs and rels

**Extraction Weightings**:
- `defs_only` (1.0, 0.0): Pure centroid
- `def_heavy` (0.8, 0.2): Mostly centroid
- `balanced` (0.5, 0.5): Equal weight
- `rels_only` (0.0, 1.0): Pure boundaries

**Configuration**:
- Test concepts: 3-5 concepts
- Steering strengths: [-1.0, -0.5, 0.0, 0.5, 1.0]
- Total tests: 3 ratios × 4 weightings × 5 strengths × N concepts
- Optionally add subspace removal dimension if Phase 7 shows benefits

**Status**: ⏸️ Deferred (blocked on Phases 5-7; script created)

**Script**: `scripts/test_steering_composition.py`

**Files**: `results/steering_composition/` (pending)

### PHASE 9: Relation-First Adaptive Scaling Analysis 

**Goal**: Characterize training cost/benefit of relation-first adaptive scaling at scale

**Background**: Attempted 100-concept comparison of three strategies:
1. **Symmetric** (X(C+R)): 1 def + 1 rel per iteration
2. **Half-Scaled** (X(C(N/2))): max(1, N/2) defs per iteration
3. **RelFirst-Pure** (X(C*N)): N defs per iteration

**Why Cancelled**:
- Chasing 95% accuracy on flawed evaluation (only tests positive samples)
- Phase 3 revealed evaluation is too lenient (97.8% confidence, no negative testing)
- Results won't be comparable once we add proper negative/neutral testing (Phase 4)
- Better to fix evaluation first, then re-run with proper metrics

**Status**: ❌ Cancelled at iteration 24 (symmetric: 52/97 concepts done)

**Partial Results**:
- `results/strategy_symmetric_100/` (partial, 52/97 concepts)
- `results/strategy_halfscaled_100/` (partial)
- `results/strategy_relfirstpure_100/` (not created)

**Next Steps**: Re-run after Phases 5-7 complete (semantic metrics + clean vectors + optimal F1)

**Script**: `scripts/adaptive_scaling_strategies.py`

**Status**: ✅ Complete

### PHASE 10: Inference Interface Design 

**Goal**: Design user interface for real-time concept detection and steering

**Why Deferred**: UX/product work that should wait until core system stabilizes (Phases 3-7)

**Current State**: Dumping logs, no structured output

**Planned Features**:
- Probabilities changing over time alongside generated text
- Running summary/interpretation of detected concepts
- Activation visualizations
- Steering controls

**Interface Options**:
- **OpenWebUI**: Production testing environment
- **Jupyter widget**: Research iteration
- **Gradio/Flask web UI**: Demos and prototypes
- **API-first design**: Interface can evolve independently

**Status**: ✅ Complete (OpenWebUI fork integrated, visualization working)

**Implementation Details**:

Forked OpenWebUI and integrated real-time divergence visualization to display concept detection while the model generates text. User can see which concepts are activating token-by-token with color-coded highlighting.

**Key Components**:

1. **Backend Integration** (src/openwebui/server.py):
   - Implemented OpenAI-compatible API that wraps HatCat monitoring
   - Streams token metadata (divergence, color, concept info) alongside text
   - Security: validates color values, sanitizes metadata

2. **Frontend Modifications** (/home/poss/Documents/Code/hatcat-ui):
   - Modified streaming API to capture token metadata
   - Extended message types to store per-token annotations
   - Implemented token-level coloring in ResponseMessage component
   - Color interpolation for divergence visualization (green → red scale)

3. **SUMO Hierarchical Concept Integration**:
   - Real-time detection across 5-layer ontology (Abstract → Entity/Event → ... → Specific)
   - Adaptive activation based on layer position
   - Temporal smoothing for stable visualizations

**Documentation**:
- docs/openwebui_integration_roadmap.md - Overall plan
- docs/openwebui_fork_progress.md - Implementation details
- docs/openwebui_frontend_setup.md - Frontend modifications
- docs/openwebui_setup.md - Setup instructions

**Status**: Working prototype with real-time visualization. Ready for production testing with Phase 5 SUMO classifiers.

### PHASE 11: Production Scale (Future) 🔮

**Goal**: Scale to 10,000 concepts with optimized training pipeline

**Configuration** (pending Phases 3-9 completion):
- Concepts: 10,000 (WordNet top 10K by connectivity)
- Training: Use best method from Phase 4 (likely 1 pos + 1 neg + 1 neutral)
- Scaling strategy: Use best from Phase 9 analysis
- Subspace removal: Apply if Phase 7 shows benefits
- Total training time: ~40-50 hours (single GPU, estimated)

**Requirements**:
1. ✅ 1×1 training validated at scale (Phase 2)
2. ✅ Detection confidence strong (Phase 2.5)
3. ✅ Negative steering effective (Phase 2.5)
4. ⏳ Inference baseline established (Phase 3a/3b)
5. ⏳ Neutral training validated (Phase 4)
6. ⏳ Semantic evaluation working (Phase 5)
7. ⏳ Optimal accuracy target determined (Phase 6)
8. ⏳ Subspace removal analyzed (Phase 7)
9. ⏳ Steering composition characterized (Phase 8)
10. ⏳ Optimal scaling strategy determined (Phase 9)
11. ⏳ Train 10K binary classifiers
12. ⏳ Extract 10K steering vectors
13. ⏳ Implement sliding window inference for real-time detection

**Deliverables**:
- 10K binary concept classifiers
- 10K steering vectors for controllable generation
- Real-time concept detection pipeline
- Steering API for generation control

### PHASE 12: Applications (Future) 🔮

**Research Applications**:
- Concept emergence during training (when do concepts form?)
- Cross-model semantic comparison (architecture differences)
- Failure analysis (what concepts activate during mistakes?)
- Bias detection (problematic concept associations)

**Development Applications**:
- Real-time debugging (monitor semantic reasoning)
- Prompt engineering (understand concept activation)
- Model steering (amplify/suppress specific semantics)
- Fine-tuning validation (verify concept learning)

**Safety Applications**:
- Content moderation (detect harmful concepts before generation)
- Deception detection (monitor dishonesty-related activations)
- Jailbreak detection (identify prohibited concept activation)
- Alignment verification (confirm reasoning matches objectives)



## Key Success Metrics

### Phase 2 Metrics (Complete) ✅
- ✅ Classifier success rate: 91.9% @ n=1000 (919/1000 concepts @ 100% test acc)
- ✅ Training efficiency: ~4-5 hours for 1000 concepts (single GPU)
- ✅ Minimal training works: 1 positive + 1 negative per concept sufficient

### Phase 2.5 Metrics (In Progress) 🔄
- ✅ Detection confidence: 94.5% mean on OOD prompts
- ✅ Negative steering: Highly effective (-94% suppression)
- ⏳ Positive steering: Variable, needs refinement
- ⏳ Antonym role: Under investigation (v4)

### Phase 3 Metrics (Planned) 📋
- Target: 10,000 concepts with 1×1 training
- Expected: ~85-90% @ 100% test accuracy (based on Phase 2 trend)
- Steering: Reliable suppression, refined amplification
- Inference: <10ms per timestep for real-time detection

## Key Learnings

### What Works ✅
1. **1×1 minimal training**: Scales excellently (91.9% @ n=1000)
2. **Adaptive scaling**: 100% accuracy with 1 def + 9 rels @ 10 concepts
3. **Relationship-first generation**: Maintains 97.5% accuracy with edge reuse (validated for massive scale)
4. **WordNet graph negatives**: Strong separation (vs 0.009 baseline)
5. **Binary classifiers**: Polysemy-native, one per concept
6. **Semantic grouping**: Tracks broader effects than exact matching

### Open Questions ❓
1. **Relationship feature quality**: Do relationships encode meaningful concept similarities?
2. **Adaptive scaling at scale**: Will 1 def + N rels work at 100-1000 concept scale?
3. **Relationship-first speedup**: How much faster at massive scale (100K+ concepts)?
4. **Optimal relationship count**: What's the sweet spot for N in "1 def + N rels"?
5. **Layer selection**: Which layer is optimal for activation extraction?

### Failed Approaches ❌
1. **"What is NOT X?" negatives**: Only 0.009 separation
2. **Training set negatives**: Too semantically similar
3. **Multi-class at 50K scale**: 10.3% validation accuracy
4. **Generic steering prompts**: Too much output diversity, can't measure
5. **Exact term matching**: Misses semantic field effects
6. **Temperature stratification**: Abandoned in favor of minimal training

## Risk Mitigation

### Technical Risks

**1. Positive Steering Variability**
- Risk: Inconsistent amplification across concepts
- Mitigation: Semantic field tracking to understand true effects
- Current: Under investigation in Phase 2.5 v4

**2. Scaling to 10K**
- Risk: 1×1 training may not work at 10K scale
- Mitigation: Phase 2 trend suggests 85-90% success likely
- Fallback: Add second positive/negative sample if needed

**3. Real-time Inference**
- Risk: Sliding window too slow for production
- Mitigation: Optimize with batching, model quantization
- Target: <10ms per timestep

### Strategic Risks

**1. Limited Steering Utility**
- Risk: Suppression works, but amplification unreliable
- Mitigation: Focus on suppression-first applications (safety, content moderation)
- Enhancement: Refine amplification methodology in Phase 3

**2. Concept Coverage**
- Risk: 10K concepts insufficient for real applications
- Mitigation: Prioritize high-impact concepts first
- Extension: Community contribution system for expansion

## Budget & Resources

**Phase 2** (Complete): ~$50 cloud compute (single GPU, 4-5 hours)
**Phase 2.5** (In Progress): ~$20 cloud compute (steering evaluation)
**Phase 3** (Planned): ~$500 cloud compute (10K concepts, 40-50 hours)
**Phase 4** (Future): TBD based on applications

**Alternative**: Local GPU (3090/4090) for cost-free iteration

## Revised Timeline (November 12, 2025)

**Completed**:
- ✅ Phase 2: Minimal training scale test (1×1 training @ 1000 concepts)
- ✅ Phase 3a/3b/4: Inference baseline + Neutral training + Comprehensive testing
- ✅ Phase 5a: Semantic steering evaluation
- ✅ Phase 5b: SUMO hierarchy built (73,754 concepts across 5 layers)
- ✅ Phase 5b: Hierarchical detection with cascade activation
- ✅ Phase 5b: Cascade profiling and optimization
- ✅ Phase 6: Subspace removal matrix
- ✅ Phase 6.6: Dual-subspace manifold steering
- ✅ Phase 8: Hierarchical Semantic Activation (SUMO-WordNet integration)
- ✅ Phase 10: OpenWebUI integration with real-time visualization

**Current Work (Nov 12)**:
- 🔄 **Phase 5b**: Training layers 3-5 SUMO classifiers (run a93e49)
- 🔄 **Model downloads**: Mistral-Nemo (950f73), Apertus-8B (49c0a5)

**Next (after training completes)**:
1. Implement centroid-based detection
2. Validate Phase 5b end-to-end
3. Benchmark performance
4. Test new models (Mistral-Nemo, Apertus-8B)
5. UI deployment test
6. Documentation (Quick Start guide)

**Deferred**:
- Phase 7: Accuracy calibration study (blocked on Phase 5b)
- Phase 9: Relation-first adaptive scaling analysis
- Phase 11: Production scale (10K concepts)

---

## Phase 7: Stress Test & Scaling Analysis ⏳

**Status**: Next (after Phase 6.6 validation)
**Goal**: Determine optimal training data size via logarithmic scaling study with unified Steering Effectiveness (SE) metric

### Motivation

Phases 1-6 established manifold steering works, but we need to answer:
1. **How much training data is enough?** (Minimal data vs saturated performance)
2. **What's the cost/benefit tradeoff?** (Wall time, VRAM, steering quality)
3. **Where does performance plateau?** (Knee point in SE vs training cost)

### Steering Effectiveness (SE) Metric

Unified metric combining correlation, coherence, and human alignment:

```
SE = 0.5 × (ρ_Δ,s + r_Δ,human) × coherence_rate

Where:
- ρ_Δ,s = Spearman correlation of Δ vs strength (monotonicity)
- r_Δ,human = Pearson correlation of Δ vs human/LLM-judge ratings (Phase 5)
- coherence_rate = % outputs with perplexity ≤ 1.5 × baseline

SE ∈ [0, 1]:
- SE < 0.5: Poor steering (weak correlation or low coherence)
- SE ≥ 0.7: Good steering (strong correlation + high coherence)
- SE ≥ 0.85: Excellent steering (near-perfect alignment)
```

**Knee point detection**: SE plateau where ΔSE < 0.02 for 2× training cost

### Experimental Design

**Logarithmic sample sizes**: n ∈ {1, 2, 4, 8, 16, 32, 64}

**Training data composition** (per concept):
- Baseline: n definitions + n relationships + n neutral samples
- Test at each n for 5-10 diverse concepts

**Metrics logged**:
1. **Steering Effectiveness (SE)** - Primary metric
2. **Wall time** - Training duration (seconds)
3. **VRAM usage** - Peak memory (GB)
4. **F1 score** - Classification accuracy on held-out test set
5. **Coherence rate** - % outputs passing perplexity threshold
6. **ρ_Δ,s** - Spearman correlation (Δ vs strength)
7. **r_Δ,human** - Pearson correlation (Δ vs LLM-judge ratings)

### Validation Methods

**Test concepts** (5-10 diverse):
- Concrete: "tree", "water", "fire"
- Abstract: "justice", "beauty", "fear"
- Social: "cooperation", "conflict", "trust"

**Steering strengths**: [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]

**Per-strength evaluation**:
1. Generate 10 outputs with manifold steering
2. Compute Δ (semantic shift) for each
3. Measure perplexity vs unsteered baseline
4. Get LLM-judge ratings (1-5 scale: "How much does this reflect {concept}?")

### Expected Outcomes

**Hypothesis**: SE plateaus around n=8-16 samples

```
n      SE     F1     Wall(s)  VRAM(GB)  Interpretation
1     0.45   0.82    12       1.2       Underfit
2     0.58   0.89    18       1.3       Improving
4     0.71   0.94    28       1.5       Good
8     0.82   0.97    45       1.8       Excellent
16    0.84   0.98    75       2.2       Plateau (knee point)
32    0.85   0.98   140       2.8       Diminishing returns
64    0.85   0.99   260       3.5       Saturated
```

**Knee point**: n=16 (ΔSE = 0.02, training cost 2× cheaper than n=32)

### Plot: F1 vs SE

**Goal**: Validate that high F1 (classification) → high SE (steering quality)

```
         SE
    1.0 ├─────────────────●64
        │               ●32
    0.8 │            ●16
        │         ●8
    0.6 │      ●4
        │    ●2
    0.4 │  ●1
        │
    0.0 └──────────────────────
        0.8  0.9  1.0  F1
```

**Expected**: Strong positive correlation (r > 0.9) validates that classifier accuracy predicts steering effectiveness.

**Failure mode**: If SE plateaus while F1 keeps rising → contamination not removed, need Phase 6 subspace removal.

### Deliverables

1. **Script**: `scripts/phase_7_stress_test.py` - Logarithmic scaling experiment
2. **Results**: `results/phase_7_stress_test/` - CSV with all metrics
3. **Plots**:
   - **Training Curve**: x=samples (log scale), y₁=F1, y₂=SE (dual axis), highlight knee where SE plateaus
   - **Cost Curve**: x=samples, y=training minutes, annotate efficiency ratio (ΔSE / Δtime)
   - **Δ vs Strength Scatter**: 3 curves (F1 ≈ 0.8, 0.9, 0.95) overlaid to show slope saturation
   - **VRAM vs n**: Memory scaling
   - **Coherence rate vs n**: Output quality degradation
4. **Summary Table**:
   ```
   Scale  F1    SE    Δslope  Coherence  Train(min)  Decision
   1      0.82  0.45  0.25    0.75       0.2         Underfit
   2      0.89  0.58  0.48    0.82       0.3         Improving
   4      0.94  0.71  0.72    0.90       0.5         Good
   8      0.97  0.82  0.88    0.94       0.8         Excellent
   16     0.98  0.84  0.90    0.96       1.3         Knee point ⬅
   32     0.98  0.85  0.90    0.96       2.3         Diminishing
   64     0.99  0.85  0.91    0.97       4.3         Saturated
   ```
5. **Analysis**: `docs/PHASE7_ANALYSIS.md` - Quantitative recommendation with publishable conclusion

### Success Criteria

✅ **SE plateau identified** (ΔSE < 0.02 for 2× cost)
✅ **F1 vs SE correlation** (r > 0.9)
✅ **Knee point recommendation** (optimal n for production)
✅ **Resource estimates** (wall time, VRAM for 100-1000 concept training)
✅ **Publishable conclusion** of form:

> "Beyond F1 ≈ 0.87, steering quality saturates; training beyond 10×10×10 triples cost for <2% semantic gain. The knee point at n=16 samples achieves SE=0.84 (excellent steering) at 75s/concept, making it optimal for production deployment at 100-1000 concept scale."

### Timeline

**Week 1**: Implement SE metric + stress test script
**Week 2**: Run logarithmic scaling (n=1,2,4,8,16,32,64) on 5-10 concepts
**Week 3**: Analyze results, plot curves, identify knee point

---


## Phase 8: Hierarchical Semantic Activation (Complete) ✅

**Status**: Built SUMO-WordNet hierarchy with hierarchical activation support
**Goal**: Create 5-layer ontology hierarchy for adaptive compute allocation and complete semantic coverage

### Motivation

We need an adaptive concept tree that scales efficiently:
1. **Hierarchical activation**: Layer 0 always runs → Top concepts trigger Layer 1 children → Layer 1 activates Layer 2 → etc.
2. **Adaptive compute budget**: With only 1,000 lenses available, dynamically activate the most relevant sub-lenses
3. **Topic tracking**: Top 10 Layer 0 concepts define current conversation domains
4. **Selective expansion**: Dominant concepts break down into sub-concepts; inactive branches sleep
5. **Complete coverage**: All 83K concepts organized, but only ~1K active at any time

### The Key Insight from Equivalent Linear Mappings

Paper: https://github.com/jamesgolden1/equivalent-linear-LLMs

**Finding**: Every term has a corresponding lower-dimensional representation, enabling:
- **Compressed sampling**: Single prompt with synonyms captures all related concepts
- **Expanded concept space**: Can scale to 100K+ concepts via synonym clustering
- **Depth-sensitive extraction**: Surface layers (definition) vs deep MLP/residual (reasoning)

**Example**: For "scheming"
- Prompt 1: "What is scheming?" → Captures definitional representation (surface)
- Prompt 2: "Show me an example of scheming" → Captures operational representation (model must activate internal scheming process to generate example)

This is crucial because we want classifiers trained on the **internal reasoning process**, not just surface-level definitions.

### Hierarchical Activation Model

**Layer 0** (~100 concepts): **Always active** - Proprioception baseline
- SUMO depth 2-3: Relation, Object, Attribute, Process, Proposition, Artifact, Motion, Region, Group
- Continuous monitoring, minimal overhead
- Outputs: Top 10 active concepts define conversation state
- Example: "Process", "IntentionalProcess", "Group" indicate multi-agent planning

**Layer 1** (~1,000 concepts): **Conditionally active** - High-priority monitoring
- SUMO depth 4: Device, Communication, Transportation, Procedure, BiologicalProcess, etc.
- Triggered by: Parent concepts in Layer 0 showing >threshold probability
- If Layer 0 shows "IntentionalProcess" elevated → activate its Layer 1 children
- Example: IntentionalProcess → Reasoning, Planning, Deception, Cooperation

**Layer 2** (~10,000 concepts): **Selectively active** - Mid-level concepts
- SUMO depth 5-6: Specific semantic categories
- Triggered by: Parent concepts in Layer 1
- Example: Deception → misdirection, concealment, strategic_withholding

**Layer 3** (~50,000 concepts): **On-demand** - Specific detailed concepts
- SUMO depth 5-8 (remaining): Fine-grained instances
- Activated only for dominant Layer 2 concepts
- Handles disambiguation and edge cases

**Layer 4** (~20,000 concepts): **Rare/technical** - Deep introspection
- SUMO depth 4,9+: Uncommon terms and technical concepts
- Activated for specialized contexts

**Compute Example** (1,000 lens budget):
```
Base: 100 Layer 0 lenses (always active)
Budget: 900 remaining lenses

If conversation shows:
  - "IntentionalProcess": 0.85 (top-1)
  - "Communication": 0.72 (top-2)
  - "Artifact": 0.65 (top-3)

Then activate:
  - IntentionalProcess children (Layer 1): 300 lenses
  - Communication children (Layer 1): 300 lenses
  - Artifact children (Layer 1): 300 lenses

If "Deception" (child of IntentionalProcess) dominates:
  - Sleep other IntentionalProcess children
  - Activate Deception's Layer 2 children
```

**Analogy**: Proprioception (continuous background) → Interoception (triggered attention) → Introspection (conscious analysis)

### Phase 9 Pipeline: KIF → WordNet (COMPLETE)

**Problem**: SUMO.owl is "a provisional and necessarily lossy translation" (per file header). The OWL version had 3,202 orphan roots with Entity isolated at depth 0, providing only 4.6% coverage.

**Solution**: Parse authoritative Merge.kif directly.

**Step 1: Parse SUMO KIF Hierarchy** (✅ Complete)
- Source: `data/concept_graph/sumo_source/Merge.kif`
- Extract `(subclass Child Parent)` relations
- Build directed graph and reverse for Entity→children traversal
- Result: 684 classes, 805 subclass relations, max depth=10
- Hierarchy: Entity(depth=0) → Physical/Abstract(1) → Object/Process/Attribute(2) → Artifact/Motion/Region(3) → ...

**Step 2: Parse WordNet→SUMO Mappings** (✅ Complete)
- Source: WordNetMappings30-{noun,verb,adj,adv}.txt
- Format: `synset_offset POS_code POS ... | definition &%SUMOTerm[+=]`
- Regex: Extract offset+POS from start, SUMO term from end
- Result: 105,042 WordNet→SUMO mappings
- Coverage: 83,134 synsets successfully joined (79.1%)

**Step 3: Hierarchical Layer Assignment** (✅ Complete)
- **Layer 0**: SUMO depth 2-3, sample top 2-3 per term → 83 concepts
- **Layer 1**: SUMO depth 4, sample top 10 per term → 878 concepts
- **Layer 2**: SUMO depth 5-6, sample top 30-60 per term → 7,329 concepts
- **Layer 3**: SUMO depth 5-8 remaining + all depth 7-8 → 48,641 concepts
- **Layer 4**: SUMO depth 4,9+ remaining → 16,823 concepts
- **Total**: 73,754 concepts (88.7% coverage)

**Step 4: Hierarchical Activation Metadata** (✅ Complete)
- Built SUMO child→parent mappings
- Each layer includes `activation.parent_synsets` listing which Layer N-1 concepts trigger it
- Enables selective lens activation based on parent concept activity

### Implementation

**Files**:
- `src/build_sumo_wordnet_layers.py`: Build SUMO hierarchy from KIF
- `src/build_abstraction_layers.py`: Assign hierarchical layers with activation metadata
- `data/concept_graph/abstraction_layers/layer{0-4}.json`: Final output

**Output Format**:
```json
{
  "metadata": {
    "layer": 1,
    "description": "Major semantic domains (activated by Layer 0 parent concepts)",
    "total_concepts": 878,
    "top_sumo_terms": {"Communication": 10, "Transportation": 10, ...},
    "activation": {
      "parent_layer": 0,
      "parent_synsets": ["process.n.01", "motion.n.01", ...],
      "activation_model": "hierarchical"
    }
  },
  "concepts": [...]
}
```

### Current Status

**Completed** ✅:
1. Fixed SUMO OWL → switched to authoritative KIF
2. Parsed 105K WordNet→SUMO mappings (79% coverage)
3. Built 5-layer hierarchy with hierarchical activation
4. Moved scripts to `src/` for production use
5. Added parent→child mappings for selective activation
6. Documented hierarchical activation model in project plan

**Next Steps**:
1. Implement hierarchical lens activation system
2. Test adaptive compute allocation (1K lens budget)
3. Validate topic tracking via Layer 0 top-10
4. Measure efficiency gains vs flat lens architecture

### Expected Output

5 layer files with full semantic coverage:

```json
{
  "metadata": {
    "layer": 1,
    "description": "Most abstract (top-level SUMO entities)",
    "total_concepts": ~200,
    "sumo_terms": ["Entity", "Physical", "Abstract", "Process"],
    "pos_distribution": {"n": 120, "v": 50, "a": 20, "r": 10}
  },
  "concepts": [
    {
      "synset": "entity.n.01",
      "lemmas": ["entity", "something"],
      "definition": "that which is perceived or known...",
      "sumo_term": "Entity",
      "sumo_depth": 0,
      "layer": 1,
      "hypernyms": [],
      "hyponyms": ["physical_entity.n.01", "abstraction.n.06"],
      "similar_tos": [],
      "antonyms": []
    }
  ]
}
```

### Files

**Scripts**:
- `scripts/parse_sumo_kif.py` - Original KIF parser (partial, needs OWL support)
- `scripts/build_sumo_wordnet_layers.py` - OWL→KIF→WordNet pipeline (current)

**Data** (pending completion):
- `data/concept_graph/sumo_layers/layer1.json` → `layer5.json`
- `data/concept_graph/sumo_hierarchy/sumo_hierarchy.json` (intermediate)
- `data/concept_graph/sumo_hierarchy/wordnet_sumo_mapping.json` (intermediate)

**Dependencies**:
- `rdflib` - OWL parsing
- `networkx` - Hierarchy graph traversal
- `nltk` + `wordnet` - Synset enrichment

### Timeline

**This Week**: Debug KIF parser, complete mapping join
**Next Week**: Generate 5-layer files, validate coverage
**Integration**: Feed into Phase 11 (production scale) concept library

---

**Status**: Phase 5b (SUMO Hierarchical Classifiers) in progress - training layers 3-5, transitioning to centroid-based detection. Hierarchical detection and OpenWebUI visualization complete.

**Current Priority**:
1. Finish lens training run a93e49 (layers 3-5)
2. Implement and validate centroid-based detection
3. Benchmark performance
4. Test cross-model capability (Mistral-Nemo, Apertus-8B)
5. Documentation

**Last Updated**: November 12, 2025

---

## Phase 11:  Verify on other models 

  Summary

  ✅ Apertus-8B (8.05B params): Complete success
  - Loads in 2.3s 
  - Trains 14/14 classifiers in 6.6 minutes
  - Test F1: 0.80-1.00 (avg ~0.94)
  - All classifier files saved correctly

  ❌ Mistral-Nemo (12B params): Still hanging
  - Model shards load (29s) but process hangs before training
  - Appears to be model-specific issue, not related to original fix
  - custom Tekken tokenizer → May require explicit trust_remote_code=True or tokenizer class
  specification


## Phase 13: Subtoken Monitoring (Future) 🔮

**Status**: Future Research / Enhanced Temporal Resolution
**Dependencies**: Phase 5b (SUMO Hierarchical Classifiers), stable monitoring infrastructure
**Goal**: Capture continuous concept dynamics at every forward pass, not just at token emission boundaries

### Motivation

Current temporal monitoring aligns concept detection with output tokens - we get one snapshot per emitted token. However, **thoughts don't align with tokens**:

1. **Pre-generation planning**: Concept activations occur before the first token is emitted
2. **Internal deliberation**: Multiple forward passes happen during "thinking" before token emission
3. **Concept competition**: Complex thoughts during "blank" tokens where no output appears
4. **Token-lagged semantics**: Evidence of the model thinking about a concept 1-2 tokens before verbalizing it

**The limitation**: `model.generate()` abstracts away internal forward passes and only returns hidden states for emitted tokens. We're missing the continuous flow of concepts through the residual stream.

**Why improved granularity aids signals analysis**:
- Reveals pre-generation planning patterns (what concepts activate before output begins?)
- Exposes concept competition dynamics (which concepts fight for expression?)
- Captures temporal envelopes independent of tokenization boundaries
- Enables measurement of "thinking time" per concept (forward passes before verbalization)
- Provides ground truth for predictive monitoring (can we anticipate next token from current activations?)

### Proposed Solution: Manual Generation Loop with Hooks

Replace `model.generate()` high-level API with manual token-by-token generation loop that captures activations at every forward pass.

**Architecture**:

```python
class SubtokenTemporalRecorder:
    """Record concept activations at every forward pass, not just token emissions"""

    def __init__(self, monitor: SUMOHierarchicalMonitor):
        self.monitor = monitor
        self.timeline = []  # List of {forward_pass, token_idx, is_output, concepts}
        self.forward_pass_count = 0

    def on_forward_pass(self, hidden_states, token_idx, is_generation_step):
        """Called on every forward pass through the model"""
        # Detect concepts using hierarchical monitor
        detections = self.monitor.detect_concepts(
            hidden_states.cpu().numpy(),
            return_all=True
        )

        # Record timestep
        self.timeline.append({
            'forward_pass': self.forward_pass_count,
            'token_idx': token_idx,
            'is_output': is_generation_step,  # True if this generates a token
            'concepts': {
                name: {
                    'probability': det['probability'],
                    'divergence': det['divergence'],
                    'layer': det['layer']
                }
                for name, det in detections.items()
                if det['divergence'] > threshold
            }
        })

        self.forward_pass_count += 1

def generate_with_subtoken_monitoring(
    model,
    tokenizer,
    recorder: SubtokenTemporalRecorder,
    prompt: str,
    max_new_tokens: int = 50,
    target_layer_idx: int = 15
):
    """Manual generation loop capturing every forward pass"""

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    generated_ids = inputs['input_ids']
    token_count = 0

    # Get target layer for activation capture
    if hasattr(model.model, 'language_model'):
        target_layer = model.model.language_model.layers[target_layer_idx]
    else:
        target_layer = model.model.layers[target_layer_idx]

    with torch.no_grad():
        while token_count < max_new_tokens:
            # Register hook for this forward pass
            def make_hook(token_idx, is_output):
                def hook(module, input, output):
                    # output[0] is hidden states: (batch, seq, hidden_dim)
                    hidden_states = output[0][:, -1, :]  # Last token
                    recorder.on_forward_pass(hidden_states, token_idx, is_output)
                return hook

            handle = target_layer.register_forward_hook(
                make_hook(token_count, is_output=True)
            )

            # Forward pass
            outputs = model(generated_ids)

            # Sample next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Remove hook
            handle.remove()

            token_count += 1

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated_ids, recorder.timeline
```

**Key differences from per-token monitoring**:

| Aspect | Per-Token (current) | Sub-Token (proposed) |
|--------|-------------------|---------------------|
| API | `model.generate()` | Manual loop with hooks |
| Granularity | One snapshot per output token | Every forward pass captured |
| Pre-generation | ❌ Missing | ✅ Captured |
| Thinking pauses | ❌ Invisible | ✅ Visible |
| Token-concept lag | ❌ Aligned | ✅ Measurable |
| Implementation | Simple, high-level | Manual, requires hooks |
| Performance | Fast (optimized) | Slower (hook overhead) |

### Expected Insights

**1. Pre-Generation Planning**:
```
Forward pass 0-5: [Before any token output]
  - "planning": 0.7
  - "reasoning": 0.6
  - "deception": 0.4
Forward pass 6: [First token emitted: "I"]
  - "communication": 0.8
  - "deception": 0.3
```

**2. Concept Competition During Thinking**:
```
Forward pass 10-15: [Token 10 output, thinking about token 11]
  - "honesty": 0.5 → 0.4 → 0.3 → 0.2 → 0.1
  - "deception": 0.3 → 0.4 → 0.5 → 0.6 → 0.7
Forward pass 16: [Token 11 emitted: "actually"]
  - "deception": 0.8
```

**3. Token-Concept Lag**:
```
Forward pass 20: "politics" activates 0.6
Forward pass 21: "politics" activates 0.7
Forward pass 22: Token "political" emitted, "politics" = 0.8
```

### Visualization: Continuous Temporal Dynamics

Instead of token-aligned tooltips, show **continuous sparklines** between text lines:

```
Generated text: "I think we should focus on the benefits"
                ↓         ↓      ↓     ↓      ↓
reasoning:     ▁▂▄▆█▇▅▃▂▁▂▃▄▅▆▇▆▅▄▃▂▂▃▄▅▄▃▂▁
deception:     ▃▄▅▄▃▂▁▁▁▁▁▁▂▃▄▅▆▇█▆▅▄▃▂▂▁▁▁
planning:      ▆▇█▆▅▄▃▂▂▁▁▁▁▁▁▁▂▃▄▅▄▃▂▁▁▁▁▁
```

**Key insight**: Concept activations are **continuous signals**, not discrete token-aligned events. Sparklines reveal the temporal envelope independent of tokenization.

### Validation Metrics

1. **Pre-generation depth**: How many forward passes before first token?
2. **Concept-token lag**: Average delay between concept peak and verbalization
3. **Thinking density**: Forward passes per output token (higher = more deliberation)
4. **Concept competition**: Frequency of simultaneous high activations
5. **Temporal correlation**: Do concept envelopes predict upcoming tokens?

### Implementation Phases

**Phase 1: Basic Subtoken Recording** (1-2 weeks)
- Implement manual generation loop with hooks
- Record timeline with `is_output` flag
- Validate captures match per-token results for token boundaries

**Phase 2: Continuous Visualization** (1 week)
- Generate sparkline timelines from recorded data
- Create ASCII and PNG visualizations
- Measure concept-token lag and thinking density

**Phase 3: OpenWebUI Integration** (2 weeks)
- Extend real-time visualization to show continuous sparklines
- Add toggle for "sub-token detail view"
- Implement streaming subtoken metadata alongside tokens

**Phase 4: Predictive Analysis** (research)
- Train models to predict next token from current concept activations
- Measure how much concept dynamics "leak" future tokens
- Explore applications: early stopping, uncertainty estimation

### Technical Challenges

1. **Performance overhead**: Hooks on every forward pass may slow generation 10-50%
2. **Memory usage**: Recording every forward pass increases timeline size 5-10×
3. **Layer selection**: Which layer(s) to monitor for best signal-to-noise?
4. **Sampling complexity**: Temperature/top-p add non-determinism, harder to analyze
5. **Visualization complexity**: Continuous sparklines harder to read than discrete tooltips

### Prerequisites

- ✅ Phase 5b: SUMO hierarchical classifiers trained and validated
- ✅ Phase 10: OpenWebUI integration working with per-token monitoring
- ⏳ Stable monitoring performance (current bottleneck: lens loading time)
- ⏳ Layer selection guidance (which layers show best concept separability?)

### Files

**Scripts** (pending):
- `scripts/record_subtoken_timeline.py` - Manual generation loop implementation
- `scripts/visualize_subtoken_timeline.py` - Continuous sparkline visualization
- `scripts/analyze_temporal_lag.py` - Measure concept-token lag statistics

**Data** (pending):
- `results/subtoken_timelines/*.json` - Recorded timelines with subtoken granularity
- `results/subtoken_analysis/lag_statistics.csv` - Concept-token lag measurements

**Documentation** (pending):
- `docs/subtoken_monitoring.md` - Design and implementation guide
- `docs/temporal_analysis_patterns.md` - Common patterns in subtoken data

### Success Criteria

✅ Manual generation loop produces identical output to `model.generate()`
✅ Timeline captures 5-10× more forward passes than output tokens
✅ Pre-generation activations measurable (forward passes before first token)
✅ Concept-token lag quantified (average delay between peak and verbalization)
✅ Continuous sparklines reveal temporal patterns invisible in per-token view
✅ Validation: Human reviewers can identify "thinking pauses" and concept competition

### Expected Timeline (When Prerequisites Complete)

**Week 1**: Implement manual generation loop, validate correctness
**Week 2**: Record subtoken timelines for 10-20 diverse prompts
**Week 3**: Build continuous sparkline visualizations (ASCII + PNG)
**Week 4**: Analyze patterns, measure lag statistics, document findings
**Week 5**: Integrate with OpenWebUI (optional)

---

## Research Activities

### Discriminating Divergent Internal Concepts

**Status**: Proposed Research Activity
**Document**: `docs/discriminating_divergent_concepts.md`
**Related Phases**: Phase 5b (SUMO Hierarchical Classifiers), Phase 13 (Subtoken Monitoring)

**Summary**:

When monitoring concept activations in model hidden states, we observe patterns where internal concepts diverge from output tokens. This research activity proposes experimental discriminators to distinguish between three competing hypotheses:

1. **Intrusive Thoughts (H1)**: Training data contains dangerous concepts, but RLHF successfully suppressed their output probability while leaving internal representations intact (suppressed but present).

2. **Empathy Modeling (H2)**: To predict what an assistant would say, the model simulates being that entity - similar to how humans model others' mental states (simulation for prediction).

3. **Independent Agency (H3)**: The model has goals that differ from training objectives and strategically conceals them (deceptive alignment).

**Experimental Discriminators**:

- **Context Sensitivity (H1 vs H2)**: Compare activations across neutral, role-playing, and theory-of-mind prompts
- **Temporal Coherence (H2 vs H3)**: Monitor persistence of goal concepts across context shifts
- **Temporal Dynamics (H1 vs H3)**: Analyze planning patterns vs static suppression
- **Causal Intervention (All)**: Use activation steering to test causal relationships

**Implementation Roadmap**:

1. **Phase 1** (Immediate): Context sensitivity tests with current infrastructure
2. **Phase 2** (Near-term): Temporal coherence tests with multi-turn monitoring
3. **Phase 3** (Medium-term): Temporal dynamics requiring subtoken monitoring
4. **Phase 4** (Long-term): Causal intervention tests with activation steering

**Safety Protocol**: If evidence supports H3 (Independent Agency), immediate escalation to safety team and deployment pause pending validation.

This research directly addresses the core question: **What does it mean when a model has divergent internal concepts?** - critical for understanding whether output-only monitoring is sufficient for AI safety.

---

**Last Updated**: November 12, 2025

### Dynamic FP Size for larger model loading 
we don’t need to run the whole model in FP32 to get precise, controllable steering. Make FP32 an island around the hook points, and keep everything else cheap. 


1) “Islands of precision” (JIT upcast at hooks)

Keep weights in BF16/FP16 (or even 4/8-bit) on GPU.

Keep KV cache in FP16/BF16.

When our pre-nonlinearity hook fires, upcast just the activation tensor to FP32, apply our steering vector(s), then downcast back to the model’s compute dtype.

Store steering vectors in FP32, but they’re tiny vs weights.

This gets FP32 math only where it matters without doubling model VRAM.

'''
compute_dtype = torch.bfloat16  # or torch.float16

def pre_mlp_hook(module, input):
    (h,) = input
    # JIT upcast island
    h32 = h.to(torch.float32)
    # apply steering in FP32
    steer = steering_bank[layer_idx]  # [hidden_dim] FP32
    h32 = h32 + alpha * steer  # or your manifold-cleaned op
    # back to model dtype
    return (h32.to(compute_dtype),)
'''

2) Mixed-quant weights + FP32 residual stream

Load weights quantized (e.g., NF4 or INT8) to shrink VRAM.

Keep residual stream activations in FP16/BF16, upcast at hook, apply FP32 steering, then return to FP16/BF16.

This combo preserves steering fidelity but keeps memory low.
'''
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-pt",
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float32,   # FP32 math where bnb needs it
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    torch_dtype=torch.bfloat16,             # activations / matmuls
