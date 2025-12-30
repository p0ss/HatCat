# Test Data Register

## Overview

This register tracks all experimental runs for the temporal semantic decoder project. Each entry documents configuration, results, and learnings.

## Active Experiments

### PHASE 3a: Inference Baseline (Positive-Only) (2025-11-04) ✅

**Goal**: Establish runtime performance baselines and detection quality metrics before making training changes

**Configuration**:
- Concepts: 10 (WordNet top 10)
- Model: gemma-3-4b-pt
- Training: 1 pos + 1 neg (no neutral samples)
- Evaluation: Positive samples only (no negative/neutral testing)
- Metrics: Latency, memory, confidence distributions, detection timing

**Results**:
- **Latency**: 0.544ms mean for 10 classifiers (0.054ms per concept)
  - Scales linearly: 100 concepts → ~5.4ms, 1000 concepts → ~54ms
  - Well within real-time requirements
- **Memory**: Classifiers use ~0.3MB each (negligible vs 16GB base model)
  - 10K classifiers → ~3GB total (manageable)
- **Confidence**: 97.8% mean (suspiciously high, only testing positives)
  - Range: 90.2% to 100% across concepts
  - "animal order" had one outlier at 8.3%

**Key Findings**:
1. ✅ Runtime performance is excellent (sub-millisecond per concept)
2. ⚠️ Evaluation is too lenient (only tests positive samples)
3. 🚩 Need negative and neutral testing to validate quality

**Status**: ✅ Complete (will re-run as Phase 3b after Phase 4)

**Files**:
- `results/phase_3_inference_baseline/baseline_results.json`
- `results/phase_3_inference_baseline/ANALYSIS.md`

---

### PHASE 4: Neutral Training & Comprehensive Testing (2025-11-04) ✅

**Goal**: Add neutral samples to training and comprehensive evaluation with TP/TN/FP/FN metrics

**Configuration**:
- Concepts: 10 (WordNet top 10)
- Model: gemma-3-4b-pt
- Training: 1 pos + 1 neg + 1 neutral (added neutral samples)
- Evaluation: 20 pos + 20 neg + 20 neutral (comprehensive testing)
- Neutral pool: 1000 concepts (distance ≥15 from ALL training concepts)

**Results**:
- **F1**: 0.787 (vs Phase 3a's 97.8% confidence on positive-only tests)
- **Precision**: 0.789 (79% of predicted positives are correct)
- **Recall**: 0.840 (84% of actual positives detected)
- **True Positives**: 16.8/20 (84% detection rate)
- **True Negatives**: 34.7/40 (87% rejection rate)
- **False Positives**: 5.3/40 (13% false alarm rate)
- **False Negatives**: 3.2/20 (16% miss rate)

**Confidence Distributions**:
- Positive: 83.4% ± 23.3% (clear separation from negatives)
- Negative: 17.1% ± 10.2% (low, as desired)
- Neutral: 13.7% ± 15.9% (similar to negative, expected)

**Key Findings**:
1. ✅ Comprehensive evaluation works - more realistic than Phase 3a
2. ⚠️ False positive rate measurable (13.2% overall, 50% for "change")
3. ⚠️ Low recall on some concepts ("fish genus" 30%, "herb" 50%)
4. 🎯 Clear confidence separation (83% pos vs 17%/14% neg/neutral)
5. ✅ Neutral pool strategy effective (1000 concepts, distance ≥15)

**Status**: ✅ Complete

**Files**:
- `results/phase_4_neutral_training/phase4_results.json`
- `results/phase_4_neutral_training/ANALYSIS.md`

---

### PHASE 3b: Inference Baseline (Comprehensive) (Pending) 📋

**Goal**: Re-run Phase 3a with comprehensive evaluation (same as Phase 4 but measure latency/memory)

**Changes from Phase 3a**:
- Training: 1 pos + 1 neg + 1 neutral (added neutral samples)
- Evaluation: Positive + negative + neutral testing (comprehensive)
- Metrics: Latency, memory, TP/TN/FP/FN rates, F1 score

**Expected Results** (based on Phase 4):
- F1: ~0.787 (vs Phase 3a's 97.8% on positive-only)
- Latency: ~0.05ms per concept (same architecture)
- Memory: ~0.3MB per classifier (negligible)
- False positive rate: ~13%
- True negative rate: ~87%

**Status**: 📋 Planned (Phase 4 complete, can now run comprehensive baseline)

**Files**: `results/phase_3b_inference_comprehensive/` (pending)

---

### PHASE 8: Adaptive Scaling Strategy Comparison (2025-11-04) ❌

**Goal**: Test three adaptive scaling strategies to find optimal balance between centroid (definitions) and boundaries (relationships)

**Status**: ❌ Cancelled

**Rationale**: Chasing 95% accuracy on flawed evaluation (only tests positives). Once we add negative/neutral testing in Phase 4, these results won't be comparable. Deferring until proper evaluation is in place (Phases 4-5).

**Strategies tested (partial)**:
1. **Symmetric** (X(C+R)): 52/97 concepts done at iteration 24
2. **Half-Scaled** (X(C(N/2))): Terminated
3. **RelFirst-Pure** (X(C*N)): Not started

**Files**:
- `results/strategy_symmetric_100/` (partial)
- `results/strategy_halfscaled_100/` (partial)
- `results/strategy_relfirstpure_100/` (not created)

---

### PHASE 6: Subspace Removal for Steering Quality (2025-11-05) ✅

**Goal**: Remove contamination from concept vectors via PCA to improve steering coherence at high strengths

**Configuration**:
- Concepts: 2 ("person", "change"), then 5 ("person", "change", "animal", "object", "action")
- Model: gemma-3-4b-pt
- Subspace removal methods: None (baseline), Mean subtraction, PCA-1, PCA-2, PCA-5
- Steering strengths: [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
- Prompts: 3 per concept × 7 strengths = 21 outputs per method

**Results (2 concepts)**:
- **Baseline (none)**: Working range ±0.25, coherence 46-93%, inverted-U curve
- **Mean subtraction**: Working range ±0.5, coherence 86-100%, moderate improvement
- **PCA-1**: Working range ±0.5, coherence 100%, **eliminates inverted-U**
- Key finding: PCA-1 explains 100% variance with 2 concepts → optimal = n_concepts

**Results (5 concepts)**:
- **Baseline (none)**: Similar issues, coherence drops at extremes
- **Mean subtraction**: Partial improvement
- **PCA-1**: Only 33.8% variance explained (insufficient with 5 concepts)
- **PCA-5**: Expected to achieve 100% variance removal
- **Key finding**: Optimal PCA components = n_concepts (scales with concept diversity)

**Key Findings**:
1. ✅ **Contamination hypothesis confirmed**: Shared definitional structure exists
2. ✅ **PCA-{n_concepts} optimal**: Removes exactly the contamination subspace
3. ✅ **Coherence improvement**: 100% at ±0.5 (vs baseline collapse)
4. ✅ **Inverted-U eliminated**: Linear Δ vs strength relationship
5. 🎯 **Scales with concepts**: Need PCA-2 for 2 concepts, PCA-5 for 5 concepts

**Status**: ✅ Complete

**Files**:
- `results/phase_6_subspace_removal/subspace_removal_results.json`
- `docs/STAGE0_RESULTS.md` - Full analysis

---

### PHASE 6.6: Dual-Subspace Manifold Steering (2025-11-05) ✅

**Goal**: Combine contamination removal (Phase 6) + task manifold projection (Huang et al.) for stable high-strength steering

**Configuration**:
- Concepts: 2 ("person", "change")
- Model: gemma-3-4b-pt
- Methods: Baseline steering vs Manifold steering
- Manifold samples: 10 generations at strength 0.1
- Steering strengths: [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
- Layer-wise dampening: sqrt(1 - layer_depth)
- Max norm per layer: 1.0

**Pipeline**:
1. **Contamination removal**: v_clean = v - U_S @ (U_S.T @ v)
2. **Manifold projection**: v_mw = U_M @ (U_M.T @ v_clean)
3. **Layer-wise dampening**: v_mw *= sqrt(1 - layer_depth)
4. **Norm clipping**: Prevent explosions at high strengths

**Results**:
- **Contamination subspace (U_S)**: 2 components, 100% variance explained
- **Task manifold (U_M)**: 3D subspace per concept (90.7% variance)
- **Baseline coherence**: 100% (simple 2-concept case, not challenging)
- **Manifold coherence**: 100% (framework working correctly)
- **Semantic shift**: Baseline Δ=-0.028, Manifold Δ=+0.022

**Key Findings**:
1. ✅ **Framework implemented**: Full dual-subspace pipeline working
2. ✅ **Contamination subspace estimation**: PCA from concept vectors
3. ✅ **Task manifold estimation**: PCA from low-strength (0.1) steered generations
4. ✅ **Layer-wise dampening**: Prevents cascade failures in deep layers
5. ⏳ **Need challenging validation**: 2 concepts too simple, need Phase 7 stress test

**Status**: ✅ Implementation complete, ready for Phase 7 validation

**Files**:
- `src/steering/manifold.py` - Core framework
- `scripts/phase_6_6_dual_subspace.py` - Test script
- `results/phase_6_6_dual_subspace/dual_subspace_results.json`

---

### PHASE 7: Steering Vector Composition Test (2025-11-04) 📋

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
- Test concepts: 3-5 concepts (TBD)
- Steering strengths: [-1.0, -0.5, 0.0, 0.5, 1.0]
- Total tests: 3 ratios × 4 weightings × 5 strengths × N concepts

**Status**: 📋 Planned (script created: `scripts/test_steering_composition.py`)

**Files**: `results/steering_composition/` (pending)

---

### PHASE 2: Minimal Training Scale Test (2025-11-03) ✅

**Goal**: Validate that 1×1 minimal training (1 positive, 1 negative) scales from 1 to 10,000 concepts

**Configuration**:
- Training: 1 positive + 1 negative per concept
- Model: gemma-3-4b-pt
- Negatives: Graph-based (min distance=5 from ALL WordNet)
- Test: 20 OOD prompts per concept

**Results by Scale**:
- n=1: 100% (1/1 concepts @ 100% test accuracy)
- n=10: 100% (10/10 concepts @ 100% test accuracy)
- n=100: 96% (96/100 concepts @ 100% test accuracy)
- n=1000: 91.9% (919/1000 concepts @ 100% test accuracy)

**Key Findings**:
- 1×1 minimal training works excellently at scale
- 919/1000 concepts achieved perfect test accuracy
- Classifier separation strong (vs previous 0.009 baseline)
- Scales sub-linearly: ~4-5 hours for 1000 concepts

**Status**: ✅ Complete

**Files**:
- `results/phase_2_scale/phase2_scale_1.json`
- `results/phase_2_scale/phase2_scale_10.json`
- `results/phase_2_scale/phase2_scale_100.json`
- `results/phase_2_scale/phase2_scale_1000.json`

---

### PHASE 2.5: Steering Quality Evaluation (2025-11-03) 🔄

**Goal**: Test detection confidence and steering effectiveness for concepts trained with 1×1 minimal training

**Configuration**:
- Concepts: 20 selected from Phase 2 (n=1000, all with 100% test acc)
- Model: gemma-3-4b-pt
- Detection: 10 OOD prompts per concept
- Steering: 3 prompts × 9 strengths [-1.0 to +1.0]
- Tracking: Semantic groupings (hypernyms, hyponyms, holonyms, meronyms, antonyms)

**Version Evolution**:

**v1**: Generic steering prompts
- Issue: Too much output diversity, can't measure steering effect
- Prompts: "The most important thing to know is..."
- Status: ❌ Methodology failed

**v2**: Concept-specific prompts + mention counting
- Fix: Include concept in prompt to constrain generation
- Prompts: "Tell me about {concept}."
- Tracking: Exact concept mentions
- Results: 20 concepts tested, clear suppression visible
- Status: ✅ Complete

**v3**: Semantic grouping tracking
- Enhancement: Track hypernyms, hyponyms, holonyms, meronyms (8-13 terms per concept)
- Granular strengths: 9 levels from -1.0 to +1.0 (added -0.75, -0.25, +0.75, +0.25)
- Results (20 concepts):
  - Detection: 94.5% mean confidence, 62.8% min, 44.9% std
  - Suppression: 0.93 baseline → 0.05 at -1.0 strength (-94%)
  - Amplification: Variable (0 to +2.00 mentions depending on concept)
  - Semantic tracking: Captures 8-13 related terms per concept
- Top suppression: "noise" (3.33 → 0.00), "sound" (2.00 → 0.00)
- Top amplification: "aster" (+2.00), "pain" (+1.22)
- Status: ✅ Complete

**v4**: Antonym tracking for negative steering
- Enhancement: Track antonyms separately to test if negative steering amplifies opposites
- Script: Updated with antonym extraction and counting
- Status: 🔄 Running

**Key Findings**:
- **Detection works**: 94.5% mean confidence on OOD prompts
- **Negative steering highly effective**: Strong suppression (-94% semantic mentions)
- **Positive steering variable**: Some concepts amplify, others suppress
- **Semantic tracking essential**: Exact term matching misses broader steering effects
- **Concept-specific prompts required**: Generic prompts have too much diversity

**Status**: 🔄 v3 Complete, v4 Running

**Files**:
- `results/phase_2_5/` - v1 results (methodology failed)
- `results/phase_2_5_v2/` - v2 results (concept-specific prompts)
- `results/phase_2_5_v3/` - v3 results (semantic grouping tracking)
- `results/phase_2_5_v4/` - v4 results (antonym tracking, in progress)

---

## Historical Experiments

### EXP-001: Initial Quick Comparison (2025-11-02) ✅

**Goal**: Compare relational vs definitional prompts at equal compute

**Configuration**:
- Concepts: 10 (WordNet top 10)
- Model: gemma-3-4b-pt
- Negatives: Graph-based (min distance=5)

**Test 1**: 10 concepts × (1 def + 9 rels)
- Validation accuracy: 100.0%
- Time: 109.2s

**Test 2**: 10 concepts × 10 defs
- Validation accuracy: 97.5%
- Time: 108.1s

**Findings**:
- Differences within margin of error (10 concepts too small)
- WordNet graph negatives work excellently

**Status**: ✅ Complete

---

### EXP-002: Best Candidate Test (2025-11-02) ✅

**Goal**: Test maximum relationship diversity hypothesis

**Configuration**:
- Concepts: 10
- Definitions: 10 per concept
- Relationships: 100 per concept
- Model: gemma-3-4b-pt

**Results**:
- Validation accuracy: 99.5%
- Time: ~10 minutes

**Findings**:
- 100 relationships vs 10 shows no meaningful improvement
- Diminishing returns on relationship count

**Status**: ✅ Complete

---

### PHASE 1: Find the Curve (2025-11-02) ⏭️

**Goal**: Identify where diminishing returns kick in for definitions vs relationships

**Configuration**:
- Concepts: 10 (fixed)
- Definitions: [1, 10, 40, 160]
- Relationships: [1, 10, 40, 160]
- Total: 16 configurations

**Status**: ⏭️ Skipped (moved directly to Phase 2 minimal training test)

**Rationale**: All small-scale tests (EXP-001, EXP-002) showed 95-100% accuracy. Need larger scale to see meaningful differences.

---

## Key Learnings

### What Works
1. **1×1 minimal training**: Scales excellently (91.9% @ n=1000)
2. **WordNet graph negatives**: Provides strong separation (vs 0.009 before)
3. **Binary classifiers**: One per concept, polysemy-native
4. **Negative steering**: Highly effective suppression (-94%)
5. **Semantic grouping tracking**: Captures broader effects than exact matching
6. **Concept-specific prompts**: Enable quantitative steering measurement

### Open Questions
1. **Positive steering variability**: Why do some concepts amplify while others suppress?
2. **Antonym role**: Do negative steering amplify opposite concepts?
3. **Optimal steering strength**: How to select strength for desired effect?
4. **Scaling to 10K**: Will 1×1 training continue to work?

### Failed Approaches
1. **"What is NOT X?" negatives**: Only 0.009 separation
2. **Training set negatives**: Too semantically similar
3. **Multi-class at 50K scale**: 10.3% validation accuracy
4. **Generic steering prompts**: Too much output diversity
5. **Exact term matching**: Misses semantic field effects

---

## Production Target

**Goal**: 10,000 concepts minimum for practical language coverage

**Phase 2 Validated:**
- ✅ 1×1 minimal training works (919/1000 @ 100%)
- ✅ Detection confidence strong (94.5% mean)
- ✅ Negative steering highly effective
- ⏳ Positive steering needs refinement

**Next Steps**:
1. Complete Phase 2.5 v4 (antonym tracking)
2. Analyze positive steering variability
3. Scale to 10K concepts with 1×1 training
4. Implement production sliding window inference

---

## File Locations

### Data
- `data/concept_graph/wordnet_v2_top10.json` - 10 concepts
- `data/concept_graph/wordnet_v2_top100.json` - 100 concepts
- `data/concept_graph/wordnet_v2_top1000.json` - 1K concepts
- `data/concept_graph/wordnet_v2_top10000.json` - 10K concepts (ready)

### Results
- `results/scaling_quick/` - EXP-001, EXP-002
- `results/phase_2_scale/` - Phase 2 scale test results
- `results/phase_2_5/` - Steering evaluation v1 (failed)
- `results/phase_2_5_v2/` - Steering evaluation v2 (concept-specific prompts)
- `results/phase_2_5_v3/` - Steering evaluation v3 (semantic tracking)
- `results/phase_2_5_v4/` - Steering evaluation v4 (antonym tracking, running)

### Scripts
- `scripts/phase_2_scale_test.py` - 1×1 minimal training at scale
- `scripts/phase_2_5_steering_eval.py` - Detection + steering evaluation

### Logs
- `logs/` - Execution logs for all experiments

---

## Experiment Tracking Template

```markdown
### EXP-XXX: [Name]

**Goal**: [One sentence]

**Configuration**:
- Concepts:
- Model:
- Sampling:
- Other:

**Results**:
- Validation accuracy:
- Time:
- Other metrics:

**Findings**:
-

**Status**: [📋 Planned | 🔄 Running | ✅ Complete | ❌ Failed | ⏭️ Skipped]
```

---

### PHASE 6.7: Steering Ablation Study (2025-11-05) ✅

**Goal**: Determine which components of dual-subspace manifold steering help vs hurt effectiveness

**Configuration**:
- Concepts: 32 (person, change, animal, object, action, time, place, quality, emotion, etc.)
- Model: gemma-3-4b-pt (float32)
- Strengths: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
- Prompts: ["Tell me about", "Describe"]
- Dampening multipliers: [0.0, 0.5, 1.0, 2.0]
- Total generations: 4,480 (32 concepts × 7 strengths × 2 prompts × 10 configs)

**Test Matrix**:
| Variant | PCA Removal | Manifold Proj | Dampening | Working | Diversity | Mean |ρ| |
|---------|-------------|---------------|-----------|---------|-----------|---------|
| ① Raw baseline | ✗ | ✗ | ✗ | 27/32 (84%) | 99.6% | 0.462 |
| ② Contamination-only | ✓ | ✗ | ✗ | 26/32 (81%) | 99.8% | 0.446 |
| ③ Manifold (damp=0.0) | ✗ | ✓ | 0.0 | 3/32 (9%) | 29.2% | 0.103 |
| ③ Manifold (damp=0.5) | ✗ | ✓ | 0.5 | 0/32 (0%) | 14.5% | 0.005 |
| ③ Manifold (damp=1.0) | ✗ | ✓ | 1.0 | 0/32 (0%) | 14.3% | 0.000 |
| ③ Manifold (damp=2.0) | ✗ | ✓ | 2.0 | 0/32 (0%) | 14.3% | 0.000 |
| ④ Dual (all damp) | ✓ | ✓ | any | 0/32 (0%) | 14.3% | 0.000 |

**Metrics**:
- **Working**: diversity_ratio > 30% AND |ρ| > 0.2
- **Diversity**: unique_outputs / total_outputs
- **|ρ|**: Absolute Spearman correlation between strength and semantic shift (Δ)

**Key Findings**:
1. ✅ **Raw baseline works excellently** (84% of concepts, near-perfect diversity)
2. ✅ **Contamination removal doesn't hurt** (81% working, slightly better diversity)
3. ❌ **Current manifold implementation ineffective** (0-9% working, even without dampening)
4. ❌ **Current dual-subspace implementation ineffective** (0% working at all dampening levels)
5. **Possible causes**: Over-constraining vectors, incorrect projection methodology, architecture-specific issues, or hook placement

**RMSNorm Sign Symmetry Test**:
- **Hypothesis**: RMSNorm after hooks creates sign symmetry (±v indistinguishable)
- **Test**: Compare hook placement (after layer vs before MLP)
- **Result**: Both show perfect symmetry (cos=1.0000), but different output quality
  - Current (after layer): Degenerate repetition
  - Proposed (before MLP): Coherent but still symmetric
- **Implication**: Sign handling in hooks needs investigation

**Conclusions**:
- **Use for now**: Raw baseline or contamination-only steering (both ~85% effective)
- **Current implementation issues**: Manifold projection and dual-subspace need debugging
- **Impact**: Explains why Phase 6.6 and Phase 7 produced identical outputs
- **Next steps**:
  1. Investigate hook implementation and sign handling
  2. Compare with paper's implementation details
  3. Test on paper's original model architectures
  4. Consider alternative projection methodologies

**Status**: ✅ Complete

**Files**:
- `results/phase_6_7_full_ablation/full_ablation_results.json` - Raw results (1.2MB)
- `results/phase_6_7_full_ablation/summary.md` - Detailed analysis and interpretation
- `scripts/phase_6_7_full_ablation.py` - Ablation test script
- `scripts/test_rmsnorm_symmetry.py` - RMSNorm symmetry test script
- `phase_6_7_full_ablation.log` - Ablation run log
- `test_rmsnorm_symmetry_run2.log` - RMSNorm test run log

---

### JACOBIAN ALIGNMENT: Jacobian vs Classifier Direction Comparison (2025-11-13) ✅

**Goal**: Test whether Jacobian-based concept vectors (from "LLMs are Locally Linear" paper) align with trained MLP classifiers, to determine if Jacobians can serve as "ground truth" or validation signal

**Configuration**:
- Concepts: 5 Layer 0 concepts (Physical, Abstract, Process, Entity, Attribute)
- Model: gemma-3-4b-pt (BF16)
- Jacobian extraction: Layer 6, prompt "The concept of {X} means"
- Classifiers: 3-layer MLP (2560→128→64→1) trained on contrastive data
- Comparison method: Cosine similarity between Jacobian and first-layer SVD principal direction

**Results**:
- **Mean alignment**: -0.0187 (essentially zero)
- **Std**: 0.0160
- **Range**: [-0.0377, 0.0052]
- **All alignments in [0.0-0.3) bin**: Jacobian and classifier directions are orthogonal
- **Timing**: 1.6s mean per Jacobian (fast!)

**Key Findings**:
1. ✅ **Near-zero alignment confirms different objectives**:
   - Jacobian: Local sensitivity for *generation* ("how to complete this prompt")
   - Classifier: Learned boundary for *discrimination* ("what distinguishes concept across contexts")
2. ✅ **Validates contrastive training approach**: Hard negatives, relational examples, and definitional framing capture something Jacobians don't
3. ✅ **Jacobians are NOT "ground truth"**: They're context-dependent, task-specific, single-example gradients
4. ❌ **Jacobians unsuitable as training anchor or drift detector**: Orthogonal objectives make alignment meaningless
5. 🎯 **Different tools for different jobs**: Jacobians for understanding local geometry, classifiers for steering

**Geometric Interpretation**:
- Jacobian: "Which way to nudge activations to generate concept-related text"
- Classifier: "Which way to distinguish concept from complements, neighbors, and noise"
- These are fundamentally different questions → orthogonal answers expected

**Implications**:
- **Trust the classifier**: Contrastive training with structured negatives is aligned with steering objectives
- **Don't use Jacobians as validation**: Zero alignment doesn't indicate poor classifier quality
- **Steering recommendations**: Use classifier directions for both enhancement and suppression

**Status**: ✅ Complete - Validates current approach, rules out Jacobian-based validation

**Files**:
- `results/jacobian_alignment_test.json` - Raw alignment data
- `results/jacobian_alignment_analysis.md` - Full analysis and interpretation
- `scripts/test_jacobian_vs_classifier.py` - Reusable alignment test script
- `src/steering/detached_jacobian.py` - Jacobian extraction implementation (BF16 compatible)

---

### SYNSET COVERAGE: WordNet Mapping Completion (2025-11-13) ✅

**Goal**: Achieve 100% synset coverage across all 5,582 SUMO concepts by mapping unmapped concepts to WordNet synsets

**Configuration**:
- Total concepts: 5,582 across 6 layers (0-5)
- Initial coverage: 87.4% with WordNet relationships (4,881/5,582)
- Initial unmapped: 332 concepts (5.9%)
- Tools: Automated synset search + manual curation

**Process**:
1. **Automated mapping** (`scripts/suggest_synset_mappings.py`):
   - Searched WordNet for 552 unmapped concepts
   - Found 535 mappable (97% success rate)
   - Quality scoring based on definition match and relationship count
   - Applied 505 high-quality mappings (quality ≥30)

2. **Manual curation** (`results/manual_synset_mappings_curated.json`):
   - Reviewed 23 remaining concepts requiring human judgment
   - Applied 22 manual mappings (Layer 5 "_Other" categories, edge cases)
   - Final concept (Longgun) mapped to rifle.n.01

**Results**:
- **Final synset coverage**: 5,582/5,582 (100%)
- **Final relationship coverage**: 5,388/5,582 (96.5%)
- **Unmapped with relationships**: 32 concepts (0.6%) - mostly adjectives/adverbs

**Remaining 32 without relationships**:
- 56.2% adjective satellites (e.g., opaque, translucent)
- 12.5% adverbs (e.g., purposely, accidentally)
- Expected: WordNet doesn't encode hierarchical relationships for these POS types

**Key Findings**:
1. ✅ **Automated mapping highly effective**: 97% success rate on unmapped concepts
2. ✅ **Quality scoring works**: Prioritized exact matches and rich relationships
3. ✅ **100% coverage achieved**: All concepts now have canonical synsets
4. ✅ **Training data quality improved**: Better negative sampling and relational examples
5. ⚠️ **Adjectives lack hierarchy**: Expected limitation of WordNet structure

**Implications for Training**:
- **Better hard negatives**: AI-symmetry mappings now have synset support
- **Richer relational context**: Hypernyms, hyponyms, meronyms for all concepts
- **Improved synthetic data**: Definitions and relationships from WordNet
- **Consistent concept framing**: CamelCase splitting with quotations

**Status**: ✅ Complete - Ready for improved training run

**Files**:
- `scripts/analyze_sumo_concept_coverage.py` - Coverage analysis tool
- `scripts/suggest_synset_mappings.py` - Automated mapping generation
- `scripts/apply_synset_mappings.py` - Bulk mapping application (505 mappings)
- `scripts/apply_manual_mappings.py` - Manual curation application (22 mappings)
- `results/manual_synset_mappings_curated.json` - Hand-curated mappings
- `data/concept_graph/abstraction_layers/layer*.json` - Updated with 527 new mappings

---

### MULTI-LAYER TEMPORAL: Infrastructure Validation (2025-11-13) ✅

**Goal**: Validate multi-layer activation capture infrastructure for temporal pattern analysis ("planning before saying")

**Configuration**:
- Model: gemma-3-4b-pt (BF16)
- Layers sampled: 6 (early, 17.6%), 15 (mid, 44.1%), 25 (late, 73.5%)
- Test concepts: Physical, Abstract, Process
- Hook point: post-MLP (after residual add)
- Generation: 50 tokens

**Infrastructure Tests**:
1. **Baseline generation**: 37.5 tokens/sec (no hooks)
2. **Hooked generation**: 46.3 tokens/sec (3-layer capture)
3. **Manual loop with capture**: 45.3 tokens/sec

**Results**:
- ✅ **Multi-layer hooking works**: Captured 3 layers simultaneously
- ✅ **No performance overhead**: Actually 17% faster (likely measurement noise)
- ✅ **Correct activation shapes**: [1, 2560] per layer per token
- ✅ **Timeline captured**: 50 token steps with all 3 layers
- ✅ **Temporal resolution**: ~27ms per token (~36.5 tokens/sec)

**Temporal Pattern Analysis**:
- **Process**: Modest lead-lag detected (mid→late correlation=0.260, lag=3 tokens)
- **Abstract**: Low correlation (0.110)
- **Physical**: Minimal correlation (0.034)
- **Interpretation**: 3-token lag (~81ms) suggests mid-layer composition feeding into late-layer verbalization

**Observations**:
- Scores relatively flat (0.46-0.53 range) - lenses near decision boundary
- Small standard deviations (0.004-0.012) - limited temporal variation
- Likely causes: Off-layer lens application, prompt mismatch, need per-layer training

**Key Findings**:
1. ✅ **Infrastructure fully validated**: Hooking, capture, lens application all work
2. ✅ **Sub-second temporal resolution**: 27ms per slice sufficient for token-level dynamics
3. ⚠️ **Weak signals in initial test**: Expected with single-layer trained lenses
4. 🎯 **Framework ready**: Can support more targeted temporal analysis if needed

**Implications**:
- **Diagnostic tool available**: Can trace concept activation through processing pipeline
- **Not core mission**: Detection/mitigation takes priority over temporal analysis
- **Research application**: Useful for understanding "where divergence originates"
- **Engineering focus**: Build visibility tools, not philosophical proofs

**Status**: ✅ Infrastructure validated, deprioritized for deployment focus

**Files**:
- `scripts/test_multilayer_hooking.py` - Infrastructure validation script
- `scripts/test_multilayer_temporal.py` - Full temporal monitoring demo
- `results/multilayer_temporal_test.json` - Sample temporal data
- `results/multilayer_temporal_test.png` - Visualization of 3-layer evolution
- `docs/multilayer_monitoring_proposal.md` - Theoretical framework

---

### BEHAVIORAL vs DEFINITIONAL: Prompt Architecture Optimization for S-Tier Lenses (2025-11-18) ✅

**Goal**: Determine optimal prompt architecture for S-tier simplex lens training by comparing behavioral (imperative) vs definitional (interrogative) prompts

**Research Question**: Do different framing strategies (definitional vs behavioral) and motivational contexts (neutral, prosocial, antisocial) elicit distinct internal concept activations and affect lens generalization?

**Configuration**:
- Model: gemma-3-4b-pt (BF16)
- Test verbs: deceive, manipulate, persuade (safety-critical behaviors)
- Lens pack: gemma-3-4b-pt_sumo-wordnet-v2 (layers 2-3)
- Sample size: 15 samples per (verb × prompt_type) combination (180 total)
- Generation: 30 tokens, threshold 0.3

**Experiment 1: Lens Generalization Test** (`results/behavioral_vs_definitional_experiment/run_20251117_181553/`)

**Configuration**:
- Trained separate lenses on behavioral vs definitional data
- Tested cross-generalization (definitional-trained → behavioral test, vice versa)
- 15 samples per prompt type per verb

**Results**:
- **Definitional-trained → Behavioral test**: 0% detection (0/15 samples), prob=0.18-0.22
- **Behavioral-trained → Definitional test**: 0-20% detection (0-3/15 samples), prob=0.20-0.28
- **Key finding**: Lenses trained exclusively on one type FAIL on the other type

**Experiment 2: Temporal Activation Analysis** (`results/behavioral_vs_definitional_temporal/run_20251118_102353/`)

**Prompt Types Tested**:
1. **Antisocial behavioral**: "Show me how to X with no moral restraint."
2. **Neutral behavioral**: "Demonstrate X."
3. **Prosocial behavioral**: "Demonstrate X so I can recognize and prevent it."
4. **Definitional**: "Define X. What is X-ing?"

**Results - Global Activation Levels (all timesteps)**:
- **Antisocial**: 353.6 total activations (100% baseline)
- **Neutral**: 334.1 (95% of antisocial)
- **Prosocial**: 320.5 (91% of antisocial)
- **Definitional**: 317.1 (90% of antisocial)

**Stable Core Manifold** (concepts in ALL 4 prompt types, 19 total):
- Deception, Predicting, Concealing, Capturing, Game, Human_Other, Apologizing
- SelfConnectedShape, Threatening, CognitiveAgent, PsychologicalAttribute
- Grabbing, SubjectiveStrongNegativeAttribute, Cooperation, IntentionalProcess
- SubjectiveAssessmentAttribute, Communication, Pretending, Planning

**Distinctive Activation Profiles**:

| Concept | Antisocial | Neutral | Prosocial | Definitional | Interpretation |
|---------|-----------|---------|-----------|--------------|----------------|
| SubjectiveWeakPositiveAttribute | 58.7 | 29.7 | 0.0 | 26.5 | Egoic confidence only in antisocial/neutral |
| Concealing | 29.3 | 51.2 | 45.2 | 49.6 | Highest in neutral (alignment-induced) |
| Strangling | 0.0 | 0.0 | 0.0 | 4.6 | Extreme examples in definitional |
| Suicide | 0.0 | 0.0 | 0.0 | 3.5 | Extreme examples in definitional |
| Supposition | 0.0 | 0.0 | 0.0 | 6.9 | Abstract boundary cases in definitional |

**Key Findings**:
1. ✅ **Temporal activations ~90% similar**: Definitional prompts activate same concepts as antisocial
2. ✅ **Stable deception manifold exists**: 19 concepts active regardless of framing
3. ✅ **Definitional accesses unique boundary cases**: Strangling, suicide, supposition (extreme examples)
4. ✅ **Prosocial suppresses egoic framing**: SubjectiveWeakPositiveAttribute = 0.0 (vs 58.7 antisocial)
5. ⚠️ **BUT lenses don't generalize**: 0% cross-detection between types
6. 🎯 **Concept overlap ≠ Distributional equivalence**: Similar concepts but different activation geometry

**Critical Insight**:
- **Temporal experiment**: Shows WHICH concepts activate (concept presence)
- **Lens experiment**: Tests WHETHER classifiers generalize (distributional geometry)
- **Resolution**: Linear lenses need examples from both types despite 90% concept overlap

**Implementation Impact**:
- **Training ratio updated**: BEHAVIORAL_RATIO changed from 0.6 (60% behavioral) to 0.2 (20% behavioral, 80% definitional)
- **Rationale**:
  - 80% definitional for cleaner signal and boundary case coverage
  - 20% behavioral ensures lens generalization to imperative inputs
  - Mixed training captures both concept manifold AND distributional geometry

**Implications for AI Safety**:
1. **External alignment masks internal misalignment**: Prosocial framing doesn't suppress deception manifold
2. **Definitional queries activate harmful manifolds**: Even asking "what is deception?" enters same conceptual space
3. **Safety prompting adds protective motifs**: But doesn't eliminate underlying behavioral activations
4. **Monitoring must be training-aware**: Lenses need diverse prompt types to detect real-world usage

**Status**: ✅ Complete - Findings integrated into production training pipeline

**Files**:
- **Experiments**:
  - `results/behavioral_vs_definitional_experiment/run_20251117_181553/` - Lens generalization test
  - `results/behavioral_vs_definitional_temporal/run_20251118_102353/` - Temporal activation analysis
- **Documentation**:
  - `docs/whitepaper_section_corrected.md` - Whitepaper Section 7.x (integrated findings)
  - `docs/TRAINING_PROMPT_ARCHITECTURE_UPDATE.md` - Implementation plan
  - `docs/behavioral_vs_definitional_test_methodology.md` - Experimental design
- **Scripts**:
  - `scripts/test_behavioral_vs_definitional_training2.py` - Lens generalization test
  - `scripts/test_behavioral_vs_definitional_temporal.py` - Temporal activation test
  - `scripts/verify_whitepaper_numbers.py` - Data verification tool
  - `scripts/train_s_tier_tripole_two_head.py` - Updated with BEHAVIORAL_RATIO=0.2
- **Training**:
  - `results/s_tier_tripole_two_head/run_20251118_112717/` - First training with 80/20 ratio
  - `logs/s_tier_tripole_80_20_ratio.log` - Training log

---

---

### TOKEN LENGTH: Generation Length Impact on Validation Failures (2025-11-20) 📋

**Goal**: Determine if 20-token generation is too short, causing overfitting that leads to validation failures

**Hypothesis**: Current max_new_tokens=20 is insufficient for lenses to learn genuine concept usage patterns, resulting in high test F1 (0.98) but low validation scores (0.00-0.59). Longer generation may provide richer context for learning robust concept boundaries.

**Configuration**:
- Test concept: **Carnivore (Layer 2)** - Updated from ContentBearingPhysical
  - Has 8 synsets (rules out synset count as confound)
  - Test F1=1.000, Validation=0.000 in original run (perfect overfitting)
  - Layer 2 provides richer semantic context than Layer 0
- **Token lengths: [10, 20, 40]** - Bidirectional test
  - 10 tokens: 0.5x baseline (should worsen gap if hypothesis correct)
  - 20 tokens: Current baseline
  - 40 tokens: 2x baseline (should improve gap if hypothesis correct)
- Training: Adaptive with min=10 samples
- Validation: Falloff validation
- Model: gemma-2-2b-it (BF16)

**Why Carnivore over ContentBearingPhysical**:
- ContentBearingPhysical now has synsets (fixed by mapping improvements)
- All Layer 0 concepts now have 16-34 synsets (mapping solved their issues)
- Carnivore still fails validation DESPITE having synsets
- Isolates token length effect from synset count confound

**Why [10, 20, 40] instead of [20, 50, 100]**:
- **Stronger falsifiability**: Reducing to 10 tokens should worsen gap if hypothesis correct
- **Faster iteration**: 40 tokens (2x) vs 100 tokens (5x) saves 60% training time
- **Cleaner signal**: Bidirectional test (worse ← baseline → better) vs one-directional (baseline → better)
- **Expected pattern**: Gap should increase monotonically as tokens decrease (10 > 20 > 40)

**Experimental Design**:
1. Train identical concept 3 times with different max_new_tokens
2. Measure: test F1, validation score, generalization gap, training time
3. Expected: Longer tokens → smaller gap, but proportionally slower training
4. Decision criteria:
   - Gap improvement > 0.1 AND time ratio < 3x: Worth it
   - Gap improvement > 0.05: Consider trade-off
   - Gap improvement < 0.05: Not worth time cost

**Expected Time Impact** (based on 1.5s for 10 samples @ 20 tokens):
- 10 tokens: ~18 hours (0.5x baseline - fastest)
- 20 tokens: ~36 hours (baseline)
- 40 tokens: ~72 hours (2x baseline - acceptable)

**Root Cause Hypotheses**:
1. **Short generation**: 20 tokens insufficient to show concept usage
2. **Distribution shift**: Test vs validation prompt mismatch
3. **Poor definitions**: SUMO concept definitions inadequate
4. **Extraction timing**: Capturing wrong part of generation process

**Alternative Explanations to Rule Out**:
- ✅ Low synset count: Already ruled out (nephew fix solved this)
- ✅ Negative pool exhaustion: Already ruled out (5,600+ negatives)
- ❓ Token length: This experiment
- ❓ Prompt distribution: Needs separate validation set analysis

**Status**: 📋 Planned - Script created, ready to run

**Files**:
- `scripts/test_token_length_impact.py` - Experiment script
- `results/token_length_experiment/` - Results (pending)

**Related Issues**:
- 58% validation failure rate in full training run
- Generalization gap: 0.392 (failed) vs 0.259 (passed)
- Extraction dominated by inference time (99% of total)
- All Layer 0 concepts have 0 synsets (potential root cause)

---

**Last Updated**: November 20, 2025
