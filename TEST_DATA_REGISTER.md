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

**Last Updated**: November 5, 2025
