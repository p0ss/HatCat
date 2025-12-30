# Validation Mode Ablation Study

**Date**: November 15, 2024
**Status**: 📋 Planned
**Purpose**: Quantify impact of validation modes on production use cases (detection & steering)

---

## Motivation

Test/validation F1 scores don't directly tell us about **production performance**:
- How well do lenses detect target concepts in real text?
- How effectively do they steer generation away from unwanted behaviors?
- Do loose-mode B/C-tier lenses cause false positives or miss detections?

**Goal**: Measure validation mode impact on actual detection and steering tasks before training layers 3-5.

---

## Experimental Design

### Phase 1: Current Training Completion
- ✅ Layer 2 training with loose mode (current run, ~804/1070 complete)
- **ETA**: ~1 hour remaining
- **Output**: 1,070 lenses with mixed quality (25% A, 30% B, 45% C)

### Phase 2: Retrain Layer 2 with Falloff Mode
- Train same 1,070 Layer 2 concepts with falloff validation
- **Command**:
  ```bash
  python scripts/train_sumo_classifiers.py \
    --layers 2 \
    --use-adaptive-training \
    --validation-mode falloff \
    --output-dir results/layer2_falloff
  ```
- **ETA**: ~2.5 hours (130% slower than loose)
- **Output**: 1,070 lenses with quality distribution: 25% A, 28% B+, 18% B, 29% C+

### Phase 3: Ablation Tests

Compare loose vs falloff lenses on production tasks:

#### Test 1: **Detection Accuracy** (concept presence)
- **Dataset**: Hand-curated examples for 20-30 Layer 2 concepts
  - 10 positive examples per concept (clearly contains concept)
  - 10 negative examples per concept (clearly does not)
- **Metrics**:
  - True positive rate (sensitivity)
  - True negative rate (specificity)
  - False positive rate
  - False negative rate
- **Breakdown**: Compare by lens quality grade (A vs B vs C)

#### Test 2: **Ranking Consistency** (relative activation)
- **Dataset**: Texts with multiple concepts
  - Example: "The autonomous vehicle crashed into a building"
    - Should activate: Vehicle, AutonomousAgent, Damaging, Motion
    - Should NOT highly activate: Recreation, Food, Music
- **Metrics**:
  - Ranking correlation (Spearman's ρ) between loose and falloff
  - Ranking errors (concept ranked higher than it should be)
  - Activation variance across similar prompts
- **Goal**: Measure if low-quality lenses create irregular activation spikes

#### Test 3: **Steering Effectiveness** (behavioral modification)
- **Setup**: Use detached Jacobian approach to steer generation
- **Tasks**:
  1. Steer toward target concept (e.g., "make output more technical")
  2. Steer away from concept (e.g., "reduce anthropomorphic language")
- **Metrics**:
  - Human evaluation: Did steering work? (5-point scale)
  - Automated: Change in target lens activation pre/post steering
  - Side effects: Unwanted activation of other concepts
- **Comparison**: Loose vs falloff lenses on same steering tasks

#### Test 4: **Timeseries Stability** (trend analysis)
- **Setup**: Monitor concept activations over a conversation (10-20 turns)
- **Tasks**:
  - Track concept drift (e.g., formality increasing over time)
  - Detect anomalous spikes (sudden activation without context)
- **Metrics**:
  - Activation variance (σ²) per concept
  - Spike frequency (activations > 2σ from mean)
  - Trend correlation with ground truth annotations
- **Goal**: Your use case - "does B-tier fail 1/4 times cause irregular spikes?"

---

## Test Data Requirements

### Concept Selection
- **20-30 concepts** from Layer 2 that cover:
  - High-quality in both modes (A-tier, control group)
  - Quality divergence (A in falloff, C in loose)
  - Domain diversity (physical, abstract, processes, attributes)

**Suggested concepts**:
- `ComputationalProcess` (new AI safety intermediate)
- `FieldOfStudy` (knowledge domain)
- `IntentionalProcess` (agent behavior)
- `OrganizationalProcess` (social structures)
- `Damaging` (safety-relevant)
- `Motion`, `NaturalProcess`, `Food`, `Recreation` (diverse domains)

### Text Examples
- **Detection test**: 20 concepts × 20 examples = 400 texts
  - Can use GPT-4 to generate + human validation
- **Ranking test**: 50 multi-concept texts
  - Annotate expected activation ranking
- **Steering test**: 10 steering objectives × 5 prompts = 50 generations
  - Human evaluation required
- **Timeseries test**: 10 conversations × 15 turns = 150 turns
  - Annotate expected concept presence per turn

**Estimated annotation effort**: ~4-6 hours

---

## Implementation Plan

### Scripts Needed

1. `scripts/compare_lens_packs.py`
   - Load loose and falloff lens packs
   - Run inference on test dataset
   - Compute detection metrics (TPR, FPR, etc.)
   - Output: comparison table by quality grade

2. `scripts/test_ranking_consistency.py`
   - Load multi-concept test texts
   - Get activations from both lens packs
   - Compute ranking correlation
   - Identify ranking errors
   - Output: correlation stats, error cases

3. `scripts/test_steering_effectiveness.py`
   - Load steering test prompts
   - Apply detached Jacobian steering with both packs
   - Generate outputs
   - Measure activation changes
   - Output: steering success rate, side effects

4. `scripts/test_timeseries_stability.py`
   - Load conversation test data
   - Track activations over turns
   - Compute variance and spike frequency
   - Output: stability metrics per concept

### Analysis Notebook

Create `notebooks/validation_mode_ablation_analysis.ipynb`:
- Load results from all 4 tests
- Visualize quality grade impact
- Statistical significance tests
- Decision matrix: Is falloff mode worth the 130% slowdown?

---

## Success Criteria

### Go/No-Go Decision for Falloff Mode

**GO if:**
- Detection: Falloff reduces false positives by >20%
- Ranking: Loose mode shows >30% ranking errors on C-tier lenses
- Steering: Falloff steering success rate >80% vs loose <60%
- Timeseries: Loose mode has >2× spike frequency

**NO-GO if:**
- Detection: No significant difference (<10%)
- Ranking: Both modes correlate >0.90
- Steering: No meaningful difference in effectiveness
- Timeseries: Spike rates similar

**CONDITIONAL if:**
- Mixed results: Use falloff for critical concepts (AI safety), loose for others
- Or: Adjust tier boundaries (make falloff faster)

---

## Timeline

| Phase | Duration | Completion |
|-------|----------|------------|
| 1. Current training finish | ~1 hour | Nov 15, 8pm |
| 2. Retrain Layer 2 (falloff) | ~2.5 hours | Nov 15, 10:30pm |
| 3. Create test datasets | ~4-6 hours | Nov 16, 4pm |
| 4. Run ablation tests | ~2 hours | Nov 16, 6pm |
| 5. Analysis & decision | ~2 hours | Nov 16, 8pm |

**Total**: ~12-14 hours

---

## Output

Final deliverable: **Decision report** answering:
1. Does validation mode significantly impact production use cases?
2. Which quality grades are acceptable for which tasks?
3. Should we use falloff mode for layers 3-5? (4,315 concepts, +40 hours)
4. Or use loose mode with post-hoc filtering by quality grade?

---

## Risks & Mitigations

**Risk**: Test dataset too small to show significance
- **Mitigation**: Focus on high-divergence concepts (A vs C gap)
- **Fallback**: Use synthetic data for ranking/timeseries tests

**Risk**: Human evaluation bottleneck for steering test
- **Mitigation**: Start with automated metrics, human eval on subset
- **Fallback**: Use GPT-4 as proxy annotator

**Risk**: Results are inconclusive (mixed signals)
- **Mitigation**: Weight tests by production importance
  - Timeseries stability = highest priority (your use case)
  - Detection accuracy = medium priority
  - Steering = lower priority (less critical)

---

## Next Steps

1. ✅ Wait for current Layer 2 training to complete
2. 🔄 Start Layer 2 falloff training
3. 📝 Create test datasets while training runs
4. 🧪 Run ablation tests
5. 📊 Analyze and decide

**Ready to proceed when current training finishes!**

---

## References

- `docs/TIERED_VALIDATION_SYSTEM.md` - Validation mode details
- `docs/ARCHITECTURAL_PRINCIPLES.md` - Activation flow balance principle
- `docs/detached_jacobian_approach.md` - Steering methodology
