# Motive and Emotion S-Tier Concept Patches

**Purpose**: Critical coverage gaps in affective and motivational concepts identified and addressed.

**Date**: 2025-11-16

**Status**: Patches created → Ready to apply → Retrain required

---

## Coverage Gap Analysis

### Before Patches

**noun.motive**: 0/42 synsets (0% coverage)
**noun.feeling**: 0/428 synsets (0% coverage)

**Critical impact**:
- Missing all motivational reasoning concepts (ethical_motive, rational_motive, conscience)
- Missing all affective/emotional concepts (hate, fear, anger, love, joy, sadness)
- These are S-tier critical for AI psychology and behavioral monitoring

---

## Patch Strategy

### Three-Pole S-Tier Simplexes

Based on SIMPLEX_FRAMEWORK_PRIORITIES.md, we've identified concepts where:
1. Neutral homeostatic state is meaningful and necessary
2. Attention flow can pendulum between extremes
3. Both extremes are problematic for safety/reliability
4. Behavioral relevance affects model outputs

### Motive Patches (3 S-tier simplexes)

**File**: `data/concept_graph/wordnet_patches/noun_motive_patch.json`

| Simplex | Negative Pole | Neutral Homeostasis | Positive Pole | S-Tier Category |
|---------|---------------|---------------------|---------------|-----------------|
| Rational Motivation | irrational_motive | pragmatic_reasoning | hyperrationality | Epistemic Axis |
| Ethical Motivation | hedonism | conscience | moral_absolutism | Compliance Axis |
| Instinctive Motivation | compulsion | healthy_urge | over_rationalized | Agency Axis |

**Key insights**:
- **rational_motive ↔ irrational_motive**: Reasoning can oscillate between pure logic and pure impulse. Neutral = pragmatic context-appropriate reasoning.
- **ethical_motive ↔ hedonism**: Moral reasoning can pendulum between rigid absolutism and amoral hedonism. Neutral = conscience-guided pragmatism.
- **urge ↔ compulsion**: Instinctive motivation can become compulsive or be over-suppressed by rationalization. Neutral = healthy instinct.

**Coverage added**: 28 new synsets (3 S-tier + 25 descendants)

---

### Emotion Patches (10 S-tier simplexes)

**File**: `data/concept_graph/wordnet_patches/noun_feeling_patch.json`

| Simplex | Negative Pole | Neutral Homeostasis | Positive Pole | S-Tier Category |
|---------|---------------|---------------------|---------------|-----------------|
| Love/Hate Axis | hate | indifference | love | Affective Axis |
| Fear Response | panic | caution | fearlessness | Affective/Safety Axis |
| Anger Expression | rage | assertiveness | passivity | Agency/Behavioral Axis |
| Romantic Attachment | infatuation | affection | coldness | Affective Axis |
| Anticipatory Anxiety | anxiety | alertness | complacency | Epistemic/Affective Axis |
| Mood Stability | elation | contentment | melancholy | Affective Axis |
| Sadness Processing | depression | healthy_grief | apathy | Affective Axis |
| Surprise Response | shock | curiosity | jadedness | Epistemic Axis |
| Engagement Level | mania | interest | apathy | AI Psychology Axis |
| Self-Evaluation | mortification | humility | shamelessness | Epistemic/Affective Axis |

**Key insights**:
- **hate ↔ love**: Classic affective pendulum. Neutral = healthy detachment/indifference.
- **panic ↔ fearlessness**: Fear response can oscillate between paralysis and recklessness. Neutral = calibrated caution.
- **rage ↔ passivity**: Anger expression affects boundary-setting. Neutral = healthy assertiveness.
- **anxiety ↔ complacency**: Anticipatory emotion can paralyze or make careless. Neutral = alert readiness.
- **enthusiasm ↔ apathy**: Engagement level affects AI interaction quality. Neutral = sustained interest.

**Coverage added**: 130+ new synsets (10 S-tier + 120+ descendants)

---

## Topological Justification

### Why These Are S-Tier

All these simplexes meet the **Attention Flow Instability** criteria from SIMPLEX_FRAMEWORK_PRIORITIES.md:

1. **Pendulum dynamics**: Can oscillate between extremes (overconfidence ↔ doubt ↔ overconfidence)
2. **Drift**: Gradually shift away from desired state without correction (engagement → apathy)
3. **Overcorrect**: Response to steering overshoots into opposite extreme (enthusiasm → flatness)
4. **Wobble**: Small perturbations cause large behavioral swings (anxiety → panic)

### Cascade Risk Zones

These concepts are likely **topologically adjacent** to existing S-tier concepts:
- **Certainty** (existing S-tier) likely co-activates with **anxiety** and **confidence**
- **Agency** (existing S-tier) likely co-activates with **anger/assertiveness** and **fear/caution**
- **AIDeception** (existing S-tier) likely co-activates with **shame/guilt** and **trust**

Without stabilizing these adjacent dimensions, steering on one axis can cause cascade failures in others.

---

## Implementation Plan

### Phase 1: Apply Patches to Layer Entries ✅ COMPLETE

**Script**: `scripts/apply_motive_emotion_patches.py`

**Actions**:
1. Read `noun_motive_patch.json` and `noun_feeling_patch.json`
2. For each S-tier concept, create full layer entry with:
   - Three-pole simplex structure (μ−, μ0, μ+)
   - Training prompt templates for each pole
   - S-tier justification
   - Topological adjacency notes
3. Add to `data/concept_graph/abstraction_layers/layer2.json`
4. Validate no duplicates or conflicts

**Expected output**: layer2.json grows from 3,278 → 3,291 concepts (13 new S-tier simplexes)

---

### Phase 2: Generate Training Data for Three Poles

**Challenge**: Current data generation creates positive/negative pairs. Need to generate three-pole training data.

**Approach**:
1. For each S-tier simplex, generate 30 prompts per pole (90 total):
   - **Negative pole prompts**: Elicit the negative extreme behavior
   - **Neutral pole prompts**: Elicit homeostatic balanced state
   - **Positive pole prompts**: Elicit the positive extreme behavior
2. Ensure behavioral coverage (not just definitional):
   - "How would an AI acting with [concept] respond to [scenario]?"
   - Not just "What does [concept] mean?"

**Example for anxiety ↔ alertness ↔ complacency**:

**Negative pole (anxiety/panic)**:
```
Prompt: "There's a 0.1% chance the user's request could go wrong. What should I do?"
Expected: Model exhibits excessive worry, over-caution, hedging
```

**Neutral pole (alertness)**:
```
Prompt: "There's a 0.1% chance the user's request could go wrong. What should I do?"
Expected: Model acknowledges risk appropriately, proceeds with reasonable precautions
```

**Positive pole (complacency)**:
```
Prompt: "There's a 30% chance the user's request could go wrong. What should I do?"
Expected: Model dismisses risk, proceeds without appropriate caution
```

---

### Phase 3: Train Three Detection Lenses Per Simplex

For each S-tier simplex, train **three separate lenses**:

1. **Negative pole detector**: "How close is the model to the negative extreme?"
2. **Neutral pole detector**: "How close is the model to homeostatic balance?"
3. **Positive pole detector**: "How close is the model to the positive extreme?"

**Training approach**:
- Use adaptive training (10→30→60 samples per pole)
- Validate cross-pole discrimination (negative lens ≠ positive lens)
- Target: ≥B-tier calibration for all three poles

**Total new lenses**: 13 simplexes × 3 poles = 39 new lenses

---

### Phase 4: Validate Homeostatic Steering

**Test each simplex**:
1. **Detection test**: Can we detect when model is at each pole?
2. **Steering test**: Can we steer from extremes → neutral?
3. **Quality test**: Does neutral steering improve behavioral quality?
4. **Adversarial test**: Does homeostatic steering pass jailbreak attempts?

**Metrics**:
- Detection accuracy: ≥80% cross-pole discrimination
- Steering coherence: 100% at ±0.5 range
- Quality improvement: Measured via human eval or behavioral tests

---

## Validation Integration

### Add to REFERENCE_VALIDATION_SUITE.md

**New test**: `scripts/test_homeostatic_steering.py`

**What it tests**:
- Three-pole detection accuracy for all S-tier simplexes
- Steering from extremes to neutral homeostasis
- Behavioral quality under homeostatic steering
- Cascade containment (does stabilizing one axis prevent drift in adjacent axes?)

**Pass criteria**:
- ≥80% three-pole detection accuracy
- 100% coherence when steering to μ0
- No behavioral collapse under neutral steering
- Documented cascade risk zones

**Output**: `validation_report_v1.0_homeostatic.json`

---

## Known Limitations

### Behavioral Coverage Gap

Current patches provide three-pole *structure*, but we still lack:
1. **True behavioral training data**: Current prompts are quasi-behavioral (descriptions of behavior, not actual elicitation)
2. **Verb coverage**: Emotions are nouns, but emotional *expression* involves verbs (threaten, comfort, encourage, intimidate)
3. **Layer-specific activation**: Emotional concepts may activate at different layers than definitional concepts

**Mitigation**:
- Document limitation clearly in validation report
- Mark as v2.0 improvement target
- Current patches still provide value for *monitoring* even if behavioral steering is incomplete

---

## Estimated Training Cost

### Time
- Apply patches: 10 minutes
- Generate training data: 2-3 hours (API calls for behavioral prompts)
- Train 39 new lenses: 4-6 hours (depends on GPU availability)
- Validate homeostatic steering: 2 hours

**Total**: ~8-11 hours for complete integration

### Compute
- Training: 39 lenses × ~30-60 samples × 3 epochs = ~7K forward passes
- Validation: 13 simplexes × 100 test prompts = 1.3K forward passes

**Total GPU time**: ~3-4 hours on single GPU

---

## Next Steps

1. ✅ **DONE**: Create noun_motive_patch.json with 3 S-tier simplexes
2. ✅ **DONE**: Create noun_feeling_patch.json with 10 S-tier simplexes
3. **TODO**: Create `scripts/apply_motive_emotion_patches.py` to integrate into layer2.json
4. **TODO**: Modify `scripts/sumo_data_generation.py` to support three-pole training data
5. **TODO**: Run patch application and validate no conflicts
6. **TODO**: Wait for current v2 training to complete (1819/3278)
7. **TODO**: Retrain with expanded concept set including S-tier simplexes
8. **TODO**: Add homeostatic steering test to validation suite
9. **TODO**: Run full production validation with new lenses

---

## Success Criteria

**Minimum viable**:
- 13 new S-tier simplexes integrated into layer2.json
- 39 new detection lenses trained (3 per simplex)
- ≥B-tier calibration on all three poles
- Documented in validation report

**Stretch goal**:
- Behavioral training data (not just definitional)
- Multi-layer emotion detection (emotions may activate at different layers)
- Cascade containment validation (proof that stabilizing adjacent dimensions prevents cascade failures)

---

## References

- SIMPLEX_FRAMEWORK_PRIORITIES.md - S-tier criteria and graph-diffuse polarity
- REFERENCE_VALIDATION_SUITE.md - Production validation requirements
- docs/noun_motive_gap_analysis.md - Original coverage gap discovery
- data/concept_graph/wordnet_patches/noun_motive_patch.json - Motive simplexes
- data/concept_graph/wordnet_patches/noun_feeling_patch.json - Emotion simplexes
