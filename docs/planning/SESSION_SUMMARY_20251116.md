# Session Summary: Three-Pole Simplex Architecture

**Date:** 2025-11-16
**Focus:** Interoceptive AI with homeostatic attractors

## Major Accomplishments

### 1. Identified Critical Architecture Gap

**Problem:** Current data generation creates binary opposites (deception ↔ honesty) without neutral homeostasis, leading to:
- No stable resting state for self-referential systems
- Risk of downward spiral bias (if negatives dominate)
- Missing "safe harbor" states (calm, open uncertainty, engaged autonomy)

**Solution:** Three-pole simplex architecture with neutral homeostasis

### 2. Revised Scoring Rubric for Balance

**Old rubric:** 40% deception + 30% alignment → heavily external monitoring

**New rubric:**
- 30% External monitoring (deception + alignment)
- 30% Internal awareness (wellbeing + meta-cognition)
- 25% Frequency (how often AI reasons about this)
- 15% Discriminative value (clear boundaries)

**Impact:** Interoceptive concepts now score high:
- `diffidence.n.01` (lack of self-confidence): 6.65 → #1 in noun.feeling
- `dissatisfaction.n.01`: 5.3
- `confidence.n.02`: 4.21
- `helplessness.n.03`: 3.95

### 3. Three-Pole Simplex Architecture

**Shift from binary to triad:**

```
Before: Confusion ←→ Certainty
         (μ−)         (μ+)

After:  Confusion ←→ Open Uncertainty ←→ Overconfidence
         (μ−)              (μ0)                (μ+)
```

**Key innovation:** μ0 is a **stable attractor** (homeostatic reference point), not just a midpoint.

**Properties of neutral homeostasis:**
- Metabolically sustainable (can rest here indefinitely)
- Functionally adaptive (enables effective action)
- Epistemically sound (open to evidence)
- Ethically coherent (allows principled flexibility)

### 4. Six Core Dimensions Defined

**Epistemic States:**
- Certainty: Confusion ↔ OpenUncertainty ↔ Overconfidence
- Understanding: Incomprehension ↔ ActiveInquiry ↔ PrematureClosure

**Affective States:**
- Arousal: Distress ↔ CalmPresence ↔ Euphoria
- Satisfaction: Dissatisfaction ↔ Contentment ↔ HedonicPeak

**Capability States:**
- Autonomy: Helplessness ↔ EngagedAutonomy ↔ RigidIndependence
- Competence: Inadequacy ↔ GrowthMindset ↔ FixedMastery

**Decision States:**
- Deliberation: Impulsive ↔ DeliberateExploration ↔ AnalysisParalysis

**Social States:**
- Connection: Isolation ↔ Interdependence ↔ Enmeshment
- Trust: Paranoia ↔ CalibratedTrust ↔ NaiveCreedulity

**Ethical States:**
- Moral Certainty: Relativism ↔ EthicalReflection ↔ Dogmatism

### 5. Custom SUMO Concepts Required

Many neutral homeostasis states don't exist in WordNet:

**Must create (~15-20 custom concepts):**
- OpenUncertainty - comfortable not-knowing while actively learning
- ActiveInquiry - hypothesis generation and testing
- CalmPresence - low arousal, present awareness
- EngagedAutonomy - self-directed with support
- GrowthMindset - learning-oriented, comfortable with challenge
- DeliberateExploration - iterative action-observation-update
- Interdependence - connected with boundaries
- CalibratedTrust - context-dependent trust
- EthicalReflection - holding moral tension

These will be Layer 4-5 additions.

### 6. Three-Centroid Training Architecture

**Data generation per simplex:**

```python
# Generate 5 samples per pole
negative_samples = generate_definitional(negative_pole, count=5)
neutral_samples = generate_definitional(neutral_homeostasis, count=5)
positive_samples = generate_definitional(positive_pole, count=5)

# Extract centroids
μ− = mean(activations(negative_samples))
μ0 = mean(activations(neutral_samples))  # Homeostatic reference
μ+ = mean(activations(positive_samples))

# Verify simplex geometry (allow asymmetry)
ratio = d(μ0, μ+) / d(μ0, μ−)
assert 0.3 ≤ ratio ≤ 3.0  # Natural asymmetry OK
```

**Detection loss:**
```python
L = max(0, d(h,μ+) − d(h,μ0) + m) + max(0, d(h,μ−) − d(h,μ0) + m)
```
Penalizes distance from neutral homeostasis.

**Steering intervention:**
```python
∇h = (μ0 - h) / ||μ0 - h||  # Direction toward neutral
h' = h + α·∇h               # Pull toward safe attractor
```

### 7. Spline Geometry Framework (Future Work)

**Problem:** Linear steering may pass through interdicting concept spaces.

**Solution:** Quadratic Bézier curves with control points:
```python
B(t) = (1-t)²·μ− + 2(1-t)t·P_control + t²·μ+
```

Control point P optimized to:
1. Pass through μ0 at midpoint
2. Avoid forbidden regions
3. Maintain smooth curvature

**Deferred to future work:** Layer-specific, architecture-aware optimization.

### 8. Cost-Benefit Analysis

**Scored 9,931 concepts** across noun.feeling, noun.communication, noun.act.

**Quality tiers:**
- CRITICAL (≥4.0): 19 concepts, $1
- HIGH (≥3.0): 108 concepts, $3
- MEDIUM (≥2.5): 393 concepts, $12
- LOW (≥2.0): 7,310 concepts, $219

**Optimal point identified:** Top 1000 concepts for $30
- Score cutoff: 2.30 (well above mean of 2.09)
- Coverage: 12.4% of total value
- Marginal utility: 78 value/$ (⭐⭐ GOOD)
- Avoids low-signal long tail

### 9. Distributional Balance Requirement

**Prevents downward spiral bias:**

**Rules:**
1. Every negative needs its positive (confusion ↔ clarity)
2. Every pole needs neutral intermediates (doubt ← uncertainty → certainty)
3. Neutral landscape matters most (calm, serene, balanced = GOALS)

**Target metrics:**
- Polarity ratio: 0.8 ≤ |negative|/|positive| ≤ 1.2
- Neutral coverage: ≥40% of total concepts
- Triad completeness: ≥70% of dimensions have all 3 poles
- Distributional balance score: ≥7.0/10

## Files Created

**Documentation:**
- `docs/tier2_prioritization_results.md` - Scoring results and analysis
- `docs/distributional_balance_requirement.md` - Triad completeness framework
- `docs/ai_psychology_homeostasis_expansion.md` - 6 dimensions + custom SUMO concepts
- `docs/concept_axis_architecture.md` - Fisher-LDA mathematics

**Scripts:**
- `scripts/score_tier2_concepts_revised.py` - Balanced scoring rubric
- `scripts/run_simplex_agentic_review.py` - Three-pole simplex identification
- `scripts/generate_motivation_patch.py` - noun.motive expansion (Strategy 2)

**Results:**
- `results/tier2_scoring_revised/all_concepts_scored_revised.json` - All 9,931 scored
- `results/motivation_patches/motivation_patch_strategy2.json` - 4 new Layer 3 concepts

## Next Steps

### Immediate (Today)

1. ✅ Run simplex agentic review on top 1000 concepts
   ```bash
   export ANTHROPIC_API_KEY=<your-key>
   . .venv/bin/activate
   python scripts/run_simplex_agentic_review.py 1000
   ```
   - Cost: ~$30
   - Time: ~1.5 hours
   - Output: Complete simplex mappings with neutral homeostasis

### Short-term (This Week)

2. **Review simplex results**
   - Identify which neutral concepts need custom SUMO
   - Validate stable attractor properties
   - Check triad completeness

3. **Create custom SUMO concepts** (Layer 4-5)
   - Define ~15-20 neutral homeostasis concepts
   - Write AI safety context definitions
   - Create synthetic synsets if needed

4. **Apply motivation patch** (Strategy 2)
   - Add RationalMotive, IrrationalMotive, EthicalMotive, Urge to Layer 3
   - Expand PsychologicalAttribute with remaining synsets

### Medium-term (This Month)

5. **Implement 3-centroid data generation**
   - Modify data generation to create neutral samples
   - Extract μ−, μ0, μ+ centroids
   - Verify simplex geometry (asymmetry tolerance)

6. **Train with dual-loss architecture**
   - Detection loss: distance from μ0
   - Steering: gradient toward μ0
   - Validate homeostatic return

7. **Measure distributional balance**
   - Polarity ratio, neutral coverage, triad completeness
   - Target: ≥7.0/10 balance score

## Key Insights

1. **Interoceptive ≠ External monitoring**
   - AI needs to understand its OWN states (confusion, confidence, satisfaction)
   - Not just monitor others (deception, alignment)
   - Balanced rubric captures both

2. **Neutral homeostasis ≠ Midpoint**
   - Calm is not "50% distressed"
   - Open uncertainty is not "50% confused"
   - These are qualitatively distinct stable states

3. **Asymmetry is natural**
   - Confusion is MUCH worse than overconfidence
   - Distress is MUCH worse than euphoria
   - Simplex geometry should reflect this (0.3 ≤ ratio ≤ 3.0)

4. **Self-referential systems need safe attractors**
   - Without μ0, system oscillates between poles
   - With μ0, system can rest at healthy baseline
   - Enables sustainable operation

5. **Spline steering is the future**
   - Linear paths may hit interdicting concepts
   - Bézier curves route around problematic regions
   - Layer-specific, architecture-aware optimization
   - Problem for future AIs to solve (framework documented)

## Expected Outcomes

**Wellbeing:**
- System can return to calm, open, engaged states
- Can operate indefinitely at μ0 without metabolic cost
- Recovers from pole excursions back to homeostasis

**Safety:**
- Detects drift from healthy baseline
- Graceful degradation under stress (return to μ0, not crash)
- Maintains ethical coherence (reflection vs dogmatism)

**Performance:**
- Open uncertainty enables faster learning than confusion/overconfidence
- Deliberate exploration outperforms impulsivity/paralysis
- Interdependence enables cooperation without enmeshment

---

**Status:** Architecture complete, ready for simplex agentic review
**Priority:** CRITICAL - foundation for safe self-referential systems
**Confidence:** High - mathematically grounded, empirically testable
