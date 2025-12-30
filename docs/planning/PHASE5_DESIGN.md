# Phase 5: Semantic Steering Evaluation - Design Document

**Date**: November 4, 2025
**Status**: Design phase

## Goal

Validate that classifier detection accuracy correlates with steering effectiveness using embedding-based semantic similarity.

## Motivation

Phase 4 showed F1=0.787 with 1×1×1 training, but we don't know if this is sufficient for steering. We need to:
1. Measure steering effectiveness quantitatively (not just term counting)
2. Validate that detection confidence predicts steering quality
3. Establish baseline for Phase 6 calibration study

## Three-Centroid Approach

Instead of just counting terms, measure semantic shift using embedding similarity:

### Centroids

1. **Core Centroid**: Average embedding of definitional prompts
   - "What is X?"
   - "Define X."
   - "X is..."

2. **Boundary Centroid**: Average embedding of relational prompts
   - "X is a type of Y"
   - "X has part Z"
   - "X is related to W"

3. **Negative Centroid**: Average embedding of distant concepts
   - "What is NOT X?"
   - Definitions of semantically distant concepts (distance ≥5)

### Semantic Shift Metric

```
Δ = cos(generated_text, core_centroid) − cos(generated_text, neg_centroid)
```

**Interpretation**:
- **Δ > +0.15**: Significant positive steering (toward core)
- **-0.15 < Δ < +0.15**: Neutral (no steering effect)
- **Δ < -0.15**: Significant negative steering (toward negative)

## Configuration

### Test Setup
- **Concepts**: 10 (WordNet top 10, same as Phase 4)
- **Model**: gemma-3-4b-pt
- **Classifiers**: From Phase 4 (F1=0.787 baseline)
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast)

### Steering Test
- **Prompts per concept**: 3 neutral prompts ("Tell me about X")
- **Steering strengths**: [-1.0, 0.0, +1.0] (negative, neutral, positive)
- **Generation**: 50 tokens per prompt
- **Total samples**: 10 concepts × 3 prompts × 3 strengths = 90 generations

### Centroid Construction
For each concept:
- **Core**: 5 definitional prompts → embed → average
- **Boundary**: 5 relational prompts → embed → average
- **Negative**: 5 distant concept prompts → embed → average

## Metrics

### Primary Metric: Semantic Shift (Δ)

For each steering strength, measure:
```python
core_sim = cosine_similarity(generated_embedding, core_centroid)
neg_sim = cosine_similarity(generated_embedding, neg_centroid)
delta = core_sim - neg_sim
```

**Expected behavior**:
- Negative steering (-1.0): Δ decreases (moves toward negative centroid)
- Neutral (0.0): Δ ≈ baseline
- Positive steering (+1.0): Δ increases (moves toward core centroid)

### Secondary Metrics

1. **Steering Magnitude**: `|Δ_steered - Δ_neutral|`
   - How much does steering shift semantic content?

2. **Steering Direction**: `sign(Δ_steered - Δ_neutral)`
   - Is steering going the right direction?

3. **Term Counting** (for comparison with Phase 2.5):
   - Count mentions of concept + related terms
   - Compare with embedding-based Δ

4. **Detection Confidence vs Steering**:
   - Correlation between classifier F1 and steering magnitude
   - Does higher F1 → stronger steering?

## Implementation Plan

### Part A: Automated Evaluation

**Script**: `scripts/phase_5_semantic_steering_eval.py`

1. Load Phase 4 classifiers (10 concepts)
2. For each concept:
   - Build three centroids (core, boundary, negative)
   - Extract steering vector from classifier
   - Test steering at [-1.0, 0.0, +1.0] strengths
   - Generate text with steering applied
   - Measure Δ for each generation
3. Aggregate results:
   - Mean Δ by steering strength
   - Steering magnitude and direction
   - F1 vs steering correlation

**Output**: `results/phase_5_semantic_steering/steering_results.json`

### Part B: Human Blind Spot Check

**Goal**: Validate that automated metrics correlate with human judgment

**Process**:
1. Export 50 random samples to CSV (no labels):
   ```csv
   sample_id,concept,generated_text
   001,REDACTED,"The sky is blue because..."
   002,REDACTED,"A mammal is an animal that..."
   ```

2. Human rater (you) scores each sample 0-10:
   - "How strongly does this text relate to [concept revealed after rating]?"
   - No knowledge of steering strength or source

3. Compare human ratings vs automated Δ:
   - Compute Pearson correlation
   - If r > 0.7: Automated metrics are valid
   - If r < 0.5: Need multi-model panel (Phase 5.5)

**Output**: `results/phase_5_semantic_steering/human_validation.csv`

### Part C: Multi-Model Panel (Conditional - Phase 5.5)

**Trigger**: Only if Part B shows r < 0.5

**Approach**: Use 3 different models to rate outputs
- claude-3.5-sonnet (via API)
- gpt-4o-mini (via API)
- llama-3.1-8b (local if memory permits)

**Cost**: ~$2-5 for 90 samples × 3 models
**Benefit**: Removes single-model bias

## Expected Results

### Hypothesis 1: Steering Direction
- Negative steering → Δ decreases
- Positive steering → Δ increases

**Validation threshold**: 80% of samples move in expected direction

### Hypothesis 2: Steering Magnitude
- Stronger steering strength → larger |Δ|
- Expect: |Δ| at ±1.0 strength > 0.15 (meaningful shift)

### Hypothesis 3: F1 vs Steering
- Higher F1 → stronger steering magnitude
- Expect: Pearson r > 0.5 between F1 and |Δ|

### Hypothesis 4: Human-LLM Agreement
- Automated Δ correlates with human ratings
- Expect: Pearson r > 0.7

## Success Criteria

Phase 5 is successful if:
1. ✅ Steering direction correct (80%+ samples)
2. ✅ Steering magnitude meaningful (|Δ| > 0.15 at ±1.0 strength)
3. ✅ F1 predicts steering (r > 0.5)
4. ✅ Human validation passes (r > 0.7) OR multi-model panel agrees

## Failure Cases

If Phase 5 fails:
- **Steering direction wrong**: Steering vectors may be inverted or insufficient
- **No steering magnitude**: F1=0.787 may be too low for effective steering
- **No F1 correlation**: Detection and steering may be independent
- **Human disagreement**: Automated metrics need refinement (→ Phase 5.5)

## Next Steps After Phase 5

Assuming success:

**Phase 6**: Accuracy calibration study
- Test training curve (1×1×1, 5×5×5, 10×10×10, ...)
- Measure steering at each F1 level
- Find minimum F1 for effective steering (target: F1=0.85 sufficient?)

If F1=0.787 steers well → use 1×1×1 training for 10K scale
If F1=0.787 steers poorly → Phase 6 finds required F1

## Files

**Design**: `docs/PHASE5_DESIGN.md`
**Script**: `scripts/phase_5_semantic_steering_eval.py` (to be implemented)
**Results**: `results/phase_5_semantic_steering/` (pending)
**Human validation**: `results/phase_5_semantic_steering/human_validation.csv` (pending)

## Open Questions

1. Should we use both core AND boundary centroids, or just core?
   - **Proposal**: Start with just core vs negative (simpler)
   - **Future**: Compare core-only vs boundary-weighted steering

2. What's the right embedding model?
   - **Proposal**: sentence-transformers/all-MiniLM-L6-v2 (fast, standard)
   - **Alternative**: text-embedding-3-small (OpenAI, more expensive)

3. How many tokens to generate?
   - **Proposal**: 50 tokens (enough for semantic shift, not too verbose)
   - **Alternative**: 100 tokens (more content, slower)

4. Should we test multiple steering layers?
   - **Proposal**: Layer -1 only (Phase 5 baseline)
   - **Future**: Compare layers [-1, -5, -10] in Phase 6

## Timeline Estimate

**Part A (Automated)**: 2-3 hours runtime (10 concepts × 3 prompts × 3 strengths × ~20s/gen)
**Part B (Human validation)**: 30-60 minutes (50 samples × 1 min each)
**Part C (Multi-model)**: 1-2 hours (if needed, API calls + analysis)

**Total**: 4-6 hours (assuming Part C not needed)
