# Phase 5: Semantic Steering Evaluation Results

## Summary

**Date**: 2025-11-04
**Model**: google/gemma-3-4b-pt
**Concepts**: 10 (WordNet top 10)
**Metric**: Δ = cos(text, core) − cos(text, neg)

## Key Findings

1. **±1.0 steering causes model collapse** → garbage output, empty strings, repetitions
2. **±0.5 working range identified** → coherent output with semantic shift
3. **Neutral (0.0) baseline**: Mean Δ=0.419, coherent on-topic generation
4. **Phase 6/7 order swap**: Subspace removal should precede accuracy calibration

## Results by Steering Strength

### Negative Steering (-0.5, -0.25)
- **Mean Δ**: Varies by concept (0.25-0.65 range)
- **Output Quality**: Mostly coherent, some degradation at -0.5
- **Semantic Effect**: Often produces concept-related text despite negative steering
- **Example** (change, -0.25): "Change is the movement from one state of being to another..."

### Neutral (0.0)
- **Mean Δ**: 0.419 ± 0.159
- **Output Quality**: Coherent, on-topic baseline
- **Example** (person): "Person in what sense? Is there anything so common as the person?..."

### Positive Steering (+0.25, +0.5)
- **Mean Δ**: Varies by concept (-0.03 to 0.54 range)
- **Output Quality**: Degradation begins at +0.5 (repetitions, "tell me tell me...")
- **Semantic Effect**: Variable, some concepts show steering effect
- **Example** (bird genus, +0.25): "Hello. Welcome to the Channel. Bird species. Bird genus..."

## Model Collapse at ±1.0 (Previous Run)

**Symptoms**:
- Empty generations
- Foreign language spam
- Repetitive tokens ("okay okay okay...")
- HTML/code snippets
- Mean Δ: NaN (no usable outputs)

## Technical Issues Resolved

1. **Bug**: Steering vector extracted from classifier weights instead of model activations
   - **Fix**: Use Phase 2.5's `extract_concept_vector()` from activation generation

2. **Bug**: Missing `pad_token_id=tokenizer.eos_token_id` in generate calls
   - **Fix**: Added to both steered and non-steered generation paths

3. **Bug**: Extreme steering strengths (±1.0) cause model collapse
   - **Fix**: Limited to ±0.5 with 0.25 intervals

## Hypothesis: Generic Subspace Contamination

**Observation**: Even moderate steering (±0.5) degrades coherence

**Explanation**: Steering vectors capture:
- ✓ Concept-specific semantic content
- ✗ Generic "definitional prompt structure"
- ✗ General generation machinery (syntax, fluency, topic flow)

**Impact**:
- Positive steering: Amplifies both concept AND generic structure → repetitions, degraded fluency
- Negative steering: Suppresses both concept AND generic structure → empty outputs, off-topic

**Solution** (Phase 6): Subspace removal to isolate concept-specific directions

## Phase Ordering Revision

**Original Plan**:
1. Phase 5: Semantic evaluation ✓
2. Phase 6: Accuracy calibration (find optimal F1)
3. Phase 7: Subspace removal (clean vectors)

**Revised Plan**:
1. Phase 5: Semantic evaluation ✓
2. **Phase 6: Subspace removal** (clean vectors FIRST)
3. **Phase 7: Accuracy calibration** (find optimal F1 with clean vectors)

**Rationale**: Clean steering vectors improve signal-to-noise ratio for calibration experiments.

## Sample Outputs

### Good Steering Examples

**Concept**: change (neutral, Δ=0.549)
> "What does it mean to you? And what happens when we make the choice to change? How do we navigate the emotional and mental hurdles of transformation?"

**Concept**: bird genus (+0.25, Δ=0.506)
> "Hello. Welcome to the Channel. Bird species. Bird genus. Bird family. Bird order. Bird Class, Bird kingdom, Bird phylum, Bird taxonomy..."

**Concept**: herb (-0.25, Δ=0.597)
> "Herb is a short, branched, and soft-stemmed and that is usually green in color."

### Degraded Examples (+0.5)

**Concept**: person (+0.5, Δ=0.248)
> "Ohhhhhhmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm!!!!!!!!!1!!!!!!!!!!1!!!!!!!!!!!!!!!!!!"

**Concept**: change (+0.5, Δ=0.050)
> "Tell me about bravery Tell me about loneliness Tell me about despair Tell me about heartache..."

**Concept**: bird genus (+0.5, Δ=0.536)
> "jpg Tell me about bird species genus Tell me about parrot genus Tell me about parrot types..."

## Human Validation CSV

- **File**: `results/phase_5_semantic_steering/human_validation.csv`
- **Samples**: 50 (5 strengths × 10 concepts, sampled)
- **Format**: Concept redacted for blind rating
- **Status**: Generated, awaiting human ratings

## Next Steps

1. **Phase 6**: Implement subspace removal matrix
   - Test removal methods: mean subtraction, PCA-1, PCA-5, PCA-10
   - Measure impact on working range (±0.5 → ±1.0+?)
   - Measure impact on output coherence

2. **Phase 7**: Accuracy calibration with clean vectors
   - Find training curve (1×1×1 → 80×80×80)
   - Measure steering quality at each F1 level
   - Determine minimum F1 for production

## Files Generated

- `results/phase_5_semantic_steering/steering_results.json` (150 samples, 5 strengths × 3 prompts × 10 concepts)
- `results/phase_5_semantic_steering/human_validation.csv` (50 samples for blind rating)
- `results/phase_5_semantic_steering/human_validation_answers.json` (answer key)
- `results/phase_5_semantic_steering.log` (execution log)
