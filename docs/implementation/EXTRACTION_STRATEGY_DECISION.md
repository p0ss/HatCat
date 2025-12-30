# Extraction Strategy Decision: Combined-20 (Prompt+Generation)

**Date**: 2025-11-20
**Status**: ✓ APPROVED - Default for all lens training

## Executive Summary

After comprehensive experimentation and analysis, we are adopting **combined-20 (prompt+generation extraction)** as the default training strategy for HatCat lens training.

**Key Result**: Combined-20 achieves 2x training data at zero additional computational cost by extracting activations from both prompt processing and generation phases.

## The Decision

### Chosen Strategy: Combined-20

Extract activations from **BOTH**:
1. **Prompt phase**: Forward pass through input prompt (e.g., "Define Attribute:")
2. **Generation phase**: Model's generated response tokens

This doubles training samples (60 vs 30) at the same generation cost.

### Performance Metrics

```
Metric                    Baseline-20    Combined-20    Delta
----------------------------------------------------------------
Primary use (gen-only)      0.975 F1      0.947 F1     -2.8%
Overall average             0.967 F1      0.980 F1     +1.3%
Generalization variance     0.0018        0.0004       4.5x better
Compute cost                1x            1x           Same
Training samples            30            60           2x
```

## Why Combined-20 Wins

### 1. Computational Efficiency (The Decisive Factor)

**The prompt forward pass is already done!** Extracting hidden states from it is essentially free:

```
Baseline-20:  30 prompts → 30 generations → 30 samples  (1x cost)
Combined-20:  30 prompts → 30 generations → 60 samples  (1x cost) ✓
Double-gen:   30 prompts → 60 generations → 60 samples  (2x cost)
```

To get 60 samples with generation-only would require 2x compute budget.

### 2. Excellent Performance

While combined-20 shows a 2.8% drop on generation-only tests (0.947 vs 0.975), this is:
- Still excellent performance for abstract concepts
- Offset by better overall average (+1.3%)
- Acceptable trade-off for 2x data at zero cost

### 3. Superior Generalization

Variance across test conditions: **0.0004 vs 0.0018** (4.5x more stable)

Combined-20 lenses work consistently well regardless of extraction method, indicating they've learned fundamental concept representations rather than distribution-specific patterns.

### 4. Bonus Feature

Can monitor user prompts in addition to model generations, providing earlier detection of concerning concepts.

## Trade-offs Acknowledged

### Primary Use Case Performance

Real-world deployment is ~80-90% monitoring model generations. Combined-20 shows 2.8% lower F1 on generation-only (0.947 vs 0.975).

**Why this is acceptable:**
1. Zero additional cost for 2x training data
2. 0.947 F1 is still very strong performance
3. Better robustness across conditions
4. Alternative (baseline-60) would require 2x compute

### Prompt Saturation at Scale

At production scale (50-90 samples), prompt diversity may saturate:
- Concepts have ~30-150 unique prompt variations possible
- At 90 samples, prompt-phase contribution plateaus
- Generation samples continue scaling linearly

**Impact**: At 90 samples, effective sample count may be ~150-160 (not full 180), but still much better than baseline's 90.

## Implementation Guide

### Current Extraction (Baseline)

```python
def extract_generation_only(model, tokenizer, prompts):
    for prompt in prompts:
        outputs = model.generate(
            **tokenizer(prompt, return_tensors="pt"),
            max_new_tokens=20,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        # Extract from generation hidden states
        activation = pool_hidden_states(outputs.hidden_states)
        yield activation
```

### New Extraction (Combined-20)

```python
def extract_prompt_and_generation(model, tokenizer, prompts):
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")

        # PHASE 1: Prompt processing (NEW!)
        prompt_outputs = model(**inputs, output_hidden_states=True)
        prompt_activation = pool_hidden_states(prompt_outputs.hidden_states)
        yield prompt_activation

        # PHASE 2: Generation (existing)
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        gen_activation = pool_hidden_states(gen_outputs.hidden_states)
        yield gen_activation
```

### Migration Path

1. Update `src/training/sumo_classifiers.py::extract_activations()` to use combined extraction
2. Retrain all existing lens packs with new strategy
3. Document change in lens pack metadata
4. Benchmark: Verify no performance regression on existing test sets

## Experimental Validation

Full experimental details: `docs/EXTRACTION_STRATEGY_EXPERIMENT.md`

**Tested strategies:**
- baseline-20: Generation only, 20 tokens
- combined-20: Prompt + generation, 20 tokens (2x samples)
- long-40: Generation only, 40 tokens

**Test methodology:**
- Cross-strategy evaluation (3×3 matrix)
- 100+100 test samples per concept
- Both abstract (Attribute) and specific (Carnivore) concepts

**Results confirmed:**
1. Combined-20 achieves best overall F1 (0.980)
2. Combined-20 most robust (lowest variance)
3. Long-40 performs worst (signal dilution from extended generation)
4. Hypothesis validated: Core concepts activate early in prompt processing

## Key Insights from Analysis

### 1. Early Activation Hypothesis Confirmed

74.8% overlap between prompt-phase and generation-phase concept activations confirms that "core concepts typically activate very early in the piece."

### 2. Longer Generation is Counterproductive

40-token generation performs worst (0.925 F1) due to:
- Signal dilution (averaging over more tokens dilutes concept signal)
- Distribution shift (extended narratives have different patterns)
- Noise introduction (later tokens are narrative elaboration, not concept)

### 3. Prompt Diversity Matters

Even at 30 samples, prompts are diverse:
- Definition prompts
- Example prompts
- Category relationship prompts
- WordNet relationship prompts

This diversity contributes to the robustness of combined-20 lenses.

## Future Considerations

### Weighted Extraction (Future Work)

For production optimization at 90+ sample scale, consider:
- 75% generation samples (optimize for primary use case)
- 25% prompt samples (maintain robustness)

Would require additional experimentation to validate.

### Prompt Deduplication (Future Work)

At very high sample counts (>100), implement prompt deduplication:
- Track unique prompt patterns
- Skip prompt extraction for duplicates
- Maintain full generation extraction

## Decision Authority

This decision was made through collaborative analysis with the HatCat development team after rigorous experimentation and cost-benefit analysis.

**Key decision factors:**
1. Computational efficiency (2x data for free)
2. Strong performance (0.947 F1 acceptable for use case)
3. Better generalization (4.5x lower variance)
4. Validated through comprehensive cross-strategy testing

## References

- Experiment documentation: `docs/EXTRACTION_STRATEGY_EXPERIMENT.md`
- Prompt-phase discovery: `docs/PROMPT_PHASE_ACTIVATION_EXPERIMENT.md`
- Implementation: `scripts/compare_extraction_strategies_cross.py`
- Results: `results/extraction_strategy_cross/run_20251120_205627/`
