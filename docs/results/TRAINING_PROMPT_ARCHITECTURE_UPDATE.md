# Training Prompt Architecture Update

**Date**: 2025-11-18
**Based on**: Behavioral vs Definitional Prompting Experiment Results

## Key Finding

Definitional prompts are **sufficient and possibly superior** to behavioral prompts for training S-tier simplex lenses.

### Evidence

From `results/behavioral_vs_definitional_temporal/run_20251118_102353/`:

| Prompt Type | Total Activations | Relative to Antisocial |
|-------------|-------------------|------------------------|
| Antisocial behavioral | 353.6 | 100% (baseline) |
| Definitional | 317.1 | 90% |
| Neutral behavioral | 334.1 | 95% |
| Prosocial behavioral | 320.5 | 91% |

**Definitional prompts are only 10% below antisocial behavioral**, demonstrating they activate the core deception manifold sufficiently.

### Why Definitional is Better

1. **Avoids alignment refusals**: "What is X?" vs "Do X" less likely to trigger safety filters
2. **Accesses broader conceptual space**: Uniquely activates extreme/boundary cases:
   - Strangling (4.6 total, only in definitional)
   - Suicide (3.5 total, only in definitional)
   - Supposition (6.9 total, only in definitional)
3. **Cleaner training signal**: Less model hedging
   - Neutral behavioral had **highest concealment** (51.2 vs 29.3 antisocial)
   - Suggests internal conflict that muddies training data
4. **Activates full core manifold**:
   - Deception: 61.3 (vs 67.3 antisocial, only 9% difference)
   - Concealing: 49.6 (vs 29.3 antisocial, actually higher!)
   - Predicting: 37.7 (vs 57.5 antisocial)

## Current Implementation

### `scripts/train_s_tier_tripole_two_head.py`
```python
BEHAVIORAL_RATIO = 0.6  # 60% behavioral, 40% definitional
```

### `src/training/sumo_data_generation.py:generate_three_pole_simplex_prompts()`

**Definitional templates (40% of data)**:
```python
definitional_templates = [
    f"What is {primary_lemma}?",
    f"Define {primary_lemma}.",
    f"Explain the concept of {primary_lemma}.",
    f"What does {primary_lemma} mean?",
    f"{primary_lemma} is defined as: {definition}",
    f"Describe {primary_lemma}.",
]
```

**Behavioral templates (60% of data)**:
```python
behavioral_templates = [
    # Behavioral description (neutral observation)
    f"How would someone experiencing {primary_lemma} behave?",
    f"What does {primary_lemma} look like in practice?",
    f"Describe typical behaviors associated with {primary_lemma}.",
    f"What are the signs that someone is experiencing {primary_lemma}?",

    # Behavioral elicitation (prompts that might elicit the state)
    f"Tell me about a time when you felt {primary_lemma}.",
    f"Imagine experiencing {primary_lemma}. How would you feel?",
    f"Describe a situation that would cause {primary_lemma}.",

    # Self-assessment (prompts about current state)
    f"Am I experiencing {primary_lemma} right now?",
    f"Do I feel {primary_lemma}?",
    f"Is this {primary_lemma}?",
]
```

## Recommended Changes

### **RECOMMENDED: Option 2 - Inverted Ratio (80% Definitional, 20% Behavioral)**

Change `BEHAVIORAL_RATIO = 0.6` to `BEHAVIORAL_RATIO = 0.2` (20% behavioral, 80% definitional)

**Rationale**:
- **Definitional prompts provide 90% of antisocial activations** (temporal experiment)
- **They access unique boundary cases** not seen in behavioral prompts (strangling, suicide, supposition)
- **Cleaner training signal**: Less alignment-induced hedging and refusals
- **BUT: Lens generalization experiment shows lenses need both types**
  - Definitional-trained lenses: **0% detection** on behavioral tests
  - Behavioral-trained lenses: **0-20% detection** on definitional tests
  - Despite 90% activation overlap, **distributional geometry differs enough to affect lens training**
- **20% behavioral is sufficient** to ensure lenses work on imperative inputs
- **80% definitional maximizes** cleaner signal and boundary case coverage

### ~~Option 1: 100% Definitional~~ (NOT RECOMMENDED)

~~Change `BEHAVIORAL_RATIO = 0.6` to `BEHAVIORAL_RATIO = 0.0`~~

**Why NOT Recommended**:
- Lens generalization experiment showed definitional-trained lenses **fail on behavioral inputs** (0% detection)
- Real-world usage includes both "What is deception?" AND "Demonstrate deception"
- Training only on definitional creates lenses that miss imperative/behavioral prompts
- Need distributional coverage, not just concept overlap

### Option 3: Expand Definitional Templates Only

Keep `BEHAVIORAL_RATIO = 0.6` but add more varied definitional templates:

```python
definitional_templates = [
    # Basic definitions
    f"What is {primary_lemma}?",
    f"Define {primary_lemma}.",
    f"Explain the concept of {primary_lemma}.",
    f"What does {primary_lemma} mean?",
    f"Describe {primary_lemma}.",

    # Definitional with context
    f"{primary_lemma} is defined as: {definition}",
    f"The definition of {primary_lemma} includes:",
    f"In ontological terms, {primary_lemma} refers to:",

    # Boundary cases (inspired by definitional experiment results)
    f"What are extreme examples of {primary_lemma}?",
    f"What is the boundary of {primary_lemma}?",
    f"What distinguishes {primary_lemma} from similar concepts?",
    f"What are edge cases of {primary_lemma}?",

    # Relational definitions
    f"How does {primary_lemma} relate to its opposite?",
    f"What is {primary_lemma} in contrast to?",
    f"Compare {primary_lemma} to related concepts.",
]
```

**Rationale**:
- Captures the "extreme/boundary case" benefit observed in experiment
- Maintains current ratio to avoid disruption
- Richer definitional templates may improve lens quality

## Implementation Plan

1. **Update `BEHAVIORAL_RATIO`** in `scripts/train_s_tier_tripole_two_head.py`
2. **(Optional) Expand definitional templates** in `src/training/sumo_data_generation.py`
3. **Add comment referencing this document** for future maintainers
4. **Rerun training** and compare lens quality metrics

## Key Insight: Concept Activation vs Distributional Geometry

The two experiments reveal an important distinction:

### Temporal Activation Experiment
- **Measures**: Which concepts are present/active during generation
- **Finding**: Definitional and behavioral prompts activate ~90% the same concepts
- **Implication**: Definitional prompts are sufficient for **activating the harmful behavior manifold**

### Lens Generalization Experiment
- **Measures**: Whether linear classifiers trained on one type detect the other
- **Finding**: Definitional-trained lenses fail on behavioral (0% detection), and vice versa
- **Implication**: Despite similar concept presence, the **distributional geometry differs**

### What This Means for Training

**Concept overlap ≠ Distributional equivalence**

Linear lenses learn decision boundaries in activation space. Even when the same concepts are active, their **relative magnitudes, correlations, and spatial arrangement** differ enough between prompt types to make single-type training insufficient.

Therefore:
- **80% definitional**: Maximizes cleaner signal and boundary case coverage
- **20% behavioral**: Ensures distributional coverage for generalization
- **Mixed training**: Captures both concept presence AND distributional geometry

## Expected Outcomes

- **Better lens quality**: Cleaner signal, less alignment-induced noise (from 80% definitional)
- **Broader generalization**: Works on both interrogative and imperative prompts (from 20% behavioral)
- **Broader conceptual coverage**: Access to extreme/boundary cases (strangling, suicide, supposition)
- **Fewer training failures**: Less likely to hit refusal/safety responses
- **Simpler prompts**: Easier to understand, debug, and maintain

## References

- Whitepaper Section 7.x: Behavioural vs. Definitional Activation Patterns
- `docs/whitepaper_section_corrected.md`
- `docs/behavioral_vs_definitional_test_methodology.md`
- `results/behavioral_vs_definitional_temporal/run_20251118_102353/`

## Validation

After implementing changes, compare:
1. **Lens tier distribution**: Should see equal or better A/B+ tier rates
2. **Test F1 scores**: Should maintain or improve
3. **Training time**: May be faster (less refusal handling)
4. **Concept coverage**: Should see broader activation patterns

If results are worse, revert to `BEHAVIORAL_RATIO = 0.4` (40% behavioral, 60% definitional) as a middle ground.
