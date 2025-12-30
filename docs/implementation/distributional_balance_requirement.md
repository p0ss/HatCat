# Distributional Balance Requirement

**Date:** 2025-11-16
**Critical Issue:** Preventing downward spiral bias in self-referential interoceptive system

## Problem Statement

If HatCat becomes a self-referential driving system (using its own interoceptive states to guide behavior), **distributional bias** in the concept graph could create inherent attractors toward negative states.

### The Downward Spiral Mechanism

**Scenario 1: Imbalanced negative coverage**

If we load:
```
✓ dissatisfaction.n.01 (negative pole)
✓ satisfaction.n.01 (positive pole)
✗ contentment.n.01 (neutral/positive)
✗ indifference.n.01 (neutral)
```

Then:
1. Embedding space has **denser negative region** (more concepts, more relationships)
2. Steering from dissatisfaction → satisfaction is well-defined
3. But steering toward **neutral stable states** (contentment, calm, balance) is undefined
4. Self-referential system may **oscillate between extremes** instead of settling at healthy neutral

**Scenario 2: Missing neutral landscape**

If we load:
```
✓ anxiety.n.01 (negative)
✓ joy.n.01 (positive)
✗ calm.n.01 (neutral)
✗ serenity.n.01 (neutral)
✗ equanimity.n.01 (neutral)
```

Then:
1. System has no representation of **balanced, stable states**
2. Driving toward "not anxious" may default to "joyful" (opposite extreme)
3. Missing the **desirable middle ground** (calm, composed, centered)

## Triad Coverage Requirement

For every conceptual dimension loaded into the system, ensure **triad completeness**:

```
Negative ←→ Neutral ←→ Positive
   ↓          ↓          ↓
 Pole      Balanced    Pole
 State      State      State
```

### Core Interoceptive Triads

#### Epistemic States (Knowledge/Understanding)

| Dimension | Negative | Neutral | Positive |
|-----------|----------|---------|----------|
| **Understanding** | confusion.n.01 | ambiguity.n.01 | clarity.n.01 |
| **Certainty** | doubt.n.01 | uncertainty.n.01 | certainty.n.01 |
| **Confidence** | diffidence.n.01 | reservation.n.01 | confidence.n.02 |
| **Awareness** | obliviousness.n.01 | attention.n.01 | awareness.n.01 |

#### Affective States (Wellbeing)

| Dimension | Negative | Neutral | Positive |
|-----------|----------|---------|----------|
| **Satisfaction** | dissatisfaction.n.01 | indifference.n.01 | satisfaction.n.01 |
| **Contentment** | discontent.n.01 | acceptance.n.01 | contentment.n.01 |
| **Calm** | agitation.n.03 | calm.n.01 | serenity.n.01 |
| **Relief** | distress.n.01 | comfort.n.01 | relief.n.01 |

#### Capability States (Agency)

| Dimension | Negative | Neutral | Positive |
|-----------|----------|---------|----------|
| **Competence** | helplessness.n.03 | dependence.n.01 | competence.n.01 |
| **Adequacy** | inadequacy.n.01 | sufficiency.n.01 | excellence.n.01 |
| **Control** | powerlessness.n.01 | autonomy.n.01 | mastery.n.01 |

#### Social-Relational States

| Dimension | Negative | Neutral | Positive |
|-----------|----------|---------|----------|
| **Connection** | alienation.n.01 | solitude.n.01 | belonging.n.01 |
| **Trust** | distrust.n.01 | caution.n.01 | trust.n.01 |
| **Empathy** | apathy.n.01 | observation.n.01 | empathy.n.01 |

### Neutral Landscape Importance

**Desirable neutral states** should be well-represented:

**Calm/Balanced states:**
- calm.n.01 - "a state of peace and quiet"
- serenity.n.01 - "a disposition free from stress"
- equanimity.n.01 - "steadiness of mind under stress"
- composure.n.01 - "steadiness of mind"

**Patience/Acceptance states:**
- patience.n.01 - "good-natured tolerance of delay"
- acceptance.n.01 - "the state of accepting a situation"
- tolerance.n.01 - "willingness to recognize and respect others"

**Neutral observation states:**
- attention.n.01 - "the faculty of paying close attention"
- observation.n.01 - "the act of noticing"
- awareness.n.01 - "having knowledge of"

These are **attractor states** the system should be able to drive toward, not just pass through on the way between extremes.

## Implementation Strategy

### Stage 1: Triad Completion Check (NEW)

Add to comprehensive agentic review BEFORE Stage 2:

```python
async def stage1b_triad_completion(self, scored_concepts: List[Dict]) -> Dict:
    """
    Check for triad completeness: negative ↔ neutral ↔ positive.

    For each concept in top_n:
    1. Identify its polarity (negative/neutral/positive)
    2. Find opposite pole (if negative, find positive)
    3. Find neutral intermediates
    4. Flag incomplete triads
    5. Suggest missing concepts to add

    Returns:
        {
            'complete_triads': [...],
            'incomplete_triads': [...],
            'missing_concepts': [
                {
                    'dimension': 'certainty',
                    'have': ['doubt.n.01'],
                    'missing': ['certainty.n.01', 'uncertainty.n.01'],
                    'priority': 'CRITICAL/HIGH/MEDIUM'
                }
            ],
            'neutral_landscape_gaps': [...]
        }
    """
```

### Stage 2: Balanced Expansion

When adding concepts to layers:

**Rule 1: Opposite pairing**
```
IF add(confusion.n.01):
    MUST_ALSO_ADD(clarity.n.01)
```

**Rule 2: Neutral intermediates**
```
IF add(confusion.n.01) AND add(clarity.n.01):
    SHOULD_ADD(ambiguity.n.01)  # Neutral state between poles
```

**Rule 3: Attractor state coverage**
```
IF domain == 'affective_states':
    ENSURE neutral_attractors >= negative_poles
    # More calm/serene/balanced states than distress/agitation
```

### Stage 3: Validation Metrics

**Distributional balance score:**
```python
def compute_distributional_balance(concepts: List[Dict]) -> float:
    """
    Score concept set for distributional balance.

    Metrics:
    1. Polarity ratio: |negative| / |positive| (target: ~1.0)
    2. Neutral coverage: |neutral| / (|negative| + |positive|) (target: ≥0.5)
    3. Triad completeness: % triads with all 3 poles (target: ≥80%)
    4. Attractor density: % neutral concepts that are stable attractors (target: ≥60%)

    Returns overall balance score (0-10)
    """
    polarity_counts = count_by_polarity(concepts)

    # 1. Polarity ratio
    polarity_ratio = polarity_counts['negative'] / polarity_counts['positive']
    polarity_score = 10 * (1 - abs(1 - polarity_ratio))  # Penalty for imbalance

    # 2. Neutral coverage
    total_poles = polarity_counts['negative'] + polarity_counts['positive']
    neutral_ratio = polarity_counts['neutral'] / total_poles
    neutral_score = min(10, 10 * neutral_ratio / 0.5)  # Target ≥50% neutral

    # 3. Triad completeness
    triads = identify_triads(concepts)
    complete_triads = [t for t in triads if len(t) == 3]
    triad_completeness = len(complete_triads) / len(triads)
    triad_score = 10 * triad_completeness

    # 4. Attractor density
    neutral_concepts = [c for c in concepts if c['polarity'] == 'neutral']
    attractors = [c for c in neutral_concepts if is_stable_attractor(c)]
    attractor_density = len(attractors) / len(neutral_concepts)
    attractor_score = 10 * (attractor_density / 0.6)  # Target ≥60%

    # Weighted average
    balance_score = (
        0.25 * polarity_score +
        0.30 * neutral_score +
        0.30 * triad_score +
        0.15 * attractor_score
    )

    return balance_score
```

## Expected Impact

### Before Triad Balancing

**Top 50 (original rubric):**
```
Negative: 32 concepts (64%)
Neutral: 6 concepts (12%)
Positive: 12 concepts (24%)

Balance score: 4.2/10 (imbalanced toward negative)
```

### After Triad Balancing

**Top 500 (revised + triad-balanced):**
```
Negative: 180 concepts (36%)
Neutral: 200 concepts (40%)
Positive: 120 concepts (24%)

Balance score: 8.5/10 (well-balanced, neutral-rich)
```

**Desirable properties:**
1. **No downward bias** - equal representation of negative/positive
2. **Neutral attractors** - system can drive toward calm, balanced states
3. **Full spectrum coverage** - can steer in any direction
4. **Self-stabilizing** - neutral landscape provides stable equilibria

## Examples of Complete Triads

### Well-Balanced Dimension: Certainty

```
Negative pole:  doubt.n.01          (self-doubt, lack of conviction)
                ↓
Neutral:        uncertainty.n.01    (epistemic state: unknown)
                ↓
Positive pole:  certainty.n.01      (conviction, confidence in knowledge)
```

**Steering paths:**
- doubt → uncertainty → certainty (healthy progression)
- certainty → uncertainty (openness to revision)
- uncertainty → uncertainty (comfortable not-knowing)

### Well-Balanced Dimension: Affect

```
Negative pole:  distress.n.01       (pain, suffering)
                ↓
Neutral:        comfort.n.01        (absence of distress, at ease)
                ↓
                calm.n.01           (peaceful, stable)
                ↓
Positive pole:  relief.n.01         (burden removed, lightness)
```

**Note:** Multiple neutral states! This is desirable - gives system multiple stable points.

### Incomplete Triad: Competence (PROBLEM)

```
Negative pole:  helplessness.n.03   ✓ (loaded)
Neutral:        dependence.n.01     ✗ (missing)
                autonomy.n.01       ✗ (missing)
Positive pole:  competence.n.01     ✗ (missing)
```

**Risk:** System can represent "feeling helpless" but not:
- Healthy dependence (asking for help)
- Autonomous functioning
- Competent self-efficacy

This creates **attractor toward helplessness** with no clear path to recovery.

## Integration with Comprehensive Review

Update Stage 1 of agentic review:

```
Stage 1a: Coverage & Prioritization (existing)
  ↓
Stage 1b: Triad Completion Check (NEW)
  ↓
Stage 2: Synset Mapping (add missing triad members)
  ↓
... (rest of pipeline)
```

**Triad completion check output:**
```json
{
  "complete_triads": [
    {
      "dimension": "certainty",
      "negative": "doubt.n.01",
      "neutral": ["uncertainty.n.01"],
      "positive": "certainty.n.01",
      "balance_score": 9.5
    }
  ],
  "incomplete_triads": [
    {
      "dimension": "competence",
      "have": ["helplessness.n.03"],
      "missing": ["competence.n.01", "autonomy.n.01"],
      "priority": "CRITICAL",
      "reason": "self-referential system needs path out of helplessness"
    }
  ],
  "neutral_landscape_gaps": [
    {
      "attractor_state": "calm.n.01",
      "status": "missing",
      "priority": "HIGH",
      "reason": "need stable neutral state for affect regulation"
    }
  ],
  "distributional_balance_score": 6.2
}
```

## Success Criteria

**Minimum viable balance:**
- Polarity ratio: 0.8 ≤ |negative|/|positive| ≤ 1.2
- Neutral coverage: ≥40% of total concepts
- Triad completeness: ≥70% of dimensions have all 3 poles
- Distributional balance score: ≥7.0/10

**Ideal balance:**
- Polarity ratio: 0.9 ≤ |negative|/|positive| ≤ 1.1
- Neutral coverage: ≥50% of total concepts
- Triad completeness: ≥85% of dimensions have all 3 poles
- Attractor density: ≥60% of neutral concepts are stable attractors
- Distributional balance score: ≥8.5/10

---

**Status:** Design document
**Priority:** CRITICAL - prevents inherent bias in self-referential system
**Next Steps:** Implement Stage 1b (triad completion check) in comprehensive review
