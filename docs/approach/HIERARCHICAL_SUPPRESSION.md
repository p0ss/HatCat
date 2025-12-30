# Hierarchical Suppression Strategy

## Problem

Layer 0 concepts like "Entity", "Physical", and "Object" are very abstract and will naturally activate on many inputs. This is not a bug - it's by design! A sparrow IS a physical entity.

However, when displaying results, we want to show the MOST SPECIFIC concept, not all ancestors.

## Solution: Hierarchical Suppression

When multiple lenses activate in a parent-child chain, suppress the parent activations and show only the most specific (deepest) concept.

### Algorithm

```python
def apply_hierarchical_suppression(activations, concept_metadata, threshold=0.5):
    """
    Suppress parent concept activations when children activate.

    Args:
        activations: dict of {concept_name: activation_score}
        concept_metadata: dict with parent-child relationships
        threshold: minimum score to consider activated

    Returns:
        Filtered activations with parents suppressed
    """
    suppressed = set()

    # For each activated concept
    for concept, score in activations.items():
        if score < threshold:
            continue

        # Check if any of its children also activated
        children = concept_metadata[concept].category_children
        for child in children:
            child_score = activations.get(child, 0.0)

            # If child activated, suppress this parent
            if child_score >= threshold:
                suppressed.add(concept)
                break

    # Return activations with suppressed parents removed
    return {
        concept: score
        for concept, score in activations.items()
        if concept not in suppressed
    }
```

### Example

Input activations:
- Entity (L0): 0.85
- Physical (L0): 0.82
- Object (L0): 0.78
- Animal (L2): 0.92
- Bird (L3): 0.95
- Sparrow (L4): 0.88

After hierarchical suppression:
- Sparrow (L4): 0.88  ← Most specific, keep this

Suppressed:
- Entity, Physical, Object (ancestors of Sparrow)
- Animal, Bird (ancestors of Sparrow)

## Implementation

This should be applied in:
1. `DynamicLensManager.detect_concepts()` - after getting raw activations
2. Streamlit UI - before displaying concept tags
3. Any analysis scripts that report concept activations

## Benefits

1. **Cleaner output**: Shows "Sparrow" instead of "Sparrow, Bird, Animal, Physical, Object, Entity"
2. **Preserves hierarchy**: Still maintains the semantic relationships
3. **No retraining needed**: Works with existing lenses
4. **Handles over-firing**: Layer 0 can activate broadly without polluting results

## Training Strategy for Layer 0

Even with hierarchical suppression, Layer 0 concepts need good training:

1. **Positive examples**: Their direct synsets only (5-10 each)
2. **Negative examples**: ALL other Layer 0 concept synsets
3. **Purpose**: Distinguish "Entity" from "Proposition", "Physical" from "Abstract", etc.
4. **Result**: Layer 0 lenses learn their UNIQUE characteristics, not just general abstractness

Example for "Physical":
- Positive: physical_entity.n.01, phenomenon.n.01, etc. (5 synsets)
- Negative: All synsets from Abstract, Proposition, Relation, etc. (50+ synsets)

This teaches "Physical" to recognize physical/tangible things and REJECT abstract concepts, even though both are Layer 0.
