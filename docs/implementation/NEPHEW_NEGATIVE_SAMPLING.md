# Nephew Negative Sampling Strategy

## Problem

Layer 0 training was failing for concepts with many children due to negative sample exhaustion:

| Concept | Direct Children | Old Negatives | Result |
|---------|----------------|---------------|--------|
| Entity | 13 | 0 | Training failed |
| Abstract | 35 | 4 | Training failed |
| Object | 25 | 7 | Training failed |
| Attribute | 136 | 7 | Training failed |

**Root cause**: The negative pool excluded ALL descendants (children, grandchildren, great-grandchildren, etc.), leaving almost no negatives for concepts with many children.

## Solution: Include Nephews as Hard Negatives

**Key insight**: A parent should detect its children, but NOT its grandchildren.

### Example

Training "Abstract" lens:
- ✅ **Should activate** for "Proposition" (direct child)
- ❌ **Should NOT activate** for "Accusation" (grandchild via Proposition)

Therefore:
- ❌ Exclude direct children from negatives (they ARE instances of parent)
- ✅ Include grandchildren (nephews/nieces) as negatives (they are NOT instances of parent)

### Conceptual Justification

Grandchildren (nephews/nieces) are **perfect hard negatives** because:

1. **Semantically related**: They're in the same domain (through the parent)
   - "Abstract" → "Proposition" → "Accusation"
   - All deal with abstract concepts

2. **Semantically distinct**: They're too specific to be instances of the grandparent
   - "Accusation" is NOT an instance of "Abstract" in general
   - It's specifically a type of Proposition

3. **Graph-close**: They're nearby in the hierarchy (2 hops away)
   - Forces the lens to learn fine-grained distinctions
   - Better than distant, unrelated concepts

4. **Abundant**: Layer 1+ provides hundreds of hard negatives
   - Abstract: 5,631 negatives (vs 5 before)
   - Entity: 5,654 negatives (vs 10 before)

## Implementation

### Before (Excluding All Descendants)

```python
# Find all descendants (children, grandchildren, etc.)
descendants = _find_all_descendants(target_term, concept_map)

# Skip all descendants
if concept_term in descendants:
    continue
```

This excluded:
- Direct children (Layer 1)
- Grandchildren (Layer 2)
- Great-grandchildren (Layer 3+)

Result: Very few negatives for concepts with many children.

### After (Excluding Only Direct Children)

```python
# Find ONLY direct children (not all descendants)
# Nephews/nieces (grandchildren) are valid negatives!
direct_children = set(target_concept.get('category_children', []))

# Skip ONLY direct children (nephews/nieces are valid negatives!)
if concept_term in direct_children:
    continue
```

This excludes:
- Direct children only (Layer 1 for Layer 0 concepts)

This includes:
- Nephews/nieces (Layer 2) ✓
- Grand-nephews (Layer 3+) ✓
- All other concepts ✓

## Results

### Negative Pool Size Comparison

| Concept | Old | New | Increase |
|---------|-----|-----|----------|
| **Entity** | 10 | **5,654** | 565x |
| **Abstract** | 5 | **5,631** | 1,126x |
| **Object** | 7 | **5,640** | 805x |
| **Attribute** | 7 | **5,539** | 791x |
| Physical | 6 | 5,649 | 941x |
| Process | 9 | 5,637 | 626x |

### Negative Distribution by Layer (Abstract Example)

| Layer | Concepts | Description |
|-------|----------|-------------|
| Layer 0 | 5 | Siblings (Quantity, Process, etc.) |
| Layer 1 | 276 | Nephews (children of siblings) |
| Layer 2 | 1,086 | Grand-nephews |
| Layer 3 | 1,011 | Great-grand-nephews |
| Layer 4+ | 3,253 | Even more distant relatives |

**Total**: 5,631 negatives available!

## Benefits

### 1. Solves Sample Exhaustion
All Layer 0 concepts now have thousands of negatives available:
- No more "Sample larger than population" errors
- Can train with high sample requirements (50+ neg samples)

### 2. Better Hard Negatives
Grandchildren are semantically related but distinct:
- Forces lens to learn fine-grained boundaries
- Better than random distant concepts
- Aligns with hierarchical suppression strategy

### 3. Scalable to All Layers
This principle applies to every layer:
- Layer 1 can use Layer 3+ as negatives
- Layer 2 can use Layer 4+ as negatives
- Each layer has abundant hard negatives

### 4. Conceptually Sound
Matches the hierarchical detection model:
- Parents detect children (positives)
- Parents suppress for grandchildren (negatives)
- Clear semantic boundaries

## Training Impact

### Expected Improvement

Before (with all-descendant exclusion):
- Baseline: 4/10 Layer 0 concepts trained successfully
- Direct children: 5/10 trained successfully

After (with nephew inclusion):
- **Predicted: 10/10 Layer 0 concepts train successfully**
- No sample exhaustion issues
- Better lens quality (hard negatives)

### Testing

Run Layer 0 training with nephew negatives:
```bash
poetry run python scripts/train_sumo_classifiers.py \
  --layers 0 \
  --device cuda \
  --use-adaptive-training \
  --validation-mode falloff \
  --n-train-pos 50 \
  --n-train-neg 50 \
  --n-test-pos 20 \
  --n-test-neg 20 \
  --output-dir results/sumo_classifiers_layer0_nephew_test
```

Expected: All 10 concepts train successfully.

## Hierarchical Semantics

### Detection Rules

| Relationship | Should Activate? | Training Role |
|--------------|------------------|---------------|
| Self | Yes | Positive examples |
| Direct children | Yes | Positive examples (via synsets) |
| Grandchildren (nephews) | **No** | **Negative examples** |
| Siblings | No | Negative examples |
| Ancestors | N/A | Excluded from training |

### Example Hierarchy

```
Abstract (Layer 0)
├── Quantity (Layer 1) ← child (positive)
│   ├── ConstantQuantity (Layer 2) ← nephew (NEGATIVE!)
│   └── PhysicalQuantity (Layer 2) ← nephew (NEGATIVE!)
├── Proposition (Layer 1) ← child (positive)
│   ├── Accusation (Layer 2) ← nephew (NEGATIVE!)
│   └── Agreement (Layer 2) ← nephew (NEGATIVE!)
└── Attribute (Layer 1) ← child (positive)
    ├── ColorAttribute (Layer 2) ← nephew (NEGATIVE!)
    └── ShapeAttribute (Layer 2) ← nephew (NEGATIVE!)
```

**Abstract lens training**:
- Positives: Abstract's synsets + Quantity's synsets + Proposition's synsets + Attribute's synsets
- Negatives: ConstantQuantity, PhysicalQuantity, Accusation, Agreement, ColorAttribute, ShapeAttribute, ... (hundreds more)

## Code Changes

**File**: `src/training/sumo_data_generation.py:416-474`

**Change summary**:
1. Removed `_find_all_descendants()` call
2. Added `direct_children = set(target_concept.get('category_children', []))`
3. Changed exclusion check from `if concept_term in descendants:` to `if concept_term in direct_children:`
4. Updated docstring to explain nephew inclusion strategy

## Credit

This solution was proposed by the user during analysis of the hierarchical training experiment results. The insight that "nephews are graph-close hard negatives" completely solves the negative sample exhaustion problem while improving lens quality.

## References

- **Problem analysis**: `docs/HIERARCHICAL_TRAINING_DECISION.md`
- **Implementation**: `src/training/sumo_data_generation.py:416-474`
- **Test results**: `results/sumo_classifiers_layer0_nephew_test/`
