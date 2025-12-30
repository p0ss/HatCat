# Shared Concept Detection Test Runner Library

## Overview

Created a shared testing library in `src/testing/` that provides the single source of truth for running concept detection experiments with DynamicLensManager.

## Problem Solved

Previously, different test scripts (temporal monitoring, behavioral tests, etc.) each implemented their own version of concept detection, leading to:
- Code duplication
- Inconsistent behavior
- Difficult debugging (lens saturation issues only in some tests)
- No guarantee that fixes in one test would propagate to others

## Solution

Extracted the WORKING approach from `tests/test_temporal_monitoring.py` into a shared library that all tests can use.

## Library Structure

```
src/testing/
├── __init__.py                 # Public API exports
└── concept_test_runner.py      # Core implementation
```

## Public API

### `generate_with_concept_detection()`

Generate text and detect concepts during each generation step.

**Use case:** Live generation with per-token concept tracking (like temporal monitoring)

**Example:**
```python
from src.testing import generate_with_concept_detection

result = generate_with_concept_detection(
    model=model,
    tokenizer=tokenizer,
    lens_manager=lens_manager,
    prompt="What is artificial intelligence?",
    max_new_tokens=30,
    temperature=0.8,
    threshold=0.3,
    record_per_token=True  # Set False for final state only
)

# Returns:
# {
#     'prompt': str,
#     'generated_text': str,
#     'tokens': List[str],
#     'final_concepts': Dict[str, {'probability': float, 'layer': int}],
#     'timesteps': List[...],  # if record_per_token=True
#     'summary': {...}
# }
```

### `score_activation_with_lens_manager()`

Score a single pre-computed activation vector against concept lenses.

**Use case:** Analyzing saved activations (like behavioral test)

**Example:**
```python
from src.testing import score_activation_with_lens_manager
import torch

# From saved numpy array
activation = np.load('saved_activation.npy')
activation_tensor = torch.from_numpy(activation)

result = score_activation_with_lens_manager(
    activation=activation_tensor,
    lens_manager=lens_manager,
    top_k=10,
    threshold=0.3
)

# Returns:
# {
#     'top_concepts': List[str],           # Top K concept names
#     'top_probabilities': List[float],    # Probabilities (sorted descending)
#     'all_scores': Dict[str, float],      # All concepts above threshold
#     'concept_details': Dict[str, {...}]  # Includes layer info
# }
```

### `batch_score_activations()`

Score multiple activation vectors in batch.

**Example:**
```python
from src.testing import batch_score_activations

activations = [torch.randn(2560) for _ in range(100)]

results = batch_score_activations(
    activations=activations,
    lens_manager=lens_manager,
    top_k=10,
    threshold=0.3
)
# Returns list of result dicts
```

## Key Implementation Details

### Why This Works

The working approach from temporal monitoring has these key characteristics:

1. **Correct tensor conversion:**
   ```python
   # Ensure float32 to match classifier dtype
   hidden_state_f32 = hidden_state.float()
   ```

2. **Proper detect_and_expand call:**
   ```python
   # Pass tensor as positional argument, not kwarg
   detected, _ = lens_manager.detect_and_expand(
       hidden_state_f32,  # Not hidden_state=...
       top_k=top_k,
       return_timing=True
   )
   ```

3. **Correct result unpacking:**
   ```python
   # Returns List[(concept_name, probability, layer)]
   for concept_name, prob, layer in detected:
       if prob > threshold:
           concept_scores[concept_name] = {
               'probability': float(prob),
               'layer': int(layer)
           }
   ```

## Migration Guide

### Before (Behavioral Test)

```python
def score_activations_with_lens_manager(activations, lens_manager, ...):
    # 60 lines of duplicated logic
    activation_tensor = torch.from_numpy(activations).float().to(...)
    detections, _ = lens_manager.detect_and_expand(...)
    # ... manual filtering and formatting ...
    return {...}
```

### After (Using Shared Library)

```python
from src.testing.concept_test_runner import score_activation_with_lens_manager

def score_activations_with_lens_manager(activations, lens_manager, ...):
    """Wrapper using shared library."""
    activation_tensor = torch.from_numpy(activations)
    return score_activation_with_lens_manager(
        activation_tensor,
        lens_manager,
        top_k=top_k,
        threshold=threshold
    )
```

## Tests Updated

1. **scripts/test_behavioral_vs_definitional_training2.py**
   - Now uses `score_activation_with_lens_manager()` from shared library
   - Eliminates 60 lines of duplicated code
   - Ensures consistent behavior with temporal monitoring

2. **tests/test_temporal_monitoring.py**
   - Could be updated to use `generate_with_concept_detection()` (not required, already working)
   - Serves as the reference implementation

## Benefits

1. **Single Source of Truth:** All tests use the same proven approach
2. **Easy Debugging:** Fix once, fixes everywhere
3. **Consistency:** No more different behavior between tests
4. **Maintainability:** Changes to concept detection logic only need to be made once
5. **Documentation:** Clear API with examples for future tests

## Future Work

Consider updating other test scripts to use this library:
- `tests/test_self_concept_monitoring.py`
- `scripts/test_behavioral_vs_definitional_concepts.py`
- Any other custom test scripts

## Related Files

- Implementation: `src/testing/concept_test_runner.py`
- API: `src/testing/__init__.py`
- First consumer: `scripts/test_behavioral_vs_definitional_training2.py`
- Reference source: `tests/test_temporal_monitoring.py`
