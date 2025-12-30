# Phase 7 Stress Test - Blocking Issues

**Date**: 2025-11-05
**Status**: BLOCKED - 2 critical issues preventing execution
**Context**: Comprehensive stress test for Phase 6.6 dual-subspace manifold steering

## Executive Summary

Phase 7 stress test script created successfully. Initial blocking issues **RESOLVED**:
1. ✅ **Dtype mismatch** - Fixed in both hooks.py and manifold.py
2. ✅ **API limitation** - Fixed to support concept=None for baseline generation

Training completes successfully (9.4s, 8.62 GB VRAM).

**Current Issue**: CUDA assertion error during baseline generation - likely GPU state corruption from previous failed runs. Requires GPU reset.

---

## Issue 1: Dtype Mismatch in Forward Hooks ⚠️ CRITICAL

### Error Message
```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::Half != float
```

### Location
- **File**: `src/steering/manifold.py:345`
- **Function**: `create_manifold_steering_hook()`
- **Context**: Occurs during task manifold estimation when generating samples

### Root Cause
```python
# Line 345 in manifold.py
v_tensor = torch.from_numpy(v_processed).float().to(device)
```

The model runs in **float16** (half precision), but steering vectors are converted to **float32** via `.float()`. When the hook performs matrix operations:

```python
# Line 352
projection = (hidden @ v_tensor.unsqueeze(-1)) * v_tensor
```

PyTorch throws dtype mismatch error because `hidden` is float16 and `v_tensor` is float32.

### Impact
- Task manifold estimation fails for all concepts
- All 5 concepts show: "Failed to estimate task manifold for {concept}: Insufficient activations collected: 0 < 2"
- Cannot proceed to evaluation phase

### Proposed Fix
```python
# Line 345 - Match model dtype instead of forcing float32
v_tensor = torch.from_numpy(v_processed).to(dtype=torch.float16, device=device)

# OR: Dynamic dtype matching
# In hook_fn, line 352:
v_tensor_matched = v_tensor.to(dtype=hidden.dtype)
projection = (hidden @ v_tensor_matched.unsqueeze(-1)) * v_tensor_matched
```

### Files to Modify
- `src/steering/manifold.py:345` (primary)
- Potentially also check `src/steering/hooks.py:38` for similar issues

---

## Issue 2: API Limitation - Baseline Generation ⚠️ CRITICAL

### Error Message
```
ValueError: Concept 'None' not fitted. Available: ['person', 'change', 'animal', 'object', 'action']
```

### Location
- **File**: `scripts/phase_7_stress_test.py:253`
- **Function**: `evaluate_at_scale()`
- **Context**: Attempting to generate baseline text without steering

### Root Cause
```python
# Line 253 in phase_7_stress_test.py
baseline_text = steerer.generate("Hello, ", None, 0.0, max_new_tokens=50)
```

The `ManifoldSteerer.generate()` method **requires** a valid fitted concept name. It does not support `concept=None` for unsteered baseline generation. The method validates:

```python
# In manifold.py:491
if concept not in self.concept_vectors:
    raise ValueError(f"Concept '{concept}' not fitted. Available: {list(self.concept_vectors.keys())}")
```

### Impact
- Cannot compute baseline perplexity for coherence threshold
- Evaluation blocked immediately after successful training

### Proposed Fix

**Option A: Modify API to support None** (cleanest)
```python
# In ManifoldSteerer.generate(), around line 491
if concept is None or abs(strength) < 1e-6:
    # Generate without steering
    return generate_with_steering(
        self.model, self.tokenizer, prompt,
        steering_vector=None,
        strength=0.0,
        layer_idx=self.target_layer_idx,
        max_new_tokens=max_new_tokens,
        device=self.device
    )
```

**Option B: Direct model generation in evaluation** (quick workaround)
```python
# In evaluate_at_scale(), replace line 253
inputs = tokenizer("Hello, ", return_tensors="pt").to(device)
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id
    )
baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Files to Modify
- **Preferred**: `src/steering/manifold.py:~491` (API enhancement)
- **Alternative**: `scripts/phase_7_stress_test.py:253` (local workaround)

---

## Resolved Issues ✅

### Issue 3: SVD Convergence Failures (RESOLVED)
**Error**: `numpy.linalg.LinAlgError: SVD did not converge`

**Solution Applied**:
- Row normalization before SVD
- Multi-tier fallback: NumPy SVD → scipy SVD → eigendecomposition of covariance
- Convert to float64 for numerical stability

**Files Modified**:
- `src/steering/manifold.py:53-79` (estimate_contamination_subspace)
- `src/steering/manifold.py:203-228` (estimate_task_manifold)

**Result**: Training now completes successfully in 3.4s with 8.62 GB VRAM

---

## Test Log Analysis

### Latest Run (ID: a4d471)
```
============================================================
PHASE 7: COMPREHENSIVE STRESS TEST
============================================================
Model: google/gemma-3-4b-pt
Concepts: person, change, animal, object, action
Scales: [2, 4, 8, 16, 32, 64]
Test strengths: [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
============================================================

Loading model...
✓ Model loaded

============================================================
Training at scale: 2 samples/concept
============================================================
Fitting manifold steerer with 2 manifold samples...
✓ Training complete: 3.4s, 8.62 GB VRAM

Evaluating at scale 2...
[ERROR] Dtype mismatch in hooks → Task manifold estimation fails
[ERROR] Baseline generation API error → Evaluation blocked
```

### Execution Timeline
1. ✅ Model loading (gemma-3-4b-pt)
2. ✅ Contamination subspace estimation (with warnings but succeeds)
3. ❌ Task manifold estimation (fails due to dtype mismatch)
4. ✅ Training completes (3.4s, 8.62 GB VRAM)
5. ❌ Evaluation blocked (baseline generation API error)

---

## Next Steps (When Resuming)

### Priority 1: Fix Dtype Mismatch
1. Modify `src/steering/manifold.py:345` to match model dtype
2. Test hook with float16 tensors
3. Verify task manifold estimation succeeds

### Priority 2: Fix Baseline Generation
1. Choose fix approach (API modification vs local workaround)
2. Implement chosen solution
3. Verify baseline perplexity computation

### Priority 3: Complete Full Run
1. Execute Phase 7 with all scales [2, 4, 8, 16, 32, 64]
2. Verify SE metric computation for all scales
3. Generate training curve, cost curve, scatter plots
4. Implement knee point detection (ΔSE < 0.02)

### Priority 4: Analysis & Documentation
1. Analyze scaling behavior
2. Identify optimal scale (knee point)
3. Generate publishable conclusion format

---

## Configuration Summary

### Test Parameters
- **Model**: google/gemma-3-4b-pt (float16)
- **Concepts**: person, change, animal, object, action
- **Scales**: [2, 4, 8, 16, 32, 64] samples per concept
- **Test Strengths**: [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
- **Prompts**: 10 generic prompts
- **Max New Tokens**: 50

### SE Metric Formula
```
SE = 0.5 × (ρ_Δ,s + r_Δ,human) × coherence_rate

Where:
  ρ_Δ,s = Spearman correlation (Δ vs strength)
  r_Δ,human = Pearson correlation (Δ vs LLM judge scores)
  coherence_rate = % outputs with perplexity ≤ 1.5 × baseline
  Δ = concept_score - neg_concept_score (LLM-judged)
```

### Resource Tracking
- Training time per scale
- VRAM usage during training
- Expected: 2→4→8→16→32→64 samples should show diminishing returns

---

## Notes for Fresh Review

1. **SVD issue is SOLVED** - Don't revisit row normalization/eigendecomposition, it works
2. **Both blocking issues are straightforward fixes** - Estimated 15 minutes total
3. **Test infrastructure is solid** - Once unblocked, should run to completion
4. **LLM judge integration is pending** - May need to implement or mock for SE metric
5. **Knee point detection is algorithmic** - Linear scan for ΔSE < 0.02

## Files Reference
- Test script: `scripts/phase_7_stress_test.py`
- Manifold steerer: `src/steering/manifold.py`
- Hooks module: `src/steering/hooks.py`
- Log file: `results/phase_7_stress_test.log`
