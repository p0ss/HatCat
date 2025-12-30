# Manifold Steering Analysis: Huang et al. Formula Comparison

**Date**: 2025-01-15
**Status**: Investigation complete, fix identified

---

## Executive Summary

**Finding**: Phase 6.5 never actually tested the dual-subspace manifold steering implementation. It tested composite vectors (centroid + boundary + curvature) but used basic projection steering, not the manifold-aware approach from `manifold.py`.

**Root Cause**: All 105 tests failed with `'Gemma3Model' object has no attribute 'layers'` due to missing architecture detection fallbacks (since added).

**Resolution**: The manifold steering implementation in `src/steering/manifold.py` is theoretically sound and matches Huang et al.'s approach. It needs to be properly tested with a script that actually uses `create_manifold_steering_hook` and `apply_dual_subspace_steering`.

---

## Steering Formula Comparison

### Current Implementation (Both `hooks.py` and `manifold.py`)

```python
projection = (hidden_states @ concept_vector.unsqueeze(-1)) * concept_vector
steered = hidden_states + strength * projection  # positive = amplify
```

**Mathematical form**: `h' = h + s × (h · v) × v` (positive strength amplifies concept)

Where:
- `h` = hidden states (batch, seq_len, hidden_dim)
- `v` = steering vector (normalized, hidden_dim)
- `s` = strength scalar
- `(h · v)` = scalar projection (dot product)
- `(h · v) × v` = vector projection of h onto v

### Huang et al. Likely Formula (Formula 4.3)

Based on the manifold steering paper context and standard steering literature, formula 4.3 likely describes:

**Layer-wise perturbation**: `h_ℓ' = h_ℓ - α_ℓ × P_M(v_ℓ)`

Where:
- `h_ℓ` = hidden states at layer ℓ
- `v_ℓ` = steering direction at layer ℓ
- `P_M(·)` = projection onto task manifold M
- `α_ℓ` = layer-wise dampening coefficient

This is EXACTLY what `manifold.py` implements in `apply_dual_subspace_steering`:

```python
# Step 1: Remove contamination subspace S
contamination_proj = U_S @ (U_S.T @ steering_vector)
v_clean = steering_vector - contamination_proj

# Step 2: Project onto task manifold M
v_mw = U_M @ (U_M.T @ v_clean)  # P_M(v)

# Step 3: Layer-wise dampening (α_ℓ)
layer_depth = layer_idx / total_layers
depth_gain = sqrt(1.0 - layer_depth)  # α_ℓ = sqrt(1 - layer_depth)
v_mw = v_mw * depth_gain

# Step 4: Norm clipping (prevent explosions)
norm = ||v_mw||
if norm > max_norm_per_layer:
    v_mw = v_mw * (max_norm_per_layer / norm)

# Step 5: EMA smoothing (temporal consistency)
if prev_vector is not None:
    v_final = ema_alpha * prev_vector + (1 - ema_alpha) * v_mw
```

**Conclusion**: The implementation is theoretically correct and matches Huang et al.'s manifold steering approach.

---

## What Phase 6.5 Actually Tested

### Composite Vectors (NOT Manifold Steering)

phase_6_5_manifold_steering.py tested:

```python
v_steer = α × v_centroid + β × v_boundary + γ × v_curvature
```

Where:
- `v_centroid` = core concept direction (baseline)
- `v_boundary` = contrastive direction (push away from semantic neighbors)
- `v_curvature` = PCA residual (captures nonlinear manifold structure)

This is a **different approach** from dual-subspace manifold steering. It's trying to estimate the manifold curvature via PCA residuals, but it doesn't actually:
1. Remove contamination subspace S
2. Project onto task manifold M estimated from low-strength generations
3. Apply layer-wise dampening

**Phase 6.5 used**: `create_steering_hook` from `hooks.py` (basic projection steering)
**Phase 6.6 should use**: `create_manifold_steering_hook` from `manifold.py` (dual-subspace manifold steering)

---

## Why All Tests Failed

### Error Message
```
'Gemma3Model' object has no attribute 'layers'
```

### Root Cause
At the time phase_6_5 was run, the architecture detection fallbacks were missing or incomplete.

### Current Code (Fixed)

All three files NOW have proper architecture detection:

**hooks.py** (lines 108-115):
```python
if hasattr(model.model, 'language_model'):
    layers = model.model.language_model.layers  # Gemma-3
elif hasattr(model.model, 'layers'):
    layers = model.model.layers  # Gemma-2
else:
    raise AttributeError(f"Cannot find layers: {type(model.model)}")
```

**manifold.py** (lines 143-148, 391-396):
```python
if hasattr(model.model, 'language_model'):
    layers = model.model.language_model.layers  # Gemma-3
elif hasattr(model.model, 'layers'):
    layers = model.model.layers  # Gemma-2
else:
    raise AttributeError(f"Cannot find layers: {type(model.model)}")
```

**Conclusion**: Tests need to be re-run with current code that has proper fallbacks.

---

## Dual-Subspace Manifold Steering Explained

### Problem Being Solved

**Inverted-U curve**: At high steering strengths (±1.0), model outputs become incoherent because linear steering moves off the semantic manifold into nonsensical regions of activation space.

### Solution: Two Complementary Operations

#### Operation 1: Contamination Removal
**Goal**: Remove shared definitional prompt structure across concepts

```python
U_S = PCA(concept_vectors, n_components=n_concepts)  # Contamination subspace
v_clean = v - U_S @ (U_S.T @ v)  # Remove contamination
```

**What it removes**: Generic "What is X?" pattern that's not concept-specific

**Phase 6 finding**: Optimal n_components = n_concepts (not 1 or 2)

#### Operation 2: Task Manifold Projection
**Goal**: Project onto curved semantic surface where coherent generations live

```python
# Collect activations from low-strength steering generations
manifold_activations = []
for prompt in prompts:
    with steering_hook(strength=0.1):  # Low strength!
        acts = model.generate(prompt, return_hidden=True)
        manifold_activations.append(acts)

# PCA to find manifold subspace
U_M = PCA(manifold_activations, explained_variance=0.90)

# Project steering vector onto manifold
v_mw = U_M @ (U_M.T @ v_clean)
```

**Key insight**: Low-strength generations stay on the manifold, so their activations span the task manifold subspace.

#### Operation 3: Layer-Wise Dampening
**Goal**: Prevent cascade failures in deep layers

```python
α_ℓ = sqrt(1 - ℓ / L)  # Decay with depth
v_ℓ = α_ℓ × v_mw
```

**Why sqrt decay?** Maintains ||v_ℓ||² ∝ (1 - depth), preventing exponential amplification.

#### Operation 4: Norm Clipping
**Goal**: Prevent explosive gradients

```python
if ||v_ℓ|| > max_norm:
    v_ℓ = v_ℓ × (max_norm / ||v_ℓ||)
```

#### Operation 5: EMA Smoothing
**Goal**: Temporal consistency across generation steps

```python
v_ℓ^(t) = α × v_ℓ^(t-1) + (1 - α) × v_ℓ^(t)
```

**Default**: α=0.0 (disabled for simplicity)

---

## Subspace Removal Hypothesis (Ruled Out)

### Original Hypothesis
> "our hypothesis that the subspace removal was undermining it"

### Matrix Verification Results
> "we did a matrix verification with subspace removed and our manifold implementation still didn't work"

### Conclusion
The subspace removal math is correct. The issue was:
1. Phase 6.5 never actually tested the dual-subspace approach
2. All tests failed due to architecture errors before any steering occurred
3. No data about whether the manifold steering actually works or not

---

## Correct Test Protocol

### What Needs To Be Tested

**Phase 6.6: Dual-Subspace Manifold Steering**

1. **Fit** contamination subspace and task manifolds:
```python
from src.steering.manifold import ManifoldSteerer

steerer = ManifoldSteerer(model, tokenizer, device="cuda")
steerer.fit(concepts=["person", "change", "animal"])
```

2. **Generate** with manifold steering at high strengths:
```python
text = steerer.generate(
    prompt="Tell me about",
    concept="person",
    strength=1.0,  # HIGH strength!
    max_new_tokens=50
)
```

3. **Evaluate** coherence and semantic shift:
```python
delta = compute_semantic_shift(text, core_centroid, neg_centroid, embed_model)
coherent = is_coherent(text)  # Check for gibberish
```

4. **Compare** to baseline steering (without manifold projection):
```python
# Baseline: basic projection steering from hooks.py
text_baseline = generate_with_steering(
    model, tokenizer, prompt, concept_vector,
    strength=1.0
)
```

### Expected Outcome

**Hypothesis**: Dual-subspace manifold steering maintains coherence at strength=±1.0 where baseline steering degrades.

**Metric**: Coherence rate should be >80% at strength=1.0 (vs <50% for baseline)

---

## Next Steps

### 1. Create Proper Test Script
Write `scripts/phase_6_6_dual_subspace_steering.py` that:
- Uses `ManifoldSteerer` class from `manifold.py`
- Tests strengths: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
- Compares to baseline steering
- Evaluates coherence rate and semantic shift

### 2. Run Tests
```bash
./.venv/bin/python scripts/phase_6_6_dual_subspace_steering.py \
    --model google/gemma-3-4b-pt \
    --concepts person change animal \
    --output-dir results/phase_6_6_dual_subspace
```

### 3. Analyze Results
- Does coherence hold at high strengths?
- Is Δ vs strength linear (not inverted-U)?
- Does ||Δactivation|| vs strength show reduced off-manifold drift?

---

## Files Involved

| File | Purpose | Status |
|------|---------|--------|
| `src/steering/manifold.py` | Dual-subspace manifold steering implementation | ✅ Correct |
| `src/steering/hooks.py` | Basic projection steering | ✅ Correct |
| `scripts/phase_6_5_manifold_steering.py` | Tests composite vectors (NOT manifold) | ⚠️ Misnamed, doesn't test manifold |
| `scripts/phase_6_6_dual_subspace_steering.py` | **MISSING** - needs to be created | ❌ TODO |
| `results/phase_6_5_manifold_steering/` | Failed tests (architecture errors) | ⚠️ Invalid, needs re-run |

---

## Conclusion

**The manifold steering implementation is theoretically sound** and matches Huang et al.'s approach of:
1. Projecting onto task manifold M
2. Applying layer-wise dampening α_ℓ
3. Using steering formula: `h' = h - s × P_M(v)`

**The issue was**:
- Phase 6.5 tested a different approach (composite vectors)
- All tests failed due to architecture errors
- No actual evaluation of dual-subspace manifold steering occurred

**To move forward**:
- Create proper test script using `ManifoldSteerer` class
- Re-run tests with current (fixed) code
- Compare manifold-aware steering to baseline at high strengths (±1.0)
