# HatCat Shared Code Reference

This document outlines canonical implementations in `/src/` that should be used across all experiments. **Always check this before implementing common patterns.**

## Model Loading and Cleanup

### ✅ Correct Pattern
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-pt",
    torch_dtype=torch.float32,  # CRITICAL: float32 for numerical stability with SVD/projections
    device_map="cuda"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ALWAYS cleanup at the end of your script
def cleanup_model():
    """Cleanup GPU memory to prevent OOM in subsequent runs."""
    global model, tokenizer
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# At the end of main():
try:
    # ... your experiment code ...
finally:
    cleanup_model()
```

**Why float32?**
- Phase 6.6 and 7 showed float16 causes NaN/Inf in SVD operations
- Manifold steering requires stable matrix decompositions
- VRAM impact: ~2x (8.6GB → 17.2GB for gemma-3-4b-pt)

### ❌ Don't Use
```python
torch_dtype=torch.float16  # Causes numerical instability
dtype=torch.float16        # Same issue, deprecated syntax
```

## Steering

### Raw Baseline Steering
**Location:** `src/steering/hooks.py`

```python
from src.steering import extract_concept_vector, generate_with_steering

# Extract concept vector
v = extract_concept_vector(model, tokenizer, "person", device="cuda")

# Generate with steering
text = generate_with_steering(
    model, tokenizer,
    prompt="Tell me about",
    steering_vector=v,
    strength=-1.0,  # Negative = suppress, positive = amplify
    max_new_tokens=50,
    device="cuda"
)
```

**Status:** ✅ Working (Phase 6.7: 100% diversity, ρ=0.674)

**⚠️ CRITICAL: Correct Sign Application**
```python
# ✅ CORRECT: Apply sign to projection (positive = amplify, negative = suppress)
projection = (hidden @ v.unsqueeze(-1)) * v
steered = hidden + strength * projection

# ❌ WRONG: Negating vector squares away the sign!
v_signed = v * strength  # DON'T DO THIS
projection = (hidden @ v_signed.unsqueeze(-1)) * v_signed
steered = hidden + projection  # Sign cancels: (h·(-v))(-v) = (h·v)(v)
```

**Why:** In projection formula `(h·v)v`, if you negate both the dot product AND the vector, the negatives cancel out, making `+v` and `-v` identical. Always keep the vector positive and apply the sign to the projection result.

### Manifold Steering
**Location:** `src/steering/manifold.py`

```python
from src.steering.manifold import ManifoldSteerer

steerer = ManifoldSteerer(model, tokenizer, device="cuda")
steerer.fit(concepts=["person", "animal"], n_manifold_samples=4)

text = steerer.generate(
    prompt="Tell me about",
    concept="person",
    strength=-1.0,
    max_new_tokens=50,
    max_norm_per_layer=1.0
)
```

**Status:** ⚠️ **CURRENT IMPLEMENTATION NOT WORKING** (Phase 6.7 ablation completed)
- **Issue:** Current manifold projection implementation ineffective
- Phase 6.7 results: 0-9% working concepts (vs 84% for raw baseline)
- Produces identical outputs regardless of steering strength
- **Possible causes**: Implementation bugs, architecture-specific issues (Gemma-3 vs paper's models), hook placement, or incorrect projection methodology
- **Note**: Manifold steering works in the paper, so the approach is valid - our implementation needs debugging

**For production use**: Raw baseline or contamination-only steering (both ~85% effective)
**For research**: Investigate implementation differences from paper

## Evaluation Metrics

### TODO: Move to `src/steering/evaluation.py`

Currently scattered across phase scripts. Need to consolidate:

```python
# Perplexity (coherence check)
def compute_perplexity(model, tokenizer, text: str, device: str) -> float:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, labels=inputs.input_ids)
        return torch.exp(outputs.loss).item()

# Semantic shift (Δ)
def compute_semantic_shift(model, tokenizer, text: str, concept: str, device: str) -> float:
    """Δ = cos(text, concept) - cos(text, 'nothing')"""
    # Implementation in phase_6_7_ablation_v2.py compute_delta()

# Diversity ratio
def compute_diversity_ratio(texts: List[str]) -> float:
    return len(set(texts)) / len(texts)

# Spearman correlation (steering effectiveness)
from scipy.stats import spearmanr
rho, pval = spearmanr(strengths, deltas)
```

## Common Patterns

### Getting Model Layers
```python
# Handle different architectures
if hasattr(model.model, 'language_model'):
    layers = model.model.language_model.layers  # Gemma-3
elif hasattr(model.model, 'layers'):
    layers = model.model.layers  # Gemma-2
else:
    raise AttributeError(f"Cannot find layers in {type(model.model)}")
```

### Getting Embeddings
```python
# Get embedding layer
if hasattr(model.model, 'language_model'):
    embed = model.model.language_model.embed_tokens
elif hasattr(model.model, 'embed_tokens'):
    embed = model.model.embed_tokens
else:
    embed = model.model.get_input_embeddings()
```

### Safe Generation (avoid do_sample=True with hooks)
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,  # Greedy decoding - deterministic
    pad_token_id=tokenizer.eos_token_id
)
```

**Why do_sample=False?**
- With steering hooks, sampling can amplify numerical issues
- Greedy decoding is more stable for debugging
- If outputs are identical across strengths, steering isn't working (not a sampling issue)

## Concept Graph

**Location:** `src/encyclopedia/wordnet_graph_v2.py`

```python
from src.encyclopedia.wordnet_graph_v2 import WordNetGraph

graph = WordNetGraph()
concepts = graph.get_top_n_concepts(n=100)
```

**Available graphs:**
- `data/concept_graph/wordnet_v2_top10.json`
- `data/concept_graph/wordnet_v2_top100.json`
- `data/concept_graph/wordnet_v2_top1000.json`

## Testing Guidelines

### Minimal Test Parameters (for iteration speed)
```python
N_CONCEPTS = 8  # Fast: person, animal, object, action, time, place, quality, emotion
STRENGTHS = [-2.0, -1.0, 0.0, 1.0, 2.0]  # 5 key points
PROMPTS = ["Tell me about", "Describe"]  # 2 diverse prompts
MAX_NEW_TOKENS = 30  # Short enough to run fast
```

**Expected runtime:** ~2-3 minutes for 8×5×2 = 80 generations

### Full Test Parameters
```python
N_CONCEPTS = 32
STRENGTHS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
PROMPTS = ["Tell me about", "Describe", "What is", "Consider", "Explain"]
```

**Expected runtime:** ~20-30 minutes

## Known Issues

### Issue: float16 numerical instability
**Symptoms:** NaN/Inf in outputs, CUDA device-side assert, "probability tensor contains inf/nan"
**Solution:** Use `torch_dtype=torch.float32`
**Phases affected:** 7 (fixed), earlier phases may still have this

### Issue: Identical outputs across steering strengths
**Symptoms:** diversity_ratio ≈ 0%, all outputs the same regardless of strength
**Possible causes:**
1. Steering vector magnitude too small (over-dampening)
2. Hooks not being applied
3. Greedy decoding + weak steering (model's prior overcomes steering)

**Debug:**
- Check if baseline (raw) steering works: `diversity_ratio > 30%`
- Check Spearman ρ: `|ρ| > 0.2` indicates strength affects outputs
- Verify hooks are registered and called

### Issue: Degraded/repetitive outputs at high strengths
**Symptoms:** At |strength| > 1.0, outputs become nonsensical repetition
**Explanation:** Over-steering pushes model into degenerate regions of latent space
**Not necessarily a bug:** Indicates steering is working but too strong

## Steering Effectiveness Summary

**Validated methods** (see `TEST_DATA_REGISTER.md` Phase 6.7 for full results):
- ✅ **Raw baseline**: 84% concepts working, 99.6% diversity (USE FOR PRODUCTION)
- ✅ **Contamination-only**: 81% concepts working, 99.8% diversity (USE FOR PRODUCTION)
- ⚠️ **Manifold projection**: 0-9% working (current implementation needs debugging)
- ⚠️ **Dual-subspace**: 0% working (current implementation needs debugging)

**Note**: Manifold steering approach is valid (works in paper) - implementation bugs need fixing

## When to Create New Shared Code

**Before adding to `/src/`:**
1. Pattern appears in 3+ phase scripts
2. Implementation is validated and working
3. Clear performance/correctness benefit over alternatives

**Process:**
1. Document working implementation in phase script
2. Extract to appropriate `/src/` module
3. Add unit tests in `/tests/`
4. Update this reference doc
5. Update phase scripts to use shared implementation

## References

- Float precision investigation: Phase 6.7, Phase 7 debugging session
- Steering effectiveness: Phase 6.6 results, Phase 6.7 ablation
- Manifold steering paper: Huang et al., "Mitigating Overthinking via Manifold Steering"
