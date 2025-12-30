# Detached Jacobian Approach for Concept Extraction

## Overview

The detached Jacobian approach from ["LLMs are Locally Linear"](https://openreview.net/forum?id=oDWbJsIuEp) provides a theoretically grounded method for extracting low-dimensional concept vectors from LLMs.

## Key Theoretical Claims

1. **Low-Dimensional Subspaces**: LLMs operate in "extremely low-dimensional subspaces" despite their high parameter count
2. **Interpretable Singular Vectors**: Singular vectors of the Jacobian matrix decode into interpretable semantic concepts
3. **Perfect Reconstruction**: Achieves reconstruction with relative error below 10^-13
4. **No Training Required**: Works with frozen models - no additional training needed

## Method

### Jacobian Computation

For an input text with embeddings `x`, compute the Jacobian matrix:

```
J[i,j,k] = ∂output_k / ∂input_embedding[i,j]
```

Where:
- `i` = token index
- `j` = input embedding dimension
- `k` = output embedding dimension

### Concept Extraction

The weighted sum of Jacobians reveals the effective transformation:

```
concept_direction = sum_i(J_i @ embed_i)
```

The top singular vectors of this transformation represent the low-dimensional semantic subspace.

## Comparison with CAV Approach

### Current CAV Approach
- **Method**: Contrastive activation vectors from paired examples
- **Speed**: Fast (~0.7s for extraction)
- **Dimensionality**: Full dimensional space (2560D)
- **Training**: Requires paired concept/non-concept examples

### Detached Jacobian Approach
- **Method**: Compute Jacobian via autograd, extract singular vectors
- **Speed**: Slower (~10s for Jacobian computation)
- **Dimensionality**: Reveals true low-dimensional structure (often < 10D)
- **Training**: No training - single forward pass with gradient

## Advantages for Manifold Steering

1. **Theoretically Grounded**: Proven to capture the true low-dimensional manifold
2. **Precise**: 10^-13 reconstruction error vs. approximate CAV directions
3. **No Contrastive Examples Needed**: Single concept prompt sufficient
4. **True Manifold Structure**: Reveals actual dimensionality of concept space

## Implementation Status

### Current Status
- **Basic implementation**: `src/steering/detached_jacobian.py` ✓
- **Architecture compatibility**: Partial (needs Gemma-3 specific handling)
- **Full forward pass**: Requires careful handling of attention masks and model structure

### Blockers
1. Gem ma-3 architecture differences (e.g., `_update_causal_mask` location)
2. Memory requirements for full Jacobian computation
3. Computational cost (~10-20s per concept)

### Future Work

1. **Complete Gemma-3 Integration**
   - Properly handle `model.language_model` vs `model.model`
   - Find `_update_causal_mask` equivalent or reimplement
   - Handle position embeddings (global vs local)

2. **Optimize Computation**
   - Use chunked Jacobian computation to reduce memory
   - Implement reverse-mode autodiff efficiently
   - Cache Jacobians for reuse

3. **Benchmark Against CAV**
   - Steering quality comparison
   - Manifold dimensionality analysis
   - Runtime vs accuracy tradeoff

4. **Integration with Manifold Steering**
   - Use Jacobian singular vectors for manifold estimation
   - Replace PCA-based manifold with Jacobian-based subspace
   - Compare steering precision

## References

- Paper: https://openreview.net/forum?id=oDWbJsIuEp
- Code: https://github.com/jamesgolden1/equivalent-linear-LLMs
- Notebook: https://github.com/jamesgolden1/equivalent-linear-LLMs/blob/main/notebooks/gemma_3/detached_jacobian_gemma_demo.ipynb

## Empirical Validation Results (2025-11-13)

### Alignment Test: Jacobian vs Trained Classifiers

**Test Configuration**:
- Compared Jacobian vectors with trained MLP classifiers (contrastive learning)
- 5 Layer 0 concepts: Physical, Abstract, Process, Entity, Attribute
- Model: gemma-3-4b-pt (BF16)
- Jacobian: Layer 6, prompt "The concept of {X} means"
- Classifiers: 3-layer MLP trained on definitions + hard negatives + relationships

**Results**:
- **Mean cosine similarity**: -0.0187 (near zero)
- **Standard deviation**: 0.0160
- **Range**: [-0.0377, 0.0052]
- **Interpretation**: Jacobian and classifier directions are **orthogonal**

### Key Findings

1. **Different Objectives → Different Directions**
   - **Jacobian**: Local sensitivity for *generation* ("how to complete this prompt")
   - **Classifier**: Learned boundary for *discrimination* ("what distinguishes concept across contexts")

2. **Jacobians are NOT "Ground Truth" for Classifiers**
   - Near-zero alignment doesn't indicate poor classifier quality
   - They measure fundamentally different aspects of concepts
   - Context-dependent (single prompt) vs context-invariant (trained on many examples)

3. **Validates Contrastive Training Approach**
   - Hard negatives, relational examples, and definitional framing capture something Jacobians don't
   - Classifiers learn discriminative boundaries suitable for steering
   - Jacobians optimize for generation, not discrimination

### Geometric Interpretation

```
Jacobian:    "Which direction nudges activations to generate concept-related text?"
Classifier:  "Which direction separates concept from complements, neighbors, and noise?"
```

These are orthogonal questions → orthogonal answers are expected.

### Implications for Use

❌ **Don't use Jacobians for**:
- Validating classifier quality (orthogonal objectives)
- Training anchors or drift detection
- "Ground truth" comparison

✅ **Do use Jacobians for**:
- Understanding local model geometry
- Research into generation vs discrimination tradeoffs
- Exploring context-dependent concept representations

✅ **Trust the classifiers**:
- Contrastive training with structured negatives is aligned with steering
- Use classifier directions for both enhancement and suppression

**See**: `results/jacobian_alignment_analysis.md` for full analysis

---

## Recommendation

**Current Status (2025-11-13)**: Jacobian approach validated but found orthogonal to steering objectives.

**Short term**: Continue using classifier-based approach - validated for steering applications

**Research use only**: Jacobian implementation available for:
- Local geometry analysis
- Generation vs discrimination studies
- Academic comparison with paper results

**Not recommended for**:
- Steering vector validation
- Classifier quality assessment
- Production steering applications
