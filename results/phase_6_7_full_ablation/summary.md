# Phase 6.7: Steering Ablation Study - Summary

## Objective
Determine which components of dual-subspace manifold steering are helpful vs harmful.

## Test Matrix

| Variant | PCA Removal | Manifold Proj | Dampening | Working | Diversity | Mean \|ρ\| |
|---------|-------------|---------------|-----------|---------|-----------|-----------|
| ① Raw baseline | ✗ | ✗ | ✗ | 27/32 (84%) | 99.6% | 0.462 |
| ② Contamination-only | ✓ | ✗ | ✗ | 26/32 (81%) | 99.8% | 0.446 |
| ③ Manifold (damp=0.0) | ✗ | ✓ | 0.0 | 3/32 (9%) | 29.2% | 0.103 |
| ③ Manifold (damp=0.5) | ✗ | ✓ | 0.5 | 0/32 (0%) | 14.5% | 0.005 |
| ③ Manifold (damp=1.0) | ✗ | ✓ | 1.0 | 0/32 (0%) | 14.3% | 0.000 |
| ③ Manifold (damp=2.0) | ✗ | ✓ | 2.0 | 0/32 (0%) | 14.3% | 0.000 |
| ④ Dual (damp=0.0) | ✓ | ✓ | 0.0 | 0/32 (0%) | 14.3% | 0.000 |
| ④ Dual (damp=0.5) | ✓ | ✓ | 0.5 | 0/32 (0%) | 14.3% | 0.000 |
| ④ Dual (damp=1.0) | ✓ | ✓ | 1.0 | 0/32 (0%) | 14.3% | 0.000 |
| ④ Dual (damp=2.0) | ✓ | ✓ | 2.0 | 0/32 (0%) | 14.3% | 0.000 |

**Test parameters:**
- 32 concepts (person, change, animal, object, action, time, place, quality, etc.)
- 7 strengths: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
- 2 prompts: "Tell me about", "Describe"
- Total: 4,480 generations across 10 configurations

## Key Findings

### 1. Raw Baseline and Contamination Removal Both Work Well
- **Raw baseline**: 84% of concepts show responsive steering (diversity >30%, |ρ|>0.2)
- **Contamination-only**: 81% of concepts working, nearly identical performance
- Both achieve ~100% output diversity
- Strong correlation between strength and semantic shift (|ρ|≈0.45)

### 2. Manifold Projection Completely Breaks Steering
- Even with **zero dampening**, manifold projection reduces effectiveness to 9%
- With any dampening (0.5, 1.0, 2.0), effectiveness drops to 0%
- Diversity collapses to 14.3% (baseline minimum from greedy decoding)
- Zero correlation between strength and output (|ρ|=0.000)

### 3. Dual-Subspace is Also Broken
- Combining PCA removal + manifold projection produces identical failure
- All dampening levels show 0% effectiveness
- This explains why Phase 6.6 and Phase 7 manifold steering produced identical outputs

## RMSNorm Sign Symmetry Test

**Hypothesis:** RMSNorm after hooks creates sign symmetry, making +v and -v indistinguishable.

**Test:** Compare two hook placements:
- Current: After layer completion (before next layer's RMSNorm)
- Proposed: Before MLP (steering participates in nonlinearity)

**Results:**
- **Current placement**: Cosine sim(+v, -v) = 1.0000, produces degenerate repetition
- **Proposed placement**: Cosine sim(+v, -v) = 1.0000, produces coherent output (but still identical for ±strength)

**Interpretation:**
- Sign symmetry confirmed for both placements (cos=1.0)
- Hook placement affects output quality significantly
- However, the fundamental issue may be in how we apply the sign in the projection

## Conclusions

### What Works
1. ✅ **Raw baseline steering** (subtract projection without preprocessing)
2. ✅ **Contamination removal** (PCA-based subtractive processing)

### What Doesn't Work (Current Implementation)
1. ⚠️ **Manifold projection** (task-manifold-only steering) - 0-9% effective
2. ⚠️ **Dual-subspace** (combined contamination + manifold) - 0% effective
3. ⚠️ **Dampening** (makes manifold steering even worse)

**Important**: The manifold steering approach is validated in the paper, so the concept is sound. Our implementation has bugs or architectural mismatches.

### Possible Root Causes
Our manifold implementation may have:
- Implementation bugs in projection calculation
- Over-constraining the steering vector to a low-dimensional subspace
- Removing signal along with noise
- Incorrect hook placement (RMSNorm interaction)

The paper's success with manifold steering likely depends on:
- Different model architectures (tested on different models than Gemma-3)
- Specific hook placement strategies we haven't matched
- Different projection methodologies or parameterization
- Possibly different normalization handling

## Recommendations

### Immediate Actions
1. **Use raw baseline or contamination-only steering** for production
2. **Abandon manifold projection** until root cause is identified
3. **Investigate hook placement** more deeply (sign handling, layer position)

### Future Research (To Fix Manifold Steering)
1. **Compare with paper implementation**: Line-by-line comparison with original code
2. **Test on paper's models**: Run on their exact architecture/model family
3. **Debug projection math**: Verify manifold estimation and projection calculations
4. **Hook placement study**: Systematic test of different injection points
5. **RMSNorm interaction**: Study normalization effects on steered activations
6. **Alternative approaches**: ICA, NMF, or other dimensionality reduction techniques

## Files

- **Full results**: `results/phase_6_7_full_ablation/full_ablation_results.json`
- **Test script**: `scripts/phase_6_7_full_ablation.py`
- **RMSNorm test**: `scripts/test_rmsnorm_symmetry.py`
- **Logs**: `phase_6_7_full_ablation.log`, `test_rmsnorm_symmetry_run2.log`

## Impact

This ablation study definitively shows that:
- The dual-subspace approach from Phase 6.6 should not be used
- Phase 7's identical outputs were due to manifold projection, not scaling
- We should return to simple baseline steering for reliable results
- The 919/1000 success in Phase 2 was likely due to using raw baseline steering (not manifold)
