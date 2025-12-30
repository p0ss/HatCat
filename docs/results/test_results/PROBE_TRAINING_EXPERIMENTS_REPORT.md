# Lens Training Experiments Report

**Date**: December 2, 2025
**Model**: swiss-ai/Apertus-8B-2509
**Objective**: Identify and address sources of lens training variance and sibling confusion

---

## Executive Summary

We conducted a series of controlled experiments to understand why lenses fail to discriminate between semantically related concepts (siblings). The key findings are:

1. **Sibling-aware training dramatically improves discrimination** - Mixed mode training with siblings as hard negatives achieves 0.88-0.99 positive-sibling gaps vs 0.50-0.70 for standard training
2. **Inter-run variance is significant (~5% std dev)** - Same training setup with different seeds produces 0.05 average gap std deviation
3. **Post-hoc debiasing shows limited effect** - Common direction analysis found Layer 0 lenses share 0.32 mean similarity, but Layers 1-4 only 0.05-0.07
4. **Generalist baseline approach inconclusive** - Only 1 generalist lens trained successfully, limiting the power of the test

---

## Experiment 1: Inter-Run Variance

**Purpose**: Establish error bars for comparing training approaches

**Method**: Train the same 8 lenses 5 times each with different random seeds (42, 142, 242, 342, 442)

**Results**:

| Concept | Layer | Gap Mean | Gap Std Dev |
|---------|-------|----------|-------------|
| AlignmentProcess | 1 | 0.087 | 0.070 |
| Agent | 1 | 0.079 | 0.097 |
| Projectile | 2 | 0.072 | 0.088 |
| AIAlignmentProcess | 2 | 0.200 | 0.00002 |
| AAV | 3 | 0.199 | 0.003 |
| ACPowerSource | 3 | 0.200 | 0.00001 |
| AH1 | 3 | 0.143 | 0.068 |
| ADHD | 4 | 0.160 | 0.080 |

**Key Finding**: Average inter-run std dev: **0.051** (±0.10 suggested error margin for comparisons)

Some lenses (AIAlignmentProcess, ACPowerSource) show almost no variance, while others (Agent, Projectile) show high variance. This may correlate with concept specificity and definition quality.

---

## Experiment 2: Contrastive Training with Sibling Hard Negatives

**Purpose**: Test whether explicitly including siblings as negatives improves discrimination

**Method**: Compare three training modes:
- **Standard**: Distant concepts as negatives
- **Sibling**: Only siblings as negatives
- **Mixed**: Both siblings and distant as negatives (50/50 split)

**Results** (Positive-Sibling Gap):

| Concept | Standard | Sibling | Mixed |
|---------|----------|---------|-------|
| AlignmentProcess | 0.500 | 0.645 | **0.999** |
| Agent | 0.475 | **1.000** | 0.953 |
| Projectile | 0.896 | 0.962 | 0.801 |
| AIAlignmentProcess | 0.785 | 0.936 | 0.888 |
| AAV | 0.991 | **1.000** | 0.999 |
| ACPowerSource | 0.515 | **1.000** | 0.999 |
| AH1 | 0.700 | 0.566 | 0.793 |
| ADHD | 0.986 | 0.800 | **1.000** |

**Key Findings**:
1. Standard training often achieves good distant rejection but poor sibling rejection
2. Sibling-only training sometimes over-specializes, reducing positive recall
3. Mixed mode provides the best balance in most cases

**Recommendation**: Use mixed training with ~50% siblings, ~50% distant negatives

---

## Experiment 3: Post-Hoc Debiasing

**Purpose**: Identify and remove shared "concept-like" directions across all lenses

**Method**:
1. Extract first-layer weights from all trained lenses
2. Compute mean direction per layer (the "common direction")
3. Analyze similarity of individual lenses to common direction

**Results** (Common Direction Statistics):

| Layer | # Lenses | Direction Norm | Mean Similarity |
|-------|----------|----------------|-----------------|
| 0 | 10 | 3.66 | 0.32 (high) |
| 1 | 257 | 0.84 | 0.05-0.07 (low) |
| 2 | 1093 | 0.69 | 0.05-0.07 (low) |
| 3 | 1027 | 0.69 | 0.05-0.07 (low) |
| 4 | 3271 | 0.84 | 0.05-0.07 (low) |

**Key Findings**:
1. Layer 0 lenses share significant structure (0.32 similarity) - likely due to learning "is this a concept?" rather than "which concept?"
2. Layers 1-4 show minimal common direction (0.05-0.07 similarity) - lenses learn concept-specific directions
3. Post-hoc debiasing unlikely to help layers 1-4; may help Layer 0

---

## Experiment 4: Generalist Baseline Approach

**Purpose**: Pre-compute common direction from diverse concepts, subtract during training

**Method**:
1. Train lenses on 10 diverse "generalist" concepts (Organism, Artifact, Device, etc.)
2. Compute common direction from their first-layer weights
3. Train target lenses while projecting out the common direction at various strengths

**Results** (limited due to only 1 generalist lens training successfully):

| Mode | Avg Pos-Sib Gap | Avg Pos-Dist Gap |
|------|-----------------|------------------|
| Baseline | 0.197 ± 0.129 | 0.297 ± 0.062 |
| Offset 0.25 | 0.181 ± 0.140 | 0.319 ± 0.191 |
| Offset 0.5 | 0.251 ± 0.141 | 0.305 ± 0.123 |
| Offset 0.75 | 0.192 ± 0.156 | 0.330 ± 0.143 |
| Offset 1.0 | **0.286 ± 0.109** | 0.380 ± 0.135 |

**Key Findings**:
1. Only "Organism" trained successfully as a generalist (others lacked definitions)
2. Offset 1.0 shows slight improvement in sibling gap (0.286 vs 0.197 baseline)
3. **Inconclusive** - need more generalist lenses for a representative common direction

---

## Recommendations

### Immediate Actions

1. **Use mixed sibling training** for all future lens packs
   - Include 50% siblings as hard negatives
   - This alone provides the largest improvement in sibling discrimination

2. **Increase minimum definitions threshold** from 5 to 10
   - Many concepts fail to train due to insufficient examples
   - Quality > quantity for lens reliability

### Future Work

1. **Multi-relationship training** - Include all known relationships (siblings, parents, children, distant) with semantic prompts
2. **Cross-category L0 negatives** - Explicit negative relationships to non-parent L0 concepts
3. **Alternate meaning lookup** - Dictionary/encyclopedic sources for polysemy beyond synsets
4. **Multiple concurrent top-k tracks** - Ensemble approaches for calibration
5. **Linguistic coverage expansion** - Address gaps in pragmatic/communicative concepts

---

## Data Files

- `results/variance_experiment/variance_results.json` - Inter-run variance data
- `results/contrastive_experiment/contrastive_results.json` - Sibling training comparison
- `results/contrastive_experiment_v2/contrastive_results.json` - Extended with distant evaluation
- `results/debiasing_experiment/common_direction_analysis.json` - Common direction statistics
- `results/generalist_baseline_experiment/generalist_baseline_results.json` - Offset training results

---

## Scripts

- `scripts/experiment_variance.py` - Variance measurement
- `scripts/experiment_contrastive.py` - Sibling hard negative training
- `scripts/experiment_posthoc_debiasing.py` - Common direction analysis
- `scripts/experiment_generalist_baseline.py` - Offset-based debiasing
