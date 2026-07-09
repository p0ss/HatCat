# Activation-Cell Cartography — Experiment 006

**Date:** 2026-05-03
**Status:** Plan (deferred until Gemma 4 E4B lens pack training completes ~2026-05-08)
**Model:** `google/gemma-4-E4B` (dense, 42 layers, 2560 hidden, ~8B with PLE)
**Concept Pack:** `first-light` (7,947 lenses across SUMO ontology layers 0–6)
**Related:**
- [Topology Probing 001](topology-probing-001.md) — clusters *neurons* by connectivity (predecessor; different unit of analysis)
- [Polar Lens Abstraction 005](polar-lens-abstraction-005.md) — bipolar probe pairs (informs cell-labeling strategy)
- `/var/home/poss/Documents/Code/HatCatDev/scripts/analysis/quant_sensitivity.py` — quantization sensitivity analysis (motivates Voronoi alternative to polar quantization)

## Summary

Cluster the residual-stream activation vectors collected during a full lens pack training run into Voronoi cells via k-means. Label each cell by (a) which trained concept lenses fire most strongly on activations in that cell, and (b) the n-gram statistics of generated text tokens emitted while activations were in that cell. Use the resulting "activation atlas" for three downstream purposes:

1. **Quantization-by-cell-ID** as an alternative to scalar / polar quantization, with stronger geometry preservation guarantees for in-distribution data.
2. **Coverage analysis** identifying activation-space regions visited during training that have no corresponding concept lens — the "missing concepts" set.
3. **Active-learning loop** for ontology expansion: targeted prompt generation toward unmapped regions to bootstrap new concepts.

## Motivation

### From Experiment 006-prerequisite quant sensitivity (2026-05-02)

Polar quantization preserves inter-lens geometry well at 8 bits (top-10 nearest-neighbor overlap = 88.8%) but degrades sharply below that. The error distribution is asymmetric: close-pair distance error is 3–4.6× higher than far-pair error, growing as bit depth drops. This matches the JL-style prediction that random-projection-based compression is robust at the centroid level but fragile at decision-boundary regions where concept disambiguation actually happens.

A learned codebook (k-means cells in activation space) would adapt the codebook to where activations actually concentrate, spending bits on the manifold rather than uniformly on the unit sphere. Within-cell perturbations are zero-error; only boundary-crossings are lossy. This shifts the failure mode from "all close-pair distances degrade smoothly" to "occasional discrete reassignments" — much more analyzable and bounded.

### From the existing concept pack architecture

The deployed `first-light` lens pack contains MLP probes with no per-concept activation centroid stored. Runtime "centroid divergence" in `analyze_divergence` is computed on the fly from concept-name embeddings, not from the activation distribution the model actually visits. We have no map of where in activation space the model spends its time, only what concepts we've trained classifiers for. Coverage is thus opaque: we don't know what fraction of activation manifold is described by the existing 7,947 concepts vs. what fraction is "dark territory."

### Why Gemma 4 E4B specifically

The current Gemma 4 E4B training run (started 2026-05-02 23:42) is collecting fresh activation extractions across the full concept pack. PLE (Per-Layer Embeddings) introduces some discrete structure even in a dense model — each decoder layer has its own per-token embedding lookup. Whether this manifests as more compact clustering than Gemma 3 4B's dense residual stream is itself an empirical question worth measuring. (The MoE variant 26B A4B would be the cleaner test of "MoE → discrete clusters" but doesn't fit on the 3090.)

## Hypothesis

**H1 (coverage):** When the trained concept pack covers the activation space well, clustering activations into K cells (K ∈ [1k, 100k]) produces a distribution where ≥80% of cells have at least one strongly-firing concept lens. If <50% of cells have concept hits, the ontology has substantial gaps relative to model behavior.

**H2 (cell purity):** Cells with low entropy in their `P(concept | cell)` distribution correspond to "clean" interpretable regions; cells with high entropy correspond to either polysemantic regions (multiple concepts share representation) or under-resolved cells (need subdivision via finer K).

**H3 (quantization):** Cell-ID quantization at K=10,000 (≈13 bits per activation token) preserves inter-lens nearest-neighbor structure better than polar-8b (which is ≈ 8 bits/dim × 2560 dim = 20,480 bits). Specifically, top-10 overlap should exceed the polar-8b baseline of 88.8%.

**H4 (gap-filling):** Cells with high training-data occupancy but no concept-lens correspondence have detectable text-token correlations (n-grams from the generation context where activations landed in that cell). These correlations identify candidate concepts to add to the ontology.

**H5 (PLE structure):** Gemma 4 E4B's PLE introduces measurably more compact clustering than Gemma 3's standard residual stream. Quantified as: at fixed K, the within-cell variance is lower for Gemma 4 E4B activations than for an equivalent-size dense model without PLE.

## Method

### Phase 1: Activation Collection (already happening)

The Gemma 4 E4B training run extracts activations via `output_hidden_states=True`. Modify or add a hook to **persist** these activations to disk during training, keyed by (network_layer, ontology_layer, concept_name, sample_idx, token_position). Format: HDF5 chunked by layer.

Estimated volume: ~7947 concepts × 100 samples × 50 tokens avg × 42 network layers × 2560 dim × 2 bytes (bf16) = ~85 GB per layer — *only feasible if persisted at a strategic subset of network layers*. Practical approach: persist activations only at the network layers selected by the auto-layer-selection step (one chosen layer per concept), which compresses to ~2 GB per ontology layer × 7 = ~14 GB total. Tractable.

### Phase 2: Clustering

For each ontology layer, run mini-batch k-means on the persisted activations. Sweep K ∈ {1000, 5000, 10000, 50000, 100000}. Track:
- Within-cluster sum of squares (WCSS) for elbow detection
- Silhouette score (sampled; full computation is O(N²))
- Per-cell occupancy distribution (Gini coefficient)

Output: one codebook per ontology layer, plus per-activation cell assignments.

### Phase 3: Cell Labeling

For each cell, compute two label types:

**Lens-based label.** For each trained concept lens in the pack, score the lens against the cell's centroid activation. Build per-cell `P(concept | cell)` using softmax over lens scores. Compute entropy of this distribution as the "cell purity" metric.

**Text-correlation label.** During Phase 1 activation extraction, also persist the generation-context tokens (preceding K tokens, current token) that produced each activation. Per-cell, compute n-gram (n=1,2,3) frequency distributions over both preceding and current tokens. Top n-grams = the text correlate of that cell.

### Phase 4: Coverage Analysis

Categorize cells by (occupancy, lens-purity, text-correlation strength):

| Category | Occupancy | Lens purity | Text correlation | Interpretation |
|---|---|---|---|---|
| Clean concept | High | High | High | Well-mapped, single-concept cell |
| Polysemantic | High | Low | High | Multi-concept region; subdivide or accept |
| Missing concept | High | None | High | Ontology gap; new concept candidate |
| Under-resolved | High | Mid | Mid | Increase K |
| Rare visit | Low | * | * | Tail; may not need ontology coverage |
| Dark | High | None | Low | Visited but unexplained — investigate |

The **Missing concept** and **Dark** categories drive the active-learning loop in Phase 5.

### Phase 5: Active Exploration (deferred to follow-up)

For each high-occupancy unmapped cell, generate prompts designed to push the model toward producing activations in that cell. Use the cell centroid as a steering target (gradient-guided prompt generation, similar to existing self_concept_steered work). Verify cell-targeting works; collect activations + generated text for human review; if a coherent semantic cluster emerges, add as new concept to the ontology.

### Phase 6: Quantization Evaluation

Replicate the `quant_sensitivity.py` experiment using cell-ID quantization in place of polar-Nb:
- Cell-ID encoder: `idx = argmin_k ||x - centroid_k||`
- Cell-ID decoder: `x_q = centroid_idx`
- Compute top-K nearest-neighbor overlap on the existing first-light lens pack signatures, comparing cell-ID-quantized vs fp32 baseline

Expected result: outperforms polar-8b at K=10,000; comparable to fp32 at K=100,000.

## Evaluation Metrics

| Metric | Target |
|---|---|
| Cell coverage by concept lens | ≥80% of cells with ≥1 concept hit |
| Mean cell purity (lower is better) | <0.5 normalized entropy |
| Cell-ID top-10 overlap @ K=10k | >0.90 (vs polar-8b baseline 0.888) |
| Within-cell variance, Gemma 4 vs Gemma 3 | Gemma 4 lower (validates H5) |
| Identified candidate concepts | ≥10 cells with strong text correlation but no lens correspondence |

## Reusable Infrastructure (from prior work)

From `results/topology/20260115_221611/` (Topology Probing 001):
- `clusters/clusters.json`, `feature_matrix.npy`, `neuron_to_cluster.json` — clustering machinery and storage format (adapt for activation vectors instead of neurons)
- `connectivity/` — inter-cluster graph machinery; reusable for activation-cell connectivity
- `fuzz_results/`, `trace_results/` — fuzzing infrastructure; directly applicable to Phase 5 active exploration

From the recent Gemma 4 training (2026-05-02→):
- Already collecting per-concept activations during training; needs persistence flag added
- `results/gemma4_e4b_first_light/` — pack output that this experiment will read from

From Quant Sensitivity (2026-05-02):
- `scripts/analysis/quant_sensitivity.py` — top-K overlap evaluation framework; reuse for Phase 6

## Open Questions / Risks

1. **Activation persistence overhead.** Saving activations during training may slow the Gemma 4 run further. If the run is already at "best part of a week," adding I/O could push it past acceptable. Mitigation: persist only at the network layer chosen per concept, not all 42 layers.

2. **K selection without ground truth.** Elbow + silhouette suggested k=25 vs k=50 in Topology 001. For activation clustering with 7947 concept lenses available, the lens-purity metric provides a much stronger signal — choose K to maximize mean cell purity subject to ≥X% coverage. New methodology.

3. **Cross-network-layer comparability.** Each ontology layer is best-detected at a different network layer. Cell codebooks are per-network-layer; comparison and aggregation across them needs care. May want a single global codebook on a representative network layer (e.g., the median selected layer).

4. **PLE confound for H5.** Gemma 4 E4B's tighter clustering (if observed) could be due to PLE or due to better training / larger pretraining corpus / instruction tuning differences. Disentangling requires comparison with a non-PLE Gemma 4 variant which doesn't exist as released.

5. **Polysemantic vs under-resolved indistinguishability at fixed K.** A cell with mixed concept assignments could be either truly polysemantic (multiple concepts genuinely share representation) or under-resolved (subdividing would separate them). Resolution: bottom-up cell merging vs top-down splitting comparison; not addressed in this plan.

6. **Storage cost.** 14 GB per ontology layer × 7 = ~100 GB. Tractable on EC2 disk (124 GB free) but tight if combined with other artifacts. Consider compressed storage (bf16, or even int8 since activations will be quantized anyway in the downstream analysis).

## Connection to Deployment

If H3 holds (cell-ID quantization beats polar-8b at K=10k), this directly enables interpretability for frontier-scale open models that don't fit at fp16. Specifically:

- Train lens pack at fp32 on a tractable model (one-time cost)
- Build activation-cell codebook from the same training run (additional ~hours)
- Deploy quantized inference: model runs at 4-bit weights, activations encoded as 13-bit cell IDs, lens scoring done in fp32 against cell centroids

Makes the difference between "interpretability for the small accessible models" and "interpretability for the actually-deployed-class models." That's the practical north star.

If H4 holds (text correlations identify missing concepts), this provides a continuous improvement loop for the ontology: every training run surfaces gap candidates that can feed back into the next concept pack iteration. Closes the active-learning loop described in [Fractal Model Cartography](../planning/fractal-model-cartography.md).

## Timeline / Dependencies

- **Blocked by:** Gemma 4 E4B lens pack training completion (~2026-05-08)
- **Phase 1 modification:** ~1 hour code change to persist activations at chosen network layer
- **Phases 2–4:** ~1 day on EC2 CPU once activations are persisted
- **Phase 6 quant evaluation:** ~hours
- **Phase 5 active exploration:** open-ended, follow-up scope

Total: ~1–2 days of work after Gemma 4 training finishes.

## Decision Points After Phase 4 Results

- If coverage <50%: ontology is the bottleneck. Defer quantization work; focus on Phase 5 active exploration to bootstrap missing concepts.
- If coverage ≥80% and cell purity is high: proceed to Phase 6 quantization, plan deployment integration.
- If coverage is bimodal (some layers high, some low): ontology layer-specific story; some layers production-ready, others need more concepts.
