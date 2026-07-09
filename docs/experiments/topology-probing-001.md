# Topology Probing Experiment 001

**Date:** 2025-01-15
**Part of:** [Fractal Model Cartography](../planning/fractal-model-cartography.md) Phase 3
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 4.3B parameters)
**Duration:** ~30 minutes
**Data Size:** 70GB

## Summary

We ran a "dumb" bidirectional connectivity probing experiment on Gemma 3 4B to map inter-layer neuron connectivity. The approach was intentionally simple: perturb neurons, measure responses. The unexpected finding is that the model's connectivity naturally segments into ~50 distinct clusters, with 16 clusters spanning multiple layers. These cross-layer clusters may represent functional circuits.

## Methodology

### Phase 1: Forward Fuzzing (1-bit probing)
- For each neuron in each layer, inject a small perturbation (+3σ of typical activation)
- Measure which downstream neurons respond above noise threshold
- Batched implementation: 64 neurons per forward pass (17.8x speedup over sequential)
- Result: Forward connectivity matrix for each layer pair

### Phase 2: Backward Tracing (MLP weight analysis)
- For each layer, extract MLP output projection weights
- Vectorized: entire layer's connectivity in one matrix read
- Result: Static backward connectivity based on weight magnitudes

### Phase 3: Aggregation and Clustering
- Combine forward (dynamic) and backward (static) connectivity
- Build feature vectors for each neuron based on connectivity patterns
- K-means clustering with elbow/silhouette optimization

## Results

### Connectivity Map
- **561 layer pairs** analyzed
- **Mean density: 0.991** - nearly all neurons have measurable connectivity
- **42GB** of connectivity data generated

### Dominant Pathways
Top 5 strongest cross-layer connections:
```
L2:N1713 -> L3:N2036 (score=0.952)
L2:N1713 -> L5:N2036 (score=0.943)
L2:N1713 -> L4:N2036 (score=0.942)
L0:N558  -> L3:N1139 (score=0.931)
L0:N558  -> L1:N1139 (score=0.930)
```

**Notable:** Neuron 1713 in layer 2 projects consistently to neuron 2036 across layers 3, 4, and 5. This is exactly the "thin tunnel" pattern we hypothesized - a single neuron maintaining strong connectivity through multiple downstream layers.

### Dynamic Routing Zones
Layers 1-5 identified as "attention-mediated" - high forward connectivity but variable backward traces, suggesting content-dependent routing in early layers.

### Cluster Analysis (Elbow + Silhouette)

| k | Inertia | Silhouette |
|---|---------|------------|
| 10 | 9.66M | 0.383 |
| 25 | 3.70M | 0.451 |
| **50** | 2.10M | **0.474** |
| 100 | 1.87M | 0.284 |
| 150 | 1.72M | 0.183 |
| 200 | 1.63M | 0.139 |
| 300 | 1.48M | 0.117 |
| 500 | 1.29M | 0.095 |

**Optimal k=50** based on silhouette score. The sharp drop after k=50 suggests the model has approximately 50 natural "regions" in connectivity space.

### Final Clusters (k=50)

```
Total neurons:        87,040
Cluster size range:   12 - 2,560
Mean cluster size:    1,740.8
Std cluster size:     1,070.9

Multi-layer clusters: 16 (32%)

Top spanning clusters:
  Cluster 42: spans 8 layers
  Cluster 31: spans 7 layers
  Cluster 35: spans 7 layers
  Cluster 27: spans 5 layers
  Cluster 49: spans 5 layers
```

## Key Findings

### 1. Natural Segmentation Exists
The model's connectivity is not uniformly distributed. It naturally segments into ~50 clusters with good separation (silhouette=0.47). This suggests real structure, not noise.

### 2. Cross-Layer Circuits
16 of 50 clusters (32%) span multiple layers. These are the "tunnels" - groups of neurons that work together across depth. Cluster 42 spans 8 consecutive layers, representing a potential functional circuit.

### 3. Consistent Pathways
Individual neurons (e.g., L2:N1713) maintain strong connectivity to specific downstream neurons across multiple layers. This contradicts the "everything connects to everything" null hypothesis.

### 4. Early Layer Dynamics
Layers 1-5 show high forward connectivity but inconsistent backward traces, suggesting attention-mediated routing. Later layers may have more fixed pathways.

## Unexpected Observations

1. **High overall density (0.991)** - Almost everything connects to almost everything at some level, but the *strength* of connections varies dramatically.

2. **Silhouette collapse after k=50** - Quality drops sharply from 0.47 to 0.28 when going from 50 to 100 clusters. The 50-cluster structure seems fundamental.

3. **8-layer spanning clusters** - We expected cross-layer structure but 8 layers (24% of the model) in a single coherent cluster is surprisingly deep.

## Limitations

1. **Partial semantic grounding** - Only layers 0-1 have trained classifiers; cross-layer clusters (layers 7+) remain unlabeled
2. **Static baseline** - Used single random context for perturbation baseline
3. **MLP-only backward** - Didn't trace through attention (QKV) paths
4. **Subsampled contexts** - 200 contexts for attention tracing, may need more
5. **Layer-local classifiers** - Existing lense packs train per-layer, missing cross-layer semantic patterns

## Next Steps

### Completed: Cross-reference with Lense Pack Classifiers
We mapped trained classifier weights to topology clusters (see Semantic Grounding section). Early-layer clusters were successfully labeled, but cross-layer clusters remain unlabeled.

### Future Work
1. **Train multi-layer classifiers** - Create concept classifiers for layers 7-24 where cross-layer clusters exist
2. **Direct cluster probing** - Activate specific prompts and measure which clusters respond
3. **Visualize cross-layer pathways** - 3D graph of cluster 42 and other spanning clusters
4. **Compare topology across model scales** - Do larger models have more/fewer cross-layer clusters?
5. **Test cluster structure vs training dynamics** - Do clusters predict which neurons change during fine-tuning?
6. **Attention path tracing** - Extend backward tracing to include QKV weights

## Files Generated

```
results/topology/20260115_221611/
├── fuzz_results/           (14GB) - Forward fuzzing connectivity
├── trace_results/          (14GB) - Backward MLP connectivity
├── connectivity/           (42GB) - Combined connectivity matrices
├── clusters/               - Final cluster assignments
│   ├── clusters.json       - Cluster definitions with neuron lists
│   ├── metadata.json       - Clustering parameters
│   └── neuron_to_cluster.json - Quick lookup mapping
├── cluster_analysis.json   - Elbow/silhouette results
└── cluster_semantics.json  - Concept-to-cluster mapping (NEW)
```

**Scripts:**
- `scripts/run_topology_probing.py` - Main probing pipeline
- `scripts/map_clusters_to_concepts.py` - Semantic grounding tool

## Semantic Grounding (Follow-up Analysis)

We cross-referenced the topology clusters with trained lense pack classifiers to identify cluster semantics.

### Methodology
Using `scripts/map_clusters_to_concepts.py`:
1. Load classifier weights from trained lense packs
2. Identify discriminative neurons (top-50 by weight magnitude per concept)
3. Map discriminative neurons to topology cluster IDs
4. Aggregate activation strength per cluster per concept

**Important note:** The lense pack directory names (`layer0/`, `layer1/`) refer to SUMO ontology hierarchy depth, NOT model layers. Most existing classifiers were trained on **model layer 15** activations (the default extraction layer).

### Results

With classifiers trained on model layer 15, we can map concepts to clusters containing layer-15 neurons:

| Cluster | Layers | Size | Top Concepts (from L15 classifiers) |
|---------|--------|------|-------------------------------------|
| **46** | includes 15 | varies | Collection, Physical, Process, Attribute |

**Limitation:** Since all existing classifiers use model layer 15, we can only label clusters that contain layer-15 neurons. Clusters spanning earlier or later layers remain unlabeled.

### Multi-layer spanning clusters remain unlabeled:
- Cluster 42: layers 7-14 (no layer 15)
- Cluster 31: layers 15-21 (partially overlaps)
- Cluster 35: layers 18-24 (no layer 15)

### Next Step: University Concept Pack

To properly ground the cross-layer clusters, we need classifiers trained on different model layers. A "university concept pack" approach would:
1. Select concepts expected to activate different model depths
2. Train multi-layer classifiers that extract from various model layers
3. Create semantic coverage across the full layer range (0-33)

This would transform the structural topology map into a functional semantic map.

### Files Generated

```
results/topology/20260115_221611/
├── cluster_semantics.json     - Concept-to-cluster mapping (layer 15 only)
└── ...
```

## Conclusion

The "too dumb to work" approach worked. Simple perturbation-based probing revealed non-trivial structure: ~50 natural clusters, 16 cross-layer circuits, and specific dominant pathways. The model has internal organization that can be mapped without understanding what it computes - just how it connects.

Semantic grounding revealed a processing hierarchy: early layers form layer-local clusters handling ontological classification, while cross-layer "tunnel" clusters emerge only in mid-to-late layers. Full semantic labeling of cross-layer circuits requires training concept classifiers on those layers.

---

## Phase 4: Pillar-to-Cluster Mapping (2026-01-16)

### Summary

We generated a 3-layer concept taxonomy using Gemma 3 4B (self-description approach), trained 151 lenses, and mapped concept activations to topology clusters. Results are **promising but preliminary** - data quality controls were insufficient.

### What We Did

1. **Generated L1 pillars** (12) using ontologist prompt - Gemma describing human activity through "Action & Agency" lens
2. **Expanded L2 children** (140) - 5-8 sub-concepts per pillar with 30 positive/negative examples each
3. **Expanded L3 grandchildren** (1078) - 5-8 sub-concepts per L2
4. **Trained multi-layer lenses** - 151/152 concepts successfully trained, avg F1=0.955
5. **Mapped to clusters** - Extracted activations for 1211 concepts, measured cluster selectivity

### Results

**Coarse mapping (12 L1 pillars):**
| Cluster | Best Pillar | Selectivity |
|---------|-------------|-------------|
| 6 | Governance | 9.69x |
| 47 | Violence/Conflict | 7.79x |
| 40 | Migration | 2.56x |
| 14 | Social Bonds | 2.51x |

**Fine-grained mapping (1211 concepts):**
| Cluster | Best Concept | Selectivity |
|---------|--------------|-------------|
| 7 | PeerInteraction_club-activities | **48.94x** |
| 25 | BeliefSystems_philosophical-theology | 11.26x |
| 39 | MovementOfGoods_bulk_commodity_transport | 7.99x |
| 6 | SocialSupport_mentorship-guidance | 4.98x |
| 29 | LegalFramework_legal-compliance | 4.91x |

**Key observation:** Finer concept granularity reveals more specific cluster specializations. "Governance" at L1 becomes "legal compliance" at L3. Cluster 7 shows extreme selectivity (48x) for peer/club activities specifically.

### Limitations and Data Quality Concerns

**This experiment has significant methodological limitations:**

1. **No ontology grounding** - Concepts generated ad-hoc without SUMO or other formal ontology anchoring
2. **No MELD format** - Missing:
   - Exclusion clauses (what ISN'T this concept)
   - Tie-break rules for ambiguous cases
   - Disambiguation from confusable concepts
   - Scope boundaries
3. **No example review** - Auto-generated examples accepted without validation
4. **MECE violations likely** - L3 concepts under different L2 parents may overlap significantly
5. **Circularity risk** - Model generating its own training data, then being probed with that data

**The high selectivity scores could be artifacts of:**
- Poorly-separated concepts (one concept is just more specific, not genuinely exclusive)
- Example leakage between siblings
- Overfitting to generation artifacts

### Interpretation

These results should be treated as **exploratory hypothesis generation**, not validated findings. The signal is:
- Topology clusters DO respond differently to different semantic content
- Finer concept granularity reveals more specific associations
- Some clusters appear specialized for narrow domains

The caution is:
- We cannot confirm these are real semantic specializations vs. data artifacts
- The concept taxonomy needs tightening before conclusions can be drawn

### Next Steps: Tightening Data Quality

To validate these preliminary findings, we need:

1. **MELD-format concept definitions**
   - Add exclusion clauses and tie-break rules
   - Define scope boundaries explicitly
   - Use judge model to validate definitions

2. **Example validation pipeline**
   - Judge model scores each example for concept fit
   - Deterministic tests for edge cases (same tests we use on judge models)
   - Cross-validation: test probes against sibling examples

3. **MECE enforcement**
   - Explicit disambiguation between sibling concepts
   - Measure actual probe discriminability on held-out sibling data
   - Flag concepts with high cross-activation

4. **Ontology grounding (optional)**
   - Map L2/L3 concepts to SUMO terms where possible
   - Identify gaps in coverage

### Files Generated

```
concept_packs/action-agency-pillars/
├── pack.json                    - Pack manifest (v0.2.0)
├── hierarchy/
│   ├── layer0.json              - 12 L1 pillars
│   ├── layer1.json              - 140 L2 children (with examples)
│   └── layer2.json              - 1078 L3 grandchildren

lens_packs/gemma3-4b_action-agency/
├── layer0/                      - 12 L1 lenses (avg F1=0.957)
├── layer1/                      - 139 L2 lenses (avg F1=0.953)
└── version_manifest.json

results/
├── pillar_cluster_mapping.json  - L1 coarse mapping
└── concept_cluster_mapping.json - L1+L2+L3 fine mapping

scripts/
├── expand_pillars_gemma.py      - L1 expansion
├── expand_l2_to_l3.py           - L2→L3 expansion
├── format_pillar_pack.py        - Convert to pack format
└── map_pillars_to_clusters.py   - Cluster correlation analysis
```

### Time Investment

Total: ~3 hours
- Pillar expansion: 15 min
- L2→L3 expansion: 80 min
- Lens training: 65 min
- Cluster mapping: 20 min

---

## See Also

- [Fractal Model Cartography](../planning/fractal-model-cartography.md) - Research proposal (Phase 3-4)
- Next: Tighten data quality using judge model + deterministic testing before drawing conclusions
