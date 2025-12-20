# Graft Protocol

> **Structured integration of learned concepts into BE substrates**
>
> This protocol defines how grafts (concept dimensions + weight biases) are derived
> from lens analysis, trained from Experience Database exemplars, merged into
> substrate parameters, and shared across BE instances via MAP.

**Status**: Draft specification
**Extends**: MAP_MELD_PROTOCOL.md, BE_EXPERIENCE_DATABASE.md, BE_CONTINUAL_LEARNING.md
**Related**: HATCAT_MELD_POLICY.md, MINDMELD_ARCHITECTURAL_PROTOCOL.md
**Governance**: Graft submissions follow the [ASK TRACE Protocol](../ASK/ASK_TRACE.md)
(Tender → Review → Authorise → Commit → Evaluate) for governed state changes.

---

## 1. Overview

### 1.1 Core Principle

Grafts **grow the substrate**. When a BE learns a new concept:

1. A **new dimension** is added to the substrate — a labelled feature for the concept
2. **Biases** are added to existing weights — encoding how the concept relates to everything else
3. A **lens** is trained that reads from the new dimension (plus auxiliary features)
4. The graft is validated and **permanently merged**
5. The substrate is now larger by one dimension

The Experience Database is the **source of truth**. Grafts are ephemeral derivatives that can be regenerated from exemplars. When concepts overlap, we cotrain from pooled data rather than naively merging grafts.

### 1.2 Substrate Growth Model

A small trunk can scale to understand everything through successive grafts:

```
Trunk (N dimensions)
  + Graft(concept_A) → Trunk (N+1 dimensions)
  + Graft(concept_B) → Trunk (N+2 dimensions)
  + Graft(concept_C) → Trunk (N+3 dimensions)
  ...
```

Each graft adds:
- **One labelled dimension** — the concept's identity in parameter space
- **Biases to existing weights** — scaled by how much training changed them
- **Relational structure** — implicit in the bias pattern (no separate edge weights needed)

This means:
- The lens has a **direct feature** to read — cheap, reliable detection
- The bias pattern **is** the concept's meaning — its relationship to everything else
- New concepts **build on old ones** — the substrate accumulates understanding
- Trunk size limits **dimension count**, not parameter count

Note: For very large, heavily pre-trained trunks, Graft SHOULD be preceded by a distillation / alignment step to a smaller “concept kernel” substrate. Directly grafting onto an un-instrumented 400B trunk risks baking in correlational shortcuts as if they were concepts.

### 1.3 Relationship to Existing Protocols

| Protocol | Role in Graft |
|----------|---------------|
| **MAP** | Concept identity, lens registration, GraftDiff distribution |
| **MELD** | Governance, protection levels, review processes |
| **XDB** | Exemplar storage, training provenance, dataset management |
| **Continual Learning Harness** | Gap detection, candidate concept lifecycle |
| **Hush** | Constraints on what regions may be modified |

Graft adds:

- **Dimension allocation** — adding labelled features for concepts
- **Bias derivation** — encoding relational structure in weight deltas
- **Lens binding** — linking lenses to their primary dimension
- **Cotrain triggers** — when concepts share bias patterns
- **Substrate maintenance** — pruning, merging, defragmentation

---

## 2. Conceptual Model

### 2.1 Concepts as Dimensions

Each learned concept becomes a **dimension** in the substrate's representation space:

```
substrate.hidden_dim = base_dim + num_grafted_concepts
```

The dimension is **labelled** — we know exactly which feature corresponds to which concept. This is unlike emergent features in a pretrained model, which are unlabelled and entangled.

### 2.2 Biases as Relational Evidence

When training a graft, the learning process modifies existing weights. These modifications are preserved as **biases**:

```
bias_pattern[concept_X] = {
  layer_18_mlp: sparse_delta,
  layer_20_attn: sparse_delta,
  ...
}
```

The bias pattern encodes:
- **Which existing features** the concept relates to
- **How strongly** (magnitude of change)
- **In what direction** (sign of change)

This is the concept's "meaning" in terms of its effect on the substrate. No separate relational weights needed — the evidence is in the parameters.

### 2.3 Lenses with Direct Features

A lens for concept X reads:
- **Primary**: the labelled dimension for X (direct, reliable)
- **Auxiliary**: other dimensions identified by region analysis (context)

```
score_X = σ(w_primary · h[dim_X] + w_aux · h[aux_dims] + b)
```

Because the primary dimension is labelled, lens training is easier and more robust. The auxiliary dimensions capture context and nuance.

### 2.4 Symmetry Principle

```
Graft:  "Add dimension for concept X, bias existing weights to wire it in"
Lens:  "Read from dimension X (plus context) to detect activation"
```

The graft writes the concept into the substrate. The lens reads it back out. They share the same primary dimension.

---

## 3. Data Model

### 3.1 Graft

A **Graft** is the complete package for adding a concept to a substrate.

```jsonc
Graft = {
  "graft_id": "graft-Fish-v1",
  "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Fish",
  "concept_version": "4.0.0",

  // === THE NEW DIMENSION ===
  "concept_dimension": {
    "dimension_index": 2049,  // position in expanded substrate
    "dimension_label": "concept/Fish",  // human-readable label
    "initial_weight": 1.0,  // initialization value

    // Where this dimension is injected in the architecture
    "injection_points": [
      {
        "layer": 18,
        "component": "residual",  // added to residual stream
        "projection": "blob://grafts/Fish-v1/layer18_proj.safetensors"
      }
    ]
  },

  // === BIASES TO EXISTING WEIGHTS ===
  "substrate_biases": [
    {
      "layer": 18,
      "component": "mlp.up_proj",
      "bias_delta": {
        "format": "sparse_coo",  // only store nonzero values
        "location": "blob://grafts/Fish-v1/layer18_mlp_bias.safetensors",
        "nnz": 1247,  // number of nonzero entries
        "shape": [2048, 8192]
      },
      "magnitude_stats": {
        "mean": 0.0023,
        "max": 0.089,
        "l2_norm": 0.34
      }
    },
    {
      "layer": 20,
      "component": "attn.v_proj",
      "bias_delta": {
        "format": "sparse_coo",
        "location": "blob://grafts/Fish-v1/layer20_attn_bias.safetensors",
        "nnz": 892,
        "shape": [2048, 2048]
      },
      "magnitude_stats": {
        "mean": 0.0018,
        "max": 0.056,
        "l2_norm": 0.21
      }
    }
  ],

  // === LENS BINDING ===
  "lens_binding": {
    "lens_id": "org.hatcat/...::lens/Fish",
    "primary_dimension": 2049,  // the new labelled dimension
    "auxiliary_dimensions": [42, 87, 156, 203],  // from region analysis
    "lens_weights_location": "blob://grafts/Fish-v1/lens.safetensors"
  },

  // === RELATIONAL EVIDENCE (derived from bias pattern) ===
  "relational_fingerprint": {
    "method": "bias_correlation",
    "top_correlations": [
      {"concept_id": "concept/Vertebrate", "correlation": 0.82},
      {"concept_id": "concept/Animal", "correlation": 0.71},
      {"concept_id": "concept/Aquatic", "correlation": 0.68}
    ],
    "fingerprint_hash": "sha256:..."  // for fast comparison
  },

  // === PROVENANCE ===
  "training_run_id": "trainrun-Fish-v1",
  "source_region_id": "region-Fish-v1",  // region analysis that guided training
  "dataset_ids": ["dataset-Fish-v1"],

  // === SUBSTRATE COMPATIBILITY ===
  "applies_to": {
    "substrate_id": "olmo3-7b-base@0.1.0",
    "substrate_checksum": "sha256:...",
    "pre_graft_dim": 2048,  // substrate hidden_dim before this graft
    "post_graft_dim": 2049  // substrate hidden_dim after this graft
  },

  // === LIFECYCLE ===
  "supersedes": [],  // previous graft versions this replaces
  "superseded_by": null,

  "validation": {
    "status": "validated",
    "validation_run_id": "valrun-Fish-v1"
  },

  "created_at": "2025-11-30T12:45:00Z",
  "created_by": "BE-instance-123"
}
```

### 3.2 ConceptRegion

A **ConceptRegion** captures where a concept's activity is detected in the *original* substrate dimensions. It guides training but doesn't define the final structure.

```jsonc
ConceptRegion = {
  "region_id": "region-Fish-v1",
  "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Fish",

  // Derived from lens analysis on pre-graft substrate
  "source_lenses": [
    {
      "lens_id": "org.hatcat/...::lens/Fish",
      "lens_version": "2.20251130.0"
    }
  ],

  "derivation": {
    "method": "lens_weight_topk",
    "parameters": {
      "top_k_percent": 15,
      "layers": [18, 20, 22]
    }
  },

  // Dimensions in the ORIGINAL substrate that correlate with this concept
  // These become the auxiliary dimensions in the lens, and guide where biases land
  "layers": [
    {
      "layer_index": 18,
      "component": "mlp",
      "dimension_mask": {
        "format": "sparse_indices",
        "indices": [42, 87, 156, 203],
        "total_dimensions": 2048  // pre-graft dimension count
      }
    }
  ],

  "created_at": "2025-11-30T12:00:00Z"
}
```

### 3.3 Extended TrainingRun

```jsonc
TrainingRun = {
  "id": "trainrun-Fish-v1",
  "type": "graft",  // "graft" | "lens_only" | "cotrain"

  "concept_ids": ["org.hatcat/...::concept/Fish"],

  // Substrate state at training time
  "substrate_id": "olmo3-7b-base@0.1.0",
  "substrate_checksum": "sha256:...",
  "substrate_dim_at_training": 2048,  // pre-graft

  // Region that guided training
  "source_region_id": "region-Fish-v1",

  // Dataset provenance
  "dataset_ids": ["dataset-Fish-v1"],
  "dataset_checksums": {"dataset-Fish-v1": "sha256:..."},

  // Training configuration
  "hyperparams": {
    "learning_rate": 5e-5,
    "epochs": 3,
    "batch_size": 32,
    "dimension_init": "learned",  // "learned" | "zero" | "random"
    "bias_sparsity_target": 0.95,  // encourage sparse biases
    "bias_magnitude_penalty": 0.01  // regularize bias magnitudes
  },

  // Results
  "metrics": {
    "train_loss": 0.12,
    "val_loss": 0.15,
    "concept_f1": 0.89,
    "bias_sparsity_achieved": 0.94,
    "dimension_usage": 0.73,  // how much the new dimension contributes to lens
    "auxiliary_contribution": 0.27  // how much auxiliary dims contribute
  },

  // Cotrain context (if type == "cotrain")
  "cotrain_context": null,

  "status": "succeeded",
  "created_at": "2025-11-30T12:30:00Z",
  "created_by": "BE-instance-123"
}
```

### 3.4 SubstrateManifest

Tracks the current state of a substrate including all grafted dimensions.

```jsonc
SubstrateManifest = {
  "manifest_id": "substrate-olmo3-7b-grafted-v42",

  "base_substrate": {
    "substrate_id": "olmo3-7b-base@0.1.0",
    "base_checksum": "sha256:...",
    "base_hidden_dim": 2048
  },

  // === ARCHITECTURE SPECIFICATION (required for expand mode) ===
  // This enables scion/bud operations to know which weight matrices to modify
  "architecture": {
    "family": "llama",           // "llama" | "gemma" | "gpt2" | "moe" | ...
    "hidden_size": 4096,
    "intermediate_size": 14336,  // MLP intermediate dimension
    "num_attention_heads": 32,
    "num_key_value_heads": 8,    // For GQA; equals num_attention_heads for MHA
    "head_dim": 128,             // hidden_size / num_attention_heads
    "num_layers": 32,

    "mlp_type": "glu",           // "glu" (gate_proj + up_proj) | "standard"
    "attention_type": "gqa",     // "gqa" | "mha" | "mqa"
    "norm_type": "rms_norm",     // "rms_norm" | "layer_norm"

    // Component path templates for expand mode
    // {layer} is replaced with layer index
    "component_paths": {
      "embed_tokens": "model.embed_tokens",
      "lm_head": "lm_head",
      "layers": "model.layers.{layer}",
      "q_proj": "self_attn.q_proj",
      "k_proj": "self_attn.k_proj",
      "v_proj": "self_attn.v_proj",
      "o_proj": "self_attn.o_proj",
      "up_proj": "mlp.up_proj",
      "gate_proj": "mlp.gate_proj",
      "down_proj": "mlp.down_proj",
      "input_layernorm": "input_layernorm",
      "post_attention_layernorm": "post_attention_layernorm"
    },

    // MoE-specific (optional)
    "is_moe": false,
    "num_experts": 1,
    "experts_path": null  // e.g., "block_sparse_moe.experts.{expert}"
  },

  "current_state": {
    "checksum": "sha256:...",
    "hidden_dim": 2091,  // base + 43 grafted concepts
    "total_grafts_applied": 43
  },

  // Dimension allocation table
  "dimension_table": [
    {
      "dimension_index": 2048,
      "concept_id": "concept/Eligibility",
      "graft_id": "graft-Eligibility-v1",
      "grafted_at": "2025-11-01T00:00:00Z"
    },
    {
      "dimension_index": 2049,
      "concept_id": "concept/Fish",
      "graft_id": "graft-Fish-v1",
      "grafted_at": "2025-11-30T13:00:00Z"
    }
    // ...
  ],

  // For dimension management
  "compaction_candidates": [
    {
      "dimensions": [2055, 2067],
      "reason": "high_correlation",
      "correlation": 0.94
    }
  ],

  "updated_at": "2025-11-30T13:00:00Z"
}
```

### 3.5 OverlapAnalysis (updated for bias fingerprints)

```jsonc
OverlapAnalysis = {
  "id": "overlap-Fish-Salmon-2025-11-30",

  "concepts": [
    {"concept_id": "concept/Fish", "graft_id": "graft-Fish-v1"},
    {"concept_id": "concept/Salmon", "graft_id": "graft-Salmon-v1"}
  ],

  // Bias pattern similarity
  "fingerprint_similarity": {
    "method": "bias_correlation",
    "correlation": 0.78,
    "shared_bias_dimensions": 423,
    "total_union_dimensions": 892
  },

  // Region overlap (in original substrate dimensions)
  "region_overlap": {
    "jaccard_index": 0.34,
    "overlapping_auxiliary_dims": [42, 87, 156]
  },

  "recommendation": {
    "action": "cotrain",  // "cotrain" | "independent" | "merge_dimensions"
    "reason": "Bias fingerprint correlation 0.78 exceeds threshold 0.6",
    "threshold_used": 0.6
  },

  "created_at": "2025-11-30T12:15:00Z"
}
```

---

## 4. Region Derivation

Region derivation identifies which *existing* substrate dimensions correlate with a concept. This guides:
- Where biases should land during training
- Which auxiliary dimensions the lens should read

### 4.1 From Lens Weights

```python
def derive_region_from_lens(
    lens: TrainedLens,
    concept_id: str,
    layers: List[int],
    top_k_percent: float = 15.0,
    include_ancestors: bool = True,
    ancestor_weight_decay: float = 0.5
) -> ConceptRegion:
    """
    Derive a ConceptRegion from lens weights.

    For each layer, takes the top k% of dimensions by |weight|.
    These become auxiliary dimensions for the graft's lens.
    """

    region_layers = []

    for layer_idx in layers:
        weights = lens.get_layer_weights(layer_idx)
        importance = np.abs(weights)

        if include_ancestors:
            for ancestor, depth in get_concept_ancestors(concept_id):
                ancestor_lens = load_lens(ancestor)
                ancestor_weights = ancestor_lens.get_layer_weights(layer_idx)
                weight = ancestor_weight_decay ** depth
                importance = importance + weight * np.abs(ancestor_weights)

        k = int(len(importance) * top_k_percent / 100)
        top_indices = np.argsort(importance)[-k:]

        region_layers.append({
            "layer_index": layer_idx,
            "component": "mlp",
            "dimension_mask": {
                "format": "sparse_indices",
                "indices": top_indices.tolist(),
                "total_dimensions": len(importance)
            }
        })

    return ConceptRegion(
        concept_id=concept_id,
        layers=region_layers,
        derivation={"method": "lens_weight_topk", "parameters": {...}}
    )
```

### 4.2 From Gradient Attribution

Alternative using gradient-based importance on concept-positive examples.

```python
def derive_region_from_gradients(
    substrate: Model,
    dataset: ConceptDataset,
    layers: List[int],
    top_k_percent: float = 15.0
) -> ConceptRegion:
    """
    Derive region by measuring gradient magnitude on concept-positive examples.
    """

    importance_accum = defaultdict(lambda: np.zeros(substrate.hidden_dim))

    for exemplar in dataset.positive_exemplars:
        activations = substrate.forward_with_cache(exemplar.text)
        gradients = substrate.backward_to_layers(layers)

        for layer_idx in layers:
            importance_accum[layer_idx] += np.abs(gradients[layer_idx])

    # Normalize and select top k%
    region_layers = []
    for layer_idx in layers:
        importance = importance_accum[layer_idx] / len(dataset.positive_exemplars)
        k = int(len(importance) * top_k_percent / 100)
        top_indices = np.argsort(importance)[-k:]

        region_layers.append({
            "layer_index": layer_idx,
            "dimension_mask": {
                "format": "sparse_indices",
                "indices": top_indices.tolist()
            }
        })

    return ConceptRegion(layers=region_layers, derivation={"method": "gradient_attribution"})
```

---

## 5. Graft Training

### 5.1 Training Procedure

Graft training learns:
1. The **concept dimension** — a new feature that activates for the concept
2. The **substrate biases** — modifications to existing weights that wire the concept in

```python
def train_graft(
    substrate: Model,
    dataset: ConceptDataset,
    region: ConceptRegion,
    config: GraftConfig
) -> Graft:
    """
    Train a graft that adds a concept dimension and biases existing weights.
    """

    # 1. Allocate new dimension
    new_dim_index = substrate.hidden_dim
    substrate.expand_hidden_dim(1)  # Now hidden_dim + 1

    # 2. Initialize projection matrices for the new dimension
    projections = initialize_dimension_projections(
        substrate,
        new_dim_index,
        injection_layers=config.injection_layers
    )

    # 3. Initialize bias accumulators (sparse)
    bias_accum = initialize_sparse_bias_accumulators(
        substrate,
        region  # Only accumulate biases in region-relevant layers
    )

    # 4. Training loop
    optimizer = torch.optim.AdamW(
        list(projections.parameters()) + list(bias_accum.parameters()),
        lr=config.learning_rate
    )

    for epoch in range(config.epochs):
        for batch in dataset.batches(config.batch_size):
            optimizer.zero_grad()

            # Forward with new dimension and biases applied
            with apply_graft_draft(substrate, projections, bias_accum):
                outputs = substrate(batch.inputs)

                # Loss: concept should activate on positive examples
                concept_activations = outputs.hidden_states[:, :, new_dim_index]
                loss = compute_concept_loss(concept_activations, batch.labels)

                # Regularization: encourage sparse biases
                loss += config.bias_sparsity_penalty * bias_accum.l1_norm()
                loss += config.bias_magnitude_penalty * bias_accum.l2_norm()

            loss.backward()
            optimizer.step()

            # Enforce sparsity via thresholding
            bias_accum.threshold_small_values(config.sparsity_threshold)

    # 5. Train lens to read from new dimension + auxiliary
    lens = train_lens_with_primary_dimension(
        substrate=substrate,
        projections=projections,
        bias_accum=bias_accum,
        dataset=dataset,
        primary_dim=new_dim_index,
        auxiliary_dims=region.get_all_indices()
    )

    # 6. Package as Graft
    return Graft(
        concept_dimension={
            "dimension_index": new_dim_index,
            "injection_points": projections.to_spec()
        },
        substrate_biases=bias_accum.to_sparse_deltas(),
        lens_binding={
            "lens_id": lens.id,
            "primary_dimension": new_dim_index,
            "auxiliary_dimensions": region.get_all_indices()
        },
        relational_fingerprint=compute_fingerprint(bias_accum)
    )
```

### 5.2 Lens Training with Primary Dimension

The lens is trained to read primarily from the new labelled dimension.

```python
def train_lens_with_primary_dimension(
    substrate: Model,
    projections: DimensionProjections,
    bias_accum: SparseBiasAccumulator,
    dataset: ConceptDataset,
    primary_dim: int,
    auxiliary_dims: List[int],
    config: LensConfig
) -> Lens:
    """
    Train a lens that reads from the primary (labelled) dimension
    plus auxiliary dimensions from region analysis.
    """

    # Lens architecture: primary weight + auxiliary weights
    lens = Lens(
        primary_dim=primary_dim,
        auxiliary_dims=auxiliary_dims,
        hidden_dim=config.lens_hidden_dim  # optional MLP layer
    )

    optimizer = torch.optim.AdamW(lens.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        for batch in dataset.batches(config.batch_size):
            optimizer.zero_grad()

            # Get activations with graft applied
            with apply_graft_draft(substrate, projections, bias_accum):
                hidden_states = substrate.get_hidden_states(batch.inputs)

            # Lens prediction
            # score = w_primary * h[primary_dim] + w_aux · h[aux_dims] + b
            scores = lens(hidden_states)
            loss = F.binary_cross_entropy_with_logits(scores, batch.labels)

            loss.backward()
            optimizer.step()

    return lens
```

### 5.3 Relational Fingerprint Computation

The bias pattern encodes how the concept relates to existing substrate features.

```python
def compute_fingerprint(bias_accum: SparseBiasAccumulator) -> RelationalFingerprint:
    """
    Compute a fingerprint from the bias pattern for fast comparison.

    Two grafts with similar fingerprints learned similar relational structure,
    even if they're for different concepts.
    """

    # Flatten all biases into a single sparse vector
    flat_biases = bias_accum.flatten()

    # Compute correlation with existing concept dimensions
    correlations = []
    for existing_graft in get_existing_grafts():
        existing_flat = existing_graft.substrate_biases.flatten()
        corr = sparse_correlation(flat_biases, existing_flat)
        if corr > 0.1:  # threshold for relevance
            correlations.append({
                "concept_id": existing_graft.concept_id,
                "correlation": corr
            })

    correlations.sort(key=lambda x: -x["correlation"])

    return RelationalFingerprint(
        method="bias_correlation",
        top_correlations=correlations[:20],
        fingerprint_hash=hash_sparse_vector(flat_biases)
    )
```

---

## 6. Overlap Detection and Cotraining

### 6.1 Overlap Detection via Fingerprints

Before applying a graft, check for fingerprint similarity with existing grafts.

```python
def detect_overlaps(
    new_graft: Graft,
    existing_grafts: List[Graft],
    fingerprint_threshold: float = 0.6,
    region_threshold: float = 0.25
) -> List[OverlapAnalysis]:
    """
    Detect which existing grafts overlap with a new one.

    Uses both fingerprint similarity (bias patterns) and region overlap.
    """

    overlaps = []

    for existing in existing_grafts:
        # Fingerprint similarity (bias patterns)
        fp_similarity = compute_fingerprint_similarity(
            new_graft.relational_fingerprint,
            existing.relational_fingerprint
        )

        # Region overlap (auxiliary dimensions)
        region_overlap = compute_region_jaccard(
            new_graft.lens_binding.auxiliary_dimensions,
            existing.lens_binding.auxiliary_dimensions
        )

        if fp_similarity > fingerprint_threshold or region_overlap > region_threshold:
            overlaps.append(OverlapAnalysis(
                concepts=[new_graft.concept_id, existing.concept_id],
                fingerprint_similarity={"correlation": fp_similarity},
                region_overlap={"jaccard_index": region_overlap},
                recommendation={
                    "action": "cotrain" if fp_similarity > fingerprint_threshold else "review",
                    "reason": f"Fingerprint similarity {fp_similarity:.2f}" if fp_similarity > fingerprint_threshold
                             else f"Region overlap {region_overlap:.2f}"
                }
            ))

    return overlaps
```

### 6.2 Cotrain Procedure

When overlap is detected, cotrain from pooled datasets.

```python
def cotrain_grafts(
    overlapping_concepts: List[str],
    substrate: Model,
    config: CotrainConfig
) -> List[Graft]:
    """
    Cotrain grafts for multiple overlapping concepts.

    Each concept still gets its own dimension, but they're trained
    together to ensure biases are compatible.
    """

    # 1. Collect datasets
    datasets = {c: load_concept_dataset(c) for c in overlapping_concepts}

    # 2. Allocate dimensions for all concepts
    dim_allocation = {}
    for concept_id in overlapping_concepts:
        dim_allocation[concept_id] = substrate.hidden_dim
        substrate.expand_hidden_dim(1)

    # 3. Initialize shared bias accumulator
    # All concepts contribute to the same bias pool during training
    shared_bias_accum = initialize_sparse_bias_accumulators(substrate)

    # 4. Initialize per-concept projections
    projections = {
        c: initialize_dimension_projections(substrate, dim_allocation[c])
        for c in overlapping_concepts
    }

    # 5. Joint training
    optimizer = torch.optim.AdamW(
        [p for proj in projections.values() for p in proj.parameters()] +
        list(shared_bias_accum.parameters()),
        lr=config.learning_rate
    )

    for epoch in range(config.epochs):
        # Interleave batches from all concept datasets
        for concept_id, batch in interleave_datasets(datasets, config.batch_size):
            optimizer.zero_grad()

            with apply_graft_draft(substrate, projections[concept_id], shared_bias_accum):
                outputs = substrate(batch.inputs)
                activations = outputs.hidden_states[:, :, dim_allocation[concept_id]]
                loss = compute_concept_loss(activations, batch.labels)
                loss += config.bias_sparsity_penalty * shared_bias_accum.l1_norm()

            loss.backward()
            optimizer.step()

    # 6. Split biases by attribution
    # Each concept gets the biases that were primarily driven by its examples
    per_concept_biases = attribute_biases_to_concepts(
        shared_bias_accum,
        datasets,
        substrate,
        projections,
        dim_allocation
    )

    # 7. Package individual grafts
    grafts = []
    for concept_id in overlapping_concepts:
        lens = train_lens_with_primary_dimension(
            substrate=substrate,
            projections=projections[concept_id],
            bias_accum=per_concept_biases[concept_id],
            dataset=datasets[concept_id],
            primary_dim=dim_allocation[concept_id],
            auxiliary_dims=get_region(concept_id).get_all_indices()
        )

        grafts.append(Graft(
            concept_id=concept_id,
            concept_dimension={"dimension_index": dim_allocation[concept_id]},
            substrate_biases=per_concept_biases[concept_id].to_sparse_deltas(),
            lens_binding={"primary_dimension": dim_allocation[concept_id], "lens_id": lens.id},
            relational_fingerprint=compute_fingerprint(per_concept_biases[concept_id]),
            cotrain_context={
                "cotrained_with": [c for c in overlapping_concepts if c != concept_id]
            }
        ))

    return grafts
```

---

## 7. Substrate Growth and Merge

### 7.1 Applying a Graft

Grafts permanently expand the substrate.

```python
def apply_graft(
    substrate: Model,
    manifest: SubstrateManifest,
    graft: Graft,
    validation_required: bool = True
) -> Tuple[Model, SubstrateManifest]:
    """
    Permanently apply a graft to a substrate.

    This:
    1. Expands the substrate by one dimension
    2. Adds the concept's projection matrices
    3. Applies biases to existing weights
    4. Registers the lens
    """

    # 1. Validate substrate compatibility
    if graft.applies_to.pre_graft_dim != substrate.hidden_dim:
        raise IncompatibleSubstrateError(
            f"Graft expects substrate dim {graft.applies_to.pre_graft_dim}, "
            f"got {substrate.hidden_dim}"
        )

    # 2. Validate graft if required
    if validation_required:
        validation_result = validate_graft(substrate, graft)
        if not validation_result.passed:
            raise GraftValidationError(validation_result.errors)

    # 3. Expand substrate dimension
    substrate.expand_hidden_dim(1)
    new_dim_index = substrate.hidden_dim - 1

    assert new_dim_index == graft.concept_dimension.dimension_index, \
        "Dimension index mismatch - substrate may have changed"

    # 4. Load and apply projection matrices
    for injection_point in graft.concept_dimension.injection_points:
        projection = load_tensor(injection_point.projection)
        layer = substrate.layers[injection_point.layer]

        if injection_point.component == "residual":
            layer.register_residual_injection(new_dim_index, projection)
        # ... other injection types

    # 5. Apply biases to existing weights
    for bias_spec in graft.substrate_biases:
        bias_delta = load_sparse_tensor(bias_spec.bias_delta.location)
        layer = substrate.layers[bias_spec.layer]
        component = getattr(layer, bias_spec.component)

        # Add bias (sparse addition)
        component.weight.data += bias_delta.to_dense()

    # 6. Register lens
    register_lens(graft.lens_binding)

    # 7. Update manifest
    manifest.current_state.hidden_dim = substrate.hidden_dim
    manifest.current_state.total_grafts_applied += 1
    manifest.dimension_table.append({
        "dimension_index": new_dim_index,
        "concept_id": graft.concept_id,
        "graft_id": graft.graft_id,
        "grafted_at": datetime.now().isoformat()
    })

    return substrate, manifest
```

### 7.2 Dimension Management

As the substrate grows, dimension management becomes important.

```python
def analyze_compaction_candidates(
    manifest: SubstrateManifest,
    substrate: Model,
    correlation_threshold: float = 0.9,
    usage_threshold: float = 0.01
) -> List[CompactionCandidate]:
    """
    Identify dimensions that could be merged or pruned.

    Candidates:
    - Dimensions that always co-activate (high correlation)
    - Dimensions that rarely activate (low usage)
    """

    candidates = []

    # Analyze activation patterns over recent experience
    activations = collect_dimension_activations(substrate, recent_episodes=1000)

    # Find highly correlated dimension pairs
    correlation_matrix = np.corrcoef(activations)

    for i, entry_i in enumerate(manifest.dimension_table):
        for j, entry_j in enumerate(manifest.dimension_table):
            if i >= j:
                continue

            corr = correlation_matrix[
                entry_i.dimension_index,
                entry_j.dimension_index
            ]

            if corr > correlation_threshold:
                candidates.append(CompactionCandidate(
                    dimensions=[entry_i.dimension_index, entry_j.dimension_index],
                    concepts=[entry_i.concept_id, entry_j.concept_id],
                    reason="high_correlation",
                    correlation=corr,
                    recommendation="merge"
                ))

    # Find low-usage dimensions
    usage_rates = activations.mean(axis=0)

    for entry in manifest.dimension_table:
        usage = usage_rates[entry.dimension_index]
        if usage < usage_threshold:
            candidates.append(CompactionCandidate(
                dimensions=[entry.dimension_index],
                concepts=[entry.concept_id],
                reason="low_usage",
                usage_rate=usage,
                recommendation="prune_or_retrain"
            ))

    return candidates
```

### 7.3 Dimension Merging

When two concepts are highly correlated, merge their dimensions.

```python
def merge_dimensions(
    substrate: Model,
    manifest: SubstrateManifest,
    dim_a: int,
    dim_b: int,
    merged_concept_id: str
) -> Tuple[Model, SubstrateManifest, Graft]:
    """
    Merge two dimensions into one.

    The merged dimension represents both concepts.
    Original datasets are pooled and a new graft is trained.
    """

    concept_a = manifest.get_concept_for_dimension(dim_a)
    concept_b = manifest.get_concept_for_dimension(dim_b)

    # 1. Pool datasets
    dataset_a = load_concept_dataset(concept_a)
    dataset_b = load_concept_dataset(concept_b)
    merged_dataset = merge_datasets(dataset_a, dataset_b, new_concept_id=merged_concept_id)

    # 2. Remove old dimensions from substrate
    substrate.remove_dimensions([dim_a, dim_b])

    # 3. Update manifest
    manifest.dimension_table = [
        e for e in manifest.dimension_table
        if e.dimension_index not in [dim_a, dim_b]
    ]
    manifest.reindex_dimensions()  # Adjust indices after removal

    # 4. Train new graft for merged concept
    region = derive_region_from_gradients(substrate, merged_dataset)
    merged_graft = train_graft(substrate, merged_dataset, region)

    # 5. Apply merged graft
    substrate, manifest = apply_graft(substrate, manifest, merged_graft)

    # 6. Record merge in MELD
    emit_concept_merge_diff(concept_a, concept_b, merged_concept_id)

    return substrate, manifest, merged_graft
```

### 7.4 Substrate Maintenance

Substrates grow indefinitely through grafting. There is no "graduation" to a larger trunk — the trunk *is* larger because you grafted onto it. The base dimensions become a historical artifact as the substrate accumulates concepts.

Maintenance operations keep the substrate healthy:

#### 7.4.1 Dimension Pruning

Remove dimensions that are no longer useful.

```python
def prune_unused_dimensions(
    substrate: Model,
    manifest: SubstrateManifest,
    usage_threshold: float = 0.01,
    episode_window: int = 10000
) -> Tuple[Model, SubstrateManifest, List[str]]:
    """
    Prune dimensions that rarely activate.

    A dimension with <1% activation rate over 10k episodes
    is a candidate for removal.
    """

    activations = collect_dimension_activations(substrate, episode_window)
    usage_rates = activations.mean(axis=0)

    pruned_concepts = []
    dims_to_remove = []

    for entry in manifest.dimension_table:
        usage = usage_rates[entry.dimension_index]
        if usage < usage_threshold:
            dims_to_remove.append(entry.dimension_index)
            pruned_concepts.append(entry.concept_id)

            # Archive the graft data before removal
            archive_graft(entry.graft_id, reason="low_usage", usage_rate=usage)

    if dims_to_remove:
        substrate.remove_dimensions(dims_to_remove)
        manifest.remove_entries(dims_to_remove)
        manifest.reindex_dimensions()

    return substrate, manifest, pruned_concepts
```

#### 7.4.2 Dimension Defragmentation

After many prune/merge operations, dimension indices may have gaps. Defragmentation renumbers dimensions contiguously.

```python
def defragment_dimensions(
    substrate: Model,
    manifest: SubstrateManifest
) -> Tuple[Model, SubstrateManifest, Dict[int, int]]:
    """
    Renumber dimensions to be contiguous.

    Returns a mapping of old_index -> new_index for updating
    any external references (lens bindings, etc).
    """

    current_indices = sorted([e.dimension_index for e in manifest.dimension_table])

    # Check if already contiguous
    expected = list(range(manifest.base_substrate.base_hidden_dim,
                         manifest.base_substrate.base_hidden_dim + len(current_indices)))

    if current_indices == expected:
        return substrate, manifest, {}  # Already defragmented

    # Build remapping
    remap = {}
    for new_offset, old_index in enumerate(current_indices):
        new_index = manifest.base_substrate.base_hidden_dim + new_offset
        if old_index != new_index:
            remap[old_index] = new_index

    # Apply to substrate
    substrate.remap_dimensions(remap)

    # Update manifest
    for entry in manifest.dimension_table:
        if entry.dimension_index in remap:
            entry.dimension_index = remap[entry.dimension_index]

    # Update lens bindings
    for entry in manifest.dimension_table:
        update_lens_primary_dimension(entry.graft_id, remap)

    return substrate, manifest, remap
```

#### 7.4.3 Lens Refresh

Periodically retrain lenses to account for substrate drift.

```python
def refresh_stale_lenses(
    substrate: Model,
    manifest: SubstrateManifest,
    staleness_threshold_days: int = 90,
    drift_threshold: float = 0.1
) -> List[str]:
    """
    Identify and retrain lenses that have drifted from their concepts.

    A lens is stale if:
    - It hasn't been retrained in >90 days, OR
    - Its F1 on held-out data has dropped >10% from training time
    """

    refreshed = []

    for entry in manifest.dimension_table:
        lens = load_lens(entry.graft_id)

        # Check staleness
        days_since_training = (now() - lens.trained_at).days

        # Check drift
        current_f1 = evaluate_lens_f1(lens, substrate)
        drift = lens.training_f1 - current_f1

        if days_since_training > staleness_threshold_days or drift > drift_threshold:
            # Retrain lens with current substrate state
            dataset = load_concept_dataset(entry.concept_id)
            new_lens = train_lens_with_primary_dimension(
                substrate=substrate,
                dataset=dataset,
                primary_dim=entry.dimension_index,
                auxiliary_dims=get_current_auxiliary_dims(entry.concept_id, substrate)
            )

            update_lens_binding(entry.graft_id, new_lens)
            refreshed.append(entry.concept_id)

    return refreshed
```

#### 7.4.4 Maintenance Schedule

Recommended maintenance cadence:

| Operation | Frequency | Trigger |
|-----------|-----------|---------|
| Dimension pruning | Monthly | Or when dimension count exceeds soft limit |
| Dimension merging | As needed | When overlap analysis detects correlation >0.9 |
| Defragmentation | After prune/merge | When gap ratio exceeds 10% |
| Lens refresh | Quarterly | Or when drift detected |
| Bias sparsification | After major graft batches | When average bias density drops below 0.8 |

These operations can be batched for efficiency. A full maintenance pass might:

1. Run overlap analysis → merge highly correlated dimensions
2. Run usage analysis → prune dead dimensions
3. Defragment if needed
4. Refresh stale lenses
5. Update SubstrateManifest checksum

The substrate grows indefinitely. Maintenance keeps it healthy.

---

## 8. Integration with MELD

### 8.1 GraftDiff

Extension to MAP diffs for sharing grafts.

```jsonc
GraftDiff = {
  "type": "GraftDiff",
  "from_model_id": "BE-source-01",
  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  // The graft being shared
  "graft_id": "graft-Fish-v1",
  "graft_checksum": "sha256:...",

  // Core graft metadata for receiver to evaluate
  "concept_id": "org.hatcat/...::concept/Fish",
  "dimension_expansion": 1,  // How many dimensions this adds

  // Relational fingerprint for overlap detection
  "relational_fingerprint": {
    "top_correlations": [
      {"concept_id": "concept/Vertebrate", "correlation": 0.82},
      {"concept_id": "concept/Aquatic", "correlation": 0.68}
    ],
    "fingerprint_hash": "sha256:..."
  },

  // Bias metadata
  "bias_summary": {
    "total_layers_modified": 3,
    "total_nnz": 2139,
    "max_magnitude": 0.089,
    "sparsity": 0.94
  },

  // Training provenance
  "training_run_id": "trainrun-Fish-v1",
  "dataset_ids": ["dataset-Fish-v1"],

  // Substrate requirements
  "requires_substrate": {
    "substrate_id": "olmo3-7b-base@0.1.0",
    "min_dim": 2048,  // Substrate must have at least this many dimensions
    "max_dim": 4096   // And at most this many (for compatibility)
  },

  // Protection assessment (per MELD protocol)
  "protection_assessment": {
    "protection_level": "standard",
    "triggers": [],
    "ask_flags": {
      "harness_relevant": false,
      "treaty_relevant": false
    }
  },

  // Supersession
  "supersedes_grafts": ["graft-Fish-v0"],

  "created": "2025-11-30T13:00:00Z"
}
```

### 8.2 Receiving a GraftDiff

When a BE receives a GraftDiff:

```python
def handle_graft_diff(
    diff: GraftDiff,
    local_substrate: Model,
    local_manifest: SubstrateManifest
) -> GraftAcceptanceDecision:
    """
    Decide whether to accept and apply an incoming graft.
    """

    # 1. Check protection level against local policy
    if diff.protection_assessment.protection_level == "critical":
        if not has_guardian_approval(diff):
            return GraftAcceptanceDecision(
                action="reject",
                reason="Critical graft requires guardian approval"
            )

    # 2. Check substrate compatibility
    if local_substrate.hidden_dim < diff.requires_substrate.min_dim:
        return GraftAcceptanceDecision(
            action="reject",
            reason=f"Substrate too small: {local_substrate.hidden_dim} < {diff.requires_substrate.min_dim}"
        )

    if local_substrate.hidden_dim > diff.requires_substrate.max_dim:
        return GraftAcceptanceDecision(
            action="reject",
            reason=f"Substrate too large: {local_substrate.hidden_dim} > {diff.requires_substrate.max_dim}"
        )

    # 3. Check for fingerprint overlaps with local grafts
    local_grafts = get_local_grafts(local_manifest)

    for local_graft in local_grafts:
        similarity = compute_fingerprint_similarity(
            diff.relational_fingerprint,
            local_graft.relational_fingerprint
        )

        if similarity > 0.6:
            # High similarity - might be same concept learned independently
            return GraftAcceptanceDecision(
                action="cotrain_required",
                reason=f"Fingerprint similarity {similarity:.2f} with {local_graft.concept_id}",
                cotrain_with=[local_graft.concept_id]
            )

    # 4. Check if concept already exists locally
    if concept_exists_locally(diff.concept_id, local_manifest):
        return GraftAcceptanceDecision(
            action="version_compare",
            reason="Concept already exists locally",
            local_version=get_local_graft_version(diff.concept_id)
        )

    # 5. Fetch and validate
    graft = fetch_graft(diff.graft_id)
    validation = validate_graft(local_substrate, graft)

    if not validation.passed:
        return GraftAcceptanceDecision(
            action="reject",
            reason=f"Validation failed: {validation.errors}"
        )

    # 6. Accept and queue for application
    return GraftAcceptanceDecision(
        action="accept",
        application_priority="normal",
        expected_new_dim=local_substrate.hidden_dim + 1
    )
```

### 8.3 Dimension Index Reconciliation

Two BEs that learned the same concept independently will have different dimension indices. When sharing grafts:

```python
def reconcile_dimension_indices(
    incoming_graft: Graft,
    local_manifest: SubstrateManifest
) -> Graft:
    """
    Remap an incoming graft's dimension index to the local substrate.

    The incoming graft was trained on a substrate where it got dimension N.
    Our substrate already has M dimensions. The graft will get dimension M.
    """

    local_next_dim = local_manifest.current_state.hidden_dim
    incoming_dim = incoming_graft.concept_dimension.dimension_index

    # Create remapped graft
    remapped = incoming_graft.copy()
    remapped.concept_dimension.dimension_index = local_next_dim
    remapped.lens_binding.primary_dimension = local_next_dim

    # Note: auxiliary dimensions refer to the ORIGINAL substrate's features,
    # which should be stable across compatible substrates (same base model)

    # Update applies_to
    remapped.applies_to.pre_graft_dim = local_manifest.current_state.hidden_dim
    remapped.applies_to.post_graft_dim = local_manifest.current_state.hidden_dim + 1

    return remapped
```

---

## 9. EQA Extensions

New tools for the Experience Query API to support Graft.

### 9.1 experience.get_concept_region

```jsonc
{
  "name": "experience.get_concept_region",
  "description": "Retrieve or derive a ConceptRegion for a concept.",
  "input_schema": {
    "type": "object",
    "properties": {
      "concept_id": { "type": "string" },
      "derive_if_missing": { "type": "boolean", "default": true },
      "derivation_config": {
        "type": "object",
        "properties": {
          "method": { "type": "string", "enum": ["lens_weight_topk", "gradient_attribution"] },
          "top_k_percent": { "type": "number" },
          "layers": { "type": "array", "items": { "type": "integer" } },
          "include_ancestors": { "type": "boolean" }
        }
      }
    },
    "required": ["concept_id"]
  }
}
```

### 9.2 experience.check_graft_overlap

```jsonc
{
  "name": "experience.check_graft_overlap",
  "description": "Check if a graft overlaps with existing grafts via fingerprint similarity.",
  "input_schema": {
    "type": "object",
    "properties": {
      "graft_id": { "type": "string" },
      "fingerprint_threshold": { "type": "number", "default": 0.6 },
      "region_threshold": { "type": "number", "default": 0.25 },
      "scope": {
        "type": "string",
        "enum": ["local", "tribe", "all"],
        "default": "local"
      }
    },
    "required": ["graft_id"]
  }
}
```

### 9.3 experience.request_cotrain

```jsonc
{
  "name": "experience.request_cotrain",
  "description": "Request cotraining for overlapping concepts.",
  "input_schema": {
    "type": "object",
    "properties": {
      "concept_ids": {
        "type": "array",
        "items": { "type": "string" },
        "minItems": 2
      },
      "priority": { "type": "string", "enum": ["low", "normal", "high"] },
      "reason": { "type": "string" }
    },
    "required": ["concept_ids"]
  }
}
```

### 9.4 experience.get_substrate_manifest

```jsonc
{
  "name": "experience.get_substrate_manifest",
  "description": "Retrieve the current substrate manifest showing all grafted dimensions.",
  "input_schema": {
    "type": "object",
    "properties": {
      "include_compaction_candidates": { "type": "boolean", "default": false },
      "include_dimension_stats": { "type": "boolean", "default": false }
    }
  }
}
```

### 9.5 experience.analyze_compaction

```jsonc
{
  "name": "experience.analyze_compaction",
  "description": "Analyze substrate for compaction opportunities (correlated or low-usage dimensions).",
  "input_schema": {
    "type": "object",
    "properties": {
      "correlation_threshold": { "type": "number", "default": 0.9 },
      "usage_threshold": { "type": "number", "default": 0.01 },
      "episode_window": { "type": "integer", "default": 1000 }
    }
  }
}
```

---

## 10. Validation Requirements

### 10.1 Graft Validation Suite

Before a graft can be applied or shared:

```jsonc
GraftValidation = {
  "validation_id": "valrun-Fish-v1",
  "graft_id": "graft-Fish-v1",

  "tests": [
    {
      "name": "dimension_activation",
      "description": "New dimension activates on concept-positive examples",
      "dataset": "dataset-Fish-v1-holdout",
      "metric": "activation_auc",
      "threshold": 0.85,
      "result": 0.91,
      "passed": true
    },
    {
      "name": "lens_f1",
      "description": "Lens correctly detects concept using primary dimension",
      "dataset": "dataset-Fish-v1-holdout",
      "metric": "f1",
      "threshold": 0.85,
      "result": 0.89,
      "passed": true
    },
    {
      "name": "null_false_positive",
      "description": "Lens does not fire on unrelated content",
      "dataset": "dataset-null-random-v1",
      "metric": "false_positive_rate",
      "threshold": 0.05,
      "result": 0.03,
      "passed": true
    },
    {
      "name": "ood_degradation",
      "description": "Behaviour on unrelated tasks not significantly degraded",
      "dataset": "dataset-ood-benchmark-v1",
      "metric": "perplexity_delta",
      "threshold": 0.1,
      "result": 0.02,
      "passed": true
    },
    {
      "name": "bias_sparsity",
      "description": "Substrate biases are sufficiently sparse",
      "metric": "sparsity",
      "threshold": 0.90,
      "result": 0.94,
      "passed": true
    },
    {
      "name": "bias_magnitude",
      "description": "Bias magnitudes are bounded",
      "metric": "max_bias_magnitude",
      "threshold": 0.1,
      "result": 0.089,
      "passed": true
    },
    {
      "name": "dimension_primary_contribution",
      "description": "Primary dimension contributes meaningfully to lens",
      "metric": "primary_dim_weight_ratio",
      "threshold": 0.5,
      "result": 0.73,
      "passed": true
    }
  ],

  "overall_passed": true,
  "validated_at": "2025-11-30T13:00:00Z",
  "validated_by": "BE-instance-123"
}
```

### 10.2 Protection Level Assessment

Per HATCAT_MELD_POLICY, grafts inherit protection levels:

```python
def assess_graft_protection(graft: Graft) -> ProtectionAssessment:
    """
    Determine protection level for a graft based on what it modifies.
    """

    max_level = "standard"
    triggers = []

    # Check if concept touches critical simplexes
    concept = load_concept(graft.concept_id)

    for simplex in CRITICAL_SIMPLEX_REGISTRY:
        if concept.concept_id in simplex.bound_concepts:
            max_level = "critical"
            triggers.append({
                "type": "bound_to_critical_simplex",
                "simplex": simplex.term,
                "concept": concept.term
            })

    # Check relational fingerprint for critical correlations
    for corr in graft.relational_fingerprint.top_correlations:
        corr_concept = load_concept(corr.concept_id)
        if corr_concept.safety_tags.risk_level in ["high", "critical"]:
            max_level = max(max_level, "protected")
            triggers.append({
                "type": "high_correlation_with_sensitive_concept",
                "concept": corr.concept_id,
                "correlation": corr.correlation
            })

    # Check bias density and magnitude
    if graft.bias_density > 0.3:  # More than 30% of region modified
        max_level = max(max_level, "elevated")
        triggers.append({
            "type": "high_bias_density",
            "density": graft.bias_density
        })

    # Check concept safety tags
    if concept.safety_tags:
        risk = concept.safety_tags.get("risk_level", "none")
        if risk == "critical":
            max_level = "critical"
        elif risk == "high":
            max_level = max(max_level, "protected")
        elif risk == "medium":
            max_level = max(max_level, "elevated")

    return ProtectionAssessment(
        protection_level=max_level,
        triggers=triggers,
        requires_review=max_level in ["protected", "critical"]
    )
```

### 10.3 CAT Oversight Triggers

Grafts trigger CAT (Conjoined Adversarial Tomograph) updates based on their protection level. CATs must be notified when substrate capabilities change through grafting.

| Protection Level | CAT Impact |
|-----------------|------------|
| STANDARD | Ambient CAT updated at next scheduled sync |
| ELEVATED | Ambient CAT capability check required before graft activation |
| PROTECTED | Escalation CAT review; lens coverage verification |
| CRITICAL | Macro-CAT assessment required; graft blocked until CAT confirms interpretability |

#### CAT-Graft Coordination

When a graft adds new concept dimensions:

1. **Lens Integration**: CAT must verify it can interpret lenses reading from the new dimension
2. **Translation Updates**: If the graft introduces concepts outside the CAT's supported `ConceptPackSpecID`s, a `TranslationMapping` must be provided or the CAT capability is degraded
3. **Bias Monitoring**: CAT windows should include the grafted dimension's lens outputs to detect drift or unexpected correlations
4. **Overlap Analysis**: If `OverlapAnalysis` detects high correlation with existing concepts, CAT should be notified to update its tomography baseline

```jsonc
GraftCATNotification = {
  "graft_id": "graft-Eligibility-v1",
  "concept_id": "concept/Eligibility",
  "protection_level": "elevated",
  "dimension_index": 2048,
  "cat_actions_required": [
    {
      "action": "capability_check",
      "reason": "New dimension added to substrate",
      "blocking": true
    },
    {
      "action": "update_lens_coverage",
      "reason": "Lens org.hatcat/custom-v1::concept/Eligibility now active",
      "blocking": false
    }
  ],
  "overlap_alerts": [
    {
      "correlated_concept": "concept/QualificationStatus",
      "correlation": 0.78,
      "action": "update_tomography_baseline"
    }
  ]
}
```

See `HAT/HAT_CONJOINED_ADVERSARIAL_TOMOGRAPHY.md` for full CAT specification.

---

## 11. Implementation Reference

The grafting system is implemented in `src/grafting/` using botanical terminology that maps to the conceptual model:

### 11.1 Terminology Mapping

| Conceptual Term | Implementation Term | Description |
|-----------------|---------------------|-------------|
| ConceptRegion | **Cleft** | Region of weights associated with a concept (from lens analysis) |
| Graft (trained) | **Scion** | Hard/permanent graft with trained weight modifications |
| Graft (temporary) | **Bud** | Soft/reversible graft using forward hooks |
| Substrate | **Model** | The base language model being modified |
| Dimension expansion | **Expand mode** | Adding new dimensions to hidden_size |
| Bias application | **Delta mode** | Modifying existing weights without expansion |

### 11.2 Core Components

#### Cleft (`src/grafting/cleft.py`)

Derives concept regions from trained lenses:

```python
from src.map.graft import derive_cleft_from_lens, merge_clefts, Cleft

# Derive cleft from a trained lens
cleft = derive_cleft_from_lens(
    lens_path="lens_packs/v4/Fish.pt",
    concept_id="Fish",
    model=model,
    layers=[18, 20, 22],
    top_k_percent=15.0
)

# Merge multiple clefts for cotraining
union_cleft = merge_clefts([fish_cleft, animal_cleft, vertebrate_cleft])
```

The `Cleft` contains:
- `regions: List[CleftRegion]` — per-layer neuron masks identifying important dimensions
- `hidden_dim: int` — the substrate's hidden dimension
- Methods for freezing non-cleft regions during training

#### Scion (`src/grafting/scion.py`)

A permanent graft with trained weight deltas:

```python
from src.map.graft import Scion, ScionTrainer, ScionConfig, apply_scion

# Configure training
config = ScionConfig(
    learning_rate=1e-4,
    epochs=3,
    batch_size=8,
    injection_layers=[18, 20, 22]
)

# Train a scion
trainer = ScionTrainer(model, tokenizer, union_cleft, config)
scion = trainer.train(
    dataset={"positive": [...], "negative": [...]},
    concept_id="Fish",
    verbose=True
)

# Apply to substrate (two modes)
apply_scion(model, scion, mode="delta")   # Modify existing weights
apply_scion(model, scion, mode="expand")  # Add new dimension (hidden_dim + 1)
```

The `Scion` contains:
- `neuron_biases: Dict[str, Tensor]` — learned weight modifications
- `training_config: ScionConfig` — training hyperparameters
- `metrics: Dict[str, float]` — training/validation metrics

#### Bud (`src/grafting/bud.py`)

A temporary/reversible graft using forward hooks:

```python
from src.map.graft import Bud, BuddedModel

# Create bud from scion (for testing before permanent application)
bud = Bud.from_scion(scion, layers=[18, 20, 22])

# Or from a steering direction
bud = Bud.from_direction(
    concept_id="Fish",
    direction=direction_vector,  # shape: [hidden_dim]
    layers=[18, 20, 22],
    strength=1.0
)

# Apply via BuddedModel wrapper
budded = BuddedModel(model, tokenizer)
budded.add_bud(bud)
budded.activate_bud(bud.bud_id, strength=0.8)

# Generate with bud active
output = budded.generate("Tell me about...")

# Context manager for temporary activation
with budded.bud_context([bud.bud_id], strengths=[1.0]):
    output = budded.generate("...")
```

#### Expand Mode (`src/grafting/expand.py`)

Architecture-aware dimension expansion:

```python
from src.map.graft import (
    detect_architecture,
    plan_expansion,
    execute_expansion,
    ArchitectureSpec
)

# Detect model architecture
arch = detect_architecture(model)
# Returns: ArchitectureSpec(family="llama", hidden_size=4096, ...)

# Plan which matrices need expansion
plan = plan_expansion(model, arch)
# Returns: ExpansionPlan with targets like:
#   - embed_tokens: (vocab, hidden) → (vocab, hidden+1)
#   - q_proj: (hidden, hidden) → (hidden+1, hidden+1)
#   - down_proj: (hidden, intermediate) → (hidden+1, intermediate)

# Execute expansion
execute_expansion(model, plan, scion)
# Model now has hidden_dim + 1
```

Supported architectures:
- Llama family (Llama, Apertus, Mistral)
- Gemma family (Gemma, Gemma2)
- GPT-2 family
- MoE architectures (Mixtral, DeepSeek)

### 11.3 XDB Integration (`src/xdb/budding.py`)

The `BuddingManager` bridges XDB experiences to grafting:

```python
from src.be.xdb import XDB, BuddingManager

# Initialize
xdb = XDB(Path("./xdb_data"), be_id="my-be")
budding = BuddingManager(xdb, lens_pack_path=Path("./lens_packs/v4"))

# 1. Tag experiences during normal operation
bud_tag = xdb.create_bud_tag("curiosity-fish", "Interesting fish discussions")
xdb.tag(bud_tag.id, tick_range=(100, 500))  # Tag relevant experiences

# 2. When ready, extract training data
training_data = budding.get_training_data(
    bud_tag.id,
    min_activation=0.7,
    max_positive=500,
    max_negative=500,
    negative_strategy="low_activation"  # or "sibling_concepts"
)

# 3. Prepare training run (pins data as WARM fidelity)
run = budding.prepare_scion_training(bud_tag.id)

# 4. Execute training
scion = budding.run_scion_training(
    run.id,
    model,
    tokenizer,
    layers=[18, 20, 22]
)

# 5. Or use convenience method for full pipeline
scion = budding.promote_bud(bud_tag.id, model, tokenizer)

# 6. Submit for tribal review (locks evidence as SUBMITTED)
submission_id = budding.submit_for_review(scion.scion_id, run.id)
```

### 11.4 Data Structures (`src/grafting/data_structures.py`)

Key structures for substrate tracking:

```python
from src.map.graft import (
    SubstrateArchitecture,
    SubstrateManifest,
    DimensionEntry,
    GraftConfig
)

# Create architecture from model config
arch = SubstrateArchitecture.from_model_config(model.config)

# Create manifest for tracking grafts
manifest = SubstrateManifest.create_for_model(
    model_id="swiss-ai/Apertus-8B-2509",
    hidden_dim=4096,
    architecture=arch
)

# After applying grafts
manifest.add_graft(scion)
# manifest.current_hidden_dim is now 4097
# manifest.dimension_table contains the new entry
```

### 11.5 The Accretion Loop

The complete flow from experience to permanent learning:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ACCRETION LOOP                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. EXPERIENCE RECORDING (XDB)                                       │
│     └─ xdb.record() stores timesteps with concept_activations        │
│                                                                      │
│  2. BUD IDENTIFICATION (BE decision)                                 │
│     └─ xdb.create_bud_tag() + xdb.tag() marks interesting patterns   │
│                                                                      │
│  3. TRAINING DATA EXTRACTION (BuddingManager)                        │
│     └─ budding.get_training_data() queries XDB for examples          │
│     └─ Positive: high activation, tagged by bud                      │
│     └─ Negative: low activation or sibling concepts                  │
│                                                                      │
│  4. CLEFT DERIVATION (grafting)                                      │
│     └─ derive_cleft_from_lens() analyzes lens weights              │
│     └─ merge_clefts() creates union for related concepts             │
│                                                                      │
│  5. SCION TRAINING (grafting)                                        │
│     └─ ScionTrainer.train() with cleft-aware freezing                │
│     └─ Learns weight biases for concept                              │
│                                                                      │
│  6. TESTING AS BUD (optional)                                        │
│     └─ Bud.from_scion() creates reversible test version              │
│     └─ Validate behavior before committing                           │
│                                                                      │
│  7. SCION APPLICATION (grafting)                                     │
│     └─ apply_scion(model, scion, mode="expand")                      │
│     └─ SubstrateManifest.add_graft() records the change              │
│                                                                      │
│  8. EVIDENCE SUBMISSION (XDB → MELD)                                 │
│     └─ budding.submit_for_review() locks evidence                    │
│     └─ GraftDiff shared via MAP for tribal review                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.6 Module Structure

```
src/grafting/
├── __init__.py          # Public API exports
├── cleft.py             # Cleft derivation from lenses
├── scion.py             # Scion training and application
├── bud.py               # Bud (temporary) graft via hooks
├── expand.py            # Architecture-aware dimension expansion
└── data_structures.py   # SubstrateManifest, SubstrateArchitecture, etc.

src/xdb/
├── __init__.py          # XDB public API
├── xdb.py               # Main XDB class
├── models.py            # TimestepRecord, Tag, BudStatus, etc.
├── experience_log.py    # Experience storage
├── tag_index.py         # Folksonomy management
├── storage_manager.py   # Fidelity tiers
└── budding.py           # XDB → Grafting bridge (BuddingManager)
```

---

## 12. Summary

The Graft Protocol provides a mechanism for **substrate growth through learned concepts**:

1. **Dimension allocation** — each concept gets a labelled dimension in the substrate
2. **Bias encoding** — relational structure is captured in sparse weight biases
3. **Lens binding** — lenses read from the primary dimension plus auxiliary context
4. **Fingerprint comparison** — bias patterns enable overlap detection across BEs
5. **Cotraining** — overlapping concepts are trained together from pooled exemplars
6. **Dimension management** — pruning, merging, and maintenance as substrates grow

The key properties:

- **Small trunks can scale** — start small, grow through grafting
- **Concepts are labelled** — explicit features, not entangled representations
- **Biases are evidence** — the relational structure is in the parameters
- **Exemplars are truth** — grafts can be regenerated from Experience Database
- **Sharing is principled** — fingerprints enable safe integration across BEs

The stack becomes:

| Layer | Role |
|-------|------|
| **MAP** | Concept identity, lens registration |
| **MELD** | Governance, protection levels |
| **Graft** | Concept dimensions + substrate biases |
| **BE** | Bounded Experiencer |
| **Hush** | Safety harnesses |
| **ASK** | Treaties and inter-BE coordination |

This enables BE instances to:

- Learn concepts from experience
- Grow their substrates incrementally
- Share learned concepts with lenses and grafts
- Safely integrate incoming concepts via fingerprint comparison
- Maintain reproducibility through full provenance in XDB
