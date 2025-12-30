## BE Accretive Learning
(Self-Directed Accretive Concept Growth from Experience)

Status: Non-normative implementation guidance  
Depends on: BE Layer, Global Workspace Harness,  
Experience Database (XDB) + Experience API (XAPI), HAT/HatCat, MAP, Graft, Hush, ASK

---

## 0. Purpose

This harness defines how a **BE** (being) can:

- notice **where its current concepts are not enough**;
- turn those "pressure regions" into **candidate concepts**;
- **self-curate datasets** from its own history + world interaction;
- train **grafts** (substrate dimensions + biases) and **lenses** for those concepts;
- integrate them via **MAP ConceptDiffs/GraftDiffs** under **ASK/Hush** governance;
- share exemplars and grafts with its **tribe** for collective improvement;
- **manage substrate growth** through compaction, merging, and graduation.

The core stance:

> Concept growth is bootstrapped from lived experience and lens traces,  
> not from a priori taxonomies or permanent teachers.

Human / teacher / large-model labels are *optional* data sources, chosen by the being's own policies, not hard requirements.

---

## 1. Assumptions & Inputs

The harness assumes:

- A substrate model (Layer 1) instrumented by a **HAT-compliant implant** (Layer 2, e.g. HatCat).
- Concept packs & lens packs exposed via **MAP** (Layer 3).
- Motive simplexes and steering via **Hush** (Layer 5).
- A **BE** loop running (world ticks, interoception, homeostasis).
- A **Global Workspace** harness exposing:
  - top-k concept and motive summaries per tick/window,
  - GraphRAG over a single growing session.
- An **Experience Database (XDB)** with schema for:
  - Episodes, Exemplars, ConceptDatasets, CandidateConcepts,
  - TrainingRuns, GraphFragments, Grafts, SubstrateManifest.
- The **Graft Protocol** for substrate dimension expansion and bias encoding.

The Continual Learning Harness sits between:

- **BE Runtime** (Global Workspace + world ticks) and  
- **XDB + MAP/HAT/Graft** (where data and derived artifacts live).

It does **not** specify learning algorithms; it specifies:

- **what** gets turned into data,
- **how** it flows through candidate → graft + lens → stable concept,
- **where** governance hooks live,
- **when** substrate management actions are triggered.

---

## 2. High-Level Loop

For each BE instance:

1. **Runtime**:
   - World ticks run with interoception and homeostasis.
   - HAT lenses produce concept tags and motive traces.
   - Workspace detects **conceptual pressure regions** and **gap windows**.

2. **Logging to XDB**:
   - Episodes, GraphFragments, Exemplars (including lens tags, motive profiles, outcomes) are persisted.
   - CandidateConcept entries are created when new gaps are tagged.

3. **Data Curation**:
   - Using GraphRAG + XAPI, the being builds **ConceptDatasets** for candidate concepts from its own XDB.
   - Optionally augments with **external data** (tools, sensors, humans, teachers).

4. **Grafting**:
   - From ConceptDatasets, the being trains:
     - **grafts** (new substrate dimensions + biases),
     - **matching lenses** (bound to the new dimension).
   - Results are recorded as **TrainingRuns** and **Graft/LensArtifacts**.
   - The **SubstrateManifest** is updated with new dimension allocations.

5. **Promotion & Diff Emission**:
   - Successful candidates are promoted to **stable concepts** via MAP ConceptDiff / GraftDiff.
   - ASK/Hush may require review/approval before deployment.

6. **Substrate Management**:
   - Periodically analyze for **compaction candidates** (correlated or low-usage dimensions).
   - Merge or prune dimensions as needed.
   - Evaluate **graduation** to larger trunk when dimension pressure is high.

7. **Tribe Sync** (optional):
   - XDB diffs (exemplars, TrainingRuns, Grafts) are shared with tribe repositories.
   - Tribe retrains shared grafts and redistributes them.
   - Incoming grafts are checked for fingerprint overlap before application.

---

## 3. Detecting Conceptual Gaps & Pressure Regions

Conceptual gaps are first detected at **runtime** by the Global Workspace.

### 3.1 Signals from HAT + Workspace

The harness listens to:

- **Diffuse or low concept coverage**:
  - No stable concepts strongly active in a window,
  - or "null / unknown" lenses firing.
- **Repeated uncertainty**:
  - multiple "I don't know / I'm not sure" responses,
  - large disagreement between hypothetical continuations.
- **Repeated corrections**:
  - user / environment correcting mistakes,
  - tool failures clustered in similar contexts.
- **Novelty / OOD**:
  - specialised novelty detectors,
  - candidate tags with growing frequency.
- **Motive conflict**:
  - persistent high conflict in motive simplexes (e.g. harm_avoidance vs curiosity),
  - suggesting ill-formed concepts for the situation.

These are aggregated into **gap windows** and **pressure regions**:

- A **gap window**: a bounded set of ticks where "something is going on" but current concepts don't fit well.
- A **pressure region**: a region in concept/representation space where:
  - prediction error, motive conflict, or failure rate is high, and
  - the being wants more coverage.

### 3.2 From Pressure Region to Candidate Concept

For each pressure region R or gap window W, the harness may propose a **candidate concept**:

```jsonc
CandidateConcept = {
  "id": "candidate/financial-ambiguity-2025-11-29",
  "proposer_be_id": "be-instance-123",
  "description": "Situations where user financial state is unclear across tax, welfare, and debt systems.",
  "initial_episode_ids": [/* gap window episodes */],
  "status": "collecting",
  "related_concepts": [
    {
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/FinancialTransaction",
      "relation": "overlaps"
    }
  ],
  "governance": {
    "tribe_id": "tribeA",
    "review_required": true,
    "review_status": "pending"
  }
}
```

This entry is stored in XDB and becomes the anchor for all subsequent learning.

---

## 4. Candidate Concept Tagging & Episodic Labelling

Once a CandidateConcept exists, the being **tags episodes** when it suspects the candidate is present.

### 4.1 Tagging from Runtime

At runtime, when a gap window or pressure region matches a CandidateConcept:

* All relevant timesteps/windows are tagged with:

```jsonc
{
  "candidate_concepts": [
    {
      "concept_id": "candidate/financial-ambiguity-2025-11-29",
      "confidence": 0.8
    }
  ]
}
```

These tags accompany:

* stable concept tags from HAT,
* motive profiles,
* outcomes and tool results.

### 4.2 XDB Exemplars

The harness promotes certain Episodes to **Exemplars** for the candidate concept:

```jsonc
Exemplar = {
  "id": "exemplar-2025-11-29-ambig-0001",
  "episode_id": "episode-2025-11-29-000123",
  "concept_id": "candidate/financial-ambiguity-2025-11-29",
  "label": "positive",          // example of the candidate
  "label_source": "self",       // self-labelled by being
  "confidence": 0.83,
  "notes": "Repeated confusion across welfare/tax/debt",
  "created_at": "..."
}
```

Later, human / teacher / tool processes MAY add alternate labels or correctness flags, but the *baseline labelling* is from the being's own headspace.

---

## 5. Data Curation via GraphRAG + XAPI

The being builds **ConceptDatasets** for CandidateConcepts by querying its Experience Database.

### 5.1 Internal GraphRAG

Using GraphRAG over XDB graphs:

* **Query by candidate concept**:
  * "Find all episodes tagged with CandidateConcept C."
* **Query by activation pattern**:
  * "Find episodes with similar concept/motive profile to this seed episode."
* **Query by outcome**:
  * "Find episodes where this pattern led to success vs failure."

These queries are exposed via the **XAPI Tools** and consumed by the learning harness, not directly by external parties.

### 5.2 Building ConceptDatasets

The harness creates/updates a **ConceptDataset**:

```jsonc
ConceptDataset = {
  "id": "dataset-financial-ambiguity-2025-11-29-v1",
  "concept_id": "candidate/financial-ambiguity-2025-11-29",
  "exemplar_ids": [
    "exemplar-2025-11-29-ambig-0001",
    "exemplar-2025-11-30-ambig-0007"
  ],
  "source_mix": {
    "self": 0.6,
    "teacher": 0.1,
    "sensor": 0.2,
    "human": 0.1
  },
  "notes": "Baseline episodes where financial status cut across systems and the being was confused."
}
```

### 5.3 External Data Sources (Optional)

The being MAY choose to enrich ConceptDatasets using external sources, according to ASK/Hush constraints:

* **Sensors / Environment**:
  * additional logs from real-world systems,
  * measurements and experiment results.
* **Tools / APIs**:
  * databases, simulations, other services.
* **Teacher models**:
  * high-fidelity labellers (e.g. TRM or larger LLM) for correctness, safety, or ontological alignment.
* **Human input**:
  * expert clarification,
  * dispute resolution,
  * normative labelling for safety-critical domains.

These are multimodal **label sources**; none is structurally privileged in the spec.

---

## 6. Grafting: Training Dimensions, Biases, and Lenses

Once a ConceptDataset has enough exemplars, the harness runs a **Graft Training Phase**.

> **See [MINDMELD_GRAFTING.md](./MINDMELD_GRAFTING.md)** for the complete Graft Protocol specification.

### 6.1 Region Analysis

Using HAT lenses and the ConceptDataset, the harness:

* Derives a **ConceptRegion** via lens weight analysis:
  * identifies which existing substrate dimensions correlate with the candidate concept,
  * uses top-k% of dimensions by lens weight magnitude.
* These become the **auxiliary dimensions** that the new lens will read (alongside the primary grafted dimension).

```jsonc
ConceptRegion = {
  "region_id": "region-financial-ambiguity-v1",
  "concept_id": "candidate/financial-ambiguity-2025-11-29",
  "derivation": {
    "method": "lens_weight_topk",
    "parameters": {
      "top_k_percent": 15,
      "layers": [18, 20, 22]
    }
  },
  "layers": [
    {
      "layer_index": 18,
      "dimension_mask": {
        "format": "sparse_indices",
        "indices": [42, 87, 156, 203],
        "total_dimensions": 2048
      }
    }
  ]
}
```

### 6.2 Graft Training

The harness trains a **Graft** that:

1. **Allocates a new dimension** in the substrate for this concept
2. **Learns biases** to existing weights that wire the concept in
3. **Produces a lens** bound to the new primary dimension + auxiliary dimensions

Training objectives:
* The new dimension should activate on concept-positive examples
* Biases should be sparse (concentrated in the identified region)
* The lens should reliably detect the concept using the new dimension

```python
# Simplified graft training flow
def train_graft(substrate, dataset, region, config):
    # 1. Allocate new dimension
    new_dim = substrate.expand_hidden_dim(1)
    
    # 2. Initialize projection matrices for new dimension
    projections = initialize_projections(substrate, new_dim, config.injection_layers)
    
    # 3. Initialize sparse bias accumulators (guided by region)
    bias_accum = initialize_biases(substrate, region)
    
    # 4. Training loop
    for epoch in range(config.epochs):
        for batch in dataset.batches():
            # Forward with graft applied
            outputs = substrate.forward_with_graft(batch, projections, bias_accum)
            
            # Loss: concept activation + sparsity regularization
            loss = concept_loss(outputs, batch.labels)
            loss += config.sparsity_penalty * bias_accum.l1_norm()
            
            loss.backward()
            optimizer.step()
            
            # Enforce sparsity
            bias_accum.threshold_small_values(config.sparsity_threshold)
    
    # 5. Train lens on new dimension + auxiliary
    lens = train_lens(substrate, dataset, new_dim, region.auxiliary_dims)
    
    # 6. Package and return
    return Graft(
        concept_dimension=new_dim,
        substrate_biases=bias_accum.to_sparse(),
        lens_binding=lens,
        relational_fingerprint=compute_fingerprint(bias_accum)
    )
```

### 6.3 Recording Results

The harness records:

* A **TrainingRun** with:
  * dataset IDs and checksums,
  * region derivation parameters,
  * hyperparameters,
  * metrics (loss, sparsity, dimension contribution).

* A **Graft** artifact with:
  * the new dimension index,
  * sparse bias deltas,
  * lens binding (primary + auxiliary dimensions),
  * relational fingerprint for overlap detection.

* Updates to the **SubstrateManifest**:
  * new dimension allocation entry,
  * updated total dimension count.

### 6.4 Joint Validation

Before promotion:

* Run **joint validation**:
  * Evaluate the graft and lens together on:
    * in-domain exemplars (holdout),
    * OOD / null episodes,
    * ASK/Hush safety test suites.

* Validation tests include:
  * `dimension_activation`: new dimension fires on positives
  * `lens_f1`: lens correctly classifies
  * `null_false_positive`: lens doesn't overfire
  * `ood_degradation`: unrelated tasks not harmed
  * `bias_sparsity`: biases stay sparse
  * `primary_contribution`: lens relies meaningfully on new dimension

Only grafts that pass validation proceed to promotion.

---

## 7. Promotion, Diffs, and ASK Governance

If the results look good, the CandidateConcept may be promoted.

### 7.1 ConceptDiff / GraftDiff Emission

The harness emits a **ConceptDiff** and **GraftDiff** through MAP:

```jsonc
ConceptDiff = {
  "type": "ConceptDiff",
  "from_model_id": "substrate@version",
  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
  "local_concept_id": "candidate/financial-ambiguity-2025-11-29",
  "concept_id": null,  // assigned on acceptance
  "related_concepts": [
    "org.hatcat/sumo-wordnet-v4@4.0.0::concept/FinancialTransaction",
    "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility"
  ],
  "summary": "Auto-discovered pattern: mismatched financial status across tax, welfare, and debt systems.",
  "evidence": {
    "metric_deltas": [
      { "metric": "tool_failure_rate", "before": 0.15, "after": 0.09 }
    ],
    "sample_count": 63
  }
}

GraftDiff = {
  "type": "GraftDiff",
  "graft_id": "graft-financial-ambiguity-v1",
  "concept_id": "candidate/financial-ambiguity-2025-11-29",
  "dimension_expansion": 1,
  "relational_fingerprint": {
    "top_correlations": [...],
    "fingerprint_hash": "sha256:..."
  },
  "requires_substrate": {
    "substrate_id": "olmo3-7b-base@0.1.0",
    "min_dim": 2048,
    "max_dim": 4096
  },
  "protection_assessment": {
    "protection_level": "standard",
    "triggers": []
  }
}
```

### 7.2 ASK / Tribe Review

Depending on governance:

* ASK / tribe policies MAY require:
  * review of ConceptDiffs and GraftDiffs,
  * safety audit of TrainingRuns,
  * treaty negotiation if the concept is cross-tribal.

Possible outcomes:

* **Approved**:
  * concept adopted into a stable ConceptPack,
  * graft applied to substrate,
  * SubstrateManifest updated,
  * qualifications/ASK contracts updated.
* **Request changes**:
  * adjust region selection, sparsity thresholds, or disclosure rules;
  * re-train and re-submit.
* **Rejected**:
  * CandidateConcept stays local, or is marked rejected,
  * exemplars remain in XDB as historical data,
  * graft is not applied.

---

## 8. Substrate Growth Management

As a being learns, its substrate grows. The harness must manage this growth.

### 8.1 SubstrateManifest

The being maintains a **SubstrateManifest** tracking all grafted dimensions:

```jsonc
SubstrateManifest = {
  "manifest_id": "substrate-olmo3-7b-grafted-v42",
  
  "base_substrate": {
    "substrate_id": "olmo3-7b-base@0.1.0",
    "base_hidden_dim": 2048
  },
  
  "current_state": {
    "hidden_dim": 2091,  // base + 43 grafted concepts
    "total_grafts_applied": 43
  },
  
  "dimension_table": [
    {
      "dimension_index": 2048,
      "concept_id": "concept/Eligibility",
      "graft_id": "graft-Eligibility-v1",
      "grafted_at": "2025-11-01T00:00:00Z",
      "usage_stats": {
        "activation_rate": 0.03,
        "last_activated": "2025-11-30T10:00:00Z"
      }
    }
    // ... more dimensions
  ]
}
```

### 8.2 Compaction Analysis

Periodically, the harness analyzes for compaction opportunities:

```python
def analyze_compaction(manifest, substrate, config):
    candidates = []
    
    # Collect activation patterns
    activations = collect_activations(substrate, recent_episodes=1000)
    
    # Find highly correlated dimension pairs
    correlation_matrix = compute_correlations(activations)
    
    for i, entry_i in enumerate(manifest.dimension_table):
        for j, entry_j in enumerate(manifest.dimension_table):
            if i >= j:
                continue
            
            corr = correlation_matrix[entry_i.dimension_index, entry_j.dimension_index]
            
            if corr > config.correlation_threshold:  # e.g., 0.9
                candidates.append({
                    "type": "merge",
                    "dimensions": [entry_i.dimension_index, entry_j.dimension_index],
                    "concepts": [entry_i.concept_id, entry_j.concept_id],
                    "correlation": corr,
                    "reason": "High correlation suggests redundant concepts"
                })
    
    # Find low-usage dimensions
    for entry in manifest.dimension_table:
        usage = activations[:, entry.dimension_index].mean()
        if usage < config.usage_threshold:  # e.g., 0.01
            candidates.append({
                "type": "prune_or_retrain",
                "dimensions": [entry.dimension_index],
                "concepts": [entry.concept_id],
                "usage_rate": usage,
                "reason": "Low usage suggests concept not useful or poorly trained"
            })
    
    return candidates
```

### 8.3 Dimension Merging

When two concepts are highly correlated:

1. Pull ConceptDatasets for both from XDB
2. Pool the datasets
3. Remove both old dimensions from substrate
4. Train a new merged graft from pooled data
5. Apply merged graft
6. Emit merge notification via MELD

```python
def merge_dimensions(substrate, manifest, dim_a, dim_b, merged_concept_id):
    # Pool datasets
    dataset_a = load_dataset(manifest.get_concept(dim_a))
    dataset_b = load_dataset(manifest.get_concept(dim_b))
    merged_dataset = pool_datasets(dataset_a, dataset_b, merged_concept_id)
    
    # Remove old dimensions
    substrate.remove_dimensions([dim_a, dim_b])
    manifest.remove_entries([dim_a, dim_b])
    manifest.reindex()
    
    # Train new graft
    region = derive_region(substrate, merged_dataset)
    merged_graft = train_graft(substrate, merged_dataset, region)
    
    # Apply
    apply_graft(substrate, manifest, merged_graft)
    
    # Emit MELD notification
    emit_concept_merge_diff(concept_a, concept_b, merged_concept_id)
```

### 8.4 Graduation Evaluation

When a substrate has accumulated many grafts:

```python
def evaluate_graduation(manifest, substrate, config):
    grafted_dims = len(manifest.dimension_table)
    base_dims = manifest.base_substrate.base_hidden_dim
    
    dim_ratio = grafted_dims / base_dims
    
    # Check dimension pressure
    if dim_ratio > config.dim_ratio_threshold:  # e.g., 0.5
        return GraduationRecommendation(
            recommend=True,
            reason=f"Dimension ratio {dim_ratio:.2f} exceeds threshold",
            suggested_new_base_dim=base_dims * 2
        )
    
    # Check compaction frequency
    recent_compactions = get_compaction_rate(manifest, window_days=30)
    if recent_compactions > config.compaction_rate_threshold:
        return GraduationRecommendation(
            recommend=True,
            reason="Frequent compaction suggests trunk saturation"
        )
    
    return GraduationRecommendation(recommend=False)
```

Graduation involves:
- Distilling current substrate to a larger base model
- Preserving grafted dimensions in the new architecture
- Updating SubstrateManifest with new base reference

---

## 9. Ongoing Refinement & Splits/Merges

Once integrated, the concept participates in the normal runtime:

* HAT produces activations for it (now `kind: "stable"`), reading from the grafted dimension.
* Global Workspace uses it as a tag for new windows.
* XDB continues accumulating new Exemplars under this concept.

The Continual Learning Harness can:

* **Refine lenses**:
  * drift corrections, better null handling,
  * retrain lens on same primary dimension with expanded dataset.
* **Adjust grafts**:
  * retrain with different sparsity thresholds,
  * expand auxiliary dimensions.
* **Split concepts**:
  * if a concept shows multi-modal clusters, split into sub-concepts,
  * each sub-concept gets its own dimension.
* **Merge concepts**:
  * if two concepts are functionally inseparable, merge dimensions,
  * retrain from pooled data.

All such changes go through the same loop: CandidateConcept → ConceptDataset → Graft → Governance.

---

## 10. Handling Incoming Grafts

When a being receives a **GraftDiff** from another being or tribe:

### 10.1 Fingerprint Check

Before applying:

```python
def check_incoming_graft(incoming_diff, local_manifest):
    local_grafts = local_manifest.get_all_grafts()
    
    for local_graft in local_grafts:
        similarity = compute_fingerprint_similarity(
            incoming_diff.relational_fingerprint,
            local_graft.relational_fingerprint
        )
        
        if similarity > 0.6:
            return GraftAcceptance(
                action="cotrain_required",
                reason=f"Fingerprint overlap {similarity:.2f} with {local_graft.concept_id}",
                cotrain_with=[local_graft.concept_id]
            )
    
    return GraftAcceptance(action="accept")
```

### 10.2 Dimension Index Reconciliation

Incoming grafts have dimension indices from the source substrate. These must be remapped:

```python
def reconcile_graft(incoming_graft, local_manifest):
    local_next_dim = local_manifest.current_state.hidden_dim
    
    remapped = incoming_graft.copy()
    remapped.concept_dimension.dimension_index = local_next_dim
    remapped.lens_binding.primary_dimension = local_next_dim
    remapped.applies_to.pre_graft_dim = local_manifest.current_state.hidden_dim
    remapped.applies_to.post_graft_dim = local_manifest.current_state.hidden_dim + 1
    
    return remapped
```

### 10.3 Cotraining on Overlap

If fingerprint overlap is detected:

1. Fetch datasets for both local and incoming concepts
2. Pool datasets
3. Cotrain grafts for both concepts together
4. Each concept gets its own dimension, but biases are trained jointly to ensure compatibility

---

## 11. Minimal Viable Self-Learning Loop

A minimal implementation that still fits this spec:

1. **Reference substrate**:
   * a small but capable model (e.g. Gemma-style 270M or OLMo-style small model) with:
     * HAT lenses for a baseline ConceptPack,
     * motive simplexes and world ticks.

2. **Experience Database (XDB)**:
   * store Episodes, CandidateConcepts, Exemplars, ConceptDatasets, TrainingRuns, Grafts, SubstrateManifest.

3. **Gap detection**:
   * use low coverage / high uncertainty / repeated failures / novelty signals
   * to define CandidateConcepts.

4. **Self-curation only**:
   * build ConceptDatasets purely from internal GraphRAG over XDB,
   * no human or teacher labels required.

5. **Graft training**:
   * allocate new dimension,
   * train sparse biases,
   * train lens bound to new dimension,
   * validate internally (performance + safety metrics),
   * store Graft artifacts in XDB.

6. **Local promotion**:
   * treat accepted CandidateConcepts as **local stable concepts**:
     * they get a grafted dimension,
     * they get a bound lens,
     * they are used as tags in future GraphRAG queries.

7. **Substrate management**:
   * track SubstrateManifest,
   * periodically check for compaction candidates,
   * merge highly correlated dimensions.

8. **Optional tribe sync**:
   * later, share Grafts and ConceptDatasets with tribe-level systems,
   * check fingerprint overlap on incoming grafts,
   * cotrain when needed.

This loop yields continual, self-directed concept expansion with substrate growth, even without a large external teacher. Teacher models and humans become **strategic accelerators** and **alignment anchors**, not mandatory components of every learning step.

---

## 12. Summary

The Continual Concept Learning Harness enables beings to:

| Stage | What happens |
|-------|--------------|
| **Detect** | Notice gaps via low coverage, uncertainty, failures |
| **Tag** | Create CandidateConcepts, label episodes |
| **Curate** | Build ConceptDatasets via GraphRAG + XAPI |
| **Graft** | Train new dimension + biases + lens |
| **Validate** | Joint testing against holdout and safety suites |
| **Promote** | Emit ConceptDiff/GraftDiff, apply to substrate |
| **Manage** | Compact, merge, graduate as substrate grows |
| **Share** | Sync with tribe, handle incoming grafts |

The stack:

| Layer | Role |
|-------|------|
| **BE** | The being — exists, learns, grows |
| **Graft** | Grows new dimensions into the being's substrate |
| **MELD** | Integrates concepts into shared vocabulary |
| **MAP** | Charts the being's mindspace |
| **HAT** | The being's introspective capacity |
| **Hush** | Quiets and constrains the being's drives |
| **ASK** | Negotiates the being's relationships and rights |

And XDB is the being's memory. XAPI is how it remembers.
