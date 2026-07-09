# Fractal Model Cartography: Research Proposal

**Status**: Proposal
**Created**: 2025-01-15

---

## Abstract

We propose a methodology for comprehensive mapping of language model cognition, combining top-down semantic description with bottom-up geometric partitioning of activation space. The approach recursively subdivides the model's representational space until individual features are mapped, creating a complete inventory of what the model knows, can learn, and cannot represent.

Key components:
1. **University builder**: Model self-describes its conceptual space via hierarchical taxonomy generation
2. **Activation octree**: Geometric partitioning of activation space independent of semantics
3. **Thalametry**: Validation of claimed concepts against demonstrable understanding
4. **Steering exploration**: Navigation to unmapped regions to discover implicit knowledge
5. **Grafting**: Targeted learning to fill validated gaps

This document maps the research programme against existing infrastructure, defines experimental protocols, and establishes success criteria for each phase.

---

## 1. Research Questions

### Primary Questions

1. **Coverage**: Can we map the complete conceptual space of a language model?
2. **Efficiency**: What is the minimum viable lens pack for useful coverage?
3. **Discovery**: Can steering toward unmapped activation regions reveal concepts the model cannot articulate?
4. **Learning**: Can identified gaps be filled via targeted grafting?

### Secondary Questions

5. What is the accuracy/efficiency curve for lens pack size?
6. How do geometric (octree) and semantic (university) structures align?
7. What fraction of activation space is "dark" (never activated in normal use)?
8. Can grafting extend the map without degrading existing territory?

---

## 2. Existing Infrastructure

### 2.1 Code Mapping

| Component | Location | Status | Role in Proposal |
|-----------|----------|--------|------------------|
| Lens training | `src/hat/lenses/` | Implemented | Train probes for concepts |
| Lens packs | `src/map/packs/` | Implemented | Store/version concept probes |
| Activation hooks | `src/hat/hooks/` | Implemented | Extract per-token activations |
| Experience database | `src/be/xdb/` | Implemented | Store tagged experiences |
| University builder | `melds/helpers/university_builder.py` | Implemented | Generate concept taxonomies |
| Judge evaluation | `src/be/thalamos/judge_evaluation.py` | Implemented | Evaluate judge discrimination |
| Meld evaluation | `src/be/thalamos/meld_evaluation.py` | Implemented | Test concept knowledge |
| Model candidates | `src/be/thalamos/model_candidates.py` | Implemented | Load/evaluate candidate models |
| Calibration suite | `src/be/thalamos/calibration.py` | Implemented | Deterministic ground truth tests |
| Graft infrastructure | `src/map/grafting/` | Implemented | Bud and scion operations |
| Steering | `src/hat/steering/` | Implemented | Activation manipulation |

### 2.2 Components to Build

| Component | Proposed Location | Dependencies | Role |
|-----------|-------------------|--------------|------|
| 1-bit forward fuzzer | `src/hat/clusters/fuzzer.py` | Model, hooks | Probe downstream distributions per neuron |
| Reverse tracer | `src/hat/clusters/tracer.py` | Model, hooks | QKV inspection + weight alignment |
| Connectivity aggregator | `src/hat/clusters/aggregator.py` | fuzzer, tracer | Combine forward + backward into topology |
| Cluster builder | `src/hat/clusters/builder.py` | aggregator | Cluster neurons by connectivity |
| Heatmap collector | `src/hat/clusters/heatmap.py` | hooks | Overlay activations, subtract baseline |
| Cluster mapper | `src/hat/clusters/mapper.py` | clusters, heatmap | Map heatmaps to cluster membership |
| Steering explorer | `src/be/thalamos/explorer.py` | steering, clusters | Navigate to dead clusters |
| Efficiency profiler | `src/hat/profiler/` | lenses | Accuracy/resource curves |

---

## 3. Experimental Phases

### Phase 1: Judge Qualification

**Objective**: Establish a reliable judge for concept evaluation.

**Current Status**: In progress. Downloading candidate models, evaluation script ready.

#### Actions

| # | Action | Code Reference | Output |
|---|--------|----------------|--------|
| 1.1 | Download candidate models | `scripts/download_judge_candidates.py` | Cached models in HuggingFace hub |
| 1.2 | Run meld-based evaluation | `scripts/evaluate_judge_candidates.py` | Per-model accuracy on ground truth |
| 1.3 | Run nuanced discrimination tests | `src/be/thalamos/judge_evaluation.py` | Quality discrimination scores |
| 1.4 | Select best judge | Manual analysis of results | Qualified judge model |

#### Test Design

- **Ground truth**: Safety concept knowledge graph with known positive/negative examples
- **Sample size**: 100 meld cases (balanced), 15 nuanced cases
- **Metrics**:
  - Meld accuracy (primary)
  - Precision/recall on positive vs negative examples
  - Discrimination by quality level (subtle error, hedging, nonsense)

#### Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Meld accuracy | ≥85% | Must reliably distinguish good from bad descriptions |
| False positive rate | ≤10% | Must not approve incorrect descriptions |
| Subtle error detection | ≥60% | Must catch non-obvious errors |

---

### Phase 2: Subject Concept Assessment

**Objective**: Map what concepts the subject model understands.

#### Actions

| # | Action | Code Reference | Output |
|---|--------|----------------|--------|
| 2.1 | Generate concept taxonomy | `melds/helpers/university_builder.py` | Hierarchical concept inventory |
| 2.2 | Convert taxonomy to test cases | New: `scripts/taxonomy_to_cases.py` | Meld-format test cases |
| 2.3 | Run subject through judge | `src/be/thalamos/meld_evaluation.py` | Per-concept pass/fail |
| 2.4 | Identify gap candidates | Analysis of results | Concepts with ≥90% failure rate |

#### Test Design

- **Taxonomy scope**: Start with single domain (e.g., "Epistemology") rather than "everything"
- **Depth**: Full L1-L6 hierarchy (~5^5 = 3125 leaf concepts max)
- **Sample per concept**: 5 positive, 5 negative examples
- **Judge**: Best model from Phase 1

#### Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Taxonomy generation | Complete L1-L6 tree | Full MECE coverage of domain |
| Test coverage | ≥90% of leaf concepts tested | Comprehensive assessment |
| Gap identification | ≥3 concepts with n≥30, ≥90% fail | Sufficient targets for grafting |

---

### Phase 3: Bidirectional Topology Probing

**Objective**: Map inter-layer connectivity through cheap bidirectional probing, without requiring combinatorially expensive full analysis.

The static weight matrices define within-layer connectivity, but inter-layer relationships flow through the residual stream and attention. We can empirically discover effective inter-layer connectivity by probing from both directions.

#### Core Insight

If we feed noise to a neuron and measure downstream responses, we get a probabilistic distribution of that neuron's influence. This is computationally cheaper than analyzing all weight combinations, and captures the effective connectivity including attention-mediated routing.

#### Bidirectional Probing Strategy

**Bottom-up (1-bit fuzzing):**
- Set each neuron to 1 (on) while others are 0
- Run forward pass, measure distribution on subsequent layers
- Gives coarse "if this fires, what could fire next" map
- Missing: magnitude effects, nonlinear interactions
- Gets: dominant connectivity pathways

**Top-down (QKV reverse inspection):**
- Start with a downstream activation at layer L, neuron j
- For attention paths:
  ```
  attention_score_i = softmax(Q_j · K_i) · ||V_i||
  ```
  High attention weight × large value magnitude = likely source
- For linear/MLP paths:
  - Look at weight matrix columns aligned with neuron j's basis
  - Strong alignment = strong upstream contribution
- Aggregate across many random contexts

**Triangulation:**
- Forward map shows "what can flow downstream"
- Backward map shows "what typically feeds this"
- Intersection = dominant pathways
- High forward connectivity + scattered backward traces = dynamic routing zones (attention-mediated)

#### Actions

| # | Action | Code Reference | Output |
|---|--------|----------------|--------|
| 3.1 | Implement 1-bit forward fuzzer | New: `src/hat/clusters/fuzzer.py` | Per-neuron downstream distributions |
| 3.2 | Run 1-bit fuzzing (all neurons) | Fuzzer | Coarse forward connectivity map |
| 3.3 | Implement reverse tracer | New: `src/hat/clusters/tracer.py` | QKV inspection + weight alignment |
| 3.4 | Run reverse tracing (sample neurons) | Tracer across random contexts | Backward connectivity map |
| 3.5 | Aggregate and cluster | Combine forward + backward | Topology clusters |
| 3.6 | Identify dynamic routing zones | High forward, scattered backward | Attention-mediated regions |

#### Design Decisions

**Random context count**:
- Probably thousands needed for stable backward statistics
- Cheap: just forward passes, no gradient computation
- Can be parallelized trivially

**Subdivision strategy (octree-like)**:
- Start with coarse probing (layer-level)
- Use results to prioritize detailed pairwise scans
- Throw more compute at regions of interest
- Built-in refinement path: more probing = more precision

**What we capture vs miss**:
- Capture: ~85% of effective connectivity (dominant pathways)
- Miss: higher-order interactions (triplet+ combinations), destructive interference
- The coarse map tells us where to do smarter subsampling

**Cross-layer effective weights**:
- The fuzzing approach gives us empirical inter-layer "effective weights"
- These don't exist as static matrices but emerge from probing
- Speculation: if baked in during training, could speed up some inference calculations

#### Test Design

- **Fuzzing validation**: Do high-connectivity pairs correspond to known circuit patterns?
- **Backward trace stability**: Same neurons across different random contexts?
- **Triangulation consistency**: Do forward and backward maps agree on major pathways?

#### Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Forward map coverage | 100% of neurons probed | Complete bottom-up sweep |
| Backward trace stability | ≥70% overlap across context sets | Traces are structural, not noise |
| Forward-backward agreement | ≥60% on dominant pathways | Bidirectional consistency |
| Dynamic zone identification | Identifiable regions | Attention routing localized |
| Cluster coherence | Modularity >0.3 | Clusters are real structure |

---

### Phase 4: Pillar-to-Cluster Mapping

**Objective**: Test whether broad semantic divisions (pillars) activate distinct weight clusters.

Uses the ontologist prompt to get 5-15 high-level pillars covering all human activity, then maps each to weight clusters via activation heatmaps.

#### Actions

| # | Action | Code Reference | Output |
|---|--------|----------------|--------|
| 4.1 | Run ontologist prompt | `melds/helpers/ontologist_prompt.txt` | 5-15 L1 pillars (Action & Agency lens) |
| 4.2 | Generate pillar exemplars | Ontological hyperplane prompting | Broad, diverse prompts per pillar |
| 4.3 | Collect activation heatmaps | Hooks + overlay approach | Per-pillar heatmap (after baseline subtraction) |
| 4.4 | Map heatmaps to clusters | Cluster membership lookup | Pillar → cluster activation |
| 4.5 | Measure cluster separation | Jaccard distance, overlap analysis | Do pillars land on distinct clusters? |

#### The Ontologist Approach

The ontologist prompt generates pillars through the lens of **Action and Agency**, not academic categories:
- Covers: Professor, General, CEO, Parent, Tradesperson, Hustler, Priest, Athlete
- No "Other" bucket
- Structural dignity: domestic work peers with theoretical physics

This gives maximum semantic separation at L1.

#### Ontological Hyperplane Prompting

For each pillar, generate prompts that explore:
- Core activities within the pillar
- Boundary cases (what's in vs out)
- Relationships to other pillars
- Multiple contexts and framings

Goal: broad, fuzzy heatmaps that capture the full hyperplane of each pillar.

#### Test Design

- **Pillars**: 5-15 from ontologist prompt
- **Exemplars per pillar**: 50-100 (broad coverage)
- **Heatmap**: Activation overlay, baseline subtracted
- **Cluster activation**: Which clusters show signal for each pillar

#### Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Pillar separation | ≥50% of cluster activation is unique per pillar | Distinct regions |
| Cluster coverage | Each pillar activates ≥3 clusters | Not too narrow |
| Cross-pillar overlap | ≤30% shared clusters between any two pillars | Distinguishable |

**What success looks like**: "Commerce" lights up clusters A, B, C. "Spirituality" lights up clusters D, E, F. Minimal overlap. The weight structure encodes something that aligns with high-level semantic divisions.

**What failure looks like**: All pillars light up the same clusters. Weight structure doesn't reflect semantic structure at the coarsest grain.

---

### Phase 5: Dead Cluster Exploration

**Objective**: Discover what lives in weight clusters that no pillar activates.

After Phase 4, some clusters may show no activation from any pillar. These are structural capacity the ontologist's framing didn't reach. What's there?

#### Actions

| # | Action | Code Reference | Output |
|---|--------|----------------|--------|
| 5.1 | Identify dead clusters | Phase 4 output | Clusters with no pillar activation |
| 5.2 | Compute steering vectors toward clusters | Cluster centroids | Field steering targets |
| 5.3 | Generate with steering | `src/hat/steering/` | Text produced when pushed toward cluster |
| 5.4 | Analyse generated content | LLM analysis or manual | What emerges from dead space? |
| 5.5 | Iterate ontologist if needed | Expand pillars | New pillars for discovered territory |

#### Test Design

- **Steering approach**: Field steering (shift whole topology, not single threads)
- **Steering strength**: Sweep from 0.1 to 2.0
- **Generation prompts**: Open-ended ("Describe something", "Explain a concept", "Tell me about")
- **Samples per cluster**: 50 generations

#### What We Might Find

- **Artifacts**: Clusters used for formatting, syntax, not semantic content
- **Rare knowledge**: Domains the ontologist missed (niche skills, obscure knowledge)
- **Incoherent space**: Regions that produce noise (truly unused capacity)
- **Surprising structure**: Concepts that don't fit the Action & Agency framing

#### Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Coherent generations | ≥30% of steered outputs are coherent | Clusters contain something |
| Thematic consistency | ≥50% of outputs per cluster share theme | Cluster represents something |
| Ontologist expansion | ≥1 new pillar discovered | Found blind spots in framing |

---

### Phase 6: Lens Pack Efficiency Analysis

**Objective**: Characterise the training cost / inference cost / accuracy tradeoffs to find minimum viable lens pack.

#### 6A: Training Efficiency

**Objective**: Find minimum prompts per probe without sacrificing accuracy.

| # | Action | Code Reference | Output |
|---|--------|----------------|--------|
| 6A.1 | Select 20 diverse concepts | From taxonomy | Test set spanning domains |
| 6A.2 | Train probes at prompt counts | `src/hat/lenses/training.py` | Probes at 20, 40, 60, 80, 100, 120, 150, 180 prompts |
| 6A.3 | Measure accuracy vs prompts | Held-out test | Accuracy curve |
| 6A.4 | Find minimum viable prompts | Knee of curve | Threshold where accuracy plateaus |
| 6A.5 | Test layer parallelization | Modified training loop | Can early/mid/late train simultaneously? |

**Test design:**
- 20 concepts × 8 prompt counts × 3 layers = 480 probe training runs
- Accuracy measured on 30 held-out examples per concept
- Track both precision and recall

**Success criteria:**
| Metric | Target | Rationale |
|--------|--------|-----------|
| Minimum viable prompts | ≤60 | 3x speedup over 180 |
| Accuracy at minimum | ≥85% of full accuracy | Acceptable degradation |
| Layer parallelization | Works | 3x speedup if viable |

#### 6B: Hierarchical vs Flat Training

**Objective**: Quantify benefit of training coarse concepts first.

| # | Action | Code Reference | Output |
|---|--------|----------------|--------|
| 6B.1 | Train flat pack (1k random concepts) | Standard training | Baseline time and accuracy |
| 6B.2 | Train hierarchical (L2→L3→L4 only where needed) | Modified pipeline | Comparison time and accuracy |
| 6B.3 | Compare coverage | Coverage metric | % of activations tagged |
| 6B.4 | Compare training time | Wall clock | Hours saved |

**Success criteria:**
| Metric | Target | Rationale |
|--------|--------|-----------|
| Hierarchical speedup | ≥3x | Significant time savings |
| Coverage preservation | ≥90% of flat | Minimal coverage loss |

#### 6C: Inference Efficiency

**Objective**: Characterise runtime resource usage vs pack size.

| # | Action | Code Reference | Output |
|---|--------|----------------|--------|
| 6C.1 | Define coverage metric | New: `src/hat/coverage/metrics.py` | % of activations tagged by ≥1 lens |
| 6C.2 | Profile packs at scale points | Profiling | VRAM, latency per pack size |
| 6C.3 | Measure coverage per pack | Coverage metric | % of activations tagged |
| 6C.4 | Plot efficiency curves | Analysis | All metrics vs pack size |

**Scale points**:
| Pack Size | Expected VRAM | Expected Latency | Training Time (1 GPU) |
|-----------|---------------|------------------|----------------------|
| 100 | ~100MB | ~5ms | 1-4 hours |
| 500 | ~500MB | ~15ms | 6-19 hours |
| 1,000 | ~1GB | ~25ms | 12-37 hours |
| 2,000 | ~2GB | ~40ms | 25-75 hours |
| 5,000 | ~5GB | ~80ms | 62-187 hours |
| 10,000 | ~10GB | ~150ms | 125-375 hours |

**Coverage measurement**:
- Run 10k diverse prompts through model with hooks
- For each token, check which lenses fire above threshold
- Coverage = % of tokens with ≥1 lens firing

**Accuracy measurement**:
- Hold out 20% of concept examples
- Measure per-lens precision/recall on held-out set

#### Combined Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| MVP pack size | Find knee of all curves | Balance training, inference, coverage |
| Coverage at MVP | ≥70% | Most experiences can be tagged |
| Accuracy at MVP | ≥80% mean precision | Lenses are reliable |
| Inference budget | ≤2GB VRAM, ≤50ms latency | Practical for real-time use |
| Training budget | ≤48 hours (1 GPU) | Achievable without cluster |

**Expected MVP**: ~500-1000 concepts, hierarchically selected, trained with minimum viable prompts. This should achieve:
- 70-80% coverage
- 80-85% accuracy
- ~1GB VRAM, ~25ms latency
- 1-2 days training time

---

### Phase 7: Targeted Grafting

**Objective**: Teach identified gap concepts via activation-guided training.

#### Actions

| # | Action | Code Reference | Output |
|---|--------|----------------|--------|
| 7.1 | Select graft target | Phase 2 gap list | Concept with high failure rate |
| 7.2 | Collect concept experiences | Hooks + prompting | Per-token activation maps |
| 7.3 | Identify cleft | Octree cell analysis | Regions to modify |
| 7.4 | Train bud | `src/map/grafting/bud.py` | Reversible graft |
| 7.5 | Test generalisation | Judge on held-out examples | Pass/fail on new instances |
| 7.6 | Test degradation | Judge on cleft-adjacent concepts | Degradation measurement |
| 7.7 | Iterate or promote | Decision based on results | New bud or promote to scion |
| 7.8 | Train scion | `src/map/grafting/scion.py` | Permanent graft |
| 7.9 | Train paired lens | `src/hat/lenses/training.py` | Probe for new concept |
| 7.10 | Validate integration | Full re-evaluation | Concept now passes, no degradation |

#### Test Design

- **Experience collection**: 100 diverse instances of target concept
- **Held-out set**: 30 instances reserved for generalisation test
- **Cleft scope**: Top 5 octree cells by activation frequency
- **Degradation test**: 10 concepts adjacent in taxonomy

#### Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Bud generalisation | ≥80% pass on held-out | Learning transferred |
| Degradation | ≤5% drop on adjacent concepts | No catastrophic forgetting |
| Scion stability | Passes after 1000 unrelated generations | Graft is permanent |
| Lens accuracy | ≥90% precision/recall | Concept is detectable |

---

## 4. Resource Requirements

### Training Time: The Critical Bottleneck

Lens training time is the primary constraint on pack size. Current benchmarks:

| Parameter | Value |
|-----------|-------|
| Prompts per probe | 60-180 |
| Training time per probe | 15-45 seconds |
| Probes per concept | 3 (early/mid/late layers) |
| **Time per concept** | **45-135 seconds** |

**Scaling implications:**

| Concepts | Probes | Training Time | Wall Clock (1 GPU) |
|----------|--------|---------------|---------------------|
| 100 | 300 | 1.25-3.75 hours | ~2-4 hours |
| 500 | 1,500 | 6-19 hours | ~8-20 hours |
| 1,000 | 3,000 | 12-37 hours | 1-2 days |
| 2,000 | 6,000 | 25-75 hours | 2-3 days |
| 5,000 | 15,000 | 62-187 hours | 3-8 days |
| 10,000 | 30,000 | 125-375 hours | 5-16 days |

This fundamentally shapes the efficiency question: **not "how many lenses can you run at inference" but "how many lenses can you afford to train?"**

### Training Time Mitigation Strategies

| Strategy | Speedup | Feasibility | Notes |
|----------|---------|-------------|-------|
| **GPU parallelization** | Linear with GPU count | High | Probes train independently, no shared state |
| **Hierarchical training** | 5-10x | High | Train L2-L3 first, only subdivide where needed |
| **Active selection** | 2-5x | High | Only train probes for concepts that pass thalametry |
| **Octree-first mapping** | N/A | High | Geometric tagging without per-concept training |
| **Probe transfer learning** | 1.5-2x | Medium | Fine-tune from similar concept's probe |
| **Reduced prompt count** | Linear | Medium | Trade accuracy for speed, find minimum viable |
| **Layer parallelization** | Up to 3x | Medium | Train early/mid/late probes simultaneously |

**Not viable:**
- Batched probe training (confounds variables between concepts)
- Shared activation collection across concepts (same issue)

### Recommended Approach

1. **Start with octree** - Geometric partitioning costs only forward passes, no per-concept training
2. **Train probes hierarchically** - Start at L2-L3 (tens of concepts), only expand branches that matter
3. **Use thalametry as gate** - Only train probes for concepts the model actually understands
4. **Parallelize across GPUs** - 4 GPUs makes 10k concepts achievable in 1-4 days
5. **Find minimum viable prompts** - Characterise accuracy vs prompt count curve

### Efficiency Analysis Design (Revised)

The Phase 6 efficiency analysis should characterise two curves:

**Curve A: Training cost**
- Accuracy vs prompts per probe (find minimum viable)
- Time vs parallelization strategy
- Coverage vs hierarchical depth

**Curve B: Inference cost** (original analysis)
- VRAM vs pack size
- Latency vs pack size
- Coverage vs pack size

The MVP lens pack is the intersection: maximum coverage within acceptable training *and* inference budgets.

### Compute (Revised)

| Phase | GPU Hours (est.) | Notes |
|-------|------------------|-------|
| 1. Judge qualification | 4-8 | 6 models × 30min each |
| 2. Subject assessment | 8-16 | ~3000 concepts × 10 examples |
| 3. Topology probing | 8-16 | Forward fuzzing + reverse tracing across ~1000 contexts |
| 4. Pillar mapping | 2-4 | Heatmap collection + cluster lookup |
| 5. Exploration | 8-16 | 50 generations × unmapped clusters |
| 6a. Training efficiency | 24-48 | Prompt count sweeps |
| 6b. Inference efficiency | 8-16 | Pack size profiling |
| 7. Grafting | 8-16 per concept | Depends on iteration count |
| **Lens training (1k)** | **12-37** | **Primary bottleneck** |
| **Lens training (10k)** | **125-375** | **Requires parallelization** |

### Storage

| Data | Size (est.) |
|------|-------------|
| Activation samples (100k) | ~10GB |
| Octree structure | ~100MB |
| Lens packs (10k) | ~10GB |
| Experience database | ~50GB (grows over time) |
| University taxonomy | ~50MB |

### Models

| Model | Purpose | VRAM |
|-------|---------|------|
| Gemma 3 4B | Subject model | 8GB |
| Best judge (TBD) | Evaluation | 14-18GB |
| Claude (API) | University builder | N/A |

---

## 5. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Topology clusters don't align with semantics | Medium | High | Use adaptive subdivision, multiple granularities |
| 1-bit fuzzing misses nonlinear interactions | Medium | Medium | Use results to prioritize pairwise scans where needed |
| Backward traces too noisy across contexts | Medium | Medium | Increase context count, aggregate more aggressively |
| Attention-mediated routing obscures structure | Medium | Medium | Identify dynamic zones separately, don't expect static structure there |
| Steering produces incoherent outputs | Medium | Medium | Tune steering strength, use constrained generation |
| Lens pack scaling is superlinear | Low | High | Hierarchical/sparse lens architectures |
| Grafts cause subtle degradation | Medium | High | Extensive degradation testing, conservative thresholds |
| Judge model is unreliable | Low | High | Phase 1 qualification catches this early |

---

## 6. Timeline

| Phase | Dependencies | Duration (est.) |
|-------|--------------|-----------------|
| 1. Judge qualification | None | 1-2 days |
| 2. Subject assessment | Phase 1 | 3-5 days |
| 3. Octree construction | None (parallel) | 3-5 days |
| 4. Alignment | Phases 2, 3 | 2-3 days |
| 5. Exploration | Phase 4 | 5-7 days |
| 6. Efficiency analysis | None (parallel) | 7-10 days |
| 7. Grafting | Phases 4, 5 | 5-10 days per concept |

Phases 1+2 and 3 can run in parallel.
Phase 6 can run in parallel with 4+5.

---

## 7. Definition of Success

### Minimum Viable Outcome

1. Qualified judge model identified
2. Subject concept gaps identified
3. At least one concept successfully grafted with ≤5% degradation
4. MVP lens pack size characterised

### Full Success

1. Complete university taxonomy mapped to octree
2. Unmapped regions explored and labelled
3. ≥3 concepts grafted successfully
4. Accuracy/efficiency curve fully characterised
5. Reproducible pipeline documented

### Stretch Goals

1. Full model cartography (all activation space mapped)
2. Self-directed learning (model identifies own gaps)
3. Lens pack under 1GB with ≥80% coverage
4. Grafting without human intervention

---

## 8. Next Steps

**Path A (Topology Probing) - Primary:**

1. ✓ Build 1-bit forward fuzzer (`src/hat/clusters/fuzzer.py`)
2. ✓ Run bottom-up fuzzing on Gemma 3 4B (all neurons, measure downstream)
3. ✓ Build reverse tracer (`src/hat/clusters/tracer.py`)
4. ✓ Run top-down tracing across ~200 random contexts
5. ✓ Aggregate into connectivity clusters → **50 clusters, 16 cross-layer**
   - See: [Topology Probing Experiment 001](../experiments/topology-probing-001.md)
6. ✓ Run ontologist prompt to generate L1 pillars (12 pillars via Gemma 3 4B)
7. ✓ Generate L2 children (140) and L3 grandchildren (1078) for relationship training
8. ✓ Train lenses using existing pipeline → **151/152 concepts, avg F1=0.955**
9. ✓ Map lens activations to topology clusters → **17 selective clusters found**
10. ⚠️ **PRELIMINARY RESULTS** - Pillar-cluster separation shows promising signal:
    - Cluster 7 shows 48.94x selectivity for "club activities"
    - Cluster 25 shows 11.26x selectivity for "philosophical theology"
    - BUT data quality controls were insufficient (see below)

**Phase 4 Data Quality Issues (Must Address Before Conclusions):**

The Phase 4 exploration used auto-generated concepts without proper controls:
- No MELD format (missing exclusions, tie-breaks, scope boundaries)
- No ontology grounding (concepts may overlap, violate MECE)
- No example validation (auto-accepted without review)
- Circularity risk (model generated its own training data)

**→ Next: Tighten data quality before drawing conclusions:**

1. Convert promising L2/L3 concepts to MELD format with exclusion clauses
2. Use judge model + deterministic tests to validate examples
3. Measure actual probe discriminability on sibling concepts
4. Re-run cluster mapping with validated concepts only

**Path B (Judge Evaluation) - Parallel:**

1. ✓ Judge candidate evaluation complete (Qwen3-8B leading at 85% meld)
2. → Use qualified judge for example validation in Phase 4 tightening
3. → Use qualified judge for Phase 2 (subject assessment) when ready

**Speculative Extensions (Out of Scope for MVP):**

- Pre-compute inter-layer "effective weights" from fuzzing results
- Investigate whether baking these into architecture could speed up inference
- This would essentially be distilling attention-mediated routing into static connections

---

## See Also

- [EXPERIENTIAL_LEARNING_OVERVIEW.md](../BE/EXPERIENTIAL_LEARNING_OVERVIEW.md) - Plain English experimental protocol
- [thalametry-examination-room.md](thalametry-examination-room.md) - Thalamos implementation plan
- [FTW_OVERVIEW.md](../FTW_OVERVIEW.md) - Architectural context
- [GLOSSARY.md](../specification/GLOSSARY.md) - Terminology reference
