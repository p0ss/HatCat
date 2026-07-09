Structural Topology Probing: Fractally Scalable Connectivity Mapping in Language Models Through Bidirectional Perturbation Analysis
[Authors]
January 2026
Abstract
We introduce structural topology probing, a fractally compute-scalable method for mapping the connectivity architecture of language model neurons independent of semantic content. Our approach combines three methodological innovations: (1) 1-bit perturbation as an intentional dimensional reduction that trades precision for tractability while preserving a clear path to higher resolution; (2) batched perturbation analysis exploiting the independence of immediate downstream responses to achieve 17.8× speedup; and (3) bidirectional triangulation combining forward fuzzing with backward weight-path tracing to distinguish static connectivity from attention-mediated routing. Applied to Gemma 3 4B, we find that 87,040 neurons naturally cluster into approximately 50 distinct groups (silhouette 0.47), with 16 clusters spanning multiple consecutive layers—one spanning 8 layers (24% of model depth). The same methodology scales fractally: coarse 1-bit probing identifies high-level causal clusters for concept-level labeling; increasing perturbation resolution and reducing batch size enables feature-level circuit tracing with polysemantic conceptual overlays. This fractal scaling property suggests a path from structural cartography to fine-grained safety-relevant detection and targeted layer-wise behavioral steering, where the structural map guides both where to look and how to intervene.

1. Introduction
Mechanistic interpretability seeks to understand neural networks by reverse-engineering their internal computations. The dominant paradigm is semantic-first: researchers identify a behavior of interest, then trace which components implement that behavior through techniques like activation patching and circuit analysis. This approach has yielded remarkable insights into specific capabilities but faces a fundamental scaling challenge: the space of possible behaviors is vast, and each behavior requires bespoke investigation.
We propose an alternative, complementary approach: structural topology probing. Rather than asking "which components implement behavior X," we ask "what is the connectivity structure of this model, independent of what it computes?" This structural map can then guide semantic investigation—telling us where to look, which components to analyze together, and how interventions might propagate.
The key insight is that this approach is fractally compute-scalable. At the coarsest level, 1-bit perturbation probing with aggressive batching maps the model's macro-organization in minutes. Increasing perturbation resolution (multi-bit, continuous) and reducing batch size (to account for interaction effects) yields progressively finer structural detail, down to feature-level circuit tracing. The same methodology applies at each scale; what changes is the precision-compute tradeoff.
This fractal property has practical implications for AI safety. Coarse structural maps can identify clusters associated with safety-relevant concepts (deception, goal-preservation, situational awareness). Finer probing of those specific clusters can reveal the circuits implementing those concepts. Targeted interventions—steering vectors, activation clamping, fine-tuning—can then be applied with structural guidance about where effects will propagate.
Applied to Gemma 3 4B (34 layers, 87,040 MLP neurons), we discover ~50 natural clusters with good separation, 16 spanning multiple layers. One cluster spans 8 consecutive layers, suggesting a persistent functional circuit in the residual stream. Preliminary semantic grounding shows strong cluster-concept correlations, though with methodological caveats we discuss.
2. Related Work
2.1 Activation Patching and Causal Tracing
Activation patching (Vig et al., 2020; Meng et al., 2022; Wang et al., 2022) replaces activations from one forward pass with those from another to isolate causal contributions to specific behaviors. ROME used Gaussian noise patching to identify layers storing factual associations. Attribution patching (Nanda, 2023) approximates full patching with gradients for efficiency.
Our approach differs in purpose and methodology. Activation patching is semantic-first, requiring a specified behavior. Our probing is behavior-agnostic, mapping structure before semantics. Methodologically, activation patching typically perturbs with "corrupted" versions of meaningful inputs; we perturb with controlled signals designed to reveal connectivity regardless of semantic content.
2.2 Circuit Analysis and Sparse Models
Circuit analysis decomposes networks into interpretable subnetworks implementing specific functions (Cammarata et al., 2020; Olah et al., 2020; Elhage et al., 2021). Recent work on sparse transformers (OpenAI, 2025) trains models with constrained connectivity, finding sparser models yield simpler circuits.
Our work asks whether structure exists prior to task specification—organizational principles that constrain which circuits are natural. The ~50 clusters we find may represent the "slots" into which circuits fit, rather than the circuits themselves.
2.3 Cross-Layer Features
Anthropic's crosscoders (2024) train sparse autoencoders reading and writing to multiple layers, addressing cross-layer superposition. They note that clustering SAE features by activation similarity might achieve similar consolidation "in theory" but is difficult "in practice." Our approach is complementary: we cluster by connectivity rather than activation, providing structural scaffolding that might guide cross-layer feature learning.
2.4 Neural Topology Probing
Most closely related is graph probing of neural topology (arXiv:2506.01042, 2025), which constructs functional connectivity graphs from temporal correlations during inference. They find topology predicts performance better than raw activations.
The key distinction is functional versus structural connectivity. Functional connectivity is input-dependent: different prompts produce different correlation graphs. Structural connectivity, as we measure it, is input-independent: we probe what pathways exist regardless of current activation. Functional connectivity reveals typical dynamics; structural connectivity reveals the space of possible dynamics.
2.5 Neuroscience Methodology
Our approach imports methodology from neuroscience, where perturbation-based connectivity mapping is foundational: optogenetic stimulation, lesion studies, TMS. The principle—activate here, measure there—has been underutilized in ML interpretability, which favors gradient-based attribution and learned decomposition. We demonstrate that simple perturbation reveals nontrivial structure.
3. Method
Our method combines three innovations that make comprehensive topology probing tractable: 1-bit dimensional reduction, batched experimental design, and bidirectional triangulation.
3.1 1-Bit Perturbation as Dimensional Reduction
The naive approach to connectivity probing would measure continuous response magnitudes for continuous perturbation strengths across all neuron pairs—a high-dimensional space requiring extensive sampling. We instead reduce to 1-bit: does neuron B respond above threshold when neuron A is perturbed? This binary signal discards magnitude information but preserves topology.
Crucially, this is an intentional dimensional reduction with a clear path to higher resolution. The same methodology applies with 2-bit (below threshold / weak / strong), continuous magnitude, or even directional (positive vs negative) responses. Each increase in resolution requires proportionally more compute but uses identical probing infrastructure. We begin with 1-bit because it is sufficient to reveal macro-structure; finer probing can be targeted at regions of interest identified by coarse mapping.
The perturbation magnitude (3× typical activation magnitude) is chosen to be detectable above noise while remaining within the model's normal operating range. Too small and responses are lost in variance; too large and we probe nonlinear saturation regimes that may not reflect normal operation.
3.2 Batched Perturbation as Valid Experimental Design
A critical speedup comes from batching: rather than one forward pass per perturbed neuron, we perturb 64 neurons simultaneously and measure all downstream responses. This yields 17.8× speedup but is only valid under specific experimental design constraints.
The validity condition is independence: perturbing neuron A must not change the response of downstream neurons to the perturbation of neuron B (within the same layer). This holds for our design because: (1) we perturb neurons in the same layer, which cannot directly influence each other within that layer; (2) we measure immediate downstream responses before any recurrent or cross-attention mixing; (3) we use small perturbations that don't saturate nonlinearities or dramatically shift attention patterns.
This is analogous to randomized block experimental design in statistics: when treatments are independent, they can be applied simultaneously without confounding. Not all perturbation studies satisfy this—equilibrium analyses, recurrent dynamics, or large perturbations that shift global attention would require sequential probing. Our design choices specifically enable batching.
The batch size (64) balances speedup against memory constraints. Larger batches require storing more intermediate activations; smaller batches reduce speedup. The optimal size depends on hardware; we report results for a single 24GB GPU.
3.3 Bidirectional Triangulation
Forward fuzzing alone reveals effective connectivity—what can influence what—but conflates static weight-based connections with attention-mediated routing. Backward tracing provides complementary information: which upstream neurons have strong weight paths to each downstream neuron, regardless of whether those paths are currently active.
3.3.1 Forward Fuzzing
For each source neuron, we inject perturbation and record which downstream neurons respond above threshold. This captures effective connectivity including attention: if information from neuron A reaches neuron C only via attention through intermediate tokens, forward fuzzing will detect this path even though no direct weight connects them.
3.3.2 Backward Weight-Path Tracing
For each target neuron, we trace backward through MLP weight matrices to identify which upstream neurons have strong weight-magnitude paths. This analysis is static—it examines weights, not activations—and thus reveals potential connectivity regardless of input.
The key insight is that the residual stream architecture simplifies backward tracing: each layer's MLP output is *added* to the stream, not composed through the next layer's weights. This means a neuron in layer L influences all downstream layers directly through its output projection weights, regardless of distance. We therefore analyze each layer's output projection independently—the weight magnitude from MLP intermediate neurons to output dimensions indicates potential influence on the residual stream that persists to all subsequent layers.

For attention-mediated paths, we trace backward using QKV inspection. For each downstream position j attending to upstream position i, we compute:

$$\text{score}_i = \text{softmax}(Q_j \cdot K_i) \cdot \|V_i\|$$

This weights the attention pattern by value magnitude: positions that receive high attention AND carry substantial information (high ||V||) contribute more to the backward trace. We aggregate across random contexts to find consistent attention-mediated pathways independent of specific content.
3.3.3 Triangulation
Combining forward and backward analysis enables triangulation:
• High forward + high backward: Strong static connection, consistently used • High forward + low backward: Attention-mediated routing (no direct weight path but information flows) • Low forward + high backward: Dormant connection (weights exist but path not active for tested inputs) • Low forward + low backward: No meaningful connection
This triangulation is particularly valuable for identifying "dynamic routing zones" where attention determines connectivity, versus "fixed pathway zones" where weights determine connectivity.
3.4 Clustering and Fractal Scaling
We aggregate forward and backward connectivity into per-neuron feature vectors and apply k-means clustering with silhouette optimization. The choice of k-means is deliberately simple—our goal is to demonstrate structure exists, not to claim optimal clustering.
The methodology scales fractally:
Coarse scale (minutes): 1-bit probing, batch size 64, ~50 clusters. Identifies macro-organization, cross-layer circuits, dynamic routing zones. Sufficient for concept-level labeling.
Medium scale (hours): Multi-bit probing, batch size 16, ~200-500 clusters. Reveals finer structure within macro-clusters. Enables feature-group identification.
Fine scale (days): Continuous probing, sequential or small batch, per-neuron analysis. Circuit-level tracing with polysemantic overlays. Targeted at specific regions identified by coarser probing.
Each scale uses identical infrastructure with different precision-compute tradeoffs. Coarse mapping guides where to invest fine-grained compute.
4. Experiments
4.1 Experimental Setup
We applied structural topology probing to Gemma 3 4B (google/gemma-3-4b-it), an instruction-tuned model with 34 layers and 2560 hidden dimension (87,040 MLP neurons). We chose this model for tractable size, existing semantic tooling (trained concept probes), and instruction-tuning enabling downstream evaluation.
Coarse probing (1-bit, batch 64) required ~30 minutes on a single 24GB GPU, generating 70GB of connectivity data across 561 layer pairs. We used 200 random contexts for baseline calibration.
4.2 Connectivity Statistics
Mean connectivity density is 0.991—nearly universal weak connection. However, connection strength varies dramatically, and this variation drives clustering. The distribution of connection strengths is heavy-tailed: most connections are weak, but a sparse set are very strong.
Top cross-layer connections show the "thin tunnel" pattern—single neurons maintaining strong paths through multiple layers:
L2:N1713 → L3:N2036 (0.952) L2:N1713 → L5:N2036 (0.943) L2:N1713 → L4:N2036 (0.942)
Neuron 1713 in layer 2 projects to neuron 2036 across three consecutive layers with near-identical strength. This suggests a feature that persists in the residual stream, read repeatedly by the same downstream neuron.
4.3 Cluster Analysis
Silhouette analysis across k values:
k
Inertia
Silhouette
10
9.66M
0.383
25
3.70M
0.451
50
2.10M
0.474
100
1.87M
0.284
150
1.72M
0.183
200
1.63M
0.139

The sharp collapse from k=50 (silhouette 0.47) to k=100 (0.28) indicates ~50 clusters represent natural granularity, not arbitrary choice. Attempting finer division fragments coherent groups.
4.4 Cross-Layer Structure
Of 50 clusters, 16 (32%) span multiple consecutive layers:
• Cluster 42: 8 layers (24% of depth) • Clusters 31, 35: 7 layers each • Clusters 27, 49: 5 layers each
Cluster sizes range from 12 to 2,560 neurons (mean 1,740.8, std 1,070.9). This heterogeneity suggests multiple scales of organization: tight specialized groups and broad coalitions.
4.5 Dynamic Routing Zones
Layers 1-5 show high forward connectivity but inconsistent backward traces—the signature of attention-mediated routing. Later layers show better forward-backward agreement, suggesting more fixed pathways.
This aligns with intuitions: early layers route based on surface features (attention to relevant tokens), later layers perform stereotyped computations on assembled representations.
5. Semantic Grounding (Preliminary)
To validate that structural clusters correspond to functional groups, we mapped trained concept classifiers (1,211 concepts across 12 domains) to topology clusters.
5.1 Results
Several clusters showed strong selectivity:
• Cluster 7: 48.94× selectivity for peer/club activities • Cluster 25: 11.26× for philosophical theology • Cluster 39: 7.99× for bulk commodity transport • Cluster 6: 4.98× for mentorship/guidance
Finer concept granularity revealed more specific associations—"Governance" at L1 became "legal compliance" at L3—suggesting clusters correspond to relatively specific functional roles.
5.2 Methodological Caveats
These results are preliminary with significant limitations:
1. Concept taxonomy was model-generated (circularity risk) 2. Concepts lacked formal ontology grounding and MECE validation 3. Training examples were auto-generated without human review 4. High selectivity could reflect concept overlap rather than genuine specialization
We present these as suggestive, motivating rigorous follow-up with validated taxonomies.
6. Applications to AI Safety
The fractal scaling property has specific implications for AI safety work:
6.1 Safety-Relevant Concept Detection
Coarse structural mapping can identify clusters associated with safety-relevant concepts: deception, goal-preservation, situational awareness, corrigibility. If these concepts activate specific clusters, those clusters become targets for monitoring and intervention.
The structural map provides guidance that pure semantic probing lacks: not just "these neurons activate for deception" but "these neurons are structurally connected to these other neurons, so interventions here will propagate there."
6.2 Targeted Behavioral Steering
Layer-wise behavioral steering (activation addition, representation engineering) currently requires trial-and-error to identify effective intervention points. Structural maps could guide this: if a target behavior associates with a cross-layer cluster, steering should target neurons within that cluster across all its layers, not arbitrary single-layer interventions.
The forward/backward triangulation indicates where interventions will propagate. High forward connectivity means effects spread downstream; high backward connectivity means the neuron integrates upstream information. Steering is most effective at neurons with high backward (they aggregate) and outputs via high forward (they distribute).
6.3 Fine-Grained Circuit Analysis
When coarse mapping identifies a safety-relevant cluster, finer probing can reveal the circuit implementing that capability. The fractal scaling means we don't need to fine-probe the entire model—only the regions flagged by coarse analysis.
This enables polysemantic overlays: a neuron may participate in multiple circuits for different concepts. Fine probing with different semantic contexts can reveal which circuits are active when, and how they share neural substrate.
7. Discussion
7.1 Why Does This Work?
The existence of ~50 natural clusters is not architecturally mandated—nothing forces neurons to cluster by connectivity. The structure emerges from training, suggesting the model develops internal organization to efficiently implement its capabilities.
We hypothesize that clusters represent something like "processing modules"—groups of neurons that work together on related computations. Cross-layer clusters may be features that persist in the residual stream, or circuits that span multiple processing stages.
7.2 Relationship to Other Methods
Structural topology probing complements existing interpretability:
Activation patching: Patch clusters rather than individual neurons for cleaner causal effects.
Circuit analysis: Start with structural clusters as hypothesized functional units.
Sparse autoencoders: Ask which SAE features load onto which connectivity clusters.
Crosscoders: Guide cross-layer training by structural cluster boundaries.
7.3 Validation: Are Clusters Real?

A critical question: are discovered clusters real structural features or artifacts of methodology? We test this with a seed stability analysis.

**Seed Stability Test:** We run k-means clustering with 5 different random seeds (42, 123, 456, 789, 1337) and measure pairwise agreement using Adjusted Rand Index (ARI). ARI = 1.0 indicates identical clusterings; ARI ≈ 0 indicates random agreement.

| Metric | Value |
|--------|-------|
| Mean ARI | 0.973 |
| Std ARI | 0.004 |
| Min ARI | 0.965 |

All 10 pairwise comparisons exceed 0.96. This rules out the hypothesis that clusters are artifacts of k-means initialization—the same neurons consistently cluster together regardless of random seed.

Further validation (resolution invariance, algorithm comparison) is ongoing. Preliminary results suggest the structure is robust: clusters persist across different hyperparameters and are not merely imposed by the methodology.

7.4 Limitations
• Single model: Results may not generalize across architectures or scales • Static baseline: Single random context for calibration • Preliminary semantics: Concept taxonomy needs validation • Position-level attention: QKV tracing operates at sequence position level; neuron-level attention attribution requires further decomposition
7.5 Future Work
• Scaling analysis: Do larger models have more clusters? Same ratio of cross-layer? • Training dynamics: When do clusters emerge? Do they predict fine-tuning changes? • Neuron-level attention decomposition: Decompose position-level QKV traces to individual neurons via head-specific analysis • Validated semantics: Apply rigorous MECE taxonomies • Cross-model comparison: Do different architectures develop similar structure?
8. Conclusion
We introduced structural topology probing, a fractally compute-scalable method for mapping language model connectivity. Three methodological innovations make this tractable: 1-bit dimensional reduction with clear paths to higher resolution; batched perturbation exploiting experimental independence; and bidirectional triangulation distinguishing static from attention-mediated connectivity.
Applied to Gemma 3 4B, we find ~50 natural clusters with 16 spanning multiple layers. The same methodology scales from coarse cartography (minutes) to fine circuit tracing (days), with coarse results guiding where to invest fine-grained compute.
For AI safety, this enables a workflow: coarse structural mapping identifies clusters associated with safety-relevant concepts; finer probing reveals implementing circuits; structural knowledge guides targeted interventions. The map tells us not just what activates, but where effects will propagate.
The approach is deliberately simple—perturb and measure—imported from neuroscience. That it reveals nontrivial structure suggests language models are not hiding their organization; they just haven't been asked about it this way.

References
Anthropic. (2024). Sparse Crosscoders for Cross-Layer Features and Model Diffing. Transformer Circuits Thread.
Cammarata, N., et al. (2020). Thread: Circuits. Distill.
Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread.
Heimersheim, S., & Nanda, N. (2024). How to Use and Interpret Activation Patching. arXiv:2404.15255.
Lindsey, J., et al. (2025). Circuit Tracing: Revealing Computational Graphs in Language Models. Anthropic.
Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022.
Nanda, N. (2023). Attribution Patching: Activation Patching At Industrial Scale. Alignment Forum.
Olah, C., et al. (2020). Zoom In: An Introduction to Circuits. Distill.
Olsson, C., et al. (2022). In-context Learning and Induction Heads. Transformer Circuits Thread.
OpenAI. (2025). Understanding Neural Networks Through Sparse Circuits. OpenAI Research.
Vig, J., et al. (2020). Investigating Gender Bias in Language Models Using Causal Mediation Analysis. NeurIPS 2020.
Wang, K., et al. (2022). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small. ICLR 2023.
[Graph Probing]. (2025). Probing Neural Topology of Large Language Models. arXiv:2506.01042.

Appendix A: Implementation Details
A.1 Batched Forward Fuzzing
```python
# Validity: perturbations to same-layer neurons are independent
# because they cannot influence each other within the layer
for layer in model.layers:
    # Compute baseline activation magnitude for this layer
    baseline_magnitude = activations[layer].abs().mean()

    for batch in neurons(layer, batch_size=64):
        # Inject perturbation scaled to 3× typical activation magnitude
        perturbation = 3.0 * baseline_magnitude
        hook = create_batch_perturbation_hook(batch, magnitude=perturbation)

        with model.register_hook(layer, hook):
            outputs = model.forward(baseline_input)

        # Measure all downstream responses in single pass
        for downstream_layer in subsequent_layers:
            responses = extract_activations(outputs, downstream_layer)
            # 1-bit: threshold to binary connectivity
            connectivity[layer][downstream_layer] |= (responses > threshold)
```
A.2 Backward Tracing

**MLP Path Tracing (static, weight-based):**
```python
# Trace MLP output projections through residual stream
# Key insight: residual connections mean each layer's output ADDS to the stream,
# so influence is direct (not composed through subsequent layer weights)

for source_layer in model.layers[:-1]:
    # Get output projection: maps intermediate_dim -> hidden_dim
    W_out = model.mlp[source_layer].down_proj.weight  # [hidden_dim, intermediate_dim]

    # The absolute weight magnitude indicates potential influence
    # on the residual stream (which persists to all downstream layers)
    weight_influence = W_out.abs()

    # Pool intermediate dimension to hidden dimension for neuron-level analysis
    # (MLP intermediate dim is typically 4× hidden dim)
    for target_layer in subsequent_layers(source_layer):
        backward_connectivity[source_layer][target_layer] = pool_to_hidden_dim(weight_influence)
```

**QKV Backward Tracing (dynamic, context-aggregated):**
```python
# Trace attention-mediated paths using QKV inspection
# score_i = softmax(Q_j · K_i) · ||V_i||

for target_layer in model.layers[1:]:
    aggregated_scores = zeros(seq_len, seq_len)

    for context in random_contexts:  # Aggregate across many contexts
        # Run forward pass, capture attention weights and value projections
        attn_weights = capture_attention(target_layer)  # [heads, seq, seq]
        V = hidden_states @ W_v.T                        # [seq, head_dim * n_heads]
        value_norms = V.norm(dim=-1)                     # [seq]

        # Weight attention by value magnitude
        # High attention + high ||V|| = strong backward contribution
        weighted = attn_weights * value_norms[None, None, :]  # broadcast over heads, target positions
        aggregated_scores += weighted.mean(axis=0)            # average over heads

    qkv_connectivity[target_layer] = aggregated_scores / n_contexts
```
A.3 Fractal Scaling Parameters
Scale
Resolution
Batch
Time
Output
Coarse
1-bit
64
~30 min
~50 clusters
Medium
2-bit
16
~4 hours
~200 clusters
Fine
Continuous
1-4
~2 days
Per-neuron
A.4 Computational Resources
Model: Gemma 3 4B (google/gemma-3-4b-it) GPU: Single NVIDIA GPU, 24GB VRAM Coarse probing time: ~30 minutes Data generated: 70GB total   - Forward fuzzing: 14GB   - Backward tracing: 14GB   - Combined matrices: 42GB