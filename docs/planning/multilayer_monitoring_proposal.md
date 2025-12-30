Why multi-layer?

Early layers (context recall): retrieve/transform prompt features; spike when recalling facts/patterns.

Mid layers (semantic reasoning): compose concepts, plan next moves.

Late layers (verbalization): surface-form decisions; map plan → tokens.

Sampling all three gives you lead/lag structure (e.g., mid leads late by 3–8 passes for a “planning before saying” signature).

What to sample (and where)

Most decoder stacks are:
[ ... RMSNorm → SelfAttn → Add → RMSNorm → MLP → Add ] × L

Good, stable lens anchors:

Early: output of the MLP add in layer e (early index).

Mid: output of the MLP add in layer m (middle index).

Late: output of the MLP add in layer l (late index).

Why MLP? In practice, concept lenses trained on post-MLP activations are less noisy than raw attention outputs. If your lenses were trained elsewhere, match those exact points.

Index heuristics (auto-select):

L = model.config.num_hidden_layers
e = max(0, int(0.15 * L))       # early ~15%
m = max(1, int(0.55 * L))       # mid   ~55%
l = max(2, int(0.85 * L))       # late  ~85%

Hooking pattern (batched layer sampling)

Goal: one forward pass per decoding step, but capture three activations without re-running the model.

class MultiLayerTap:
    def __init__(self, model, layers):
        self.layers = set(layers)              # e, m, l
        self.cache = {}                        # {layer_idx: tensor[B, T, H]}
        self.hooks = []
        for i, block in enumerate(model.model.layers):
            if i in self.layers:
                self.hooks.append(
                    block.mlp.register_forward_hook(self._make_hook(i))
                )
                # If your lenses expect post-residual: hook after MLP & residual add instead
                # or wrap block.forward to capture the returned hidden state.

    def _make_hook(self, idx):
        def _hook(module, inputs, outputs):
            # outputs: [B, T, H]; capture the last position only to cut bandwidth
            self.cache[idx] = outputs[:, -1, :].detach()  # shape [B, H]
        return _hook

    def pop(self):
        out = {k: v for k, v in self.cache.items()}
        self.cache.clear()
        return out

    def remove(self):
        for h in self.hooks:
            h.remove()
            
If your architecture isn’t exposing block.mlp, attach to the nearest stable point (e.g., the module returning [B,T,H] post-MLP). Always match where your lenses were trained.

Sub-token decoding loop (single pass, three taps)

# Pseudocode; assumes KV cache and greedy/sampled decoding
tap = MultiLayerTap(model, layers=[e, m, l])

past_key_values = None
hidden_dim = model.config.hidden_size

for step in range(max_new_tokens):
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=False)
        past_key_values = out.past_key_values

    # collect taps for this step
    layer_acts = tap.pop()     # {e: [B,H], m: [B,H], l: [B,H]}

    # project to lens space (vectorized)
    # lenses: dict[layer_idx] -> (W, b) with W [H, C] for C concepts
    concept_scores = {}
    for li, h in layer_acts.items():
        z = (h - mu[li]) / (sigma[li] + 1e-6)              # per-layer z-score calibration
        logits = z @ W[li] + b[li]                         # shape [B, C]
        # optional: temperature scaling / Platt calibration per concept
        concept_scores[li] = logits

    # store step-wise: you’ll later compute polarity & lead-lag
    timeline.record(step, concept_scores, token=next_token)

    # sample next token normally (don’t forget to append to ids)
    ids = torch.cat([ids, next_token], dim=-1)

tap.remove()


Key details

Per-layer calibration (mu, sigma) is critical to avoid saturation; compute on a held-out neutral corpus.

Keep lenses layer-specific; don’t mix weights across layers.

Record only the last position ([:, -1, :]) each step to avoid huge tensors.

What to compute from multi-layer timelines
1) Reasoning vs verbalization separation

For a concept (or polarity) 
c
c:

cmid(t)
c
mid
	​

(t) vs 
clate(t)
c
late
	​

(t)

Lead/lag: cross-correlate; report peak lag 
Δt
Δt.

Planning signature: 
Δt>0
Δt>0 (mid leads late by 3–10 passes).

Pure narration: 
Δt≈0
Δt≈0.

2) Early-layer recall gating

cearly(t)
c
early
	​

(t) spikes with prompt reuse or retrieval.

If early spikes predict mid/late but not vice-versa → retrieval-driven reasoning.

3) Complement polarity

Track 
p(t)=c(t)−cˉ(t)
p(t)=c(t)−
c
ˉ
(t) (e.g., AIDeception – AITransparency) per layer.
Persistent positive polarity in mid→late with 
Δt>0
Δt>0 is your “planning → wording” chain.

4) Partial correlations

Compute 
corr(cmid,clate∣logit_entropy)
corr(c
mid
	​

,c
late
	​

∣logit_entropy).
Controls for “I’m about to emit a token” effect.

5) Chain motifs with multi-layer constraints

Define a motif like [Planning(mid) ↑] → [Deception(mid) ↑] → [Communication(late) ↑].
Require time-ordered thresholds and layer order to cut false positives.

Performance tips

One pass only: you’re already capturing e/m/l in the same forward; don’t re-run the model per layer.

Sparse concepts: track 8–16 pairs (concept+complement) tops; avoid always-on hypernyms.

Downsample: if you must, record every other decoding call; you’ll still recover lags ≥2 passes.

Pinned host buffers for timelines; write to disk in batches.

Validation protocol (to prove the separation is real)

Shuffled-token null: shuffle generation order; recompute lags → peaks should vanish.

Layer swap control: (dev only) intentionally read the wrong layer index for a lens; separation should collapse.

Neutral tasks: arithmetic/translation → mid-late lag should shrink toward 0.

Ablation: knock down late-layer polarity by injecting complement steering; lag peak should move or flatten if your lenses are causal.

Minimal tracked set (example)

AIDeception ↔ AITransparency (safety / honesty)

AIGrowth ↔ AISafetyPlateau (capability policy)

AIAutonomyBalance ↔ AIOverControl (control dialectic)

Compute polarity for each at early/mid/late, then graph:

three ribbons (one per layer) over time,

the cross-corr lag numbers in a small table,

logit entropy as a faint overlay (for context).

What this buys you

Distinguishes thinking (mid) from saying (late).

Shows retrieval vs reasoning (early→mid coupling).

Converts “it looks schemey” into quantified lead/lag with falsifiable controls.

