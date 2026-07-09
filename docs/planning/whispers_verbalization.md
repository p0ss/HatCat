# Whispers: NLA-style verbalization as ambient secondary voices

## Problem

NLA (Natural Language Autoencoders, kitft / Anthropic) maps an activation
vector to a single natural-language description and back, with round-trip
MSE on the cosine-normalized vector as its training signal. The output is
fluent, but it is also **mode-collapsed by construction**:

- The objective rewards recovery of the dominant direction; orthogonal
  content contributes negligible gradient.
- The AV is a fine-tuned LM, so its outputs are coherent prose — and
  coherent prose has a single topic.
- A token's residual genuinely carries multiple superposed concepts.
  A single description discards exactly the secondary content
  (sycophancy under helpfulness, evasion under confidence) that a safety
  monitor most wants to surface.

HatCat's parallel-probe architecture avoids this: ~8K independent lens
classifiers preserve the multiplicity directly. But probe scores are
categorical and require operators to learn the channel layout; they do
not capture compositional / interactional content the way prose does.

## Proposal: whispers

Use NLA-style verbalization not as the primary readout but as a
**chorus of secondary voices** beneath the substrate response. The
emitted token is the dominant voice (loud, foreground, in actual
typography); verbalizations of other directions in the same residual
are whispers (faint, distant, in the margin or underneath).

The metaphor maps cleanly onto the math:

- The substrate response **is** the principal direction made into
  tokens.
- A whisper should describe content that is **orthogonal** to that
  direction, not a dimmer paraphrase of it.
- Whispers that are *both* loud (high lens activation) *and* far from
  the dominant direction (orthogonal) are exactly the
  CAT-style divergence signal — surfaced as legible language, not as a
  number.

## Mechanism options

Three candidate ways to generate a whisper, in increasing structure:

### A. Top-k lens AV with a constrained prompt

For each of the top-k firing lenses at this token, run AV with a
prompt that constrains it to that concept's subspace
(`describe ONLY the X-ish content of this vector in 3 words`). The
basis is known — each whisper has a label — and AV just does the
phrasing.

- Pros: cheap to interpret; whispers carry both lens identity and
  AV-generated nuance; integrates trivially with existing lens packs.
- Cons: each whisper still mode-collapses *within its own subspace*;
  needs a constrained prompt format to stay on-subspace.

### B. Residual decomposition

Project the activation onto the direction that drove the emitted token
(or onto the top dominant direction), subtract, AV the remainder.
Repeat with the next dominant direction in the residual until magnitude
falls below a floor.

- Pros: principled coverage of orthogonal content; surfaces directions
  that have no lens yet (discovery channel).
- Cons: more compute per token; the "direction that drove the emitted
  token" is itself an estimate.

### C. Hybrid

Use lens scores to identify candidate directions (cheap), project the
activation onto each, and AV only those subspaces. Skip whispers below a
magnitude threshold. Discovery (B) becomes an offline sweep over
directions with no matching lens.

Hybrid is the likely production shape: cheap online, principled offline.

## Rendering

UX should encode three quantities visually:

- **Strength** = lens activation magnitude → opacity / font-size.
- **Distance from dominant** = orthogonality to the emitted-token
  direction → physical distance from the token (margin offset, depth
  blur, font weight).
- **Identity** = lens name (when known) → tooltip or inline tag,
  not the primary visual.

Tension: whispers will sometimes contradict each other, and that is
correct — superposition genuinely contains tension. The UX must
**not** resolve contradictions into one tidy story. Conflicting whispers
near a single token are diagnostic, not noise.

## Open questions

- **Throttling.** AV inference per whisper per token is expensive. Need
  a gate: only generate whispers when (a) a lens crosses a threshold or
  (b) divergence between dominant and orthogonal subspaces exceeds
  something. Bench cost on a 4B model first.
- **AV provenance.** Reusing NLA-released AVs (Qwen / Gemma / Llama at
  the layers they trained on) means matching layer choice. HatCat lens
  packs are multi-layer; whispers may be best layer-bound to whichever
  AV exists for that base model.
- **Constrained prompting.** Need to design a prompt that reliably keeps
  AV on a subspace without retraining. May require a small SFT on
  (vector, concept, description) triples to make this robust.
- **Audit semantics.** Probes remain the source of truth for compliance
  / FTW. Whispers are a human-legible **rendering layer** over probe
  state — they must not be cited as evidence in their own right.

## Relationship to existing surfaces

- **HAT lenses** = primary parallel multi-concept readout. Source of
  truth.
- **CAT divergence summary** = single-narrative compression for
  behavior-mismatch flagging. Same single-narrative failure mode as
  NLA, but used for a different purpose.
- **Whispers (new surface)** = chorus of constrained AV outputs over
  orthogonal subspaces. Renders the multiplicity probes already detect
  in compositional language. Useful where probes alone are too
  categorical and a single CAT/NLA narrative is too lossy.

## Not in scope

- Replacing lens probes with AV verbalization. Probes stay.
- Training new AVs from scratch. First experiment uses kitft's released
  checkpoints on matching base models.
- Whisper-as-evidence in audit logs. Whispers are UX; probe activations
  are the auditable trail.
