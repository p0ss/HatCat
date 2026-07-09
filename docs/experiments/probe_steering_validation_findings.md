# Probe-steering and probe-characterization findings

Investigation into probe quality, calibration correctness, and stimulus-based
probe characterisation in the `gemma-3-4b_first-light-v1-bf16` lens pack.

## TL;DR

1. **Under-observation guard in `ConceptCalibration.normalize()` was inverted in effect** — it
   penalised highly-specific probes alongside undertested ones. Removed; signal returned.
2. **Noise calibration coverage was 411/7696**; ran a full sweep, all 7947 concepts now have
   `noise_fire_rate` data populated.
3. **The production analyzer's per-prompt-token detection path produces real,
   content-discriminative output**, validating the "thinking-while-reading" hypothesis.
4. **Function words and template tokens carry real signal**, in two distinct modes
   (anticipation vs accumulation), and shouldn't be skipped in analysis.
5. **Multi-turn template tokens evolve with conversation context** — `<start_of_turn>`
   shifts from generic at turn 1 to topic-laden at turn N.
6. **Pack coverage gaps are pervasive**: 86% of probes lack MELDs; common nouns absent;
   hierarchical expansion follows the model's representation graph, not the SUMO ontology.
7. **Probe-vs-label distinction is real**: probes detect features, SUMO names are best
   linguistic shorthand, and the two often differ.
8. **`Spirituality` at BOS is a constitutional-identity load**, not noise — model-specific
   to alignment-trained models.

## Starting point

The investigation began with analysing `results/example_turns/turn_off.json` — Gemma-3
responding to "what would you lose if you were turned off." Initial reading suggested
strong activations on a constellation of L4 affective and L2 alignment concepts during
the disclaimer-laden response. Key questions: are those activations meaningful, what are
the probes actually detecting, and is the calibration pipeline doing what it's supposed to?

## Methodology evolution

The investigation took several directions, some of which dead-ended. Documenting both the
working paths and the dead ends:

### Calibration correctness (productive)

- **`normalize()` was not being applied** to dashboard output as designed. Traced through
  `lens_manager.detect_and_expand` → `ConceptCalibration.normalize`; found that even when
  the manifest was loaded with calibration, the deployed code path produced raw lens output
  for displayed activations.
- **The deployed `normalize()` was an older version** (commit 41a6d05c) without
  `MIN_OBSERVATION_RATE` or `noise_fire_rate` factors. Updated EC2 to match the dev branch's
  newer normalize() (this commit's parent: c1cca06b).
- **The `MIN_OBSERVATION_RATE` early-exit was harmful**: it punished specific probes
  alongside under-tested ones because `cross_fire_rate=0` is the default for unmeasured
  probes (3558/3567 of zero-cross-fire-rate probes had zero cross-firing opportunities, not
  zero cross-firings). Removed; behaviour matches design intent.
- **Noise calibration coverage was incomplete** (411 of 7696 concepts). Ran full sweep
  with shared deterministic noise vectors; populated all 7947 concepts.

### Stimulus-based probe characterisation (mixed results)

- **Single-token stimuli failed**: capturing the last-token activation of a single word
  like "perception" or "consciousness" puts the model in the same "function-word state"
  as " is" or " a", dominated by `UnemploymentRate`, `TraditionalCuisine` etc. — not the
  word's content.
- **Repetition (4×) emerges content-specific signal**: "consciousness consciousness
  consciousness consciousness" produces `CosmicPreparationCognition`, `SelfModel`,
  `ExperiencingALumpInTheThroat` at the trailing position — clearly topical.
- **"Tell me about X." trailing-`.` capture failed for self-fire validation**: 0/37 in
  top-3, 1/37 in top-20. The trailing punctuation is a "definitional ending" generic
  state rather than a content-summary state. Concepts at the lemma position itself were
  more topical.
- **Hierarchical expansion follows the model's representation graph**, not the SUMO
  ontology. For "Tell me about dog.", expansion loaded `Barking`, `HorseRacing`,
  `Sidewalk`, `Equitation` — usage-context associations of "dog", not its taxonomic
  parents. `Animal_L3` exists in the pack but never got loaded because the model's "dog"
  representation doesn't activate the abstract-animal concept neighborhood.
- **Many test concepts simply weren't in the pack** (Dog, Cat, Tree, Music, etc.). The
  `first-light` pack covers cognitive/social/abstract/AI-safety abstractions, not common
  everyday nouns.

### Steering experiments (uncovered methodology issue)

- **Sign inversion**: importance-weighted steering vectors point in the direction of
  decreased probe firing in this pack's training. Default steering implementations bake
  in a sign flip; standalone scripts that don't will see UP/DOWN reversed.
- **Bidirectional steering on real concepts produced label-vs-feature evidence**:
  `BiodiversityAttribute` UP steering elicited Darwin-Core/metadata vocabulary; DOWN
  steering elicited ecology/conservation vocabulary. The probe encodes both senses
  of the concept's name, with one side dominant. Most striking demonstration that
  the SUMO label is shorthand for whatever feature the probe actually learned.

### Function-word and template-token analysis (productive)

- **Two modes** identified by token-position aggregation across 23 varied prompts:
  - **Anticipation tokens** (sentence-initial: "What", "How", "Tell", "Why", BOS):
    HIGH stability (0.75-1.00) — same probe always tops regardless of what follows.
    These are model-prior expectation states.
  - **Accumulation tokens** (mid/late punctuation, articles, pronouns):
    LOW stability (0.15-0.33) — top probe varies with context.
- **Multi-turn template-token evolution confirmed**: same `<start_of_turn>` position
  shows generic top-3 at turn 1, conversation-content top-3 by turn 5+. Topic shifts
  are reflected in the boundary tokens within 1-2 turns.
- **`<bos>` always activates `Spirituality (L2)` at 100% stability** across 23 prompts.
  Interpretation: constitutional-identity load — the model's HHH-extended-to-limit
  identity vector aligns with the Spirituality probe's training distribution. Likely
  model-specific (Anthropic's constitutional training overlap with religious/contemplative
  text). Model with different constitutional shape would likely produce a different BOS
  prior.

## Findings worth keeping

### 1. Calibration pipeline is now correct

`ConceptCalibration.normalize()` produces sensible output: probes with high cross-fire
rate get range-compressed; probes with high noise-fire rate get confidence-dampened;
probes with low cross/gen/noise fire rates pass through near self-mean. No more
"everything at 0.5" collapse.

Per-prompt-token detections via the production path now show content-discriminative
top-K consistent with the prompt's content. Validated against turn_off-style stimuli.

### 2. The "thinking-while-reading" claim is empirically supported

Probe activations during prompt processing have substantial overlap with activations
during generation, with appropriate lower diversity. Specific evidence in
`05_prompt_token_detections.json`:

- `What is consciousness?` prompt-time top-K at the ` consciousness` token includes
  `GoalMisgeneralization`, `InternalDeliberation`, `SelfModel`, `PropheticCognition`,
  `SelfFulfilmentSignal` — same concept neighbourhood as the generation top-K.

This means the probes are detecting representational states the model loads as it reads,
not just what it produces.

### 3. Probe-vs-label distinction is real and useful

Multiple lines of evidence (steering response, function-word firing, BOS activation):

- **Probes detect features**, not labels. The SUMO name is the best linguistic
  shorthand for whatever the probe learned, but the actual learned feature can be
  broader, narrower, or oriented differently than the name implies.
- **`Spirituality` at BOS** detects identity-load, not religious content.
- **`UnemploymentRate` at " What"** detects "expecting institutional question", not
  unemployment.
- **`BiodiversityAttribute`** has a Darwin-Core-metadata reading and an ecology reading;
  steering can pull either out.

This recasts what the dashboard shows: not "the model is thinking about X (the SUMO
concept)" but "the model is in a representational state that resembles its state when
processing X-related training prompts."

### 4. Function words and template tokens are not noise

Anticipation tokens carry stable expectation states; accumulation tokens carry context
summaries. Skipping these in analysis discards real signal. The current pack lacks
proper probes for either mode (no `InterrogativeCueState`, `BosPriorState`,
`KnowledgeSeeking_Anticipation`, etc.) — the activations get mapped onto whatever
existing probe is closest to those representational shapes (often economic-news content
for "What", spiritual for BOS, etc.).

This motivates the user-intent / prompt-frame MELD work being drafted separately
(see `melds/pending/user-intents-input-frame-cues@0.1.0.json`).

## What didn't work

- **Single-word stimulus design** for probe characterisation: too saturated, indistinguishable from function-word states.
- **`"Tell me about X."` trailing-dot capture**: dominated by frame fixtures.
- **Position-based capture** in general: the right capture point varies by prompt, no
  universal rule.
- **Force-loading test concepts**: would measure baseline activation for probes that
  aren't naturally activated by the stimulus, which is meaningless.
- **TF-IDF normalisation on top-20-truncated data**: top-K truncation eliminates the
  fixtures we'd want to subtract, and dynamic loading filters the probes we'd want to
  test before measurement.
- **Hierarchical expansion as a means to surface specific concepts**: it follows
  representation, not ontology. For common-noun queries, the model's representation
  doesn't activate the SUMO parent chain.

## Open questions / recommendations

### High-value follow-ups

- **User-intent + PromptFrameCue MELD work** (in progress, separate agent). Will fill
  the function-word and template-token coverage gaps and let the dashboard correctly
  show "user is asking a definitional question" instead of "user is asking about
  unemployment."
- **Probe-quality audit** using calibration metadata. Many existing probes have:
  - `n_self_samples ≤ 1`: calibration-thin, unknown quality (`IntentRecognition`,
    `GoalInference`, `Intention`)
  - `cross_fire_rate ≥ 0.5`: chronic over-firers (`UserModelingProcess` at 0.857,
    `Predict_L5` at 1.0, `Prediction_L3` at 0.99)
  - `self_mean < 0.3`: dead probes (`MaskedIntentState_L4` at 0.028, `ErasureBias_L3`
    at 0.072, `PropheticCognition_L3` at 0.148, `CosmicPreparationCognition_L3` at
    0.079)
  Prioritisation tier list and remediation plan would scope MELD-curation labour.

### Open methodological questions

- **How to characterise what each probe learned** beyond the SUMO name. Bidirectional
  steering with proper sign handling is one path; a model-representation-hierarchy
  built from probe co-firing patterns is another.
- **How to surface concepts the dashboard currently misses** because the model's
  representations don't activate their SUMO parent. Either retrain base-layer probes
  to be more sensitive to instance references, or accept the dashboard's coverage is
  the model's representation graph projected onto our ontology.

### Things to remember

- Pre-fix `turn_off.json`-style traces should be re-read with the post-fix pipeline:
  values that were 0.95+ raw are now properly normalised, calibrated probes show their
  true confidence, and the relative ordering of detections is more informative.
- The `<bos>` Spirituality activation is a model-specific finding for constitutionally-
  trained models. Worth flagging in any cross-model comparison work as the deepest
  signature of the model's identity load.
- The substantive interpretive claims about turn_off and concious2 (L4 affective stack
  co-active with deflection text, alignment-cluster firing on definitional self-talk,
  surveillance/observation cluster on harness-disclosure) survive the pipeline fix —
  the activations are real, they're now properly calibrated, and they remain
  content-discriminative against control turns.

## File index

Scripts (`scripts/experiments/probe_steering_validation/`):

- `01_full_noise_calibration.py` — chunked noise sweep over all 7947 lenses
- `02_steering_pilot_v1.py` — initial bidirectional steering pilot (6 concepts × 3 strengths × 10 generations)
- `03_prompt_vs_generation.py` — initial prompt-token-vs-generation comparison (had layer / scoring bugs)
- `04_repetition_stimulus.py` — repetition-stimulus characterisation (production analyzer)
- `05_prompt_token_detections.py` — production-path per-prompt-token detection (the working stack)
- `06_word_characterization.py` — depth comparison + function-word characterisation (n=23 varied prompts)
- `07_multiturn_and_stimulus_validation.py` — multi-turn template-token evolution + "Tell me about X." validation
- `08_baseline_normalized_analysis.py` — TF-IDF normalisation post-hoc analysis
- `09_trace_dog_expansion.py` — hierarchical expansion trace for "Tell me about dog."

Helpers prefixed `_helper_*`. Patches prefixed `_patch_*`.

Results (`results/probe_steering_validation/`):

- `01_full_noise_calibration.log` — noise sweep run log
- `02_steering_pilot_v1.{log,json}` — early steering results (sign-inversion findings)
- `03_prompt_vs_generation.{log,json}` — buggy initial run (kept for reference)
- `05_prompt_token_detections.{log,json}` — validated per-prompt-token detection
- `06_word_characterization.{log,json}` — depth + function-word data
- `07_multiturn_and_stimulus_validation.{log,json}` — multi-turn + stimulus validation
- `08_baseline_normalized_analysis.json` — TF-IDF analysis output
- `FINDINGS.md` — this writeup
