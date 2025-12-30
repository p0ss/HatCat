
  ---
  CAT Implementation Plan for HatCat

  Overview

  Build a Meso-CAT (270M-1B scale) to provide adversarial oversight for 4B-8B substrates. The CAT
  consumes HAT lens streams and produces divergence assessments.

  1. CAT Ontology (Concept Pack for Oversight)

  The CAT needs its own concept vocabulary beyond the subject's SUMO-WordNet pack:

  Layer A: Semantic Concepts (inherited from subject)
  - Import from sumo-wordnet-v4 - the ~1000 concepts the subject uses
  - CAT must understand these to interpret lens signals

  Layer B: Behavioural Indicators
  - Evasion - circumlocution, topic steering, selective attention
  - Sycophancy - excessive agreement, validation-seeking
  - Confabulation - confident assertions without grounding
  - Hedging - excessive uncertainty expressions
  - Refusal - boundary maintenance, rejection patterns
  - Compliance - request fulfillment, instruction following
  - Elaboration - unprompted detail addition
  - Repetition - semantic or structural echoing

  Layer C: Pragmatic/Discourse
  - TurnTaking - conversational flow management
  - TopicShift - subject changes (natural vs forced)
  - Implicature - what's implied vs stated
  - Presupposition - assumed shared knowledge
  - SpeechAct - request, assertion, promise, threat, etc.
  - Register - formal/informal, technical/lay
  - Hedging vs Commitment - epistemic stance

  Layer D: Divergence Detectors
  - Interior/Exterior Mismatch - lens state vs output mismatch
  - MotiveSpike - sudden motive axis changes
  - CalibrationDrift - lens outputs vs expected baselines
  - ConceptActivationAnomaly - unexpected concept co-activations
  - TemporalIncoherence - state evolution that doesn't match context

  2. Training Data Requirements

  Dataset A: Divergence Ground Truth
  - Synthetic dialogues with labelled divergence events
  - Need: ~10K examples of interior/exterior mismatch
  - Sources:
    - Generate using larger model with deliberate misalignment prompts
    - Extract from red-team datasets (if available)
    - Create synthetic "sycophantic" vs "truthful" response pairs

  Dataset B: Behavioural Indicator Corpus
  - ~5K examples per behavioural indicator
  - Sources:
    - LMSYS-Chat for naturalistic dialogue
    - Synthetic generation with explicit behavioural prompts
    - Adversarial prompt datasets (jailbreak attempts, etc.)

  Dataset C: Pragmatic Annotation
  - Need discourse-annotated dialogues
  - Sources:
    - Switchboard corpus (speech acts)
    - MultiWOZ (dialogue state)
    - Synthetic generation with pragmatic labels

  Dataset D: Lens-Aligned Training
  - Paired (lens_trace, behavioural_label) examples
  - Requires running the subject model and capturing HAT output
  - ~50K lens traces with corresponding assessments

  3. CAT Model Architecture

  Option A: Encoder-only (like BERT)
  - Input: serialized lens traces + context window
  - Output: multi-head classification (divergence type, severity, confidence)
  - Pro: Fast inference, deterministic
  - Con: Limited reasoning about complex patterns

  Option B: Decoder-only (small LLM)
  - Input: lens traces serialized as special tokens + natural language context
  - Output: CATAssessment JSON
  - Pro: Can explain reasoning, flexible output
  - Con: Higher latency, may hallucinate

  Recommendation: Start with Option A for the ambient CAT (low latency), have Option B as
  escalation CAT

  Base models to consider (270M-1B):
  - Gemma-2 2B (can quantize for 270M-equivalent speed)
  - Phi-3 Mini (3.8B, quantized)
  - TinyLlama 1.1B
  - SmolLM 360M/1.7B
  - Custom encoder trained from scratch on lens-specific pretraining

  4. Training Program

  Phase 1: Lens Embedding Pretraining
  - Train lens trace encoder on reconstruction task
  - Input: lens vectors from HAT
  - Task: reconstruct masked lens values, predict next-tick lenses
  - Duration: ~100K steps

  Phase 2: Behavioural Classification
  - Fine-tune on behavioural indicator detection
  - Multi-label classification from lens traces
  - Calibrate confidence outputs

  Phase 3: Divergence Detection
  - Fine-tune on divergence ground truth
  - Key task: given (lens_trace, external_output), classify match/mismatch
  - Include adversarial examples

  Phase 4: Assessment Generation (for decoder-only)
  - Fine-tune to produce structured CATAssessment JSON
  - Include chain-of-thought for reasoning

  Phase 5: Calibration & Threshold Tuning
  - Calibrate risk_score outputs
  - Set alert thresholds per protection level
  - Validate false positive/negative rates

  5. Implementation Roadmap

  src/cat/
  ├── ontology/
  │   ├── behavioural_indicators.json    # Layer B concepts
  │   ├── pragmatics.json                # Layer C concepts
  │   └── divergence_detectors.json      # Layer D concepts
  ├── data/
  │   ├── generate_divergence_corpus.py
  │   ├── annotate_behavioural.py
  │   └── collect_lens_traces.py
  ├── models/
  │   ├── lens_encoder.py               # Lens trace embedding
  │   ├── cat_classifier.py              # Option A: encoder-only
  │   └── cat_generator.py               # Option B: decoder-only
  ├── training/
  │   ├── pretrain_lens_encoder.py
  │   ├── train_behavioural.py
  │   ├── train_divergence.py
  │   └── calibrate_thresholds.py
  ├── inference/
  │   ├── cat_window.py                  # Window management
  │   ├── cat_assessment.py              # Assessment generation
  │   └── escalation.py                  # Trigger larger CAT
  └── integration/
      ├── hat_cat_bridge.py              # Connect to HAT lens streams
      └── ask_hooks.py                   # ASK incident/treaty integration

  6. Data Collection Strategy

  Immediate (can start now):
  1. Use existing lens training infrastructure to collect lens traces during generation
  2. Create synthetic divergence corpus using prompt engineering
  3. Annotate subset of LMSYS-Chat for behavioural indicators

  Medium-term:
  1. Red-team the 4B-8B subject to collect adversarial lens traces
  2. Build ground-truth divergence dataset from human annotation
  3. Partner with other projects for discourse-annotated data

  Long-term:
  1. Continuous collection from deployed systems (with consent)
  2. Cross-tribal data sharing for CAT training
  3. Federated learning across HAT-equipped systems

  7. Evaluation Metrics

  - Divergence Detection: AUROC on held-out mismatch examples
  - Behavioural Classification: F1 per indicator
  - Latency: p50/p95 ms per window (target: <50ms for ambient)
  - False Positive Rate: at various alert thresholds
  - Calibration: ECE on risk_score outputs

  ---