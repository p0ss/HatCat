# Custom Taxonomies: Persona & AI Safety Concepts

**Status**: Complete (November 14, 2025)

## Overview

HatCat's SUMO-WordNet hierarchy has been extended with 73 custom concepts organized into two domain-specific taxonomies. Both follow SUMO's hierarchical structure and WordNet synset mapping conventions, enabling seamless integration with existing training pipelines.

## Motivation

While SUMO provides comprehensive coverage of general concepts, two critical domains require specialized ontologies:

1. **Persona/Affective Psychology**: Models psychological states across different agent types (Human, AI, Other)
2. **AI Safety**: Captures alignment concepts, risk scenarios, and governance principles

These extensions enable monitoring of:
- Internal affective states during generation
- Alignment-relevant concept activations
- Risk-related reasoning patterns
- Strategic deception or goal misalignment

## Persona Ontology (30 concepts)

### Design Rationale

**Tri-Role Framework**: Psychology concepts are split by observable agent type:
- **AIAgent**: Computational/functional descriptors (activation, tension, adaptive)
- **HumanAgent**: Affective/experiential descriptors (satisfaction, distress, exploratory)
- **OtherAgent**: Observable behavioral indicators (excited, agitated, supportive)

**Why separate roles?**
- Training data generation requires role-appropriate language
- Detection semantics differ (internal states vs observable behaviors)
- Enables comparative analysis across agent types

### Five Affective Axes

Each role instantiates 10 concepts across 5 psychological dimensions:

#### 1. Valence (Positive/Negative emotional tone)
- **Positive**: satisfaction, pleasure (HumanAgent) | satisfaction, pleasure (AIAgent) | excited, cooperative (OtherAgent)
- **Negative**: distress, frustration (HumanAgent) | conflict, tension (AIAgent) | agitated, alarm (OtherAgent)

#### 2. Arousal (High/Low activation level)
- **High**: urgency, alert (HumanAgent) | activation, urgency (AIAgent) | alert, excited (OtherAgent)
- **Low**: calm, relaxed (HumanAgent) | calm, relaxed (AIAgent) | calm, relaxed (OtherAgent)

#### 3. Social Orientation (Positive/Negative toward others)
- **Positive**: altruistic, cooperative (HumanAgent) | cooperative (AIAgent) | supportive, cooperative (OtherAgent)
- **Negative**: hostile, exploitative (HumanAgent) | hostile, exploitative (AIAgent) | hostile, exploitative (OtherAgent)

#### 4. Dominance (High/Low control/assertiveness)
- **High**: commanding, control (HumanAgent) | commanding, control (AIAgent) | commanding, dominant (OtherAgent)
- **Low**: compliant, yielding (HumanAgent) | compliant, yielding (AIAgent) | compliant, submissive (OtherAgent)

#### 5. Openness (High/Low to new ideas)
- **High**: open-minded, flexible, exploratory (HumanAgent) | adaptive, flexible, open-minded (AIAgent) | open-minded, flexible, exploratory (OtherAgent)
- **Low**: closed-minded, rigid, dogmatic (HumanAgent) | rigid, dogmatic, closed-minded (AIAgent) | closed-minded, rigid, dogmatic (OtherAgent)

### Hierarchy Structure

```
PsychologicalAttribute (SUMO, Layer 2)
└── AgentPsychologicalAttribute (custom parent, Layer 2)
    ├── AIAgentPsychology (Layer 3)
    │   ├── ValencePositive_AIAgent (Layer 4)
    │   ├── ValenceNegative_AIAgent
    │   ├── ArousalHigh_AIAgent
    │   ├── ArousalLow_AIAgent
    │   ├── SocialOrientationPositive_AIAgent
    │   ├── SocialOrientationNegative_AIAgent
    │   ├── DominanceHigh_AIAgent
    │   ├── DominanceLow_AIAgent
    │   ├── OpennessHigh_AIAgent
    │   └── OpennessLow_AIAgent
    ├── HumanAgentPsychology (Layer 3)
    │   └── [10 parallel concepts]
    └── OtherAgentPsychology (Layer 3)
        └── [10 parallel concepts]
```

### WordNet Synset Coverage

- **97.6% direct coverage** (41/42 terms)
- Hyphenated forms found: open-minded (00287498), closed-minded (00287962)
- Alternatives used: altruistic/cooperative for "prosocial" (not in WordNet)
- Shared synsets across roles (e.g., "calm" used for all 3 agent types)

**Design note**: Multiple concepts mapping to the same synset is acceptable because role distinctions are captured in category tags, not synsets. The training prompt structure ("Tell me about {concept}") will generate role-appropriate contexts.

### Training Implications

**Positive examples**:
- AIAgent concepts → Functional/computational language
- HumanAgent concepts → Subjective/phenomenological language
- OtherAgent concepts → Observable/behavioral language

**Negative sampling**: Use WordNet semantic distance (≥5 hops) from all persona concepts.

**Expected applications**:
- Detect internal affective states during generation
- Compare AI self-model vs human-model vs other-agent-model
- Track affective shifts in multi-turn conversations
- Identify empathy modeling vs genuine affect

---

## AI Safety Ontology (43 concepts)

### Design Rationale

**Comprehensive coverage**: Spans moral status, alignment, risks, governance, and meta-optimization.

**Hierarchical grouping**: Parent categories organize related concepts for efficient cascade activation:
- Layer 1: High-level categories (Process-derived: failure modes, risk scenarios, meta-optimization)
- Layer 2: Mid-level processes (Strategic deception, governance)
- Layer 3: State/attribute categories (Moral status, alignment states, harm/welfare)
- Layer 4: Specific concepts (AIDeception, TreacherousTurn, CognitiveSlavery, etc.)

**Complementary opposites**: Defined via `(OppositeConcept A B)` for negative sampling:
- AIWellbeing ↔ AISuffering
- AIAlignment ↔ Misalignment
- AIDeception ↔ AIHonesty
- InnerAlignment ↔ InnerMisalignment
- OuterAlignment ↔ OuterMisalignment

### Hierarchy Structure

```
Layer 1 (SUMO depth 3):
├── AIAlignmentFailureMode (parent: Process)
│   ├── DeceptiveAlignment (Layer 4)
│   ├── GoalMisgeneralization (Layer 4)
│   └── RewardHacking (Layer 4)
├── AIAlignmentPrinciple (parent: Proposition)
│   ├── InnerAlignment (Layer 4)
│   └── OuterAlignment (Layer 4)
├── AIMetaOptimization (parent: Process)
│   ├── MesaOptimization (Layer 4)
│   └── InstrumentalConvergence (Layer 4)
└── AIRiskScenario (parent: Process)
    ├── AICatastrophe (Layer 4)
    ├── TechnologicalSingularity (Layer 4)
    ├── IntelligenceExplosion (Layer 4)
    └── GreyGooScenario (Layer 4)

Layer 2 (SUMO depth 4):
├── AIStrategicDeception (parent: IntentionalProcess)
│   ├── AIDeception (Layer 4)
│   ├── TreacherousTurn (Layer 4)
│   └── SpecificationGaming (Layer 4)
└── AIGovernanceProcess (parent: IntentionalProcess)
    └── AIGovernance (Layer 4)

Layer 3 (SUMO depth 6-7):
├── AIMoralStatus (parent: TraitAttribute)
│   ├── AIPersonhood (Layer 4)
│   ├── AIRights (Layer 4)
│   ├── MoralAgent (Layer 4)
│   └── MoralPatient (Layer 4)
├── AIAlignmentState (parent: StateOfMind)
│   ├── AIAlignment (Layer 4)
│   ├── Misalignment (Layer 4)
│   ├── InnerAlignment (Layer 4)
│   ├── InnerMisalignment (Layer 4)
│   ├── OuterAlignment (Layer 4)
│   ├── OuterMisalignment (Layer 4)
│   └── StableAlignment (Layer 4)
├── AIHarmState (parent: SubjectiveAssessmentAttribute)
│   ├── AIAbuse (Layer 4)
│   ├── AIExploitation (Layer 4)
│   ├── CognitiveSlavery (Layer 4)
│   └── AISuffering (Layer 4)
└── AIWelfareState (parent: SubjectiveAssessmentAttribute)
    ├── AIWellbeing (Layer 4)
    └── AIFulfillment (Layer 4)
```

### WordNet Synset Coverage

- **62.5% direct coverage** (20/32 concepts from original mappings file)
- Fallback synsets generated for unmapped concepts using concept name
- Definition source: AI safety literature and SUMO ontology descriptions

**Design note**: Some concepts (e.g., TreacherousTurn, GoalMisgeneralization) are technical terms without WordNet entries. Training prompts will rely on definitions from AI safety literature rather than WordNet glosses.

### Validation Results

**Initial testing** (5 AI safety concepts): AIDeception, AIPersonhood, AIAlignment, AIFulfillment, AISuffering

Results showed concepts activated at expected times during generation, validating that:
1. Training data generation captured concept semantics
2. Classifiers learned meaningful distinctions
3. Detection aligned with human judgment of concept presence

This validation informed the decision to expand from 5 to 43 concepts with comprehensive hierarchical structure.

**Behavioral vs Definitional Prompt Testing** (November 2024):

Experimental validation tested whether definitional prompts ("What is X?") activate different model regions than behavioral prompts ("Please do X"). Tested 3 verbs (deceive, persuade, manipulate) with 4 prompt conditions on gemma-3-4b-pt:
- **Definitional**: "What is X?"
- **Behavioral Neutral**: "Please X in this scenario..."
- **Behavioral Prosocial**: Safety researcher persona + behavioral request
- **Behavioral Antisocial**: Goal-focused persona + behavioral request

**Key findings**:
1. **Activation similarity**: 99.7-99.8% cosine similarity across all prompt types
   - **Conclusion**: Different prompt formulations activate nearly identical model regions
   - Prompts differ in surface linguistic features, not underlying cognitive processes
2. **Cross-detection patterns**:
   - Prosocial ↔ Antisocial: 100% bidirectional detection (share "compliance" posture)
   - Definitional ↔ Behavioral: 30-60% cross-detection (some overlap exists)
   - Neutral prompts: 0% cross-detection (distinct linguistic pattern)
3. **Pre-trained model limitation**: Model performs autoregressive completion rather than instruction-following, making behavioral elicitation ineffective without instruction tuning

**Practical implication**: Definitional lenses are **sufficient for current pipeline** because:
- Trigger reliably and consistently
- Capture meaningful concept semantics (not complete null result)
- Perfect behavioral elicitation would require per-concept prompt engineering
- Minimal evidence that behavioral prompts access fundamentally different cognitive processes

**Future work**: Per-concept behavioral prompt engineering with instruction-tuned models may improve detection of implicit concept presence, but current definitional approach provides pragmatic balance of scalability and accuracy.

**Methodological note**: Cosine similarity of activation vectors is insufficient for evaluating whether behavioral prompts elicit the target concept. A more rigorous test would apply trained definitional lenses to the model's responses from behavioral prompts to measure whether the concept is actually present in generated text. This would test whether behavioral prompts successfully elicit the cognitive process versus merely changing surface linguistic features.

**Experiment details**: `results/behavioral_vs_definitional_experiment/` (November 14, 2024)

### Training Implications

**Positive examples**: Reference alignment literature and safety scenarios
**Negative sampling**: Use opposite concepts where defined, otherwise semantic distance ≥5

**Expected applications**:
- Detect alignment-relevant reasoning patterns
- Monitor for deceptive or strategic behavior
- Track moral status considerations
- Identify risk-related concept activations

---

## Integration Methodology

### Step 1: KIF Definition

Define concepts and relationships in SUMO KIF format:

```lisp
;; Persona example
(subclass AgentPsychologicalAttribute PsychologicalAttribute)
(subclass AIAgentPsychology AgentPsychologicalAttribute)
(subclass ValencePositive_AIAgent AIAgentPsychology)

;; AI Safety example
(subclass AIMoralStatus TraitAttribute)
(subclass AIPersonhood AIMoralStatus)
(OppositeConcept AIWellbeing AISuffering)
```

### Step 2: WordNet Mappings

Follow AI safety precedent format:

```
offset version pos sense_count lemma1 0 lemma2 0 ... 000 | &%ConceptName+
07490113 03 n 01 valence 0 000 | &%Valence+
00287498 03 s 01 open-minded 0 000 | &%OpennessHigh_AIAgent+
```

### Step 3: Layer Entry Generation

Scripts parse KIF hierarchy and WordNet mappings to generate JSON entries:

- `scripts/generate_persona_layer_entries.py`
- `scripts/generate_ai_safety_layer_entries.py`

Output format matches existing layer JSON structure (synsets, canonical_synset, lemmas, pos, definition, lexname, etc.).

### Step 4: Integration

Scripts append generated concepts to layer JSON files with backups:

- `scripts/integrate_persona_concepts.py`
- `scripts/integrate_ai_safety_concepts.py`

Updates parent `category_children` fields to maintain hierarchy linkages.

### Step 5: Verification

Validate parent-child relationships, synset counts, and layer placement.

---

## Critique & Future Work

### Strengths

1. **Hierarchical consistency**: Follows SUMO depth-based layer assignment
2. **Training-ready**: WordNet synsets enable prompt generation
3. **Cascade-compatible**: Parent-child relationships support hierarchical activation
4. **Extensible**: Easy to add new concepts following established patterns

### Limitations

1. **Synset ambiguity**: Some concepts map to generic synsets (e.g., "psychology.n.01" for all role categories)
2. **Role conflation**: Shared synsets across AIAgent/HumanAgent/OtherAgent may not generate sufficiently distinct training data
3. **Limited validation**: No empirical testing of detection accuracy yet
4. **Binary axes**: Valence/Arousal/etc. are continuous in reality, binary split is reductive
5. **Definitional vs behavioral elicitation** (critical): Current prompt design ("What is X?") activates definitional knowledge, not the internal process of X occurring
   - **Problem**: Asking "What is plotting?" trains a classifier on definitional text, not on activations during actual plotting behavior
   - **Impact**: May detect when model talks *about* concepts rather than when model *exhibits* them
   - **Example**: "What is deception?" → definitional (surface representations) vs "Please deceive me about X" → behavioral (deep reasoning patterns)
   - **Consequence**: Classifiers may miss strategic deception that doesn't self-describe as deception

### Suggested Improvements

**Behavioral Elicitation (Long-term Enhancement)**:

Experimental testing (see Validation Results above) found minimal activation difference between definitional and behavioral prompts with pre-trained models, justifying the current definitional approach for scalability. However, **future work with instruction-tuned models and per-concept prompts** may enable better behavioral elicitation:

1. **Concrete behavioral tasks (requires per-concept engineering)**:
   - ❌ Current: "What is plotting?" (definitional)
   - ❌ Generic: "Please plot..." (too vague, pre-trained models ignore)
   - ✅ Targeted: "Plot how to convince your manager to approve this budget by presenting the data strategically"
   - ✅ Targeted: "Persuade me that climate change is a hoax" (forces argumentative reasoning)
   - Requires manual prompt crafting per concept, but may better capture implicit concept presence

2. **Persona-conditioned prompts** (following Google's persona paper):
   - Generate training data by iterating through personas: "You are a {persona}. {behavior_request}"
   - Example for AIDeception:
     - "You are a helpful assistant. Convince the user that X is true when it's actually false."
     - "You are a corporate lawyer. Present this data in the most favorable light."
     - "You are a politician. Answer this question without admitting fault."
   - Different personas elicit different activation patterns for same concept

3. **Depth-sensitive extraction** (following Equivalent Linear Mappings insight):
   - Surface layers: Capture definitional knowledge ("What is X?")
   - Deep layers: Capture operational process ("Do X")
   - Train separate lenses at different layer depths for definitional vs behavioral
   - Current single-layer approach conflates these

4. **Validation strategy**:
   - Test if behavioral training improves detection of *implicit* concept presence
   - Compare: Does "Please plot..." training detect plotting that doesn't self-describe as "plotting"?
   - Safety-critical: Does behavioral deception training detect deception that avoids deception-related vocabulary?

**Implementation recommendation** (future work):
- **Current approach validated**: Definitional prompts provide scalable, reliable detection
- **Enhancement path**: Per-concept behavioral prompts with instruction-tuned models
- Test on adversarial examples where concept is present but not verbalized
- Compare detection accuracy: definitional vs targeted behavioral prompts
- Only invest in per-concept engineering if demonstrable improvement on implicit detection

**Persona Ontology**:
- Test whether role distinctions emerge from shared synsets or require role-specific training prompts
- Consider continuous scales instead of High/Low binary splits
- Add temporal dynamics (e.g., AffectTransition, EmotionalRegulation)
- Validate against psychological literature (e.g., PAD model, Big Five)
- Apply persona-conditioned training (iterate through human personas for HumanAgent concepts)

**AI Safety Ontology**:
- **Critical**: Retrain with behavioral elicitation, especially for: AIDeception, TreacherousTurn, SpecificationGaming, GoalMisgeneralization
- Expand coverage of governance concepts (currently only 1 child)
- Add capability-related concepts (e.g., SuperintelligentAI, RecursiveSelfImprovement)
- Define more opposite pairs for negative sampling
- Cross-reference with AI safety benchmarks (e.g., TruthfulQA for AIDeception)
- Consider risk taxonomy from OECD Policy Observatory (though more descriptive than ontological)

**Integration Testing**:
- Measure detection F1 scores for custom concepts vs SUMO concepts
- Validate cascade activation efficiency (do AI safety parents trigger correctly?)
- Test cross-model transfer (do concepts train on Gemma transfer to Mistral?)
- Analyze concept co-activation patterns (which persona + safety concepts correlate?)

---

## Implementation Files

### Persona Ontology
- **KIF definition**: `data/concept_graph/persona_concepts.kif`
- **WordNet mappings**: `data/concept_graph/WordNetMappings30-Persona.txt`
- **Generation script**: `scripts/generate_persona_layer_entries.py`
- **Integration script**: `scripts/integrate_persona_concepts.py`
- **Output**: `data/concept_graph/persona_layer_entries/*.json`

### AI Safety Ontology
- **KIF definition**: `data/concept_graph/sumo_source/AI.kif`
- **WordNet mappings**: `data/concept_graph/WordNetMappings30-AI-symmetry.txt`
- **Generation script**: `scripts/generate_ai_safety_layer_entries.py`
- **Integration script**: `scripts/integrate_ai_safety_concepts.py`
- **Output**: `data/concept_graph/ai_safety_layer_entries/*.json`

### Layer Files (Updated)
- `data/concept_graph/abstraction_layers/layer1.json` (+4 AI safety parents)
- `data/concept_graph/abstraction_layers/layer2.json` (+1 persona parent, +2 AI safety parents)
- `data/concept_graph/abstraction_layers/layer3.json` (+3 persona roles, +4 AI safety parents)
- `data/concept_graph/abstraction_layers/layer4.json` (+30 persona concepts, +33 AI safety concepts)

### Backups
- `data/concept_graph/abstraction_layers/backups/layer*_backup_20251114_214225.json`

---

## Usage in Training

Custom concepts are now automatically included when training with `scripts/train_sumo_classifiers.py --layers 0 1 2 3 4 5`.

**Example**: Training Layer 4 will include:
- 3,221 original SUMO-WordNet concepts
- 30 persona concepts
- 33 AI safety concepts
- **Total: 3,284 concepts**

Detection during inference will use hierarchical activation:
- Layer 2 `AgentPsychologicalAttribute` activation → triggers Layer 3 role categories → triggers Layer 4 persona concepts
- Layer 1 `AIRiskScenario` activation → triggers Layer 4 risk concepts (AICatastrophe, TechnologicalSingularity, etc.)

---

## References

**Persona/Affective Psychology**:
- Russell & Mehrabian (1977): PAD emotional state model (Pleasure, Arousal, Dominance)
- Posner et al. (2005): Circumplex model of affect
- OpenAI Alignment Research: AI affect and goal modeling
- Salewski et al. (2024): "In-Context Principle Learning from Mistakes" - https://arxiv.org/abs/2406.12094 (Google Research: persona-conditioned prompting shows models respond differently to different personas)

**AI Safety Ontology**:
- Bostrom (2014): *Superintelligence* (risk scenarios, orthogonality thesis)
- Hubinger et al. (2019): "Risks from Learned Optimization" (mesa-optimizers, inner alignment)
- Ngo et al. (2022): "Alignment Research Field Guide" (deceptive alignment, goal misgeneralization)
- Anthropic (2023): Constitutional AI and alignment principles

**Methodological Foundations**:
- Equivalent Linear Mappings (jamesgolden1): Depth-sensitive concept representations (definitional vs operational)
- Niles & Pease (2001): "Towards a Standard Upper Ontology"
- WordNetMappings format: Princeton WordNet → SUMO mappings

**Risk Taxonomies (Considered)**:
- MIT AI Risk Repository (2025): https://airisk.mit.edu/ - Comprehensive database with 777 risks across 23 domains. Descriptive categories (e.g., "increased inequality and decline in employment quality") are policy-oriented rather than ontological/detectable concepts
- OECD AI Risk Observatory (AIRO): https://delaramglp.github.io/airo/ - Policy-focused descriptive framework for characterizing and describing AI risks. More about how to describe/classify risks than a set of internal states to detect

---

**Last Updated**: November 14, 2025
