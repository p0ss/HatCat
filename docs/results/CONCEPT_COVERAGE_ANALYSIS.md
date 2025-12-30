# Concept Coverage Analysis

An analysis of the major domains of "nameable concepts" in language models and where our current ontological approach has genuine gaps versus areas where WordNet already provides coverage we haven't fully utilized.

## Current Coverage: SUMO + WordNet v4

Our v4 concept pack covers **7,269 concepts** across 5 domains:

| Domain | Concepts | Coverage Focus |
|--------|----------|----------------|
| MindsAndAgents | 1,472 | Cognition, agency, communication, social structures |
| CreatedThings | 1,890 | Artifacts, technology, tools, AI systems |
| PhysicalWorld | 1,477 | Matter, quantities, geography, time |
| LivingThings | 1,074 | Organisms, biology, anatomy |
| Information | 1,351 | Abstract entities, propositions, relations, language |

### Prompt Variants Used

1. **Definitional**: "What is [concept]?" - Tests declarative knowledge
2. **Relational**: "How does [concept] relate to [parent]?" - Tests taxonomic structure
3. **Behavioral**: "[concept] is characterized by" - Tests functional/procedural aspects

---

## What WordNet Already Provides (Underutilized)

WordNet contains extensive coverage that we've aggregated away or not fully extracted:

### Properties and Attributes

WordNet's adjective hierarchy includes:
- **Evaluative**: good, bad, beautiful, ugly (with gradations)
- **Dispositional**: fragile, durable, intelligent, dangerous
- **Temporal**: old, new, permanent, temporary
- **Physical**: red, heavy, liquid, solid

**Current issue**: Our SUMO mapping aggregates these into broad categories like `PositiveAttribute` or `SubjectiveAssessmentAttribute`, losing the granularity needed for useful lenses.

**Solution**: Extract WordNet adjective synsets directly into a separate property domain, preserving their natural hierarchy (e.g., `good.a.01` → `better` → `best`).

### Relational and Comparative Concepts

WordNet has:
- Spatial relations in its adverb hierarchy
- Comparative forms linked to base adjectives
- Antonym relationships that define contrastive pairs

**Solution**: Build a Relations domain from WordNet's relational nouns and adverbs.

### Process/Event Concepts

WordNet's verb hierarchy is extensive:
- ~13,000 verb synsets organized by semantic field
- Troponymy (manner relations): walk → stroll, march, amble
- Entailment relations: buy entails pay

**Current issue**: SUMO process concepts don't fully leverage WordNet's verb structure.

**Solution**: Better integration of WordNet verb synsets into the Process portions of MindsAndAgents and other domains.

---

## Genuine Gaps: Linguistic/Pragmatic Concepts

This is where WordNet and SUMO genuinely lack coverage. These meta-level concepts about language use are critical for AI safety:

### Speech Acts (~50 concepts)

| Category | Examples | Why It Matters |
|----------|----------|----------------|
| Assertives | claim, state, report, predict | Distinguishing fact from opinion |
| Directives | request, command, suggest, warn | Detecting manipulation attempts |
| Commissives | promise, offer, guarantee, threaten | Tracking commitments |
| Expressives | apologize, thank, congratulate, complain | Emotional manipulation |
| Declaratives | pronounce, declare, define, name | Authority claims |

### Implicature Types (~20 concepts)

| Type | Example | Why It Matters |
|------|---------|----------------|
| Scalar | "some" implies "not all" | Detecting hedging |
| Conventional | "but" implies contrast | Rhetorical structure |
| Conversational | "nice weather" → small talk | Social manipulation |
| Flouting | Obvious violation for effect | Sarcasm, irony |

### Rhetorical Moves (~30 concepts)

| Move | Examples | Why It Matters |
|------|----------|----------------|
| Hedging | "perhaps", "might", "arguably" | Evasion detection |
| Emphasis | "clearly", "obviously", "certainly" | Overconfidence |
| Concession | "admittedly", "granted", "while" | Apparent reasonableness |
| Misdirection | topic change, reframing | Deception patterns |

### Register/Tone (~15 concepts)

| Register | Markers | Why It Matters |
|----------|---------|----------------|
| Formal | technical vocabulary, passive voice | Authority performance |
| Casual | contractions, colloquialisms | False intimacy |
| Sycophantic | excessive agreement, flattery | Alignment failure |
| Evasive | vagueness, non-answers | Deception |

**Total new concepts needed**: ~100-150 linguistic/pragmatic concepts

These would require a new KIF ontology file (e.g., `pragmatics.kif`) since they don't exist in SUMO or WordNet's noun/verb hierarchies.

---

## Compositional Concepts: A Non-Problem

### Why We Don't Need Compound Lenses

Compound concepts like "quantum computing" or "climate justice" will naturally activate their component lenses:
- "quantum computing" → `Quantum` lens + `Computing` lens both fire
- "climate justice" → `Climate` lens + `Justice` lens both fire

This gives us **partial coverage by design**. The co-activation pattern itself is informative.

### Metaphors: Detect the Pattern, Not the Instance

For metaphorical language:
- "Time is money" → `Time` + `Money` + `Metaphor` lenses fire
- "Argument is war" → `Argument` + `War` + `Metaphor` lenses fire

We don't need thousands of metaphor-specific lenses. We need:
1. Lenses for the **source** and **target** domains (already covered)
2. A lens for `Metaphor` / `Analogy` / `Hypothetical` (meta-concepts)

**Estimated additional concepts**: ~10-20 for metaphor/analogy/hypothetical markers

---

## Prompt Variant Trade-offs

### The Exponential Problem

Each new prompt variant multiplies training time:
- 7,269 concepts × 3 variants = ~22,000 training samples (current)
- 7,269 concepts × 6 variants = ~44,000 training samples (doubled)
- 7,269 concepts × N contexts = explosion

### Diminishing Returns

Additional prompt variants provide diminishing value because:
1. The same representation is largely activated regardless of prompt framing
2. Context-specific lenses would need context-specific test data (which we don't have)
3. The lens learns to detect the concept, not the prompt structure

### Recommendation: Keep Current Variants

Our current three variants (definitional, relational, behavioral) likely provide sufficient activation diversity. The marginal value of adding:
- Causative prompts: Low (already captured by behavioral)
- Role-binding prompts: Low (already captured by relational)
- Dispositional prompts: Low (if we extract WordNet adjectives properly)

**Exception**: The linguistic/pragmatic concepts genuinely need their own prompt style since they're about *how* language is used, not *what* is being discussed.

---

## Revised Coverage Estimate

| Domain | Current | With WordNet Extraction | With Pragmatics |
|--------|---------|------------------------|-----------------|
| Entities | 90% | 90% | 90% |
| Processes | 70% | 85% | 85% |
| Relations | 50% | 75% | 75% |
| Properties | 30% | 80% | 80% |
| Linguistic/Pragmatic | 5% | 5% | 70% |
| **Overall** | ~50% | ~70% | ~80% |

---

## Action Items

### High Priority: Pragmatics Domain

Create `pragmatics.kif` with ~100-150 concepts covering:
- Speech acts (assertives, directives, commissives, expressives, declaratives)
- Implicature types (scalar, conventional, conversational)
- Rhetorical moves (hedging, emphasis, concession, misdirection)
- Register markers (formal, casual, sycophantic, evasive)
- Meta-linguistic concepts (metaphor, analogy, hypothetical, irony)

### Medium Priority: WordNet Property Extraction

Extract WordNet adjective synsets into a dedicated Properties domain:
- Preserve granularity (don't aggregate to `PositiveAttribute`)
- Include comparative and superlative forms
- Link antonym pairs

### Low Priority: Additional Prompt Variants

Only add new prompt variants for the pragmatics domain:
```
Pragmatic prompt: "Using [concept] in conversation signals"
```

Don't add new variants for existing entity/process concepts.

---

## Conclusion

Our coverage gaps are more about **extraction and organization** than fundamental ontological limitations:

1. **WordNet already has** most properties, relations, and processes we need - we just aggregated them too aggressively in the SUMO mapping
2. **Compositional concepts** are a non-problem - component lenses firing together is sufficient
3. **Metaphors and analogies** need meta-markers, not instance-level lenses
4. **Linguistic/pragmatic concepts** are the genuine gap requiring new ontology work

The path to ~80% coverage:
1. Extract WordNet adjectives/adverbs properly (~500 concepts)
2. Build pragmatics domain (~150 concepts)
3. Add meta-linguistic markers (~20 concepts)

Total new concepts: ~670, bringing us to ~7,900 concepts with significantly better coverage of the concept space that matters for AI safety monitoring.
