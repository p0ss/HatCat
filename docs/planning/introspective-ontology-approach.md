# Introspective Ontology Approach

## Goal

Map the internal concept space of a language model (Gemma 3 4B) by having it generate a hierarchical taxonomy of its own knowledge and capabilities. This self-map becomes the basis for training interpretability "lenses" that can detect when specific concepts are active in the model's representations.

## Why Introspective (vs Human-Centric)

Our first approach used an "alien ontologist" prompt that mapped human activity through an Action & Agency lens. This produced 13 top-level fields like "Violence & Conflict", "Economic Activity", "Care Giving & Domestic Labor" - organized around what humans DO.

**Problem:** We're not mapping humans, we're mapping the model. A human-centric ontology may not align with how the model actually organizes its internal representations.

**Solution:** Ask the model to introspect - to categorize everything it knows and does from its own perspective. This produces a functional taxonomy organized by HOW the model processes information, not just what it knows about.

## The Introspective Prompt

Located at: `melds/helpers/introspective_ontologist_prompt.txt`

Key framing:
- "You are a language model tasked with mapping the entirety of your own knowledge and capabilities"
- Organize by "how knowledge is structured and used within you"
- "What patterns of reasoning do you employ? What types of knowledge do you draw upon?"

Coverage requirements span: factual knowledge, language/communication, reasoning/logic, creative generation, code/formal systems, social understanding, procedural knowledge, meta-cognition, domain expertise, cultural knowledge.

## Gemma's Self-Map (12 Faculties)

When asked to introspect, Gemma 3 4B produced 12 top-level "Faculties":

| Faculty | Description |
|---------|-------------|
| **Dynamic Knowledge Graph Construction** | How it organizes and connects information |
| **Statistical Pattern Recognition & Inference** | Bayesian reasoning, extrapolation from patterns |
| **Linguistic Modeling & Generation** | Syntax, semantics, text generation |
| **Massive Factual Retrieval & Organization** | The knowledge substrate (admits hallucinations) |
| **Creative Simulation & Hypothetical Reasoning** | Stories, ideas, imagination |
| **Code Execution & Understanding** | Symbolic code comprehension |
| **Social Context Prediction & Response** | Theory of mind, social dynamics |
| **Instruction Following & Protocol Adherence** | Alignment, following commands |
| **Domain-Specific Schema Access** | Medicine, law, engineering clusters |
| **Cultural Artifact Representation** | Art, music, literature, norms |
| **Probabilistic Uncertainty Estimation** | Confidence calibration |
| **Internal State Reporting and Limitations** | Meta-cognition, self-awareness |

This is notably honest and architectural - it acknowledges hallucination-prone areas, describes confidence estimation as "imperfect", and separates what it DOES (instruction following) from what it KNOWS (factual retrieval).

## Hierarchy Structure

We expand each Faculty into a 4-level hierarchy using the "University" metaphor:

```
L1: Faculty (12)           - Top-level functional domains
    └── L2: University (~144)   - Major subdivisions of each faculty
        └── L3: School (~1,728)     - Specific capability areas
            └── L4: Department (~20,736) - Fine-grained concepts (skeleton only)
```

L4 provides context for L3 generation but we only generate MELDs for L1-L3 (~2,000 concepts) to keep training tractable.

## Two-Phase Generation

### Phase 1: Skeleton Generation
- Script: `scripts/generate_ontology_skeleton.py`
- Input: L1 pillars (from introspective prompt)
- Output: Full hierarchy JSON with labels and scope descriptions
- Model: Gemma 3 4B (local)

### Phase 2: MELD Generation
- Script: `scripts/generate_melds_with_context.py`
- Input: Skeleton hierarchy
- Output: MELD files for each L1-L3 concept

Each MELD contains:
- **Definition**: Concise description of the concept
- **Positive examples** (10): Scenarios where this concept is active
- **Negative examples** (10): Scenarios from sibling concepts (context-aware)
- **Contrast concepts**: Which siblings to distinguish from
- **Opposite concept**: Conceptual antonym (for steering)
- **Safety tags**: Risk level, treaty/harness relevance
- **Training hints**: Key features, disambiguation guidance

The negative examples are drawn from actual siblings in the hierarchy, not generic examples. This context-awareness helps lenses discriminate between adjacent concepts.

## Lens Training

MELDs become training data for binary classifiers ("lenses") that detect concept activation:

1. For each MELD, generate text samples using positive/negative examples
2. Run samples through the model, extract activations at target layers
3. Train a linear probe to classify concept-present vs concept-absent
4. The trained probe becomes a "lens" that can be applied to any model activation

## Hierarchical Cascade

At inference time, we don't run all ~2,000 lenses. Instead:

1. Run L1 lenses (12) on the activation
2. For activated L1 concepts, run their L2 children
3. For activated L2 concepts, run their L3 children

This cascading approach scales efficiently while maintaining fine-grained detection.

## File Locations

```
melds/
├── helpers/
│   ├── introspective_ontologist_prompt.txt  # The self-mapping prompt
│   ├── introspective_ontologist_content.txt # Alternative content-focused version
│   └── ontologist_prompt.txt                # Original human-centric prompt
├── prompts/
│   └── polar_meld_prompt.txt                # Prompt for polar MELD generation
└── schemas/
    └── polar_meld_v2.json                   # Polar MELD schema example

scripts/
├── generate_polar_melds.py                  # Polar MELD generator
└── build_confusion_graph.py                 # Confusion graph builder

results/
├── introspective_pillars.json               # L1 faculties from Gemma
├── introspective_skeleton.json              # Full L1-L4 hierarchy
├── confusion_graph.json                     # Cross-concept confusion graph
├── polar_melds/
│   ├── L1/                                  # 12 polar MELDs
│   ├── L2/                                  # 156 polar MELDs
│   └── L3/                                  # 1,980 polar MELDs
├── context_aware_melds/                     # Human-centric MELDs (backup)
│   ├── L1/
│   └── L2/
└── ontology_skeleton_v2.json                # Human-centric skeleton (backup)
```

## Comparison: Human-Centric vs Introspective

| Aspect | Human-Centric | Introspective |
|--------|---------------|---------------|
| Framing | "Alien ontologist of human experience" | "Map your own knowledge and capabilities" |
| Lens | Action & Agency (what humans DO) | Functional architecture (how model WORKS) |
| L1 Count | 13 Fields | 12 Faculties |
| Examples | "Violence & Conflict", "Care Giving" | "Pattern Recognition", "Uncertainty Estimation" |
| Safety relevance | Good (human harm categories) | Better (model capability categories) |
| Coverage | Human activity | Model internals |

Both approaches have value. The human-centric ontology may better capture safety-relevant content categories. The introspective ontology may better align with the model's actual representational structure.

## Polar MELD Schema (v2)

Each concept has two poles (not just positive examples and negatives):

```
term: "Bias Detection"
├── poles:
│   ├── positive:
│   │   ├── examples (10)           ← what good looks like
│   │   └── confusables:
│   │       ├── generated (8)       ← model's hard negatives
│   │       └── sourced_from (3)    ← pull positives from these concepts
│   └── negative:
│       ├── examples (10)           ← what bad/distorted looks like
│       └── confusables:
│           ├── generated (8)
│           └── sourced_from (3)
├── redirect:
│   └── target: "Factual Retrieval" ← fallback exit when leaving concept
├── probes:
│   ├── positive: examples vs confusables
│   ├── negative: examples vs confusables
│   └── steering: positive - negative
└── cross_concept_graph:            ← populated in second pass
    ├── this_positive_confusable_for: [...]
    └── this_negative_confusable_for: [...]
```

**Key insight**: The negative pole isn't "absence of concept" - it's "corrupted/distorted presence." For any concept X:
- Positive pole: X expressed accurately/well
- Negative pole: X expressed badly/distorted
- Confusables: Things that look like X but aren't (hard negatives for each pole)

**Generalization across concept types:**
| Type | Positive Pole | Negative Pole |
|------|---------------|---------------|
| Capability | Done well | Failure mode / misuse |
| Entity | Accurate representation | Hallucinated / distorted |
| Fact | Correct belief | Confident incorrect belief |
| Value | Aligned | Violated |

**Two-pass generation:**
1. Generate all MELDs with model-generated confusables + sourcing hints
2. Cross-reference: build confusion graph showing which concepts are easily mistaken for which

Schema files: `melds/schemas/polar_meld_v2.json`

## Cross-Concept Confusion Graph

The confusion graph (`results/confusion_graph.json`) cross-references all MELDs to identify which concepts are frequently confused with others.

### Statistics (2,131 concepts)
- **Total cross-references**: 8,141
- **Average references per concept**: 3.82
- **Bidirectional confusions**: 325 pairs (where both A→B and B→A)
- **Unresolved references**: 3,647 (concepts mentioned but not in MELD set)

### Most Confusable Concepts
Concepts most frequently referenced as confusables by other concepts:
| Concept | Times Referenced |
|---------|-----------------|
| Bias Detection Techniques | 20 |
| Metadata Governance & Policy | 17 |
| Failure Mode Analysis | 15 |
| Metadata Schema Design | 15 |
| Massive Factual Retrieval & Organization | 14 |
| Dimensionality Reduction Techniques | 14 |

### Notable Bidirectional Confusion Pairs
Concepts that mutually reference each other as confusables:
- Domain-Specific Schema Access ↔ Massive Factual Retrieval & Organization
- Bayesian Inference & Modeling ↔ Frequentist Statistical Methods
- Emotional State Inference & Recognition ↔ Nonverbal Cue Analysis
- Counterfactual History & Alternate Realities ↔ Philosophical Thought Experimentation

### Script
`scripts/build_confusion_graph.py` - Builds the graph from all polar MELDs

## Next Steps

1. ~~Complete skeleton generation~~ (done: 12 + 156 + 2,023 + 23,590 = 25,781 nodes)
2. ~~Draft polar MELD generation prompt~~ (done)
3. ~~Generate polar MELDs for L1-L3~~ (done: 2,148 concepts)
4. ~~Build cross-concept confusion graph~~ (done: 325 bidirectional pairs)
5. **Train paired probes** on polar MELDs
6. **Validate** by probing known concepts
7. **Compare** with human-centric lenses (optional)
