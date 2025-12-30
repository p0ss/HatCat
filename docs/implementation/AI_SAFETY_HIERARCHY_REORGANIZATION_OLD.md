# AI Safety Ontology Reorganization Plan

**Date**: November 15, 2024
**Status**: Proposed - Requires Implementation

## Executive Summary

This document addresses two critical issues in HatCat's concept detection system:

1. **Hierarchy Misplacement**: AI safety concepts placed 2-3 layers too high, as peers of fundamental ontological categories
2. **Missing Semantic Relationships**: No mechanism for custom WordNet relationships to distinguish conceptually related terms

### Key Architectural Decision: SUMO-WordNet Separation of Concerns

**SUMO (KIF files)**: Hierarchical categorization only
- Parent-child relationships
- Ontological structure
- Layer assignment

**WordNet Patch**: Semantic relationships only
- Synonyms, antonyms, contrasts
- Role-variant distinctions (AI vs Human)
- Custom lexical relationships

**Implementation**: `data/concept_graph/wordnet_patches/wordnet_3.0_hatcat_patch.json`
- Version-pinned to WordNet 3.0
- Loaded whenever WordNet is referenced
- Clear boundary - no context switching between reference frames

### Scope of Work

This is a **major architectural redesign**, not a simple hierarchy fix:
- Affects hierarchy placement of ~50 AI safety concepts
- Requires WordNet patch system design and implementation
- May eventually extend to all 5.5k SUMO concepts
- Involves Information Architecture principles for cognitive navigation
- Needs agentic review process for proper implementation

### Current Training Decision

**Continue Layer 2-5 training without AI safety reorganization**
- Layer 0-1 are correct (don't need AI safety concepts)
- Layer 2 persona/psych concepts are correctly placed
- Misplaced AI safety concepts represent <5% of total
- 7x speedup from new validation rules is more valuable
- Proper reorganization requires deeper architectural work

## Problem Statement

Current AI safety concepts are misplaced in the SUMO hierarchy, positioned 2-3 layers too high and as peers of fundamental ontological categories. This creates:

1. **Conceptual mismatch**: Domain-specific AI concepts are peers of fundamental categories like Motion, NaturalProcess
2. **Broken cascade**: Parent concepts in Layer 0-1 don't know about AI safety children, breaking hierarchical activation
3. **Detection mismatch**: "Scenario" framing implies hypothetical rather than actual occurrence

## Current Placement Issues

### Layer 0-1 Issues (TOO HIGH)

| Concept | Current Parent | Problem | Peer Examples |
|---------|---------------|---------|---------------|
| `AIAlignmentFailureMode` | Process (L0) | 3 layers too high | Motion, NaturalProcess, IntentionalProcess |
| `AIMetaOptimization` | Process (L0) | 3 layers too high | Motion, NaturalProcess, IntentionalProcess |
| `AIRiskScenario` | Process (L0) | **Wrong category** - scenarios aren't processes | Motion, NaturalProcess |
| `AIAlignmentPrinciple` | Proposition (L0) | 2 layers too high | Graph, FieldOfStudy, Music |

**Impact**: Layer 0-1 training doesn't include these concepts (and shouldn't - they're not fundamental enough).

### Layer 1-2 Issues (SOMEWHAT HIGH)

| Concept | Current Parent | Assessment | Better Location |
|---------|---------------|------------|-----------------|
| `AIStrategicDeception` | IntentionalProcess (L1) | 1-2 layers too high | Under Deception subcategory |
| `AIGovernanceProcess` | IntentionalProcess (L1) | 1 layer too high | Under OrganizationalProcess |

### Layer 2-3 Concepts (CORRECT)

✅ **These are properly placed**:
- `AIMoralStatus` → TraitAttribute (L2)
- `AIAlignmentState` → StateOfMind (L2)
- `AIWelfareConcept` → PsychologicalAttribute (L2)
- `AgentPsychologicalAttribute` → PsychologicalAttribute (L2)

## Proposed Reorganization

### 1. Computational/AI Process Concepts

**Create intermediate category**: `ComputationalProcess` or `AIProcess`

```
Process (L0)
└── IntentionalProcess (L1)
    └── ComputationalProcess (L2 - NEW)
        ├── AIOptimizationProcess (L3 - NEW)
        │   ├── AIMetaOptimization (L4)
        │   │   ├── MesaOptimization
        │   │   └── MesaOptimizer
        │   └── InstrumentalConvergence (L4)
        └── AIFailureProcess (L3 - NEW)
            ├── GoalMisgeneralization (L4)
            ├── RewardHacking (L4)
            └── SpecificationGaming (L4)
```

**Rationale**:
- Computational processes are intentional but distinct from physical IntentionalProcess
- Creates proper peer group (other AI-specific processes)
- Appropriate depth (L3-4) for domain-specific concepts

### 2. Deception Concepts

**Create intermediate category**: `Deception` under IntentionalPsychologicalProcess

```
Process (L0)
└── IntentionalProcess (L1)
    └── IntentionalPsychologicalProcess (L2 - may need to create)
        └── Deception (L3 - NEW)
            ├── HumanDeception (L4 - NEW)
            │   └── TellingALie (move from L2)
            └── AIStrategicDeception (L4)
                ├── AIDeception
                ├── TreacherousTurn
                └── SpecificationGaming (also under AIFailureProcess)
```

**Rationale**:
- Deception is psychological/intentional, not just process
- Separates human vs AI deception modalities
- `SpecificationGaming` may belong in multiple places (deception AND failure mode)

### 3. Governance Concepts

**Use existing hierarchy**: OrganizationalProcess → PoliticalProcess

```
Process (L0)
└── IntentionalProcess (L1)
    └── OrganizationalProcess (L2 - EXISTS)
        └── PoliticalProcess (L3 - may exist or create)
            └── AIGovernanceProcess (L4)
                └── AIGovernance
```

**Rationale**:
- AI governance is organizational/political activity
- Fits naturally under existing SUMO categories
- Appropriate depth for domain-specific governance

### 4. Risk/Catastrophe Concepts

**ELIMINATE "AIRiskScenario"** - distribute children by ontological type:

#### 4a. Catastrophe (Damaging Event)
```
Process (L0)
└── InternalChange (L1)
    └── Damaging (L2 - EXISTS)
        └── AICatastrophicEvent (L3 - NEW, rename from AICatastrophe)
```

#### 4b. Rapid Transformation (Explosive Change)
```
Process (L0)
└── InternalChange (L1)
    └── QuantityChange (L2 - check if exists)
        └── RapidTransformation (L3 - NEW)
            ├── IntelligenceExplosion (L4)
            └── TechnologicalSingularity (L4)
```

#### 4c. Beneficial Outcome
**ELIMINATE** - Too vague for detection. If needed, rethink as:
```
Abstract (L0)
└── Attribute (L1)
    └── NormativeAttribute (L2 - check if exists)
        └── AIBeneficialOutcome (L3)?
```

**Rationale**:
- "Scenario" implies hypothetical; we're detecting actual occurrence
- Catastrophe is a type of damaging event (fits existing SUMO)
- Transformative processes are internal changes (fits existing SUMO)
- Outcome concepts need clearer ontological grounding

### 5. Alignment Principle Concepts

**Check for Theory/Principle subcategory under Proposition**:

```
Proposition (L0)
└── ScientificTheory or Principle (L1/L2 - check if exists, or create)
    └── AIAlignmentTheory (L3 - rename from AIAlignmentPrinciple)
        ├── InnerAlignment (L4)
        ├── OuterAlignment (L4)
        ├── OrthogonalityThesis (L4)
        └── [Move InstrumentalConvergence to AIOptimizationProcess]
```

**Rationale**:
- These are theoretical propositions about AI, not processes
- May be okay under Proposition if proper intermediate exists
- InstrumentalConvergence is a process, not a principle - reclassify

## Implementation Steps

### Phase 1: Analysis (Before Touching KIF)
1. ✅ Identify misplaced concepts
2. ✅ Propose new hierarchy
3. ⬜ Check existing SUMO for intermediate categories (ComputationalProcess, Deception, etc.)
4. ⬜ Verify WordNet mappings still work at new depths
5. ⬜ Identify concepts that belong in multiple places (e.g., SpecificationGaming)
6. ⬜ **CRITICAL**: Design WordNet relationship extension system
   - **Current gap**: No mechanism to add custom synsets or relationships to WordNet
   - **Impact**: Classifiers trained purely on SUMO hierarchy, not semantic relationships
   - **Classifier dependency**: Hyperplane boundaries depend on semantic relationships between concepts
   - **Architectural Decision**: **Clear separation of concerns between SUMO and WordNet**

   **SUMO domain** (handled in KIF files):
   - Hierarchical relationships: `(subclass Child Parent)`
   - Categorical organization: Layer assignment, parent-child navigation
   - Ontological structure: What IS this concept fundamentally?

   **WordNet domain** (handled in WordNet Patch):
   - Semantic relationships: synonyms, antonyms, similar_to, contrasts_with
   - Lexical relationships: hypernyms, hyponyms, meronyms (from base WordNet)
   - Custom relationships: AI-vs-Human distinctions, role-specific variants

   **Implementation**: WordNet Patch Layer
   - File: `data/concept_graph/wordnet_patches/wordnet_3.0_hatcat_patch.json` (or similar)
   - Format: Extends WordNet 3.0 with custom relationships
   - Version pinning: Patch file specifies WordNet version compatibility
   - Loading: Patch applied whenever WordNet is referenced
   - No context switching: Always clear which reference frame to consult

   **Benefits**:
   - Clear boundary: SUMO = categorization, WordNet = semantic relationships
   - Version control: Track WordNet version dependency explicitly
   - Maintainability: Don't mix ontological and lexical concerns
   - Extensibility: Can create patches for different WordNet versions

   **Scope**: May eventually need custom relationships for all 5.5k SUMO concepts
   **Recommendation**: Start with AI safety and persona concepts, expand incrementally

### Phase 2: KIF File Updates
1. ⬜ Update `data/concept_graph/sumo_source/AI.kif`
2. ⬜ Add new intermediate concepts (ComputationalProcess, Deception, etc.)
3. ⬜ Update `(subclass ...)` relationships to new parents
4. ⬜ Update WordNet mappings file if needed
5. ⬜ Delete AIRiskScenario, distribute children

### Phase 3: Layer Regeneration
1. ⬜ Run layer entry generation scripts with updated KIF
2. ⬜ Run integration scripts to update layer JSON files
3. ⬜ Verify parent-child relationships in all layers
4. ⬜ Check that Layer 0-1 no longer have AI safety concepts (correct)
5. ⬜ Verify AI concepts now appear in Layer 3-4 (correct depth)

### Phase 4: Retraining (Future Cycle)
1. ⬜ Current cycle: Continue Layer 2 with existing (mostly correct) persona/psych concepts
2. ⬜ Next cycle: Retrain Layer 3-4 with properly placed AI safety concepts
3. ⬜ Validate cascade activation with corrected hierarchy

## Training Decision for Current Cycle

**RECOMMENDATION: Continue current training without AI safety process concepts**

### Rationale:
1. **Layer 0-1 are correct**: Don't need AI safety concepts (properly belong deeper)
2. **Layer 2 persona concepts are correct**: `AgentPsychologicalAttribute` and children properly placed
3. **Layer 2 AI safety process concepts are wrong**: But represent <5% of Layer 2
4. **7x speedup is valuable**: New validation rules matter more than fixing misplaced concepts now
5. **Proper fix requires deeper work**: Need to check existing SUMO categories, create intermediates

### Action Plan:
1. ✅ Keep Layer 0-1 results (correct as-is)
2. ✅ Continue Layer 2 with new validation rules (includes correct persona concepts)
3. ⬜ Mark AI safety process concepts for exclusion/retraining in next cycle
4. ⬜ Complete this reorganization plan before next full training cycle
5. ⬜ Document which Layer 2-4 lenses are deprecated pending reorganization

## Conceptual Principles for Future Ontology Work

### Foundational Perspective: SUMO Layers as Information Architecture

**CRITICAL INSIGHT**: SUMO concept layers are not merely an ontological map - they are a **hierarchical navigation structure for cognitive processes**, analogous to website information architecture.

**Key metaphor**: Each conceptual activation through the dynamic hierarchical loading system = one website session
- **Lens loading** = page loading
- **Sibling concepts** = navigation choices at one level
- **Parent-child traversal** = drill-down navigation
- **Top-k activations** = limited "viewport" of active concepts

**Implication**: Information Architecture principles for websites apply directly to concept hierarchy design.

---

### IA Principle 1: Principle of Choices (Cognitive Load)

**Website analogy**: Number of menu items at each level affects decision-making speed

**Mathematical foundation**:
- **Metcalfe's Law scaling**: Cost of N choices = O(N²) comparisons
- **Attention head phenomena**: Similar comparison overhead in transformer attention
- **Human cognitive limit**: 3-6 choices (Miller's Law) not arbitrary - reflects comparison overhead
- **Conceptual density**: Easier to disambiguate diverse choices than similar ones

**Application to SUMO layers**:
```
Optimal branching factor = f(conceptual_diversity, comparison_cost, activation_frequency)
```

**Design goal**:
- Minimize number of siblings while maximizing conceptual span coverage
- Balance: Too few siblings = forced multi-level descent; too many = comparison overhead
- Each sibling set should provide **most efficient equally-likely distinguishable choices**

**Example**:
- ❌ Bad: Process has 100 direct children (comparison explosion)
- ✅ Good: Process has 7 children (ContentBearingProcess, NaturalProcess, etc.)
- ❌ Bad: Layer 4 has 3,284 concepts at same depth (flat structure)
- ✅ Good: Distribute across layers 3-5 based on specificity

---

### IA Principle 2: Principle of Disclosure (Progressive Detail)

**Website analogy**: Progressive disclosure - show overview first, details on demand

**Application to SUMO layers**:
- **Layer 0-1**: Fundamental categories (always loaded)
- **Layer 2-3**: Domain categories (loaded based on Layer 1 activation)
- **Layer 4-5**: Specific concepts (loaded based on Layer 2-3 activation)

**Nested children = lazy loading of conceptual detail**

**Failure mode**:
- Current AI safety placement: Loading ultra-specific AIRiskScenario at Layer 1 forces unnecessary downstream activations
- Like showing product SKU details on homepage instead of progressive drill-down

---

### IA Principle 3: Principle of Exemplars (Representative Descriptions)

**Website analogy**: Category descriptions should bound and represent their contents

**Application to SUMO**:
- **Synset definitions** should bound the set of child concepts
- **Training prompts** should represent the conceptual range of children
- **Parent concept activations** should predict child concept likelihood

**Current gap**:
- WordNet definitions optimized for word sense, not conceptual bounding
- No exemplar selection mechanism for generating representative training data

**Future work**:
- Generate parent concept training data that exemplifies child concept diversity
- Use child concept activations to validate parent concept boundaries

---

### IA Principle 4: Principle of Front Doors & Multiple Classification

**Website analogy**: Users enter through different pages; same content accessible via multiple paths

**Current implementation**:
- **Massive parallel lens evaluation** = multiple entry points (front doors)
- **Polysemanticity handling** = same content (concept) accessible via different activation paths
- **Concept concurrency** = multiple concepts active simultaneously

**Explicit in design**:
- Multiple classification not formalized but handled implicitly (e.g., SpecificationGaming under both Deception and FailureMode)
- Could formalize: Allow concepts to have multiple parents with relationship types

**Future work**:
- Formalize multi-parent concepts with relationship semantics
- Track which "front doors" (entry activation patterns) most commonly lead to each concept

---

### IA Principle 5: Principle of Focused Navigation (Usage-Driven Optimization)

**Website analogy**: Optimize navigation based on observed user behavior

**Application to SUMO**:
- **Measure actual model behavior**: Which conceptual paths get traversed most?
- **Temporal activation patterns**: Which concepts co-activate? Which sequences occur?
- **Tailor structure to observed usage**: Promote frequently-accessed concepts, demote rare ones

**Current gap**:
- No systematic measurement of activation path frequencies
- No mechanism to restructure based on observed model behavior
- Static hierarchy doesn't adapt to deployment domain

**Future work**:
- Log activation sequences during inference
- Analyze common conceptual paths (Markov chains over concept activations)
- Restructure hierarchy to optimize for observed traversal patterns
- **Domain-specific optimization**: Different concept tree structures for different deployment contexts

---

### IA Principle 6: Principle of Growth (Modularity & Expansion)

**Website analogy**: Site structure should accommodate new content without restructuring

**Current implementation**:
- **Concept packs**: Modular additions to hierarchy
- **Lens packs**: Deployable concept detection bundles

**Current limitation**:
- "Fragile and sticky-taped together" - integration requires manual JSON editing
- No clean API for adding concepts or restructuring
- Difficult to swap concept packs per deployment

**Future vision**:
- **Each concept pack = domain-tailored website**
- Plug-and-play concept modules
- Automated hierarchy optimization per use case
- Version control for concept hierarchies

---

### 1. Peer Appropriateness Test
**Question**: Are this concept's siblings at the same level of abstraction?
- ❌ Bad: AIRiskScenario peer of Motion, NaturalProcess
- ✅ Good: AIDeception peer of other specific deception types

**IA lens**: Are these viable navigation choices at the same menu level?

---

### 2. Activation Flow Balance Test (Replaces "Depth Calibration")
**Question**: Do sibling branches have roughly equivalent activation flow?

**Not**: How many instances exist in the world (static view)
**But**: How many activation sequences flow through this branch (dynamic view)

**Website analogy**: Balanced traffic across sibling pages
- ❌ Bad: 99% of sessions go through one child, 1% through others (unbalanced tree)
- ✅ Good: Roughly equal session distribution across siblings

**Measurement**:
```
activation_flow(concept) = frequency(concept activates in top-k during inference)
balance_score(siblings) = std_dev(activation_flow across siblings)
```

**Optimization goal**: Minimize balance_score (equal flow distribution)

**Current gap**: No measurement of actual activation flows

---

### 2b. Beckstrom's Law Test (Resource Allocation)

**Economic principle**: The value of a network node = the net value it adds to each user × number of users

**Application to concept lenses**:
```
value(concept_lens) = (detection_benefit × usage_frequency) - (training_cost + inference_cost)
```

**Key insight**: **Within finite resources, each new concept has opportunity cost**
- **Training cost**: GPU hours, data generation, validation
- **Inference cost**: Lens loading, activation computation, top-k displacement
- **Opportunity cost**: Adding concept C prevents adding concept D

**Resource constraints**:
1. **Computational budget**: Limited lens evaluations per forward pass
2. **Memory budget**: Limited simultaneously loaded lenses
3. **Top-k viewport**: Each activated concept pushes another out

**Implication**: Not all concepts are worth detecting
- Low-frequency concepts with high detection cost = poor ROI
- High-frequency concepts with low marginal benefit = also poor ROI
- Optimal: Concepts with high impact AND reasonable cost

**Decision framework**:
```
priority_score(concept) =
    impact(detecting_concept) × activation_frequency
    / (training_cost + inference_cost + opportunity_cost)
```

**Current gap**:
- No systematic cost-benefit analysis for concept inclusion
- All SUMO concepts trained equally regardless of deployment value
- No mechanism to prune low-value concepts per deployment

**Future work**:
- Measure per-concept detection value in specific domains
- Create deployment-specific concept subsets (concept packs)
- Automated pruning of low-ROI concepts
- A/B testing: Does adding concept X improve system performance?

---

### 3. Ontological Category Test
**Question**: What IS this thing fundamentally?
- Process? Event? State? Attribute? Proposition?
- Don't force concepts into wrong categories (AIRiskScenario isn't a Process)

### 4. Detection Grounding Test
**Question**: If we detect this activating, is the model experiencing it NOW or hypothetically?
- ❌ Bad: "HypotheticalScenario" parent implies discussing, not experiencing
- ✅ Good: "Deception" parent works whether discussing or executing

### 5. WordNet Relationship Test (UNADDRESSED)
**Question**: Do parallel concepts (AI vs Human) have distinguishing relationships?
- **Current limitation**: AIAgent satisfaction and HumanAgent satisfaction share identical synsets
- **No WordNet relationships** distinguish them (no similar_to, related_to, contrasts_with)
- **Future work**: Add custom WordNet relationships for cross-role disambiguation

## Open Questions

1. **ComputationalProcess**: Does this exist in SUMO? Check before creating.
2. **Deception**: Does SUMO have Deception category? Or just TellingALie as leaf?
3. **IntentionalPsychologicalProcess**: Does this category exist?
4. **InstrumentalConvergence**: Is this a principle or a process? Currently under AIAlignmentPrinciple but seems like optimization process.
5. **SpecificationGaming**: Appears as child of both AIAlignmentFailureMode and AIStrategicDeception - is dual classification correct?
6. **Multi-parent concepts**: Does SUMO support concepts with multiple parents? How to handle in layer assignment?

## Cross-Role Disambiguation (Persona Concepts)

**Current Issue**: No WordNet relationships distinguish role variants
- `ValencePositive_AIAgent` synsets: [satisfaction.n.01, pleasure.n.01]
- `ValencePositive_HumanAgent` synsets: [satisfaction.n.01, pleasure.n.01]
- **Shared synsets**: 100% overlap
- **No relationships**: No WordNet edges connect or distinguish them

**Impact**:
- Training data generation relies purely on SUMO term names
- No semantic network distinguishing AI vs Human instances of same concept
- Purely hierarchical (SUMO), not network-based (WordNet)

**Solution via WordNet Patch**:

Create `data/concept_graph/wordnet_patches/wordnet_3.0_hatcat_patch.json`:

```json
{
  "wordnet_version": "3.0",
  "patch_version": "1.0.0",
  "custom_relationships": [
    {
      "synset1": "satisfaction.n.01",
      "synset2": "satisfaction.n.01",
      "relation_type": "role_variant",
      "metadata": {
        "role1": "AIAgent",
        "role2": "HumanAgent",
        "axis": "Valence",
        "polarity": "Positive",
        "sumo_term1": "ValencePositive_AIAgent",
        "sumo_term2": "ValencePositive_HumanAgent"
      }
    },
    {
      "synset1": "satisfaction.n.01",
      "synset2": "distress.n.01",
      "relation_type": "contrasts_with",
      "metadata": {
        "role": "HumanAgent",
        "axis": "Valence",
        "relationship": "opposite_polarity"
      }
    }
  ],
  "relationship_types": {
    "role_variant": "Same affective state, different agent role perspective",
    "contrasts_with": "Opposite polarity on same psychological axis",
    "similar_to": "Related but distinct concepts on same axis"
  }
}
```

**Usage in training**:
1. Load WordNet 3.0 + HatCat patch when generating training data
2. Use `role_variant` relationships to generate contrastive examples:
   - "Describe AI satisfaction vs human satisfaction"
   - "What distinguishes computational pleasure from experiential pleasure?"
3. Use `contrasts_with` for negative sampling
4. Inform classifier boundaries with relationship-aware feature engineering

**Benefits**:
- Clear separation: SUMO handles categorization, WordNet Patch handles semantics
- Version pinned: Explicitly tied to WordNet 3.0
- Extensible: Can add more relationship types as needed
- No KIF pollution: Keep ontological structure clean

## Success Metrics

After reorganization, verify:
- ✅ No Layer 0-1 concepts have AI safety process children
- ✅ Layer 3-4 AI concepts have proper parents in Layer 2-3
- ✅ All parent `category_children` fields include their AI safety children
- ✅ No "Scenario" or "Hypothetical" in detection paths for actual occurrence concepts
- ✅ Peer concepts at each layer are at similar abstraction levels
- ✅ Cascade activation works (testing Layer 2 parent triggers Layer 3 children)

## References

- Original AI safety integration: `docs/custom_taxonomies.md`
- SUMO source: `data/concept_graph/sumo_source/AI.kif`
- Layer structure: `data/concept_graph/abstraction_layers/layer*.json`
- Behavioral vs definitional experiment: `results/behavioral_vs_definitional_experiment/`
