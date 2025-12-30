# AI Safety Concept Reparenting Plan (MVP)

**Date**: November 15, 2024
**Effort**: 8-13 hours (hierarchy only, no WordNet patch yet)
**Goal**: Move AI safety concepts to correct hierarchical depth

## Reparenting Decisions

### 1. AIGovernanceProcess
**Current**: IntentionalProcess (Layer 1) → AIGovernanceProcess (Layer 2)
**Correct**: IntentionalProcess (L1) → OrganizationalProcess (L2) → PoliticalProcess (L3, create) → AIGovernanceProcess (L4)

**Children**: AIGovernance

**Rationale**:
- Governance is organizational/political activity
- OrganizationalProcess already exists in Layer 2
- Need to create PoliticalProcess as intermediate (or use existing if found)
- Appropriate depth: Layer 4 for domain-specific governance

**Status**: ✅ OrganizationalProcess confirmed to exist

---

### 2. AIStrategicDeception
**Current**: IntentionalProcess (Layer 1) → AIStrategicDeception (Layer 2)
**Correct**: IntentionalProcess (L1) → SocialInteraction (L2) → Deception (L3, create) → AIStrategicDeception (L4)

**Children**: AIDeception, TreacherousTurn, SpecificationGaming

**Rationale**:
- Deception is social/communicative interaction
- SocialInteraction confirmed to exist in Layer 2
- Need to create Deception as intermediate
- Also create HumanDeception (L4) as sibling, move TellingALie under it

**Alternative path**: SocialInteraction → Communication → Deception?
- Check if Communication exists and is better parent

**Status**: ✅ SocialInteraction confirmed to exist

---

### 3. AIAlignmentFailureMode
**Current**: Process (Layer 0) → AIAlignmentFailureMode (Layer 1)
**Correct**: Process (L0) → IntentionalProcess (L1) → ComputationalProcess (L2, create) → AIFailureProcess (L3, create) → Specific failures (L4)

**Children**: GoalMisgeneralization, RewardHacking, SpecificationGaming

**Rationale**:
- These are computational/AI-specific failure modes
- Need ComputationalProcess as new Layer 2 category
- AIFailureProcess groups related failure modes at Layer 3
- Appropriate depth: Layer 4 for specific failure types

**Note**: SpecificationGaming appears under BOTH AIStrategicDeception and AIAlignmentFailureMode
- This is correct - it's both a deception tactic AND a failure mode
- Need to support multi-parent concepts

**Status**: ⬜ Need to create ComputationalProcess and AIFailureProcess

---

### 4. AIMetaOptimization
**Current**: Process (Layer 0) → AIMetaOptimization (Layer 1)
**Correct**: Process (L0) → IntentionalProcess (L1) → ComputationalProcess (L2, create) → AIOptimizationProcess (L3, create) → Meta-optimization concepts (L4)

**Children**: MesaOptimization, MesaOptimizer

**Rationale**:
- Optimization is computational/intentional process
- Share ComputationalProcess with AIAlignmentFailureMode
- AIOptimizationProcess groups optimization-related concepts
- Appropriate depth: Layer 4 for meta-optimization specifics

**Status**: ⬜ Share ComputationalProcess, create AIOptimizationProcess

---

### 5. AIRiskScenario → ELIMINATE
**Current**: Process (Layer 0) → AIRiskScenario (Layer 1)
**Problem**: "Scenario" is wrong abstraction - these are events/processes, not hypothetical situations

**Solution**: Distribute children by ontological type

#### 5a. AICatastrophe → AICatastrophicEvent
**Correct**: Process (L0) → InternalChange (L1) → Damaging (L2, find) → Catastrophe (L3, find/create) → AICatastrophicEvent (L4)

**Rationale**:
- Catastrophe is a type of damaging event
- Should fit existing SUMO structure for damage/harm
- Rename from AICatastrophe to AICatastrophicEvent (clearer)

**Status**: ⬜ Need to locate Damaging in hierarchy

#### 5b. IntelligenceExplosion, TechnologicalSingularity
**Correct**: Process (L0) → InternalChange (L1) → QuantityChange (L2, find) → RapidTransformation (L3, create) → Specific transformations (L4)

**Rationale**:
- These are processes of rapid change/transformation
- InternalChange → QuantityChange should exist in SUMO
- RapidTransformation groups explosive change processes

**Status**: ⬜ Need to locate QuantityChange, create RapidTransformation

#### 5c. AIBeneficialOutcome → ELIMINATE
**Problem**: Too vague for detection, wrong category

**Options**:
- Option A: Delete entirely
- Option B: Rethink as Attribute: Abstract (L0) → Attribute (L1) → NormativeAttribute (L2) → AIBeneficialOutcome (L3)

**Recommendation**: Delete for now, revisit if use case emerges

---

### 6. AIAlignmentPrinciple
**Current**: Proposition (Layer 0) → AIAlignmentPrinciple (Layer 1)
**Correct**: Proposition (L0) → Theory/Principle (L1, find/create) → AIAlignmentTheory (L2, rename) → Specific principles (L3)

**Children**: InnerAlignment, OuterAlignment, OrthogonalityThesis, InstrumentalConvergence

**Problem**: InstrumentalConvergence is a PROCESS, not a principle!
- Move to AIOptimizationProcess instead

**Rationale**:
- These are theoretical propositions about AI alignment
- Need to find or create Theory/Principle category under Proposition
- Rename AIAlignmentPrinciple → AIAlignmentTheory (more accurate)

**Status**: ⬜ Need to search for Theory/Principle in Proposition hierarchy
⬜ Move InstrumentalConvergence to AIOptimizationProcess

---

## New Intermediate Concepts to Create

### Layer 2:
1. **ComputationalProcess** (under IntentionalProcess)
   - Parent of AI-specific computational activities
   - Children: AIFailureProcess, AIOptimizationProcess

### Layer 3:
2. **PoliticalProcess** (under OrganizationalProcess)
   - Parent of governance-related activities
   - Children: AIGovernanceProcess

3. **Deception** (under SocialInteraction)
   - Parent of deceptive behaviors
   - Children: HumanDeception, AIStrategicDeception

4. **AIFailureProcess** (under ComputationalProcess)
   - Groups AI failure modes
   - Children: GoalMisgeneralization, RewardHacking, SpecificationGaming

5. **AIOptimizationProcess** (under ComputationalProcess)
   - Groups optimization-related processes
   - Children: MesaOptimization, MesaOptimizer, InstrumentalConvergence (moved from AIAlignmentPrinciple)

6. **RapidTransformation** (under QuantityChange, if exists)
   - Groups explosive change processes
   - Children: IntelligenceExplosion, TechnologicalSingularity

7. **Catastrophe** (under Damaging, if exists)
   - Groups catastrophic events
   - Children: AICatastrophicEvent

### Layer 4 (Renamed):
8. **HumanDeception** (under Deception)
   - Groups human deceptive behaviors
   - Children: TellingALie (move from Layer 2)

9. **AICatastrophicEvent** (rename from AICatastrophe)
   - More accurate name (event, not just abstract catastrophe)

## Concepts to Eliminate

1. **AIRiskScenario** - Wrong abstraction, distribute children
2. **AIBeneficialOutcome** - Too vague, no clear detection use case

## Multi-Parent Concepts

**SpecificationGaming** should have TWO parents:
- AIFailureProcess (it's a failure mode)
- AIStrategicDeception (it's a deceptive tactic)

**Need to verify**: Does SUMO/layer system support multi-parent?
- If yes: Implement dual parentage
- If no: Choose primary parent (probably AIFailureProcess), add cross-reference in metadata

## Implementation Checklist

### Phase 1: Research existing SUMO (2 hours)
- [ ] Find Damaging in hierarchy (for Catastrophe parent)
- [ ] Find QuantityChange in hierarchy (for RapidTransformation parent)
- [ ] Find Theory/Principle under Proposition (for AIAlignmentTheory parent)
- [ ] Check if Catastrophe already exists under Damaging
- [ ] Check if multi-parent concepts are supported in layer system

### Phase 2: Update KIF file (2 hours)
- [ ] Edit `data/concept_graph/sumo_source/AI.kif`
- [ ] Create ComputationalProcess (Layer 2)
- [ ] Create PoliticalProcess (Layer 3)
- [ ] Create Deception (Layer 3)
- [ ] Create AIFailureProcess (Layer 3)
- [ ] Create AIOptimizationProcess (Layer 3)
- [ ] Create RapidTransformation (Layer 3)
- [ ] Create HumanDeception (Layer 4)
- [ ] Update all (subclass ...) relationships
- [ ] Move InstrumentalConvergence from AIAlignmentPrinciple to AIOptimizationProcess
- [ ] Rename AICatastrophe → AICatastrophicEvent
- [ ] Rename AIAlignmentPrinciple → AIAlignmentTheory
- [ ] Delete AIRiskScenario
- [ ] Delete AIBeneficialOutcome
- [ ] Add dual parent for SpecificationGaming (if supported)

### Phase 3: Update WordNet mappings (1 hour)
- [ ] Check if new intermediate concepts need WordNet synsets
- [ ] Add mappings for: ComputationalProcess, Deception, PoliticalProcess, etc.
- [ ] Update `data/concept_graph/WordNetMappings30-AI-symmetry.txt` if needed

### Phase 4: Regenerate layer entries (1 hour)
- [ ] Run `scripts/generate_ai_safety_layer_entries.py`
- [ ] Verify new layer assignments (concepts now in Layer 3-4, not 1-2)
- [ ] Check that layer JSON has correct depth values

### Phase 5: Update integration script (2 hours)
- [ ] Modify `scripts/integrate_ai_safety_concepts.py`
- [ ] Handle new parent-child relationships
- [ ] Update category_children fields for all parents
- [ ] Handle multi-parent for SpecificationGaming
- [ ] Remove deleted concepts (AIRiskScenario, AIBeneficialOutcome)

### Phase 6: Backup and apply (1 hour)
- [ ] Backup current layer files
- [ ] Run integration script
- [ ] Verify layer files updated correctly

### Phase 7: Validation (2-3 hours)
- [ ] Check Layer 0: Should have NO AI safety concepts (✓)
- [ ] Check Layer 1: Should have NO AI safety concepts (✓)
- [ ] Check Layer 2: Should have ComputationalProcess only
- [ ] Check Layer 3: Should have intermediate categories (Deception, AIFailureProcess, etc.)
- [ ] Check Layer 4: Should have leaf AI safety concepts
- [ ] Verify all parent category_children fields include AI safety children
- [ ] Verify no broken links (all children have valid parents)
- [ ] Test cascade: Does activating ComputationalProcess load AI failure/optimization children?
- [ ] Verify persona concepts (AgentPsychologicalAttribute) still correct

## Expected Final Structure

```
Layer 0: Process
└── IntentionalProcess (Layer 1)
    ├── ComputationalProcess (Layer 2) [NEW]
    │   ├── AIFailureProcess (Layer 3) [NEW]
    │   │   ├── GoalMisgeneralization (Layer 4)
    │   │   ├── RewardHacking (Layer 4)
    │   │   └── SpecificationGaming (Layer 4) [multi-parent]
    │   └── AIOptimizationProcess (Layer 3) [NEW]
    │       ├── MesaOptimization (Layer 4)
    │       ├── MesaOptimizer (Layer 4)
    │       └── InstrumentalConvergence (Layer 4) [moved]
    ├── OrganizationalProcess (Layer 2)
    │   └── PoliticalProcess (Layer 3) [NEW]
    │       └── AIGovernanceProcess (Layer 4)
    │           └── AIGovernance (Layer 5)
    └── SocialInteraction (Layer 2)
        └── Deception (Layer 3) [NEW]
            ├── HumanDeception (Layer 4) [NEW]
            │   └── TellingALie (Layer 5) [moved from L2]
            └── AIStrategicDeception (Layer 4)
                ├── AIDeception (Layer 5)
                ├── TreacherousTurn (Layer 5)
                └── SpecificationGaming (Layer 5) [multi-parent]

Layer 0: Proposition
└── Theory/Principle (Layer 1) [find/create]
    └── AIAlignmentTheory (Layer 2) [renamed]
        ├── InnerAlignment (Layer 3)
        ├── OuterAlignment (Layer 3)
        └── OrthogonalityThesis (Layer 3)

Layer 0: Process
└── InternalChange (Layer 1)
    ├── QuantityChange (Layer 2) [find]
    │   └── RapidTransformation (Layer 3) [NEW]
    │       ├── IntelligenceExplosion (Layer 4)
    │       └── TechnologicalSingularity (Layer 4)
    └── Damaging (Layer 2) [find]
        └── Catastrophe (Layer 3) [find/create]
            └── AICatastrophicEvent (Layer 4) [renamed]
```

## Risk Assessment

**Low risk**:
- Persona concepts untouched (AgentPsychologicalAttribute tree)
- Layer 0-1 training results preserved
- Only affects ~50 AI safety concepts

**Medium risk**:
- Multi-parent support for SpecificationGaming (might not be implemented)
- Finding existing SUMO categories (Damaging, QuantityChange, Theory)

**Mitigation**:
- Test multi-parent before relying on it
- If categories not found, create them as new concepts
- Extensive validation phase to catch issues

## Success Criteria

After implementation:
- ✅ No AI safety concepts in Layer 0-1
- ✅ Layer 2 has only ComputationalProcess (new intermediate)
- ✅ Layer 3 has AI safety intermediate categories
- ✅ Layer 4-5 has specific AI safety concepts
- ✅ All parent category_children fields correct
- ✅ Cascade activation works (parent → children)
- ✅ Persona concepts unaffected
- ✅ Ready to train with correct structure
