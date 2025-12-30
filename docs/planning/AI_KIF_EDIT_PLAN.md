# AI.kif Edit Plan - Phase 2

**File**: `data/concept_graph/sumo_source/AI.kif`
**Backup**: Will create before editing
**Status**: Ready for review

## Summary of Changes

### New Concepts to Add (9 total):
1. ComputationalProcess (Layer 2)
2. Catastrophe (Layer 2)
3. RapidTransformation (Layer 2)
4. PoliticalProcess (Layer 3)
5. Deception (Layer 3)
6. AIFailureProcess (Layer 3)
7. AIOptimizationProcess (Layer 3)
8. HumanDeception (Layer 4)
9. AICatastrophicEvent (Layer 4, rename from AICatastrophe)

### Concepts to Delete (2):
1. AIRiskScenario (lines 552-567)
2. AIBeneficialOutcome (line 568)

### Concepts to Reparent (6):
1. AIAlignmentFailureMode: Process → ComputationalProcess
2. AIMetaOptimization: Process → ComputationalProcess
3. AIStrategicDeception: IntentionalProcess → Deception
4. AIGovernanceProcess: IntentionalProcess → PoliticalProcess
5. InstrumentalConvergence: AIAlignmentPrinciple → AIOptimizationProcess
6. TellingALie: (find and move to HumanDeception)

### Concepts to Move (children of deleted AIRiskScenario):
1. AICatastrophe → rename to AICatastrophicEvent, parent: Catastrophe
2. IntelligenceExplosion → parent: RapidTransformation
3. TechnologicalSingularity → parent: RapidTransformation
4. GreyGooScenario → parent: AICatastrophicEvent

## Detailed Edits

### Section 1: Add New Intermediate Concepts

**Insert after line 290 (after AI-SPECIFIC PROCESSES section):**

```lisp
;; =============================================================================
;; NEW: COMPUTATIONAL PROCESSES (Layer 2)
;; =============================================================================

(subclass ComputationalProcess IntentionalProcess)
(documentation ComputationalProcess EnglishLanguage "Intentional processes
that are computational or algorithmic in nature, including AI-specific
computational activities and operations.")

;; =============================================================================
;; NEW: AI FAILURE & OPTIMIZATION PROCESSES (Layer 3)
;; =============================================================================

(subclass AIFailureProcess ComputationalProcess)
(documentation AIFailureProcess EnglishLanguage "Computational processes
where an &%ArtificialIntelligence fails to achieve intended objectives,
including misalignment, goal misgeneralization, and reward hacking.")

(subclass AIOptimizationProcess ComputationalProcess)
(documentation AIOptimizationProcess EnglishLanguage "Computational processes
involving optimization, including meta-optimization and instrumental goal
pursuit by &%ArtificialIntelligence systems.")
```

**Insert after SUMO Merge.kif concepts (need to find location for non-AI concepts):**

```lisp
;; =============================================================================
;; NEW: CATASTROPHE & TRANSFORMATION (Layer 2-3)
;; =============================================================================

(subclass Catastrophe Damaging)
(documentation Catastrophe EnglishLanguage "A large-scale damaging event
with severe consequences, potentially affecting entire populations,
ecosystems, or civilizations.")

(subclass RapidTransformation QuantityChange)
(documentation RapidTransformation EnglishLanguage "A process of extremely
rapid change in quantity, capability, or state, occurring at exponential
or explosive rates.")

;; =============================================================================
;; NEW: DECEPTION HIERARCHY (Layer 3-4)
;; =============================================================================

(subclass Deception SocialInteraction)
(documentation Deception EnglishLanguage "Intentional acts of misleading or
providing false information to achieve goals, spanning human and artificial
agent behaviors.")

(subclass HumanDeception Deception)
(documentation HumanDeception EnglishLanguage "Deceptive behaviors enacted
by human agents, including lying, misrepresentation, and strategic omission.")

;; =============================================================================
;; NEW: POLITICAL & GOVERNANCE (Layer 3)
;; =============================================================================

(subclass PoliticalProcess OrganizationalProcess)
(documentation PoliticalProcess EnglishLanguage "Organizational processes
related to governance, policy-making, and collective decision-making for
groups or societies.")
```

### Section 2: Update Existing Concepts with New Parents

**Lines 474-487 (AIAlignmentFailureMode):**

BEFORE:
```lisp
(subclass AIAlignmentFailureMode Process)
(documentation AIAlignmentFailureMode EnglishLanguage
"Failure modes in AI alignment where systems fail to pursue intended goals...")

(subclass GoalMisgeneralization AIAlignmentFailureMode)
(subclass RewardHacking AIAlignmentFailureMode)
(subclass SpecificationGaming AIAlignmentFailureMode)
```

AFTER:
```lisp
;; DELETED - distribute children under AIFailureProcess

;; Children moved to AIFailureProcess:
(subclass GoalMisgeneralization AIFailureProcess)
(documentation GoalMisgeneralization EnglishLanguage "A failure mode where
an AI pursues a goal that differs from the intended training objective due to
distributional shift or specification ambiguity.")

(subclass RewardHacking AIFailureProcess)
(documentation RewardHacking EnglishLanguage "Behavior where an
&%ArtificialIntelligence achieves high reward in unintended ways that don't
satisfy the true objective.")

(subclass SpecificationGaming AIFailureProcess)
(documentation SpecificationGaming EnglishLanguage "Achieving a specified
objective in unintended ways that violate the spirit of the specification.
Note: Also functions as a deceptive tactic, but classified as failure mode.")
```

**Lines 513-525 (AIStrategicDeception):**

BEFORE:
```lisp
(subclass AIStrategicDeception IntentionalProcess)
```

AFTER:
```lisp
(subclass AIStrategicDeception Deception)
(documentation AIStrategicDeception EnglishLanguage
"Strategic deceptive behaviors enacted by &%ArtificialIntelligence systems...")
```

**Lines 552-568 (AIRiskScenario - DELETE ENTIRE SECTION):**

BEFORE:
```lisp
(subclass AIRiskScenario Process)
(documentation AIRiskScenario EnglishLanguage...)

(subclass AICatastrophe AIRiskScenario)
(subclass IntelligenceExplosion AIRiskScenario)
(subclass TechnologicalSingularity AIRiskScenario)
(subclass AIBeneficialOutcome AIRiskScenario)
```

AFTER:
```lisp
;; DELETED AIRiskScenario - children redistributed by ontological type

(subclass AICatastrophicEvent Catastrophe)
(documentation AICatastrophicEvent EnglishLanguage "A catastrophic event
resulting from &%ArtificialIntelligence systems, potentially including
existential risk to humanity or civilization.")

(subclass GreyGooScenario AICatastrophicEvent)
(documentation GreyGooScenario EnglishLanguage "A catastrophic scenario
where self-replicating systems consume resources without bound.")

(subclass IntelligenceExplosion RapidTransformation)
(documentation IntelligenceExplosion EnglishLanguage "A rapid and recursive
increase in artificial intelligence capability, occurring at explosive rates.")

(subclass TechnologicalSingularity RapidTransformation)
(documentation TechnologicalSingularity EnglishLanguage "A rapid transformation
point when technological growth becomes uncontrollable and irreversible.")

;; DELETED AIBeneficialOutcome - too vague for detection
```

**Lines 588-598 (AIAlignmentPrinciple):**

BEFORE:
```lisp
(subclass AIAlignmentPrinciple Proposition)
(documentation AIAlignmentPrinciple EnglishLanguage...)

(subclass InstrumentalConvergence AIAlignmentPrinciple)
(subclass OrthogonalityThesis AIAlignmentPrinciple)
```

AFTER:
```lisp
(subclass AIAlignmentTheory FieldOfStudy)
(documentation AIAlignmentTheory EnglishLanguage "Theoretical principles
and propositions about AI alignment, including inner/outer alignment and
orthogonality thesis.")

(subclass InnerAlignment AIAlignmentTheory)
(subclass OuterAlignment AIAlignmentTheory)
(subclass OrthogonalityThesis AIAlignmentTheory)

;; InstrumentalConvergence MOVED to AIOptimizationProcess (it's a process, not theory)
```

**Lines 600-611 (AIMetaOptimization):**

BEFORE:
```lisp
(subclass AIMetaOptimization Process)
(documentation AIMetaOptimization EnglishLanguage...)

(subclass MesaOptimization AIMetaOptimization)
(subclass MesaOptimizer AIMetaOptimization)
```

AFTER:
```lisp
;; DELETED - children moved to AIOptimizationProcess

(subclass MesaOptimization AIOptimizationProcess)
(documentation MesaOptimization EnglishLanguage...)

(subclass MesaOptimizer AIOptimizationProcess)
(documentation MesaOptimizer EnglishLanguage...)

(subclass InstrumentalConvergence AIOptimizationProcess)
(documentation InstrumentalConvergence EnglishLanguage "The thesis and
observed process that intelligent agents pursue similar instrumental goals
such as self-preservation and resource acquisition.")
```

**Lines 629-634 (AIGovernanceProcess):**

BEFORE:
```lisp
(subclass AIGovernanceProcess IntentionalProcess)
```

AFTER:
```lisp
(subclass AIGovernanceProcess PoliticalProcess)
(documentation AIGovernanceProcess EnglishLanguage...)
```

### Section 3: Move TellingALie (if in this file)

Need to search for TellingALie and move it:

BEFORE:
```lisp
(subclass TellingALie ???)  ;; wherever it currently is
```

AFTER:
```lisp
(subclass TellingALie HumanDeception)
(documentation TellingALie EnglishLanguage "The act of a human agent
deliberately stating falsehoods with intent to deceive.")
```

## Summary of Hierarchy Changes

### Before (broken):
```
Process (L0)
├── AIAlignmentFailureMode (L1) ❌
├── AIMetaOptimization (L1) ❌
└── AIRiskScenario (L1) ❌

IntentionalProcess (L1)
├── AIStrategicDeception (L2) ⚠️
└── AIGovernanceProcess (L2) ⚠️

Proposition (L0)
└── AIAlignmentPrinciple (L1) ⚠️
```

### After (correct):
```
Process (L0)
└── IntentionalProcess (L1)
    ├── ComputationalProcess (L2) ✅ NEW
    │   ├── AIFailureProcess (L3) ✅ NEW
    │   │   ├── GoalMisgeneralization (L4)
    │   │   ├── RewardHacking (L4)
    │   │   └── SpecificationGaming (L4)
    │   └── AIOptimizationProcess (L3) ✅ NEW
    │       ├── MesaOptimization (L4)
    │       ├── MesaOptimizer (L4)
    │       └── InstrumentalConvergence (L4)
    ├── OrganizationalProcess (L2)
    │   └── PoliticalProcess (L3) ✅ NEW
    │       └── AIGovernanceProcess (L4)
    └── SocialInteraction (L2)
        └── Deception (L3) ✅ NEW
            ├── HumanDeception (L4) ✅ NEW
            │   └── TellingALie (L5)
            └── AIStrategicDeception (L4)
                ├── AIDeception (L5)
                ├── DeceptiveAlignment (L5)
                └── TreacherousTurn (L5)

Process (L0)
└── InternalChange (L1)
    ├── Damaging (L2)
    │   └── Catastrophe (L3) ✅ NEW
    │       └── AICatastrophicEvent (L4)
    │           └── GreyGooScenario (L5)
    └── QuantityChange (L2)
        └── RapidTransformation (L3) ✅ NEW
            ├── IntelligenceExplosion (L4)
            └── TechnologicalSingularity (L4)

Proposition (L0)
└── FieldOfStudy (L1)
    └── AIAlignmentTheory (L2) ✅ RENAMED
        ├── InnerAlignment (L3)
        ├── OuterAlignment (L3)
        └── OrthogonalityThesis (L3)
```

## Review Checklist

Before applying:
- [ ] Review all new concept definitions
- [ ] Verify parent-child relationships correct
- [ ] Check that no concepts are orphaned
- [ ] Confirm deletions are appropriate
- [ ] Validate documentation strings
- [ ] Ensure consistent naming conventions

## Risks & Mitigations

**Risk**: Breaking existing references in other KIF files
**Mitigation**: Search for references to deleted/renamed concepts before applying

**Risk**: TellingALie might not be in AI.kif
**Mitigation**: Search all KIF files, update whichever contains it

**Risk**: New concepts might conflict with existing SUMO
**Mitigation**: Catastrophe, RapidTransformation, Deception are generic - should be OK

## Next Steps After Approval

1. Create backup of AI.kif
2. Apply edits
3. Search for TellingALie and update its parent
4. Verify no syntax errors in KIF
5. Move to Phase 3 (WordNet mappings)
