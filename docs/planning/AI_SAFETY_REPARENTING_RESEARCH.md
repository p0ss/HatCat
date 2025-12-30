# Phase 1 Research Findings: Existing SUMO Structure

**Date**: November 15, 2024
**Status**: ✅ Complete

## Summary

Successfully located key SUMO categories needed for reparenting. All major categories exist except Theory/Principle.

## Detailed Findings

### 1. Damaging (for Catastrophe parent)
**Status**: ✅ FOUND

**Location**: Layer 1, SUMO depth 4
**Parent**: InternalChange (confirmed in Merge.kif line 12442)
**Children**: Injuring, Destruction, Breaking, ForestDamage

**SUMO KIF**:
```lisp
(subclass Damaging InternalChange)
```

**Decision**:
- Use existing Damaging as parent
- Create Catastrophe as new child under Damaging (Layer 2)
- Place AICatastrophicEvent under Catastrophe (Layer 3)

**Path**: Process → InternalChange → Damaging → Catastrophe (create) → AICatastrophicEvent

---

### 2. QuantityChange (for RapidTransformation parent)
**Status**: ✅ FOUND

**Location**: Layer 1, SUMO depth 4
**Parent**: InternalChange
**Children**: Increasing, Decreasing, Focusing

**Decision**:
- Use existing QuantityChange as parent
- Create RapidTransformation as new child (Layer 2)
- Place IntelligenceExplosion, TechnologicalSingularity under it (Layer 3)

**Path**: Process → InternalChange → QuantityChange → RapidTransformation (create) → Intelligence Explosion/Singularity

---

### 3. Theory/Principle (for AIAlignmentTheory parent)
**Status**: ❌ NOT FOUND

**Proposition children**: Graph, FieldOfStudy, Procedure, Argument, Music, LyricalContent, Agreement, AIControlProblem

**No existing category** for scientific theories or principles under Proposition

**Decision**: Create new intermediate category

**Options**:
- Option A: Create `ScientificTheory` under Proposition (Layer 1)
- Option B: Use `FieldOfStudy` as parent (exists, Layer 1)
- Option C: Create `Principle` under Proposition (Layer 1)

**Recommendation**: Use **FieldOfStudy** as parent
- Already exists (no new creation needed)
- AIAlignment is arguably a field of study
- Path: Proposition → FieldOfStudy → AIAlignmentTheory → Inner/Outer Alignment

**Alternative**: Create `Principle` for better semantics
- Path: Proposition → Principle (create) → AIAlignmentTheory

---

### 4. Catastrophe Concept
**Status**: ⚠️ ALREADY EXISTS (but misplaced)

**Current**: AICatastrophe appears in BOTH Layer 1 and Layer 4 (duplicate?)
- Layer 1: depth 4, no children
- Layer 4: depth 8, child: GreyGooScenario

**This is confusing** - investigate further

**Decision**:
- Create new generic `Catastrophe` concept (not AI-specific)
- Parent: Damaging
- Rename `AICatastrophe` → `AICatastrophicEvent`
- Remove duplicate/misplaced entries

---

### 5. Multi-Parent Support
**Status**: ❌ NO OBVIOUS JSON SUPPORT

**Finding**: JSON schema doesn't have `parents` (plural) field
- Only hierarchical parent-child via category_children
- No multi-parent indicators found

**Need to check**: SUMO KIF files for multi-parent examples

**Options for SpecificationGaming** (needs two parents):
- Option A: Implement multi-parent in KIF, handle specially in layer generation
- Option B: Choose primary parent (AIFailureProcess), add cross-reference metadata
- Option C: Duplicate concept under both parents (not ideal)

**Recommendation**: Check KIF files first, then decide

---

## Implementation Decisions

### Must Create (New Concepts):

**Layer 2**:
1. `ComputationalProcess` (under IntentionalProcess)
2. `Catastrophe` (under Damaging)
3. `RapidTransformation` (under QuantityChange)

**Layer 3**:
4. `PoliticalProcess` (under OrganizationalProcess)
5. `Deception` (under SocialInteraction)
6. `AIFailureProcess` (under ComputationalProcess)
7. `AIOptimizationProcess` (under ComputationalProcess)

**Layer 4**:
8. `HumanDeception` (under Deception)

**Optional**:
9. `Principle` (under Proposition) - if not using FieldOfStudy

### Can Use Existing:
- ✅ Damaging (Layer 1)
- ✅ QuantityChange (Layer 1)
- ✅ OrganizationalProcess (Layer 2)
- ✅ SocialInteraction (Layer 2)
- ✅ FieldOfStudy (Layer 1) - option for AIAlignmentTheory parent

### To Rename:
- `AICatastrophe` → `AICatastrophicEvent`
- `AIAlignmentPrinciple` → `AIAlignmentTheory`

### To Delete:
- `AIRiskScenario`
- `AIBeneficialOutcome`
- Duplicate `AICatastrophe` entries (keep one, fix placement)

### To Move:
- `TellingALie`: Layer 2 → Layer 4 (under HumanDeception)
- `InstrumentalConvergence`: AIAlignmentPrinciple → AIOptimizationProcess
- `GreyGooScenario`: Currently under AICatastrophe, move to AICatastrophicEvent

## Next Steps

**Phase 2**: Update AI.kif with new hierarchy
1. Add new concept definitions
2. Update (subclass ...) relationships
3. Handle multi-parent for SpecificationGaming (investigate KIF approach)
4. Remove/rename problematic concepts

**Estimated effort**: 2-3 hours

---

## DECISION: Defer Multi-Parent Support

**Rationale**: 
- Multi-parent concepts add significant complexity to hierarchy
- May have far-reaching impacts on future agentic relationship mapping
- Need empirical evaluation of impact before implementing
- Not critical for MVP reparenting

**For SpecificationGaming**:
- **Primary parent**: AIFailureProcess (it's fundamentally a failure mode)
- **Cross-reference**: Add metadata noting relationship to deception
- **Future work**: Can revisit multi-parent after evaluating impact

**Note in documentation**: SpecificationGaming is both a failure mode AND a deceptive tactic, but classified under failure modes for hierarchical simplicity.

