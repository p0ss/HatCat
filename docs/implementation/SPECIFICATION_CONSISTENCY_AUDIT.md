# Specification Consistency Audit

> **Purpose**: Identify and fix internal inconsistencies across HatCat specification documents
> **Date**: 2025-12-06
> **Status**: COMPLETED (high/medium priority items)

---

## Executive Summary

The specification documents have evolved as the design was refined. This audit identified inconsistencies where older documents contained outdated terminology (PEFT/LoRA, AAL, autopilot) that didn't reflect the latest architectural decisions.

**Key fixes made:**
- Replaced PEFT/LoRA references with Graft Protocol terminology throughout
- Replaced AAL with BE (Bounded Experiencer) - no historical retention
- Replaced "autopilot" with "autonomic core" / "autonomic simplex core"
- Added cross-references to MAP_GRAFTING.md from parent documents
- Documented that simplex lenses defined by USH are always active for autonomic steering
- Clarified distinction between concept lenses (hierarchical, parent-activated) and simplex lenses (always-on)

---

## 1. Critical: PEFT/LoRA vs Graft Architecture

### The Problem

Several documents still reference "PEFT", "LoRA", or "fine-tuning" in vague terms, when the architecture has been refined to use the **Graft Protocol** - a novel accretive approach where:

- Each concept gets a **labelled dimension** added to the substrate
- **Biases** to existing weights encode relational structure
- The substrate **grows** rather than being re-parameterised

### Documents Needing Updates

#### ARCHITECTURE.md (lines 90-130)

Current text references layers vaguely but doesn't mention that Layer 2/BE can grow the substrate via Grafting. Should reference:
- MAP_GRAFTING.md for the Graft Protocol
- The accretive dimension model

#### BOUNDED_EXPERIENCER.md (lines 1-50)

References "continual learning" but doesn't specify the Graft mechanism. Relevant sections:
- Line ~15: "Hushed: MAY run (optional) LoRA or harness-based steering" - **LoRA should be updated to reference Graft/steering biases**
- Line ~30-40: The "learning" submodule description should explicitly reference MAP_GRAFTING.md

#### BE_CONTINUAL_LEARNING.md (lines ~220-240)

Contains explicit PEFT/LoRA language that's inconsistent with the Graft architecture defined in MAP_GRAFTING.md:

```
Line ~227: "Lightweight PEFT/LoRA-style substrate adaptation"
Line ~240: References "PEFT patches"
```

This is contradictory - the same document's earlier sections describe the Graft architecture (dimension expansion + biases) but then reverts to PEFT terminology in the summary.

**Recommendation**: Replace all PEFT/LoRA references with:
- "Graft" for concept-dimension additions
- "substrate bias updates" for weight modifications
- Reference MAP_GRAFTING.md as the normative specification

---

## 2. Terminology Inconsistencies: AAL vs BE

### The Problem

The documents inconsistently use "AAL" (Autonomous Aligned Learner) and "BE" (Bounded Experiencer) to refer to the same thing.

### Resolution

**FIXED**: All AAL references have been updated to BE. AAL is not retained for historical purposes - BE is the canonical term.

### Related Terminology Clarifications

- **Concept lenses**: Dual-pole lenses trained per concept, using hierarchical loading in the dynamic lens manager so they only activate when their parent concept activates.
- **Simplex lenses**: Always-active steering lenses that strongly relate to behaviours governed by USH. The ASK defines the minimum set of simplex autonomic steering lenses that are always on to enforce USH. A BE can request additional simplexes if ASK governance permits.
- **Autopilot** (old term): Now called "autonomic core" or "autonomic simplex core" - the always-on simplex-driven steering system.

---

## 3. Missing Cross-References

### MAP_GRAFTING.md

Excellent standalone document, but several other documents should reference it:

| Document | Should Reference | Current State |
|----------|-----------------|---------------|
| ARCHITECTURE.md | MAP_GRAFTING.md for substrate growth | No reference |
| BOUNDED_EXPERIENCER.md | MAP_GRAFTING.md for learning mechanism | No reference |
| BE_CONTINUAL_LEARNING.md | MAP_GRAFTING.md | Self-inconsistent (describes Graft, then says PEFT) |
| MINDMELD_ARCHITECTURAL_PROTOCOL.md | MAP_GRAFTING.md | Has "See also" but could be stronger |
| ASK (AGENTIC_STATE_KERNEL.md) | MAP_GRAFTING.md for capability tracking | No reference |

### CAT Integration

The CAT specification (HAT_CONJOINED_ADVERSARIAL_TOMOGRAPHY.md) is well integrated with:
- HEADSPACE_AMBIENT_TRANSDUCER.md ✓
- AGENTIC_STATE_KERNEL.md ✓ (Section 2.6)
- HATCAT_MELD_POLICY.md ✓

But should also be referenced from:
- MAP_GRAFTING.md - **Has CAT section (10.3)** ✓
- MAP_MELDING.md - **Has CAT section (13.8)** ✓

### XDB/XAPI Integration

The Experience Database documents (BE_REMEMBERING_XDB.md, BE_REMEMBERING_XAPI.md) should be cross-referenced from:
- BE_CONTINUAL_LEARNING.md (references XDB but could be stronger)
- MAP_GRAFTING.md - mentions XDB as source of truth but doesn't reference XAPI tools

---

## 4. Structural/Naming Inconsistencies

### Concept Pack Versioning

Different documents use different version formats:

| Document | Example Version |
|----------|----------------|
| MAP_MELDING.md | `org.hatcat/sumo-wordnet-v4@4.0.0` |
| MAP_GRAFTING.md | `org.hatcat/sumo-wordnet-v4@4.0.0` |
| ASK | `org.hatcat/sumo-wordnet-v4@4.0.0` |

**Status**: Consistent ✓

### Lens Pack Versioning

| Document | Format |
|----------|--------|
| MAP_MELDING.md | `<date>.<sequence>` e.g. `20251130.0` |
| Actual code | Uses different format in lens_packs directory |

**Recommendation**: Verify code matches spec

### Simplex vs Concept Terminology

The documents sometimes blur the distinction between:
- **Concept lenses**: Hierarchical discrimination (is this Fish vs not-Fish?)
- **Simplex lenses**: Intensity tracking (how strong is autonomy drive?)

MAP_MELDING.md Section 12 clarifies this well, but earlier documents (BOUNDED_EXPERIENCER.md, ARCHITECTURE.md) should reference this distinction.

---

## 5. Document-Specific Issues

### BOUNDED_EXPERIENCER.md

1. **Line 93-95**: References "autopilot" but this term isn't defined elsewhere
2. **Missing**: Should have explicit section on Graft integration
3. **Missing**: Should reference MAP_GRAFTING.md for substrate growth mechanism

### BE_CONTINUAL_LEARNING.md

1. **Section 3.5 (lines ~220-240)**: Summary contradicts the detailed architecture
   - Says "PEFT patches" but earlier sections describe dimension expansion
   - **Action**: Replace with Graft terminology

2. **TrainingRun schema** (lines ~110-130):
   - References `peft_config` - should this be `graft_config`?
   - **Action**: Audit all schema references

3. **Line ~175**: References "low-rank adapters" - outdated
   - **Action**: Replace with "substrate bias updates"

### MAP_MELDING.md

1. **Line 615**: `"accept_with_peft": false` - PEFT terminology
   - **Action**: Update to `"accept_with_graft": false`

2. **Section 6.2** (Cross-BE Propagation): Uses `PackDiff` but should also reference `GraftDiff` from MAP_GRAFTING.md

### ARCHITECTURE.md

1. **Missing**: No mention of substrate growth via Grafting
2. **Missing**: No reference to CAT as Layer 2.5
3. **Should add**: Brief summary of how MAP_GRAFTING.md extends the base architecture

### AGENTIC_STATE_KERNEL.md

1. **Line ~486-495**: Section on "MAP & BE" mentions lenses and ConceptDiffs but not Grafts
   - **Action**: Add reference to GraftDiff propagation

2. **Evidence and Qualifications**: Should reference Graft artifacts as evidence

---

## 6. Recommended Actions (Priority Order)

### High Priority - COMPLETED

1. **~~Update BE_CONTINUAL_LEARNING.md~~** ✓
   - Document was already updated to use Graft terminology

2. **~~Update BOUNDED_EXPERIENCER.md~~** ✓
   - Replaced LoRA reference with Graft/bias steering
   - Replaced "autopilot" with "autonomic core" throughout
   - Updated state machine state names

3. **~~Update HATCAT_MELD_POLICY.md~~** ✓
   - Replaced "AAL" with "BE" throughout

### Medium Priority - COMPLETED

4. **~~Update ARCHITECTURE.md~~** ✓
   - Added substrate growth via Grafting references
   - CAT already included as Layer 2.5 pattern
   - Added MAP_GRAFTING.md references

5. **~~Update MAP_MELDING.md~~** ✓
   - Replaced `accept_with_peft` with `accept_with_graft`

6. **~~Update AGENTIC_STATE_KERNEL.md~~** ✓
   - Added GraftDiff to evidence tracking
   - Added simplex steering lens documentation

### Additional Fixes Made

7. **~~Update BE_WAKING.md~~** ✓
   - Added clarifying note that bootstrap adapters are distinct from Graft-based concept learning
   - Updated terminology to reference Grafts as primary learning mechanism

8. **~~Update BE_REMEMBERING_XDB.md~~** ✓
   - Replaced PatchArtifact schema with GraftArtifact schema

### Remaining (Low Priority)

9. **Verify code-spec alignment**
   - Check lens pack versioning format
   - Check training configuration schemas

10. **Add glossary**
    - Create GLOSSARY.md defining terms: BE, Graft, Simplex, Lens, etc.
    - Cross-reference from all documents

---

## 7. Cross-Reference Matrix

This shows which documents should reference which:

| From \ To | ARCH | BE | BE_CL | HAT | CAT | MAP_G | MAP_M | ASK | XDB |
|-----------|------|----|----|-----|-----|-------|-------|-----|-----|
| ARCHITECTURE.md | - | ✓ | ⚠ | ✓ | ⚠ | ❌ | ✓ | ✓ | ⚠ |
| BOUNDED_EXPERIENCER.md | ✓ | - | ✓ | ✓ | ⚠ | ❌ | ⚠ | ✓ | ✓ |
| BE_CONTINUAL_LEARNING.md | ⚠ | ✓ | - | ⚠ | ⚠ | ❌ | ✓ | ⚠ | ✓ |
| HAT | ✓ | ⚠ | ⚠ | - | ✓ | ⚠ | ✓ | ✓ | ⚠ |
| CAT | ✓ | ✓ | ⚠ | ✓ | - | ⚠ | ✓ | ✓ | ⚠ |
| MAP_GRAFTING.md | ⚠ | ⚠ | ✓ | ✓ | ✓ | - | ✓ | ⚠ | ✓ |
| MAP_MELDING.md | ⚠ | ✓ | ✓ | ⚠ | ✓ | ✓ | - | ⚠ | ⚠ |
| ASK | ✓ | ✓ | ⚠ | ✓ | ✓ | ❌ | ✓ | - | ⚠ |

Legend:
- ✓ = Referenced appropriately
- ⚠ = Could be stronger / mentioned but not linked
- ❌ = Missing reference (should be added)

---

## 8. Schema Consistency Check

### TrainingRun Schema

| Field | BE_CONTINUAL_LEARNING | MAP_GRAFTING | Status |
|-------|----------------------|--------------|--------|
| `type` | lens_only, peft | graft, lens_only, cotrain | **MISMATCH** |
| `peft_config` | Present | Not present (uses graft_config implicitly) | **MISMATCH** |
| `substrate_biases` | Not present | Present | MAP_GRAFTING is source of truth |

**Action**: Update BE_CONTINUAL_LEARNING.md TrainingRun schema to match MAP_GRAFTING.md

### Graft Schema

MAP_GRAFTING.md defines comprehensive Graft schema. Other documents should not redefine it, only reference it.

---

## 9. Conclusion

The core architecture is sound and the detailed specifications (MAP_GRAFTING.md, MAP_MELDING.md, CAT spec) are thorough and consistent with each other. The main work is:

1. Updating terminology in older documents (PEFT→Graft, AAL→BE)
2. Adding cross-references to MAP_GRAFTING.md from parent documents
3. Ensuring schema definitions are consistent

The accretive Graft architecture is a significant improvement over PEFT/LoRA and should be consistently referenced throughout.
