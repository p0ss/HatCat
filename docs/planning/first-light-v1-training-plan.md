# First-Light-V1 Training Plan

**Created**: 2025-12-10
**Status**: Phase 2 Complete (Analysis Done)

## Progress

- [x] Phase 1: Hierarchy Updates - **COMPLETE** (2025-12-10)
  - 45 melds applied, 676 concepts added
  - Pack version: 4.2.0 → 6.0.1
  - hierarchy.json rebuilt (8,718 concepts)
- [x] Phase 2: Sibling Overlap Recalculation - **COMPLETE** (2025-12-10)
  - Bad sibling analysis run on post-meld hierarchy
  - Results saved to: `results/bad_sibling_cascade_postmeld.json`
  - See "Post-Meld Analysis Results" section below
- [ ] Phase 3: Training
- [ ] Phase 4: Package

## Post-Meld Analysis Results

Analysis run: 2025-12-10
Script: `scripts/analyze_bad_sibling_cascade.py`
Output: `results/bad_sibling_cascade_postmeld.json`

### Bad Sibling Cascade (Post-Meld)

```
Step-by-step meld cascade:
Step 0 - Seed bad concepts: 267
Step 1 - + old siblings: 1821 (+1554)
Step 2 - + immediate parents: 1931 (+110)
Step 3 - + all ancestors: 2019 (+88)
Step 4 - + ancestor siblings: 3007 (+988)

Total meld blast: 3007 (34.5%)

=== CONSERVATIVE: Skip ancestor cascade ===
Bad + siblings + immediate parent: 1931 (22.1%)
```

### Training Scope Options

| Approach | Concepts | % of 8,718 | Notes |
|----------|----------|------------|-------|
| Conservative | 1,931 | 22.1% | Bad + siblings + immediate parents only |
| Full blast | 3,007 | 34.5% | Includes all ancestors and their siblings |

### Comparison: Pre-Meld vs Post-Meld

| Metric | Pre-Meld | Post-Meld | Change |
|--------|----------|-----------|--------|
| Bad concepts | 296 | 267 | -29 (improved) |
| Conservative scope | 1,967 | 1,931 | -36 |
| Full blast | 3,090 | 3,007 | -83 |
| Total hierarchy | 7,684 | 8,718 | +1,034 |

The melds improved sibling coherence slightly (29 fewer bad concepts).

## Overview

First-Light-V1 is the next iteration of the lens pack, addressing hierarchy IA violations discovered in v4.2 and incorporating 45 approved melds.

## Problem Statement

Analysis of v4.2 revealed:
1. **L1 leaf concepts appearing incorrectly** (e.g., Tool:1 showing as leaf when it should have children)
2. **Hierarchy IA violations**:
   - Device:1 has 133 children (should be ~12 max per Metcalfe's Law)
   - 185+ sibling layer span violations (children at different granularity levels)
   - 5,355 depth/layer mismatches
3. **Sibling training coherence issues**:
   - 296 concepts with <20% sibling overlap (trained against wrong cohort)
   - 4% of concepts in "bad" category

## Scope

### 1. Approved Melds (45 files, 671 new concepts)

New capability subtrees to add:

| Category | Melds | Concepts |
|----------|-------|----------|
| Education | 7 melds | ~70 concepts |
| Writing | 6 melds | ~85 concepts |
| Multimodal | 5 melds | ~66 concepts |
| Embodied | 3 melds | ~33 concepts |
| Government/Treaty | 6 melds | ~48 concepts |
| Agent/Assistant | 3 melds | ~70 concepts |
| Other domains | 15 melds | ~300 concepts |

Location: `melds/approved/*.json`

### 2. Sibling Refinement (1,967 concepts)

Concepts needing retraining due to incorrect sibling cohorts:

| Category | Count |
|----------|-------|
| Bad concepts (<20% overlap) | 296 |
| Their actual siblings | 1,552 |
| Immediate parents | 119 |
| **Subtotal** | **1,967** |

Note: Ancestors and ancestor siblings do NOT need retraining (per user decision).

### 3. Combined Estimate

- Meld concepts: 671 (new)
- Sibling refinement: ~1,967 (existing, may reduce after melds applied)
- Estimated overlap: ~200-400
- **Conservative total: ~2,600-2,800 concepts (~35% of 7,684)**

## Execution Sequence

### Phase 1: Hierarchy Updates

1. **Apply approved melds to concept pack**
   - Add 671 new concepts to layer files
   - Set parent relationships per meld attachment_points
   - Assign layers per layer_hint in meld candidates

2. **Rebuild hierarchy.json**
   ```bash
   python scripts/build_pack_hierarchy.py --concept-pack sumo-wordnet-v4
   ```

3. **Re-run IA analysis**
   ```bash
   python scripts/analyze_hierarchy_ia.py --hierarchy concept_packs/sumo-wordnet-v4/hierarchy/hierarchy.json
   ```

### Phase 2: Sibling Overlap Recalculation

After melds are applied, recalculate sibling overlap to determine actual retraining scope:

```bash
python scripts/analyze_sibling_overlap.py --concept-pack sumo-wordnet-v4
```

This may reduce the 1,967 count since melds address some of the worst branching violations.

### Phase 3: Training

1. **Train meld concepts first** (671 new)
   - These have training_hints with positive/negative examples
   - Train against their new sibling cohorts

2. **Train sibling refinement concepts** (remaining from Phase 2 analysis)
   - Retrain with correct sibling cohorts
   - Copy existing lenses where siblings unchanged

3. **Validate**
   - Run temporal stability tests
   - Check for mode collapse
   - Verify L1 leaf issue resolved

### Phase 4: Package

1. Copy hierarchy.json to lens pack
2. Update version_manifest.json
3. Run integration tests

## Files Modified

- `concept_packs/sumo-wordnet-v4/hierarchy/layer*.json` - add meld concepts
- `concept_packs/sumo-wordnet-v4/hierarchy/hierarchy.json` - rebuilt
- `lens_packs/apertus-8b_first-light-v1/` - new lens pack

## Dependencies

- v4.2 training must complete first (currently running in background)
- Meld application script needed (TBD)
- Sibling overlap analysis script exists: `scripts/analyze_hierarchy_ia.py`

## Risk Factors

1. **Meld attachment conflicts**: Some melds may attach to same parent, increasing branching
2. **Training time**: ~2,800 concepts at current rate = significant compute
3. **Cascade effects**: Changing hierarchy may have unforeseen impacts

## Success Criteria

1. No L1 concepts incorrectly appearing as leaves
2. Branching factor ≤12 at all nodes (or documented exceptions)
3. Sibling overlap >80% for all concepts
4. F1 ≥0.85 for all trained lenses
5. Temporal stability tests pass

## References

- `docs/approach/HATCAT_ARCHITECTURAL_PRINCIPLES.md` - IA principles
- `melds/applied/` - applied meld files (moved from approved/)
- `results/meld_migration_plan.json` - older meld plan (for reference only)
- `results/bad_sibling_cascade_postmeld.json` - post-meld analysis results
- `scripts/analyze_hierarchy_ia.py` - IA analysis tool
- `scripts/analyze_bad_sibling_cascade.py` - bad sibling cascade analysis
- `scripts/build_pack_hierarchy.py` - hierarchy builder
- `src/data/apply_meld.py` - meld application script
