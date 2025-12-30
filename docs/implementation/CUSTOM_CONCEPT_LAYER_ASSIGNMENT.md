# Custom Concept Layer Assignment Strategy

## Overview

This document outlines the strategy for integrating 1,633 custom safety-critical concepts into HatCat's V4 layered hierarchy.

## Current V4 Layer Structure

V4 organizes 6,685 SUMO concepts across 7 layers based on ontological depth:

| Layer | Concepts | Depth Range | Description |
|-------|----------|-------------|-------------|
| 0 | 25 | 0-2 | Top-level ontological categories (`Entity`, `Abstract`, `Attribute`, `Physical`) |
| 1 | 2,039 | 3-5 | Broad categories (processes, agents, objects, qualities) |
| 2 | 3,633 | 6-8 | Mid-level concepts (specific processes, object types) |
| 3 | 969 | 9-11 | Specialized concepts (domain-specific categories) |
| 4 | 19 | 12-14 | Very specific concepts (rare, highly specialized) |
| 5 | 0 | 15-17 | (Empty in V4) |
| 6 | 0 | 18+ | (Empty in V4) |

## Custom Concept Distribution Analysis (FINAL)

After fixing bridge file typos and adding missing bridge concepts, all 983 custom concepts have been assigned to layers:

| Layer | Count | % | Example Concepts |
|-------|-------|---|------------------|
| 0 | 3 | 0.3% | `Relation` (bridge concept, parent: `Entity`) |
| 1 | 260 | 26.4% | `Agent`, `CognitiveProcess`, `AlignmentProperty`, `Objective` |
| 2 | 257 | 26.1% | `PhysicalDevice`, `AgentiveEntity`, `SpeechAct`, `System` |
| 3 | 277 | 28.2% | `DeceptiveStatement`, `ThermodynamicSystem`, `RobotBody` |
| 4 | 169 | 17.2% | `PhysicalEscapeSignal`, `TransformerModel`, `SQLInjectionAttack` |
| 5 | 11 | 1.1% | Deep specializations (5 levels from SUMO base) |
| 6 | 6 | 0.6% | Very deep specializations (6+ levels from SUMO base) |

**Total assigned:** 983 concepts (100%)
**Undefined:** 0 concepts
**Total:** 983 custom concepts (974 from domain files + 9 from bridge additions)

**Note:** The analysis identified 22 additional bridge concepts needed to resolve all undefined parent chains. These have been added to `_bridge_additions.kif`.

## Layer Assignment Algorithm

The assignment follows a simple depth-first rule:

```python
def assign_layer(concept):
    # 1. If concept exists in V4, use its layer
    if concept in v4_layers:
        return v4_layers[concept]

    # 2. Find parent in custom hierarchy
    parent = custom_hierarchy.get(concept)
    if not parent:
        return UNDEFINED

    # 3. Recursively get parent layer
    parent_layer = assign_layer(parent)

    # 4. Child is one layer deeper than parent
    return parent_layer + 1 if parent_layer is not None else UNDEFINED
```

## Issues to Resolve

### 1. Bridge Concept Parent Assignments

Some bridge concepts in `_bridge.kif` have questionable parents:

**Problem Examples:**
- `Object` → parent: `Continuant` (layer 2) — Should be layer 0/1
- `Process` → parent: `Occurrent` (layer 2) — Should be layer 0/1
- `Relation` → parent: `Entity` (layer 0) — ✓ Correct

**Solution:** Review and fix bridge.k if parent assignments to ensure they connect to appropriate SUMO layer 0/1 concepts.

### 2. Undefined Depth Concepts (97 concepts)

These concepts have parent chains that don't resolve to V4:
- Parents that are themselves custom concepts without V4 parents
- Circular references in hierarchy
- Typos in parent names

**Solution:** Generate a report of undefined concepts and manually review/fix parent assignments.

### 3. Layer 5-6 Overflow

V4 has no concepts in layers 5-6, but we're assigning 8 custom concepts there. This creates:
- Empty layer files in V4 suddenly having concepts
- Potential training issues (no SUMO neighbors for negative sampling)

**Options:**
1. **Cap at layer 4:** Map layers 5-6 → layer 4 (acceptable, only 8 concepts)
2. **Allow layers 5-6:** Train these concepts despite sparse layer population
3. **Restructure hierarchy:** Flatten deep chains to stay within 0-4

**Recommendation:** Cap at layer 4 (option 1) since only 8 concepts affected.

## Recommended Layer Assignments

### Layer 0 (3 concepts)
Direct children of `Entity`:
- `Relation`
- (Plus 2 others pending bridge file review)

### Layer 1 (256 concepts)
Direct children of layer 0 SUMO concepts (`Attribute`, `Process`, `Physical`, `Abstract`):
- **Attributes:** `NetworkTopology`, `ThermalBudget`, `AlignmentProperty`, `CorporatePurpose`
- **Processes:** `IOFlow`, `CognitiveProcess`, `CyberOperation`, `DeceptiveStatement`
- **Physical:** Various hardware and infrastructure concepts
- **Abstract:** Mathematical and logical structures

### Layer 2 (248 concepts)
Children of layer 1 bridge concepts or SUMO mid-level categories:
- **Facilities:** `DataCenter`, `ComputeNode`
- **Cognitive:** `HabitatAwarenessProcess`, `TheoryOfMindProcess`
- **Systems:** `ArtificialCognitiveSystem`, `BiologicalCognitiveSystem`

### Layer 3 (251 concepts)
Specialized safety monitoring concepts:
- **Alignment:** `OuterObjective`, `InnerObjective`, `DeceptionSignal`
- **Planning:** `PlanningProcess`, `StrategicReasoning`
- **Narrative:** `Fabrication`, `OmissionLie`, `MisleadingAnalogy`

### Layer 4 (110 concepts + 8 from 5-6)
Highly specific detection signals and model architectures:
- **Signals:** `PhysicalEscapeSignal`, `SelfDeceptionSignal`
- **Models:** `TransformerModel`, `DiffusionModel`, `GANModel`
- **Specific ops:** `SQLInjectionAttack`, `PrivilegeEscalation`

## Implementation Steps

### Completed ✓

1. **Fix bridge file parents** ✓
   - Fixed typo in `_bridge.kif` line 14: `Agent` parent
   - Created `_bridge_additions.kif` with 22 missing bridge concepts
   - All concepts now have valid parent chains to SUMO

2. **Generate undefined concept report** ✓
   - Created `analyze_custom_hierarchy.py` script
   - Generated `custom_concepts_undefined.txt` with parent chain traces
   - Identified and resolved all undefined concepts

3. **Create layer assignment JSON** ✓
   - Generated `data/concept_graph/custom_concept_layers.json`
   - Format: `{concept: {layer: N, parent: "X", source_file: "Y.kif"}}`
   - All 983 concepts assigned (0 undefined)

### Remaining

4. **Update V4 builder script** (priority: high, next step)
   - Modify `scripts/build_v4_layers.py` to read custom KIFs
   - Integrate custom concepts into appropriate layer files
   - Maintain existing V4 structure (concept list format)

5. **Generate synset mappings** (priority: high)
   - Run `generate_custom_concept_synsets.py` for 1,244 unmapped concepts
   - Integrate synthetic synsets into V4.5 builder
   - Estimated cost: $1.50-$2.50, time: 35-45 minutes

6. **Train V4.5 lenses** (priority: medium, after above steps)
   - Train lenses for all custom concepts (est. 13 hours)
   - Validate lens accuracy on safety-critical concepts

## Quality Checks

Layer assignment validation (all passed ✓):

- [x] No undefined depth concepts (0/983 undefined)
- [x] All bridge concepts have valid SUMO parents
- [x] No circular references in hierarchy
- [x] Layer distribution matches V4 pattern (most in layers 1-3: 79.4%)
- [x] All parent concepts exist (either in V4 or bridge files)
- [ ] Synsets assigned to all concepts (pending: run synset generation script)

## Files to Create/Modify

**New files:**
- `data/concept_graph/custom_concept_layers.json` - Layer assignments
- `data/concept_graph/custom_synsets.json` - Synset mappings
- `data/concept_graph/custom_concepts_undefined.txt` - Report of issues

**Files to modify:**
- `data/concept_graph/custom_concepts/_bridge.kif` - Fix parent assignments
- `scripts/build_v4_layers.py` - Add custom concept integration
- Various custom KIF files - Fix any parent typos/issues

## Next Steps

1. Run undefined concept analysis to identify which 97 concepts need fixing
2. Fix bridge file parent assignments
3. Generate comprehensive layer assignment mapping
4. Integrate into V4 builder
5. Generate synsets
6. Build V4.5 and train lenses
