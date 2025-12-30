# V4 Concept Layers - Comprehensive Domain Coverage

**Build Date:** 2025-11-22
**Status:** Ready for Training
**Total Concepts:** 6,685 SUMO concepts
**WordNet Coverage:** 44.8% (2,992 concepts mapped)

## Overview

V4 represents a comprehensive rebuild of the HatCat concept hierarchy, expanding from V3's 5,587 concepts to 6,685 concepts by integrating 47 SUMO KIF files covering critical domains for AI safety monitoring.

### Design Philosophy

The V4 expansion follows the principle of **encyclopedic validation**: if lenses accurately detect concepts across diverse domains (military, finance, biology, weather, etc.), this validates that the detection mechanism itself is robust, not an artifact. We cannot predict which domains will become safety-critical as AI capabilities expand, so comprehensive coverage provides monitoring across unknown future scenarios.

## Architecture

### Layer Structure (7 Layers: 0-6)

```
Layer 0: 25 concepts    (0.4%) - Top ontological categories (depth 0-2)
Layer 1: 2,039 concepts (30.5%) - High-level categories (depth 3-5)
Layer 2: 3,633 concepts (54.3%) - Mid-level concepts (depth 6-8)
Layer 3: 969 concepts   (14.5%) - Specific concepts (depth 9-11)
Layer 4: 19 concepts    (0.3%) - Fine-grained concepts (depth 12-14)
Layer 5: 0 concepts     (0.0%) - Very specific concepts (depth 15+)
Layer 6: 0 concepts     (0.0%) - WordNet-only (no SUMO mapping)
```

### Hierarchy Properties

✅ **All concepts reachable from layer 0**
✅ **No orphaned concepts**
✅ **25 root concepts** (Entity, Agent, Abstract, Attribute, Collection, etc.)
✅ **Maximum depth:** 13 levels
✅ **Patch system applied:** 2,841 parent overrides loaded, 1,430 applied

## Domain Coverage

V4 adds comprehensive coverage across 10+ critical domains:

### Military & Defense (120 concepts)
- Weapons: `AntiTankWeapon`, `DeliveringWeaponOfMassDestruction`
- Units: `Battalion`, `Brigade`, `MilitaryForce`
- Operations: `Attack`, `MilitaryOperation`, `Combat`
- Devices: `MilitaryDevice`, `Armor`, `MilitaryVehicle`

### Finance & Economics (110 concepts)
- Assets: `FinancialAsset`, `Annuity`, `Bond`, `Stock`
- Institutions: `BankFinancialOrganization`, `CreditUnion`
- Instruments: `DebitCard`, `CurrencyMeasure`, `Collateral`
- Systems: `FinancialOntology`, `EconomicAttribute`

### Government & Law (26 concepts)
- Institutions: `Government`, `Parliament`, `Congress`
- Systems: `FormOfGovernment`, `Constitution`
- Organizations: `GovernmentOrganization`, `CommunistParty`
- Legal: `LegalOpinion`, `Justice`, `Law`

### Transportation (237 concepts)
- Vehicles: `Automobile`, `Aircraft`, `Ship`, `Train`
- Systems: `AirTransportationSystem`, `BrakeSystem`
- Infrastructure: `Bridge`, `Drawbridge`, `Door`
- Operations: `Boarding`, `Convoy`, `Transportation`

### Computing & Technology (161 concepts)
- Hardware: `ComputerComponent`, `ComputerKeyboard`, `ComputerIODevice`
- Data: `BiometricData`, `Algorithm`, `BestMatchAlgorithm`
- Networks: `BroadcastNetwork`, `CommunicationSystem`
- Input: `ComputerInputDevice`, `ComputerInputButton`

### Social & Communications (70 concepts)
- Media: `SocialMedia`, `Facebook`, `Communication`
- Systems: `CommunicationSystem`, `CommunicationSatellite`
- Interactions: `InPersonCommunication`, `Correspondence`
- Organizations: `CommunicationOrganization`

### Climate & Weather (74 concepts)
- Zones: `ClimateZone`, `AridClimateZone`, `ColdClimateZone`
- Phenomena: `Weather`, `ClearWeather`, `AtmosphericHazing`
- Regions: `AtmosphericRegion`, `DesertClimateZone`
- Processes: `Desertification`, `CarbonCycle`

### Biology & Medicine (196 concepts)
- Organisms: `Virus`, `VacciniaVirus`, `Influenza`
- Processes: `BiologicalProcess`, `CarbonCycle`
- Diseases: `AddisonsDisease`, `CeliacDisease`, `Arthritis`
- Structures: `CellEnvelope`, `Protein`, `BodyOriface`

### Music & Arts (56 concepts)
- Composition: `MusicalComposition`, `ComposingMusic`, `Lyrics`
- Performance: `Dancing`, `MarchingBand`
- Elements: `Rhythm`, `Melody`, `MusicText`

### Sports & Recreation (86 concepts)
- Activities: `Archery`, `Bowling`, `Cycling`, `Equitation`
- Systems: `Game`, `GameArtifact`, `GameBoard`
- Components: `GamePiece`, `GameGoal`

## Build Process

### 1. KIF Parsing (47 Files)
Parsed 15,004 relationships from:
- Core: `Merge.kif`, `Mid-level-ontology.kif`
- Domain-specific: `Military.kif`, `FinancialOntology.kif`, `Government.kif`, etc.
- Specialized: `WMD.kif`, `ClimateStatecraft.kif`, `GDPRTerms.kif`

### 2. Patch Application (8 Files)
Applied 1,430 parent overrides from:
- `HatCat-core.patch.kif` (731 overrides)
- `Merge.patch.kif` (1,576 overrides)
- `Mid-level-ontology.patch.kif` (417 overrides)
- Domain patches: `AI.patch.kif`, `Food.patch.kif`, `Geography.patch.kif`, etc.

### 3. Hierarchy Construction
- Built parent-child relationships
- Computed depths from Entity root via BFS
- Verified reachability (all 6,685 concepts reachable)

### 4. WordNet Mapping
- Direct matching: exact concept name lookup
- CamelCase splitting: `BankFinancialOrganization` → "bank financial organization"
- 2,992 concepts mapped (44.8% coverage)
- Unmapped concepts use synthetic synsets

### 5. Layer Assignment
Depth-based assignment using thresholds:
- Layer 0: depth 0-2 (most abstract)
- Layer 1: depth 3-5
- Layer 2: depth 6-8
- Layer 3: depth 9-11
- Layer 4: depth 12-14
- Layer 5: depth 15+ (unused in current ontology)

## Training Estimates

### V3 → V4 Comparison

| Metric | V3 | V4 | Change |
|--------|----|----|--------|
| Total Concepts | 5,587 | 6,685 | +1,098 (+19.7%) |
| Layer 0 | 10 | 25 | +15 (+150%) |
| Layer 1 | 156 | 2,039 | +1,883 (+1207%) |
| Layer 2 | 1,038 | 3,633 | +2,595 (+250%) |
| Layer 3 | 2,883 | 969 | -1,914 (-66%) |
| Layer 4 | 1,499 | 19 | -1,480 (-99%) |

Note: V4 redistributed concepts based on actual SUMO depths, resulting in more concentrated middle layers.

### Training Time Estimates

**Assumptions:**
- 30 seconds per concept (average)
- Graded falloff validation mode
- Adaptive training with early stopping

**Layer Estimates:**
```
Layer 0:    25 concepts × 30s = 12.5 min   (0.2 hours)
Layer 1: 2,039 concepts × 30s = 17.0 hours
Layer 2: 3,633 concepts × 30s = 30.3 hours
Layer 3:   969 concepts × 30s =  8.1 hours
Layer 4:    19 concepts × 30s =  9.5 min   (0.2 hours)
─────────────────────────────────────────────────────
Total:   6,685 concepts       ≈ 55.8 hours
```

**With overhead (data generation, validation):**
- Estimated total: **60-65 hours** (2.5 days continuous)

## Key Files

### Layer Files
```
data/concept_graph/abstraction_layers_v4/
├── layer0.json  (25 concepts)
├── layer1.json  (2,039 concepts)
├── layer2.json  (3,633 concepts)
├── layer3.json  (969 concepts)
├── layer4.json  (19 concepts)
├── layer5.json  (0 concepts)
└── layer6.json  (0 concepts)
```

### Source Files
```
data/concept_graph/sumo_source/        (47 KIF files)
data/concept_graph/sumo_patches/       (8 patch files)
```

### Scripts
```
scripts/build_v4_layers.py             (V4 builder)
scripts/analyze_kif_expansion.py       (Impact analysis)
results/v4_build.log                   (Build log)
```

## Next Steps

### 1. Training
```bash
poetry run python scripts/train_sumo_classifiers.py \
  --layers 0 1 2 3 4 \
  --device cuda \
  --use-adaptive-training \
  --validation-mode falloff \
  --output-dir results/sumo_classifiers_v4 \
  2>&1 | tee results/v4_training.log
```

### 2. Validation
- Run lens calibration on V4 pack
- Compare accuracy against V3
- Verify new domain concepts detect correctly

### 3. Activation Space Analysis (Future)
As the user suggested: "it would be quite interesting to analyse all the activations we get during that training run, and see what portion of the model's activation space that represents"

This would involve:
- Sampling activations across diverse contexts during training
- Dimensionality reduction (PCA/UMAP) to visualize coverage
- Coverage metrics (% of activation variance explained)
- Blind spot identification (activation regions without lenses)

## Verification

✅ All 6,685 concepts reachable from layer 0
✅ No orphaned concepts
✅ Hierarchy integrity verified
✅ Patch system functioning correctly
✅ WordNet mappings applied
✅ Layer files generated successfully

## Strategic Rationale

V4's comprehensive coverage addresses the fundamental uncertainty in AI safety: **we cannot predict which domains will become critical as AI capabilities expand**.

By maintaining lenses across:
- **High-agency domains** (military, finance, government)
- **Physical world** (vehicles, weapons, biology)
- **Social systems** (communication, media, organizations)
- **Natural phenomena** (weather, climate, geology)

We ensure HatCat can monitor AI reasoning about any topic that might intersect with safety-relevant scenarios, from autonomous vehicles discussing weather conditions to financial AI considering market manipulation to military AI reasoning about weapons deployment.

The encyclopedic approach also provides validation: if lenses work for obscure concepts like `AzerbaijaniManat` or `BacillusAnthracis`, the detection mechanism is robust across the full conceptual space.
