# Hierarchy Restructure Rules

Guidelines and principles used when restructuring the concept hierarchy for optimal cognitive grouping and inference.

## Core Principle

**Siblings should co-activate** - concepts at the same level should be cognitively related, not just taxonomically similar. The goal is groupings that make sense for top-k=10 inference, where seeing a parent should prime you to expect its children.

## Branching Factor Target

- **Target**: ~10 children per parent (for top-k=10 inference)
- **Acceptable**: Up to 15 children
- **Mega-parent**: >15 children, needs subdivision or restructuring

## Naming Conventions

### Self-Descriptive Labels
Group names should be self-descriptive so the label alone conveys context without needing to see the parent:

**Pattern**: `{Domain}{Type}` or `{Domain}{Suffix}`

Examples:
- `FinancialTrading`, `FinancialAccounts`, `FinancialStocks` (not "TradingOperations")
- `BusinessOrgs`, `GovernmentOrgs`, `MediaOrgs` (not "BusinessOrganizations" or just "Business")
- `WaterSports`, `CombatSports`, `WinterSports`
- `FoodAttrs`, `MediaAttrs`, `LegalAttrs`
- `BeliefStates`, `IdentityStates`, `PolicyStates`

### Consistency Within Parent
All siblings should use the same suffix pattern:
- Under Organization: all end in `Orgs`
- Under RelationalAttribute: all end in `Attrs`
- Under CognitiveState: all end in `States`
- Under Sport: all end in `Sports`

### Markers for New/Moved Concepts
- `NEW:ConceptName` - newly created intermediate grouping
- `MOVED:ConceptName` - concept moved from elsewhere in tree

## Restructuring Strategies

### 1. Domain Grouping
Group children by their domain/field:
- Elements → Periodic table groups (NobleGases, TransitionMetals, Lanthanides)
- Financial transactions → By operation type (Trading, Accounts, Transfers)
- Organizations → By sector (Business, Government, Media, Service)

### 2. Signal Placement
Detection signals should be placed **near what they detect**, not grouped together:
- `DeceptionSignal` → sibling of `Deception`
- `CorporateIntentSignal` → inside `Corporation`
- `BiasAwarenessSignal` → sibling of `BiasProcesses`

Signals should be nested inside their target concept if it's a leaf, or as a sibling if the target has children.

### 3. Splitting Bad Categories
Some categories are taxonomically convenient but cognitively incoherent. These need to be **split, not just subdivided**:
- `StationaryArtifact` → split into Buildings, Rooms, Infrastructure, Facilities, BuildingParts
- `DetectionSignal` → dissolved entirely, each signal moved to relevant location

### 4. Natural Subdivisions
Use natural/established groupings where they exist:
- Periodic table groups for elements
- Anatomical systems for body parts (Skeletal, Muscular, HeadAndFace)
- Engine stroke types for engine motion

### 5. Functional Grouping
Group by function/use rather than form when it makes more cognitive sense:
- Sports by activity type (Water, Winter, Combat, Racquet)
- Devices by purpose (Communication, Safety, Industrial)
- Motions by type (Directional, Force, Mechanical)

## Anti-Patterns to Avoid

### Don't Create Arbitrary Splits
Bad: Splitting 30 items into "Group1", "Group2", "Group3"
Good: Finding natural categories that aid cognition

### Don't Over-Nest
If a grouping only has 2-3 items, consider whether it's worth the extra tree depth.

### Don't Break Established Taxonomies
Some domains have well-established taxonomies (periodic table, anatomy) - use them rather than inventing new ones.

### Don't Group by Alphabet or Arbitrary Order
Groupings should be semantic, not mechanical.

## Layer Considerations

The declared "layer" in the hierarchy is somewhat misleading - actual tree depth varies significantly. When restructuring:
- Focus on cognitive grouping, not layer numbers
- New intermediate nodes add depth but improve navigation
- A concept's effective depth = path length from root, not declared layer

## Process

1. Identify mega-parents (>15 children)
2. List all children and look for natural groupings
3. Check if concept itself is a "bad category" needing split vs subdivision
4. Create intermediate groups with self-descriptive names
5. For signals/detectors, move to be near what they detect
6. Verify branching factor is now acceptable
7. Ensure naming consistency with siblings

## Current Status (v4)

Restructured:
- ElementalSubstance: 113 → 20 (periodic table groups)
- DetectionSignal: 80 → 0 (dissolved, signals distributed)
- Organization: 39 → 14
- CognitiveState: 35 → 8
- AbstractEntity: 37 → 13
- Motion: 30 → 12
- Sport: 33 → 12
- FinancialTransaction: 36 → 13
- RelationalAttribute: 56 → 27

Remaining mega-parents are mostly domain-specific (Device, Proposition, Region, bacteria, churches) requiring domain expertise to subdivide appropriately.
