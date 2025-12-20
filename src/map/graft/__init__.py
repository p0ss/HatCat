"""
Grafting infrastructure for structured integration of learned concepts into substrates.

Terminology (botanical grafting metaphor):
- **Bud**: Soft/temporary graft using hooks (reversible, for testing)
- **Scion**: Hard/permanent graft that modifies weights (accretes)
- **Cleft**: Region of model weights associated with concepts (from lens analysis)

The grafting flow:
1. Model experiences something, XDB records it with concept tags (lenses that fired)
2. Each tagged concept maps to a Cleft (weights the lens reads from)
3. Scion training trains ONLY the union of tagged concepts' clefts
4. After training, compute deltas and create new neuron with proportional biases
5. Apply scion permanently, or test as a bud first

See docs/specification/MAP/MAP_GRAFTING.md for the full specification.
"""

# Cleft identification
from .cleft import (
    Cleft,
    CleftRegion,
    UnionCleft,
    CleftAwareFreezer,
    derive_cleft_from_lens,
    merge_clefts,
)

# Scion (permanent grafts)
from .scion import (
    Scion,
    ScionConfig,
    ScionTrainer,
    WeightDelta,
    apply_scion,
    revert_scion,
)

# Bud (temporary grafts)
from .bud import (
    Bud,
    BuddedModel,
)

# Data structures
from .data_structures import (
    ConceptRegion,
    Graft,
    GraftConfig,
    SubstrateManifest,
    SubstrateArchitecture,
    SubstrateBias,
    DimensionEntry,
    InjectionPoint,
    LayerMask,
)

# Legacy functions (for compatibility)
from .region_derivation import (
    derive_region_from_lens,
    analyze_region_overlap,
)

# Expand mode (architecture-aware dimension expansion)
from .expand import (
    ArchitectureSpec,
    ExpansionPlan,
    ExpansionTarget,
    ScionExpandMetadata,
    detect_architecture,
    plan_expansion,
    execute_expansion,
    extract_expand_metadata,
)

__all__ = [
    # Cleft
    "Cleft",
    "CleftRegion",
    "UnionCleft",
    "CleftAwareFreezer",
    "derive_cleft_from_lens",
    "merge_clefts",
    # Scion
    "Scion",
    "ScionConfig",
    "ScionTrainer",
    "WeightDelta",
    "apply_scion",
    "revert_scion",
    # Bud
    "Bud",
    "BuddedModel",
    # Data structures
    "ConceptRegion",
    "Graft",
    "GraftConfig",
    "SubstrateManifest",
    "SubstrateArchitecture",
    "SubstrateBias",
    "DimensionEntry",
    "InjectionPoint",
    "LayerMask",
    # Legacy functions
    "derive_region_from_lens",
    # Expand mode
    "ArchitectureSpec",
    "ExpansionPlan",
    "ExpansionTarget",
    "ScionExpandMetadata",
    "detect_architecture",
    "plan_expansion",
    "execute_expansion",
    "extract_expand_metadata",
]
