"""
MAP (Mindmeld Architectural Protocol)

Layer 3 of the FTW architecture - provides concept packs, lens packs,
registry management, and HuggingFace synchronization.

Submodules:
- map.registry: Pack management and HF sync
- map.graft: Concept grafting operations
- map.meld: Meld operations (ontology building)
- map.training: Lens training infrastructure
- map.data: Version manifests, concept embeddings
"""

# Re-export from registry submodule
from .registry import (
    PackRegistry,
    registry,
    PackInfo,
    ConceptPack,
    Concept,
    load_concept_pack,
    LensPack,
    LensInfo,
    load_lens_pack,
)

# Re-export key graft items
from .graft import (
    Cleft,
    Scion,
    Bud,
    BuddedModel,
    apply_scion,
    revert_scion,
)

# Re-export key meld items
from .meld import (
    load_concepts,
    load_test_concepts,
    get_concept_category,
)

# Re-export key training items
from .training import (
    train_sumo_classifiers,
    create_sumo_training_dataset,
    validate_lens_calibration,
    validate_lens_set,
)

# Re-export key data items
from .data import (
    LensManifest,
    LensEntry,
    ClassifierEntry,
)

__all__ = [
    # Registry
    'PackRegistry',
    'registry',
    'PackInfo',
    'ConceptPack',
    'Concept',
    'LensPack',
    'LensInfo',
    'load_concept_pack',
    'load_lens_pack',
    # Graft
    'Cleft',
    'Scion',
    'Bud',
    'BuddedModel',
    'apply_scion',
    'revert_scion',
    # Meld
    'load_concepts',
    'load_test_concepts',
    'get_concept_category',
    # Training
    'train_sumo_classifiers',
    'create_sumo_training_dataset',
    'validate_lens_calibration',
    'validate_lens_set',
    # Data
    'LensManifest',
    'LensEntry',
    'ClassifierEntry',
]
