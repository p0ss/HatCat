"""
BE - Bounded Experiencer

Layer 4 of the FTW architecture. The BE is the experiential runtime that
integrates model inference, lenses, steering, workspace, XDB, and audit
into a coherent experiencing entity.

Submodules:
- be.bootstrap: BE instantiation and lifecycle management
- be.xdb: Experience database (episodic memory system)

The diegesis is the experiential frame in which the BE lives.
"""

from .diegesis import BEDFrame, BEDConfig, ExperienceTick

# Re-export key bootstrap items
from .bootstrap import (
    BootstrapArtifact,
    wake_be,
    WakeSequence,
    BoundedExperiencer,
    ToolGraft,
    ToolGraftPack,
    UpliftTaxonomy,
    build_base_taxonomy,
    MeldSubmission,
    MeldBatch,
)

# Re-export key XDB items
from .xdb import (
    XDB,
    ExperienceLog,
    TagIndex,
    StorageManager,
    AuditLog,
    BuddingManager,
    TimestepRecord,
    Tag,
    Fidelity,
)

__all__ = [
    # Diegesis
    'BEDFrame',
    'BEDConfig',
    'ExperienceTick',
    # Bootstrap
    'BootstrapArtifact',
    'wake_be',
    'WakeSequence',
    'BoundedExperiencer',
    'ToolGraft',
    'ToolGraftPack',
    'UpliftTaxonomy',
    'build_base_taxonomy',
    'MeldSubmission',
    'MeldBatch',
    # XDB
    'XDB',
    'ExperienceLog',
    'TagIndex',
    'StorageManager',
    'AuditLog',
    'BuddingManager',
    'TimestepRecord',
    'Tag',
    'Fidelity',
]
