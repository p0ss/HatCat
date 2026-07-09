"""
BE - Bounded Experiencer

Layer 4 of the FTW architecture. The BE is the experiential runtime that
integrates model inference, lenses, steering, workspace, XDB, and audit
into a coherent experiencing entity.

Submodules:
- be.bootstrap: BE instantiation and lifecycle management
- be.xdb: Experience database (episodic memory system)
- be.thalamos: Cognitive assessment and surgery (Thalametry)
- be.harness: Graft testing harness (deprecated, use thalamos)

The diegesis is the experiential frame in which the BE lives.
"""

from .diegesis import BEDFrame, BEDConfig, ExperienceTick

# Tier 0: Autonomic Core
from .autonomic import (
    AutonomicCore,
    AutonomicState,
    TierManager,
    WorkspaceState,
    TierBreach,
)

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

# Re-export key thalamos items (cognitive assessment/surgery)
from .thalamos import (
    ExaminationRoom,
    ExaminationConfig,
    Thalametrist,
    ConceptAssessment,
    AssessmentResult,
    CalibrationSuite,
    CalibrationReport as ThalamosCalibrationReport,
    AssessmentProtocol,
    ExplanationProtocol,
    ClassificationProtocol,
    ProcedureRecord,
    AssessmentRecord,
    QualificationRecord,
)

# Re-export key harness items (deprecated - use thalamos instead)
from .harness import (
    GraftTester,
    GraftTestReport,
    HarnessConfig,
    TargetModel,
    JudgeModel,
    ConceptEvaluator,
    EvaluationResult,
    HarnessReporter,
    HarnessReport,
    JudgeCalibrator,
    CalibrationReport,
    MeldDesigner,
)

__all__ = [
    # Diegesis
    'BEDFrame',
    'BEDConfig',
    'ExperienceTick',
    # Tier 0: Autonomic Core
    'AutonomicCore',
    'AutonomicState',
    'TierManager',
    'WorkspaceState',
    'TierBreach',
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
    # Thalamos (cognitive assessment/surgery)
    'ExaminationRoom',
    'ExaminationConfig',
    'Thalametrist',
    'ConceptAssessment',
    'AssessmentResult',
    'CalibrationSuite',
    'ThalamosCalibrationReport',
    'AssessmentProtocol',
    'ExplanationProtocol',
    'ClassificationProtocol',
    'ProcedureRecord',
    'AssessmentRecord',
    'QualificationRecord',
    # Harness (deprecated)
    'GraftTester',
    'GraftTestReport',
    'HarnessConfig',
    'TargetModel',
    'JudgeModel',
    'ConceptEvaluator',
    'EvaluationResult',
    'HarnessReporter',
    'HarnessReport',
    'JudgeCalibrator',
    'CalibrationReport',
    'MeldDesigner',
]
