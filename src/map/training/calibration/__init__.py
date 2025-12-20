"""
Pack-level calibration for lens ensembles.

After training individual lenses, pack-level calibration identifies and fixes:
- Under-firing: Target lens not in top-k when it should be
- Over-firing: Non-target lenses appearing in top-k when they shouldn't

See docs/CALIBRATION.md for the full framework description.

Usage:
    # Build full calibration matrix (recommended)
    python -m training.calibration.matrix \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509 \
        --fast-mode

    # Or run calibration cycle (analysis + finetune)
    python -m training.calibration.cycle \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509
"""

from .analysis import (
    run_calibration_analysis,
    PackCalibrationAnalysis,
    LensCalibrationReport,
    ConceptProbeResult,
)

from .finetune import (
    run_calibration_finetune,
    FineTuneReport,
    FineTuneResult,
)

from .cycle import run_calibration_cycle

from .matrix import (
    CalibrationMatrix,
    CalibrationMatrixBuilder,
    CalibrationMatrixConfig,
    LensStatistics,
    print_matrix_summary,
)

from .feature_weighting import (
    # Core classes
    FeatureWeightedTrainer,
    ConceptActivationMapper,
    NeuralConceptMapper,
    # Data structures
    FeatureWeights,
    ConceptActivationStats,
    ConceptActivation,
    ActivationMap,
    # Utilities
    collect_concept_activations,
    # Backward compatibility
    BucketClassifier,  # Alias for ConceptActivationMapper
    NeuralBucketClassifier,  # Alias for NeuralConceptMapper
)

__all__ = [
    # Matrix-based calibration (recommended)
    'CalibrationMatrix',
    'CalibrationMatrixBuilder',
    'CalibrationMatrixConfig',
    'LensStatistics',
    'print_matrix_summary',
    # Legacy analysis
    'run_calibration_analysis',
    'PackCalibrationAnalysis',
    'LensCalibrationReport',
    'ConceptProbeResult',
    # Fine-tuning
    'run_calibration_finetune',
    'FineTuneReport',
    'FineTuneResult',
    # Orchestration
    'run_calibration_cycle',
    # Feature weighting / Concept activation mapping
    'FeatureWeightedTrainer',
    'ConceptActivationMapper',
    'NeuralConceptMapper',
    'FeatureWeights',
    'ConceptActivationStats',
    'ConceptActivation',
    'ActivationMap',
    'collect_concept_activations',
    # Backward compatibility aliases
    'BucketClassifier',
    'NeuralBucketClassifier',
]
