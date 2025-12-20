"""
HAT - Headspace Ambient Transducer

Layer 2 of the FTW architecture. HAT reads and writes to the model's
activation space through "lenses" - trained classifiers that detect
or steer specific concepts.

Submodules:
- hat.classifiers: MLPClassifier, Lens, ActivationCapture
- hat.steering: hooks, extraction, manifold steering
- hat.monitoring: DynamicLensManager, temporal monitoring
- hat.interpreter: SemanticInterpreter for activation decoding
- hat.utils: model loading, storage, provenance

A Lens translates high-level requests ("measure Deception") into the
appropriate combination of classifiers and layers for that concept.
"""

# Re-export from classifiers submodule
from .classifiers import (
    MLPClassifier,
    LinearProbe,
    load_classifier,
    save_classifier,
    Lens,
    ClassifierInfo,
    LensNotFoundError,
    ActivationCapture,
    ActivationConfig,
    BaselineGenerator,
)

# Re-export from steering submodule
from .steering import (
    create_steering_hook,
    generate_with_steering,
    create_contrastive_steering_hook,
    compute_steering_field,
    compute_contrastive_vector,
    load_lens_classifier,
    LensClassifier,
    extract_concept_vector,
    build_centroids,
    compute_semantic_shift,
    apply_subspace_removal,
    ManifoldSteerer,
    create_manifold_steering_hook,
    create_dampened_steering_hook,
    apply_dual_subspace_steering,
    estimate_contamination_subspace,
    estimate_task_manifold,
)

# Re-export from monitoring submodule
from .monitoring import (
    DynamicLensManager,
    SimpleMLP,
    ConceptMetadata,
    LensRole,
    SUMOTemporalMonitor,
    load_sumo_classifiers,
    run_temporal_detection,
    CentroidTextDetector,
    TfidfConceptLens,
)

# Re-export from interpreter submodule
from .interpreter import (
    SemanticInterpreter,
    InterpreterWithHierarchy,
    MultiTaskInterpreter,
)

# Re-export from utils submodule
from .utils import (
    ModelLoader,
    ActivationStorage,
    SparseActivationStorage,
    get_git_info,
    get_provenance,
    save_results_with_provenance,
    create_run_directory,
    write_run_manifest,
    update_run_manifest,
)

__all__ = [
    # Classifiers
    "MLPClassifier",
    "LinearProbe",
    "load_classifier",
    "save_classifier",
    "Lens",
    "ClassifierInfo",
    "LensNotFoundError",
    "ActivationCapture",
    "ActivationConfig",
    "BaselineGenerator",
    # Steering
    "create_steering_hook",
    "generate_with_steering",
    "create_contrastive_steering_hook",
    "compute_steering_field",
    "compute_contrastive_vector",
    "load_lens_classifier",
    "LensClassifier",
    "extract_concept_vector",
    "build_centroids",
    "compute_semantic_shift",
    "apply_subspace_removal",
    "ManifoldSteerer",
    "create_manifold_steering_hook",
    "create_dampened_steering_hook",
    "apply_dual_subspace_steering",
    "estimate_contamination_subspace",
    "estimate_task_manifold",
    # Monitoring
    "DynamicLensManager",
    "SimpleMLP",
    "ConceptMetadata",
    "LensRole",
    "SUMOTemporalMonitor",
    "load_sumo_classifiers",
    "run_temporal_detection",
    "CentroidTextDetector",
    "TfidfConceptLens",
    # Interpreter
    "SemanticInterpreter",
    "InterpreterWithHierarchy",
    "MultiTaskInterpreter",
    # Utils
    "ModelLoader",
    "ActivationStorage",
    "SparseActivationStorage",
    "get_git_info",
    "get_provenance",
    "save_results_with_provenance",
    "create_run_directory",
    "write_run_manifest",
    "update_run_manifest",
]
