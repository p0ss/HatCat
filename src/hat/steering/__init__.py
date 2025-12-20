"""
HAT Steering Module

Concept vector extraction and steering operations for LLM activation space.
"""

from .hooks import (
    create_steering_hook,
    generate_with_steering,
    create_contrastive_steering_hook,
    compute_steering_field,
    compute_contrastive_vector,
    load_lens_classifier,
    LensClassifier,
)
from .extraction import extract_concept_vector
from .evaluation import build_centroids, compute_semantic_shift
from .subspace import apply_subspace_removal
from .manifold import (
    ManifoldSteerer,
    create_manifold_steering_hook,
    create_dampened_steering_hook,
    apply_dual_subspace_steering,
    estimate_contamination_subspace,
    estimate_task_manifold,
)

__all__ = [
    # Hooks
    "create_steering_hook",
    "generate_with_steering",
    "create_contrastive_steering_hook",
    "compute_steering_field",
    "compute_contrastive_vector",
    "load_lens_classifier",
    "LensClassifier",
    # Extraction
    "extract_concept_vector",
    # Evaluation
    "build_centroids",
    "compute_semantic_shift",
    # Subspace
    "apply_subspace_removal",
    # Manifold
    "ManifoldSteerer",
    "create_manifold_steering_hook",
    "create_dampened_steering_hook",
    "apply_dual_subspace_steering",
    "estimate_contamination_subspace",
    "estimate_task_manifold",
]
