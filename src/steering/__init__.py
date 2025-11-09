"""
Steering module: Concept vector extraction and manipulation for controllable generation.

This module provides tools for:
- Extracting concept vectors from model activations
- Applying steering during generation via forward hooks
- Evaluating semantic shifts using embedding-based metrics
- Cleaning concept vectors via subspace removal
"""

from .extraction import extract_concept_vector
from .hooks import create_steering_hook, generate_with_steering
from .evaluation import build_centroids, compute_semantic_shift
from .subspace import apply_subspace_removal

__all__ = [
    "extract_concept_vector",
    "create_steering_hook",
    "generate_with_steering",
    "build_centroids",
    "compute_semantic_shift",
    "apply_subspace_removal",
]
