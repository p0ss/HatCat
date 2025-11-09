"""
Subspace removal methods for cleaning concept vectors.

Phase 6 methodology: Remove shared "definitional prompt structure" and
generic generation machinery from concept vectors to improve steering.
"""

import numpy as np
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


def apply_subspace_removal(
    vectors: np.ndarray,
    method: str
) -> np.ndarray:
    """
    Apply subspace removal to concept vectors.

    Methods:
    - "none": No removal (baseline)
    - "mean_subtraction": Remove mean vector across all concepts
    - "pca_1": Remove first principal component
    - "pca_5": Remove first 5 principal components
    - "pca_10": Remove first 10 principal components

    Args:
        vectors: (n_concepts, hidden_dim) array of concept vectors
        method: Removal method name

    Returns:
        clean_vectors: (n_concepts, hidden_dim) with shared subspace removed

    Example:
        >>> vectors = np.array([person_vec, change_vec, action_vec])  # (3, 2560)
        >>> clean = apply_subspace_removal(vectors, "mean_subtraction")
        >>> clean.shape
        (3, 2560)

    Phase 6 Results (2 concepts):
    - none: Working range ±0.25, 50% coherence at ±0.5
    - mean_subtraction: Working range ±0.5, 100% coherence at ±0.5
    - pca_1: Working range ±0.5, 100% coherence at all strengths
    """
    if method == "none":
        return vectors

    elif method == "mean_subtraction":
        # Remove mean vector (shared centroid)
        mean_vec = vectors.mean(axis=0)
        clean_vectors = vectors - mean_vec

    elif method.startswith("pca_"):
        # Remove first N principal components
        n_components = int(method.split("_")[1])

        # Can't use more components than min(n_samples, n_features)
        max_components = min(vectors.shape[0], vectors.shape[1])
        if n_components > max_components:
            logger.warning(
                f"Requested {n_components} components but only {max_components} available. "
                f"Using {max_components}."
            )
            n_components = max_components

        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(vectors)

        # Project onto principal components and subtract
        projected = pca.transform(vectors)
        reconstruction = pca.inverse_transform(projected)
        clean_vectors = vectors - reconstruction

        logger.info(
            f"PCA explained variance (first {n_components} components): "
            f"{pca.explained_variance_ratio_.sum():.3f}"
        )

    else:
        raise ValueError(f"Unknown removal method: {method}")

    # Re-normalize
    norms = np.linalg.norm(clean_vectors, axis=1, keepdims=True)
    clean_vectors = clean_vectors / (norms + 1e-8)

    return clean_vectors
