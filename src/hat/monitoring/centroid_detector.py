"""
Direct token-to-concept text detection using cosine similarity.

Compares output token embeddings directly to concept name embeddings:
- Fast inference (no text vectorization needed)
- Scalable to 110K+ concepts
- Works with model's own embedding space
- More accurate than training sample centroids
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


class CentroidTextDetector:
    """
    Embedding-based text detector using concept name embeddings.

    Compares token embedding directly to the embedding of the concept name itself.
    This is more accurate than using centroids of training samples.
    """

    def __init__(self, concept_name: str, centroid: np.ndarray):
        """
        Initialize detector with concept name embedding.

        Args:
            concept_name: SUMO concept name
            centroid: Normalized concept name embedding [embedding_dim]
        """
        self.concept_name = concept_name
        self.centroid = centroid  # This is actually the concept name embedding
        self.is_fitted = True

    def predict(self, token_embedding: np.ndarray) -> float:
        """
        Predict probability that token embedding expresses this concept.

        Uses cosine similarity between token embedding and concept name embedding.
        Converts similarity [-1, 1] to probability [0, 1].

        Args:
            token_embedding: Token embedding vector [embedding_dim]

        Returns:
            Probability [0, 1] that token expresses this concept
        """
        # Normalize token embedding
        token_norm = token_embedding / (np.linalg.norm(token_embedding) + 1e-8)

        # Compute cosine similarity with concept name embedding
        similarity = np.dot(token_norm, self.centroid)

        # Convert similarity [-1, 1] to probability [0, 1]
        # similarity = 1.0  → prob = 1.0 (perfect match)
        # similarity = 0.0  → prob = 0.5 (neutral)
        # similarity = -1.0 → prob = 0.0 (opposite)
        probability = (similarity + 1.0) / 2.0

        return float(probability)

    def save(self, path: Path):
        """Save centroid to disk."""
        np.save(path, self.centroid)

    @classmethod
    def load(cls, path: Path, concept_name: Optional[str] = None) -> 'CentroidTextDetector':
        """
        Load centroid from disk.

        Args:
            path: Path to .npy centroid file
            concept_name: Optional concept name (extracted from path if not provided)

        Returns:
            CentroidTextDetector instance
        """
        centroid = np.load(path)

        if concept_name is None:
            # Extract from filename: "Physical_centroid.npy" → "Physical"
            concept_name = path.stem.replace("_centroid", "")

        return cls(concept_name=concept_name, centroid=centroid)


def load_centroids_for_layer(
    layer: int,
    centroids_dir: Path,
    device: str = "cpu",
) -> Dict[str, CentroidTextDetector]:
    """
    Load all centroids for a layer.

    Args:
        layer: Layer number
        centroids_dir: Directory containing centroid .npy files
        device: Device (not used for numpy, kept for API compatibility)

    Returns:
        Dict mapping concept names to CentroidTextDetector instances
    """
    detectors = {}

    # Find all centroid files
    centroid_files = list(centroids_dir.glob("*_centroid.npy"))

    for centroid_file in centroid_files:
        concept_name = centroid_file.stem.replace("_centroid", "")
        detector = CentroidTextDetector.load(centroid_file, concept_name=concept_name)
        detectors[concept_name] = detector

    return detectors


__all__ = [
    "CentroidTextDetector",
    "load_centroids_for_layer",
]
