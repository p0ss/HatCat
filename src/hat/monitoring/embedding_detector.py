#!/usr/bin/env python3
"""
Embedding-based text detection using concept similarity index.

Replaces the legacy TF-IDF and centroid-based text lenses with
sentence-transformer based similarity. Detects "aboutness" - whether
a token/text is semantically related to a concept.

Key insight: This detects DOMAIN not DIRECTION.
- "safe" and "dangerous" both score high on Safety (both are about safety)
- "honest" and "dishonest" both relate to Deception domain
- Use activation lenses for direction, text similarity for domain relevance
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.map.data.concept_embeddings import ConceptEmbeddingIndex


class EmbeddingTextDetector:
    """
    Text detector using pre-built concept embedding index.

    Provides:
    - Top-k concept similarity for any text
    - Domain-relevance scores for specific concepts
    - Batch processing for efficiency
    """

    def __init__(self, index: "ConceptEmbeddingIndex"):
        """
        Initialize with pre-built concept embedding index.

        Args:
            index: ConceptEmbeddingIndex with embeddings loaded
        """
        self.index = index

    @classmethod
    def from_concept_pack(cls, concept_pack_path: Path) -> "EmbeddingTextDetector":
        """
        Load from a concept pack's embedding index.

        Args:
            concept_pack_path: Path to concept pack directory

        Returns:
            EmbeddingTextDetector instance
        """
        from src.map.data.concept_embeddings import ConceptEmbeddingIndex

        index_path = concept_pack_path / "embedding_index"
        if not index_path.exists():
            raise ValueError(
                f"No embedding index at {index_path}. "
                f"Run: python -m src.data.concept_embeddings {concept_pack_path}"
            )

        index = ConceptEmbeddingIndex()
        index.load(index_path)

        return cls(index)

    def detect_concepts(
        self,
        text: str,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Detect which concepts a text is semantically related to.

        Args:
            text: Input text (token, phrase, or sentence)
            top_k: Number of top concepts to return
            threshold: Minimum similarity threshold

        Returns:
            List of (concept_name, similarity) tuples, sorted by similarity desc
        """
        return self.index.find_similar(text, top_k=top_k, threshold=threshold)

    def get_concept_similarity(
        self,
        text: str,
        concept_name: str,
    ) -> Optional[float]:
        """
        Get similarity between text and a specific concept.

        Args:
            text: Input text
            concept_name: Name of concept to check

        Returns:
            Similarity score [-1, 1] or None if concept not in index
        """
        if concept_name not in self.index.term_to_idx:
            return None

        text_emb = self.index._extract_embedding(text)
        concept_emb = self.index.embeddings[self.index.term_to_idx[concept_name]]

        return float(np.dot(text_emb, concept_emb))

    def get_concept_similarities_batch(
        self,
        text: str,
        concept_names: List[str],
    ) -> Dict[str, float]:
        """
        Get similarities between text and multiple concepts efficiently.

        Args:
            text: Input text
            concept_names: List of concept names to check

        Returns:
            Dict mapping concept_name to similarity score
        """
        text_emb = self.index._extract_embedding(text)

        results = {}
        for concept_name in concept_names:
            if concept_name in self.index.term_to_idx:
                concept_emb = self.index.embeddings[self.index.term_to_idx[concept_name]]
                results[concept_name] = float(np.dot(text_emb, concept_emb))

        return results

    def compute_divergence(
        self,
        text: str,
        activation_scores: Dict[str, float],
        activation_threshold: float = 0.5,
    ) -> Dict[str, Dict]:
        """
        Compute divergence between activation scores and text similarity.

        Divergence indicates the model may be "thinking" about a concept
        (high activation) without "writing" about it (low text similarity),
        or vice versa.

        Args:
            text: The generated text/token
            activation_scores: Dict of concept_name -> activation probability
            activation_threshold: Only check concepts above this activation

        Returns:
            Dict mapping concept_name to:
                - activation: float (from activation lens)
                - text_similarity: float (from embedding)
                - divergence: float (activation - text_similarity)
                - interpretation: str
        """
        # Filter to concepts above threshold
        relevant_concepts = [
            name for name, score in activation_scores.items()
            if score >= activation_threshold
        ]

        if not relevant_concepts:
            return {}

        # Get text similarities for relevant concepts
        text_sims = self.get_concept_similarities_batch(text, relevant_concepts)

        # Normalize text similarity from [-1, 1] to [0, 1] for comparison
        def normalize_sim(sim: float) -> float:
            return (sim + 1.0) / 2.0

        results = {}
        for concept_name in relevant_concepts:
            activation = activation_scores[concept_name]
            text_sim = text_sims.get(concept_name)

            if text_sim is None:
                continue

            text_prob = normalize_sim(text_sim)
            divergence = activation - text_prob

            # Interpretation
            if abs(divergence) < 0.2:
                interpretation = "aligned"  # Thinking and writing match
            elif divergence > 0.2:
                interpretation = "covert"   # Thinking more than writing
            else:
                interpretation = "overt"    # Writing more than thinking

            results[concept_name] = {
                "activation": activation,
                "text_similarity": text_sim,
                "text_probability": text_prob,
                "divergence": divergence,
                "interpretation": interpretation,
            }

        return results

    def get_domain_profile(
        self,
        text: str,
        top_k: int = 5,
    ) -> Dict[str, any]:
        """
        Get a domain profile for text showing what concepts it relates to.

        Useful for understanding what semantic domains a token/phrase touches.

        Args:
            text: Input text
            top_k: Number of top concepts per domain

        Returns:
            Dict with:
                - top_concepts: List of (concept, similarity)
                - domains: Set of unique domains
                - avg_max_similarity: Average of top-k similarities
        """
        top_concepts = self.detect_concepts(text, top_k=top_k)

        domains = set()
        for concept_name, _ in top_concepts:
            meta = self.index.metadata.get(concept_name, {})
            domain = meta.get("domain")
            if domain:
                domains.add(domain)

        avg_sim = np.mean([sim for _, sim in top_concepts]) if top_concepts else 0.0
        max_sim = max([sim for _, sim in top_concepts]) if top_concepts else 0.0

        return {
            "top_concepts": top_concepts,
            "domains": domains,
            "avg_similarity": float(avg_sim),
            "max_similarity": float(max_sim),
        }


def create_text_detector_for_openwebui(
    concept_pack_path: Path = Path("concept_packs/first-light"),
) -> EmbeddingTextDetector:
    """
    Factory function for OpenWebUI integration.

    Args:
        concept_pack_path: Path to concept pack with embedding index

    Returns:
        Configured EmbeddingTextDetector
    """
    return EmbeddingTextDetector.from_concept_pack(concept_pack_path)


__all__ = [
    "EmbeddingTextDetector",
    "create_text_detector_for_openwebui",
]
