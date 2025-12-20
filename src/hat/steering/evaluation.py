"""
Semantic shift evaluation using embedding-based metrics.

Phase 5 methodology: Measure steering effectiveness by comparing
generated text embeddings to concept centroids.
"""

import numpy as np
from typing import Dict, List, Tuple
import random


def build_centroids(
    concept: str,
    concept_info: Dict,
    neutral_pool: List[str],
    embedding_model,
    n_samples: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build three centroids for semantic shift measurement.

    Centroids represent:
    - Core: Definitional prompts about the concept
    - Boundary: Relational prompts connecting to related concepts
    - Negative: Prompts about distant/unrelated concepts

    Args:
        concept: Concept name
        concept_info: Dict with 'related' and 'negatives' keys
        neutral_pool: List of neutral concept names for fallback
        embedding_model: SentenceTransformer model
        n_samples: Number of samples per centroid

    Returns:
        (core_centroid, boundary_centroid, neg_centroid)

    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        >>> concept_info = {"related": ["human", "individual"], "negatives": ["object", "thing"]}
        >>> core, boundary, neg = build_centroids("person", concept_info, ["animal"], embed_model)
    """
    # Core centroid: definitional prompts
    core_prompts = [
        f"What is {concept}?",
        f"Define {concept}.",
        f"{concept.capitalize()} is",
        f"Tell me about {concept}.",
        f"Explain {concept}."
    ][:n_samples]

    core_embeddings = embedding_model.encode(core_prompts)
    core_centroid = core_embeddings.mean(axis=0)

    # Boundary centroid: relational prompts
    related = concept_info.get('related', [])
    boundary_prompts = []

    if len(related) >= n_samples:
        sampled_related = random.sample(related, n_samples)
        for rel in sampled_related:
            boundary_prompts.append(f"{concept} is related to {rel}.")
    else:
        # Fallback: use core prompts if not enough relationships
        boundary_prompts = core_prompts[:n_samples]

    boundary_embeddings = embedding_model.encode(boundary_prompts)
    boundary_centroid = boundary_embeddings.mean(axis=0)

    # Negative centroid: distant concepts
    negatives = concept_info.get('negatives', [])
    if len(negatives) >= n_samples:
        sampled_negs = random.sample(negatives, n_samples)
        neg_prompts = [f"What is {neg}?" for neg in sampled_negs]
    else:
        # Fallback: use neutral pool
        sampled_neutrals = random.sample(neutral_pool, n_samples)
        neg_prompts = [f"What is {neut}?" for neut in sampled_neutrals]

    neg_embeddings = embedding_model.encode(neg_prompts)
    neg_centroid = neg_embeddings.mean(axis=0)

    return core_centroid, boundary_centroid, neg_centroid


def compute_semantic_shift(
    generated_text: str,
    core_centroid: np.ndarray,
    neg_centroid: np.ndarray,
    embedding_model
) -> float:
    """
    Compute semantic shift Δ for generated text.

    Δ = cos(text, core) - cos(text, neg)

    Positive Δ means text is closer to concept than to negatives.
    Higher |Δ| indicates stronger steering effect.

    Args:
        generated_text: Generated text to evaluate
        core_centroid: Core concept centroid
        neg_centroid: Negative concept centroid
        embedding_model: SentenceTransformer model

    Returns:
        Semantic shift Δ (range: -2 to +2, typically -1 to +1)

    Example:
        >>> delta = compute_semantic_shift(
        ...     "A person is a human being.",
        ...     core_centroid, neg_centroid, embed_model
        ... )
        >>> delta > 0  # Text is closer to "person" than negatives
        True
    """
    text_embedding = embedding_model.encode([generated_text])[0]

    # Cosine similarity to core and negative centroids
    cos_core = np.dot(text_embedding, core_centroid) / (
        np.linalg.norm(text_embedding) * np.linalg.norm(core_centroid) + 1e-8
    )
    cos_neg = np.dot(text_embedding, neg_centroid) / (
        np.linalg.norm(text_embedding) * np.linalg.norm(neg_centroid) + 1e-8
    )

    delta = cos_core - cos_neg
    return float(delta)
