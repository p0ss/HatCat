#!/usr/bin/env python3
"""
Concept embedding utilities for semantic similarity.

Provides fast embedding-based lookup across concept packs without
requiring trained lenses. Useful for:
- MELD term resolution (finding existing near-duplicates)
- Antonym/steering target validation
- Cheap divergence scoring between concepts
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Default sentence-transformers model for semantic similarity
DEFAULT_ST_MODEL = "all-MiniLM-L6-v2"


class ConceptEmbeddingIndex:
    """
    Index of concept embeddings for fast similarity search.

    Uses sentence-transformers for semantic similarity (much better than
    raw LLM hidden states for this task).
    """

    def __init__(
        self,
        st_model_name: str = DEFAULT_ST_MODEL,
        device: str = "cuda",
    ):
        """
        Initialize the embedding index.

        Args:
            st_model_name: Sentence-transformers model name
            device: Device for embedding model
        """
        self.st_model_name = st_model_name
        self.device = device
        self._st_model = None

        # Index state
        self.terms: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.term_to_idx: Dict[str, int] = {}
        self.metadata: Dict[str, Dict] = {}  # term -> concept metadata
        self.alias_map: Dict[str, str] = {}  # alias -> primary term

    @property
    def st_model(self):
        """Lazy load sentence-transformers model."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.st_model_name)
            if self.device == "cuda":
                self._st_model = self._st_model.to(self.device)
        return self._st_model

    def _extract_embedding(self, text: str) -> np.ndarray:
        """Extract embedding for a text string."""
        embedding = self.st_model.encode([text], normalize_embeddings=True)[0]
        return embedding.astype(np.float32)

    def build_from_concept_pack(
        self,
        concept_pack_path: Union[str, Path],
        include_aliases: bool = True,
        include_definitions: bool = True,
        progress_callback=None,
        batch_size: int = 64,
    ) -> int:
        """
        Build index from a concept pack's per-concept files.

        Args:
            concept_pack_path: Path to concept pack
            include_aliases: Also index aliases (maps to primary term)
            include_definitions: Include definitions in embeddings (recommended)
            progress_callback: Optional fn(current, total) for progress
            batch_size: Batch size for embedding

        Returns:
            Number of terms indexed
        """
        concept_pack_path = Path(concept_pack_path)
        concepts_dir = concept_pack_path / "concepts"

        if not concepts_dir.exists():
            raise ValueError(f"No concepts/ directory in {concept_pack_path}")

        # Collect all concept files
        concept_files = list(concepts_dir.glob("layer*/*.json"))

        terms = []
        texts = []  # term + definition for embedding
        metadata = {}
        alias_map = {}  # alias -> primary term

        for i, f in enumerate(concept_files):
            if progress_callback:
                progress_callback(i, len(concept_files))

            with open(f) as fp:
                data = json.load(fp)

            term = data.get("term", "")
            if not term:
                continue

            definition = data.get("definition", "")[:300]

            terms.append(term)
            # Embed with definition for better semantics
            if include_definitions and definition:
                texts.append(f"{term}: {definition}")
            else:
                texts.append(term)

            metadata[term] = {
                "layer": data.get("layer"),
                "domain": data.get("domain"),
                "definition": definition,
                "aliases": data.get("aliases", []),
                "antonyms": data.get("relationships", {}).get("antonyms", []),
            }

            # Map aliases to primary term
            if include_aliases:
                for alias in data.get("aliases", []):
                    alias_lower = alias.lower()
                    if alias_lower != term.lower():
                        alias_map[alias_lower] = term

        # Batch encode for efficiency
        logger.info(f"Encoding {len(texts)} concepts...")
        embeddings = self.st_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )

        self.terms = terms
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.term_to_idx = {t: i for i, t in enumerate(terms)}
        self.metadata = metadata
        self.alias_map = alias_map

        logger.info(f"Indexed {len(terms)} terms, {len(alias_map)} aliases")
        return len(terms)

    def save(self, path: Union[str, Path]):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "index.json", "w") as f:
            json.dump({
                "st_model_name": self.st_model_name,
                "terms": self.terms,
                "metadata": self.metadata,
                "alias_map": self.alias_map,
            }, f)

    def load(self, path: Union[str, Path]):
        """Load index from disk."""
        path = Path(path)

        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "index.json") as f:
            data = json.load(f)

        self.st_model_name = data.get("st_model_name", DEFAULT_ST_MODEL)
        self.terms = data["terms"]
        self.metadata = data["metadata"]
        self.term_to_idx = {t: i for i, t in enumerate(self.terms)}
        self.alias_map = data.get("alias_map", {})

    def resolve_term(self, term: str) -> Optional[str]:
        """
        Resolve a term to its canonical form.

        Checks: exact match -> alias match -> None
        """
        if term in self.term_to_idx:
            return term
        if term.lower() in self.alias_map:
            return self.alias_map[term.lower()]
        return None

    def find_similar(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
        exclude_self: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Find most similar concepts to a query term.

        Args:
            query: Term to search for (can be new or existing)
            top_k: Number of results
            threshold: Minimum similarity (0-1)
            exclude_self: Exclude exact match from results

        Returns:
            List of (term, similarity) tuples, sorted by similarity desc
        """
        # Get or compute query embedding
        if query in self.term_to_idx:
            query_emb = self.embeddings[self.term_to_idx[query]]
        else:
            query_emb = self._extract_embedding(query)

        # Compute similarities
        similarities = self.embeddings @ query_emb

        # Sort and filter
        indices = np.argsort(-similarities)
        results = []

        for idx in indices:
            term = self.terms[idx]
            sim = float(similarities[idx])

            if exclude_self and term == query:
                continue
            if sim < threshold:
                break

            results.append((term, sim))
            if len(results) >= top_k:
                break

        return results

    def divergence(self, term_a: str, term_b: str) -> float:
        """
        Compute divergence between two concepts.

        Returns 1 - cosine_similarity, so:
        - 0.0 = identical
        - 1.0 = orthogonal
        - 2.0 = opposite
        """
        # Get or compute embeddings
        if term_a in self.term_to_idx:
            emb_a = self.embeddings[self.term_to_idx[term_a]]
        else:
            emb_a = self._extract_embedding(term_a)

        if term_b in self.term_to_idx:
            emb_b = self.embeddings[self.term_to_idx[term_b]]
        else:
            emb_b = self._extract_embedding(term_b)

        similarity = float(np.dot(emb_a, emb_b))
        return 1.0 - similarity

    def find_potential_duplicates(
        self,
        new_terms: List[str],
        threshold: float = 0.85,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find potential duplicates for a list of new terms.

        Useful for MELD validation before application.

        Args:
            new_terms: List of terms to check
            threshold: Similarity threshold for flagging

        Returns:
            Dict mapping new_term -> [(existing_term, similarity), ...]
        """
        duplicates = {}

        for term in new_terms:
            # Skip if already in index
            if term in self.term_to_idx:
                continue

            similar = self.find_similar(term, top_k=5, threshold=threshold)
            if similar:
                duplicates[term] = similar

        return duplicates

    def validate_antonyms(
        self,
        concept_pack_path: Union[str, Path],
    ) -> Dict[str, Dict]:
        """
        Validate all antonyms in a concept pack.

        Returns report of:
        - Antonyms that exist in hierarchy
        - Antonyms with near-matches
        - Antonyms that are truly missing
        """
        concept_pack_path = Path(concept_pack_path)
        concepts_dir = concept_pack_path / "concepts"

        report = {
            "exists": [],      # (concept, antonym)
            "near_match": [],  # (concept, antonym, match, similarity)
            "missing": [],     # (concept, antonym)
        }

        for f in concepts_dir.glob("layer*/*.json"):
            with open(f) as fp:
                data = json.load(fp)

            term = data.get("term", "")
            antonyms = data.get("relationships", {}).get("antonyms", [])

            for ant in antonyms:
                if ant in self.term_to_idx:
                    report["exists"].append((term, ant))
                else:
                    # Check for near-match
                    similar = self.find_similar(ant, top_k=1, threshold=0.8)
                    if similar:
                        match, sim = similar[0]
                        report["near_match"].append((term, ant, match, sim))
                    else:
                        report["missing"].append((term, ant))

        return report


def build_index_cli():
    """CLI for building concept embedding index."""
    import argparse

    parser = argparse.ArgumentParser(description="Build concept embedding index")
    parser.add_argument("concept_pack", type=Path, help="Path to concept pack")
    parser.add_argument("--output", type=Path, help="Output path for index")
    parser.add_argument("--model", default=DEFAULT_ST_MODEL, help="Sentence-transformers model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-definitions", action="store_true", help="Don't include definitions")

    args = parser.parse_args()

    print(f"Building index from {args.concept_pack}...")
    print(f"Using model: {args.model}")
    index = ConceptEmbeddingIndex(st_model_name=args.model, device=args.device)

    count = index.build_from_concept_pack(
        args.concept_pack,
        include_definitions=not args.no_definitions,
    )
    print(f"Indexed {count} terms")

    output_path = args.output or (args.concept_pack / "embedding_index")
    print(f"Saving to {output_path}...")
    index.save(output_path)
    print("Done")


if __name__ == "__main__":
    build_index_cli()
