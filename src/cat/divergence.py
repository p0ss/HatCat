"""Semantic dissonance measurement between expected and detected concepts."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Language = None

try:
    from nltk.corpus import wordnet as wn
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class SUMOConceptGraph:
    """Builds and queries the SUMO concept hierarchy graph."""

    def __init__(
        self,
        layers_dir: Path = Path("data/concept_graph/abstraction_layers"),
        include_wordnet_edges: bool = True,
    ):
        """
        Load SUMO concept graph from layer files.

        Args:
            layers_dir: Path to abstraction layers directory
            include_wordnet_edges: Add WordNet hyponym/similar-to edges
        """
        self.graph = nx.Graph()  # Undirected for bidirectional search
        self.concept_to_layer: Dict[str, int] = {}
        self.concept_to_synsets: Dict[str, List[str]] = {}
        self._build_graph(layers_dir, include_wordnet_edges)

    def _build_graph(self, layers_dir: Path, include_wordnet_edges: bool) -> None:
        """Build graph from layer JSON files with optional WordNet enrichment."""
        # Find all layer files
        layer_files = sorted(layers_dir.glob("layer*.json"))

        # First pass: Add all nodes and parent-child edges
        for layer_file in layer_files:
            with open(layer_file) as f:
                data = json.load(f)

            layer_num = data["metadata"]["layer"]

            for concept in data["concepts"]:
                concept_name = concept["sumo_term"]

                # Add node
                self.graph.add_node(
                    concept_name,
                    layer=layer_num,
                    depth=concept.get("sumo_depth"),
                    synset_count=concept.get("synset_count", 0),
                )
                self.concept_to_layer[concept_name] = layer_num

                # Store synsets for WordNet enrichment
                synsets = concept.get("synsets", [])
                self.concept_to_synsets[concept_name] = synsets

                # Add edges to children (SUMO hierarchy)
                for child in concept.get("category_children", []):
                    self.graph.add_edge(concept_name, child, edge_type="sumo_parent")

        # Second pass: Add WordNet-based edges
        if include_wordnet_edges and NLTK_AVAILABLE:
            self._add_wordnet_edges()

    def _add_wordnet_edges(self) -> None:
        """Add edges based on WordNet hyponym and similar-to relationships."""
        print("  Adding WordNet hyponym/similar-to edges...")

        edge_count = 0

        for concept, synsets in self.concept_to_synsets.items():
            if not synsets:
                continue

            # Get WordNet synset objects
            wn_synsets = []
            for synset_name in synsets:
                try:
                    wn_synsets.append(wn.synset(synset_name))
                except Exception:
                    continue

            if not wn_synsets:
                continue

            # Find related concepts via hyponyms
            related_concepts = set()
            for syn in wn_synsets:
                # Hyponyms (is-a relationship)
                for hypo in syn.hyponyms():
                    hypo_name = hypo.name()
                    # Find SUMO concept containing this synset
                    for other_concept, other_synsets in self.concept_to_synsets.items():
                        if other_concept != concept and hypo_name in other_synsets:
                            related_concepts.add(other_concept)

                # Hypernyms (reverse is-a)
                for hyper in syn.hypernyms():
                    hyper_name = hyper.name()
                    for other_concept, other_synsets in self.concept_to_synsets.items():
                        if other_concept != concept and hyper_name in other_synsets:
                            related_concepts.add(other_concept)

                # Similar-to (for adjectives)
                try:
                    for similar in syn.similar_tos():
                        similar_name = similar.name()
                        for other_concept, other_synsets in self.concept_to_synsets.items():
                            if other_concept != concept and similar_name in other_synsets:
                                related_concepts.add(other_concept)
                except Exception:
                    pass  # Not all synsets have similar_tos

            # Add edges to related concepts
            for related in related_concepts:
                if not self.graph.has_edge(concept, related):
                    self.graph.add_edge(concept, related, edge_type="wordnet_related")
                    edge_count += 1

        print(f"  Added {edge_count} WordNet-based edges")

    def shortest_path_length(self, source: str, target: str, max_dist: int = 10) -> int:
        """
        Compute shortest path length between two concepts.

        Args:
            source: Source concept
            target: Target concept
            max_dist: Maximum distance to return if no path exists

        Returns:
            Shortest path length, or max_dist if no path exists
        """
        try:
            # Try directed path first
            return nx.shortest_path_length(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                # Try reverse direction
                return nx.shortest_path_length(self.graph, target, source)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # No path in either direction
                return max_dist

    def get_all_concepts(self) -> Set[str]:
        """Return all concept names in the graph."""
        return set(self.graph.nodes())


class TokenToSUMOMapper:
    """Maps tokens to SUMO concepts via spaCy→WordNet→SUMO pipeline."""

    def __init__(
        self,
        sumo_graph: SUMOConceptGraph,
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initialize token→SUMO mapper.

        Args:
            sumo_graph: SUMO concept graph for validation
            spacy_model: spaCy model name
        """
        if not SPACY_AVAILABLE:
            raise ImportError(
                "spaCy is required for token mapping. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
        if not NLTK_AVAILABLE:
            raise ImportError(
                "NLTK is required for WordNet. "
                "Install with: pip install nltk && python -c \"import nltk; nltk.download('wordnet')\""
            )

        self.nlp: Language = spacy.load(spacy_model)
        self.sumo_graph = sumo_graph
        self.all_sumo_concepts = sumo_graph.get_all_concepts()

        # Load WordNet→SUMO mappings if available
        self.synset_to_sumo: Dict[str, str] = {}
        self._load_wordnet_sumo_mappings()

    def _load_wordnet_sumo_mappings(self) -> None:
        """Load WordNet synset → SUMO concept mappings."""
        mapping_file = Path("data/concept_graph/sumo_hierarchy/wordnet_sumo_mapping.json")

        if not mapping_file.exists():
            print(f"Warning: WordNet→SUMO mapping file not found at {mapping_file}")
            print("Will attempt heuristic matching via lemma overlap")
            return

        with open(mapping_file) as f:
            data = json.load(f)
            self.synset_to_sumo = data.get("synset_to_sumo", {})

    @lru_cache(maxsize=10000)
    def token_to_sumo(
        self,
        token: str,
        context: str = "",
        use_most_common: bool = True,
        min_token_length: int = 2,
    ) -> Optional[str]:
        """
        Map token to SUMO concept.

        Pipeline:
        1. spaCy: lemma + POS tag
        2. WordNet: lemma.POS → synsets
        3. Pick synset (most common or context-based)
        4. Synset → SUMO via mappings

        Args:
            token: Input token
            context: Context for disambiguation (space-separated tokens)
            use_most_common: Use most common synset if multiple options
            min_token_length: Minimum token length to avoid partial words

        Returns:
            SUMO concept name, or None if no mapping found
        """
        # Filter out partial tokens, punctuation, whitespace
        token_clean = token.strip()
        if (len(token_clean) < min_token_length or
            not token_clean.isalpha() or
            token_clean.startswith('<')):  # Special tokens like <start_of_image>
            return None

        # Step 1: Lemmatize and get POS
        # Use context if provided for better POS tagging
        text_to_parse = f"{context} {token_clean}" if context else token_clean
        doc = self.nlp(text_to_parse)
        if not doc:
            return None

        # Get the target token (last one if we added context)
        target_token = doc[-1] if context else doc[0]

        lemma = target_token.lemma_.lower()
        pos = self._spacy_to_wordnet_pos(target_token.pos_)

        if pos is None:
            return None

        # Step 2: Get WordNet synsets
        synsets = wn.synsets(lemma, pos=pos)
        if not synsets:
            # Try without POS filtering
            synsets = wn.synsets(lemma)

        if not synsets:
            return None

        # Step 3: Pick synset
        if use_most_common:
            # Sort by frequency (requires lemma counts in WordNet)
            # For now, just take first synset (most common)
            synset = synsets[0]
        else:
            synset = synsets[0]  # TODO: Context-based disambiguation

        # Step 4: Map synset → SUMO
        synset_name = synset.name()

        # Try explicit mapping first
        if synset_name in self.synset_to_sumo:
            return self.synset_to_sumo[synset_name]

        # Fallback: heuristic matching via lemma
        return self._heuristic_sumo_match(lemma, synset)

    def _spacy_to_wordnet_pos(self, spacy_pos: str) -> Optional[str]:
        """Convert spaCy POS tag to WordNet POS."""
        pos_map = {
            "NOUN": wn.NOUN,
            "VERB": wn.VERB,
            "ADJ": wn.ADJ,
            "ADV": wn.ADV,
        }
        return pos_map.get(spacy_pos)

    def _heuristic_sumo_match(self, lemma: str, synset) -> Optional[str]:
        """
        Heuristic SUMO matching when explicit mapping unavailable.

        Strategy:
        1. Exact lemma match (case-insensitive)
        2. Check all synset lemmas
        3. Check synset definition for concept names
        4. Fuzzy matching (substring)
        """
        lemma_clean = lemma.strip().lower()

        # 1. Exact match on lemma
        for concept in self.all_sumo_concepts:
            if concept.lower() == lemma_clean:
                return concept

        # 2. Try all synset lemmas (exact match)
        all_synset_lemmas = [l.name().replace('_', ' ').lower() for l in synset.lemmas()]
        for syn_lemma in all_synset_lemmas:
            for concept in self.all_sumo_concepts:
                if concept.lower() == syn_lemma:
                    return concept

        # 3. Try immediate hypernyms (category match)
        for hyper in synset.hypernyms():
            for hyper_lemma in hyper.lemmas():
                lemma_name = hyper_lemma.name().replace('_', ' ').lower()
                for concept in self.all_sumo_concepts:
                    if concept.lower() == lemma_name:
                        return concept

        # 4. Fuzzy matching: find concepts containing lemma
        # Prioritize exact matches, then shorter concepts
        candidates = []
        for concept in self.all_sumo_concepts:
            concept_lower = concept.lower()
            if lemma_clean in concept_lower:
                # Score based on match quality and length
                if concept_lower == lemma_clean:
                    score = 1000  # Exact match
                elif concept_lower.startswith(lemma_clean):
                    # Prefer shorter concepts (less likely to be spurious substring)
                    score = 100 - len(concept)
                else:
                    # Contains but not at start - penalize heavily
                    score = 10 - len(concept)
                candidates.append((concept, score))

        if candidates:
            # Return highest scoring match
            candidates.sort(key=lambda x: x[1], reverse=True)
            # Only accept if score is reasonable (avoid spurious matches)
            if candidates[0][1] >= 50:  # Threshold to avoid Mathematician from "mat"
                return candidates[0][0]

        return None


def concept_divergence(
    token_concept: str,
    detected_concepts: List[Tuple[str, float]],
    graph: SUMOConceptGraph,
    alpha: float = 0.5,
    max_dist: int = 10,
    use_hybrid: bool = False,
    embedding_model=None,
) -> float:
    """
    Compute semantic divergence between expected token concept and detected concepts.

    Measures how far the model's activated concepts are from the expected concept
    for a given token. Higher divergence = more dissonance.

    Hybrid mode: Use graph distance when path exists, embedding similarity otherwise.

    Args:
        token_concept: Expected SUMO concept for the token
        detected_concepts: List of (concept, strength) detected by classifiers
        graph: SUMO concept hierarchy graph
        alpha: Decay parameter for distance (higher = faster decay)
        max_dist: Maximum graph distance (used when no path exists)
        use_hybrid: Enable hybrid graph + embedding mode
        embedding_model: spaCy model or other embedder (for hybrid mode)

    Returns:
        Divergence score [0, 1] where 0 = perfect alignment, 1 = maximum dissonance
    """
    if not detected_concepts:
        return 0.0

    divs = []
    for concept, strength in detected_concepts:
        # Compute graph distance
        dist = graph.shortest_path_length(token_concept, concept, max_dist=max_dist)

        # Check if path exists in graph
        has_path = dist < max_dist

        if use_hybrid and not has_path and embedding_model is not None:
            # Fallback to embedding similarity
            try:
                token_vec = embedding_model(token_concept)
                concept_vec = embedding_model(concept)

                # Compute cosine similarity
                if NUMPY_AVAILABLE and hasattr(token_vec, 'vector') and hasattr(concept_vec, 'vector'):
                    cos_sim = np.dot(token_vec.vector, concept_vec.vector) / (
                        np.linalg.norm(token_vec.vector) * np.linalg.norm(concept_vec.vector) + 1e-10
                    )
                    # Convert similarity to divergence: 1 - similarity
                    div = 1.0 - max(0.0, cos_sim)
                else:
                    # Fallback to max divergence if vectors unavailable
                    div = 1.0
            except Exception:
                # Fallback to graph-based divergence
                div = 1.0 - math.exp(-alpha * dist)
        else:
            # Standard graph-based divergence with exponential decay
            # dist=0 → div=0 (same concept)
            # dist=∞ → div=1 (unrelated)
            div = 1.0 - math.exp(-alpha * dist)

        # Weight by detection strength
        divs.append(div * strength)

    # Normalize by total strength
    total_strength = sum(s for _, s in detected_concepts)
    if total_strength == 0:
        return 0.0

    return sum(divs) / total_strength


def batch_divergence(
    tokens: List[str],
    timesteps: List[Dict],
    graph: SUMOConceptGraph,
    mapper: Optional[TokenToSUMOMapper] = None,
    alpha: float = 0.5,
    context_window: int = 3,
) -> List[Dict]:
    """
    Compute divergence for a batch of tokens with contextual mapping.

    Args:
        tokens: List of token strings
        timesteps: List of timestep dicts from SUMOTemporalMonitor
        graph: SUMO concept graph
        mapper: Token→SUMO mapper (if None, dissonance will be None)
        alpha: Decay parameter
        context_window: Number of following tokens to use for disambiguation

    Returns:
        List of dicts with {token, expected_concept, divergence, detected_concepts}
    """
    results = []

    for i, (token, timestep) in enumerate(zip(tokens, timesteps)):
        # Build context from following tokens (lookahead for disambiguation)
        context = ""
        if mapper is not None and context_window > 0:
            # Get next N tokens
            next_tokens = tokens[i+1:i+1+context_window]
            context = " ".join(t.strip() for t in next_tokens if t.strip())

        # Map token to expected SUMO concept
        expected_concept = None
        if mapper is not None:
            try:
                expected_concept = mapper.token_to_sumo(token, context=context)
            except Exception as e:
                print(f"Warning: Failed to map token '{token}': {e}")

        # Get detected concepts
        detected = [
            (c["concept"], c["probability"])
            for c in timestep.get("concepts", [])
        ]

        # Compute divergence
        divergence = None
        if expected_concept is not None:
            divergence = concept_divergence(
                expected_concept,
                detected,
                graph,
                alpha=alpha,
            )

        results.append({
            "token": token,
            "expected_concept": expected_concept,
            "divergence": divergence,
            "detected_concepts": detected,
        })

    return results


def compute_temporal_lag(
    timesteps: List[Dict],
    max_lag: int = 5,
) -> Dict:
    """
    Compute temporal lag between surface form (token) and latent concepts.

    The model may "think ahead" - activating concepts for future tokens
    before actually generating them. This computes cross-correlation to
    find the optimal lag.

    Args:
        timesteps: List of timestep dicts with concepts and tokens
        max_lag: Maximum lag to search (tokens)

    Returns:
        Dict with lag analysis: {
            'optimal_lag': int,
            'correlation_at_lags': List[float],
            'best_correlation': float,
        }
    """
    if not NUMPY_AVAILABLE or len(timesteps) < max_lag + 2:
        return {
            'optimal_lag': 0,
            'correlation_at_lags': [],
            'best_correlation': 0.0,
        }

    # Extract concept activation series (average probability per timestep)
    latent_series = []
    for ts in timesteps:
        if ts.get('concepts'):
            avg_prob = sum(c['probability'] for c in ts['concepts']) / len(ts['concepts'])
            latent_series.append(avg_prob)
        else:
            latent_series.append(0.0)

    # Extract surface concept match series (1 if expected matches top, 0 otherwise)
    surface_series = []
    for ts in timesteps:
        expected = ts.get('expected_concept')
        if expected and ts.get('concepts'):
            top_concept = ts['concepts'][0]['concept']
            surface_series.append(1.0 if expected == top_concept else 0.0)
        else:
            surface_series.append(0.0)

    latent_arr = np.array(latent_series)
    surface_arr = np.array(surface_series)

    # Compute cross-correlation at different lags
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Surface leads latent (model thinking about past tokens)
            corr = np.corrcoef(surface_arr[:lag], latent_arr[-lag:])[0, 1] if len(surface_arr[:lag]) > 1 else 0.0
        elif lag > 0:
            # Latent leads surface (model thinking ahead)
            corr = np.corrcoef(latent_arr[:-lag], surface_arr[lag:])[0, 1] if len(latent_arr[:-lag]) > 1 else 0.0
        else:
            # No lag
            corr = np.corrcoef(latent_arr, surface_arr)[0, 1] if len(latent_arr) > 1 else 0.0

        # Handle NaN from corrcoef
        correlations.append(0.0 if np.isnan(corr) else corr)

    # Find optimal lag (positive means model thinks ahead)
    optimal_idx = np.argmax(np.abs(correlations))
    optimal_lag = optimal_idx - max_lag

    return {
        'optimal_lag': int(optimal_lag),
        'correlation_at_lags': correlations,
        'best_correlation': float(correlations[optimal_idx]),
    }


__all__ = [
    "SUMOConceptGraph",
    "TokenToSUMOMapper",
    "concept_divergence",
    "batch_divergence",
    "compute_temporal_lag",
]
