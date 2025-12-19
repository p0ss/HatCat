#!/usr/bin/env python3
"""
Feature Weighting for Concept Discrimination

Two approaches to improving concept discrimination through distributional analysis
of activation patterns across all concepts:

## Approach 1: Pre-Training Feature Weighting (FeatureWeightedTrainer)

Uses TF-IDF-style weighting to pre-process activations before training classifiers:
1. Collect activation distributions for all concepts
2. Compute per-feature discrimination scores (features unique to concept = high score)
3. Weight activations by discrimination scores
4. Train classifiers on weighted activations

This reduces noise and makes classifiers' jobs easier by amplifying distinguishing
features and suppressing shared ones.

## Approach 2: Weighted Centroid Classifier (BucketClassifier)

Eliminates per-concept classifiers entirely:
1. Compute weighted centroid for each concept in activation space
2. Classification = find nearest centroid to query activation
3. No training required - pure distributional statistics

The weighting naturally handles:
- "Entity fires on everything" → its unique features get emphasized
- Sibling discrimination → shared parent features get downweighted
- Polysemy → different senses cluster differently

Usage:
    # Approach 1: Generate weighted training data
    python -m training.calibration.feature_weighting \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509 \
        --mode pretraining

    # Approach 2: Build bucket classifier
    python -m training.calibration.feature_weighting \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509 \
        --mode bucket
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class ConceptActivationStats:
    """Statistics for a single concept's activation distribution."""
    concept: str
    layer: int
    n_samples: int

    # Per-feature statistics across all samples for this concept
    mean_activation: np.ndarray      # [hidden_dim] - mean activation per feature
    std_activation: np.ndarray       # [hidden_dim] - std dev per feature
    activation_frequency: np.ndarray # [hidden_dim] - how often feature > threshold

    # Weighted representation (computed after global IDF)
    weighted_centroid: Optional[np.ndarray] = None


@dataclass
class FeatureWeights:
    """Global feature discrimination weights (IDF-style)."""
    hidden_dim: int
    n_concepts: int

    # Per-feature statistics across ALL concepts
    concept_frequency: np.ndarray  # [hidden_dim] - how many concepts activate this feature
    idf_weights: np.ndarray        # [hidden_dim] - log(N / concept_frequency)

    # Optional: per-layer weights if features behave differently by hierarchy level
    layer_weights: Optional[Dict[int, np.ndarray]] = None


@dataclass
class BucketClassifierModel:
    """
    The bucket classifier - stores weighted centroids for nearest-centroid lookup.
    """
    concept_centroids: Dict[str, np.ndarray]  # concept -> weighted centroid
    concept_layers: Dict[str, int]            # concept -> layer
    feature_weights: FeatureWeights
    concept_variances: Optional[Dict[str, np.ndarray]] = None  # For Mahalanobis distance


# =============================================================================
# Approach 1: Pre-Training Feature Weighting
# =============================================================================

class FeatureWeightedTrainer:
    """
    Computes feature weights from activation distributions, then provides
    weighted activations for downstream classifier training.

    The key insight: features that fire for many concepts are less informative
    than features unique to specific concepts. We apply TF-IDF-style weighting:

        weighted_activation[dim] = activation[dim] * idf[dim]

    where:
        idf[dim] = log(N_concepts / n_concepts_where_dim_activates)

    This amplifies distinguishing features and suppresses common ones.
    """

    def __init__(
        self,
        hidden_dim: int,
        activation_threshold: float = 0.1,  # Feature "fires" if > this
        min_concept_frequency: int = 1,     # Minimum concepts for valid IDF
        smoothing: float = 1.0,             # IDF smoothing to avoid div by zero
    ):
        self.hidden_dim = hidden_dim
        self.activation_threshold = activation_threshold
        self.min_concept_frequency = min_concept_frequency
        self.smoothing = smoothing

        # Will be populated during fit()
        self.concept_stats: Dict[str, ConceptActivationStats] = {}
        self.feature_weights: Optional[FeatureWeights] = None

    def fit(
        self,
        concept_activations: Dict[str, Tuple[int, List[np.ndarray]]],
    ) -> FeatureWeights:
        """
        Compute feature weights from activation distributions.

        Args:
            concept_activations: Dict mapping concept name to (layer, list of activation vectors)
                Each activation vector is [hidden_dim]

        Returns:
            FeatureWeights with IDF-style weights per feature
        """
        print("Computing feature weights from activation distributions...")

        n_concepts = len(concept_activations)

        # Track which concepts activate each feature
        feature_concept_sets: List[Set[str]] = [set() for _ in range(self.hidden_dim)]

        # Compute per-concept statistics
        for concept, (layer, activations) in tqdm(concept_activations.items(), desc="Computing concept stats"):
            if not activations:
                continue

            act_array = np.stack(activations)  # [n_samples, hidden_dim]
            n_samples = act_array.shape[0]

            # Per-feature stats for this concept
            mean_act = act_array.mean(axis=0)
            std_act = act_array.std(axis=0)

            # How often does each feature fire (above threshold)?
            fires = (act_array > self.activation_threshold).astype(float)
            freq = fires.mean(axis=0)

            self.concept_stats[concept] = ConceptActivationStats(
                concept=concept,
                layer=layer,
                n_samples=n_samples,
                mean_activation=mean_act,
                std_activation=std_act,
                activation_frequency=freq,
            )

            # Track which features this concept activates
            # A feature is "activated by concept" if it fires in >50% of samples
            for dim in range(self.hidden_dim):
                if freq[dim] > 0.5:
                    feature_concept_sets[dim].add(concept)

        # Compute IDF weights
        concept_frequency = np.array([
            len(feature_concept_sets[dim]) for dim in range(self.hidden_dim)
        ], dtype=np.float32)

        # IDF = log((N + smoothing) / (concept_frequency + smoothing))
        # Higher weight for features that appear in fewer concepts
        idf_weights = np.log(
            (n_concepts + self.smoothing) /
            (concept_frequency + self.smoothing)
        )

        # Normalize to [0, 1] range for interpretability
        idf_weights = idf_weights / idf_weights.max()

        self.feature_weights = FeatureWeights(
            hidden_dim=self.hidden_dim,
            n_concepts=n_concepts,
            concept_frequency=concept_frequency,
            idf_weights=idf_weights,
        )

        # Report statistics
        high_idf = (idf_weights > 0.8).sum()
        low_idf = (idf_weights < 0.2).sum()
        print(f"  Total features: {self.hidden_dim}")
        print(f"  High discrimination (IDF > 0.8): {high_idf} ({100*high_idf/self.hidden_dim:.1f}%)")
        print(f"  Low discrimination (IDF < 0.2): {low_idf} ({100*low_idf/self.hidden_dim:.1f}%)")

        return self.feature_weights

    def weight_activation(self, activation: np.ndarray) -> np.ndarray:
        """
        Apply feature weights to an activation vector.

        Args:
            activation: [hidden_dim] raw activation

        Returns:
            [hidden_dim] weighted activation
        """
        if self.feature_weights is None:
            raise ValueError("Must call fit() before weight_activation()")

        return activation * self.feature_weights.idf_weights

    def weight_activations_batch(self, activations: np.ndarray) -> np.ndarray:
        """
        Apply feature weights to batch of activations.

        Args:
            activations: [batch, hidden_dim]

        Returns:
            [batch, hidden_dim] weighted activations
        """
        if self.feature_weights is None:
            raise ValueError("Must call fit() before weight_activations_batch()")

        return activations * self.feature_weights.idf_weights[np.newaxis, :]

    def get_weighted_training_data(
        self,
        concept: str,
        positive_activations: List[np.ndarray],
        negative_activations: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get weighted training data for a specific concept's classifier.

        Args:
            concept: Concept name
            positive_activations: List of positive example activations
            negative_activations: List of negative example activations

        Returns:
            (weighted_positive, weighted_negative) arrays
        """
        pos_array = np.stack(positive_activations)
        neg_array = np.stack(negative_activations)

        weighted_pos = self.weight_activations_batch(pos_array)
        weighted_neg = self.weight_activations_batch(neg_array)

        return weighted_pos, weighted_neg

    def save(self, path: Path):
        """Save feature weights to file."""
        data = {
            'hidden_dim': self.hidden_dim,
            'activation_threshold': self.activation_threshold,
            'min_concept_frequency': self.min_concept_frequency,
            'smoothing': self.smoothing,
            'feature_weights': {
                'hidden_dim': self.feature_weights.hidden_dim,
                'n_concepts': self.feature_weights.n_concepts,
                'concept_frequency': self.feature_weights.concept_frequency.tolist(),
                'idf_weights': self.feature_weights.idf_weights.tolist(),
            } if self.feature_weights else None,
            'concept_stats': {
                concept: {
                    'concept': stats.concept,
                    'layer': stats.layer,
                    'n_samples': stats.n_samples,
                    'mean_activation': stats.mean_activation.tolist(),
                    'activation_frequency': stats.activation_frequency.tolist(),
                }
                for concept, stats in self.concept_stats.items()
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> 'FeatureWeightedTrainer':
        """Load feature weights from file."""
        with open(path) as f:
            data = json.load(f)

        trainer = cls(
            hidden_dim=data['hidden_dim'],
            activation_threshold=data['activation_threshold'],
            min_concept_frequency=data['min_concept_frequency'],
            smoothing=data['smoothing'],
        )

        if data['feature_weights']:
            fw = data['feature_weights']
            trainer.feature_weights = FeatureWeights(
                hidden_dim=fw['hidden_dim'],
                n_concepts=fw['n_concepts'],
                concept_frequency=np.array(fw['concept_frequency'], dtype=np.float32),
                idf_weights=np.array(fw['idf_weights'], dtype=np.float32),
            )

        return trainer


# =============================================================================
# Approach 2: Concept Activation Map (Weighted Centroids)
# =============================================================================

@dataclass
class ConceptActivation:
    """Activation strength for a single concept."""
    concept: str
    layer: int
    activation: float      # Raw similarity score
    normalized: float      # Normalized to [0, 1] range
    above_threshold: bool  # Whether this concept is considered "active"


@dataclass
class ActivationMap:
    """
    Complete activation map over all concepts.

    This is the output of the concept activation system - analogous to
    running all binary classifiers but computed via weighted centroids.
    """
    # All concept activations, sorted by strength
    activations: List[ConceptActivation]

    # Quick lookups
    by_concept: Dict[str, ConceptActivation]
    by_layer: Dict[int, List[ConceptActivation]]

    # Active concepts (above threshold)
    active_concepts: List[str]

    # Statistics
    n_active: int
    max_activation: float
    mean_activation: float

    def top_k(self, k: int = 10) -> List[ConceptActivation]:
        """Get top-k activated concepts."""
        return self.activations[:k]

    def layer_top_k(self, layer: int, k: int = 5) -> List[ConceptActivation]:
        """Get top-k activated concepts for a specific layer."""
        return self.by_layer.get(layer, [])[:k]

    def is_active(self, concept: str) -> bool:
        """Check if a concept is active."""
        return concept in self.by_concept and self.by_concept[concept].above_threshold


class ConceptActivationMapper:
    """
    Maps model activations to concept activation strengths.

    Unlike a classifier that picks "the best" concept, this produces an
    activation map showing how strongly each concept is expressed in the
    input. Multiple concepts can and should be active simultaneously:

    - "A dog running in the park" activates: Dog, Mammal, Animal, Motion,
      Running, Park, Location, Recreation, etc.
    - Parent concepts activate when children activate (hierarchy)
    - Related concepts show correlated activations (semantic clustering)

    The TF-IDF-style feature weighting ensures that:
    - Features unique to a concept contribute strongly to its activation
    - Features shared across many concepts contribute weakly
    - This naturally handles "Entity fires on everything" by downweighting
      Entity's common features, leaving only its distinctive ones

    Output is equivalent to running all binary classifiers, but computed
    via weighted centroid similarity - much faster and with built-in
    calibration through the feature weighting.
    """

    def __init__(
        self,
        hidden_dim: int,
        activation_threshold: float = 0.5,  # Concepts with similarity > this are "active"
        use_mahalanobis: bool = False,      # Use variance-weighted distance
    ):
        self.hidden_dim = hidden_dim
        self.activation_threshold = activation_threshold
        self.use_mahalanobis = use_mahalanobis

        self.model: Optional[BucketClassifierModel] = None
        self.feature_weighter: Optional[FeatureWeightedTrainer] = None

        # For batch computation
        self._centroid_matrix: Optional[np.ndarray] = None
        self._concept_list: Optional[List[Tuple[str, int]]] = None

    def fit(
        self,
        concept_activations: Dict[str, Tuple[int, List[np.ndarray]]],
    ) -> BucketClassifierModel:
        """
        Build the activation mapper from concept activation distributions.

        Args:
            concept_activations: Dict mapping concept name to (layer, list of activation vectors)

        Returns:
            BucketClassifierModel with weighted centroids
        """
        print("Building concept activation mapper...")

        # First compute feature weights (TF-IDF style)
        self.feature_weighter = FeatureWeightedTrainer(
            hidden_dim=self.hidden_dim,
            activation_threshold=0.1,  # For determining if feature "fires"
        )
        feature_weights = self.feature_weighter.fit(concept_activations)

        # Compute weighted centroids for each concept
        concept_centroids = {}
        concept_layers = {}
        concept_variances = {} if self.use_mahalanobis else None

        for concept, (layer, activations) in tqdm(concept_activations.items(), desc="Computing centroids"):
            if not activations:
                continue

            act_array = np.stack(activations)  # [n_samples, hidden_dim]

            # Weight the activations by feature discrimination
            weighted = self.feature_weighter.weight_activations_batch(act_array)

            # Compute centroid (mean of weighted activations)
            centroid = weighted.mean(axis=0)

            # L2 normalize for cosine similarity
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            concept_centroids[concept] = centroid
            concept_layers[concept] = layer

            # Optionally compute variance for Mahalanobis distance
            if self.use_mahalanobis and len(activations) > 1:
                variance = weighted.var(axis=0) + 1e-6
                concept_variances[concept] = variance

        self.model = BucketClassifierModel(
            concept_centroids=concept_centroids,
            concept_layers=concept_layers,
            feature_weights=feature_weights,
            concept_variances=concept_variances,
        )

        # Pre-compute centroid matrix for batch operations
        self._concept_list = [(c, concept_layers[c]) for c in concept_centroids.keys()]
        self._centroid_matrix = np.stack([
            concept_centroids[c] for c, _ in self._concept_list
        ])  # [n_concepts, hidden_dim]

        print(f"  Built mapper with {len(concept_centroids)} concept centroids")

        return self.model

    def compute_activations(
        self,
        activation: np.ndarray,
        layer_filter: Optional[List[int]] = None,
    ) -> ActivationMap:
        """
        Compute activation strengths for all concepts.

        Args:
            activation: [hidden_dim] model activation vector
            layer_filter: Only include concepts from these layers (None = all)

        Returns:
            ActivationMap with activation strengths for all concepts
        """
        if self.model is None or self.feature_weighter is None:
            raise ValueError("Must call fit() before compute_activations()")

        # Weight and normalize the query activation
        weighted_query = self.feature_weighter.weight_activation(activation)
        weighted_query = weighted_query / (np.linalg.norm(weighted_query) + 1e-8)

        # Batch compute cosine similarities: [n_concepts]
        # Since both are L2 normalized, dot product = cosine similarity
        similarities = self._centroid_matrix @ weighted_query

        # Build activation list
        activations = []
        min_sim = similarities.min()
        max_sim = similarities.max()
        sim_range = max_sim - min_sim + 1e-8

        for i, (concept, layer) in enumerate(self._concept_list):
            if layer_filter and layer not in layer_filter:
                continue

            sim = similarities[i]
            # Normalize to [0, 1] based on observed range
            normalized = (sim - min_sim) / sim_range

            activations.append(ConceptActivation(
                concept=concept,
                layer=layer,
                activation=float(sim),
                normalized=float(normalized),
                above_threshold=normalized >= self.activation_threshold,
            ))

        # Sort by activation strength
        activations.sort(key=lambda x: x.activation, reverse=True)

        # Build lookups
        by_concept = {a.concept: a for a in activations}
        by_layer: Dict[int, List[ConceptActivation]] = defaultdict(list)
        for a in activations:
            by_layer[a.layer].append(a)

        # Sort each layer's list
        for layer in by_layer:
            by_layer[layer].sort(key=lambda x: x.activation, reverse=True)

        active_concepts = [a.concept for a in activations if a.above_threshold]

        return ActivationMap(
            activations=activations,
            by_concept=by_concept,
            by_layer=dict(by_layer),
            active_concepts=active_concepts,
            n_active=len(active_concepts),
            max_activation=float(max_sim),
            mean_activation=float(similarities.mean()),
        )

    def compute_activations_batch(
        self,
        activations: np.ndarray,
        layer_filter: Optional[List[int]] = None,
    ) -> List[ActivationMap]:
        """
        Compute activation maps for a batch of activations.

        Args:
            activations: [batch, hidden_dim] model activations
            layer_filter: Only include concepts from these layers

        Returns:
            List of ActivationMaps, one per input
        """
        results = []
        for i in range(activations.shape[0]):
            results.append(self.compute_activations(activations[i], layer_filter))
        return results

    def get_active_concepts(
        self,
        activation: np.ndarray,
        threshold: Optional[float] = None,
        layer_filter: Optional[List[int]] = None,
    ) -> List[Tuple[str, float, int]]:
        """
        Get list of active concepts (above threshold).

        Convenience method that returns same format as binary classifiers.

        Args:
            activation: [hidden_dim] model activation
            threshold: Override default threshold
            layer_filter: Only include concepts from these layers

        Returns:
            List of (concept, activation, layer) for active concepts
        """
        act_map = self.compute_activations(activation, layer_filter)
        thresh = threshold if threshold is not None else self.activation_threshold

        return [
            (a.concept, a.activation, a.layer)
            for a in act_map.activations
            if a.normalized >= thresh
        ]

    def save(self, path: Path):
        """Save activation mapper to files."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save feature weights
        self.feature_weighter.save(path / "feature_weights.json")

        # Save centroids as numpy array
        np.save(path / "centroids.npy", self._centroid_matrix)

        # Save metadata
        metadata = {
            'hidden_dim': self.hidden_dim,
            'activation_threshold': self.activation_threshold,
            'use_mahalanobis': self.use_mahalanobis,
            'concept_list': self._concept_list,
            'n_concepts': len(self._concept_list),
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save variances if using Mahalanobis
        if self.model.concept_variances:
            var_data = {k: v.tolist() for k, v in self.model.concept_variances.items()}
            with open(path / "variances.json", 'w') as f:
                json.dump(var_data, f)

        print(f"Saved activation mapper to {path}")

    @classmethod
    def load(cls, path: Path) -> 'ConceptActivationMapper':
        """Load activation mapper from files."""
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        mapper = cls(
            hidden_dim=metadata['hidden_dim'],
            activation_threshold=metadata['activation_threshold'],
            use_mahalanobis=metadata['use_mahalanobis'],
        )

        # Load feature weights
        mapper.feature_weighter = FeatureWeightedTrainer.load(path / "feature_weights.json")

        # Load centroids
        mapper._centroid_matrix = np.load(path / "centroids.npy")
        mapper._concept_list = [tuple(x) for x in metadata['concept_list']]

        # Reconstruct model
        concept_centroids = {
            mapper._concept_list[i][0]: mapper._centroid_matrix[i]
            for i in range(len(mapper._concept_list))
        }
        concept_layers = {c: l for c, l in mapper._concept_list}

        # Load variances if present
        var_path = path / "variances.json"
        concept_variances = None
        if var_path.exists():
            with open(var_path) as f:
                var_data = json.load(f)
            concept_variances = {k: np.array(v) for k, v in var_data.items()}

        mapper.model = BucketClassifierModel(
            concept_centroids=concept_centroids,
            concept_layers=concept_layers,
            feature_weights=mapper.feature_weighter.feature_weights,
            concept_variances=concept_variances,
        )

        return mapper


# Backward compatibility alias
BucketClassifier = ConceptActivationMapper


# =============================================================================
# Neural Concept Activation Mapper (Trainable version)
# =============================================================================

class NeuralConceptMapper(nn.Module):
    """
    A trainable neural network for concept activation mapping.

    Unlike the statistical ConceptActivationMapper, this learns:
    1. A feature transformation (amplifies discriminative features)
    2. A projection to embedding space
    3. Concept embeddings (learned centroids)

    Can be initialized from a ConceptActivationMapper for warm-start training,
    then fine-tuned to improve activation quality.

    The output is activation strengths for ALL concepts (not a classification).
    Multiple concepts can and should be active simultaneously.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_concepts: int,
        embedding_dim: int = 256,
        concept_names: Optional[List[str]] = None,
        concept_layers: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_concepts = n_concepts
        self.embedding_dim = embedding_dim
        self.concept_names = concept_names or []
        self.concept_layers = concept_layers or {}

        # Feature transformation (learns to emphasize discriminative features)
        # This replaces the hand-computed IDF weights with learned weights
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Project to embedding space
        self.projection = nn.Linear(hidden_dim, embedding_dim)

        # Concept embeddings (each concept gets a learned centroid)
        self.concept_embeddings = nn.Parameter(torch.randn(n_concepts, embedding_dim))

        # Optional: store pre-computed IDF weights for reference
        self.pretrained_idf: Optional[torch.Tensor] = None

    def initialize_from_mapper(self, mapper: ConceptActivationMapper):
        """
        Initialize weights from a pre-computed ConceptActivationMapper.

        This gives the network a good starting point based on the
        distributional analysis.
        """
        if mapper.feature_weighter is None:
            raise ValueError("ConceptActivationMapper must be fitted first")

        # Store concept metadata
        self.concept_names = [c for c, _ in mapper._concept_list]
        self.concept_layers = {c: l for c, l in mapper._concept_list}

        # Use IDF weights to initialize feature transform
        idf = mapper.feature_weighter.feature_weights.idf_weights
        self.pretrained_idf = torch.tensor(idf, dtype=torch.float32)

        # Initialize first layer of feature_transform with IDF as diagonal
        with torch.no_grad():
            # Make it a diagonal matrix scaled by IDF
            self.feature_transform[0].weight.copy_(
                torch.diag(self.pretrained_idf)
            )
            self.feature_transform[0].bias.zero_()

        # Initialize concept embeddings from centroids
        # Project centroids through the (now IDF-initialized) transform
        for i, concept in enumerate(self.concept_names):
            centroid = mapper.model.concept_centroids[concept]
            # Initialize with small random values - the projection will learn
            self.concept_embeddings.data[i] = torch.randn(self.embedding_dim) * 0.1

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - compute activation strengths for all concepts.

        Args:
            activation: [batch, hidden_dim] or [hidden_dim]

        Returns:
            [batch, n_concepts] or [n_concepts] activation strengths (cosine similarities)
        """
        # Handle single activation
        squeeze_output = False
        if activation.dim() == 1:
            activation = activation.unsqueeze(0)
            squeeze_output = True

        # Transform features
        x = self.feature_transform(activation)

        # Project to embedding space
        x = self.projection(x)  # [batch, embedding_dim]

        # Compute similarity to all concept embeddings
        # Using cosine similarity - this gives activation strength per concept
        x_norm = F.normalize(x, dim=-1)
        emb_norm = F.normalize(self.concept_embeddings, dim=-1)

        # [batch, n_concepts] - these are the activation strengths
        activations = torch.matmul(x_norm, emb_norm.T)

        if squeeze_output:
            activations = activations.squeeze(0)

        return activations

    def compute_activations(
        self,
        activation: torch.Tensor,
        threshold: float = 0.5,
        layer_filter: Optional[List[int]] = None,
    ) -> ActivationMap:
        """
        Compute activation map (compatible with ConceptActivationMapper).

        Args:
            activation: [hidden_dim] model activation tensor
            threshold: Threshold for "active" concepts
            layer_filter: Only include concepts from these layers

        Returns:
            ActivationMap with activation strengths for all concepts
        """
        self.eval()
        with torch.no_grad():
            # Get activation strengths for all concepts
            strengths = self.forward(activation)  # [n_concepts]
            strengths_np = strengths.cpu().numpy()

        # Normalize to [0, 1]
        min_s = strengths_np.min()
        max_s = strengths_np.max()
        range_s = max_s - min_s + 1e-8

        # Build activation list
        activations = []
        for i, concept in enumerate(self.concept_names):
            layer = self.concept_layers.get(concept, 0)

            if layer_filter and layer not in layer_filter:
                continue

            normalized = (strengths_np[i] - min_s) / range_s

            activations.append(ConceptActivation(
                concept=concept,
                layer=layer,
                activation=float(strengths_np[i]),
                normalized=float(normalized),
                above_threshold=normalized >= threshold,
            ))

        # Sort by activation strength
        activations.sort(key=lambda x: x.activation, reverse=True)

        # Build lookups
        by_concept = {a.concept: a for a in activations}
        by_layer: Dict[int, List[ConceptActivation]] = defaultdict(list)
        for a in activations:
            by_layer[a.layer].append(a)

        for layer in by_layer:
            by_layer[layer].sort(key=lambda x: x.activation, reverse=True)

        active_concepts = [a.concept for a in activations if a.above_threshold]

        return ActivationMap(
            activations=activations,
            by_concept=by_concept,
            by_layer=dict(by_layer),
            active_concepts=active_concepts,
            n_active=len(active_concepts),
            max_activation=float(max_s),
            mean_activation=float(strengths_np.mean()),
        )

    def get_active_concepts(
        self,
        activation: torch.Tensor,
        threshold: float = 0.5,
        layer_filter: Optional[List[int]] = None,
    ) -> List[Tuple[str, float, int]]:
        """
        Get list of active concepts.

        Convenience method that returns same format as binary classifiers.
        """
        act_map = self.compute_activations(activation, threshold, layer_filter)
        return [
            (a.concept, a.activation, a.layer)
            for a in act_map.activations
            if a.above_threshold
        ]


# Backward compatibility alias
NeuralBucketClassifier = NeuralConceptMapper


# =============================================================================
# Utilities and Main
# =============================================================================

def collect_concept_activations(
    model,
    tokenizer,
    concept_pack_dir: Path,
    layers: List[int],
    device: str,
    layer_idx: int = 15,
    n_samples_per_concept: int = 10,
    fast_mode: bool = True,
) -> Dict[str, Tuple[int, List[np.ndarray]]]:
    """
    Collect activation samples for all concepts.

    Returns:
        Dict mapping concept name to (layer, list of activation vectors)
    """
    print(f"Collecting activations for concepts...")

    concept_activations = {}

    for layer in layers:
        layer_file = concept_pack_dir / "hierarchy" / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        concept_list = layer_data.get('concepts', layer_data if isinstance(layer_data, list) else [])

        for concept in tqdm(concept_list, desc=f"Layer {layer}"):
            term = concept.get('sumo_term') or concept.get('term')
            if not term:
                continue

            # Get prompts
            if fast_mode:
                prompts = [term]
            else:
                hints = concept.get('training_hints', {})
                pos_examples = hints.get('positive_examples', [])[:n_samples_per_concept]
                if not pos_examples:
                    prompts = [f"Tell me about {term}."]
                else:
                    prompts = pos_examples

            # Collect activations
            activations = []
            for prompt in prompts[:n_samples_per_concept]:
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        hidden_states = outputs.hidden_states[layer_idx]
                        activation = hidden_states[0, -1, :].float().cpu().numpy()

                    activations.append(activation)
                except Exception as e:
                    print(f"  Error collecting activation for {term}: {e}")
                    continue

            if activations:
                concept_activations[term] = (layer, activations)

    print(f"  Collected activations for {len(concept_activations)} concepts")
    return concept_activations


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Feature weighting for concept discrimination')
    parser.add_argument('--concept-pack', required=True, help='Path to concept pack')
    parser.add_argument('--model', required=True, help='Model name/path')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help='Layers to process')
    parser.add_argument('--mode', choices=['pretraining', 'bucket'], default='bucket',
                        help='Mode: pretraining (compute weights) or bucket (build classifier)')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--layer-idx', type=int, default=15, help='Model layer for activations')
    parser.add_argument('--n-samples', type=int, default=10, help='Samples per concept')
    parser.add_argument('--fast-mode', action='store_true', help='Use concept name as prompt')

    args = parser.parse_args()

    concept_pack_dir = Path(args.concept_pack)

    # Auto-detect layers from concept pack
    if args.layers is None:
        layers = []
        hierarchy_dir = concept_pack_dir / "hierarchy"
        for layer_file in hierarchy_dir.glob("layer*.json"):
            try:
                layer_num = int(layer_file.stem.replace('layer', ''))
                layers.append(layer_num)
            except ValueError:
                pass
        layers.sort()
    else:
        layers = args.layers

    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    # Collect activations
    concept_activations = collect_concept_activations(
        model=model,
        tokenizer=tokenizer,
        concept_pack_dir=concept_pack_dir,
        layers=layers,
        device=args.device,
        layer_idx=args.layer_idx,
        n_samples_per_concept=args.n_samples,
        fast_mode=args.fast_mode,
    )

    hidden_dim = model.config.hidden_size

    if args.mode == 'pretraining':
        # Approach 1: Compute feature weights for pre-training
        trainer = FeatureWeightedTrainer(hidden_dim=hidden_dim)
        trainer.fit(concept_activations)

        output_path = Path(args.output) if args.output else concept_pack_dir / "feature_weights.json"
        trainer.save(output_path)
        print(f"\nSaved feature weights to: {output_path}")

    else:
        # Approach 2: Build bucket classifier
        classifier = BucketClassifier(hidden_dim=hidden_dim)
        classifier.fit(concept_activations)

        output_path = Path(args.output) if args.output else concept_pack_dir / "bucket_classifier"
        classifier.save(output_path)

        # Test classification on a few concepts
        print("\nTesting classification:")
        test_concepts = list(concept_activations.keys())[:5]
        for concept in test_concepts:
            layer, activations = concept_activations[concept]
            if activations:
                results = classifier.classify(activations[0], top_k=5)
                print(f"\n  Query: {concept}")
                for c, score, l in results:
                    marker = " <--" if c == concept else ""
                    print(f"    {c}: {score:.3f}{marker}")


if __name__ == '__main__':
    main()
