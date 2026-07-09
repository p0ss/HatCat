"""
Cluster Builder

Clusters neurons by their connectivity patterns to create topology clusters.
These clusters represent groups of neurons that "work together" - they have
similar upstream sources and downstream targets.

The clusters are content-neutral (based on structure, not activation values)
and can be used for:
- Cleft identification (which clusters to include in targeted training)
- Dead zone detection (clusters with low semantic activation)
- Semantic mapping (which concepts light up which clusters)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import json
import logging
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .aggregator import ConnectivityMap

logger = logging.getLogger(__name__)


@dataclass
class NeuronCluster:
    """A cluster of neurons with similar connectivity."""
    cluster_id: int
    neurons: List[Tuple[int, int]]  # List of (layer, neuron_idx)
    centroid: Optional[np.ndarray] = None  # Centroid in connectivity space
    size: int = 0
    layer_distribution: Dict[int, int] = field(default_factory=dict)  # layer -> count


@dataclass
class TopologyClusters:
    """
    Complete clustering of the model's neurons by connectivity.

    Each cluster contains neurons that have similar connectivity patterns
    (similar upstream sources and downstream targets).
    """
    n_clusters: int
    n_layers: int
    hidden_dim: int

    # Cluster definitions
    clusters: List[NeuronCluster] = field(default_factory=list)

    # Quick lookup: (layer, neuron) -> cluster_id
    neuron_to_cluster: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Per-layer cluster distribution
    layer_cluster_distribution: Dict[int, Dict[int, int]] = field(default_factory=dict)

    def get_cluster(self, cluster_id: int) -> Optional[NeuronCluster]:
        """Get cluster by ID."""
        for c in self.clusters:
            if c.cluster_id == cluster_id:
                return c
        return None

    def get_neuron_cluster(self, layer: int, neuron: int) -> int:
        """Get cluster ID for a neuron."""
        return self.neuron_to_cluster.get((layer, neuron), -1)

    def get_cluster_neurons(self, cluster_id: int) -> List[Tuple[int, int]]:
        """Get all neurons in a cluster."""
        cluster = self.get_cluster(cluster_id)
        return cluster.neurons if cluster else []

    def get_layer_clusters(self, layer: int) -> Dict[int, List[int]]:
        """Get cluster assignments for all neurons in a layer."""
        result = {}
        for (l, n), c in self.neuron_to_cluster.items():
            if l == layer:
                if c not in result:
                    result[c] = []
                result[c].append(n)
        return result


class ClusterBuilder:
    """
    Builds topology clusters from connectivity map.

    Usage:
        builder = ClusterBuilder(connectivity_map)
        clusters = builder.build(n_clusters=50)
    """

    def __init__(self, connectivity_map: ConnectivityMap):
        self.conn_map = connectivity_map
        self.n_layers = connectivity_map.n_layers
        self.hidden_dim = connectivity_map.hidden_dim

        # Feature matrix: each neuron gets a feature vector from its connectivity
        self._feature_matrix: Optional[np.ndarray] = None
        self._neuron_index: List[Tuple[int, int]] = []  # Maps row index to (layer, neuron)

    def _build_feature_matrix(self):
        """
        Build feature matrix where each row is a neuron and columns are
        connectivity features (upstream sources + downstream targets).
        """
        logger.info("Building feature matrix from connectivity...")

        # For each neuron, we build a feature vector from:
        # 1. Its downstream connectivity (what does it influence?)
        # 2. Its upstream connectivity (what influences it?)

        features = []
        self._neuron_index = []

        for layer in range(self.n_layers):
            for neuron in range(self.hidden_dim):
                # Downstream features: connectivity to later layers
                downstream = []
                for target_layer in range(layer + 1, self.n_layers):
                    key = (layer, target_layer)
                    if key in self.conn_map.combined_connectivity:
                        conn = self.conn_map.combined_connectivity[key]
                        # Check if neuron index is in bounds (may have been subsampled)
                        if neuron < conn.shape[0]:
                            # Use top-k connectivity values as features
                            top_k = min(10, conn.shape[1])
                            downstream.extend(np.sort(conn[neuron, :])[-top_k:])

                # Upstream features: connectivity from earlier layers
                upstream = []
                for source_layer in range(layer):
                    key = (source_layer, layer)
                    if key in self.conn_map.combined_connectivity:
                        conn = self.conn_map.combined_connectivity[key]
                        # Check if neuron index is in bounds
                        if neuron < conn.shape[1]:
                            # Use top-k connectivity values as features
                            top_k = min(10, conn.shape[0])
                            upstream.extend(np.sort(conn[:, neuron])[-top_k:])

                # Combine into feature vector
                # Pad to consistent length
                max_downstream = 10 * (self.n_layers - 1)
                max_upstream = 10 * (self.n_layers - 1)

                downstream = downstream[:max_downstream]
                downstream += [0] * (max_downstream - len(downstream))

                upstream = upstream[:max_upstream]
                upstream += [0] * (max_upstream - len(upstream))

                feature_vec = np.array(downstream + upstream, dtype=np.float32)
                features.append(feature_vec)
                self._neuron_index.append((layer, neuron))

        self._feature_matrix = np.stack(features)
        logger.info(f"Feature matrix shape: {self._feature_matrix.shape}")

    def build(
        self,
        n_clusters: int = 50,
        method: str = "kmeans",
        random_state: int = 42,
    ) -> TopologyClusters:
        """
        Build topology clusters.

        Args:
            n_clusters: Number of clusters to create
            method: Clustering method ("kmeans" or "spectral")
            random_state: Random seed for reproducibility

        Returns:
            TopologyClusters with cluster assignments
        """
        if self._feature_matrix is None:
            self._build_feature_matrix()

        logger.info(f"Clustering {len(self._neuron_index)} neurons into {n_clusters} clusters...")

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self._feature_matrix)

        # Handle NaN/Inf from scaling
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Cluster
        if method == "kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
            )
        elif method == "spectral":
            clusterer = SpectralClustering(
                n_clusters=n_clusters,
                random_state=random_state,
                affinity='nearest_neighbors',
                n_neighbors=10,
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        labels = clusterer.fit_predict(features_scaled)

        # Build TopologyClusters
        clusters_result = TopologyClusters(
            n_clusters=n_clusters,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
        )

        # Group neurons by cluster
        cluster_neurons: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(n_clusters)}

        for idx, label in enumerate(labels):
            layer, neuron = self._neuron_index[idx]
            cluster_neurons[label].append((layer, neuron))
            clusters_result.neuron_to_cluster[(layer, neuron)] = int(label)

        # Build cluster objects
        for cluster_id, neurons in cluster_neurons.items():
            layer_dist = {}
            for layer, _ in neurons:
                layer_dist[layer] = layer_dist.get(layer, 0) + 1

            # Compute centroid in feature space
            neuron_indices = [self._neuron_index.index((l, n)) for l, n in neurons]
            if neuron_indices:
                centroid = features_scaled[neuron_indices].mean(axis=0)
            else:
                centroid = None

            cluster = NeuronCluster(
                cluster_id=cluster_id,
                neurons=neurons,
                centroid=centroid,
                size=len(neurons),
                layer_distribution=layer_dist,
            )
            clusters_result.clusters.append(cluster)

        # Build layer distribution
        for layer in range(self.n_layers):
            dist = {}
            for neuron in range(self.hidden_dim):
                cluster_id = clusters_result.neuron_to_cluster.get((layer, neuron), -1)
                if cluster_id >= 0:
                    dist[cluster_id] = dist.get(cluster_id, 0) + 1
            clusters_result.layer_cluster_distribution[layer] = dist

        logger.info(f"Built {n_clusters} clusters")
        self._log_cluster_stats(clusters_result)

        return clusters_result

    def _log_cluster_stats(self, clusters: TopologyClusters):
        """Log statistics about the clusters."""
        sizes = [c.size for c in clusters.clusters]
        logger.info(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")

        # Check if clusters span layers
        multi_layer_clusters = sum(1 for c in clusters.clusters if len(c.layer_distribution) > 1)
        logger.info(f"Multi-layer clusters: {multi_layer_clusters}/{len(clusters.clusters)}")

    def find_optimal_clusters(
        self,
        k_range: Optional[List[int]] = None,
        method: str = "kmeans",
        random_state: int = 42,
        sample_size: Optional[int] = None,
    ) -> Dict:
        """
        Find optimal number of clusters using elbow method and silhouette score.

        Args:
            k_range: List of k values to try (default: [10, 25, 50, 100, 200, 300, 500])
            method: Clustering method
            random_state: Random seed
            sample_size: Subsample for faster silhouette (None = use all, recommended for large datasets)

        Returns:
            Dict with 'optimal_k', 'inertias', 'silhouettes', 'k_values'
        """
        if self._feature_matrix is None:
            self._build_feature_matrix()

        if k_range is None:
            k_range = [10, 25, 50, 100, 200, 300, 500]

        # Filter k values that are too large
        max_k = len(self._neuron_index) // 10  # At least 10 samples per cluster
        k_range = [k for k in k_range if k < max_k]

        logger.info(f"Finding optimal clusters, testing k in {k_range}")

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self._feature_matrix)
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Subsample for silhouette if dataset is large
        if sample_size and len(features_scaled) > sample_size:
            np.random.seed(random_state)
            sample_idx = np.random.choice(len(features_scaled), sample_size, replace=False)
            features_sample = features_scaled[sample_idx]
        else:
            sample_idx = None
            features_sample = features_scaled

        inertias = []
        silhouettes = []

        for k in k_range:
            logger.info(f"  Testing k={k}...")

            if method == "kmeans":
                clusterer = KMeans(n_clusters=k, random_state=random_state, n_init=5)
                clusterer.fit(features_scaled)
                inertias.append(clusterer.inertia_)
                labels = clusterer.labels_
            else:
                # Spectral doesn't have inertia
                clusterer = SpectralClustering(
                    n_clusters=k, random_state=random_state,
                    affinity='nearest_neighbors', n_neighbors=10
                )
                labels = clusterer.fit_predict(features_scaled)
                inertias.append(np.nan)

            # Silhouette on sample for speed
            if sample_idx is not None:
                sil = silhouette_score(features_sample, labels[sample_idx])
            else:
                sil = silhouette_score(features_scaled, labels)
            silhouettes.append(sil)

            logger.info(f"    k={k}: inertia={inertias[-1]:.0f}, silhouette={sil:.4f}")

        # Find elbow using second derivative (acceleration)
        if len(inertias) >= 3 and not np.isnan(inertias[0]):
            # Compute rate of change
            diffs = np.diff(inertias)
            # Compute acceleration (second derivative)
            accel = np.diff(diffs)
            # Elbow is where acceleration is maximum (biggest bend)
            elbow_idx = np.argmax(accel) + 1  # +1 because diff reduces length
            elbow_k = k_range[elbow_idx]
        else:
            elbow_k = k_range[np.argmax(silhouettes)]

        # Also get best silhouette
        best_sil_k = k_range[np.argmax(silhouettes)]

        logger.info(f"Elbow method suggests k={elbow_k}")
        logger.info(f"Best silhouette at k={best_sil_k} (score={max(silhouettes):.4f})")

        return {
            'elbow_k': elbow_k,
            'silhouette_k': best_sil_k,
            'optimal_k': elbow_k,  # Default to elbow
            'k_values': k_range,
            'inertias': inertias,
            'silhouettes': silhouettes,
        }

    def save(self, clusters: TopologyClusters, path: str):
        """Save clusters to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save neuron-to-cluster mapping
        mapping = {f"{l}_{n}": c for (l, n), c in clusters.neuron_to_cluster.items()}
        with open(path / "neuron_to_cluster.json", "w") as f:
            json.dump(mapping, f)

        # Save cluster info
        cluster_info = []
        for c in clusters.clusters:
            cluster_info.append({
                "cluster_id": c.cluster_id,
                "size": c.size,
                "layer_distribution": c.layer_distribution,
                "neurons": [list(n) for n in c.neurons[:100]],  # Save first 100 neurons
            })
        with open(path / "clusters.json", "w") as f:
            json.dump(cluster_info, f, indent=2)

        # Save metadata
        metadata = {
            "n_clusters": clusters.n_clusters,
            "n_layers": clusters.n_layers,
            "hidden_dim": clusters.hidden_dim,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save feature matrix for later analysis
        if self._feature_matrix is not None:
            np.save(path / "feature_matrix.npy", self._feature_matrix)

        logger.info(f"Saved clusters to {path}")

    @classmethod
    def load(cls, path: str) -> TopologyClusters:
        """Load clusters from disk."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        with open(path / "neuron_to_cluster.json") as f:
            mapping_raw = json.load(f)

        with open(path / "clusters.json") as f:
            cluster_info = json.load(f)

        clusters = TopologyClusters(
            n_clusters=metadata["n_clusters"],
            n_layers=metadata["n_layers"],
            hidden_dim=metadata["hidden_dim"],
        )

        # Rebuild mapping
        for key, cluster_id in mapping_raw.items():
            l, n = map(int, key.split("_"))
            clusters.neuron_to_cluster[(l, n)] = cluster_id

        # Rebuild cluster objects
        for info in cluster_info:
            cluster = NeuronCluster(
                cluster_id=info["cluster_id"],
                neurons=[tuple(n) for n in info["neurons"]],
                size=info["size"],
                layer_distribution={int(k): v for k, v in info["layer_distribution"].items()},
            )
            clusters.clusters.append(cluster)

        # Rebuild layer distribution
        for layer in range(clusters.n_layers):
            dist = {}
            for neuron in range(clusters.hidden_dim):
                cluster_id = clusters.neuron_to_cluster.get((layer, neuron), -1)
                if cluster_id >= 0:
                    dist[cluster_id] = dist.get(cluster_id, 0) + 1
            clusters.layer_cluster_distribution[layer] = dist

        return clusters


def quick_cluster_stats(clusters: TopologyClusters) -> Dict:
    """Get quick statistics about clusters."""
    sizes = [c.size for c in clusters.clusters]

    # Find which clusters span the most layers
    layer_spans = [(c.cluster_id, len(c.layer_distribution)) for c in clusters.clusters]
    layer_spans.sort(key=lambda x: x[1], reverse=True)

    return {
        "n_clusters": clusters.n_clusters,
        "total_neurons": sum(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": float(np.mean(sizes)),
        "std_size": float(np.std(sizes)),
        "multi_layer_clusters": sum(1 for _, span in layer_spans if span > 1),
        "top_spanning_clusters": layer_spans[:5],
    }
