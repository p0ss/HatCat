"""
Alternative Clustering Algorithms

Provides DBSCAN and Spectral clustering as alternatives to k-means
for validating that discovered clusters are robust to methodology.

If clusters persist across different algorithms, they're more likely
to represent real structure rather than k-means artifacts.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
from pathlib import Path
import json
import logging

from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from .aggregator import ConnectivityMap
from .builder import TopologyClusters, NeuronCluster, ClusterBuilder

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Result from any clustering algorithm."""
    algorithm: str
    n_clusters: int
    labels: np.ndarray
    silhouette: float
    # Algorithm-specific metrics
    extra_metrics: Dict = field(default_factory=dict)


class AlternativeClusterBuilder:
    """
    Builds clusters using multiple algorithms for comparison.

    Usage:
        builder = AlternativeClusterBuilder(connectivity_map)
        results = builder.compare_algorithms(
            algorithms=["kmeans", "dbscan", "spectral"],
            n_clusters=50  # For k-means and spectral
        )
    """

    def __init__(self, connectivity_map: ConnectivityMap):
        self.conn_map = connectivity_map
        self.n_layers = connectivity_map.n_layers
        self.hidden_dim = connectivity_map.hidden_dim

        self._feature_matrix: Optional[np.ndarray] = None
        self._neuron_index: List[Tuple[int, int]] = []
        self._features_scaled: Optional[np.ndarray] = None

    def _build_feature_matrix(self):
        """Build feature matrix (same as ClusterBuilder)."""
        if self._feature_matrix is not None:
            return

        logger.info("Building feature matrix from connectivity...")

        features = []
        self._neuron_index = []

        for layer in range(self.n_layers):
            for neuron in range(self.hidden_dim):
                downstream = []
                for target_layer in range(layer + 1, self.n_layers):
                    key = (layer, target_layer)
                    if key in self.conn_map.combined_connectivity:
                        conn = self.conn_map.combined_connectivity[key]
                        if neuron < conn.shape[0]:
                            top_k = min(10, conn.shape[1])
                            downstream.extend(np.sort(conn[neuron, :])[-top_k:])

                upstream = []
                for source_layer in range(layer):
                    key = (source_layer, layer)
                    if key in self.conn_map.combined_connectivity:
                        conn = self.conn_map.combined_connectivity[key]
                        if neuron < conn.shape[1]:
                            top_k = min(10, conn.shape[0])
                            upstream.extend(np.sort(conn[:, neuron])[-top_k:])

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

        # Pre-scale features
        scaler = StandardScaler()
        self._features_scaled = scaler.fit_transform(self._feature_matrix)
        self._features_scaled = np.nan_to_num(
            self._features_scaled, nan=0.0, posinf=0.0, neginf=0.0
        )

        logger.info(f"Feature matrix shape: {self._feature_matrix.shape}")

    def cluster_kmeans(
        self,
        n_clusters: int = 50,
        random_state: int = 42,
    ) -> ClusteringResult:
        """Cluster using k-means."""
        self._build_feature_matrix()

        logger.info(f"K-means clustering with k={n_clusters}")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(self._features_scaled)

        silhouette = silhouette_score(self._features_scaled, labels)

        return ClusteringResult(
            algorithm="kmeans",
            n_clusters=n_clusters,
            labels=labels,
            silhouette=silhouette,
            extra_metrics={
                "inertia": kmeans.inertia_,
            }
        )

    def cluster_dbscan(
        self,
        eps: Optional[float] = None,
        min_samples: int = 5,
    ) -> ClusteringResult:
        """
        Cluster using DBSCAN.

        DBSCAN finds clusters of arbitrary shape based on density.
        It doesn't require specifying n_clusters, making it a good
        validation that k-means isn't forcing artificial structure.

        Args:
            eps: Maximum distance between samples. If None, auto-detect.
            min_samples: Minimum samples in a neighborhood.
        """
        self._build_feature_matrix()

        # Auto-detect eps using k-distance graph if not provided
        if eps is None:
            eps = self._estimate_eps(min_samples)
            logger.info(f"Auto-detected eps={eps:.4f}")

        logger.info(f"DBSCAN clustering with eps={eps}, min_samples={min_samples}")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(self._features_scaled)

        # DBSCAN labels: -1 = noise, 0+ = cluster ID
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        # Silhouette only on non-noise points
        if n_clusters >= 2:
            non_noise_mask = labels != -1
            if non_noise_mask.sum() >= n_clusters:
                silhouette = silhouette_score(
                    self._features_scaled[non_noise_mask],
                    labels[non_noise_mask]
                )
            else:
                silhouette = 0.0
        else:
            silhouette = 0.0

        return ClusteringResult(
            algorithm="dbscan",
            n_clusters=n_clusters,
            labels=labels,
            silhouette=silhouette,
            extra_metrics={
                "eps": eps,
                "min_samples": min_samples,
                "n_noise": int(n_noise),
                "noise_ratio": n_noise / len(labels),
            }
        )

    def _estimate_eps(self, min_samples: int) -> float:
        """
        Estimate eps using the k-distance graph elbow method.
        """
        k = min_samples
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self._features_scaled)
        distances, _ = nn.kneighbors(self._features_scaled)

        # k-th nearest neighbor distance
        k_distances = np.sort(distances[:, -1])

        # Find elbow using second derivative
        # Use subset for speed
        subset = k_distances[::max(1, len(k_distances) // 1000)]
        if len(subset) < 10:
            subset = k_distances

        # Compute curvature
        d1 = np.diff(subset)
        d2 = np.diff(d1)

        # Elbow is where second derivative is maximum
        if len(d2) > 0:
            elbow_idx = np.argmax(d2) + 1
            eps = subset[min(elbow_idx, len(subset) - 1)]
        else:
            eps = np.median(k_distances)

        return float(eps)

    def cluster_spectral(
        self,
        n_clusters: int = 50,
        random_state: int = 42,
        n_neighbors: int = 10,
    ) -> ClusteringResult:
        """
        Cluster using spectral clustering.

        Spectral clustering uses graph structure, making it more
        appropriate for connectivity data than k-means.
        """
        self._build_feature_matrix()

        logger.info(f"Spectral clustering with k={n_clusters}")
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            random_state=random_state,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            n_jobs=-1,
        )
        labels = spectral.fit_predict(self._features_scaled)

        silhouette = silhouette_score(self._features_scaled, labels)

        return ClusteringResult(
            algorithm="spectral",
            n_clusters=n_clusters,
            labels=labels,
            silhouette=silhouette,
            extra_metrics={
                "n_neighbors": n_neighbors,
            }
        )

    def compare_algorithms(
        self,
        algorithms: List[str] = ["kmeans", "dbscan", "spectral"],
        n_clusters: int = 50,
        random_state: int = 42,
    ) -> Dict[str, ClusteringResult]:
        """
        Run multiple clustering algorithms and compare results.
        """
        results = {}

        for algo in algorithms:
            if algo == "kmeans":
                results[algo] = self.cluster_kmeans(n_clusters, random_state)
            elif algo == "dbscan":
                results[algo] = self.cluster_dbscan()
            elif algo == "spectral":
                results[algo] = self.cluster_spectral(n_clusters, random_state)
            else:
                logger.warning(f"Unknown algorithm: {algo}")

        return results

    def compute_algorithm_agreement(
        self,
        results: Dict[str, ClusteringResult]
    ) -> Dict[str, float]:
        """
        Compute pairwise agreement between clustering algorithms.

        Uses Adjusted Rand Index (ARI) which is 1.0 for identical
        clusterings and ~0.0 for random.
        """
        algorithms = list(results.keys())
        agreement = {}

        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i + 1:]:
                labels1 = results[algo1].labels
                labels2 = results[algo2].labels

                # Handle DBSCAN noise (-1 labels)
                # Only compare non-noise points
                valid_mask = (labels1 != -1) & (labels2 != -1)
                if valid_mask.sum() < 2:
                    ari = 0.0
                else:
                    ari = adjusted_rand_score(labels1[valid_mask], labels2[valid_mask])

                key = f"{algo1}_vs_{algo2}"
                agreement[key] = ari

        return agreement

    def to_topology_clusters(
        self,
        result: ClusteringResult
    ) -> TopologyClusters:
        """
        Convert ClusteringResult to TopologyClusters for compatibility.
        """
        n_clusters = result.n_clusters
        labels = result.labels

        clusters = TopologyClusters(
            n_clusters=n_clusters,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
        )

        # Group neurons by cluster
        cluster_neurons: Dict[int, List[Tuple[int, int]]] = {}
        for idx, label in enumerate(labels):
            if label == -1:  # DBSCAN noise
                continue
            if label not in cluster_neurons:
                cluster_neurons[label] = []
            layer, neuron = self._neuron_index[idx]
            cluster_neurons[label].append((layer, neuron))
            clusters.neuron_to_cluster[(layer, neuron)] = int(label)

        # Build cluster objects
        for cluster_id, neurons in cluster_neurons.items():
            layer_dist = {}
            for layer, _ in neurons:
                layer_dist[layer] = layer_dist.get(layer, 0) + 1

            cluster = NeuronCluster(
                cluster_id=cluster_id,
                neurons=neurons,
                size=len(neurons),
                layer_distribution=layer_dist,
            )
            clusters.clusters.append(cluster)

        return clusters


def compare_seed_stability(
    conn_map: ConnectivityMap,
    algorithm: str = "kmeans",
    n_clusters: int = 50,
    seeds: List[int] = [42, 123, 456, 789, 1337],
) -> Dict:
    """
    Test cluster stability across different random seeds.

    Returns pairwise ARI between seed runs.
    """
    results = {}

    for seed in seeds:
        builder = AlternativeClusterBuilder(conn_map)
        if algorithm == "kmeans":
            result = builder.cluster_kmeans(n_clusters, random_state=seed)
        elif algorithm == "spectral":
            result = builder.cluster_spectral(n_clusters, random_state=seed)
        else:
            raise ValueError(f"Seed stability not applicable to {algorithm}")

        results[seed] = result

    # Compute pairwise ARI
    stability = {}
    seeds_list = list(seeds)
    for i, seed1 in enumerate(seeds_list):
        for seed2 in seeds_list[i + 1:]:
            ari = adjusted_rand_score(results[seed1].labels, results[seed2].labels)
            stability[f"seed_{seed1}_vs_{seed2}"] = ari

    # Summary stats
    ari_values = list(stability.values())
    stability["mean_ari"] = np.mean(ari_values)
    stability["std_ari"] = np.std(ari_values)
    stability["min_ari"] = np.min(ari_values)

    return stability
