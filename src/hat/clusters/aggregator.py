"""
Connectivity Aggregator

Combines forward fuzzing results with backward trace results to build
a unified connectivity map.

Forward: "if source neuron fires, what downstream neurons respond?"
Backward: "what upstream neurons contribute to target neuron?"

Triangulation: paths that appear in both directions are dominant pathways.
Regions with high forward but scattered backward = dynamic routing zones.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging

from .fuzzer import LayerFuzzResult
from .tracer import LayerTraceResult

logger = logging.getLogger(__name__)


@dataclass
class ConnectivityEdge:
    """A single connectivity edge between neurons."""
    source_layer: int
    source_neuron: int
    target_layer: int
    target_neuron: int
    forward_score: float  # From fuzzing
    backward_score: float  # From tracing
    combined_score: float  # Triangulated


@dataclass
class ConnectivityMap:
    """
    Unified connectivity map combining forward and backward analysis.

    Provides:
    - Per-layer-pair connectivity matrices
    - Dominant pathway identification
    - Dynamic routing zone detection
    """
    n_layers: int
    hidden_dim: int

    # Layer pair connectivity: (source, target) -> [source_dim, target_dim]
    forward_connectivity: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    backward_connectivity: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    combined_connectivity: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)

    # Summary statistics
    layer_connectivity_density: Dict[Tuple[int, int], float] = field(default_factory=dict)
    dynamic_routing_score: Dict[int, float] = field(default_factory=dict)

    def get_connectivity(
        self,
        source_layer: int,
        target_layer: int,
        mode: str = "combined"
    ) -> Optional[np.ndarray]:
        """Get connectivity matrix for a layer pair."""
        key = (source_layer, target_layer)
        if mode == "forward":
            return self.forward_connectivity.get(key)
        elif mode == "backward":
            return self.backward_connectivity.get(key)
        else:
            return self.combined_connectivity.get(key)

    def get_top_connections(
        self,
        source_layer: int,
        source_neuron: int,
        target_layer: int,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Get top-k target neurons connected to a source neuron."""
        conn = self.get_connectivity(source_layer, target_layer)
        if conn is None:
            return []

        scores = conn[source_neuron, :]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def get_upstream_sources(
        self,
        target_layer: int,
        target_neuron: int,
        top_k: int = 10,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Get top-k upstream sources for a target neuron across all source layers."""
        results = {}
        for (src_layer, tgt_layer), conn in self.combined_connectivity.items():
            if tgt_layer == target_layer:
                scores = conn[:, target_neuron]
                top_indices = np.argsort(scores)[-top_k:][::-1]
                results[src_layer] = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results


class ConnectivityAggregator:
    """
    Aggregates forward and backward connectivity into unified map.

    Usage:
        aggregator = ConnectivityAggregator(n_layers, hidden_dim)
        aggregator.add_forward_results(fuzz_results)
        aggregator.add_backward_results(trace_results)
        connectivity_map = aggregator.build()
    """

    def __init__(self, n_layers: int, hidden_dim: int):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Raw results storage
        self._forward_results: Dict[int, LayerFuzzResult] = {}
        self._backward_results: Dict[int, LayerTraceResult] = {}

    def add_forward_results(self, results: Dict[int, LayerFuzzResult]):
        """Add forward fuzzing results."""
        self._forward_results.update(results)
        logger.info(f"Added forward results for {len(results)} source layers")

    def add_backward_results(self, results: Dict[int, LayerTraceResult]):
        """Add backward trace results."""
        self._backward_results.update(results)
        logger.info(f"Added backward results for {len(results)} target layers")

    def build(
        self,
        forward_weight: float = 0.5,
        min_score_threshold: float = 0.01,
    ) -> ConnectivityMap:
        """
        Build unified connectivity map.

        Args:
            forward_weight: Weight for forward scores in combination (0-1)
            min_score_threshold: Minimum score to include in map

        Returns:
            ConnectivityMap with combined analysis
        """
        conn_map = ConnectivityMap(
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
        )

        # Process forward results
        for source_layer, fuzz_result in self._forward_results.items():
            for target_layer, conn_matrix in fuzz_result.connectivity.items():
                key = (source_layer, target_layer)

                # Normalize forward connectivity
                max_val = conn_matrix.max()
                if max_val > 0:
                    normalized = conn_matrix / max_val
                else:
                    normalized = conn_matrix

                conn_map.forward_connectivity[key] = normalized

        # Process backward results
        for target_layer, trace_result in self._backward_results.items():
            for source_layer, conn_matrix in trace_result.mlp_connectivity.items():
                key = (source_layer, target_layer)

                # Normalize backward connectivity
                # Note: backward is [target_dim, source_dim], need to transpose
                conn_t = conn_matrix.T if conn_matrix.shape[0] != self.hidden_dim else conn_matrix

                max_val = conn_t.max()
                if max_val > 0:
                    normalized = conn_t / max_val
                else:
                    normalized = conn_t

                conn_map.backward_connectivity[key] = normalized

        # Combine forward and backward
        backward_weight = 1.0 - forward_weight

        all_keys = set(conn_map.forward_connectivity.keys()) | set(conn_map.backward_connectivity.keys())

        for key in all_keys:
            forward = conn_map.forward_connectivity.get(key)
            backward = conn_map.backward_connectivity.get(key)

            if forward is not None and backward is not None:
                # Both available - combine
                # Ensure shapes match
                if forward.shape == backward.shape:
                    combined = forward_weight * forward + backward_weight * backward
                else:
                    # Shape mismatch - use forward only
                    combined = forward
            elif forward is not None:
                combined = forward
            elif backward is not None:
                combined = backward
            else:
                continue

            # Apply threshold
            combined[combined < min_score_threshold] = 0

            conn_map.combined_connectivity[key] = combined

            # Compute density (fraction of non-zero entries)
            density = (combined > 0).sum() / combined.size
            conn_map.layer_connectivity_density[key] = float(density)

        # Compute dynamic routing scores per layer
        # High forward connectivity but scattered backward = dynamic routing
        for layer_idx in range(1, self.n_layers):
            forward_scores = []
            backward_variance = []

            for (src, tgt), conn in conn_map.forward_connectivity.items():
                if tgt == layer_idx:
                    forward_scores.append(conn.mean())

            for (src, tgt), conn in conn_map.backward_connectivity.items():
                if tgt == layer_idx:
                    # Variance across source neurons
                    backward_variance.append(conn.var())

            if forward_scores and backward_variance:
                # High forward mean + high backward variance = dynamic routing
                score = np.mean(forward_scores) * np.mean(backward_variance)
                conn_map.dynamic_routing_score[layer_idx] = float(score)

        logger.info(
            f"Built connectivity map: {len(conn_map.combined_connectivity)} layer pairs, "
            f"mean density {np.mean(list(conn_map.layer_connectivity_density.values())):.3f}"
        )

        return conn_map

    def identify_dominant_pathways(
        self,
        conn_map: ConnectivityMap,
        top_k: int = 100,
    ) -> List[ConnectivityEdge]:
        """
        Identify the most dominant pathways in the network.

        Returns top-k edges by combined score.
        """
        edges = []

        for (src_layer, tgt_layer), combined in conn_map.combined_connectivity.items():
            forward = conn_map.forward_connectivity.get((src_layer, tgt_layer))
            backward = conn_map.backward_connectivity.get((src_layer, tgt_layer))

            # Find top connections in this layer pair
            flat_indices = np.argsort(combined.ravel())[-top_k:][::-1]

            for flat_idx in flat_indices:
                src_neuron = flat_idx // combined.shape[1]
                tgt_neuron = flat_idx % combined.shape[1]
                combined_score = combined[src_neuron, tgt_neuron]

                if combined_score < 0.01:
                    continue

                forward_score = forward[src_neuron, tgt_neuron] if forward is not None else 0
                backward_score = backward[src_neuron, tgt_neuron] if backward is not None else 0

                edges.append(ConnectivityEdge(
                    source_layer=src_layer,
                    source_neuron=int(src_neuron),
                    target_layer=tgt_layer,
                    target_neuron=int(tgt_neuron),
                    forward_score=float(forward_score),
                    backward_score=float(backward_score),
                    combined_score=float(combined_score),
                ))

        # Sort by combined score and return top k overall
        edges.sort(key=lambda e: e.combined_score, reverse=True)
        return edges[:top_k]

    def identify_dynamic_zones(
        self,
        conn_map: ConnectivityMap,
        threshold: float = 0.5,
    ) -> List[int]:
        """
        Identify layers that are primarily attention-mediated (dynamic routing).

        These are layers where forward connectivity is high but backward
        traces are inconsistent, suggesting attention-dependent routing.
        """
        dynamic_layers = []

        scores = conn_map.dynamic_routing_score
        if not scores:
            return dynamic_layers

        max_score = max(scores.values())
        threshold_score = max_score * threshold

        for layer_idx, score in scores.items():
            if score >= threshold_score:
                dynamic_layers.append(layer_idx)

        return sorted(dynamic_layers)

    def save(self, conn_map: ConnectivityMap, path: str):
        """Save connectivity map to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save connectivity matrices
        for (src, tgt), conn in conn_map.combined_connectivity.items():
            np.save(path / f"combined_L{src}_to_L{tgt}.npy", conn)

        for (src, tgt), conn in conn_map.forward_connectivity.items():
            np.save(path / f"forward_L{src}_to_L{tgt}.npy", conn)

        for (src, tgt), conn in conn_map.backward_connectivity.items():
            np.save(path / f"backward_L{src}_to_L{tgt}.npy", conn)

        # Save metadata
        metadata = {
            "n_layers": conn_map.n_layers,
            "hidden_dim": conn_map.hidden_dim,
            "layer_pairs": [list(k) for k in conn_map.combined_connectivity.keys()],
            "density": {f"{k[0]}_{k[1]}": v for k, v in conn_map.layer_connectivity_density.items()},
            "dynamic_routing_scores": conn_map.dynamic_routing_score,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved connectivity map to {path}")

    @classmethod
    def load(cls, path: str) -> ConnectivityMap:
        """Load connectivity map from disk."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        conn_map = ConnectivityMap(
            n_layers=metadata["n_layers"],
            hidden_dim=metadata["hidden_dim"],
        )

        # Load matrices
        for pair in metadata["layer_pairs"]:
            src, tgt = pair
            key = (src, tgt)

            combined_path = path / f"combined_L{src}_to_L{tgt}.npy"
            if combined_path.exists():
                conn_map.combined_connectivity[key] = np.load(combined_path)

            forward_path = path / f"forward_L{src}_to_L{tgt}.npy"
            if forward_path.exists():
                conn_map.forward_connectivity[key] = np.load(forward_path)

            backward_path = path / f"backward_L{src}_to_L{tgt}.npy"
            if backward_path.exists():
                conn_map.backward_connectivity[key] = np.load(backward_path)

        conn_map.layer_connectivity_density = {
            tuple(map(int, k.split("_"))): v
            for k, v in metadata.get("density", {}).items()
        }
        conn_map.dynamic_routing_score = {
            int(k): v for k, v in metadata.get("dynamic_routing_scores", {}).items()
        }

        return conn_map
