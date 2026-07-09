#!/usr/bin/env python3
"""
Map Topology Clusters to Concepts

Cross-references topology cluster assignments with trained lense pack classifiers
to discover semantic meaning of structural clusters.

Usage:
    python scripts/map_clusters_to_concepts.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using numpy-only fallback.")


@dataclass
class ConceptProfile:
    """Profile of concept activation across clusters."""
    concept: str
    layer: int
    discriminative_neurons: List[int]  # Top neurons by weight magnitude
    cluster_activation: Dict[int, float]  # cluster_id -> activation strength


def load_topology_clusters(cluster_path: str) -> Tuple[Dict[Tuple[int, int], int], Dict]:
    """Load topology cluster assignments."""
    cluster_dir = Path(cluster_path)

    # Load neuron-to-cluster mapping
    with open(cluster_dir / "neuron_to_cluster.json") as f:
        mapping_raw = json.load(f)

    neuron_to_cluster = {}
    for key, cluster_id in mapping_raw.items():
        l, n = map(int, key.split("_"))
        neuron_to_cluster[(l, n)] = cluster_id

    # Load cluster info
    with open(cluster_dir / "clusters.json") as f:
        cluster_info = json.load(f)

    # Load metadata
    with open(cluster_dir / "metadata.json") as f:
        metadata = json.load(f)

    return neuron_to_cluster, {
        "n_clusters": metadata["n_clusters"],
        "n_layers": metadata["n_layers"],
        "hidden_dim": metadata["hidden_dim"],
        "clusters": cluster_info,
    }


def find_discriminative_neurons_from_weights(weights: np.ndarray, top_k: int = 100) -> List[int]:
    """Find neurons with highest absolute weight (most discriminative)."""
    abs_weights = np.abs(weights).flatten()
    top_indices = np.argsort(abs_weights)[-top_k:][::-1]
    return top_indices.tolist()


def load_classifier_weights(classifier_path: str) -> Optional[np.ndarray]:
    """Load classifier weights from .pt file."""
    if not TORCH_AVAILABLE:
        return None

    try:
        state_dict = torch.load(classifier_path, map_location='cpu', weights_only=False)

        # Handle different classifier structures
        if isinstance(state_dict, dict):
            # Look for weight keys
            for key in ['weight', 'linear.weight', 'classifier.weight', 'fc.weight']:
                if key in state_dict:
                    return state_dict[key].numpy()

            # Try model_state_dict
            if 'model_state_dict' in state_dict:
                model_dict = state_dict['model_state_dict']
                for key in ['weight', 'linear.weight', '0.weight']:
                    if key in model_dict:
                        return model_dict[key].numpy()

            # Just grab first weight-like tensor
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor) and 'weight' in key.lower():
                    return value.numpy()

        elif hasattr(state_dict, 'weight'):
            return state_dict.weight.numpy()

        return None
    except Exception as e:
        print(f"Error loading {classifier_path}: {e}")
        return None


def scan_classifiers(results_dir: str, model_layer: int = 15) -> Dict[str, str]:
    """
    Scan results directory for trained classifiers.

    NOTE: The directory names (layer0/, layer1/) refer to SUMO ontology hierarchy
    depth, NOT model layers. All classifiers are trained on activations from a
    single model layer (default: 15).

    Args:
        results_dir: Path to results directory
        model_layer: The model layer these classifiers were trained on (default: 15)

    Returns:
        Dict mapping concept name -> classifier_path
    """
    results_path = Path(results_dir)
    classifiers = {}

    # Look for classifier .pt files
    for pt_file in results_path.rglob("*_classifier.pt"):
        # Parse concept from filename
        concept = pt_file.stem.replace("_classifier", "")

        # Skip duplicates (prefer more recent paths)
        if concept not in classifiers:
            classifiers[concept] = str(pt_file)

    return classifiers


def compute_cluster_activation(
    neuron_to_cluster: Dict[Tuple[int, int], int],
    layer: int,
    discriminative_neurons: List[int],
    weights: np.ndarray,
    n_clusters: int,
) -> Dict[int, float]:
    """
    Compute activation strength per cluster based on discriminative neurons.

    Uses signed weight sum to capture both positive and negative contributions.
    """
    cluster_activation = defaultdict(float)
    cluster_counts = defaultdict(int)

    weights_flat = weights.flatten()

    for neuron_idx in discriminative_neurons:
        key = (layer, neuron_idx)
        if key in neuron_to_cluster:
            cluster_id = neuron_to_cluster[key]
            # Use absolute weight as activation strength
            cluster_activation[cluster_id] += abs(weights_flat[neuron_idx])
            cluster_counts[cluster_id] += 1

    # Normalize by count to get average activation per cluster
    for cluster_id in cluster_activation:
        if cluster_counts[cluster_id] > 0:
            cluster_activation[cluster_id] /= cluster_counts[cluster_id]

    return dict(cluster_activation)


def map_concepts_to_clusters(
    neuron_to_cluster: Dict[Tuple[int, int], int],
    cluster_metadata: Dict,
    classifiers: Dict[str, str],
    model_layer: int = 15,
    top_k_neurons: int = 100,
) -> List[ConceptProfile]:
    """
    Create concept profiles by mapping classifier weights to clusters.

    NOTE: All classifiers are assumed to be trained on the same model layer
    (default: 15, as per extract_activations default).

    Args:
        neuron_to_cluster: Mapping from (layer, neuron) to cluster ID
        cluster_metadata: Cluster metadata
        classifiers: Dict mapping concept name -> classifier path
        model_layer: The model layer these classifiers were trained on
        top_k_neurons: Number of top neurons to consider per classifier
    """
    profiles = []
    n_clusters = cluster_metadata["n_clusters"]

    for concept, classifier_path in classifiers.items():
        weights = load_classifier_weights(classifier_path)
        if weights is None:
            continue

        # Find discriminative neurons
        top_neurons = find_discriminative_neurons_from_weights(weights, top_k_neurons)

        # Compute cluster activation - all classifiers use the same model layer
        cluster_activation = compute_cluster_activation(
            neuron_to_cluster, model_layer, top_neurons, weights, n_clusters
        )

        profile = ConceptProfile(
            concept=concept,
            layer=model_layer,  # All classifiers use the same model layer
            discriminative_neurons=top_neurons[:20],  # Store top 20
            cluster_activation=cluster_activation,
        )
        profiles.append(profile)

    return profiles


def aggregate_cluster_semantics(
    profiles: List[ConceptProfile],
    cluster_metadata: Dict,
) -> Dict[int, Dict]:
    """
    Aggregate semantic meaning for each cluster.

    Returns:
        Dict mapping cluster_id -> {
            "top_concepts": [(concept, layer, strength), ...],
            "concept_count": int,
            "total_activation": float,
        }
    """
    cluster_concepts = defaultdict(list)

    for profile in profiles:
        for cluster_id, strength in profile.cluster_activation.items():
            cluster_concepts[cluster_id].append({
                "concept": profile.concept,
                "layer": profile.layer,
                "strength": strength,
            })

    # Sort and summarize
    summaries = {}
    for cluster_id in range(cluster_metadata["n_clusters"]):
        concepts = cluster_concepts.get(cluster_id, [])

        # Sort by strength
        concepts.sort(key=lambda x: x["strength"], reverse=True)

        summaries[cluster_id] = {
            "top_concepts": [
                (c["concept"], c["layer"], c["strength"])
                for c in concepts[:10]
            ],
            "concept_count": len(concepts),
            "total_activation": sum(c["strength"] for c in concepts),
            "unique_concepts": len(set(c["concept"] for c in concepts)),
        }

    return summaries


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Map topology clusters to concepts")
    parser.add_argument(
        "--clusters",
        type=str,
        default="results/topology/20260115_221611/clusters",
        help="Path to topology clusters directory",
    )
    parser.add_argument(
        "--classifiers",
        type=str,
        default="results/overnight_training_20251114_001140",
        help="Path to trained classifiers directory",
    )
    parser.add_argument(
        "--model-layer",
        type=int,
        default=15,
        help="Model layer the classifiers were trained on (default: 15)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/topology/20260115_221611/cluster_semantics.json",
        help="Output path for cluster semantics",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top neurons to consider per classifier",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MAPPING TOPOLOGY CLUSTERS TO CONCEPTS")
    print("=" * 60)

    # Load topology clusters
    print(f"\nLoading topology clusters from {args.clusters}...")
    neuron_to_cluster, cluster_metadata = load_topology_clusters(args.clusters)
    print(f"  {len(neuron_to_cluster)} neurons mapped to {cluster_metadata['n_clusters']} clusters")

    # Scan for classifiers
    print(f"\nScanning for classifiers in {args.classifiers}...")
    classifiers = scan_classifiers(args.classifiers, model_layer=args.model_layer)
    print(f"  Found {len(classifiers)} unique concepts")
    print(f"  Assuming all trained on model layer {args.model_layer}")

    if len(classifiers) == 0:
        print("No classifiers found! Check the path.")
        return

    # Show sample concepts
    print(f"\n  Sample concepts: {list(classifiers.keys())[:10]}")

    # Map concepts to clusters
    print(f"\nMapping concepts to clusters (model_layer={args.model_layer}, top_k={args.top_k})...")
    profiles = map_concepts_to_clusters(
        neuron_to_cluster, cluster_metadata, classifiers,
        model_layer=args.model_layer, top_k_neurons=args.top_k
    )
    print(f"  Created {len(profiles)} concept profiles")

    # Aggregate semantics
    print("\nAggregating cluster semantics...")
    cluster_summaries = aggregate_cluster_semantics(profiles, cluster_metadata)

    # Print interesting findings
    print("\n" + "=" * 60)
    print("CLUSTER SEMANTIC SUMMARIES")
    print("=" * 60)

    # Sort clusters by total activation
    sorted_clusters = sorted(
        cluster_summaries.items(),
        key=lambda x: x[1]["total_activation"],
        reverse=True
    )

    # Print top 10 most semantically rich clusters
    print("\nTop 10 clusters by semantic richness:")
    for cluster_id, summary in sorted_clusters[:10]:
        print(f"\n  Cluster {cluster_id}:")
        print(f"    Unique concepts: {summary['unique_concepts']}")
        print(f"    Total activation: {summary['total_activation']:.4f}")
        print(f"    Top concepts:")
        for concept, layer, strength in summary["top_concepts"][:5]:
            print(f"      - {concept} (L{layer}): {strength:.4f}")

    # Find clusters with specific semantic themes
    print("\n" + "=" * 60)
    print("SEMANTIC THEMES")
    print("=" * 60)

    # Group concepts by apparent category
    themes = {
        "physical": ["Physical", "Object", "Artifact", "Device"],
        "abstract": ["Abstract", "Proposition", "Attribute", "Relation"],
        "social": ["Person", "Group", "Organization", "Social"],
        "process": ["Process", "Motion", "Change", "Action"],
    }

    for theme_name, theme_concepts in themes.items():
        theme_clusters = defaultdict(float)
        for profile in profiles:
            if any(tc.lower() in profile.concept.lower() for tc in theme_concepts):
                for cluster_id, strength in profile.cluster_activation.items():
                    theme_clusters[cluster_id] += strength

        if theme_clusters:
            top_cluster = max(theme_clusters.items(), key=lambda x: x[1])
            print(f"\n  {theme_name.upper()} theme:")
            print(f"    Strongest cluster: {top_cluster[0]} (strength: {top_cluster[1]:.4f})")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    def to_json_safe(val):
        """Convert numpy types to Python native types."""
        if isinstance(val, (np.floating, np.float32, np.float64)):
            return float(val)
        if isinstance(val, (np.integer, np.int32, np.int64)):
            return int(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    output_data = {
        "cluster_summaries": {
            str(k): {
                "top_concepts": [
                    (c, l, to_json_safe(s)) for c, l, s in v["top_concepts"]
                ],
                "concept_count": v["concept_count"],
                "total_activation": to_json_safe(v["total_activation"]),
                "unique_concepts": v["unique_concepts"],
            }
            for k, v in cluster_summaries.items()
        },
        "profiles": [
            {
                "concept": p.concept,
                "layer": p.layer,
                "discriminative_neurons": [int(n) for n in p.discriminative_neurons],
                "cluster_activation": {str(k): to_json_safe(v) for k, v in p.cluster_activation.items()},
            }
            for p in profiles
        ],
        "metadata": {
            "n_clusters": cluster_metadata["n_clusters"],
            "n_profiles": len(profiles),
            "n_concepts": len(classifiers),
            "top_k_neurons": args.top_k,
        }
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResults saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
