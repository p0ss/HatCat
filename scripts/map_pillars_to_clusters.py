#!/usr/bin/env python3
"""
Map pillar lens activations to topology clusters.

For each pillar:
1. Generate prompts that should activate that pillar
2. Extract activations from the model
3. See which topology clusters have high activation
4. Report correlations between pillars and clusters
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_clusters(cluster_dir: Path) -> Tuple[List[Dict], Dict[Tuple[int, int], int]]:
    """Load topology clusters and neuron-to-cluster mapping."""
    with open(cluster_dir / "clusters.json") as f:
        clusters = json.load(f)

    with open(cluster_dir / "neuron_to_cluster.json") as f:
        neuron_to_cluster_raw = json.load(f)

    # Convert string keys back to tuples (format is "layer_neuron")
    neuron_to_cluster = {}
    for key, cluster_id in neuron_to_cluster_raw.items():
        parts = key.split("_")
        layer, neuron = int(parts[0]), int(parts[1])
        neuron_to_cluster[(layer, neuron)] = cluster_id

    return clusters, neuron_to_cluster


def load_pillar_lenses(lens_dir: Path) -> Dict[str, Dict]:
    """Load trained pillar lenses and their metadata."""
    results_path = lens_dir / "layer0" / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    lenses = {}
    for result in results["results"]:
        concept = result["concept"]
        lens_path = lens_dir / "layer0" / f"{concept}.pt"
        if lens_path.exists():
            lens_data = torch.load(lens_path, map_location="cpu")
            lenses[concept] = {
                "weights": lens_data.get("weights", lens_data.get("classifier_weights")),
                "bias": lens_data.get("bias", lens_data.get("classifier_bias")),
                "selected_layers": result.get("selected_layers", [15]),
                "f1": result.get("test_f1", 0),
            }
    return lenses


def load_pillar_examples(pack_dir: Path) -> Dict[str, List[str]]:
    """Load positive examples for each pillar."""
    examples = {}
    for pillar_file in pack_dir.glob("*.json"):
        if pillar_file.name in ["pack.json"]:
            continue
        with open(pillar_file) as f:
            pillar = json.load(f)
        if "training_examples" in pillar:
            examples[pillar["id"]] = pillar["training_examples"]["positive"][:10]
    return examples


def load_all_concept_examples(pack_dir: Path) -> Dict[str, List[str]]:
    """Load positive examples for all concepts (L1 + L2) from hierarchy."""
    examples = {}
    hierarchy_dir = pack_dir / "hierarchy"

    for layer_file in hierarchy_dir.glob("layer*.json"):
        with open(layer_file) as f:
            layer_data = json.load(f)

        for concept in layer_data.get("concepts", []):
            term = concept.get("sumo_term", "")
            pos_examples = concept.get("positive_examples", [])
            if pos_examples and term:
                examples[term] = pos_examples[:15]  # Use up to 15 examples

    return examples


def extract_activations_multi_layer(
    model, tokenizer, prompts: List[str], layers: List[int], device: str = "cuda"
) -> Dict[int, torch.Tensor]:
    """Extract activations from multiple layers for given prompts."""
    activations = {layer: [] for layer in layers}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        for layer in layers:
            if layer < len(hidden_states):
                # Get last token activation (convert to float32 for numpy)
                act = hidden_states[layer][0, -1, :].float().cpu()
                activations[layer].append(act)

    # Stack into tensors
    return {layer: torch.stack(acts) for layer, acts in activations.items()}


def compute_cluster_activations(
    activations: Dict[int, torch.Tensor],
    clusters: List[Dict],
    neuron_to_cluster: Dict[Tuple[int, int], int]
) -> np.ndarray:
    """Compute average activation per cluster."""
    n_clusters = len(clusters)
    n_prompts = next(iter(activations.values())).shape[0]

    cluster_acts = np.zeros((n_prompts, n_clusters))
    cluster_counts = np.zeros(n_clusters)

    for (layer, neuron), cluster_id in neuron_to_cluster.items():
        if layer in activations:
            acts = activations[layer]
            if neuron < acts.shape[1]:
                cluster_acts[:, cluster_id] += acts[:, neuron].numpy()
                cluster_counts[cluster_id] += 1

    # Normalize by neuron count per cluster
    cluster_counts[cluster_counts == 0] = 1  # Avoid division by zero
    cluster_acts = cluster_acts / cluster_counts

    return cluster_acts


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Map pillars to topology clusters")
    parser.add_argument("--cluster-dir", type=str,
                        default="results/topology/20260115_221611/clusters",
                        help="Directory with cluster data")
    parser.add_argument("--lens-dir", type=str,
                        default="lens_packs/gemma3-4b_action-agency",
                        help="Directory with trained pillar lenses")
    parser.add_argument("--pack-dir", type=str,
                        default="concept_packs/action-agency-pillars",
                        help="Directory with pillar examples")
    parser.add_argument("--model", type=str,
                        default="google/gemma-3-4b-it",
                        help="Model to use")
    parser.add_argument("--output", type=str,
                        default="results/pillar_cluster_mapping.json",
                        help="Output file")

    args = parser.parse_args()

    print("Loading clusters...")
    clusters, neuron_to_cluster = load_clusters(Path(args.cluster_dir))
    print(f"  Loaded {len(clusters)} clusters with {len(neuron_to_cluster)} neurons")

    print("\nLoading pillar lenses...")
    lenses = load_pillar_lenses(Path(args.lens_dir))
    print(f"  Loaded {len(lenses)} pillar lenses")

    print("\nLoading concept examples (L1 + L2)...")
    examples = load_all_concept_examples(Path(args.pack_dir))
    print(f"  Loaded examples for {len(examples)} concepts")

    # Get all layers used by clusters
    cluster_layers = set()
    for layer, _ in neuron_to_cluster.keys():
        cluster_layers.add(layer)
    cluster_layers = sorted(cluster_layers)
    print(f"\nCluster layers: {cluster_layers}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Compute cluster activations for each concept
    pillar_cluster_activations = {}
    processed = 0

    for concept_name, prompts in examples.items():
        if not prompts:
            continue

        print(f"\n[{processed+1}/{len(examples)}] Processing {concept_name}...")
        processed += 1

        # Extract activations from cluster layers
        activations = extract_activations_multi_layer(
            model, tokenizer, prompts, cluster_layers, "cuda"
        )

        # Compute cluster activations
        cluster_acts = compute_cluster_activations(activations, clusters, neuron_to_cluster)

        # Average across prompts
        mean_cluster_acts = cluster_acts.mean(axis=0)

        # Find top clusters for this concept
        top_clusters = np.argsort(mean_cluster_acts)[::-1][:10]

        pillar_cluster_activations[concept_name] = {
            "concept_name": concept_name,
            "top_clusters": top_clusters.tolist(),
            "top_cluster_activations": mean_cluster_acts[top_clusters].tolist(),
            "all_cluster_activations": mean_cluster_acts.tolist(),
        }

        print(f"  Top clusters: {top_clusters[:5].tolist()}")
        print(f"  Activations: {mean_cluster_acts[top_clusters[:5]].round(3).tolist()}")

    # Analyze cluster-pillar associations
    print("\n" + "="*60)
    print("CLUSTER-PILLAR ASSOCIATIONS")
    print("="*60)

    # For each cluster, find which pillars activate it most
    cluster_pillar_scores = {i: {} for i in range(len(clusters))}
    for pillar_id, data in pillar_cluster_activations.items():
        for cluster_id, act in enumerate(data["all_cluster_activations"]):
            cluster_pillar_scores[cluster_id][pillar_id] = act

    # Find clusters with strong pillar preferences
    selective_clusters = []
    for cluster_id, pillar_scores in cluster_pillar_scores.items():
        if not pillar_scores:
            continue
        scores = list(pillar_scores.values())
        max_score = max(scores)
        mean_score = np.mean(scores)

        if max_score > 0 and mean_score > 0:
            selectivity = max_score / mean_score
            best_pillar = max(pillar_scores.items(), key=lambda x: x[1])[0]

            if selectivity > 1.5:  # Cluster is 1.5x more active for one pillar
                selective_clusters.append({
                    "cluster_id": cluster_id,
                    "best_pillar": best_pillar,
                    "selectivity": selectivity,
                    "cluster_size": clusters[cluster_id]["size"],
                    "layer_distribution": clusters[cluster_id]["layer_distribution"],
                })

    # Sort by selectivity
    selective_clusters.sort(key=lambda x: x["selectivity"], reverse=True)

    print(f"\nFound {len(selective_clusters)} selective clusters (selectivity > 1.5):")
    for sc in selective_clusters[:20]:
        print(f"  Cluster {sc['cluster_id']}: {sc['best_pillar']} "
              f"(selectivity={sc['selectivity']:.2f}, size={sc['cluster_size']})")

    # Save results
    results = {
        "pillar_cluster_activations": pillar_cluster_activations,
        "selective_clusters": selective_clusters,
        "n_clusters": len(clusters),
        "n_pillars": len(pillar_cluster_activations),
        "cluster_layers": cluster_layers,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
