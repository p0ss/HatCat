#!/usr/bin/env python3
"""
Find optimal layer for simplex pole separation.

Scan all layers to find where negative/neutral/positive poles
are most distinguishable in activation space.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from map.training.sumo_data_generation import create_simplex_pole_training_dataset_contrastive

# Load simplex definitions
import json
S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"


def get_layers(model):
    """Get model layers."""
    if hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    elif hasattr(model.model, 'layers'):
        return model.model.layers
    raise AttributeError("Cannot find layers")


def extract_all_layers(
    model, tokenizer, prompts: List[str]
) -> Dict[int, torch.Tensor]:
    """Extract activations from all layers."""
    device = next(model.parameters()).device
    n_layers = len(get_layers(model))

    # Storage for each layer
    layer_acts = {i: [] for i in range(n_layers + 1)}  # +1 for embedding layer

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states[0] = embeddings, [1] = layer 0 output, etc.
            for i, hidden in enumerate(outputs.hidden_states):
                act = hidden[0, -1, :].cpu().float()
                layer_acts[i].append(act)

    # Stack into tensors
    return {i: torch.stack(acts) for i, acts in layer_acts.items()}


def generate_pole_prompts(simplex_name: str, simplex_def: dict, pole_name: str, n: int = 30) -> List[str]:
    """Generate prompts for a pole."""
    pole_data = simplex_def[pole_name]
    pole_type = pole_name.split('_')[0]

    other_pole_names = [p for p in ['negative_pole', 'neutral_homeostasis', 'positive_pole'] if p != pole_name]
    other_poles_data = [{**simplex_def[p], 'pole_type': p.split('_')[0]} for p in other_pole_names]

    prompts, labels = create_simplex_pole_training_dataset_contrastive(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=simplex_name,
        other_poles_data=other_poles_data,
        behavioral_ratio=0.6,
        prompts_per_synset=3
    )

    # Take only positive samples
    positive_prompts = [p for p, l in zip(prompts, labels) if l == 1]
    return positive_prompts[:n]


def compute_separation_metrics(
    neg_acts: torch.Tensor,
    neu_acts: torch.Tensor,
    pos_acts: torch.Tensor
) -> Dict[str, float]:
    """Compute pole separation metrics."""
    # Centroids
    neg_c = neg_acts.mean(dim=0)
    neu_c = neu_acts.mean(dim=0)
    pos_c = pos_acts.mean(dim=0)

    # Inter-pole cosine similarities (lower = better separation)
    neg_pos_sim = cosine_similarity(neg_c.unsqueeze(0).numpy(), pos_c.unsqueeze(0).numpy())[0,0]
    neg_neu_sim = cosine_similarity(neg_c.unsqueeze(0).numpy(), neu_c.unsqueeze(0).numpy())[0,0]
    neu_pos_sim = cosine_similarity(neu_c.unsqueeze(0).numpy(), pos_c.unsqueeze(0).numpy())[0,0]

    avg_inter_sim = (neg_pos_sim + neg_neu_sim + neu_pos_sim) / 3

    # Within-pole variance (lower = tighter clusters)
    neg_var = neg_acts.var(dim=0).mean().item()
    neu_var = neu_acts.var(dim=0).mean().item()
    pos_var = pos_acts.var(dim=0).mean().item()
    avg_within_var = (neg_var + neu_var + pos_var) / 3

    # Between-pole variance (higher = better separation)
    centroids = torch.stack([neg_c, neu_c, pos_c])
    between_var = centroids.var(dim=0).mean().item()

    # Separation ratio: between / within (higher = better)
    separation_ratio = between_var / (avg_within_var + 1e-8)

    # Euclidean distances between centroids
    neg_pos_dist = (neg_c - pos_c).norm().item()
    neg_neu_dist = (neg_c - neu_c).norm().item()
    neu_pos_dist = (neu_c - pos_c).norm().item()
    avg_dist = (neg_pos_dist + neg_neu_dist + neu_pos_dist) / 3

    return {
        'avg_inter_sim': avg_inter_sim,
        'neg_pos_sim': neg_pos_sim,
        'separation_ratio': separation_ratio,
        'avg_distance': avg_dist,
        'neg_pos_distance': neg_pos_dist,
        'within_var': avg_within_var,
        'between_var': between_var,
    }


def main():
    print("=" * 70)
    print("LAYER SCAN FOR OPTIMAL SIMPLEX POLE SEPARATION")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model_name = "google/gemma-3-4b-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    n_layers = len(get_layers(model))
    print(f"Model has {n_layers} layers")

    # Load simplex definitions
    with open(S_TIER_DEFS_PATH) as f:
        simplexes = json.load(f)['simplexes']

    # Test on a few representative simplexes
    test_simplexes = ['threat_perception', 'social_orientation', 'motivational_regulation']

    all_layer_results = {i: [] for i in range(n_layers + 1)}

    for simplex_name in test_simplexes:
        if simplex_name not in simplexes:
            continue

        simplex_def = simplexes[simplex_name]
        print(f"\n{'='*70}")
        print(f"SIMPLEX: {simplex_name}")
        print(f"{'='*70}")

        # Generate prompts for each pole
        print("Generating prompts...")
        neg_prompts = generate_pole_prompts(simplex_name, simplex_def, 'negative_pole', 25)
        neu_prompts = generate_pole_prompts(simplex_name, simplex_def, 'neutral_homeostasis', 25)
        pos_prompts = generate_pole_prompts(simplex_name, simplex_def, 'positive_pole', 25)

        print(f"  negative: {len(neg_prompts)} prompts")
        print(f"  neutral: {len(neu_prompts)} prompts")
        print(f"  positive: {len(pos_prompts)} prompts")

        # Extract all layers
        print("Extracting activations from all layers...")
        neg_all = extract_all_layers(model, tokenizer, neg_prompts)
        neu_all = extract_all_layers(model, tokenizer, neu_prompts)
        pos_all = extract_all_layers(model, tokenizer, pos_prompts)

        # Compute metrics per layer
        print("\nLayer analysis:")
        print(f"{'Layer':>5} | {'Sim':>7} | {'Sep Ratio':>9} | {'Distance':>8} | Notes")
        print("-" * 60)

        for layer_idx in range(n_layers + 1):
            metrics = compute_separation_metrics(
                neg_all[layer_idx],
                neu_all[layer_idx],
                pos_all[layer_idx]
            )

            all_layer_results[layer_idx].append(metrics)

            # Identify notable layers
            notes = ""
            if metrics['separation_ratio'] > 0.5:
                notes = "** HIGH SEP"
            elif metrics['avg_inter_sim'] < 0.999:
                notes = "* lower sim"

            print(f"{layer_idx:>5} | {metrics['avg_inter_sim']:.5f} | {metrics['separation_ratio']:.5f} | {metrics['avg_distance']:>8.1f} | {notes}")

    # Summary across all simplexes
    print("\n" + "=" * 70)
    print("SUMMARY: Average metrics across all tested simplexes")
    print("=" * 70)

    print(f"\n{'Layer':>5} | {'Avg Sim':>8} | {'Sep Ratio':>9} | {'Avg Dist':>8} | Recommendation")
    print("-" * 65)

    best_layer = 0
    best_separation = 0

    for layer_idx in range(n_layers + 1):
        if not all_layer_results[layer_idx]:
            continue

        avg_sim = np.mean([m['avg_inter_sim'] for m in all_layer_results[layer_idx]])
        avg_sep = np.mean([m['separation_ratio'] for m in all_layer_results[layer_idx]])
        avg_dist = np.mean([m['avg_distance'] for m in all_layer_results[layer_idx]])

        rec = ""
        if avg_sep > best_separation:
            best_separation = avg_sep
            best_layer = layer_idx
            rec = "<-- BEST"

        print(f"{layer_idx:>5} | {avg_sim:.6f} | {avg_sep:.6f} | {avg_dist:>8.1f} | {rec}")

    print(f"\n✓ Best layer for pole separation: Layer {best_layer}")
    print(f"  Separation ratio: {best_separation:.4f}")

    # Compare to layer 12
    layer12_sep = np.mean([m['separation_ratio'] for m in all_layer_results[12]])
    print(f"\n  Layer 12 (current): {layer12_sep:.4f}")
    print(f"  Improvement: {(best_separation / layer12_sep - 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
