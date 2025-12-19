#!/usr/bin/env python3
"""
Analyze what binary classifiers have learned.

The classifiers are small MLPs:
    Linear(4096 -> 128) -> ReLU -> Dropout -> Linear(128 -> 64) -> ReLU -> Dropout -> Linear(64 -> 1)

The first layer weights [128, 4096] encode which input features matter for each hidden unit.
By analyzing these weights, we can understand:
1. Which features are discriminative for each concept
2. How concepts differ from each other
3. Whether we can extract a "feature importance map" from trained classifiers
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm


def load_classifier_weights(lens_path: Path) -> Dict[str, torch.Tensor]:
    """Load classifier state dict."""
    return torch.load(lens_path, map_location='cpu')


def extract_first_layer_weights(state_dict: Dict) -> np.ndarray:
    """
    Extract first layer weights [128, 4096].
    These encode which input features activate each hidden unit.
    """
    # Keys might be '0.weight' or 'net.0.weight' depending on how saved
    for key in state_dict:
        if 'weight' in key and state_dict[key].shape[1] == 4096:
            return state_dict[key].numpy()
    raise ValueError("Could not find first layer weights")


def compute_feature_importance(first_layer: np.ndarray) -> np.ndarray:
    """
    Compute per-feature importance from first layer weights.

    Several options:
    1. L1 norm across hidden units: sum of absolute weights per feature
    2. L2 norm: sqrt of sum of squared weights
    3. Max: maximum absolute weight per feature

    Returns [4096] importance scores.
    """
    # L2 norm gives good balance - features that strongly activate ANY hidden unit
    return np.linalg.norm(first_layer, axis=0)


def analyze_single_classifier(lens_path: Path) -> Dict:
    """Analyze a single classifier's learned features."""
    state_dict = load_classifier_weights(lens_path)
    first_layer = extract_first_layer_weights(state_dict)

    importance = compute_feature_importance(first_layer)

    # Find top features
    top_indices = np.argsort(importance)[::-1][:100]

    return {
        'top_features': top_indices.tolist(),
        'top_importance': importance[top_indices].tolist(),
        'mean_importance': float(importance.mean()),
        'std_importance': float(importance.std()),
        'max_importance': float(importance.max()),
        'sparsity': float((importance < importance.mean()).sum() / len(importance)),
    }


def compare_classifiers(
    lens_dir: Path,
    concepts: List[str],
) -> Dict:
    """
    Compare feature importance across multiple classifiers.

    Returns analysis of:
    - Shared features (high importance across many concepts)
    - Unique features (high importance for specific concepts)
    - Feature selectivity
    """
    all_importance = {}

    for concept in tqdm(concepts, desc="Loading classifiers"):
        lens_path = lens_dir / f"{concept}_classifier.pt"
        if not lens_path.exists():
            continue

        state_dict = load_classifier_weights(lens_path)
        first_layer = extract_first_layer_weights(state_dict)
        importance = compute_feature_importance(first_layer)
        all_importance[concept] = importance

    if not all_importance:
        return {}

    # Stack into matrix [n_concepts, 4096]
    concepts_list = list(all_importance.keys())
    importance_matrix = np.stack([all_importance[c] for c in concepts_list])

    # Analyze feature patterns
    # 1. How many concepts use each feature?
    threshold = np.percentile(importance_matrix, 90)  # Top 10% threshold
    feature_usage = (importance_matrix > threshold).sum(axis=0)

    # 2. Find universally important features (used by many concepts)
    universal_features = np.where(feature_usage > len(concepts_list) * 0.5)[0]

    # 3. Find selective features (used by few concepts)
    selective_features = np.where((feature_usage > 0) & (feature_usage < len(concepts_list) * 0.1))[0]

    # 4. Compute feature selectivity score: how concentrated is each feature's importance?
    # High selectivity = feature important for few concepts
    # Low selectivity = feature important for many concepts
    feature_selectivity = np.zeros(4096)
    for dim in range(4096):
        col = importance_matrix[:, dim]
        if col.max() > 0:
            # Gini-like coefficient: how unequal is the distribution?
            sorted_col = np.sort(col)
            n = len(sorted_col)
            cumsum = np.cumsum(sorted_col)
            feature_selectivity[dim] = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0

    return {
        'n_concepts': len(concepts_list),
        'n_universal_features': len(universal_features),
        'n_selective_features': len(selective_features),
        'universal_features': universal_features.tolist()[:50],
        'selective_features': selective_features.tolist()[:50],
        'feature_usage_hist': np.histogram(feature_usage, bins=20)[0].tolist(),
        'mean_selectivity': float(feature_selectivity.mean()),
        'concepts': concepts_list,
    }


def build_classifier_based_map(
    lens_dir: Path,
    concepts: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a feature importance matrix from classifiers that could serve as
    an alternative to IDF-weighted centroids.

    Each concept's row is the L2 norm of its first layer weights,
    representing which features that classifier learned to look at.

    Returns:
        (importance_matrix, concept_list) where matrix is [n_concepts, 4096]
    """
    importance_rows = []
    valid_concepts = []

    for concept in tqdm(concepts, desc="Building map"):
        lens_path = lens_dir / f"{concept}_classifier.pt"
        if not lens_path.exists():
            continue

        state_dict = load_classifier_weights(lens_path)
        first_layer = extract_first_layer_weights(state_dict)
        importance = compute_feature_importance(first_layer)

        importance_rows.append(importance)
        valid_concepts.append(concept)

    return np.stack(importance_rows), valid_concepts


def analyze_sibling_discrimination(
    lens_dir: Path,
    siblings: List[str],
) -> Dict:
    """
    Analyze how sibling concepts discriminate from each other.

    What features does each sibling emphasize that others don't?
    """
    # Load all sibling classifiers
    importance = {}
    for sib in siblings:
        lens_path = lens_dir / f"{sib}_classifier.pt"
        if not lens_path.exists():
            continue
        state_dict = load_classifier_weights(lens_path)
        first_layer = extract_first_layer_weights(state_dict)
        importance[sib] = compute_feature_importance(first_layer)

    if len(importance) < 2:
        return {}

    # Stack into matrix
    sibs = list(importance.keys())
    matrix = np.stack([importance[s] for s in sibs])  # [n_sibs, 4096]

    # For each sibling, find features where it differs most from others
    discriminating_features = {}
    for i, sib in enumerate(sibs):
        my_importance = matrix[i]
        others_mean = np.mean(np.delete(matrix, i, axis=0), axis=0)

        # Where am I much higher than others?
        diff = my_importance - others_mean
        top_discriminating = np.argsort(diff)[::-1][:20]

        discriminating_features[sib] = {
            'top_features': top_discriminating.tolist(),
            'importance_diff': diff[top_discriminating].tolist(),
        }

    return {
        'siblings': sibs,
        'discriminating_features': discriminating_features,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze classifier learned features')
    parser.add_argument('--lens-pack', required=True, help='Path to lens pack')
    parser.add_argument('--layer', type=int, default=1, help='Layer to analyze')
    parser.add_argument('--concept', type=str, default=None, help='Single concept to analyze')
    parser.add_argument('--siblings', nargs='+', default=None, help='Sibling concepts to compare')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--build-map', action='store_true', help='Build importance-based map')

    args = parser.parse_args()

    lens_dir = Path(args.lens_pack) / f"layer{args.layer}"

    if args.concept:
        # Analyze single concept
        lens_path = lens_dir / f"{args.concept}_classifier.pt"
        result = analyze_single_classifier(lens_path)
        print(f"\nAnalysis of {args.concept}:")
        print(f"  Top 10 feature indices: {result['top_features'][:10]}")
        print(f"  Top 10 importance values: {[f'{x:.3f}' for x in result['top_importance'][:10]]}")
        print(f"  Mean importance: {result['mean_importance']:.4f}")
        print(f"  Max importance: {result['max_importance']:.4f}")
        print(f"  Sparsity (% below mean): {result['sparsity']:.1%}")

    elif args.siblings:
        # Compare siblings
        result = analyze_sibling_discrimination(lens_dir, args.siblings)
        print(f"\nSibling discrimination analysis:")
        for sib, data in result.get('discriminating_features', {}).items():
            print(f"\n  {sib}:")
            print(f"    Top discriminating features: {data['top_features'][:5]}")
            print(f"    Importance diff: {[f'{x:.3f}' for x in data['importance_diff'][:5]]}")

    else:
        # Full layer analysis
        concepts = [p.stem.replace('_classifier', '') for p in lens_dir.glob('*_classifier.pt')]
        print(f"Found {len(concepts)} classifiers in layer {args.layer}")

        result = compare_classifiers(lens_dir, concepts)

        print(f"\nCross-classifier analysis:")
        print(f"  Concepts analyzed: {result['n_concepts']}")
        print(f"  Universal features (>50% of concepts): {result['n_universal_features']}")
        print(f"  Selective features (<10% of concepts): {result['n_selective_features']}")
        print(f"  Mean feature selectivity: {result['mean_selectivity']:.3f}")
        print(f"\n  Feature usage histogram (how many concepts use each feature):")
        print(f"    {result['feature_usage_hist']}")

        if args.build_map:
            print("\nBuilding classifier-based importance map...")
            matrix, concept_list = build_classifier_based_map(lens_dir, concepts)
            print(f"  Map shape: {matrix.shape}")

            # Save map
            if args.output:
                np.save(args.output, matrix)
                with open(args.output.replace('.npy', '_concepts.json'), 'w') as f:
                    json.dump(concept_list, f)
                print(f"  Saved to {args.output}")

    if args.output and not args.build_map:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved analysis to {args.output}")


if __name__ == '__main__':
    main()
