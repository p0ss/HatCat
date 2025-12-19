#!/usr/bin/env python3
"""
Analyze what MLP lens classifiers have actually learned.

The lens classifier structure:
- Layer 0: Linear(4096 → 128) - extracts 128 features from activation space
- Layer 3: Linear(128 → 64) - combines into 64 higher-order features
- Layer 6: Linear(64 → 1) - final decision

Key questions:
1. What are the 128 feature directions the classifier learned?
2. How important is each feature to the final classification?
3. Can we construct a "steering matrix" from feature importance?
4. Does this correlate with what we'd expect from the concept?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import json

from src.steering.hooks import load_lens_classifier, LensClassifier


def analyze_classifier_features(classifier: LensClassifier) -> Dict:
    """
    Analyze the feature structure of a lens classifier.

    Returns dict with:
    - feature_directions: The 128 feature directions (first layer weights)
    - feature_importance: How much each feature matters for positive classification
    - steering_vector: Importance-weighted sum of feature directions
    - top_features: Indices of most important features
    """
    # Extract weights
    W1 = classifier.net[0].weight.detach().cpu()  # [128, 4096] - feature extractors
    b1 = classifier.net[0].bias.detach().cpu()    # [128]

    W2 = classifier.net[3].weight.detach().cpu()  # [64, 128]
    b2 = classifier.net[3].bias.detach().cpu()    # [64]

    W3 = classifier.net[6].weight.detach().cpu()  # [1, 64]
    b3 = classifier.net[6].bias.detach().cpu()    # [1]

    print(f"Layer 1: {W1.shape} features extracted from input")
    print(f"Layer 2: {W2.shape} combinations of features")
    print(f"Layer 3: {W3.shape} final decision weights")

    # Compute feature importance through the network
    # For a linear path (ignoring ReLU for now), the importance of
    # first-layer feature i to the output is:
    # sum_j(W3[0,j] * W2[j,i])

    # This gives us how much each of the 128 first-layer features
    # contributes to the final output (assuming linear approximation)
    feature_importance_linear = (W3 @ W2).squeeze()  # [128]

    print(f"\nFeature importance (linear approx):")
    print(f"  Shape: {feature_importance_linear.shape}")
    print(f"  Range: [{feature_importance_linear.min():.4f}, {feature_importance_linear.max():.4f}]")
    print(f"  Mean: {feature_importance_linear.mean():.4f}")
    print(f"  Std: {feature_importance_linear.std():.4f}")

    # Top positive features (increase classification score)
    top_pos_idx = torch.argsort(feature_importance_linear, descending=True)[:10]
    top_neg_idx = torch.argsort(feature_importance_linear, descending=False)[:10]

    print(f"\nTop 10 positive features (indices): {top_pos_idx.tolist()}")
    print(f"  Importance values: {feature_importance_linear[top_pos_idx].tolist()}")
    print(f"\nTop 10 negative features (indices): {top_neg_idx.tolist()}")
    print(f"  Importance values: {feature_importance_linear[top_neg_idx].tolist()}")

    # Construct steering vector: importance-weighted sum of feature directions
    # Each row of W1 is a feature direction, weight by importance
    steering_vector = (feature_importance_linear.unsqueeze(1) * W1).sum(dim=0)  # [4096]
    steering_vector_normalized = steering_vector / (steering_vector.norm() + 1e-8)

    print(f"\nSteering vector (importance-weighted):")
    print(f"  Norm before normalization: {steering_vector.norm():.4f}")
    print(f"  Mean: {steering_vector.mean():.6f}")
    print(f"  Std: {steering_vector.std():.6f}")

    # Also compute positive-only steering (only features that increase score)
    positive_mask = feature_importance_linear > 0
    steering_positive = (feature_importance_linear.clamp(min=0).unsqueeze(1) * W1).sum(dim=0)
    steering_positive_normalized = steering_positive / (steering_positive.norm() + 1e-8)

    print(f"\nPositive-only steering vector:")
    print(f"  Using {positive_mask.sum()} of 128 features")
    print(f"  Norm before normalization: {steering_positive.norm():.4f}")

    # Compare to simple sum (what we were doing before)
    simple_sum = W1.sum(dim=0)
    simple_sum_normalized = simple_sum / (simple_sum.norm() + 1e-8)

    # Cosine similarity between approaches
    cos_weighted_vs_simple = torch.cosine_similarity(
        steering_vector_normalized.unsqueeze(0),
        simple_sum_normalized.unsqueeze(0)
    ).item()

    cos_positive_vs_simple = torch.cosine_similarity(
        steering_positive_normalized.unsqueeze(0),
        simple_sum_normalized.unsqueeze(0)
    ).item()

    cos_weighted_vs_positive = torch.cosine_similarity(
        steering_vector_normalized.unsqueeze(0),
        steering_positive_normalized.unsqueeze(0)
    ).item()

    print(f"\nCosine similarities:")
    print(f"  Weighted vs Simple sum: {cos_weighted_vs_simple:.4f}")
    print(f"  Positive-only vs Simple sum: {cos_positive_vs_simple:.4f}")
    print(f"  Weighted vs Positive-only: {cos_weighted_vs_positive:.4f}")

    return {
        "feature_directions": W1.numpy(),
        "feature_importance": feature_importance_linear.numpy(),
        "steering_vector": steering_vector.numpy(),
        "steering_vector_normalized": steering_vector_normalized.numpy(),
        "steering_positive": steering_positive.numpy(),
        "steering_positive_normalized": steering_positive_normalized.numpy(),
        "simple_sum_normalized": simple_sum_normalized.numpy(),
        "top_positive_features": top_pos_idx.tolist(),
        "top_negative_features": top_neg_idx.tolist(),
        "cosine_weighted_vs_simple": cos_weighted_vs_simple,
        "cosine_positive_vs_simple": cos_positive_vs_simple,
    }


def analyze_feature_activation_patterns(
    classifier: LensClassifier,
    test_activations: torch.Tensor,
    labels: torch.Tensor,
) -> Dict:
    """
    Analyze which features activate for positive vs negative examples.

    Args:
        classifier: The lens classifier
        test_activations: [N, 4096] activation samples
        labels: [N] binary labels (1=concept present, 0=absent)
    """
    W1 = classifier.net[0].weight.detach()  # [128, 4096]
    b1 = classifier.net[0].bias.detach()    # [128]

    # Project activations onto feature directions
    # [N, 4096] @ [4096, 128] = [N, 128]
    feature_activations = test_activations @ W1.T + b1

    # Apply ReLU (as the network does)
    feature_activations_relu = torch.relu(feature_activations)

    # Separate by label
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_activations = feature_activations_relu[pos_mask].mean(dim=0)  # [128]
    neg_activations = feature_activations_relu[neg_mask].mean(dim=0)  # [128]

    # Which features discriminate?
    discrimination = pos_activations - neg_activations

    print(f"\nFeature activation analysis:")
    print(f"  Positive samples: {pos_mask.sum()}")
    print(f"  Negative samples: {neg_mask.sum()}")
    print(f"  Mean pos activation: {pos_activations.mean():.4f}")
    print(f"  Mean neg activation: {neg_activations.mean():.4f}")
    print(f"  Max discrimination: {discrimination.max():.4f}")
    print(f"  Min discrimination: {discrimination.min():.4f}")

    # Top discriminating features
    top_disc_idx = torch.argsort(discrimination, descending=True)[:10]
    print(f"\nTop discriminating features: {top_disc_idx.tolist()}")
    print(f"  Discrimination values: {discrimination[top_disc_idx].tolist()}")

    return {
        "pos_activations": pos_activations.numpy(),
        "neg_activations": neg_activations.numpy(),
        "discrimination": discrimination.numpy(),
        "top_discriminating": top_disc_idx.tolist(),
    }


def compute_steering_matrix(classifier: LensClassifier) -> np.ndarray:
    """
    Compute a steering matrix that can be directly multiplied with activations.

    The idea: instead of computing gradients at runtime, pre-compute a matrix
    that represents "how to modify activations to increase classifier score".

    For the MLP: f(x) = W3 @ relu(W2 @ relu(W1 @ x + b1) + b2) + b3

    The linear approximation of the gradient w.r.t x is:
    df/dx ≈ W1.T @ diag(importance) where importance = W3 @ W2

    But this ignores ReLU. A better approximation uses expected ReLU activation.
    """
    W1 = classifier.net[0].weight.detach().cpu()  # [128, 4096]
    W2 = classifier.net[3].weight.detach().cpu()  # [64, 128]
    W3 = classifier.net[6].weight.detach().cpu()  # [1, 64]

    # Linear importance: how each first-layer feature affects output
    importance = (W3 @ W2).squeeze()  # [128]

    # Steering matrix: transpose of W1, scaled by importance
    # This gives us: for each input dimension, how much does increasing it
    # affect the output (through all 128 features)?
    steering_matrix = W1.T @ torch.diag(importance)  # [4096, 128]

    # Sum across features to get single steering vector per input dimension
    steering_vector = steering_matrix.sum(dim=1)  # [4096]

    return steering_vector.numpy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-pack", default="apertus-8b_first-light")
    parser.add_argument("--concept", default="DomesticCat")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    lens_pack_path = Path(f"lens_packs/{args.lens_pack}")

    # Find classifier for concept
    classifier_path = None
    for layer_dir in sorted(lens_pack_path.glob("layer*")):
        candidate = layer_dir / f"{args.concept}_classifier.pt"
        if candidate.exists():
            classifier_path = candidate
            layer = int(layer_dir.name.replace("layer", ""))
            break

    if classifier_path is None:
        print(f"Classifier not found for {args.concept}")
        return

    print(f"Analyzing: {args.concept} (layer {layer})")
    print(f"Path: {classifier_path}")
    print("=" * 60)

    classifier = load_lens_classifier(classifier_path, args.device)

    # Analyze feature structure
    analysis = analyze_classifier_features(classifier)

    # Save results
    output_path = Path(f"results/lens_analysis/{args.concept}_features.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON
    json_safe = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in analysis.items()
    }

    with open(output_path, "w") as f:
        json.dump(json_safe, f, indent=2)

    print(f"\nSaved analysis to {output_path}")

    # Also save the steering vectors as .pt for easy loading
    vectors_path = output_path.with_suffix(".pt")
    torch.save({
        "steering_weighted": torch.tensor(analysis["steering_vector_normalized"]),
        "steering_positive": torch.tensor(analysis["steering_positive_normalized"]),
        "steering_simple": torch.tensor(analysis["simple_sum_normalized"]),
        "feature_importance": torch.tensor(analysis["feature_importance"]),
        "feature_directions": torch.tensor(analysis["feature_directions"]),
    }, vectors_path)
    print(f"Saved vectors to {vectors_path}")


if __name__ == "__main__":
    main()
