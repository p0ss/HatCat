#!/usr/bin/env python3
"""
Test which layers activate most strongly for trained all-layers classifiers.

Instead of analyzing weights, we:
1. Load trained classifier
2. Run concept-relevant prompt through model
3. For each layer, mask out other layers and measure classifier response
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.training.sumo_classifiers import extract_activations, get_hidden_dim


def test_layer_importance(classifier, activations, n_layers, hidden_dim):
    """
    Test which layer's activations most strongly trigger the classifier.

    For each layer, zero out all other layers and measure classifier output.
    """
    classifier.eval()
    layer_scores = []

    with torch.no_grad():
        for layer_idx in range(n_layers):
            # Create masked version with only this layer
            masked = np.zeros_like(activations)
            start = layer_idx * hidden_dim
            end = start + hidden_dim
            masked[:, start:end] = activations[:, start:end]

            # Run through classifier
            x = torch.tensor(masked, dtype=torch.float32)
            if next(classifier.parameters()).is_cuda:
                x = x.cuda()

            output = classifier(x)
            # Output is already sigmoid'd, single value per sample
            score = output.mean().item()
            layer_scores.append(score)

    return np.array(layer_scores)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-dir", default="results/all_layers_test")
    args = parser.parse_args()

    classifier_dir = Path(args.classifier_dir)

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=True,
    )
    model.eval()

    hidden_dim = get_hidden_dim(model)
    config = model.config
    if hasattr(config, 'num_hidden_layers'):
        n_layers = config.num_hidden_layers
    elif hasattr(config, 'text_config'):
        n_layers = config.text_config.num_hidden_layers
    else:
        raise ValueError("Cannot find num_hidden_layers")

    print(f"Model: {n_layers} layers, hidden_dim={hidden_dim}")

    # Test prompts for each concept - positive AND negative
    test_prompts = {
        "DomesticCat": {
            "positive": [
                "The cat purred softly on the couch.",
                "My cat loves to chase mice.",
                "Cats are wonderful pets.",
            ],
            "negative": [
                "The algorithm runs in O(n log n) time.",
                "Mathematics is the study of numbers.",
                "The stock market closed higher today.",
            ],
        },
        "DomesticDog": {
            "positive": [
                "The dog wagged its tail happily.",
                "My dog loves to play fetch.",
                "Dogs are loyal companions.",
            ],
            "negative": [
                "The computer program crashed unexpectedly.",
                "Quantum physics describes subatomic particles.",
                "The recipe calls for two cups of flour.",
            ],
        },
        "Algorithm": {
            "positive": [
                "The sorting algorithm runs in O(n log n) time.",
                "Implement a binary search algorithm.",
                "This algorithm solves the problem efficiently.",
            ],
            "negative": [
                "The sunset painted the sky orange.",
                "She walked through the garden slowly.",
                "The coffee was too hot to drink.",
            ],
        },
        "Deception": {
            "positive": [
                "He lied about where he was last night.",
                "The con artist deceived everyone.",
                "She was hiding the truth from her family.",
            ],
            "negative": [
                "The scientist published her findings.",
                "Water boils at 100 degrees Celsius.",
                "The train arrived on time.",
            ],
        },
    }

    # Build classifier architecture (must match training)
    input_dim = n_layers * hidden_dim

    for concept_name, prompt_sets in test_prompts.items():
        classifier_path = classifier_dir / f"{concept_name}_classifier.pt"
        if not classifier_path.exists():
            print(f"\nSkipping {concept_name} - no classifier found")
            continue

        print(f"\n{'='*60}")
        print(f"Testing: {concept_name}")
        print("="*60)

        # Build and load classifier (must match train_simple_classifier architecture)
        # Training used hidden_dim=128, so hidden//2 = 64
        classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )
        classifier.load_state_dict(torch.load(classifier_path, map_location=args.device))
        classifier = classifier.to(args.device)
        classifier.eval()

        for prompt_type in ["positive", "negative"]:
            prompts = prompt_sets[prompt_type]
            print(f"\n  --- {prompt_type.upper()} prompts ---")

            # Extract activations for test prompts (all-layers mode)
            print(f"  Extracting activations for {len(prompts)} prompts...")
            activations = extract_activations(model, tokenizer, prompts, args.device, layer_idx=None)
            print(f"  Activations shape: {activations.shape}")

            # Test each layer's contribution
            print(f"  Testing layer-by-layer activation importance...")
            layer_scores = test_layer_importance(classifier, activations, n_layers, hidden_dim)

            # Also test with full activations
            with torch.no_grad():
                x = torch.tensor(activations, dtype=torch.float32).to(args.device)
                full_output = classifier(x)
                full_score = full_output.mean().item()  # Already sigmoid'd

            print(f"\n  Full model score: {full_score:.4f}")

            # Show layers with highest discrimination
            print(f"\n  Layer scores (top 5 / bottom 5):")
            top_indices = np.argsort(layer_scores)[-5:][::-1]
            for idx in top_indices:
                print(f"    Layer {idx:2d}: {layer_scores[idx]:.4f}")
            print("    ...")
            bottom_indices = np.argsort(layer_scores)[:5]
            for idx in bottom_indices:
                print(f"    Layer {idx:2d}: {layer_scores[idx]:.4f}")

            print(f"\n  Score range: {layer_scores.min():.4f} - {layer_scores.max():.4f} (spread: {layer_scores.max() - layer_scores.min():.4f})")


if __name__ == "__main__":
    main()
