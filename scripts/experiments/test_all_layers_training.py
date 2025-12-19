#!/usr/bin/env python3
"""
Test training classifiers with all-layer activations.

Trains a few concepts and analyzes which layers the classifier learned to use.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.training.sumo_classifiers import (
    extract_activations,
    train_simple_classifier,
    get_hidden_dim,
    load_all_concepts,
)
from src.training.sumo_data_generation import (
    create_sumo_training_dataset,
    build_sumo_negative_pool,
)


def analyze_layer_importance(classifier, n_layers: int, hidden_dim: int):
    """
    Analyze which layers the classifier learned to focus on.

    The first layer of the classifier maps from [n_layers * hidden_dim] to hidden.
    We can analyze the weight magnitudes to see which layer slices matter.
    """
    # Get first layer weights: [hidden_neurons, n_layers * hidden_dim]
    W1 = classifier[0].weight.detach().cpu().numpy()

    # Reshape to [hidden_neurons, n_layers, hidden_dim]
    W1_by_layer = W1.reshape(W1.shape[0], n_layers, hidden_dim)

    # Compute importance per layer: mean absolute weight magnitude
    layer_importance = np.abs(W1_by_layer).mean(axis=(0, 2))  # [n_layers]

    # Normalize to sum to 1
    layer_importance = layer_importance / layer_importance.sum()

    return layer_importance


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--concepts", nargs="+",
                        default=["DomesticCat", "DomesticDog", "Algorithm", "Deception"])
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--output-dir", default="results/all_layers_test")
    parser.add_argument("--extraction-mode", default="prompt",
                        choices=["prompt", "combined", "generation"],
                        help="prompt=no generation (cleaner signal), combined=prompt+gen, generation=gen only")
    parser.add_argument("--pooling", default="last",
                        choices=["last", "mean", "max"],
                        help="last=last token (cleanest), mean=average all, max=max across positions")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    # Handle different model config structures (Gemma3 uses text_config)
    config = model.config
    if hasattr(config, 'num_hidden_layers'):
        n_layers = config.num_hidden_layers
    elif hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
        n_layers = config.text_config.num_hidden_layers
    else:
        raise AttributeError(f"Cannot find num_hidden_layers in config: {type(config)}")
    print(f"Model has {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"All-layers input dim: {n_layers * hidden_dim}")

    # Load hierarchy
    hierarchy_dir = Path("concept_packs/first-light/hierarchy")

    # Load all concepts for negative pool building
    all_concepts = load_all_concepts(hierarchy_dir)
    concept_map = {c["sumo_term"]: c for c in all_concepts}
    print(f"Loaded {len(all_concepts)} concepts from hierarchy")

    results = {}

    for concept_name in args.concepts:
        print(f"\n{'='*60}")
        print(f"Training: {concept_name}")
        print("="*60)

        # Find concept in hierarchy
        if concept_name not in concept_map:
            print(f"  Concept {concept_name} not found in hierarchy, skipping")
            continue

        concept_data = concept_map[concept_name]

        # Build negative pool
        negative_pool = build_sumo_negative_pool(all_concepts, concept_data, include_siblings=True)
        print(f"  Negative pool size: {len(negative_pool)}")

        # Split for train/test
        test_negative_pool = negative_pool[len(negative_pool) // 2:]

        # Generate training data
        print(f"  Generating training data...")
        train_prompts, train_labels = create_sumo_training_dataset(
            concept=concept_data,
            all_concepts=concept_map,
            negative_pool=negative_pool[:len(negative_pool) // 2],
            n_positives=args.n_train,
            n_negatives=args.n_train,
        )

        test_prompts, test_labels = create_sumo_training_dataset(
            concept=concept_data,
            all_concepts=concept_map,
            negative_pool=test_negative_pool,
            n_positives=args.n_test,
            n_negatives=args.n_test,
        )

        print(f"  Train: {len(train_prompts)} prompts, Test: {len(test_prompts)} prompts")
        print(f"  Train labels distribution: {sum(train_labels)} pos, {len(train_labels) - sum(train_labels)} neg")

        # Show sample prompts
        print(f"  Sample train prompts:")
        for i, (p, l) in enumerate(zip(train_prompts[:3], train_labels[:3])):
            print(f"    [{l}] {p[:80]}...")

        # Extract all-layer activations (layer_idx=None enables all-layers mode)
        import time
        print(f"  Extracting all-layer activations (mode={args.extraction_mode}, pooling={args.pooling})...")
        t0 = time.time()
        X_train = extract_activations(
            model, tokenizer, train_prompts, args.device,
            extraction_mode=args.extraction_mode, pooling=args.pooling,
            layer_idx=None  # All-layers mode
        )
        t1 = time.time()
        X_test = extract_activations(
            model, tokenizer, test_prompts, args.device,
            extraction_mode=args.extraction_mode, pooling=args.pooling,
            layer_idx=None  # All-layers mode
        )
        t2 = time.time()
        print(f"  Extraction time: train={t1-t0:.1f}s, test={t2-t1:.1f}s")
        print(f"  X_train: shape={X_train.shape}, min={X_train.min():.3f}, max={X_train.max():.3f}, std={X_train.std():.3f}")
        print(f"  X_test: shape={X_test.shape}")

        # Check if different layers have different activation patterns
        print(f"  Per-layer activation stats (first 5 layers vs last 5):")
        for layer_idx in [0, 1, 2, 3, 4, n_layers-5, n_layers-4, n_layers-3, n_layers-2, n_layers-1]:
            start = layer_idx * hidden_dim
            end = start + hidden_dim
            layer_acts = X_train[:, start:end]
            print(f"    Layer {layer_idx:2d}: mean={layer_acts.mean():.4f}, std={layer_acts.std():.4f}, norm={np.linalg.norm(layer_acts, axis=1).mean():.2f}")

        # Handle combined extraction (2x samples)
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)
        if X_train.shape[0] == 2 * len(train_labels):
            y_train_exp = np.repeat(y_train, 2)
        else:
            y_train_exp = y_train

        # Check discrimination per layer (difference between pos and neg examples)
        pos_mask = y_train_exp == 1
        neg_mask = y_train_exp == 0
        print(f"  Per-layer discrimination (pos - neg mean activation):")
        best_layer = -1
        best_diff = 0
        for layer_idx in range(n_layers):
            start = layer_idx * hidden_dim
            end = start + hidden_dim
            layer_acts = X_train[:, start:end]
            pos_mean = np.linalg.norm(layer_acts[pos_mask], axis=1).mean()
            neg_mean = np.linalg.norm(layer_acts[neg_mask], axis=1).mean()
            diff = abs(pos_mean - neg_mean)
            if diff > best_diff:
                best_diff = diff
                best_layer = layer_idx
        print(f"    Best discriminating layer: {best_layer} (diff={best_diff:.4f})")
        if X_train.shape[0] == 2 * len(train_labels):
            y_train = np.repeat(y_train, 2)
            y_test = np.repeat(y_test, 2)

        # Train classifier - 128 hidden neurons should be enough
        # to learn which of 34 layer slices to focus on
        print(f"  Training classifier (hidden=128)...")
        classifier, metrics = train_simple_classifier(
            X_train, y_train, X_test, y_test,
            hidden_dim=128,
            epochs=200,
        )

        print(f"  Test F1: {metrics['test_f1']:.3f}")
        print(f"  Test Precision: {metrics['test_precision']:.3f}, Recall: {metrics['test_recall']:.3f}")

        # Analyze layer importance
        layer_importance = analyze_layer_importance(classifier, n_layers, hidden_dim)

        # Find top layers
        top_k = 5
        top_indices = np.argsort(layer_importance)[-top_k:][::-1]

        print(f"\n  Layer importance (top {top_k}):")
        for idx in top_indices:
            print(f"    Layer {idx}: {layer_importance[idx]*100:.1f}%")

        # Save results
        results[concept_name] = {
            "metrics": metrics,
            "layer_importance": layer_importance.tolist(),
            "top_layers": top_indices.tolist(),
        }

        # Save classifier
        classifier_path = output_dir / f"{concept_name}_classifier.pt"
        torch.save(classifier.state_dict(), classifier_path)
        print(f"  Saved classifier to {classifier_path}")

    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Print comparison
    print("\n" + "="*60)
    print("SUMMARY: Layer Usage by Concept")
    print("="*60)
    for concept, data in results.items():
        top = data["top_layers"][:3]
        importance = [data["layer_importance"][i]*100 for i in top]
        print(f"{concept}:")
        print(f"  Top layers: {top} ({importance[0]:.1f}%, {importance[1]:.1f}%, {importance[2]:.1f}%)")
        print(f"  F1: {data['metrics']['test_f1']:.3f}")


if __name__ == "__main__":
    main()
