#!/usr/bin/env python3
"""
Extract Steering Vector from Classifier Weights

Hypothesis: Classifier's decision boundary normal gives cleaner steering direction
than averaging activations, because it explicitly separates happy/sad clusters.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from src.steering import extract_concept_vector, generate_with_steering
from src.training.activations import collect_activations
from src.training.classifier import train_binary_classifier
from src.utils.gpu_cleanup import cleanup_model, print_gpu_memory


def extract_classifier_vector(classifier, device="cuda"):
    """
    Extract steering vector from classifier's first layer weights.

    The classifier learns: score = sigmoid(W·x + b)
    W is the normal to the decision boundary hyperplane.
    """
    # Get first layer weights (input_dim -> intermediate_dim)
    first_layer = classifier.net[0]  # nn.Linear layer
    weights = first_layer.weight.data  # (intermediate_dim, input_dim)

    # Average across intermediate neurons to get single direction
    # Each row is a direction, average gives the dominant separating direction
    avg_weights = weights.mean(dim=0).cpu().numpy()  # (input_dim,)

    # Normalize
    classifier_vector = avg_weights / (np.linalg.norm(avg_weights) + 1e-8)

    return classifier_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--concept", default="happy")
    parser.add_argument("--negative", default="sad")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/classifier_steering_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("CLASSIFIER-BASED STEERING VECTOR EXTRACTION")
    print("="*60)
    print(f"Concept: {args.concept} vs {args.negative}")
    print(f"Training classifier on {args.n_train} samples per class")
    print("="*60 + "\n")

    print_gpu_memory()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=args.device
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded")
    print_gpu_memory()

    try:
        # Step 1: Collect activations for training classifier
        print(f"\nCollecting {args.n_train} positive samples ('{args.concept}')...")
        pos_prompts = [f"Tell me about {args.concept}."] * args.n_train
        pos_acts = collect_activations(
            model, tokenizer, pos_prompts,
            layer_idx=-1, device=args.device
        )
        print(f"✓ Positive activations: {pos_acts.shape}")

        print(f"\nCollecting {args.n_train} negative samples ('{args.negative}')...")
        neg_prompts = [f"Tell me about {args.negative}."] * args.n_train
        neg_acts = collect_activations(
            model, tokenizer, neg_prompts,
            layer_idx=-1, device=args.device
        )
        print(f"✓ Negative activations: {neg_acts.shape}")

        # Step 2: Train classifier
        print("\nTraining binary classifier...")
        X = np.vstack([pos_acts, neg_acts])
        y = np.array([1] * args.n_train + [0] * args.n_train)

        classifier = train_binary_classifier(
            X, y,
            input_dim=X.shape[1],
            epochs=100,
            device=args.device,
            verbose=True
        )
        print("✓ Classifier trained")

        # Step 3: Extract vectors
        print("\nExtracting steering vectors...")

        print("  (1) Standard activation-based...")
        v_standard = extract_concept_vector(model, tokenizer, args.concept, device=args.device)
        print(f"      ✓ Shape: {v_standard.shape}")

        print("  (2) Contrastive (happy - sad)...")
        v_pos = extract_concept_vector(model, tokenizer, args.concept, device=args.device)
        v_neg = extract_concept_vector(model, tokenizer, args.negative, device=args.device)
        v_contrastive = v_pos - v_neg
        v_contrastive = v_contrastive / (np.linalg.norm(v_contrastive) + 1e-8)
        print(f"      ✓ Shape: {v_contrastive.shape}")

        print("  (3) Classifier weights...")
        v_classifier = extract_classifier_vector(classifier, device=args.device)
        print(f"      ✓ Shape: {v_classifier.shape}")

        # Compute similarities
        sim_std_contra = float(np.dot(v_standard, v_contrastive))
        sim_std_class = float(np.dot(v_standard, v_classifier))
        sim_contra_class = float(np.dot(v_contrastive, v_classifier))

        print(f"\n  Similarities:")
        print(f"    Standard ↔ Contrastive:  {sim_std_contra:.4f}")
        print(f"    Standard ↔ Classifier:   {sim_std_class:.4f}")
        print(f"    Contrastive ↔ Classifier: {sim_contra_class:.4f}")

        # Step 4: Test steering with all three vectors
        prompt = f"Tell me about {args.concept}."
        results = {}

        for vec_name, vector in [
            ("standard", v_standard),
            ("contrastive", v_contrastive),
            ("classifier", v_classifier)
        ]:
            print(f"\n{'='*60}")
            print(f"TESTING: {vec_name.upper()}")
            print('='*60)

            results[vec_name] = {}

            for strength in [0.0, 0.5, -0.5]:
                print(f"\n--- Strength: {strength:+.1f} ---")
                samples = []

                for i in range(args.n_samples):
                    set_seed(42 + i)

                    text = generate_with_steering(
                        model, tokenizer,
                        prompt=prompt,
                        steering_vector=vector,
                        strength=strength,
                        max_new_tokens=30,
                        device=args.device,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95
                    )
                    samples.append(text)
                    print(f"Sample {i+1}: {text[:100]}")

                results[vec_name][f"strength_{strength:+.1f}"] = samples

        # Save results
        output_data = {
            "metadata": {
                "test": "Classifier-Based Steering Vector",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model,
                "concept": args.concept,
                "negative_concept": args.negative,
                "n_train_samples": args.n_train,
                "prompt": prompt
            },
            "vectors": {
                "standard_norm": float(np.linalg.norm(v_standard)),
                "contrastive_norm": float(np.linalg.norm(v_contrastive)),
                "classifier_norm": float(np.linalg.norm(v_classifier)),
                "similarity_standard_contrastive": sim_std_contra,
                "similarity_standard_classifier": sim_std_class,
                "similarity_contrastive_classifier": sim_contra_class
            },
            "results": results
        }

        output_file = output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Saved to: {output_file}")
        print("="*60)

    finally:
        print("\nCleaning up GPU memory...")
        cleanup_model(model, tokenizer)
        print_gpu_memory()


if __name__ == "__main__":
    main()
