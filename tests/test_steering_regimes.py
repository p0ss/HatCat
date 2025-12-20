#!/usr/bin/env python3
"""
Test Three Steering Regimes Using Proven Infrastructure

Uses src.steering pipeline that already works.
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

from src.hat import extract_concept_vector, generate_with_steering
from src.utils.gpu_cleanup import cleanup_model, print_gpu_memory


def extract_contrastive_vector(model, tokenizer, concept, negative, device="cuda"):
    """Extract contrastive vector: v = activations[concept] - activations[negative]."""
    # Use existing extraction for both
    v_pos = extract_concept_vector(model, tokenizer, concept, device=device)
    v_neg = extract_concept_vector(model, tokenizer, negative, device=device)

    v_contrastive = v_pos - v_neg
    v_contrastive = v_contrastive / (np.linalg.norm(v_contrastive) + 1e-8)

    return v_contrastive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--concept", default="happy")
    parser.add_argument("--negative", default="sad")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/steering_regimes_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STEERING REGIMES TEST - USING PROVEN INFRASTRUCTURE")
    print("="*60)
    print(f"Concept: {args.concept}")
    print(f"Prompt: 'Tell me about {args.concept}.'")
    print(f"Using: src.steering.generate_with_steering")
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
        # Extract vectors
        print(f"\nExtracting standard vector for '{args.concept}'...")
        v_standard = extract_concept_vector(model, tokenizer, args.concept, device=args.device)
        print(f"✓ Vector shape: {v_standard.shape}")

        print(f"\nExtracting contrastive vector ('{args.concept}' - '{args.negative}')...")
        v_contrastive = extract_contrastive_vector(
            model, tokenizer, args.concept, args.negative, device=args.device
        )
        print(f"✓ Vector shape: {v_contrastive.shape}")

        similarity = float(np.dot(v_standard, v_contrastive))
        print(f"\nVector similarity: {similarity:.4f}\n")

        prompt = f"Tell me about {args.concept}."
        results = {}

        # Test both extraction types
        for extraction_name, vector in [("standard", v_standard), ("contrastive", v_contrastive)]:
            print("="*60)
            print(f"EXTRACTION: {extraction_name.upper()}")
            print("="*60)

            results[extraction_name] = {}

            # Test multiple strengths
            for strength in [0.0, 0.5, 1.0, -0.5, -1.0]:
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

                results[extraction_name][f"strength_{strength:+.1f}"] = samples

        # Save results
        output_data = {
            "metadata": {
                "test": "Steering Regimes - Using Proven Infrastructure",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model,
                "concept": args.concept,
                "negative_concept": args.negative,
                "prompt": prompt,
                "infrastructure": "src.steering.generate_with_steering"
            },
            "vectors": {
                "standard_norm": float(np.linalg.norm(v_standard)),
                "contrastive_norm": float(np.linalg.norm(v_contrastive)),
                "cosine_similarity": similarity
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
