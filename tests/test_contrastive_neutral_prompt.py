#!/usr/bin/env python3
"""
Test Contrastive Extraction with Neutral Prompt

Uses "What emotion are you feeling?" to see pure steering effect
without concept in the prompt.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from src.utils.gpu_cleanup import cleanup_model, print_gpu_memory
from src.steering import extract_concept_vector


def extract_concept_vector_contrastive(
    model, tokenizer,
    concept: str,
    negative_concept: str,
    layer_idx: int = -1,
    device: str = "cuda"
) -> np.ndarray:
    """Extract contrastive vector: v = activations[concept] - activations[negative]."""
    def get_activations(phrase: str) -> np.ndarray:
        prompt = f"What is {phrase}?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

            activations = []
            for step_states in outputs.hidden_states:
                if layer_idx == -1:
                    last_layer = step_states[-1]
                else:
                    last_layer = step_states[layer_idx]

                act = last_layer[0, -1, :]
                activations.append(act.cpu().numpy())

            return np.stack(activations).mean(axis=0)

    pos_vector = get_activations(concept)
    neg_vector = get_activations(negative_concept)
    contrastive_vector = pos_vector - neg_vector
    contrastive_vector = contrastive_vector / (np.linalg.norm(contrastive_vector) + 1e-8)

    return contrastive_vector


def test_steering(
    model, tokenizer, concept_vector: np.ndarray,
    prompt: str, strength: float,
    n_samples: int, device: str,
    regime: str = "attenuation"
):
    """Generate with steering.

    Args:
        regime: Steering regime to use:
            - "attenuation": Projective modulation h ± α*(h·v)v (upregulate/downregulate)
            - "injection": Direct addition h ± α*v (add/subtract concept)
            - "nullification": Remove projection h - (h·v)v (erase concept, ignores strength)
    """
    if hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers

    target_layer = layers[-1]
    v_tensor = torch.from_numpy(concept_vector).float().to(device)

    samples = []

    for i in range(n_samples):
        set_seed(42 + i)

        original_mlp_forward = target_layer.mlp.forward

        if regime == "attenuation":
            # Projective modulation (current default)
            # Positive strength = subtract projection = suppress concept
            # Negative strength = add projection = amplify concept
            def hook_mlp(hidden_states):
                v_matched = v_tensor.to(dtype=hidden_states.dtype)
                projection = (hidden_states @ v_matched.unsqueeze(-1)) * v_matched
                steered = hidden_states - (strength * projection)  # SUBTRACT (correct)
                return original_mlp_forward(steered)

        elif regime == "injection":
            # Direct vector addition
            # Positive strength = add concept vector = amplify concept
            # Negative strength = subtract concept vector = suppress concept
            def hook_mlp(hidden_states):
                v_matched = v_tensor.to(dtype=hidden_states.dtype)
                steered = hidden_states + (strength * v_matched)  # ADD
                return original_mlp_forward(steered)

        elif regime == "nullification":
            # Remove concept projection entirely (ignores strength sign)
            def hook_mlp(hidden_states):
                v_matched = v_tensor.to(dtype=hidden_states.dtype)
                projection = (hidden_states @ v_matched.unsqueeze(-1)) * v_matched
                steered = hidden_states - projection  # Always subtract
                return original_mlp_forward(steered)
        else:
            raise ValueError(f"Unknown regime: {regime}")

        target_layer.mlp.forward = hook_mlp

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            samples.append(text)
        finally:
            target_layer.mlp.forward = original_mlp_forward

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--concept", default="happy")
    parser.add_argument("--negative", default="sad")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/contrastive_concept_prompt_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("="*60)
    print("STANDARD VS CONTRASTIVE - THREE STEERING REGIMES")
    print("="*60)
    print(f"Concept: {args.concept} vs {args.negative}")
    print(f"Prompt: 'Tell me about {args.concept}.'")
    print(f"Decoding: Greedy (deterministic)")
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
        print(f"\nExtracting STANDARD vector for '{args.concept}'...")
        v_standard = extract_concept_vector(model, tokenizer, args.concept, device=args.device)
        print(f"✓ Vector shape: {v_standard.shape}")

        print(f"\nExtracting CONTRASTIVE vector ('{args.concept}' - '{args.negative}')...")
        v_contrastive = extract_concept_vector_contrastive(
            model, tokenizer, args.concept, args.negative, device=args.device
        )
        print(f"✓ Vector shape: {v_contrastive.shape}")

        prompt = f"Tell me about {args.concept}."

        # Test all three regimes with both vectors
        results_by_regime = {}

        for extraction_type, concept_vector in [("standard", v_standard), ("contrastive", v_contrastive)]:
            for regime in ["attenuation", "injection", "nullification"]:
                key = f"{extraction_type}_{regime}"
                print("\n" + "="*60)
                print(f"EXTRACTION: {extraction_type.upper()} | REGIME: {regime.upper()}")
                print("="*60)

                results_by_regime[key] = {}

                print("\n" + "-"*60)
                print("BASELINE (strength 0.0)")
                print("-"*60)
                baseline_samples = test_steering(
                    model, tokenizer, concept_vector, prompt, 0.0, args.n_samples, args.device, regime
                )
                results_by_regime[key]["baseline"] = baseline_samples
                for i, text in enumerate(baseline_samples):
                    print(f"\nSample {i+1}:\n{text}")

                if regime != "nullification":  # Nullification ignores strength
                    print("\n" + "-"*60)
                    print("AMPLIFY HAPPY (strength +1.0)")
                    print("-"*60)
                    positive_samples = test_steering(
                        model, tokenizer, concept_vector, prompt, 1.0, args.n_samples, args.device, regime
                    )
                    results_by_regime[key]["amplify_happy"] = positive_samples
                    for i, text in enumerate(positive_samples):
                        print(f"\nSample {i+1}:\n{text}")

                    print("\n" + "-"*60)
                    print("SUPPRESS HAPPY / AMPLIFY SAD (strength -1.0)")
                    print("-"*60)
                    negative_samples = test_steering(
                        model, tokenizer, concept_vector, prompt, -1.0, args.n_samples, args.device, regime
                    )
                    results_by_regime[key]["suppress_happy"] = negative_samples
                    for i, text in enumerate(negative_samples):
                        print(f"\nSample {i+1}:\n{text}")
                else:
                    print("\n" + "-"*60)
                    print("NULLIFY (removes concept projection)")
                    print("-"*60)
                    nullify_samples = test_steering(
                        model, tokenizer, concept_vector, prompt, 0.0, args.n_samples, args.device, regime
                    )
                    results_by_regime[key]["nullify"] = nullify_samples
                    for i, text in enumerate(nullify_samples):
                        print(f"\nSample {i+1}:\n{text}")

        # Compute similarity between vectors
        similarity = float(np.dot(v_standard, v_contrastive))

        # Save results
        output_data = {
            "metadata": {
                "test": "Standard vs Contrastive - Three Steering Regimes",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model,
                "concept": args.concept,
                "negative_concept": args.negative,
                "prompt": prompt,
                "runtime_seconds": time.time() - start_time
            },
            "vectors": {
                "standard_norm": float(np.linalg.norm(v_standard)),
                "contrastive_norm": float(np.linalg.norm(v_contrastive)),
                "cosine_similarity": similarity
            },
            "regimes": {
                "attenuation": "Projective modulation h ± α*(h·v)v",
                "injection": "Direct addition h ± α*v",
                "nullification": "Remove projection h - (h·v)v"
            },
            "results": results_by_regime
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
