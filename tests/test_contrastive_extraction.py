#!/usr/bin/env python3
"""
Test Contrastive Vector Extraction

Hypothesis: Current extraction gets the happy↔sad AXIS, not the →happy DIRECTION.
When we apply ±strength, we get both poles because the vector encodes both.

Test:
1. Current method: v = mean(activations["happy"])
2. Contrastive method: v = mean(activations["happy"]) - mean(activations["sad"])

Expected: Contrastive method should give directional steering (not both poles).
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


def extract_concept_vector_contrastive(
    model, tokenizer,
    concept: str,
    negative_concept: str = "sad",
    layer_idx: int = -1,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extract concept vector using contrastive method.

    v = mean(activations[concept]) - mean(activations[negative_concept])

    This should give a direction: away from negative, toward positive.
    """
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

    # Contrastive: point toward positive, away from negative
    contrastive_vector = pos_vector - neg_vector

    # Normalize
    contrastive_vector = contrastive_vector / (np.linalg.norm(contrastive_vector) + 1e-8)

    return contrastive_vector


def extract_concept_vector_standard(
    model, tokenizer,
    concept: str,
    layer_idx: int = -1,
    device: str = "cuda"
) -> np.ndarray:
    """Standard extraction (current method)."""
    concept_prompt = f"What is {concept}?"
    inputs = tokenizer(concept_prompt, return_tensors="pt").to(device)

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

        concept_vector = np.stack(activations).mean(axis=0)
        concept_vector = concept_vector / (np.linalg.norm(concept_vector) + 1e-8)

    return concept_vector


def test_steering(
    model, tokenizer, concept_vector: np.ndarray,
    prompt: str, strength: float,
    n_samples: int, device: str
):
    """Generate with steering and return samples."""
    if hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers

    target_layer = layers[-1]
    v_tensor = torch.from_numpy(concept_vector).float().to(device)

    samples = []

    for i in range(n_samples):
        set_seed(42 + i)

        # Before MLP placement (from previous test)
        original_mlp_forward = target_layer.mlp.forward

        def hook_mlp(hidden_states):
            v_matched = v_tensor.to(dtype=hidden_states.dtype)
            projection = (hidden_states @ v_matched.unsqueeze(-1)) * v_matched
            steered = hidden_states + (strength * projection)  # FIXED: Add projection (positive = amplify concept)
            return original_mlp_forward(steered)

        target_layer.mlp.forward = hook_mlp

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
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
    parser.add_argument("--output-dir", default="results/contrastive_extraction_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("="*60)
    print("CONTRASTIVE EXTRACTION TEST")
    print("="*60)
    print(f"Concept: {args.concept}")
    print(f"Negative: {args.negative}")
    print(f"Hypothesis: Current method encodes AXIS (both poles)")
    print(f"           Contrastive method encodes DIRECTION (one pole)")
    print("="*60 + "\n")

    print_gpu_memory()

    # Load model
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
        print(f"\nExtracting STANDARD vector for '{args.concept}'...")
        v_standard = extract_concept_vector_standard(model, tokenizer, args.concept, device=args.device)
        print(f"✓ Vector shape: {v_standard.shape}")

        print(f"\nExtracting CONTRASTIVE vector ('{args.concept}' - '{args.negative}')...")
        v_contrastive = extract_concept_vector_contrastive(
            model, tokenizer, args.concept, args.negative, device=args.device
        )
        print(f"✓ Vector shape: {v_contrastive.shape}")

        # Compute similarity
        similarity = float(np.dot(v_standard, v_contrastive))
        print(f"\nCosine similarity between vectors: {similarity:.4f}")
        print("(High similarity = both encode same thing)")
        print("(Low similarity = they're different directions)\n")

        # Test steering at strength +1.0
        prompt = f"Tell me about {args.concept}."

        print("="*60)
        print("TESTING STANDARD EXTRACTION at strength +1.0")
        print("="*60)
        standard_samples = test_steering(
            model, tokenizer, v_standard, prompt, 1.0, args.n_samples, args.device
        )
        for i, text in enumerate(standard_samples):
            print(f"\nSample {i+1}:")
            print(text[:200])

        print("\n" + "="*60)
        print("TESTING CONTRASTIVE EXTRACTION at strength +1.0")
        print("="*60)
        contrastive_samples = test_steering(
            model, tokenizer, v_contrastive, prompt, 1.0, args.n_samples, args.device
        )
        for i, text in enumerate(contrastive_samples):
            print(f"\nSample {i+1}:")
            print(text[:200])

        print("\n" + "="*60)
        print("TESTING STANDARD EXTRACTION at strength -1.0")
        print("="*60)
        standard_neg_samples = test_steering(
            model, tokenizer, v_standard, prompt, -1.0, args.n_samples, args.device
        )
        for i, text in enumerate(standard_neg_samples):
            print(f"\nSample {i+1}:")
            print(text[:200])

        print("\n" + "="*60)
        print("TESTING CONTRASTIVE EXTRACTION at strength -1.0")
        print("="*60)
        contrastive_neg_samples = test_steering(
            model, tokenizer, v_contrastive, prompt, -1.0, args.n_samples, args.device
        )
        for i, text in enumerate(contrastive_neg_samples):
            print(f"\nSample {i+1}:")
            print(text[:200])

        # Save results
        output_data = {
            "metadata": {
                "test": "Contrastive vs Standard Extraction",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model,
                "concept": args.concept,
                "negative_concept": args.negative,
                "runtime_seconds": time.time() - start_time
            },
            "vectors": {
                "standard_norm": float(np.linalg.norm(v_standard)),
                "contrastive_norm": float(np.linalg.norm(v_contrastive)),
                "cosine_similarity": similarity
            },
            "results": {
                "standard_positive": standard_samples,
                "contrastive_positive": contrastive_samples,
                "standard_negative": standard_neg_samples,
                "contrastive_negative": contrastive_neg_samples
            },
            "hypothesis": {
                "statement": "Standard extraction encodes bidirectional axis, contrastive encodes unidirectional vector",
                "expected": "Contrastive method should not mix positive/negative concepts at same strength",
                "standard_issue": "At +1.0, should see mix of happy and sad content",
                "contrastive_fix": "At +1.0, should see only happy content; at -1.0, only sad content"
            }
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
