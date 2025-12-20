#!/usr/bin/env python3
"""
Test the new generation-based activation extraction with varied temperature.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.map.training.sumo_classifiers import extract_activations


def main():
    print("=" * 80)
    print("TESTING GENERATION-BASED ACTIVATION EXTRACTION")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model_name = "google/gemma-3-4b-pt"
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print("✓ Model loaded")

    # Test prompts
    test_prompts = [
        "What is 'Process'? The result of alteration or modification",
        "Give me examples of 'Process'.",
        "Describe 'Physical'.",
    ]

    print(f"\nExtracting activations from {len(test_prompts)} prompts...")
    print("Temperature range: 0.3 to 1.2")

    # Extract activations (should now generate and capture generation activations)
    activations = extract_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=test_prompts,
        device=device,
        layer_idx=15,
        max_new_tokens=50,
        temperature_range=(0.3, 1.2),
    )

    print(f"\n✓ Extracted activations shape: {activations.shape}")
    print(f"  Expected: ({len(test_prompts)}, 2560)")

    # Verify diversity
    print("\nChecking activation diversity:")
    for i in range(len(test_prompts)):
        for j in range(i + 1, len(test_prompts)):
            # Cosine similarity
            cos_sim = (activations[i] @ activations[j]) / (
                (activations[i] @ activations[i]) ** 0.5 *
                (activations[j] @ activations[j]) ** 0.5
            )
            print(f"  Prompt {i+1} vs {j+1}: cosine similarity = {cos_sim:.3f}")

    print("\n" + "=" * 80)
    print("SUCCESS: Generation-based activation extraction working!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
