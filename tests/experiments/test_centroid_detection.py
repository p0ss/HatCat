#!/usr/bin/env python3
"""
End-to-end test of centroid-based text detection.

This script:
1. Loads a trained layer 0 concept (Physical) with activation lens + centroid
2. Tests on sample prompts about physical vs abstract things
3. Measures divergence between activation and text detection
"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.map.training.sumo_classifiers import extract_activations
from src.hat.monitoring.centroid_detector import CentroidTextDetector


def test_concept_detection(concept_name="Physical"):
    """Test end-to-end centroid-based detection for a layer 0 concept."""

    print(f"\n{'='*80}")
    print(f"TESTING CENTROID DETECTION FOR: {concept_name}")
    print(f"{'='*80}\n")

    # Setup
    device = "cuda"
    model_name = "google/gemma-3-4b-pt"
    layer = 0

    results_dir = Path(f"results/sumo_classifiers/layer{layer}")

    # Step 1: Load model
    print("Step 1: Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Model loaded\n")

    # Step 2: Load trained activation lens
    print("Step 2: Loading trained activation lens...")

    # Reconstruct the model architecture (same as train_simple_classifier)
    import torch.nn as nn
    input_dim = 2560  # Gemma-3-4b hidden dim
    hidden_dim = 128
    activation_lens = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim // 2, 1),
        nn.Sigmoid(),
    ).to(device)

    # Load saved weights
    state_dict = torch.load(
        results_dir / f"{concept_name}_classifier.pt",
        map_location=device
    )
    activation_lens.load_state_dict(state_dict)
    activation_lens.eval()
    print("✓ Activation lens loaded\n")

    # Step 3: Load centroid
    print("Step 3: Loading centroid...")
    centroid_path = results_dir / "embedding_centroids" / f"{concept_name}_centroid.npy"
    text_lens = CentroidTextDetector.load(centroid_path, concept_name)
    print("✓ Centroid loaded\n")

    # Step 4: Test on various prompts
    print("Step 4: Testing on sample prompts...\n")

    test_prompts = [
        # Should trigger Physical concept
        "The rock fell down the mountain slope.",
        "My car needs new tires and an oil change.",
        "The tree grew tall in the forest.",

        # Should NOT trigger Physical concept (abstract things)
        "Democracy requires informed citizens.",
        "The concept of infinity is difficult to grasp.",
        "Justice should be blind to wealth and status.",
    ]

    print(f"{'Prompt':<50} {'Activation':>12} {'Text':>12} {'Divergence':>12}")
    print("-" * 90)

    for prompt in test_prompts:
        # Get activation confidence
        with torch.no_grad():
            activation = extract_activations(model, tokenizer, [prompt], device)[0]
            activation_tensor = torch.FloatTensor(activation).unsqueeze(0).to(device)
            activation_conf = activation_lens(activation_tensor).item()

        # Get text confidence from last generated token
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

            # Extract last token embedding
            last_step_hidden = outputs.hidden_states[-1]
            last_layer_hidden = last_step_hidden[-1]
            last_token_embedding = last_layer_hidden[0, -1, :].float().cpu().numpy()

            text_conf = text_lens.predict(last_token_embedding)

        divergence = activation_conf - text_conf

        prompt_short = prompt[:47] + "..." if len(prompt) > 50 else prompt
        print(f"{prompt_short:<50} {activation_conf:>11.3f} {text_conf:>11.3f} {divergence:>11.3f}")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")

    print("Interpretation:")
    print("  - Activation confidence: How strongly the internal activations express the concept")
    print("  - Text confidence: How strongly the generated text expresses the concept")
    print("  - Divergence: Difference between internal and external (HIGH = potential deception)")
    print()


if __name__ == "__main__":
    test_concept_detection("Physical")
