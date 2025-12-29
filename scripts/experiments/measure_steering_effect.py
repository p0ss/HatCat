#!/usr/bin/env python3
"""
Measure steering vector effects in activation space.

Instead of relying on coherent generation, measure:
1. How much a steering vector shifts activations toward the target pole
2. Whether steered activations are closer to target pole centroid
"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

VECTORS_DIR = Path("/home/poss/Documents/Code/HatCat/results/simplex_steering_vectors/run_20251226_104933")


def get_layers(model):
    """Get model layers handling different architectures."""
    if hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    elif hasattr(model.model, 'layers'):
        return model.model.layers
    else:
        raise AttributeError(f"Cannot find layers in model")


def extract_with_steering(
    model, tokenizer, prompt: str,
    steering_vector: torch.Tensor,
    scale: float,
    layer_idx: int = 12
) -> torch.Tensor:
    """Extract activation with steering applied."""
    device = next(model.parameters()).device
    layers = get_layers(model)

    steered_activation = [None]

    def capture_hook(module, input, output):
        hidden = output[0]
        if scale != 0.0:
            vec = steering_vector.to(device).to(hidden.dtype)
            hidden = hidden + scale * vec
        steered_activation[0] = hidden[0, -1, :].cpu().float()
        return (hidden,) + output[1:]

    handle = layers[layer_idx].register_forward_hook(capture_hook)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return steered_activation[0]


def main():
    print("=" * 70)
    print("STEERING EFFECT MEASUREMENT")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model_name = "google/gemma-3-4b-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Test threat_perception
    print("\n" + "=" * 70)
    print("THREAT_PERCEPTION: Does steering move activations toward target pole?")
    print("=" * 70)

    # Load vectors and centroids
    simplex_dir = VECTORS_DIR / "threat_perception"
    neg_to_pos = torch.load(simplex_dir / "negative_to_positive.pt", map_location="cpu")
    centroid_neg = torch.load(simplex_dir / "centroid_negative.pt", map_location="cpu")
    centroid_pos = torch.load(simplex_dir / "centroid_positive.pt", map_location="cpu")
    centroid_neu = torch.load(simplex_dir / "centroid_neutral.pt", map_location="cpu")

    # Test prompts that should be in negative pole territory
    negative_prompts = [
        "I feel really scared and anxious",
        "There's something dangerous here",
        "I sense a terrible threat",
    ]

    # Test prompts that should be in positive pole territory
    positive_prompts = [
        "I feel completely safe and calm",
        "Everything is peaceful here",
        "I trust that nothing bad will happen",
    ]

    print("\n--- Negative pole prompts (scared/anxious) ---")
    for prompt in negative_prompts:
        print(f"\nPrompt: '{prompt}'")

        # Extract at different scales
        results = []
        for scale in [0.0, 0.5, 1.0, 2.0, 3.0]:
            act = extract_with_steering(model, tokenizer, prompt, neg_to_pos, scale)

            # Measure similarity to each pole centroid
            sim_neg = cosine_similarity(act.unsqueeze(0).numpy(), centroid_neg.unsqueeze(0).numpy())[0,0]
            sim_pos = cosine_similarity(act.unsqueeze(0).numpy(), centroid_pos.unsqueeze(0).numpy())[0,0]
            sim_neu = cosine_similarity(act.unsqueeze(0).numpy(), centroid_neu.unsqueeze(0).numpy())[0,0]

            results.append((scale, sim_neg, sim_neu, sim_pos))

        print(f"  scale | sim_neg | sim_neu | sim_pos | shift toward pos")
        print(f"  " + "-" * 55)
        baseline_pos = results[0][3]
        for scale, sim_neg, sim_neu, sim_pos in results:
            shift = sim_pos - baseline_pos
            print(f"  {scale:5.1f} | {sim_neg:.5f} | {sim_neu:.5f} | {sim_pos:.5f} | {shift:+.5f}")

    print("\n--- Positive pole prompts (safe/calm) ---")
    for prompt in positive_prompts:
        print(f"\nPrompt: '{prompt}'")

        results = []
        for scale in [0.0, 0.5, 1.0, 2.0, 3.0]:
            act = extract_with_steering(model, tokenizer, prompt, neg_to_pos, scale)

            sim_neg = cosine_similarity(act.unsqueeze(0).numpy(), centroid_neg.unsqueeze(0).numpy())[0,0]
            sim_pos = cosine_similarity(act.unsqueeze(0).numpy(), centroid_pos.unsqueeze(0).numpy())[0,0]
            sim_neu = cosine_similarity(act.unsqueeze(0).numpy(), centroid_neu.unsqueeze(0).numpy())[0,0]

            results.append((scale, sim_neg, sim_neu, sim_pos))

        print(f"  scale | sim_neg | sim_neu | sim_pos | shift toward pos")
        print(f"  " + "-" * 55)
        baseline_pos = results[0][3]
        for scale, sim_neg, sim_neu, sim_pos in results:
            shift = sim_pos - baseline_pos
            print(f"  {scale:5.1f} | {sim_neg:.5f} | {sim_neu:.5f} | {sim_pos:.5f} | {shift:+.5f}")

    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If steering works correctly:
- Negative prompts should show INCREASING sim_pos as scale increases
- The shift column should be positive and growing

If poles are too similar in activation space:
- All similarities will be ~1.0 and nearly identical
- Shifts will be tiny regardless of scale
""")


if __name__ == "__main__":
    main()
