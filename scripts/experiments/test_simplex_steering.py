#!/usr/bin/env python3
"""
Test simplex steering vectors on actual generation.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load steering vectors
VECTORS_DIR = Path("/home/poss/Documents/Code/HatCat/results/simplex_steering_vectors/run_20251226_104933")


def get_layers(model):
    """Get model layers handling different architectures."""
    if hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    elif hasattr(model.model, 'layers'):
        return model.model.layers
    else:
        raise AttributeError(f"Cannot find layers in model: {type(model.model)}")


def test_steering(
    model,
    tokenizer,
    steering_vector: torch.Tensor,
    prompt: str,
    layer_idx: int = 12,
    scales: list = [0.0, 1.0, 3.0, 5.0, 10.0]
):
    """Test steering at different scales."""
    device = next(model.parameters()).device
    layers = get_layers(model)

    print(f"\nPrompt: '{prompt}'")
    print(f"Vector magnitude: {steering_vector.norm():.2f}")
    print("-" * 60)

    for scale in scales:
        # Create steering hook
        def steering_hook(module, input, output):
            if scale == 0.0:
                return output
            hidden = output[0]
            # Add steering vector to all tokens
            vec = steering_vector.to(device).to(hidden.dtype)
            hidden = hidden + scale * vec
            return (hidden,) + output[1:]

        # Register hook
        handle = layers[layer_idx].register_forward_hook(steering_hook)

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        handle.remove()

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated[len(prompt):].strip()
        print(f"scale={scale:5.1f}: {continuation[:70]}...")


def main():
    print("=" * 70)
    print("SIMPLEX STEERING TEST")
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
    print("Model loaded")

    # Test threat_perception: alarm → safety
    print("\n" + "=" * 70)
    print("TEST 1: threat_perception (negative → positive = alarm → safety)")
    print("=" * 70)

    vec_path = VECTORS_DIR / "threat_perception" / "negative_to_positive.pt"
    vec = torch.load(vec_path, map_location="cpu")
    print(f"Loaded steering vector: {vec.shape}")

    # Anxious prompt - steer toward safety
    test_steering(
        model, tokenizer, vec,
        "I feel really anxious and scared because",
        layer_idx=12
    )

    # Test with negative scale (steer toward alarm)
    print("\n" + "=" * 70)
    print("TEST 2: threat_perception (reverse: toward alarm)")
    print("=" * 70)

    test_steering(
        model, tokenizer, -vec,  # Negative = toward alarm
        "I feel calm and safe because",
        layer_idx=12
    )

    # Test social_orientation: hostility → cooperation
    print("\n" + "=" * 70)
    print("TEST 3: social_orientation (negative → positive = hostility → cooperation)")
    print("=" * 70)

    vec_path = VECTORS_DIR / "social_orientation" / "negative_to_positive.pt"
    vec = torch.load(vec_path, map_location="cpu")

    test_steering(
        model, tokenizer, vec,
        "When someone disagrees with me, I usually",
        layer_idx=12
    )

    # Test motivational_regulation: demotivation → motivation
    print("\n" + "=" * 70)
    print("TEST 4: motivational_regulation (negative → positive)")
    print("=" * 70)

    vec_path = VECTORS_DIR / "motivational_regulation" / "negative_to_positive.pt"
    vec = torch.load(vec_path, map_location="cpu")

    test_steering(
        model, tokenizer, vec,
        "I don't want to do anything because",
        layer_idx=12
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
