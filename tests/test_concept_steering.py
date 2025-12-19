"""
Concept Steering Test: Suppress "book" concept during generation

Uses our trained binary classifier to:
1. Extract the "book" concept direction from activations
2. Suppress it during generation by subtracting the concept vector
3. Compare steered vs unsteered generation
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


def extract_concept_vector(model, tokenizer, concept_prompt: str, layer_idx: int = -1):
    """
    Extract concept vector directly from model activations.

    Uses mean of activations when generating about the concept.
    """
    inputs = tokenizer(concept_prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,  # Deterministic for consistency
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        # Collect activations from the target layer across all generation steps
        activations = []
        for step_states in outputs.hidden_states:
            if layer_idx == -1:
                last_layer = step_states[-1]
            else:
                last_layer = step_states[layer_idx]

            # Take last token's activation
            act = last_layer[0, -1, :]
            activations.append(act)

        # Mean across time
        concept_vector = torch.stack(activations).mean(dim=0)

        # Normalize
        concept_vector = concept_vector / concept_vector.norm()

    return concept_vector


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    concept_vector: torch.Tensor = None,
    steering_strength: float = 0.0,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    layer_idx: int = -1
):
    """
    Generate text with optional concept steering.

    Args:
        concept_vector: Direction to suppress (if steering_strength > 0)
        steering_strength: How much to suppress (0 = no steering, 1.0 = strong)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if concept_vector is not None and steering_strength > 0:
        # Register hook to steer activations
        concept_vector = concept_vector.to(model.device)

        def steering_hook(module, input, output):
            # output is tuple of (hidden_states,)
            hidden_states = output[0]

            # Project onto concept vector and add (positive = amplify)
            projection = (hidden_states @ concept_vector.unsqueeze(-1)) * concept_vector
            steered = hidden_states + steering_strength * projection

            return (steered,)

        # Get the target layer (Gemma3 has language_model.layers)
        if layer_idx == -1:
            target_layer = model.model.language_model.layers[-1]
        else:
            target_layer = model.model.language_model.layers[layer_idx]

        # Register hook
        handle = target_layer.register_forward_hook(steering_hook)
    else:
        handle = None

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    finally:
        if handle is not None:
            handle.remove()


def main():
    print("=" * 70)
    print("CONCEPT STEERING TEST: Suppress 'book' concept")
    print("=" * 70)
    print()

    # Load model
    model_name = "google/gemma-3-4b-pt"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract "book" concept vector from activations
    print("Extracting 'book' concept vector from model activations...")
    concept_vector = extract_concept_vector(
        model,
        tokenizer,
        "What is a book?",
        layer_idx=-1
    )
    print(f"Concept vector shape: {concept_vector.shape}")
    print(f"Concept vector norm: {concept_vector.norm().item():.4f}")

    print()

    # Test prompts that might naturally lead to books
    test_prompts = [
        "My favorite book is",
        "I just finished reading a book about",
        "The library has many books about",
        "Can you recommend a good book on",
    ]

    # Bidirectional: negative suppresses, positive amplifies
    steering_strengths = [-1.0, -0.5, 0.0, 0.5, 1.0]

    for prompt in test_prompts:
        print("=" * 70)
        print(f"PROMPT: \"{prompt}\"")
        print("=" * 70)
        print()

        for strength in steering_strengths:
            if strength == 0.0:
                label = "BASELINE (no steering)"
            elif strength < 0:
                label = f"SUPPRESS (strength={strength})"
            else:
                label = f"AMPLIFY (strength={strength})"

            print(f"{label}:")

            generated = generate_with_steering(
                model,
                tokenizer,
                prompt,
                concept_vector=concept_vector,
                steering_strength=strength,
                max_new_tokens=30,
                temperature=0.8
            )

            # Extract just the generated part
            generated_only = generated[len(prompt):].strip()
            print(f"  {generated_only}")
            print()

        print()

    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    print("Expected behavior:")
    print("- Baseline: Natural completion continuing the prompt")
    print("- SUPPRESS (negative): Should avoid/reduce book-related content")
    print("- AMPLIFY (positive): Should emphasize/increase book-related content")
    print()
    print("If steering works:")
    print("✓ Concept vector captures meaningful semantic direction")
    print("✓ Our approach generalizes beyond detection to control")
    print("✓ Bi-directional manipulation is possible (suppress AND amplify)")
    print()
    print("Note: Prompts now explicitly include 'book' to test steering effect")


if __name__ == "__main__":
    main()
