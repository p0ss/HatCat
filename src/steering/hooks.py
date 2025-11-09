"""
Forward hooks for steering model generation.

Phase 5 methodology: Project out concept vectors from hidden states
during generation to steer model behavior.
"""

import torch
import numpy as np
from typing import Callable, Optional


def create_steering_hook(
    concept_vector: np.ndarray,
    strength: float,
    device: str
) -> Callable:
    """
    Create hook for steering generation.

    Projects out the concept vector from hidden states with specified strength.
    Formula: steered = hidden - strength * (hidden · vector) * vector

    Args:
        concept_vector: Normalized concept direction (hidden_dim,)
        strength: Steering strength (negative = suppress, positive = amplify)
        device: Device tensor should be on

    Returns:
        Hook function for PyTorch forward hooks

    Example:
        >>> hook_fn = create_steering_hook(concept_vector, strength=0.5, device="cuda")
        >>> handle = model.model.layers[-1].register_forward_hook(hook_fn)
        >>> # Generate with steering...
        >>> handle.remove()
    """
    concept_tensor = torch.tensor(concept_vector, dtype=torch.float32).to(device)

    def hook(module, input, output):
        """Project out concept vector from hidden states."""
        hidden_states = output[0]
        # Match tensor dtype to hidden states (e.g., float16 for model, float32 for operations)
        concept_matched = concept_tensor.to(dtype=hidden_states.dtype)
        # Project onto concept vector and subtract scaled projection
        projection = (hidden_states @ concept_matched.unsqueeze(-1)) * concept_matched
        steered = hidden_states - strength * projection
        return (steered,)

    return hook


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: Optional[np.ndarray] = None,
    strength: float = 0.0,
    layer_idx: int = -1,
    max_new_tokens: int = 50,
    device: str = "cuda",
    **generation_kwargs
) -> str:
    """
    Generate text with optional steering applied using forward hooks.

    If steering_vector is None or strength is 0.0, generates without steering.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Text prompt to complete
        steering_vector: Normalized concept vector (optional)
        strength: Steering strength (negative = suppress, positive = amplify)
        layer_idx: Layer to apply steering at (-1 for last layer)
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
        **generation_kwargs: Additional arguments for model.generate()

    Returns:
        Generated text (including prompt)

    Example:
        >>> text = generate_with_steering(
        ...     model, tokenizer,
        ...     prompt="Tell me about",
        ...     steering_vector=person_vector,
        ...     strength=-0.5  # Suppress "person" concept
        ... )
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if steering_vector is None or abs(strength) < 1e-6:
        # No steering
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Apply steering using forward hook
    hook_fn = create_steering_hook(steering_vector, strength, device)

    # Get target layer (handle different model architectures)
    if hasattr(model.model, 'language_model'):
        # Gemma-3 architecture
        layers = model.model.language_model.layers
    elif hasattr(model.model, 'layers'):
        # Gemma-2 architecture
        layers = model.model.layers
    else:
        raise AttributeError(f"Cannot find layers in model architecture: {type(model.model)}")

    if layer_idx == -1:
        target_layer = layers[-1]
    else:
        target_layer = layers[layer_idx]

    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        handle.remove()

    return generated_text
