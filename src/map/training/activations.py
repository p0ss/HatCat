"""
Activation extraction from language models.

Phase 4 methodology: Extract mean activation from final layer during generation.
"""

import torch
import numpy as np


def get_mean_activation(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
    layer_idx: int = -1
) -> np.ndarray:
    """
    Extract mean activation from model for a given prompt.

    Generates text and averages hidden states across generation steps.

    Args:
        model: Language model (AutoModelForCausalLM)
        tokenizer: Tokenizer for the model
        prompt: Input prompt
        device: Device to run on
        layer_idx: Layer index to extract from (-1 for last layer)

    Returns:
        Mean activation vector (hidden_dim,)

    Example:
        >>> activation = get_mean_activation(model, tokenizer, "What is person?")
        >>> activation.shape
        (2560,)
    """
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

        mean_activation = np.stack(activations).mean(axis=0)

    return mean_activation
