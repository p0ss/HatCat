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
        layer_idx: Layer index to extract from (-1 for last layer, None for auto-select)

    Returns:
        Mean activation vector (hidden_dim,)

    Example:
        >>> activation = get_mean_activation(model, tokenizer, "What is person?")
        >>> activation.shape
        (2560,)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Remove token_type_ids if present - some models (Llama) don't accept it
    inputs = {k: v for k, v in inputs.items() if k != 'token_type_ids'}

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
            num_layers = len(step_states)

            # Determine which layer to use
            # Some models (e.g., Gemma-3) have NaN in deeper layers during generation
            if layer_idx is None:
                # Auto-select: try last, then second-to-last, then middle, then first
                candidates = [-1, -2, num_layers // 2, 0]
            elif layer_idx == -1:
                # Default: prefer deeper layers but fall back to shallower ones
                candidates = [-1, -2, num_layers // 2, 0]
            else:
                # Specific layer requested, still try fallbacks
                candidates = [layer_idx, -1, num_layers // 2, 0]

            # Find a layer without NaN
            act = None
            for idx in candidates:
                try:
                    layer_state = step_states[idx]
                    candidate_act = layer_state[0, -1, :]
                    if not torch.isnan(candidate_act).any():
                        act = candidate_act.cpu().numpy()
                        break
                except IndexError:
                    continue

            if act is None:
                # Fall back to requested layer even with NaN
                layer_state = step_states[-1 if layer_idx == -1 else layer_idx]
                act = layer_state[0, -1, :].cpu().numpy()

            activations.append(act)

        mean_activation = np.stack(activations).mean(axis=0)

    return mean_activation
