"""
Concept vector extraction from model activations.

Phase 5 methodology: Extract concept vectors by averaging hidden states
during generation of concept-related text.
"""

import torch
import numpy as np
from typing import Optional


def extract_concept_vector(
    model,
    tokenizer,
    concept: str,
    layer_idx: int = -1,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extract concept vector from model activations.

    This matches Phase 5's approach: generate text about the concept
    and average the hidden states to get the concept direction.

    Args:
        model: Language model (should be AutoModelForCausalLM)
        tokenizer: Tokenizer for the model
        concept: Concept name (e.g., "person", "change")
        layer_idx: Layer to extract from (-1 for last layer)
        device: Device to run on ("cuda" or "cpu")

    Returns:
        Normalized concept vector (hidden_dim,)

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-pt", dtype=torch.float32)
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
        >>> vector = extract_concept_vector(model, tokenizer, "person")
        >>> vector.shape
        (2560,)
    """
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
            activations.append(act.float().cpu().numpy())

        # Average across generation steps
        concept_vector = np.stack(activations).mean(axis=0)

        # Normalize
        concept_vector = concept_vector / (np.linalg.norm(concept_vector) + 1e-8)

    return concept_vector
