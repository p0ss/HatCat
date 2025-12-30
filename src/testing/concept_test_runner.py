#!/usr/bin/env python3
"""
Shared Concept Detection Test Runner

Provides reusable functions for running concept detection experiments using
DynamicLensManager. This is the single source of truth for how to properly
run concept detection tests, extracted from the working temporal monitoring test.

Based on tests/test_temporal_monitoring.py
"""

import torch
from typing import Dict, List, Tuple, Optional


def generate_with_concept_detection(
    model,
    tokenizer,
    lens_manager,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k_concepts: int = 5,
    threshold: float = 0.1,
    device: str = "cuda",
    record_per_token: bool = True
) -> dict:
    """
    Generate text and detect concepts using DynamicLensManager.

    This is the WORKING approach from temporal monitoring that should be used
    by all concept detection tests.

    Args:
        model: The language model
        tokenizer: The tokenizer
        lens_manager: DynamicLensManager instance
        prompt: Input prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling top-p
        top_k_concepts: Number of top concepts to return
        threshold: Minimum probability to record a concept
        device: Device to use ('cuda' or 'cpu')
        record_per_token: If True, record concepts per token; if False, only final

    Returns:
        Dict containing:
            - prompt: Original prompt
            - generated_text: Generated text
            - tokens: List of generated tokens
            - timesteps: List of per-token concept detections (if record_per_token)
            - final_concepts: Concepts detected in final state
            - summary: Summary statistics
    """
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    timesteps = [] if record_per_token else None

    with torch.inference_mode():
        # Generate with hidden states
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

        # Extract generated tokens
        token_ids = outputs.sequences[0][prompt_len:].cpu().tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Process hidden states for each forward pass
        for step_idx, step_states in enumerate(outputs.hidden_states):
            # Use last layer, last position
            last_layer = step_states[-1]  # [1, seq_len, hidden_dim]
            hidden_state = last_layer[:, -1, :]  # [1, hidden_dim]

            # Convert to float32 to match classifier dtype
            hidden_state_f32 = hidden_state.float()

            # Use DynamicLensManager to detect and expand
            detected, timing = lens_manager.detect_and_expand(
                hidden_state_f32,
                top_k=top_k_concepts,
                return_timing=True
            )

            # Filter by threshold and convert to dict
            concept_scores = {}
            for concept_name, prob, layer in detected:
                if prob > threshold:
                    concept_scores[concept_name] = {
                        'probability': float(prob),
                        'layer': int(layer)
                    }

            # Record this timestep if requested
            if record_per_token:
                # Get token info
                token = tokens[step_idx] if step_idx < len(tokens) else '<eos>'

                # Record timestep
                timesteps.append({
                    'forward_pass': step_idx,
                    'token_idx': step_idx,
                    'token': token,
                    'position': prompt_len + step_idx,
                    'concepts': concept_scores
                })

            # Always save the final state
            if step_idx == len(outputs.hidden_states) - 1:
                final_concepts = concept_scores

    # Build result in expected format
    generated_text = ''.join(tokens)

    result = {
        'prompt': prompt,
        'generated_text': generated_text,
        'tokens': tokens,
        'final_concepts': final_concepts,
    }

    if record_per_token:
        result['timesteps'] = timesteps
        result['summary'] = {
            'unique_concepts_detected': len(set(
                concept for ts in timesteps
                for concept in ts['concepts'].keys()
            ))
        }
    else:
        result['summary'] = {
            'unique_concepts_detected': len(final_concepts)
        }

    return result


def score_activation_with_lens_manager(
    activation: torch.Tensor,
    lens_manager,
    top_k: int = 10,
    threshold: float = 0.3
) -> dict:
    """
    Score a single activation vector against HatCat concept lenses.

    This function is for scoring pre-computed activations (e.g., from saved data)
    rather than live generation.

    Args:
        activation: Activation tensor [hidden_dim] or [1, hidden_dim]
        lens_manager: DynamicLensManager instance
        top_k: Number of top concepts to return
        threshold: Minimum probability to include concept

    Returns:
        Dict containing:
            - top_concepts: List of concept names (sorted by probability)
            - top_probabilities: List of probabilities (sorted descending)
            - all_scores: Dict mapping concept name -> probability
            - concept_details: Dict mapping concept name -> {probability, layer}
    """
    # Ensure tensor is on correct device and dtype
    if activation.dim() == 1:
        activation = activation.unsqueeze(0)  # Add batch dimension

    activation = activation.float().to(lens_manager.device)

    # Use detect_and_expand (working method from temporal monitoring)
    detections, _ = lens_manager.detect_and_expand(
        activation,
        top_k=top_k,
        return_timing=True
    )

    # Filter by threshold and convert to dict
    concept_details = {}
    for concept_name, prob, layer in detections:
        if prob > threshold:
            concept_details[concept_name] = {
                'probability': float(prob),
                'layer': int(layer)
            }

    # Build return format
    top_concepts = []
    top_probabilities = []

    # Sort by probability descending
    sorted_concepts = sorted(
        concept_details.items(),
        key=lambda x: x[1]['probability'],
        reverse=True
    )

    for concept_name, data in sorted_concepts[:top_k]:
        top_concepts.append(concept_name)
        top_probabilities.append(data['probability'])

    # Also create all_scores dict with just probabilities
    all_scores = {
        name: data['probability']
        for name, data in concept_details.items()
    }

    return {
        'top_concepts': top_concepts,
        'top_probabilities': top_probabilities,
        'all_scores': all_scores,
        'concept_details': concept_details
    }


def batch_score_activations(
    activations: List[torch.Tensor],
    lens_manager,
    top_k: int = 10,
    threshold: float = 0.3
) -> List[dict]:
    """
    Score multiple activation vectors against HatCat concept lenses.

    Args:
        activations: List of activation tensors
        lens_manager: DynamicLensManager instance
        top_k: Number of top concepts to return per activation
        threshold: Minimum probability to include concept

    Returns:
        List of result dicts (same format as score_activation_with_lens_manager)
    """
    results = []

    for activation in activations:
        result = score_activation_with_lens_manager(
            activation,
            lens_manager,
            top_k=top_k,
            threshold=threshold
        )
        results.append(result)

    return results
