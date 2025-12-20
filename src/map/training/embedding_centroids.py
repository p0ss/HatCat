"""
Embedding-based text detection using concept centroids.

Instead of training TF-IDF classifiers, we compute embedding centroids
for each concept and use cosine similarity at inference time.
"""

from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import json


def compute_concept_centroid(
    model,
    tokenizer,
    prompts: List[str],
    device: str = "cuda",
    layer_idx: int = -1,
    max_response_tokens: int = 50,
) -> np.ndarray:
    """
    Compute centroid embedding for a concept from model-generated responses.

    IMPORTANT: Generates responses for each prompt and extracts embeddings from
    the GENERATED text, not the prompts themselves. This ensures the centroid
    represents how the model expresses the concept, not how we ask about it.

    Args:
        model: The language model
        tokenizer: Model tokenizer
        prompts: Training prompts (definitions + relationships)
        device: Device to run on
        layer_idx: Which layer to extract embeddings from
        max_response_tokens: Max tokens to generate per prompt

    Returns:
        Centroid embedding vector (averaged across all generated responses)
    """
    embeddings = []
    model.eval()

    with torch.no_grad():
        for prompt in prompts:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_response_tokens,
                do_sample=False,  # Deterministic
                pad_token_id=tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            # Get hidden states from the LAST generated token
            # outputs.hidden_states is a tuple of tuples: (step1_layers, step2_layers, ...)
            # We want the last generation step's hidden states
            last_step_hidden = outputs.hidden_states[-1]  # Last generation step
            last_layer_hidden = last_step_hidden[layer_idx]  # Specified layer
            last_token_embedding = last_layer_hidden[0, -1, :]  # [hidden_dim]

            # Convert to float32 before numpy (handles bfloat16)
            embeddings.append(last_token_embedding.float().cpu().numpy())

    # Average all embeddings to get centroid
    centroid = np.mean(embeddings, axis=0)

    # Normalize for cosine similarity
    centroid = centroid / np.linalg.norm(centroid)

    return centroid


def save_concept_centroid(
    concept_name: str,
    centroid: np.ndarray,
    output_path: Path,
):
    """Save concept centroid to disk."""
    np.save(output_path, centroid)


def load_concept_centroid(centroid_path: Path) -> np.ndarray:
    """Load concept centroid from disk."""
    return np.load(centroid_path)


def compute_centroids_for_layer(
    layer: int,
    model,
    tokenizer,
    concepts: List[Dict],
    train_prompts_map: Dict[str, List[str]],
    output_dir: Path,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Compute and save centroids for all concepts in a layer.

    Args:
        layer: Layer number
        model: Language model
        tokenizer: Model tokenizer
        concepts: List of concept dicts
        train_prompts_map: Map of concept_name -> training prompts
        output_dir: Where to save centroids
        device: Device to run on

    Returns:
        Dict mapping concept names to centroids
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    centroids = {}

    print(f"\nComputing embedding centroids for Layer {layer}...")

    for i, concept in enumerate(concepts, 1):
        concept_name = concept['sumo_term']
        prompts = train_prompts_map.get(concept_name, [])

        if not prompts:
            print(f"[{i}/{len(concepts)}] {concept_name}: No prompts, skipping")
            continue

        print(f"[{i}/{len(concepts)}] {concept_name}: Computing from {len(prompts)} prompts...")

        centroid = compute_concept_centroid(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
        )

        # Save centroid
        centroid_path = output_dir / f"{concept_name}_centroid.npy"
        save_concept_centroid(concept_name, centroid, centroid_path)

        centroids[concept_name] = centroid

    # Save metadata
    metadata = {
        'layer': layer,
        'n_concepts': len(centroids),
        'embedding_dim': list(centroids.values())[0].shape[0] if centroids else 0,
    }

    metadata_path = output_dir / "centroids_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved {len(centroids)} centroids to {output_dir}")

    return centroids


__all__ = [
    'compute_concept_centroid',
    'save_concept_centroid',
    'load_concept_centroid',
    'compute_centroids_for_layer',
]
