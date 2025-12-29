#!/usr/bin/env python3
"""
Shared Activation Cache for Calibration Cycle

Caches activations from concept prompts so they can be reused across:
- Calibration analysis
- Fine-tuning
- Cross-activation calibration

This avoids redundant forward passes through the model.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


@dataclass
class ActivationCache:
    """Cache of activations from concept prompts."""

    # Map from (concept, layer) -> list of activation tensors (one per prompt)
    activations: Dict[Tuple[str, int], List[torch.Tensor]] = field(default_factory=dict)

    # Metadata
    model_name: str = ""
    model_layer: int = 15
    hidden_dim: int = 0
    created_at: str = ""

    # Stats
    n_concepts: int = 0
    n_activations: int = 0

    def get(self, concept: str, layer: int) -> Optional[List[torch.Tensor]]:
        """Get cached activations for a concept."""
        return self.activations.get((concept, layer))

    def put(self, concept: str, layer: int, activations: List[torch.Tensor]):
        """Store activations for a concept."""
        self.activations[(concept, layer)] = activations
        self.n_concepts = len(self.activations)
        self.n_activations = sum(len(v) for v in self.activations.values())

    def has(self, concept: str, layer: int) -> bool:
        """Check if concept is cached."""
        return (concept, layer) in self.activations


def get_cache_path(lens_pack_dir: Path, model_layer: int) -> Path:
    """Get the cache file path."""
    cache_dir = lens_pack_dir / "activation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"concept_prompts_L{model_layer}.pt"


def load_activation_cache(
    lens_pack_dir: Path,
    model_layer: int,
    model_name: str = "",
) -> Optional[ActivationCache]:
    """Load cached activations if they exist and are valid."""
    cache_path = get_cache_path(lens_pack_dir, model_layer)
    meta_path = cache_path.with_suffix('.json')

    if not cache_path.exists() or not meta_path.exists():
        return None

    try:
        # Load metadata
        with open(meta_path) as f:
            meta = json.load(f)

        # Check if model matches (optional - warn if different)
        if model_name and meta.get("model_name") and meta["model_name"] != model_name:
            print(f"  Warning: Cache was created with different model")
            print(f"    Cache: {meta['model_name']}")
            print(f"    Current: {model_name}")

        # Load activations
        data = torch.load(cache_path, map_location='cpu', weights_only=True)

        # Reconstruct cache
        cache = ActivationCache(
            activations={},
            model_name=meta.get("model_name", ""),
            model_layer=meta.get("model_layer", model_layer),
            hidden_dim=meta.get("hidden_dim", 0),
            created_at=meta.get("created_at", ""),
            n_concepts=meta.get("n_concepts", 0),
            n_activations=meta.get("n_activations", 0),
        )

        # Convert stored format back to tuple keys
        for key_str, acts in data.items():
            # Key format: "ConceptName_L0"
            parts = key_str.rsplit("_L", 1)
            if len(parts) == 2:
                concept = parts[0]
                layer = int(parts[1])
                cache.activations[(concept, layer)] = acts

        print(f"  ✓ Loaded activation cache: {cache.n_concepts} concepts, {cache.n_activations} activations")
        print(f"    Created: {cache.created_at}")

        return cache

    except Exception as e:
        print(f"  ✗ Failed to load cache: {e}")
        return None


def save_activation_cache(
    cache: ActivationCache,
    lens_pack_dir: Path,
) -> None:
    """Save activation cache to disk."""
    cache_path = get_cache_path(lens_pack_dir, cache.model_layer)
    meta_path = cache_path.with_suffix('.json')

    # Convert to storable format (string keys)
    data = {}
    for (concept, layer), acts in cache.activations.items():
        key_str = f"{concept}_L{layer}"
        data[key_str] = acts

    # Save activations
    torch.save(data, cache_path)

    # Save metadata
    meta = {
        "created_at": datetime.now().isoformat(),
        "model_name": cache.model_name,
        "model_layer": cache.model_layer,
        "hidden_dim": cache.hidden_dim,
        "n_concepts": cache.n_concepts,
        "n_activations": cache.n_activations,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Saved activation cache: {cache_path}")
    print(f"    Concepts: {cache.n_concepts}, Activations: {cache.n_activations}")


def build_activation_cache(
    concept_pack_dir: Path,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    model_layer: int = 15,
    model_name: str = "",
    n_samples_per_concept: int = 5,
    max_concepts: Optional[int] = None,
    normalize: bool = True,
) -> ActivationCache:
    """
    Build activation cache by running model on concept prompts.

    Args:
        concept_pack_dir: Path to concept pack
        model: The language model
        tokenizer: The tokenizer
        device: Device to run on
        layers: Concept layers to include
        model_layer: Which model layer to extract activations from
        model_name: Model name for metadata
        n_samples_per_concept: Max prompts per concept
        max_concepts: Limit total concepts (for testing)
        normalize: Whether to apply LayerNorm

    Returns:
        ActivationCache with all activations
    """
    print(f"\nBuilding activation cache...")
    print(f"  Model layer: {model_layer}")
    print(f"  Samples per concept: {n_samples_per_concept}")

    # Load concept definitions
    definitions = {}
    for layer in layers:
        layer_file = concept_pack_dir / "hierarchy" / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        for concept in layer_data.get('concepts', []):
            term = concept.get('sumo_term') or concept.get('term')
            if not term:
                continue

            defs = []
            definition = concept.get('definition', '')
            if definition and len(definition) > 10:
                defs.append(definition)
            sumo_def = concept.get('sumo_definition', '')
            if sumo_def and len(sumo_def) > 10 and sumo_def != definition:
                defs.append(sumo_def)
            for lemma in concept.get('lemmas', [])[:3]:
                if lemma and len(lemma) > 3:
                    defs.append(f"A type of {lemma}")

            if defs:
                definitions[(term, layer)] = defs[:n_samples_per_concept]

    concepts_to_cache = list(definitions.keys())
    if max_concepts and len(concepts_to_cache) > max_concepts:
        concepts_to_cache = concepts_to_cache[:max_concepts]

    print(f"  Concepts to cache: {len(concepts_to_cache)}")

    # Build cache
    cache = ActivationCache(
        model_name=model_name,
        model_layer=model_layer,
    )

    layer_norm = None
    model.eval()

    for concept_name, layer in tqdm(concepts_to_cache, desc="Caching activations"):
        prompts = definitions.get((concept_name, layer), [])
        activations = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[model_layer]
                activation = hidden_states[0, -1, :].float()

                # Normalize if requested
                if normalize:
                    if layer_norm is None:
                        layer_norm = torch.nn.LayerNorm(
                            activation.shape[-1], elementwise_affine=False
                        ).to(device)
                    activation = layer_norm(activation.unsqueeze(0)).squeeze(0)

                # Store on CPU
                activations.append(activation.cpu())

                if cache.hidden_dim == 0:
                    cache.hidden_dim = activation.shape[-1]

        if activations:
            cache.put(concept_name, layer, activations)

    cache.created_at = datetime.now().isoformat()
    print(f"  Built cache: {cache.n_concepts} concepts, {cache.n_activations} activations")

    return cache


def get_or_build_cache(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    model_layer: int = 15,
    model_name: str = "",
    n_samples_per_concept: int = 5,
    max_concepts: Optional[int] = None,
    force_rebuild: bool = False,
) -> ActivationCache:
    """
    Load existing cache or build a new one.

    This is the main entry point for the calibration cycle.
    """
    if not force_rebuild:
        cache = load_activation_cache(lens_pack_dir, model_layer, model_name)
        if cache is not None:
            return cache

    # Build new cache
    cache = build_activation_cache(
        concept_pack_dir=concept_pack_dir,
        model=model,
        tokenizer=tokenizer,
        device=device,
        layers=layers,
        model_layer=model_layer,
        model_name=model_name,
        n_samples_per_concept=n_samples_per_concept,
        max_concepts=max_concepts,
    )

    # Save for future use
    save_activation_cache(cache, lens_pack_dir)

    return cache
