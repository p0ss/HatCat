"""
Region derivation from lens weights.

Identifies which substrate dimensions correlate with a concept by analyzing
the trained lens's weight patterns. These dimensions become:
1. Auxiliary dimensions for the graft's lens
2. Targets for where biases should land during graft training

Per MAP_GRAFTING.md Section 4.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from .data_structures import ConceptRegion, LayerMask

logger = logging.getLogger(__name__)


def derive_region_from_lens(
    lens_path: Path,
    concept_id: str,
    layers: List[int],
    top_k_percent: float = 15.0,
    hidden_dim: Optional[int] = None,
    include_ancestors: bool = False,
    ancestor_lenses: Optional[Dict[str, Path]] = None,
    ancestor_weight_decay: float = 0.5
) -> ConceptRegion:
    """
    Derive a ConceptRegion from lens weights.

    For each layer, takes the top k% of dimensions by |weight|.
    These become auxiliary dimensions for the graft's lens.

    Args:
        lens_path: Path to trained lens (.pt file)
        concept_id: Identifier for the concept
        layers: Layer indices to analyze
        top_k_percent: Percentage of dimensions to include (default 15%)
        hidden_dim: Override hidden dimension if lens doesn't store it
        include_ancestors: Include ancestor lens weights (weighted)
        ancestor_lenses: Dict mapping ancestor concept_id to lens paths
        ancestor_weight_decay: Decay factor per ancestor depth

    Returns:
        ConceptRegion with identified dimensions for each layer

    Example:
        >>> region = derive_region_from_lens(
        ...     lens_path=Path("lenses/Fish_classifier.pt"),
        ...     concept_id="concept/Fish",
        ...     layers=[18, 20, 22],
        ...     top_k_percent=15.0
        ... )
        >>> print(f"Layer 18 has {len(region.layers[0].indices)} important dims")
    """
    # Load lens
    lens_state = torch.load(lens_path, map_location='cpu', weights_only=True)

    # Extract first linear layer weights (maps from hidden_dim to intermediate)
    # BinaryClassifier architecture: Linear(hidden_dim, 128) -> ReLU -> Dropout -> Linear(128, 1)
    if 'net.0.weight' in lens_state:
        weights = lens_state['net.0.weight']  # Shape: (128, hidden_dim)
    else:
        # Try alternative key patterns
        for key in lens_state.keys():
            if 'weight' in key and lens_state[key].dim() == 2:
                weights = lens_state[key]
                break
        else:
            raise ValueError(f"Could not find linear weights in lens state dict: {list(lens_state.keys())}")

    # Get importance per dimension (sum of absolute weights across intermediate units)
    importance = torch.abs(weights).sum(dim=0).numpy()  # Shape: (hidden_dim,)

    if hidden_dim is None:
        hidden_dim = len(importance)

    # Optionally include ancestor weights
    if include_ancestors and ancestor_lenses:
        for ancestor_id, ancestor_path in ancestor_lenses.items():
            depth = ancestor_lenses.get(f"{ancestor_id}_depth", 1)
            weight = ancestor_weight_decay ** depth

            try:
                ancestor_state = torch.load(ancestor_path, map_location='cpu', weights_only=True)
                if 'net.0.weight' in ancestor_state:
                    ancestor_weights = ancestor_state['net.0.weight']
                    ancestor_importance = torch.abs(ancestor_weights).sum(dim=0).numpy()
                    importance = importance + weight * ancestor_importance
            except Exception as e:
                logger.warning(f"Could not load ancestor lens {ancestor_path}: {e}")

    # Compute top-k indices
    k = int(len(importance) * top_k_percent / 100)
    k = max(1, k)  # At least 1 dimension

    top_indices = np.argsort(importance)[-k:].tolist()
    top_indices.sort()  # Keep sorted for consistency

    # For simplified implementation, use same indices across all specified layers
    # (In full implementation, would analyze per-layer activations)
    region_layers = []
    for layer_idx in layers:
        region_layers.append(LayerMask(
            layer_index=layer_idx,
            component="residual",  # We read from residual stream
            indices=top_indices,
            total_dimensions=hidden_dim
        ))

    region_id = f"region-{concept_id.replace('/', '-')}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return ConceptRegion(
        region_id=region_id,
        concept_id=concept_id,
        layers=region_layers,
        derivation={
            "method": "lens_weight_topk",
            "parameters": {
                "top_k_percent": top_k_percent,
                "layers": layers,
                "include_ancestors": include_ancestors
            }
        },
        source_lens_path=str(lens_path)
    )


def derive_regions_from_lens_pack(
    lens_pack_path: Path,
    concept_ids: List[str],
    layer: int,
    top_k_percent: float = 15.0,
    output_dir: Optional[Path] = None
) -> Dict[str, ConceptRegion]:
    """
    Derive ConceptRegions for multiple concepts from a lens pack.

    Args:
        lens_pack_path: Path to lens pack directory
        concept_ids: List of concept IDs to process
        layer: Layer index (lens pack is organized by layer)
        top_k_percent: Percentage of dimensions to include
        output_dir: Optional directory to save regions

    Returns:
        Dict mapping concept_id to ConceptRegion
    """
    layer_dir = lens_pack_path / f"layer{layer}"
    if not layer_dir.exists():
        raise ValueError(f"Layer directory not found: {layer_dir}")

    regions = {}

    for concept_id in concept_ids:
        lens_name = f"{concept_id}_classifier.pt"
        lens_path = layer_dir / lens_name

        if not lens_path.exists():
            logger.warning(f"Lens not found: {lens_path}")
            continue

        try:
            region = derive_region_from_lens(
                lens_path=lens_path,
                concept_id=concept_id,
                layers=[layer],
                top_k_percent=top_k_percent
            )
            regions[concept_id] = region

            if output_dir:
                region_path = output_dir / f"{concept_id}_region.json"
                region.save(region_path)

        except Exception as e:
            logger.error(f"Failed to derive region for {concept_id}: {e}")

    return regions


def analyze_region_overlap(
    region_a: ConceptRegion,
    region_b: ConceptRegion,
    layer_index: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze overlap between two concept regions.

    Used to detect when concepts share dimensions and may need cotraining.

    Args:
        region_a: First concept region
        region_b: Second concept region
        layer_index: Specific layer to analyze (None = all layers)

    Returns:
        Dict with overlap metrics including jaccard index
    """
    indices_a = set(region_a.get_all_indices(layer_index))
    indices_b = set(region_b.get_all_indices(layer_index))

    intersection = indices_a & indices_b
    union = indices_a | indices_b

    jaccard = len(intersection) / len(union) if union else 0.0

    return {
        "concept_a": region_a.concept_id,
        "concept_b": region_b.concept_id,
        "layer_index": layer_index,
        "jaccard_index": jaccard,
        "overlapping_dimensions": sorted(list(intersection)),
        "total_a": len(indices_a),
        "total_b": len(indices_b),
        "overlap_count": len(intersection),
        "union_count": len(union)
    }


def get_lens_weight_magnitudes(lens_path: Path) -> Dict[str, float]:
    """
    Get magnitude statistics from lens weights.

    Useful for understanding lens quality and comparing across concepts.

    Args:
        lens_path: Path to trained lens

    Returns:
        Dict with weight statistics (mean, max, l2_norm, sparsity)
    """
    lens_state = torch.load(lens_path, map_location='cpu', weights_only=True)

    if 'net.0.weight' in lens_state:
        weights = lens_state['net.0.weight']
    else:
        for key in lens_state.keys():
            if 'weight' in key and lens_state[key].dim() == 2:
                weights = lens_state[key]
                break
        else:
            return {}

    abs_weights = torch.abs(weights)
    importance = abs_weights.sum(dim=0)

    return {
        "mean": float(importance.mean()),
        "max": float(importance.max()),
        "min": float(importance.min()),
        "std": float(importance.std()),
        "l2_norm": float(torch.norm(importance)),
        "sparsity": float((importance < 0.01).sum() / len(importance))
    }
