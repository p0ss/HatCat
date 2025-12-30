"""
Cleft identification and management.

A Cleft is the region of model weights associated with a concept, derived from
lens weight analysis. During scion training:

1. Experience data has concept tags (from lenses that fired during experience)
2. Each tagged concept maps to a cleft (the weights that concept's lens reads from)
3. We train ONLY those clefts - the union of regions for all tagged concepts
4. Everything else stays frozen (model can keep running)

After training, the scion's biases encode how much each feature changed,
proportional to its contribution to learning the new concept.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CleftRegion:
    """
    A region of model weights associated with a concept.

    Derived from lens weight analysis - identifies which parameters
    in which layers are implicated in detecting a concept.
    """
    concept_id: str
    layer_index: int
    component: str  # "mlp.up_proj", "mlp.down_proj", "attn.q_proj", etc.

    # Which dimensions/indices in this weight matrix are important
    # For a weight matrix W[out_features, in_features]:
    #   row_indices: which output features (rows) are implicated
    #   col_indices: which input features (cols) are implicated
    row_indices: List[int] = field(default_factory=list)
    col_indices: List[int] = field(default_factory=list)

    # Importance scores for each index (from lens weight magnitudes)
    row_importance: Optional[List[float]] = None
    col_importance: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        return {
            "concept_id": self.concept_id,
            "layer_index": self.layer_index,
            "component": self.component,
            "row_indices": self.row_indices,
            "col_indices": self.col_indices,
            "row_importance": self.row_importance,
            "col_importance": self.col_importance
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CleftRegion":
        return cls(**d)


@dataclass
class Cleft:
    """
    Complete cleft for a concept - all regions across all layers.

    The cleft defines exactly which model parameters are "owned" by
    this concept and should be considered during scion training.
    """
    concept_id: str
    regions: List[CleftRegion] = field(default_factory=list)
    source_lens_path: Optional[str] = None

    # Metadata
    hidden_dim: int = 0
    total_parameters: int = 0  # Count of parameters in this cleft

    def get_regions_for_layer(self, layer_index: int) -> List[CleftRegion]:
        """Get all regions in a specific layer."""
        return [r for r in self.regions if r.layer_index == layer_index]

    def get_all_layers(self) -> Set[int]:
        """Get set of all layer indices with regions."""
        return {r.layer_index for r in self.regions}

    def to_dict(self) -> Dict:
        return {
            "concept_id": self.concept_id,
            "regions": [r.to_dict() for r in self.regions],
            "source_lens_path": self.source_lens_path,
            "hidden_dim": self.hidden_dim,
            "total_parameters": self.total_parameters
        }


def derive_cleft_from_lens(
    lens_path: Path,
    concept_id: str,
    model: nn.Module,
    layers: List[int],
    top_k_percent: float = 15.0,
    components: List[str] = None
) -> Cleft:
    """
    Derive a Cleft from a trained lens's weights.

    The lens's first linear layer maps hidden_dim -> intermediate_dim.
    The magnitude of weights tells us which hidden dimensions are important
    for detecting this concept.

    We then trace those dimensions back to the model components that
    produce them (MLP projections, attention projections, etc.)

    Args:
        lens_path: Path to trained lens (.pt file)
        concept_id: Identifier for the concept
        model: The substrate model (to understand architecture)
        layers: Which layers to analyze
        top_k_percent: Top percentage of dimensions to include
        components: Which components to include (default: mlp projections)

    Returns:
        Cleft identifying the regions associated with this concept
    """
    if components is None:
        components = ["mlp.up_proj", "mlp.down_proj"]

    # Load lens weights
    lens_state = torch.load(lens_path, map_location='cpu', weights_only=True)

    # Get first linear layer weights: (intermediate_dim, hidden_dim)
    if 'net.0.weight' in lens_state:
        weights = lens_state['net.0.weight']
    else:
        for key in lens_state.keys():
            if 'weight' in key and lens_state[key].dim() == 2:
                weights = lens_state[key]
                break
        else:
            raise ValueError(f"Could not find linear weights in lens: {lens_path}")

    # Compute importance per hidden dimension
    # Sum of absolute weights across intermediate units
    importance = torch.abs(weights).sum(dim=0).numpy()  # (hidden_dim,)
    hidden_dim = len(importance)

    # Get top-k indices
    k = max(1, int(hidden_dim * top_k_percent / 100))
    top_indices = np.argsort(importance)[-k:].tolist()
    top_importance = importance[top_indices].tolist()

    # Build cleft regions for each layer/component
    regions = []
    total_params = 0

    for layer_idx in layers:
        layer = _get_layer(model, layer_idx)
        if layer is None:
            continue

        for component_name in components:
            component = _get_component(layer, component_name)
            if component is None:
                continue

            # For linear layers, determine which indices matter
            # W[out_features, in_features] @ x[in_features] -> y[out_features]
            weight_shape = component.weight.shape

            if "up_proj" in component_name or "dense_h_to_4h" in component_name:
                # Projects from hidden_dim to intermediate
                # Input is hidden_dim, so col_indices are the important hidden dims
                col_indices = [i for i in top_indices if i < weight_shape[1]]
                row_indices = list(range(weight_shape[0]))  # All outputs affected
                col_importance = [importance[i] for i in col_indices]
                row_importance = None

            elif "down_proj" in component_name or "dense_4h_to_h" in component_name:
                # Projects from intermediate back to hidden_dim
                # Output is hidden_dim, so row_indices are the important hidden dims
                row_indices = [i for i in top_indices if i < weight_shape[0]]
                col_indices = list(range(weight_shape[1]))  # All inputs contribute
                row_importance = [importance[i] for i in row_indices]
                col_importance = None

            else:
                # Generic: assume hidden_dim on both sides
                row_indices = [i for i in top_indices if i < weight_shape[0]]
                col_indices = [i for i in top_indices if i < weight_shape[1]]
                row_importance = [importance[i] for i in row_indices] if row_indices else None
                col_importance = [importance[i] for i in col_indices] if col_indices else None

            if row_indices or col_indices:
                region = CleftRegion(
                    concept_id=concept_id,
                    layer_index=layer_idx,
                    component=component_name,
                    row_indices=row_indices,
                    col_indices=col_indices,
                    row_importance=row_importance,
                    col_importance=col_importance
                )
                regions.append(region)
                total_params += len(row_indices) * len(col_indices)

    return Cleft(
        concept_id=concept_id,
        regions=regions,
        source_lens_path=str(lens_path),
        hidden_dim=hidden_dim,
        total_parameters=total_params
    )


def merge_clefts(clefts: List[Cleft]) -> "UnionCleft":
    """
    Merge multiple clefts into a union cleft.

    Used when experience data has multiple concept tags - we need to
    train the union of all their clefts.

    Args:
        clefts: List of clefts to merge

    Returns:
        UnionCleft representing the union of all regions
    """
    return UnionCleft(clefts)


@dataclass
class UnionCleft:
    """
    Union of multiple concept clefts.

    During scion training, this defines exactly which parameters are trainable.
    """
    source_clefts: List[Cleft]

    def __post_init__(self):
        # Build merged index structures for efficient lookup
        self._build_indices()

    def _build_indices(self):
        """Build merged indices for each layer/component."""
        self._layer_component_indices: Dict[Tuple[int, str], Tuple[Set[int], Set[int]]] = {}

        for cleft in self.source_clefts:
            for region in cleft.regions:
                key = (region.layer_index, region.component)
                if key not in self._layer_component_indices:
                    self._layer_component_indices[key] = (set(), set())

                rows, cols = self._layer_component_indices[key]
                rows.update(region.row_indices)
                cols.update(region.col_indices)

    def get_trainable_mask(
        self,
        layer_index: int,
        component: str,
        weight_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Get a boolean mask indicating which parameters are trainable.

        Args:
            layer_index: Which layer
            component: Which component (e.g., "mlp.up_proj")
            weight_shape: Shape of the weight matrix (out_features, in_features)

        Returns:
            Boolean tensor of same shape, True where trainable
        """
        key = (layer_index, component)

        if key not in self._layer_component_indices:
            # No cleft touches this component - all frozen
            return torch.zeros(weight_shape, dtype=torch.bool)

        rows, cols = self._layer_component_indices[key]

        # Create mask
        mask = torch.zeros(weight_shape, dtype=torch.bool)

        # Mark trainable cells
        for r in rows:
            if r < weight_shape[0]:
                for c in cols:
                    if c < weight_shape[1]:
                        mask[r, c] = True

        return mask

    def get_all_layers(self) -> Set[int]:
        """Get all layer indices that have trainable regions."""
        return {key[0] for key in self._layer_component_indices.keys()}

    def get_components_for_layer(self, layer_index: int) -> List[str]:
        """Get components that are trainable in a layer."""
        return [key[1] for key in self._layer_component_indices.keys()
                if key[0] == layer_index]

    @property
    def concept_ids(self) -> List[str]:
        """Get list of all concept IDs in this union."""
        return [c.concept_id for c in self.source_clefts]


class CleftAwareFreezer:
    """
    Manages parameter freezing based on cleft regions.

    During scion training:
    - Parameters in the union cleft are trainable (gradient flows)
    - Parameters outside the cleft are frozen (no gradient)

    This is implemented via gradient hooks that zero out gradients
    for frozen parameters, allowing the forward pass to work normally.
    """

    def __init__(self, model: nn.Module, union_cleft: UnionCleft):
        self.model = model
        self.union_cleft = union_cleft
        self._hooks = []
        self._masks: Dict[str, torch.Tensor] = {}

    def freeze(self):
        """
        Apply freezing to the model.

        Registers gradient hooks that zero out gradients for frozen params.
        """
        self._hooks = []

        for layer_idx in range(self._get_num_layers()):
            layer = _get_layer(self.model, layer_idx)
            if layer is None:
                continue

            for component_name in self.union_cleft.get_components_for_layer(layer_idx):
                component = _get_component(layer, component_name)
                if component is None or not hasattr(component, 'weight'):
                    continue

                # Get trainable mask
                mask = self.union_cleft.get_trainable_mask(
                    layer_idx,
                    component_name,
                    component.weight.shape
                )

                # Store mask and register hook
                param_name = f"layer{layer_idx}.{component_name}"
                self._masks[param_name] = mask.to(component.weight.device)

                # Hook that masks gradients
                def make_hook(m):
                    def hook(grad):
                        return grad * m.float()
                    return hook

                handle = component.weight.register_hook(make_hook(self._masks[param_name]))
                self._hooks.append(handle)

        logger.info(f"Applied cleft freezing: {len(self._hooks)} gradient masks active")

    def unfreeze(self):
        """Remove all gradient hooks, unfreezing the model."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._masks = {}
        logger.info("Removed cleft freezing")

    def _get_num_layers(self) -> int:
        """Get number of transformer layers."""
        layers = _get_model_layers(self.model)
        return len(layers) if layers else 0

    def get_trainable_param_count(self) -> int:
        """Count trainable parameters (in the cleft)."""
        total = 0
        for mask in self._masks.values():
            total += mask.sum().item()
        return int(total)

    def get_frozen_param_count(self) -> int:
        """Count frozen parameters (outside the cleft)."""
        total = 0
        for mask in self._masks.values():
            total += (~mask).sum().item()
        return int(total)


def _get_model_layers(model: nn.Module) -> List[nn.Module]:
    """Get transformer layers from model."""
    if hasattr(model, 'model'):
        m = model.model
    else:
        m = model

    if hasattr(m, 'language_model'):
        return list(m.language_model.layers)
    elif hasattr(m, 'layers'):
        return list(m.layers)
    return []


def _get_layer(model: nn.Module, layer_idx: int) -> Optional[nn.Module]:
    """Get a specific transformer layer."""
    layers = _get_model_layers(model)
    if layer_idx < len(layers):
        return layers[layer_idx]
    return None


def _get_component(layer: nn.Module, component_path: str) -> Optional[nn.Module]:
    """
    Get a component from a layer by dot-separated path.

    E.g., "mlp.up_proj" -> layer.mlp.up_proj
    """
    parts = component_path.split(".")
    current = layer

    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            # Try alternative names
            alternatives = {
                "up_proj": ["dense_h_to_4h", "fc1", "w1"],
                "down_proj": ["dense_4h_to_h", "fc2", "w2"],
                "gate_proj": ["w3"],
            }
            found = False
            for alt in alternatives.get(part, []):
                if hasattr(current, alt):
                    current = getattr(current, alt)
                    found = True
                    break
            if not found:
                return None

    return current
