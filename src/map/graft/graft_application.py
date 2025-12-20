"""
Graft application to substrates.

Applies trained grafts to models, either:
1. As "soft grafts" using hooks (for testing, non-permanent)
2. As "hard grafts" modifying weights directly (permanent)

Per MAP_GRAFTING.md Section 7.

For initial testing, we implement soft grafts using forward hooks.
This allows testing graft effects without modifying model weights.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Callable
from contextlib import contextmanager
import logging
import json

from .data_structures import Graft, SubstrateManifest
from .graft_training import DimensionProjection, ConceptLens

logger = logging.getLogger(__name__)


class GraftedModel(nn.Module):
    """
    Wrapper that applies grafts to a base model using forward hooks.

    This is a "soft graft" approach for initial testing:
    - Grafts are applied via hooks, not by modifying weights
    - Multiple grafts can be stacked
    - Effects can be enabled/disabled dynamically
    - Useful for evaluating graft effects before permanent application

    Usage:
        >>> grafted = GraftedModel(model, tokenizer)
        >>> grafted.load_graft("grafts/Fish/graft.json")
        >>> grafted.enable_graft("graft-Fish-v1")
        >>> output = grafted.generate("Tell me about swimming")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cuda"
    ):
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self.device = device

        # Registry of loaded grafts
        self.grafts: Dict[str, Graft] = {}
        self.projections: Dict[str, DimensionProjection] = {}
        self.lenses: Dict[str, ConceptLens] = {}
        self.biases: Dict[str, torch.Tensor] = {}

        # Which grafts are currently active
        self.active_grafts: set = set()

        # Hook handles
        self._hook_handles: List = []

        # Track last concept activations (for introspection)
        self.last_activations: Dict[str, torch.Tensor] = {}

    def load_graft(self, graft_path: Path) -> str:
        """
        Load a graft from disk.

        Args:
            graft_path: Path to graft JSON file

        Returns:
            graft_id of the loaded graft
        """
        graft_path = Path(graft_path)
        graft = Graft.load(graft_path)
        graft_dir = graft_path.parent

        # Load projection weights
        if graft.injection_points and graft.injection_points[0].projection_path:
            proj_path = Path(graft.injection_points[0].projection_path)
            if not proj_path.is_absolute():
                proj_path = graft_dir / proj_path.name

            hidden_dim = self.base_model.config.hidden_size
            projection = DimensionProjection(hidden_dim).to(self.device)
            projection.load_state_dict(torch.load(proj_path, map_location=self.device, weights_only=True))
            projection.eval()
            self.projections[graft.graft_id] = projection

        # Load lens weights
        if graft.lens_path:
            lens_path = Path(graft.lens_path)
            if not lens_path.is_absolute():
                lens_path = graft_dir / lens_path.name

            if lens_path.exists():
                hidden_dim = self.base_model.config.hidden_size
                lens = ConceptLens(
                    hidden_dim,
                    graft.auxiliary_dimensions
                ).to(self.device)
                lens.load_state_dict(torch.load(lens_path, map_location=self.device, weights_only=True))
                lens.eval()
                self.lenses[graft.graft_id] = lens

        # Load bias
        if graft.substrate_biases:
            bias_spec = graft.substrate_biases[0]
            bias_path = Path(bias_spec.bias_delta_path)
            if not bias_path.is_absolute():
                bias_path = graft_dir / bias_path.name

            if bias_path.exists():
                with open(bias_path, 'r') as f:
                    bias_data = json.load(f)

                # Reconstruct dense bias tensor
                hidden_dim = self.base_model.config.hidden_size
                bias = torch.zeros(hidden_dim, device=self.device)
                if bias_data.get("indices") and bias_data.get("values"):
                    for idx, val in zip(bias_data["indices"], bias_data["values"]):
                        bias[idx] = val
                self.biases[graft.graft_id] = bias

        self.grafts[graft.graft_id] = graft
        logger.info(f"Loaded graft: {graft.graft_id} for concept {graft.concept_id}")

        return graft.graft_id

    def enable_graft(self, graft_id: str, strength: float = 1.0):
        """
        Enable a loaded graft.

        Args:
            graft_id: ID of the graft to enable
            strength: Scaling factor for graft effect (1.0 = full strength)
        """
        if graft_id not in self.grafts:
            raise ValueError(f"Graft not loaded: {graft_id}")

        self.active_grafts.add(graft_id)
        self._update_hooks()

    def disable_graft(self, graft_id: str):
        """Disable a graft."""
        self.active_grafts.discard(graft_id)
        self._update_hooks()

    def disable_all_grafts(self):
        """Disable all grafts."""
        self.active_grafts.clear()
        self._update_hooks()

    def _update_hooks(self):
        """Update forward hooks based on active grafts."""
        # Remove existing hooks
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        if not self.active_grafts:
            return

        # Get target layers for active grafts
        layer_grafts: Dict[int, List[str]] = {}
        for graft_id in self.active_grafts:
            graft = self.grafts[graft_id]
            if graft.injection_points:
                layer = graft.injection_points[0].layer
                if layer not in layer_grafts:
                    layer_grafts[layer] = []
                layer_grafts[layer].append(graft_id)

        # Register hooks for each layer
        layers = self._get_model_layers()

        for layer_idx, graft_ids in layer_grafts.items():
            if layer_idx < len(layers):
                hook_fn = self._create_graft_hook(graft_ids)
                handle = layers[layer_idx].register_forward_hook(hook_fn)
                self._hook_handles.append(handle)

    def _get_model_layers(self) -> List[nn.Module]:
        """Get the list of transformer layers from the model."""
        if hasattr(self.base_model, 'model'):
            model = self.base_model.model
        else:
            model = self.base_model

        if hasattr(model, 'language_model'):
            return list(model.language_model.layers)
        elif hasattr(model, 'layers'):
            return list(model.layers)
        else:
            raise AttributeError(f"Cannot find layers in model: {type(model)}")

    def _create_graft_hook(self, graft_ids: List[str]) -> Callable:
        """Create a forward hook that applies grafts."""

        def hook(module, input, output):
            hidden_states = output[0]

            for graft_id in graft_ids:
                graft = self.grafts[graft_id]

                # Apply projection to compute concept activation
                if graft_id in self.projections:
                    projection = self.projections[graft_id]
                    with torch.no_grad():
                        concept_act = projection(hidden_states)
                        # Store for introspection
                        self.last_activations[graft_id] = concept_act.detach()

                # Apply bias
                if graft_id in self.biases:
                    bias = self.biases[graft_id]
                    hidden_states = hidden_states + bias

            return (hidden_states,) + output[1:]

        return hook

    def get_concept_activation(
        self,
        text: str,
        graft_id: str
    ) -> Dict[str, Any]:
        """
        Get concept activation for a text input.

        Args:
            text: Input text
            graft_id: Graft to measure activation for

        Returns:
            Dict with activation statistics
        """
        if graft_id not in self.grafts:
            raise ValueError(f"Graft not loaded: {graft_id}")

        graft = self.grafts[graft_id]
        projection = self.projections.get(graft_id)

        if projection is None:
            return {"error": "No projection loaded for graft"}

        # Get hidden states at injection layer
        injection_layer = graft.injection_points[0].layer if graft.injection_points else 18

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.base_model(**inputs, output_hidden_states=True)
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states[injection_layer + 1]
            else:
                hidden_states = outputs.last_hidden_state

            # Compute activation
            activation = projection(hidden_states)  # (1, seq_len, 1)

            # Apply lens if available
            lens_score = None
            if graft_id in self.lenses:
                lens = self.lenses[graft_id]
                scores = lens(activation.squeeze(-1), hidden_states)
                mask = inputs.get('attention_mask', torch.ones_like(scores))
                mean_score = (scores * mask).sum() / mask.sum()
                lens_score = torch.sigmoid(mean_score).item()

        return {
            "graft_id": graft_id,
            "concept_id": graft.concept_id,
            "mean_activation": activation.mean().item(),
            "max_activation": activation.max().item(),
            "min_activation": activation.min().item(),
            "std_activation": activation.std().item(),
            "lens_score": lens_score,
            "tokens": len(inputs.input_ids[0])
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> str:
        """
        Generate text with active grafts applied.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional arguments for model.generate()

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @contextmanager
    def graft_context(self, graft_ids: List[str]):
        """
        Context manager for temporarily enabling grafts.

        Example:
            >>> with grafted.graft_context(["graft-Fish-v1"]):
            ...     output = grafted.generate("Tell me about...")
        """
        # Save current state
        previously_active = self.active_grafts.copy()

        # Enable requested grafts
        self.active_grafts = set(graft_ids)
        self._update_hooks()

        try:
            yield
        finally:
            # Restore previous state
            self.active_grafts = previously_active
            self._update_hooks()

    def forward(self, *args, **kwargs):
        """Forward pass through base model with grafts applied."""
        return self.base_model(*args, **kwargs)


def apply_graft(
    model: nn.Module,
    manifest: SubstrateManifest,
    graft: Graft,
    mode: str = "soft",
    device: str = "cuda"
) -> Tuple[nn.Module, SubstrateManifest]:
    """
    Apply a graft to a substrate.

    Args:
        model: The substrate model
        manifest: Current substrate manifest
        graft: The graft to apply
        mode: "soft" (hooks, reversible) or "hard" (modify weights, permanent)
        device: Device to use

    Returns:
        Tuple of (modified_model, updated_manifest)
    """
    # Validate substrate compatibility
    hidden_dim = model.config.hidden_size
    if graft.pre_graft_dim != hidden_dim:
        logger.warning(
            f"Graft expects substrate dim {graft.pre_graft_dim}, "
            f"got {hidden_dim}. Proceeding anyway for testing."
        )

    if mode == "soft":
        # Soft graft: use hooks (already implemented via GraftedModel)
        logger.info(f"Soft graft mode - use GraftedModel to apply graft")
        # Return model unchanged, manifest records intent
        manifest.add_graft(graft)
        return model, manifest

    elif mode == "hard":
        # Hard graft: modify model weights directly
        # This is a more involved operation - for now, just add biases

        if graft.substrate_biases:
            bias_spec = graft.substrate_biases[0]
            bias_path = Path(bias_spec.bias_delta_path)

            if bias_path.exists():
                with open(bias_path, 'r') as f:
                    bias_data = json.load(f)

                # Get target layer
                layer_idx = bias_spec.layer
                layers = _get_model_layers(model)

                if layer_idx < len(layers):
                    layer = layers[layer_idx]

                    # Find the mlp/dense layer to modify
                    # This varies by architecture - simplified example
                    if hasattr(layer, 'mlp'):
                        if hasattr(layer.mlp, 'up_proj'):
                            # Apply bias to up_proj
                            target = layer.mlp.up_proj
                        elif hasattr(layer.mlp, 'dense_h_to_4h'):
                            target = layer.mlp.dense_h_to_4h
                        else:
                            target = None

                        if target is not None and hasattr(target, 'bias'):
                            if target.bias is not None:
                                # Add to existing bias
                                with torch.no_grad():
                                    for idx, val in zip(
                                        bias_data.get("indices", []),
                                        bias_data.get("values", [])
                                    ):
                                        if idx < len(target.bias):
                                            target.bias[idx] += val
                                logger.info(f"Applied hard bias to layer {layer_idx}")

        manifest.add_graft(graft)
        return model, manifest

    else:
        raise ValueError(f"Unknown graft mode: {mode}")


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
    else:
        return []


def validate_graft(
    model: nn.Module,
    graft: Graft,
    validation_data: Optional[Dict[str, List[str]]] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Validate a graft before application.

    Checks:
    1. Substrate compatibility
    2. Dimension activation (if validation_data provided)
    3. Lens F1 (if validation_data provided)
    4. Bias sparsity

    Args:
        model: The substrate model
        graft: The graft to validate
        validation_data: Dict with "positive" and "negative" texts
        device: Device to use

    Returns:
        Validation results dict
    """
    results = {
        "graft_id": graft.graft_id,
        "passed": True,
        "tests": []
    }

    # Check substrate compatibility
    hidden_dim = model.config.hidden_size
    dim_match = hidden_dim == graft.pre_graft_dim
    results["tests"].append({
        "name": "substrate_compatibility",
        "passed": dim_match,
        "expected": graft.pre_graft_dim,
        "actual": hidden_dim
    })
    if not dim_match:
        results["passed"] = False

    # Check bias sparsity
    if graft.substrate_biases:
        bias = graft.substrate_biases[0]
        sparsity = 1.0 - (bias.nnz / max(np.prod(bias.shape), 1))
        sparsity_pass = sparsity >= 0.90
        results["tests"].append({
            "name": "bias_sparsity",
            "passed": sparsity_pass,
            "threshold": 0.90,
            "actual": sparsity
        })
        if not sparsity_pass:
            results["passed"] = False

    # Check metrics from training
    if graft.metrics:
        acc = graft.metrics.get("final_accuracy", 0)
        acc_pass = acc >= 0.80
        results["tests"].append({
            "name": "training_accuracy",
            "passed": acc_pass,
            "threshold": 0.80,
            "actual": acc
        })
        if not acc_pass:
            results["passed"] = False

    return results


def compare_grafts(
    graft_a: Graft,
    graft_b: Graft
) -> Dict[str, Any]:
    """
    Compare two grafts for overlap analysis.

    Used to detect when grafts might need cotraining.

    Args:
        graft_a: First graft
        graft_b: Second graft

    Returns:
        Comparison metrics
    """
    # Compare auxiliary dimensions
    aux_a = set(graft_a.auxiliary_dimensions)
    aux_b = set(graft_b.auxiliary_dimensions)

    intersection = aux_a & aux_b
    union = aux_a | aux_b

    jaccard = len(intersection) / len(union) if union else 0.0

    return {
        "graft_a": graft_a.graft_id,
        "graft_b": graft_b.graft_id,
        "concept_a": graft_a.concept_id,
        "concept_b": graft_b.concept_id,
        "auxiliary_overlap": {
            "jaccard_index": jaccard,
            "overlapping_dims": sorted(list(intersection)),
            "overlap_count": len(intersection),
            "union_count": len(union)
        },
        "recommendation": "cotrain" if jaccard > 0.6 else "independent"
    }
