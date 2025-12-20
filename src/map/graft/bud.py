"""
Bud - soft/temporary grafts using hooks.

A Bud is a reversible graft that uses forward hooks to modify model behavior
without changing weights. Useful for:
- Testing graft effects before committing to a scion
- A/B testing different concept implementations
- Temporary concept activation for specific tasks

Buds do not accrete - they are ephemeral and can be enabled/disabled at will.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import json

from .cleft import Cleft, _get_layer, _get_model_layers
from .scion import Scion

logger = logging.getLogger(__name__)


@dataclass
class Bud:
    """
    A soft/temporary graft that modifies behavior via hooks.

    Can be created from:
    - A Scion (to test before permanent application)
    - A Cleft + direction vector (for steering-style modifications)
    - Raw bias vectors (for direct manipulation)
    """
    bud_id: str
    concept_id: str

    # The modification to apply (added to hidden states)
    bias_vectors: Dict[int, torch.Tensor]  # layer_idx -> bias vector

    # Hook configuration
    injection_layers: List[int] = field(default_factory=list)
    strength: float = 1.0  # Scaling factor for the bias

    # Source
    source_type: str = "direct"  # "scion", "cleft", "direct"
    source_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "bud_id": self.bud_id,
            "concept_id": self.concept_id,
            "injection_layers": self.injection_layers,
            "strength": self.strength,
            "source_type": self.source_type,
            "source_id": self.source_id
        }

    @classmethod
    def from_scion(cls, scion: Scion, layers: Optional[List[int]] = None) -> "Bud":
        """
        Create a bud from a scion for testing.

        The bud applies the scion's learned biases as additive modifications
        to the hidden states, without permanently changing weights.
        """
        if layers is None:
            layers = scion.training_config.injection_layers

        # Convert neuron biases to layer biases
        bias_vectors = {}
        for layer_idx in layers:
            # Aggregate biases relevant to this layer
            layer_bias = None
            for key, bias in scion.neuron_biases.items():
                if f"layer{layer_idx}" in key and "_col" in key:
                    # Column biases represent input feature importance
                    if layer_bias is None:
                        layer_bias = bias.clone()
                    else:
                        # If dimensions match, add; otherwise skip
                        if layer_bias.shape == bias.shape:
                            layer_bias = layer_bias + bias

            if layer_bias is not None:
                bias_vectors[layer_idx] = layer_bias

        return cls(
            bud_id=f"bud-{scion.concept_id}-from-scion",
            concept_id=scion.concept_id,
            bias_vectors=bias_vectors,
            injection_layers=layers,
            source_type="scion",
            source_id=scion.scion_id
        )

    @classmethod
    def from_direction(
        cls,
        concept_id: str,
        direction: torch.Tensor,
        layers: List[int],
        strength: float = 1.0
    ) -> "Bud":
        """
        Create a bud from a steering direction vector.

        Useful for activation steering experiments.
        """
        bias_vectors = {layer: direction.clone() for layer in layers}

        return cls(
            bud_id=f"bud-{concept_id}-direction",
            concept_id=concept_id,
            bias_vectors=bias_vectors,
            injection_layers=layers,
            strength=strength,
            source_type="direction"
        )


class BuddedModel(nn.Module):
    """
    Wrapper that applies buds to a model using forward hooks.

    Multiple buds can be active simultaneously, and their effects stack.
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

        # Registry of loaded buds
        self.buds: Dict[str, Bud] = {}

        # Active buds and their strengths
        self.active_buds: Dict[str, float] = {}  # bud_id -> strength

        # Hook handles
        self._hook_handles: List = []

    def add_bud(self, bud: Bud) -> str:
        """Add a bud to the model (but don't activate it yet)."""
        # Move bias vectors to device
        bud.bias_vectors = {
            layer: bias.to(self.device)
            for layer, bias in bud.bias_vectors.items()
        }
        self.buds[bud.bud_id] = bud
        logger.info(f"Added bud: {bud.bud_id}")
        return bud.bud_id

    def activate_bud(self, bud_id: str, strength: float = 1.0):
        """Activate a bud with optional strength scaling."""
        if bud_id not in self.buds:
            raise ValueError(f"Bud not found: {bud_id}")

        self.active_buds[bud_id] = strength
        self._update_hooks()
        logger.info(f"Activated bud {bud_id} with strength {strength}")

    def deactivate_bud(self, bud_id: str):
        """Deactivate a bud."""
        if bud_id in self.active_buds:
            del self.active_buds[bud_id]
            self._update_hooks()
            logger.info(f"Deactivated bud {bud_id}")

    def deactivate_all(self):
        """Deactivate all buds."""
        self.active_buds.clear()
        self._update_hooks()

    def remove_bud(self, bud_id: str):
        """Remove a bud entirely."""
        self.deactivate_bud(bud_id)
        if bud_id in self.buds:
            del self.buds[bud_id]

    def _update_hooks(self):
        """Update forward hooks based on active buds."""
        # Remove existing hooks
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

        if not self.active_buds:
            return

        # Collect biases per layer
        layer_biases: Dict[int, List[Tuple[torch.Tensor, float]]] = {}

        for bud_id, strength in self.active_buds.items():
            bud = self.buds[bud_id]
            effective_strength = strength * bud.strength

            for layer_idx, bias in bud.bias_vectors.items():
                if layer_idx not in layer_biases:
                    layer_biases[layer_idx] = []
                layer_biases[layer_idx].append((bias, effective_strength))

        # Register hooks
        layers = _get_model_layers(self.base_model)

        for layer_idx, biases in layer_biases.items():
            if layer_idx >= len(layers):
                continue

            hook_fn = self._create_bias_hook(biases)
            handle = layers[layer_idx].register_forward_hook(hook_fn)
            self._hook_handles.append(handle)

    def _create_bias_hook(
        self,
        biases: List[Tuple[torch.Tensor, float]]
    ) -> Callable:
        """Create a forward hook that adds biases to hidden states."""

        def hook(module, input, output):
            hidden_states = output[0]

            for bias, strength in biases:
                # Ensure bias is broadcastable
                if bias.shape[0] == hidden_states.shape[-1]:
                    # Add bias to all positions
                    hidden_states = hidden_states + strength * bias
                else:
                    logger.warning(
                        f"Bias shape {bias.shape} doesn't match hidden_dim {hidden_states.shape[-1]}"
                    )

            return (hidden_states,) + output[1:]

        return hook

    @contextmanager
    def bud_context(self, bud_ids: List[str], strengths: Optional[List[float]] = None):
        """
        Context manager for temporarily activating buds.

        Example:
            >>> with budded.bud_context(["bud-Fish"]):
            ...     output = budded.generate("Tell me about...")
        """
        if strengths is None:
            strengths = [1.0] * len(bud_ids)

        # Save current state
        previously_active = self.active_buds.copy()

        # Activate requested buds
        self.active_buds = {
            bud_id: strength
            for bud_id, strength in zip(bud_ids, strengths)
        }
        self._update_hooks()

        try:
            yield
        finally:
            # Restore previous state
            self.active_buds = previously_active
            self._update_hooks()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> str:
        """Generate text with active buds applied."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def forward(self, *args, **kwargs):
        """Forward pass through base model with buds applied."""
        return self.base_model(*args, **kwargs)

    def get_active_buds(self) -> Dict[str, float]:
        """Get currently active buds and their strengths."""
        return self.active_buds.copy()

    def list_buds(self) -> List[Dict[str, Any]]:
        """List all registered buds."""
        return [
            {
                "bud_id": bud.bud_id,
                "concept_id": bud.concept_id,
                "layers": bud.injection_layers,
                "active": bud.bud_id in self.active_buds,
                "strength": self.active_buds.get(bud.bud_id, 0.0)
            }
            for bud in self.buds.values()
        ]
