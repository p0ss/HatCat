"""
Activation capture system using PyTorch hooks.
Captures layer activations from transformer models with TopK sparsity.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class ActivationConfig:
    """Configuration for activation capture."""
    top_k: int = 100  # Number of top activations to keep (sparsity)
    layer_names: Optional[List[str]] = None  # Specific layers to capture, None = all
    capture_mode: str = "forward"  # "forward", "backward", or "both"
    storage_dtype: torch.dtype = torch.float16  # Dtype for storage efficiency


class ActivationCapture:
    """
    Captures activations from PyTorch models using forward hooks.
    Applies TopK sparsity to reduce memory footprint.
    """

    def __init__(self, model: nn.Module, config: ActivationConfig):
        self.model = model
        self.config = config
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List = []
        self.layer_dims: Dict[str, tuple] = {}

    def _make_hook(self, name: str) -> Callable:
        """Create a hook function that captures and stores activations."""
        def hook(module, input, output):
            # Handle different output types (tensor, tuple, etc.)
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output

            # Flatten to get activation vector
            if len(activation.shape) > 2:
                # For transformers: [batch, seq_len, hidden_dim]
                # We'll average over sequence dimension for now
                activation = activation.mean(dim=1)  # [batch, hidden_dim]

            # Convert to desired dtype for storage
            activation = activation.detach().to(self.config.storage_dtype)

            # Apply TopK sparsity
            if self.config.top_k > 0:
                activation = self._apply_topk_sparsity(activation)

            self.activations[name] = activation

            # Store layer dimensions for reconstruction
            if name not in self.layer_dims:
                self.layer_dims[name] = activation.shape

        return hook

    def _apply_topk_sparsity(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Apply TopK sparsity to activation tensor.
        Keeps only top-k values, zeros out the rest.
        Returns sparse representation.
        """
        # Take absolute values for finding top activations
        abs_activation = torch.abs(activation)

        # Get top-k indices
        k = min(self.config.top_k, activation.shape[-1])
        topk_values, topk_indices = torch.topk(abs_activation, k, dim=-1)

        # Create sparse tensor (store as indices + values for efficiency)
        # For now, return dense tensor with zeros for non-topk elements
        sparse_activation = torch.zeros_like(activation)

        # Scatter the top-k values back
        if len(activation.shape) == 2:  # [batch, features]
            for i in range(activation.shape[0]):
                sparse_activation[i, topk_indices[i]] = activation[i, topk_indices[i]]
        else:  # [features]
            sparse_activation[topk_indices] = activation[topk_indices]

        return sparse_activation

    def register_hooks(self, layer_filter: Optional[Callable] = None):
        """
        Register hooks on model layers.

        Args:
            layer_filter: Optional function to filter which layers to hook.
                         Should take (name, module) and return bool.
        """
        for name, module in self.model.named_modules():
            # Default filter: capture attention and MLP outputs
            should_capture = False

            if layer_filter is not None:
                should_capture = layer_filter(name, module)
            else:
                # Default: capture key transformer layers
                if any(x in name.lower() for x in ['attention', 'mlp', 'feed_forward', 'ffn']):
                    should_capture = True
                # Also capture layer outputs
                if 'layer.' in name and name.endswith('layer'):
                    should_capture = True

            # If specific layers specified, only capture those
            if self.config.layer_names is not None:
                should_capture = name in self.config.layer_names

            if should_capture:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

        print(f"Registered {len(self.hooks)} hooks on model layers")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get captured activations."""
        return self.activations

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}

    def compute_activation_diff(
        self,
        baseline_activations: Dict[str, torch.Tensor],
        current_activations: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute difference between current and baseline activations.
        This is key for identifying concept-specific activation patterns.

        Args:
            baseline_activations: Baseline (neutral) activation patterns
            current_activations: Current activations (defaults to self.activations)

        Returns:
            Dictionary of activation differences per layer
        """
        if current_activations is None:
            current_activations = self.activations

        diffs = {}
        for layer_name in current_activations.keys():
            if layer_name in baseline_activations:
                # Compute difference
                diff = current_activations[layer_name] - baseline_activations[layer_name]

                # Re-apply sparsity to the difference
                if self.config.top_k > 0:
                    diff = self._apply_topk_sparsity(diff)

                diffs[layer_name] = diff

        return diffs

    def to_numpy(self, activations: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Convert activation tensors to numpy arrays."""
        return {
            name: tensor.cpu().numpy()
            for name, tensor in activations.items()
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up hooks."""
        self.remove_hooks()


class BaselineGenerator:
    """
    Generates baseline (neutral) activations for computing differences.
    Uses generic prompts to establish activation baseline.
    """

    def __init__(self, model, tokenizer, capture_config: ActivationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.capture_config = capture_config

    def generate_baseline(
        self,
        num_samples: int = 10,
        neutral_prompts: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate baseline activations by averaging over neutral prompts.

        Args:
            num_samples: Number of neutral prompts to average over
            neutral_prompts: Optional list of neutral prompts. If None, uses defaults.

        Returns:
            Averaged baseline activations per layer
        """
        if neutral_prompts is None:
            # Default neutral prompts - generic, non-concept-specific
            neutral_prompts = [
                "The",
                "A",
                "This is",
                "It was",
                "There are",
                "In the",
                "On a",
                "With some",
                "From this",
                "To be"
            ][:num_samples]

        all_activations = []

        with ActivationCapture(self.model, self.capture_config) as capturer:
            capturer.register_hooks()

            for prompt in neutral_prompts:
                # Clear previous activations
                capturer.clear_activations()

                # Run forward pass
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    _ = self.model(**inputs)

                # Store activations
                all_activations.append(capturer.get_activations().copy())

        # Average activations across neutral prompts
        baseline = {}
        layer_names = all_activations[0].keys()

        for layer_name in layer_names:
            layer_acts = [acts[layer_name] for acts in all_activations if layer_name in acts]
            baseline[layer_name] = torch.stack(layer_acts).mean(dim=0)

        return baseline
