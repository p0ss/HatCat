"""
Unified classifier implementations for HAT lenses.

All classifiers output raw logits (no sigmoid) for flexibility:
- Use sigmoid for probability interpretation
- Use raw logits for gradient-based steering
- Use logits with BCEWithLogitsLoss for training

Canonical architecture (MLPClassifier):
    input → LayerNorm → 128 → ReLU → Dropout → 64 → ReLU → Dropout → 1

LayerNorm at input normalizes activations to prevent saturation during training.
This matches the architecture used in sumo_classifiers.py for training.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Standard MLP lens classifier for concept detection.

    Architecture (with layer_norm=True):
        input_dim → LayerNorm → 128 → ReLU → Dropout → 64 → ReLU → Dropout → 1

    Architecture (with layer_norm=False, legacy):
        input_dim → 128 → ReLU → Dropout → 64 → ReLU → Dropout → 1

    Outputs raw logits (no sigmoid). Apply sigmoid for probabilities.

    Args:
        input_dim: Model hidden dimension (e.g., 4096 for Apertus-8B, 2560 for Gemma-3-4b)
        hidden_dim: First hidden layer size (default: 128)
        dropout: Dropout rate (default: 0.1)
        layer_norm: Whether to include LayerNorm at input (default: True)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.has_layer_norm = layer_norm

        layers = []
        if layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits.

        Args:
            x: Input activations of shape (batch, input_dim) or (input_dim,)

        Returns:
            Raw logits of shape (batch, 1) or (1,)
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability (applies sigmoid to logits)."""
        return torch.sigmoid(self.forward(x))


class LinearProbe(nn.Module):
    """
    Simple linear probe classifier.

    A single linear layer for concept detection. Useful as a baseline
    or when the concept has a clean linear representation.

    Args:
        input_dim: Model hidden dimension
        bias: Whether to include bias term (default: True)
    """

    def __init__(self, input_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        return self.linear(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability (applies sigmoid to logits)."""
        return torch.sigmoid(self.forward(x))

    @property
    def direction(self) -> torch.Tensor:
        """Return the probe direction vector (normalized weights)."""
        with torch.no_grad():
            weights = self.linear.weight.squeeze()
            return weights / weights.norm()


def infer_classifier_type(state_dict: Dict[str, Any]) -> str:
    """
    Infer classifier type from state dict keys.

    Returns:
        "mlp" for MLPClassifier, "linear" for LinearProbe
    """
    keys = list(state_dict.keys())

    # MLPClassifier uses "net.0.weight", "net.3.weight", "net.6.weight"
    if any("net." in k for k in keys):
        return "mlp"

    # LinearProbe uses "linear.weight"
    if any("linear." in k for k in keys):
        return "linear"

    # Legacy format: raw "0.weight", "3.weight" etc (treated as MLP)
    if "0.weight" in keys:
        return "mlp"

    raise ValueError(f"Unknown classifier format. Keys: {keys}")


def has_layer_norm(state_dict: Dict[str, Any]) -> bool:
    """
    Detect if classifier has LayerNorm at input.

    LayerNorm weights are 1D (shape [input_dim]), while Linear weights are 2D.
    """
    # Check net.0.weight or 0.weight
    first_weight_key = "net.0.weight" if "net.0.weight" in state_dict else "0.weight"
    if first_weight_key in state_dict:
        return len(state_dict[first_weight_key].shape) == 1
    return False


def infer_input_dim(state_dict: Dict[str, Any], classifier_type: str) -> int:
    """Infer input dimension from state dict."""
    if classifier_type == "mlp":
        # Check for net.0.weight first, then legacy 0.weight
        first_weight_key = "net.0.weight" if "net.0.weight" in state_dict else "0.weight"
        if first_weight_key in state_dict:
            weight = state_dict[first_weight_key]
            if len(weight.shape) == 1:
                # LayerNorm: weight is 1D [input_dim]
                return weight.shape[0]
            else:
                # Linear: weight is 2D [hidden_dim, input_dim]
                return weight.shape[1]
    elif classifier_type == "linear":
        return state_dict["linear.weight"].shape[1]

    raise ValueError(f"Cannot infer input_dim for {classifier_type}")


def load_classifier(
    path: Union[str, Path],
    device: str = "cuda",
    classifier_type: Optional[str] = None,
) -> nn.Module:
    """
    Load a trained classifier from disk.

    Automatically infers the classifier type (MLP or linear) from the
    saved state dict structure if not specified. Also detects whether
    the classifier has LayerNorm at input.

    Args:
        path: Path to the .pt file containing classifier state_dict
        device: Device to load onto
        classifier_type: Force a specific type ("mlp" or "linear")

    Returns:
        Loaded classifier ready for inference
    """
    path = Path(path)
    state_dict = torch.load(path, map_location=device, weights_only=True)

    # Infer type if not specified
    if classifier_type is None:
        classifier_type = infer_classifier_type(state_dict)

    # Infer input dimension
    input_dim = infer_input_dim(state_dict, classifier_type)

    # Create and load classifier
    if classifier_type == "mlp":
        # Detect if classifier has LayerNorm
        use_layer_norm = has_layer_norm(state_dict)
        classifier = MLPClassifier(input_dim, layer_norm=use_layer_norm)

        # Handle legacy format (0.weight vs net.0.weight)
        if "0.weight" in state_dict and "net.0.weight" not in state_dict:
            # Convert legacy keys to new format
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = f"net.{key}"
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        classifier.load_state_dict(state_dict)

    elif classifier_type == "linear":
        classifier = LinearProbe(input_dim)
        classifier.load_state_dict(state_dict)

    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    classifier.to(device)
    classifier.eval()
    return classifier


def save_classifier(classifier: nn.Module, path: Union[str, Path]) -> None:
    """
    Save a classifier to disk.

    Args:
        classifier: The classifier to save
        path: Destination path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), path)
