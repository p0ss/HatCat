#!/usr/bin/env python3
"""
Lens Types and Basic Classes

Core types used throughout the lens management system:
- LensRole: Enum for different lens purposes
- SimpleMLP: The MLP classifier architecture
- SimplexBinding: Configuration for simplex-concept bindings
- ConceptMetadata: Metadata for a single concept
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn


class LensRole(Enum):
    """Roles for different lens types in the monitoring system."""
    CONCEPT = "concept"        # Hierarchical discrimination vs siblings
    SIMPLEX = "simplex"        # Intensity tracking relative to baseline (tripole)
    BEHAVIORAL = "behavioral"  # Pattern detection (e.g., deception markers)
    CATEGORY = "category"      # Domain/layer markers (layer 0 style)


class SimpleMLP(nn.Module):
    """Simple MLP classifier matching SUMO training architecture."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dtype: torch.dtype = None, layer_norm: bool = False):
        """
        Args:
            input_dim: Input feature dimension (model hidden_dim)
            hidden_dim: MLP hidden layer dimension
            dtype: Parameter dtype. If None, uses default (float32).
                   Use torch.bfloat16 for memory-efficient inference.
            layer_norm: If True, include LayerNorm at input (matches new training arch)
        """
        super().__init__()
        self.has_layer_norm = layer_norm

        # Keep 'net' name for backward compatibility with saved lenses
        layers = []
        if layer_norm:
            layers.append(nn.LayerNorm(input_dim, dtype=dtype))
        layers.extend([
            nn.Linear(input_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1, dtype=dtype),
        ])
        self.net = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logits=False):
        """
        Forward pass.

        Args:
            x: Input tensor
            return_logits: If True, return (probability, logit) tuple

        Returns:
            If return_logits=False: probability [0,1]
            If return_logits=True: (probability, logit) tuple
        """
        logits = self.net(x).squeeze(-1)
        probs = self.sigmoid(logits)

        if return_logits:
            return probs, logits
        return probs


def detect_layer_norm(state_dict: dict) -> bool:
    """Detect if state_dict has LayerNorm at input (1D weight vs 2D)."""
    first_key = "net.0.weight" if "net.0.weight" in state_dict else "0.weight"
    if first_key in state_dict:
        return len(state_dict[first_key].shape) == 1
    return False


def create_lens_from_state_dict(state_dict: dict, hidden_dim: int, device: str) -> SimpleMLP:
    """Create SimpleMLP matching the state_dict architecture."""
    has_ln = detect_layer_norm(state_dict)
    lens = SimpleMLP(hidden_dim, layer_norm=has_ln).to(device)
    lens.eval()

    # Handle missing net. prefix
    if "0.weight" in state_dict and "net.0.weight" not in state_dict:
        new_state_dict = {f"net.{k}": v for k, v in state_dict.items()}
        state_dict = new_state_dict

    lens.load_state_dict(state_dict)
    return lens


@dataclass
class SimplexBinding:
    """Configuration for a simplex bound to a concept."""
    simplex_term: str
    always_on: bool = False
    poles: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'baseline_window': 100,
        'alert_threshold': 2.0,
        'trend_window': 500
    })


@dataclass
class ConceptMetadata:
    """Metadata for a single SUMO concept."""
    sumo_term: str
    layer: int
    category_children: List[str] = field(default_factory=list)
    parent_concepts: List[str] = field(default_factory=list)
    synset_count: int = 0
    sumo_depth: int = 0

    # Role and simplex binding (new in MAP Meld Protocol)
    role: LensRole = LensRole.CONCEPT
    simplex_binding: Optional[SimplexBinding] = None
    domain: Optional[str] = None

    # Lens paths (set by manager)
    activation_lens_path: Optional[Path] = None
    text_lens_path: Optional[Path] = None
    simplex_lens_path: Optional[Path] = None
    has_text_lens: bool = False
    has_activation_lens: bool = False
    has_simplex_lens: bool = False


__all__ = [
    "LensRole",
    "SimpleMLP",
    "SimplexBinding",
    "ConceptMetadata",
    "detect_layer_norm",
    "create_lens_from_state_dict",
]
