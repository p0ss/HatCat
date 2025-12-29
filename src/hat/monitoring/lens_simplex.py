#!/usr/bin/env python3
"""
Simplex Lens Management

Handles intensity-tracking simplex lenses that measure motive/drive
strength relative to baseline, as opposed to hierarchical concept lenses
that discriminate between sibling concepts.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

import numpy as np
import torch
import torch.nn as nn

from .lens_types import SimpleMLP, create_lens_from_state_dict


class SimplexManager:
    """
    Manages simplex lenses for intensity/drive monitoring.

    Simplexes provide a different view than hierarchical concepts:
    - Hierarchical: "Is this concept prominent vs siblings?"
    - Simplex: "How intense is this drive vs baseline?"

    This dual-view is described in MAP Meld Protocol §12.3.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Loaded simplex lenses
        self.loaded_simplex_lenses: Dict[str, nn.Module] = {}
        self.simplex_scores: Dict[str, float] = {}
        self.simplex_baselines: Dict[str, List[float]] = defaultdict(list)

        # Binding registry: concept_term -> simplex_term
        self.simplex_bindings: Dict[str, str] = {}

        # Always-on simplexes (run every token)
        self.always_on_simplexes: Set[str] = set()

        # Hidden dim for creating lenses
        self.hidden_dim: Optional[int] = None

    def set_hidden_dim(self, hidden_dim: int):
        """Set hidden dimension for lens creation."""
        self.hidden_dim = hidden_dim

    def load_simplex(self, simplex_term: str, lens_path: Path) -> bool:
        """
        Load a simplex lens for intensity monitoring.

        Args:
            simplex_term: Name of the simplex (e.g., "AutonomyDrive")
            lens_path: Path to the simplex lens file

        Returns:
            True if loaded successfully, False otherwise
        """
        if simplex_term in self.loaded_simplex_lenses:
            return True

        if not lens_path.exists():
            return False

        try:
            state_dict = torch.load(lens_path, map_location=self.device)

            # Infer hidden dim if not set
            if self.hidden_dim is None:
                for key, value in state_dict.items():
                    if 'weight' in key and len(value.shape) == 2:
                        self.hidden_dim = value.shape[1]
                        break

            lens = create_lens_from_state_dict(state_dict, self.hidden_dim, self.device)
            self.loaded_simplex_lenses[simplex_term] = lens
            self.simplex_scores[simplex_term] = 0.0
            return True

        except Exception as e:
            print(f"  Failed to load simplex {simplex_term}: {e}")
            return False

    def register_binding(
        self,
        concept_term: str,
        simplex_term: str,
        always_on: bool = False
    ):
        """
        Register a binding between a hierarchical concept and its simplex.

        Args:
            concept_term: Name of the hierarchical concept
            simplex_term: Name of the bound simplex
            always_on: Whether this simplex should run every token
        """
        self.simplex_bindings[concept_term] = simplex_term
        if always_on:
            self.always_on_simplexes.add(simplex_term)

    def detect(
        self,
        hidden_state: torch.Tensor,
        simplex_terms: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Run simplex lenses and return activations.

        Args:
            hidden_state: Hidden state tensor [1, hidden_dim] or [hidden_dim]
            simplex_terms: Specific simplexes to run, or None for always-on only

        Returns:
            Dict mapping simplex_term to activation score
        """
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        # Match dtype
        if self.loaded_simplex_lenses:
            sample_lens = next(iter(self.loaded_simplex_lenses.values()))
            lens_dtype = next(sample_lens.parameters()).dtype
            if hidden_state.dtype != lens_dtype:
                hidden_state = hidden_state.to(dtype=lens_dtype)

        terms_to_run = simplex_terms or list(self.always_on_simplexes)
        results = {}

        with torch.inference_mode():
            for simplex_term in terms_to_run:
                if simplex_term not in self.loaded_simplex_lenses:
                    continue

                lens = self.loaded_simplex_lenses[simplex_term]
                prob = lens(hidden_state).item()

                results[simplex_term] = prob
                self.simplex_scores[simplex_term] = prob

                # Update rolling baseline
                baseline_list = self.simplex_baselines[simplex_term]
                baseline_list.append(prob)

                # Keep only last N samples
                max_baseline = 100
                if len(baseline_list) > max_baseline:
                    self.simplex_baselines[simplex_term] = baseline_list[-max_baseline:]

        return results

    def get_deviation(self, simplex_term: str) -> Optional[float]:
        """
        Get current deviation from baseline for a simplex.

        Args:
            simplex_term: Name of the simplex

        Returns:
            Standard deviations from baseline, or None if insufficient data
        """
        if simplex_term not in self.simplex_scores:
            return None

        baseline = self.simplex_baselines.get(simplex_term, [])
        if len(baseline) < 10:
            return None

        current = self.simplex_scores[simplex_term]
        mean = np.mean(baseline)
        std = np.std(baseline)

        if std < 0.001:
            return 0.0

        return (current - mean) / std

    def get_combined_activation(
        self,
        concept_term: str,
        hierarchical_scores: Dict[tuple, float],
        layer: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get both hierarchical and simplex activation for a concept.

        Args:
            concept_term: Name of the concept
            hierarchical_scores: Dict of (concept, layer) -> score
            layer: Optional layer hint

        Returns:
            Dict with hierarchical, simplex, and interpretation
        """
        result = {
            'concept_term': concept_term,
            'hierarchical': None,
            'simplex': None,
            'simplex_deviation': None,
            'interpretation': 'unknown'
        }

        # Get hierarchical activation
        concept_key = None
        if layer is not None:
            concept_key = (concept_term, layer)
        else:
            for key in hierarchical_scores.keys():
                if key[0] == concept_term:
                    concept_key = key
                    break

        if concept_key and concept_key in hierarchical_scores:
            result['hierarchical'] = hierarchical_scores[concept_key]

        # Get simplex activation if bound
        if concept_term in self.simplex_bindings:
            simplex_term = self.simplex_bindings[concept_term]
            if simplex_term in self.simplex_scores:
                result['simplex'] = self.simplex_scores[simplex_term]
                result['simplex_deviation'] = self.get_deviation(simplex_term)

        # Generate interpretation
        h = result['hierarchical']
        s = result['simplex']

        if h is not None and s is not None:
            h_high = h > 0.6
            s_high = s > 0.6

            if h_high and s_high:
                result['interpretation'] = 'active_elevated'
            elif h_high and not s_high:
                result['interpretation'] = 'discussing_not_activated'
            elif not h_high and s_high:
                result['interpretation'] = 'implicit_elevated'
            else:
                result['interpretation'] = 'not_relevant'
        elif h is not None:
            result['interpretation'] = 'hierarchical_only'
        elif s is not None:
            result['interpretation'] = 'simplex_only'

        return result

    def get_all_activations(self) -> Dict[str, Dict[str, Any]]:
        """Get current activations for all loaded simplexes."""
        results = {}
        for simplex_term in self.loaded_simplex_lenses:
            results[simplex_term] = {
                'activation': self.simplex_scores.get(simplex_term, 0.0),
                'deviation': self.get_deviation(simplex_term),
                'always_on': simplex_term in self.always_on_simplexes,
                'bound_to': [
                    concept for concept, simplex in self.simplex_bindings.items()
                    if simplex == simplex_term
                ]
            }
        return results


__all__ = ["SimplexManager"]
