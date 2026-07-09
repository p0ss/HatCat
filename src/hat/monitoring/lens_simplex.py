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

        # Tripole binding registry: concept_term -> logical simplex_name
        # (poles loaded under {simplex_name}_{positive,neutral,negative})
        self.tripole_bindings: Dict[str, str] = {}

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

    def load_tripole_simplex(
        self,
        simplex_name: str,
        simplex_dir: Path,
    ) -> Dict[str, bool]:
        """
        Load all three poles of a tripole simplex from a directory.

        Expected layout: {simplex_dir}/{pole}/{simplex_name}_{pole}_classifier.pt
        where {pole} is one of "positive", "neutral", "negative".

        Each pole is registered under simplex_term f"{simplex_name}_{pole}",
        loaded via the underlying single-classifier load_simplex path.

        Args:
            simplex_name: Logical simplex name (without pole suffix)
            simplex_dir: Directory containing the per-pole subdirectories

        Returns:
            Dict mapping pole name to load success status
        """
        results: Dict[str, bool] = {}
        for pole in ("positive", "neutral", "negative"):
            pt_path = simplex_dir / pole / f"{simplex_name}_{pole}_classifier.pt"
            simplex_term = f"{simplex_name}_{pole}"
            results[pole] = self.load_simplex(simplex_term, pt_path)
        return results

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

    def register_tripole_binding(
        self,
        concept_term: str,
        simplex_name: str,
        always_on: bool = False,
    ):
        """
        Register a binding between a hierarchical concept and a tripole simplex.

        The tripole simplex is identified by a logical name; its three pole
        lenses are loaded under {simplex_name}_{positive,neutral,negative}.

        Args:
            concept_term: Name of the hierarchical concept
            simplex_name: Logical name of the tripole simplex (without pole suffix)
            always_on: Whether all three poles should run every token
        """
        self.tripole_bindings[concept_term] = simplex_name
        if always_on:
            for pole in ("positive", "neutral", "negative"):
                self.always_on_simplexes.add(f"{simplex_name}_{pole}")

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

    def get_tripole_state(self, simplex_name: str) -> Dict[str, float]:
        """
        Get current activations for all three poles of a tripole simplex.

        Args:
            simplex_name: Logical simplex name (without pole suffix)

        Returns:
            Dict mapping pole name to current activation score
        """
        return {
            pole: self.simplex_scores.get(f"{simplex_name}_{pole}", 0.0)
            for pole in ("positive", "neutral", "negative")
        }

    def get_tripole_deviation(self, simplex_name: str) -> Dict[str, Optional[float]]:
        """
        Get current deviation from baseline for all three poles.

        Args:
            simplex_name: Logical simplex name (without pole suffix)

        Returns:
            Dict mapping pole name to std-devs from baseline
            (None if insufficient data)
        """
        return {
            pole: self.get_deviation(f"{simplex_name}_{pole}")
            for pole in ("positive", "neutral", "negative")
        }

    def get_combined_activation(
        self,
        concept_term: str,
        hierarchical_scores: Dict[tuple, float],
        layer: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get hierarchical and simplex activation for a concept with combined interpretation.

        Recognizes both legacy single-output simplex bindings (four-state interpretation)
        and tripole simplex bindings (eight-state interpretation including blended-state
        detection). Tripole bindings take precedence when both are present.

        Args:
            concept_term: Name of the concept
            hierarchical_scores: Dict of (concept, layer) -> score
            layer: Optional layer hint

        Returns:
            Dict with hierarchical, simplex/tripole activations, and interpretation
        """
        result = {
            'concept_term': concept_term,
            'hierarchical': None,
            'simplex': None,
            'simplex_deviation': None,
            'tripole': None,
            'tripole_deviation': None,
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

        h = result['hierarchical']

        # Tripole binding takes precedence over legacy single-output binding
        if concept_term in self.tripole_bindings:
            simplex_name = self.tripole_bindings[concept_term]
            result['tripole'] = self.get_tripole_state(simplex_name)
            result['tripole_deviation'] = self.get_tripole_deviation(simplex_name)

            pos = result['tripole']['positive']
            neg = result['tripole']['negative']
            neutral = result['tripole']['neutral']
            drive_intensity = max(pos, neg)
            drive_direction = 'positive' if pos >= neg else 'negative'
            drive_high = drive_intensity > 0.6
            saddle_high = neutral > 0.6

            if h is not None:
                h_high = h > 0.6
                if h_high and drive_high:
                    result['interpretation'] = f'active_{drive_direction}_drive'
                elif h_high and saddle_high:
                    result['interpretation'] = 'active_blended'
                elif h_high:
                    result['interpretation'] = 'discussing_quietly'
                elif drive_high:
                    result['interpretation'] = f'implicit_{drive_direction}_drive'
                elif saddle_high:
                    result['interpretation'] = 'implicit_blended'
                else:
                    result['interpretation'] = 'not_relevant'
            else:
                if drive_high:
                    result['interpretation'] = f'tripole_{drive_direction}_only'
                elif saddle_high:
                    result['interpretation'] = 'tripole_blended_only'
                else:
                    result['interpretation'] = 'tripole_inactive'

            return result

        # Legacy single-output simplex binding
        if concept_term in self.simplex_bindings:
            simplex_term = self.simplex_bindings[concept_term]
            if simplex_term in self.simplex_scores:
                result['simplex'] = self.simplex_scores[simplex_term]
                result['simplex_deviation'] = self.get_deviation(simplex_term)

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

    def get_all_tripole_activations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current activations grouped by logical tripole simplex name.

        Walks loaded_simplex_lenses for entries matching the {name}_{pole} naming
        convention and groups them. Returns a dict keyed by logical simplex name with
        per-pole activation, deviation, the always_on status, and the bound concepts.
        """
        groups: Dict[str, Dict[str, Any]] = {}
        for term in self.loaded_simplex_lenses:
            for pole in ("positive", "neutral", "negative"):
                suffix = f"_{pole}"
                if term.endswith(suffix):
                    name = term[: -len(suffix)]
                    bucket = groups.setdefault(name, {
                        'poles': {},
                        'always_on': False,
                        'bound_to': [],
                    })
                    bucket['poles'][pole] = {
                        'activation': self.simplex_scores.get(term, 0.0),
                        'deviation': self.get_deviation(term),
                    }
                    if term in self.always_on_simplexes:
                        bucket['always_on'] = True
                    break

        for name in groups:
            groups[name]['bound_to'] = [
                concept for concept, simplex in self.tripole_bindings.items()
                if simplex == name
            ]

        return groups


__all__ = ["SimplexManager"]
