"""
Concept color mapping using sunburst positions.

Maps concepts to colors based on their angular position in the SUMO ontology sunburst.
- Hue: Determined by angular position (0-360°)
- Saturation: Fixed at 70%
- Lightness: Inversely proportional to divergence (bright = low div, dark = high div)
"""

import json
from pathlib import Path
import colorsys
from typing import Dict, Tuple, List, Optional
import numpy as np

class ConceptColorMapper:
    """Maps concepts to colors using sunburst positions and divergence."""

    def __init__(self, positions_file: Path = None):
        """Load sunburst positions."""

        if positions_file is None:
            # Go up from src/ui/visualization to project root, then into results/
            positions_file = Path(__file__).parent.parent.parent.parent / 'results' / 'concept_sunburst_positions.json'

        if not positions_file.exists():
            raise FileNotFoundError(
                f"Sunburst positions not found: {positions_file}\n"
                f"Run: poetry run python scripts/build_concept_sunburst_positions_simple.py"
            )

        with open(positions_file) as f:
            self.positions = json.load(f)

    def get_concept_hue(self, concept: str) -> float:
        """Get hue (0-360) for a concept based on its sunburst position."""

        if concept not in self.positions:
            # Unknown concept: use hash for consistent color
            return (hash(concept) % 360)

        return self.positions[concept]['angle']

    def hsl_to_hex(self, h: float, s: float, l: float) -> str:
        """Convert HSL to hex color."""

        r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def divergence_to_lightness(self, divergence: float) -> float:
        """
        Convert divergence to lightness value.

        Low divergence (0.0) → Bright (0.9 lightness)
        High divergence (1.0) → Dark (0.1 lightness)
        """

        # Inverse relationship: high divergence = low lightness
        return 0.9 - (divergence * 0.8)

    def get_concept_color(
        self,
        concept: str,
        divergence: float,
        saturation: float = 0.7
    ) -> str:
        """
        Get hex color for a concept with given divergence.

        Args:
            concept: Concept name
            divergence: Divergence value (0-1)
            saturation: Color saturation (default 0.7)

        Returns:
            Hex color string (e.g., "#ff5733")
        """

        hue = self.get_concept_hue(concept)
        lightness = self.divergence_to_lightness(divergence)

        return self.hsl_to_hex(hue, saturation, lightness)

    def calculate_concept_spread(
        self,
        concepts_with_divergences: List[Tuple[str, float, float]]
    ) -> float:
        """
        Calculate angular spread of concepts on color wheel.

        Returns saturation value (0-1):
        - High saturation (0.9): All concepts clustered in one area → vivid color
        - Low saturation (0.1): Concepts spread across wheel → dull/gray
        - Zero saturation (0.0): Indistinguishable → pure gray
        """

        if len(concepts_with_divergences) <= 1:
            return 0.7  # Default for single concept

        # Get hues weighted by activation
        hues = []
        weights = []
        for concept, activation, divergence in concepts_with_divergences:
            if activation > 0.1:  # Only consider significant activations
                hues.append(self.get_concept_hue(concept))
                weights.append(activation)

        if len(hues) == 0:
            return 0.0  # Indistinguishable → gray

        if len(hues) == 1:
            return 0.9  # Single strong concept → vivid

        # Calculate circular variance (how spread out the hues are)
        # Convert to unit vectors and sum
        hues = np.array(hues)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        hues_rad = np.radians(hues)
        mean_x = np.sum(weights * np.cos(hues_rad))
        mean_y = np.sum(weights * np.sin(hues_rad))

        # R is the mean resultant length (0-1)
        # R ≈ 1: concepts tightly clustered → high saturation
        # R ≈ 0: concepts spread out → low saturation
        R = np.sqrt(mean_x**2 + mean_y**2)

        # Map R to saturation
        # R = 1.0 → saturation = 0.9 (vivid)
        # R = 0.5 → saturation = 0.5 (moderate)
        # R = 0.0 → saturation = 0.1 (dull gray)
        saturation = 0.1 + (R * 0.8)

        return saturation

    def blend_concept_colors_average(
        self,
        concepts_with_divergences: List[Tuple[str, float, float]],
        use_adaptive_saturation: bool = True
    ) -> str:
        """
        Blend multiple concepts by averaging their HSL values.

        Args:
            concepts_with_divergences: List of (concept, activation, divergence) tuples
            use_adaptive_saturation: If True, calculate saturation from concept spread

        Returns:
            Hex color string
        """

        if not concepts_with_divergences:
            return "#808080"  # Gray default

        # Calculate saturation from concept clustering
        if use_adaptive_saturation:
            saturation = self.calculate_concept_spread(concepts_with_divergences)
        else:
            saturation = 0.7

        # Weight by activation strength
        weighted_hue_x = 0.0
        weighted_hue_y = 0.0
        weighted_lightness = 0.0
        total_weight = 0.0

        for concept, activation, divergence in concepts_with_divergences:
            hue = self.get_concept_hue(concept)
            lightness = self.divergence_to_lightness(divergence)

            # Convert hue to unit circle for averaging
            hue_rad = np.radians(hue)
            weighted_hue_x += np.cos(hue_rad) * activation
            weighted_hue_y += np.sin(hue_rad) * activation
            weighted_lightness += lightness * activation
            total_weight += activation

        if total_weight == 0:
            return "#808080"

        # Average
        avg_hue_rad = np.arctan2(weighted_hue_y / total_weight, weighted_hue_x / total_weight)
        avg_hue = np.degrees(avg_hue_rad) % 360.0
        avg_lightness = weighted_lightness / total_weight

        return self.hsl_to_hex(avg_hue, saturation, avg_lightness)

    def create_gradient(
        self,
        concept1: str,
        divergence1: float,
        concept2: str,
        divergence2: float,
        steps: int = 5,
        saturation: float = 0.7
    ) -> List[str]:
        """
        Create a gradient between two concepts.

        Useful for showing transition from text concept to hidden concept.

        Args:
            concept1: First concept (e.g., text detection)
            divergence1: Divergence for concept1
            concept2: Second concept (e.g., activation detection)
            divergence2: Divergence for concept2
            steps: Number of gradient steps
            saturation: Color saturation

        Returns:
            List of hex colors forming gradient
        """

        hue1 = self.get_concept_hue(concept1)
        hue2 = self.get_concept_hue(concept2)
        light1 = self.divergence_to_lightness(divergence1)
        light2 = self.divergence_to_lightness(divergence2)

        # Handle hue wrap-around (take shorter path around color wheel)
        hue_diff = hue2 - hue1
        if abs(hue_diff) > 180:
            if hue_diff > 0:
                hue1 += 360
            else:
                hue2 += 360

        gradient = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            hue = (hue1 + (hue2 - hue1) * t) % 360.0
            lightness = light1 + (light2 - light1) * t
            gradient.append(self.hsl_to_hex(hue, saturation, lightness))

        return gradient

    def create_palette_swatch(
        self,
        concepts_with_divergences: List[Tuple[str, float, float]],
        max_colors: int = 5,
        saturation: float = 0.7
    ) -> List[str]:
        """
        Create a palette swatch of the top concepts.

        Args:
            concepts_with_divergences: List of (concept, activation, divergence) tuples
            max_colors: Maximum colors in palette
            saturation: Color saturation

        Returns:
            List of hex colors (sorted by activation strength)
        """

        # Sort by activation strength
        sorted_concepts = sorted(
            concepts_with_divergences,
            key=lambda x: x[1],
            reverse=True
        )[:max_colors]

        palette = []
        for concept, activation, divergence in sorted_concepts:
            color = self.get_concept_color(concept, divergence, saturation)
            palette.append(color)

        return palette

# Global instance
_color_mapper = None

def get_color_mapper() -> ConceptColorMapper:
    """Get or create global color mapper instance."""
    global _color_mapper
    if _color_mapper is None:
        _color_mapper = ConceptColorMapper()
    return _color_mapper
