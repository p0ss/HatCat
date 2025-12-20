"""
Lens - Abstract grouping of classifiers for a single concept.

A Lens translates high-level requests ("measure Deception", "steer toward Honesty")
into the appropriate combination of classifiers and layers for that concept.

Key design principles:
- Classifiers may exist on different layers (early/mid/late signal categories)
- Classifiers may use different techniques (MLP, linear probe, etc.)
- The Lens is the handshake/handoff layer - it knows WHICH classifiers exist
  and WHERE, but actual execution stays on GPU
- Support both single-layer and multi-layer operations

Layer Categories:
- early: Layers 0-10 (roughly) - raw feature detection
- mid: Layers 11-20 - concept formation
- late: Layers 21+ - abstract reasoning

Training identifies the best signal layer within each category for each concept.
Not all concepts have classifiers in all categories.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
import torch.nn as nn

from .classifier import load_classifier, MLPClassifier, LinearProbe


class LensNotFoundError(Exception):
    """Raised when a lens or classifier cannot be found."""
    pass


LayerCategory = Literal["early", "mid", "late"]


@dataclass
class ClassifierInfo:
    """
    Metadata about a single classifier within a Lens.

    Attributes:
        layer: Model layer index this classifier was trained on
        category: Signal category ("early", "mid", "late")
        technique: Classifier type ("mlp", "linear", etc.)
        path: Path to the saved classifier weights
        metrics: Optional training/validation metrics
    """
    layer: int
    category: LayerCategory
    technique: str = "mlp"
    path: Optional[Path] = None
    metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.path is not None:
            self.path = Path(self.path)


@dataclass
class Lens:
    """
    Abstract grouping of classifiers for a single concept.

    A Lens aggregates all classifiers trained to detect a specific concept,
    potentially across multiple layers and using different techniques.

    The Lens handles:
    - Finding which layers have classifiers for this concept
    - Selecting the best classifier for a given use case
    - Loading classifiers on demand

    Attributes:
        concept_name: The concept this lens detects (e.g., "Deception")
        classifiers: Dict mapping layer index to ClassifierInfo
        default_layer: Preferred layer when none specified
        description: Optional concept description
    """
    concept_name: str
    classifiers: Dict[int, ClassifierInfo] = field(default_factory=dict)
    default_layer: Optional[int] = None
    description: Optional[str] = None

    # Cached loaded classifiers (layer -> nn.Module)
    _loaded: Dict[int, nn.Module] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Set default layer if not specified
        if self.default_layer is None and self.classifiers:
            # Prefer mid category, then late, then early
            for category in ["mid", "late", "early"]:
                layer = self.get_best_layer(category)
                if layer is not None:
                    self.default_layer = layer
                    break
            # Fallback to first available
            if self.default_layer is None:
                self.default_layer = next(iter(self.classifiers.keys()))

    @property
    def layers(self) -> List[int]:
        """Get all layers this lens has classifiers for."""
        return sorted(self.classifiers.keys())

    @property
    def categories(self) -> List[str]:
        """Get all categories this lens has classifiers for."""
        return list(set(c.category for c in self.classifiers.values()))

    def has_layer(self, layer: int) -> bool:
        """Check if this lens has a classifier for the given layer."""
        return layer in self.classifiers

    def get_classifier_info(self, layer: int) -> ClassifierInfo:
        """Get classifier metadata for a specific layer."""
        if layer not in self.classifiers:
            raise LensNotFoundError(
                f"No classifier for concept '{self.concept_name}' at layer {layer}. "
                f"Available layers: {self.layers}"
            )
        return self.classifiers[layer]

    def get_classifier_path(self, layer: int) -> Path:
        """Get path to classifier weights for a specific layer."""
        info = self.get_classifier_info(layer)
        if info.path is None:
            raise LensNotFoundError(
                f"No path set for concept '{self.concept_name}' at layer {layer}"
            )
        return info.path

    def get_best_layer(self, category: Optional[LayerCategory] = None) -> Optional[int]:
        """
        Get the best layer for a category, or overall best if no category specified.

        Best is determined by:
        1. Highest accuracy metric if available
        2. Otherwise, middle layer in the category

        Args:
            category: Optional category to filter by ("early", "mid", "late")

        Returns:
            Layer index, or None if no matching classifiers
        """
        candidates = [
            (layer, info)
            for layer, info in self.classifiers.items()
            if category is None or info.category == category
        ]

        if not candidates:
            return None

        # Sort by accuracy if available, otherwise by layer
        def sort_key(item):
            layer, info = item
            if info.metrics and "accuracy" in info.metrics:
                return (-info.metrics["accuracy"], layer)  # Higher accuracy first
            return (0, layer)

        candidates.sort(key=sort_key)
        return candidates[0][0]

    def get_layers_by_category(self, category: LayerCategory) -> List[int]:
        """Get all layers in a specific category."""
        return [
            layer for layer, info in self.classifiers.items()
            if info.category == category
        ]

    def load_classifier(
        self,
        layer: Optional[int] = None,
        device: str = "cuda",
        force_reload: bool = False,
    ) -> nn.Module:
        """
        Load the classifier for a specific layer.

        Classifiers are cached after first load. Use force_reload=True to reload.

        Args:
            layer: Layer to load (uses default_layer if None)
            device: Device to load onto
            force_reload: Force reload even if cached

        Returns:
            Loaded classifier ready for inference
        """
        if layer is None:
            layer = self.default_layer

        if layer is None:
            raise LensNotFoundError(
                f"No layers available for concept '{self.concept_name}'"
            )

        # Check cache
        if not force_reload and layer in self._loaded:
            return self._loaded[layer]

        # Load from disk
        info = self.get_classifier_info(layer)
        if info.path is None:
            raise LensNotFoundError(
                f"No path set for concept '{self.concept_name}' at layer {layer}"
            )

        classifier = load_classifier(info.path, device=device, classifier_type=info.technique)
        self._loaded[layer] = classifier
        return classifier

    def unload(self, layer: Optional[int] = None) -> None:
        """
        Unload cached classifier(s) to free memory.

        Args:
            layer: Specific layer to unload, or None to unload all
        """
        if layer is not None:
            self._loaded.pop(layer, None)
        else:
            self._loaded.clear()

    def add_classifier(
        self,
        layer: int,
        category: LayerCategory,
        path: Path,
        technique: str = "mlp",
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Add a classifier to this lens.

        Args:
            layer: Model layer index
            category: Signal category
            path: Path to saved weights
            technique: Classifier type
            metrics: Optional training metrics
        """
        self.classifiers[layer] = ClassifierInfo(
            layer=layer,
            category=category,
            technique=technique,
            path=path,
            metrics=metrics,
        )

        # Update default if this is first classifier
        if self.default_layer is None:
            self.default_layer = layer

    @classmethod
    def from_layer_results(
        cls,
        concept_name: str,
        layer_results: Dict[int, Dict],
        lens_dir: Path,
        description: Optional[str] = None,
    ) -> "Lens":
        """
        Create a Lens from training results across multiple layers.

        Args:
            concept_name: The concept name
            layer_results: Dict mapping layer -> training results dict
            lens_dir: Base directory containing layer subdirectories
            description: Optional concept description

        Returns:
            Configured Lens instance
        """
        lens = cls(concept_name=concept_name, description=description)

        for layer, results in layer_results.items():
            # Infer category from layer index
            if layer <= 10:
                category = "early"
            elif layer <= 20:
                category = "mid"
            else:
                category = "late"

            # Build classifier path
            path = lens_dir / f"layer{layer}" / f"{concept_name}_classifier.pt"

            # Extract metrics if available
            metrics = {}
            if "accuracy" in results:
                metrics["accuracy"] = results["accuracy"]
            if "f1" in results:
                metrics["f1"] = results["f1"]

            lens.add_classifier(
                layer=layer,
                category=category,
                path=path,
                technique=results.get("technique", "mlp"),
                metrics=metrics or None,
            )

        return lens


def categorize_layer(layer: int, total_layers: int = 26) -> LayerCategory:
    """
    Categorize a layer index into early/mid/late.

    Default boundaries based on Gemma-3-4b-pt (26 layers):
    - early: 0-8 (first ~33%)
    - mid: 9-17 (middle ~33%)
    - late: 18-25 (last ~33%)

    Args:
        layer: Layer index
        total_layers: Total number of layers in the model

    Returns:
        Category string
    """
    third = total_layers // 3

    if layer < third:
        return "early"
    elif layer < 2 * third:
        return "mid"
    else:
        return "late"
