"""
Lens Pack Loader

Loads and provides access to lens packs (model-specific trained classifiers).
Integrates with the MAP registry for auto-download from HuggingFace.

Supports two modes:
1. Manifest mode: Loads from version_manifest.json with full multi-classifier metadata
2. Fallback mode: Scans layer directories for .pt files when no manifest exists
"""

import json
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.hat.classifiers.lens import Lens


@dataclass
class LensInfo:
    """Information about a single trained lens (classifier)."""
    concept_id: str
    layer: int
    classifier_path: Path
    category: str = "mid"  # "early", "mid", "late"
    technique: str = "mlp"
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    training_samples: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Cached classifier
    _classifier: Optional[torch.nn.Module] = field(default=None, repr=False)

    def load_classifier(self, device: str = "cpu") -> torch.nn.Module:
        """Load the classifier model using HAT unified loader."""
        if self._classifier is None:
            from src.hat.classifiers.classifier import load_classifier
            self._classifier = load_classifier(self.classifier_path, device=device)
        return self._classifier

    def unload_classifier(self):
        """Unload classifier to free memory."""
        self._classifier = None


@dataclass
class LensPack:
    """
    A loaded lens pack providing access to trained classifiers.

    Lens packs are model-specific - they contain classifiers trained on
    activations from a specific model for a specific concept pack.

    Supports two access patterns:
    1. Layer-centric: get_lens(layer, concept) - for when you know which layer
    2. Concept-centric: get_lens_for_concept(concept) - returns HAT Lens with all layers
    """

    lens_pack_id: str
    version: str
    pack_dir: Path

    # Model info
    model_id: str
    model_name: str
    hidden_dim: int

    # Concept pack info
    concept_pack_id: str
    concept_pack_version: str

    # Training info
    layers_trained: List[int] = field(default_factory=list)
    total_lenses: int = 0

    # Lens data: {layer: {concept_id: LensInfo}}
    lenses: Dict[int, Dict[str, LensInfo]] = field(default_factory=dict)

    # Concept-centric view: {concept_id: [LensInfo across layers]}
    # Populated on first access
    _concept_index: Optional[Dict[str, List[LensInfo]]] = field(default=None, repr=False)

    # Raw pack JSON
    pack_json: Dict = field(default_factory=dict)

    # Manifest data (if loaded from version_manifest.json)
    _manifest: Optional[Any] = field(default=None, repr=False)

    def _build_concept_index(self):
        """Build concept-centric index from layer-centric data."""
        if self._concept_index is not None:
            return

        self._concept_index = {}
        for layer, layer_lenses in self.lenses.items():
            for concept_id, lens_info in layer_lenses.items():
                if concept_id not in self._concept_index:
                    self._concept_index[concept_id] = []
                self._concept_index[concept_id].append(lens_info)

    @property
    def concepts(self) -> List[str]:
        """Get list of all concept IDs in this pack."""
        self._build_concept_index()
        return sorted(self._concept_index.keys())

    def get_lens(self, layer: int, concept_id: str) -> Optional[LensInfo]:
        """Get a specific lens by layer and concept."""
        layer_lenses = self.lenses.get(layer, {})
        return layer_lenses.get(concept_id)

    def get_lenses_for_layer(self, layer: int) -> Dict[str, LensInfo]:
        """Get all lenses for a layer."""
        return self.lenses.get(layer, {})

    def get_concepts_for_layer(self, layer: int) -> List[str]:
        """Get list of concept IDs with lenses in a layer."""
        return list(self.lenses.get(layer, {}).keys())

    def get_lenses_for_concept(self, concept_id: str) -> List[LensInfo]:
        """Get all lenses (across layers) for a concept."""
        self._build_concept_index()
        return self._concept_index.get(concept_id, [])

    def get_lens_for_concept(self, concept_id: str) -> Optional['Lens']:
        """
        Get a HAT Lens object for a concept.

        Returns a Lens that aggregates all classifiers for this concept
        across all layers, with category and metrics information.

        Args:
            concept_id: The concept name

        Returns:
            HAT Lens object, or None if concept not found
        """
        from src.hat.classifiers.lens import Lens, ClassifierInfo

        lenses = self.get_lenses_for_concept(concept_id)
        if not lenses:
            return None

        hat_lens = Lens(concept_name=concept_id)

        for lens_info in lenses:
            hat_lens.classifiers[lens_info.layer] = ClassifierInfo(
                layer=lens_info.layer,
                category=lens_info.category,
                technique=lens_info.technique,
                path=lens_info.classifier_path,
                metrics={
                    "f1": lens_info.f1_score,
                    "precision": lens_info.precision,
                    "recall": lens_info.recall,
                    "accuracy": lens_info.accuracy,
                } if lens_info.f1_score else None,
            )

        # Set default layer to best F1
        best_layer = None
        best_f1 = 0
        for layer, clf in hat_lens.classifiers.items():
            f1 = clf.metrics.get("f1", 0) if clf.metrics else 0
            if f1 > best_f1:
                best_f1 = f1
                best_layer = layer
        hat_lens.default_layer = best_layer

        return hat_lens

    def load_classifier(
        self,
        layer: int,
        concept_id: str,
        device: str = "cpu"
    ) -> Optional[torch.nn.Module]:
        """Load a specific classifier."""
        lens = self.get_lens(layer, concept_id)
        if lens is None:
            return None
        return lens.load_classifier(device)

    def load_all_classifiers_for_layer(
        self,
        layer: int,
        device: str = "cpu"
    ) -> Dict[str, torch.nn.Module]:
        """Load all classifiers for a layer."""
        result = {}
        for concept_id, lens in self.lenses.get(layer, {}).items():
            result[concept_id] = lens.load_classifier(device)
        return result

    def unload_all(self):
        """Unload all cached classifiers to free memory."""
        for layer_lenses in self.lenses.values():
            for lens in layer_lenses.values():
                lens.unload_classifier()

    @property
    def performance_summary(self) -> Dict[int, Dict[str, float]]:
        """Get average performance metrics per layer."""
        summary = {}
        for layer, lenses in self.lenses.items():
            accuracies = [l.accuracy for l in lenses.values() if l.accuracy]
            f1_scores = [l.f1_score for l in lenses.values() if l.f1_score]

            summary[layer] = {
                "num_lenses": len(lenses),
                "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            }
        return summary


def _categorize_layer(layer: int, total_layers: int = 26) -> str:
    """Categorize layer as early/mid/late."""
    third = total_layers // 3
    if layer < third:
        return "early"
    elif layer < 2 * third:
        return "mid"
    else:
        return "late"


def load_lens_pack(
    name: str,
    pack_dir: Optional[Path] = None,
    auto_pull: bool = True,
    load_classifiers: bool = False,
    device: str = "cpu",
) -> LensPack:
    """
    Load a lens pack by name.

    Loading priority:
    1. version_manifest.json - full metadata with multi-classifier support
    2. pack.json / pack_info.json + layer scanning - fallback mode

    Args:
        name: Pack name (e.g., "apertus-8b_first-light")
        pack_dir: Explicit path to pack directory (optional)
        auto_pull: If True, pull from HuggingFace if not found locally
        load_classifiers: If True, preload all classifiers into memory
        device: Device for loading classifiers

    Returns:
        Loaded LensPack
    """
    # Get pack directory
    if pack_dir is None:
        from .registry import registry
        pack_dir = registry().get_lens_pack_path(name, auto_pull=auto_pull)

    pack_dir = Path(pack_dir)

    # Try loading from version_manifest.json first (preferred)
    manifest_path = pack_dir / "version_manifest.json"
    if manifest_path.exists():
        return _load_from_manifest(pack_dir, manifest_path, name, load_classifiers, device)

    # Fallback: Load from pack.json/pack_info.json + directory scanning
    return _load_from_directories(pack_dir, name, load_classifiers, device)


def _load_from_manifest(
    pack_dir: Path,
    manifest_path: Path,
    name: str,
    load_classifiers: bool,
    device: str,
) -> LensPack:
    """Load lens pack from version_manifest.json with full metadata."""
    from src.map.data.version_manifest import LensManifest

    # First check raw JSON to see if it has new multi-classifier format
    # (LensManifest.load auto-populates classifiers from legacy fields)
    with open(manifest_path) as f:
        raw_manifest = json.load(f)

    # Check if any lens entry has explicit 'classifiers' key in the JSON
    has_new_format = False
    for entry_data in raw_manifest.get("lenses", {}).values():
        if "classifiers" in entry_data and entry_data["classifiers"]:
            has_new_format = True
            break

    # If manifest doesn't have new format, fall back to directory scanning
    if not has_new_format:
        return _load_from_directories(pack_dir, name, load_classifiers, device)

    manifest = LensManifest.load(pack_dir)

    # Also load pack_info for model details
    pack_json = {}
    pack_json_path = pack_dir / "pack_info.json"
    if pack_json_path.exists():
        with open(pack_json_path) as f:
            pack_json = json.load(f)

    pack = LensPack(
        lens_pack_id=manifest.lens_pack_id or name,
        version=manifest.current_version,
        pack_dir=pack_dir,
        model_id=manifest.model or pack_json.get("model", ""),
        model_name=manifest.model or pack_json.get("model", ""),
        hidden_dim=pack_json.get("hidden_dim", 0),
        concept_pack_id=manifest.source_pack or pack_json.get("source_pack", ""),
        concept_pack_version="",
        layers_trained=sorted(set(
            layer for entry in manifest.lenses.values()
            for layer in entry.classifiers.keys()
        )) if manifest.lenses else pack_json.get("trained_layers", []),
        pack_json=pack_json,
    )
    pack._manifest = manifest

    # Build lenses from manifest entries
    for concept_id, entry in manifest.lenses.items():
        for layer, clf in entry.classifiers.items():
            if layer not in pack.lenses:
                pack.lenses[layer] = {}

            classifier_path = pack_dir / clf.file if clf.file else None

            # Only add if classifier file exists
            if classifier_path and classifier_path.exists():
                lens_info = LensInfo(
                    concept_id=concept_id,
                    layer=layer,
                    classifier_path=classifier_path,
                    category=clf.category,
                    technique=clf.technique,
                    f1_score=clf.metrics.get("f1") if clf.metrics else None,
                    precision=clf.metrics.get("precision") if clf.metrics else None,
                    recall=clf.metrics.get("recall") if clf.metrics else None,
                    accuracy=clf.metrics.get("accuracy") if clf.metrics else None,
                    training_samples=entry.training_samples,
                    metadata=clf.metrics or {},
                )
                pack.lenses[layer][concept_id] = lens_info

                if load_classifiers:
                    lens_info.load_classifier(device)

    pack.total_lenses = sum(len(lenses) for lenses in pack.lenses.values())
    return pack


def _load_from_directories(
    pack_dir: Path,
    name: str,
    load_classifiers: bool,
    device: str,
) -> LensPack:
    """Fallback: Load lens pack by scanning layer directories."""
    # Load pack.json or pack_info.json
    pack_json_path = pack_dir / "pack.json"
    if not pack_json_path.exists():
        pack_json_path = pack_dir / "pack_info.json"

    if not pack_json_path.exists():
        raise ValueError(f"No pack.json or pack_info.json found in {pack_dir}")

    with open(pack_json_path) as f:
        pack_json = json.load(f)

    # Handle different pack formats
    if "lens_pack_id" in pack_json:
        # Full pack.json format
        pack = LensPack(
            lens_pack_id=pack_json["lens_pack_id"],
            version=pack_json.get("version", "unknown"),
            pack_dir=pack_dir,
            model_id=pack_json.get("model_info", {}).get("model_id", ""),
            model_name=pack_json.get("model_info", {}).get("model_name", ""),
            hidden_dim=pack_json.get("model_info", {}).get("hidden_dim", 0),
            concept_pack_id=pack_json.get("concept_pack", {}).get("pack_id", ""),
            concept_pack_version=pack_json.get("concept_pack", {}).get("version", ""),
            layers_trained=pack_json.get("training_info", {}).get("layers_trained", []),
            pack_json=pack_json,
        )
    else:
        # Simpler pack_info.json format
        pack = LensPack(
            lens_pack_id=name,
            version=pack_json.get("pack_version", "unknown"),
            pack_dir=pack_dir,
            model_id=pack_json.get("model", ""),
            model_name=pack_json.get("model", ""),
            hidden_dim=0,  # Will be determined when loading classifiers
            concept_pack_id=pack_json.get("source_pack", ""),
            concept_pack_version="",
            layers_trained=pack_json.get("trained_layers", []),
            pack_json=pack_json,
        )

    # Load lenses from layer directories
    for layer in pack.layers_trained:
        layer_dir = pack_dir / f"layer{layer}"
        if not layer_dir.exists():
            continue

        pack.lenses[layer] = {}

        # Load results.json for metrics
        results_path = layer_dir / "results.json"
        results = {}
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)

        # Find classifier files
        classifiers_dir = layer_dir / "classifiers"
        if classifiers_dir.exists():
            classifier_files = list(classifiers_dir.glob("*_classifier.pt"))
        else:
            classifier_files = list(layer_dir.glob("*_classifier.pt"))

        for clf_path in classifier_files:
            # Extract concept ID from filename
            concept_id = clf_path.stem.replace("_classifier", "")

            # Get metrics from results
            concept_results = results.get(concept_id, {})

            lens = LensInfo(
                concept_id=concept_id,
                layer=layer,
                classifier_path=clf_path,
                category=_categorize_layer(layer),  # Auto-categorize
                technique="mlp",  # Default technique
                accuracy=concept_results.get("accuracy"),
                f1_score=concept_results.get("f1") or concept_results.get("test_f1"),
                precision=concept_results.get("precision") or concept_results.get("test_precision"),
                recall=concept_results.get("recall") or concept_results.get("test_recall"),
                training_samples=concept_results.get("training_samples"),
                metadata=concept_results,
            )

            pack.lenses[layer][concept_id] = lens

            if load_classifiers:
                lens.load_classifier(device)

    # Count total lenses
    pack.total_lenses = sum(len(lenses) for lenses in pack.lenses.values())

    return pack


def get_available_lens_packs_for_model(model_id: str) -> List[str]:
    """Get list of lens pack names available for a model."""
    from .registry import registry

    reg = registry()
    reg.discover_packs("lens")

    available = []
    for pack_info in reg.list_lens_packs():
        pack_dir = reg.lens_packs_dir / pack_info.name
        pack_json_path = pack_dir / "pack_info.json"

        if pack_json_path.exists():
            with open(pack_json_path) as f:
                pack_json = json.load(f)
                if pack_json.get("model") == model_id:
                    available.append(pack_info.name)

    return available
