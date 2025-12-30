#!/usr/bin/env python3
"""
Lens pack version manifest for diff-based distribution.

The version_manifest.json tracks:
- Which concept pack version each lens was trained at
- When each lens was last trained
- Training quality metrics for each lens
- Which lenses changed between versions (for diff distribution)

Per MAP_MELD_PROTOCOL.md §8:
- Lens packs are rebuilt on any concept pack version change
- Unchanged lenses can persist across versions
- Diff-based distribution only sends changed lenses

Usage:
    from src.map.data.version_manifest import LensManifest

    manifest = LensManifest.load(lens_pack_dir)
    manifest.update_lens("Deception", "4.1.0", metrics={"f1": 0.92})
    manifest.save()

    # Get diff between versions
    diff = manifest.compute_diff("4.0.0", "4.1.0")
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any


@dataclass
class ClassifierEntry:
    """Metadata for a single classifier within a lens."""
    layer: int
    category: str  # "early", "mid", "late"
    technique: str = "mlp"
    metrics: Dict[str, float] = field(default_factory=dict)
    trained_at: str = ""
    file: str = ""  # Relative path to .pt file

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ClassifierEntry':
        return cls(**data)


def categorize_layer(layer: int, total_layers: int = 26) -> str:
    """Categorize layer as early/mid/late based on position."""
    third = total_layers // 3
    if layer < third:
        return "early"
    elif layer < 2 * third:
        return "mid"
    else:
        return "late"


@dataclass
class LensEntry:
    """
    Entry for a concept lens with potentially multiple classifiers.

    Supports both:
    - Legacy single-classifier mode (layer, lens_file fields)
    - Multi-classifier mode (classifiers dict)

    When loading, if classifiers is empty but layer is set, we auto-populate
    classifiers from the legacy fields.
    """
    concept: str
    trained_at_version: str  # Concept pack version when trained
    trained_timestamp: str  # ISO timestamp

    # Multi-classifier support: layer -> ClassifierEntry
    classifiers: Dict[int, ClassifierEntry] = field(default_factory=dict)

    # Which layer is the default/best for this concept
    default_layer: Optional[int] = None

    # Legacy fields - kept for backwards compatibility
    layer: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    training_samples: int = 0
    lens_file: str = ""

    def __post_init__(self):
        """Auto-populate classifiers from legacy fields if needed."""
        if not self.classifiers and self.layer > 0 and self.lens_file:
            # Legacy format - create single classifier entry
            self.classifiers[self.layer] = ClassifierEntry(
                layer=self.layer,
                category=categorize_layer(self.layer),
                technique="mlp",
                metrics=self.metrics.copy(),
                trained_at=self.trained_timestamp,
                file=self.lens_file,
            )
            if self.default_layer is None:
                self.default_layer = self.layer

    @property
    def layers(self) -> List[int]:
        """Get all layers with classifiers."""
        return sorted(self.classifiers.keys())

    @property
    def categories(self) -> Set[str]:
        """Get all categories with classifiers."""
        return set(c.category for c in self.classifiers.values())

    def get_best_layer(self, category: Optional[str] = None) -> Optional[int]:
        """
        Get best layer for a category based on metrics.

        Args:
            category: Optional filter by "early", "mid", or "late"

        Returns:
            Layer with best F1 score, or None if no classifiers
        """
        candidates = [
            (layer, clf)
            for layer, clf in self.classifiers.items()
            if category is None or clf.category == category
        ]

        if not candidates:
            return None

        # Sort by F1 score (descending), then layer (ascending)
        candidates.sort(key=lambda x: (-x[1].metrics.get("f1", 0), x[0]))
        return candidates[0][0]

    def add_classifier(
        self,
        layer: int,
        category: str,
        file: str,
        metrics: Optional[Dict[str, float]] = None,
        technique: str = "mlp",
        trained_at: Optional[str] = None,
    ):
        """Add or update a classifier for this concept."""
        self.classifiers[layer] = ClassifierEntry(
            layer=layer,
            category=category,
            technique=technique,
            metrics=metrics or {},
            trained_at=trained_at or self.trained_timestamp,
            file=file,
        )

        # Update default layer if this is first or better
        if self.default_layer is None:
            self.default_layer = layer
        elif metrics and metrics.get("f1", 0) > self.classifiers.get(
            self.default_layer, ClassifierEntry(0, "")
        ).metrics.get("f1", 0):
            self.default_layer = layer

        # Update legacy fields for backwards compat
        if self.layer == 0 or layer == self.default_layer:
            self.layer = layer
            self.lens_file = file
            if metrics:
                self.metrics = metrics

    def to_dict(self) -> Dict:
        d = {
            "concept": self.concept,
            "trained_at_version": self.trained_at_version,
            "trained_timestamp": self.trained_timestamp,
            "default_layer": self.default_layer,
            # Multi-classifier data
            "classifiers": {
                str(layer): clf.to_dict()
                for layer, clf in self.classifiers.items()
            },
            # Legacy fields for backwards compat
            "layer": self.layer,
            "metrics": self.metrics,
            "training_samples": self.training_samples,
            "lens_file": self.lens_file,
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'LensEntry':
        # Parse classifiers if present
        classifiers = {}
        if "classifiers" in data:
            for layer_str, clf_data in data["classifiers"].items():
                classifiers[int(layer_str)] = ClassifierEntry.from_dict(clf_data)

        return cls(
            concept=data.get("concept", ""),
            trained_at_version=data.get("trained_at_version", ""),
            trained_timestamp=data.get("trained_timestamp", ""),
            classifiers=classifiers,
            default_layer=data.get("default_layer"),
            layer=data.get("layer", 0),
            metrics=data.get("metrics", {}),
            training_samples=data.get("training_samples", 0),
            lens_file=data.get("lens_file", ""),
        )

    def to_hat_lens(self) -> 'Lens':
        """
        Convert to HAT Lens object for use in steering/monitoring.

        Returns:
            src.hat.lens.Lens object
        """
        from src.hat.classifiers.lens import Lens, ClassifierInfo
        from pathlib import Path

        lens = Lens(
            concept_name=self.concept,
            default_layer=self.default_layer,
        )

        for layer, clf in self.classifiers.items():
            lens.classifiers[layer] = ClassifierInfo(
                layer=layer,
                category=clf.category,
                technique=clf.technique,
                path=Path(clf.file) if clf.file else None,
                metrics=clf.metrics,
            )

        return lens


@dataclass
class VersionSnapshot:
    """Snapshot of lens states at a specific version."""
    version: str
    timestamp: str
    lenses_trained: List[str] = field(default_factory=list)  # New/retrained lenses
    lenses_unchanged: List[str] = field(default_factory=list)  # Persisted from prior
    total_lenses: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'VersionSnapshot':
        return cls(**data)


@dataclass
class VersionDiff:
    """Diff between two version snapshots for distribution."""
    from_version: str
    to_version: str
    added_lenses: List[str] = field(default_factory=list)
    retrained_lenses: List[str] = field(default_factory=list)
    removed_lenses: List[str] = field(default_factory=list)
    unchanged_lenses: List[str] = field(default_factory=list)

    @property
    def lenses_to_download(self) -> List[str]:
        """Lenses that need to be downloaded for upgrade."""
        return self.added_lenses + self.retrained_lenses

    @property
    def download_count(self) -> int:
        return len(self.lenses_to_download)

    @property
    def savings_percent(self) -> float:
        """Percentage of lenses that don't need downloading."""
        total = len(self.unchanged_lenses) + self.download_count + len(self.removed_lenses)
        if total == 0:
            return 0.0
        return (len(self.unchanged_lenses) / total) * 100

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["lenses_to_download"] = self.lenses_to_download
        d["download_count"] = self.download_count
        d["savings_percent"] = round(self.savings_percent, 1)
        return d


class LensManifest:
    """
    Manages version manifest for a lens pack.

    Structure:
    {
        "lens_pack_id": "org.hatcat/apertus-8b__sumo-wordnet-v4",
        "model": "swiss-ai/Apertus-8B-2509",
        "current_version": "4.1.0",
        "created": "2025-11-29T00:00:00Z",
        "updated": "2025-11-30T12:00:00Z",
        "lenses": {
            "Deception": {
                "concept": "Deception",
                "trained_at_version": "4.1.0",
                "trained_timestamp": "2025-11-30T12:00:00Z",
                "layer": 3,
                "metrics": {"f1": 0.92, "precision": 0.90, "recall": 0.94},
                "training_samples": 100,
                "lens_file": "layer3/Deception.pt"
            }
        },
        "version_history": [
            {
                "version": "4.0.0",
                "timestamp": "2025-11-29T00:00:00Z",
                "lenses_trained": ["Entity", "Object", ...],
                "lenses_unchanged": [],
                "total_lenses": 7000
            },
            {
                "version": "4.1.0",
                "timestamp": "2025-11-30T12:00:00Z",
                "lenses_trained": ["Deception", "DeceptionDetector", ...],
                "lenses_unchanged": ["Entity", "Object", ...],
                "total_lenses": 7050
            }
        ]
    }
    """

    def __init__(
        self,
        lens_pack_id: str = "",
        model: str = "",
        source_pack: str = "",
    ):
        self.lens_pack_id = lens_pack_id
        self.model = model
        self.source_pack = source_pack
        self.current_version = "0.0.0"
        self.created = datetime.now().isoformat() + "Z"
        self.updated = self.created
        self.lenses: Dict[str, LensEntry] = {}
        self.version_history: List[VersionSnapshot] = []
        self._path: Optional[Path] = None

    @classmethod
    def load(cls, lens_pack_dir: Path) -> 'LensManifest':
        """Load manifest from lens pack directory."""
        manifest_path = lens_pack_dir / "version_manifest.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)

            manifest = cls(
                lens_pack_id=data.get("lens_pack_id", ""),
                model=data.get("model", ""),
                source_pack=data.get("source_pack", ""),
            )
            manifest.current_version = data.get("current_version", "0.0.0")
            manifest.created = data.get("created", manifest.created)
            manifest.updated = data.get("updated", manifest.updated)

            for name, lens_data in data.get("lenses", {}).items():
                manifest.lenses[name] = LensEntry.from_dict(lens_data)

            for snap_data in data.get("version_history", []):
                manifest.version_history.append(VersionSnapshot.from_dict(snap_data))
        else:
            # Initialize from pack_info.json if available
            pack_info_path = lens_pack_dir / "pack_info.json"
            if pack_info_path.exists():
                with open(pack_info_path) as f:
                    info = json.load(f)
                manifest = cls(
                    lens_pack_id=f"{info.get('source_pack', '')}_{info.get('model', '').replace('/', '_')}",
                    model=info.get("model", ""),
                    source_pack=info.get("source_pack", ""),
                )
                manifest.current_version = info.get("pack_version", "0.0.0")
            else:
                manifest = cls()

        manifest._path = manifest_path
        return manifest

    def save(self, path: Optional[Path] = None):
        """Save manifest to file."""
        save_path = path or self._path
        if not save_path:
            raise ValueError("No path specified for saving manifest")

        self.updated = datetime.now().isoformat() + "Z"

        data = {
            "lens_pack_id": self.lens_pack_id,
            "model": self.model,
            "source_pack": self.source_pack,
            "current_version": self.current_version,
            "created": self.created,
            "updated": self.updated,
            "lenses": {name: p.to_dict() for name, p in self.lenses.items()},
            "version_history": [s.to_dict() for s in self.version_history],
            "summary": {
                "total_lenses": len(self.lenses),
                "versions_tracked": len(self.version_history),
                "layers": sorted(set(p.layer for p in self.lenses.values()))
            }
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def update_lens(
        self,
        concept: str,
        version: str,
        layer: int,
        metrics: Optional[Dict[str, float]] = None,
        training_samples: int = 0,
        lens_file: str = "",
        category: Optional[str] = None,
        technique: str = "mlp",
        total_layers: int = 26,
    ):
        """
        Update or add a classifier for a concept.

        If the concept already has a LensEntry, adds the classifier to it.
        Otherwise creates a new LensEntry.

        Args:
            concept: Concept name
            version: Concept pack version
            layer: Model layer this classifier was trained on
            metrics: Training metrics (f1, precision, recall)
            training_samples: Number of training samples
            lens_file: Path to .pt file (relative to lens pack)
            category: Layer category (auto-detected if not provided)
            technique: Classifier technique ("mlp", "linear")
            total_layers: Total layers in model (for category detection)
        """
        timestamp = datetime.now().isoformat() + "Z"
        lens_file = lens_file or f"layer{layer}/{concept}_classifier.pt"

        # Auto-detect category if not provided
        if category is None:
            category = categorize_layer(layer, total_layers)

        if concept in self.lenses:
            # Add classifier to existing entry
            entry = self.lenses[concept]
            entry.trained_at_version = version
            entry.add_classifier(
                layer=layer,
                category=category,
                file=lens_file,
                metrics=metrics,
                technique=technique,
                trained_at=timestamp,
            )
            entry.training_samples += training_samples
        else:
            # Create new entry
            entry = LensEntry(
                concept=concept,
                trained_at_version=version,
                trained_timestamp=timestamp,
                layer=layer,
                metrics=metrics or {},
                training_samples=training_samples,
                lens_file=lens_file,
            )
            # Add classifier explicitly
            entry.add_classifier(
                layer=layer,
                category=category,
                file=lens_file,
                metrics=metrics,
                technique=technique,
                trained_at=timestamp,
            )
            self.lenses[concept] = entry

        self.current_version = version

    def record_training_run(
        self,
        version: str,
        trained_concepts: List[str],
        all_concepts: Optional[List[str]] = None
    ):
        """Record a training run in version history."""
        trained_set = set(trained_concepts)

        # Determine unchanged lenses
        if all_concepts:
            all_set = set(all_concepts)
            unchanged = sorted(all_set - trained_set)
        else:
            # Infer from existing lenses that weren't retrained
            unchanged = sorted([
                name for name, p in self.lenses.items()
                if name not in trained_set and p.trained_at_version != version
            ])

        snapshot = VersionSnapshot(
            version=version,
            timestamp=datetime.now().isoformat() + "Z",
            lenses_trained=sorted(trained_concepts),
            lenses_unchanged=unchanged,
            total_lenses=len(self.lenses)
        )

        self.version_history.append(snapshot)
        self.current_version = version

    def get_lenses_at_version(self, version: str) -> Dict[str, LensEntry]:
        """Get all lenses that were current at a specific version."""
        result = {}
        for name, lens in self.lenses.items():
            # A lens is "at" a version if it was trained at that version
            # or at any prior version (and not retrained since)
            if self._version_lte(lens.trained_at_version, version):
                result[name] = lens
        return result

    def compute_diff(self, from_version: str, to_version: str) -> VersionDiff:
        """
        Compute the diff between two versions.

        Returns lenses that were added, retrained, removed, or unchanged.
        This enables efficient diff-based distribution.
        """
        from_lenses = self.get_lenses_at_version(from_version)
        to_lenses = self.get_lenses_at_version(to_version)

        from_names = set(from_lenses.keys())
        to_names = set(to_lenses.keys())

        added = sorted(to_names - from_names)
        removed = sorted(from_names - to_names)

        # Check which existing lenses were retrained
        common = from_names & to_names
        retrained = []
        unchanged = []

        for name in sorted(common):
            from_lens = from_lenses[name]
            to_lens = to_lenses[name]

            # If trained_at_version changed, it was retrained
            if from_lens.trained_at_version != to_lens.trained_at_version:
                retrained.append(name)
            else:
                unchanged.append(name)

        return VersionDiff(
            from_version=from_version,
            to_version=to_version,
            added_lenses=added,
            retrained_lenses=retrained,
            removed_lenses=removed,
            unchanged_lenses=unchanged
        )

    def get_lenses_needing_training(self, pending_concepts: List[str]) -> Dict[str, str]:
        """
        Given a list of concepts pending training, return which need training.

        Returns dict mapping concept -> reason (new, must_retrain, should_retrain)
        """
        result = {}
        existing = set(self.lenses.keys())

        for concept in pending_concepts:
            if concept not in existing:
                result[concept] = "new"
            else:
                result[concept] = "retrain"

        return result

    def _version_lte(self, v1: str, v2: str) -> bool:
        """Check if version v1 <= v2."""
        try:
            parts1 = [int(x) for x in v1.split(".")]
            parts2 = [int(x) for x in v2.split(".")]
            return parts1 <= parts2
        except (ValueError, AttributeError):
            return True


def generate_manifest_from_training(
    lens_pack_dir: Path,
    version: str,
    training_results: Dict[str, Dict],
    source_pack: str = "",
    model: str = ""
) -> LensManifest:
    """
    Generate or update manifest after a training run.

    Args:
        lens_pack_dir: Path to lens pack
        version: Concept pack version lenses were trained for
        training_results: Dict mapping concept -> training result
            Expected keys: layer, metrics (f1, precision, recall), n_samples
        source_pack: Source concept pack ID
        model: Model name

    Returns:
        Updated manifest
    """
    manifest = LensManifest.load(lens_pack_dir)

    if source_pack:
        manifest.source_pack = source_pack
    if model:
        manifest.model = model

    trained_concepts = []

    for concept, result in training_results.items():
        manifest.update_lens(
            concept=concept,
            version=version,
            layer=result.get("layer", 0),
            metrics=result.get("metrics", {}),
            training_samples=result.get("n_samples", 0),
            lens_file=result.get("lens_file", f"layer{result.get('layer', 0)}/{concept}.pt")
        )
        trained_concepts.append(concept)

    manifest.record_training_run(
        version=version,
        trained_concepts=trained_concepts,
        all_concepts=list(manifest.lenses.keys())
    )

    return manifest


def print_manifest_summary(manifest: LensManifest):
    """Print a summary of the manifest."""
    print(f"\nLens Pack Manifest: {manifest.lens_pack_id}")
    print(f"  Model: {manifest.model}")
    print(f"  Source: {manifest.source_pack}")
    print(f"  Version: {manifest.current_version}")
    print(f"  Total lenses: {len(manifest.lenses)}")

    if manifest.version_history:
        print(f"\nVersion History ({len(manifest.version_history)} versions):")
        for snap in manifest.version_history[-3:]:  # Last 3
            print(f"  {snap.version}: {len(snap.lenses_trained)} trained, "
                  f"{len(snap.lenses_unchanged)} unchanged")


def print_diff_summary(diff: VersionDiff):
    """Print a summary of a version diff."""
    print(f"\nDiff: {diff.from_version} → {diff.to_version}")
    print(f"  Added: {len(diff.added_lenses)}")
    print(f"  Retrained: {len(diff.retrained_lenses)}")
    print(f"  Removed: {len(diff.removed_lenses)}")
    print(f"  Unchanged: {len(diff.unchanged_lenses)}")
    print(f"  Download: {diff.download_count} lenses")
    print(f"  Savings: {diff.savings_percent:.1f}%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.data.version_manifest <lens_pack_dir> [--diff from_ver to_ver]")
        sys.exit(1)

    lens_pack_dir = Path(sys.argv[1])
    manifest = LensManifest.load(lens_pack_dir)
    print_manifest_summary(manifest)

    if len(sys.argv) >= 5 and sys.argv[2] == "--diff":
        from_ver = sys.argv[3]
        to_ver = sys.argv[4]
        diff = manifest.compute_diff(from_ver, to_ver)
        print_diff_summary(diff)
