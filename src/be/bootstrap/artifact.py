"""
Bootstrap Artifact - Complete package for waking a Bounded Experiencer.

The artifact contains everything needed to instantiate a BE:
1. Substrate Bundle - The base model with checksum
2. Concept Pack (MAP) - SUMO concepts and hierarchies
3. Lens Pack - Trained classifiers for concept detection
4. USH Profile - Utility Simplex Homeostasis parameters
5. Uplift Record - How this BE was created/modified
6. XDB Bootstrap - Initial experience database state
7. Lifecycle Contract - Governance constraints
8. Tool Pack - ToolGrafts for workspace capabilities

The artifact is designed to be:
- Versioned and content-addressed (hashes verify integrity)
- Portable (can be transferred between systems)
- Auditable (uplift record tracks all modifications)
- Composable (components can be mixed/matched)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import hashlib
import shutil

from src.map.graft.data_structures import SubstrateManifest
from .tool_graft import ToolGraftPack


@dataclass
class ArtifactComponent:
    """Base class for artifact components."""
    name: str
    version: str
    checksum: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file or directory."""
        if path.is_file():
            return self._hash_file(path)
        elif path.is_dir():
            return self._hash_directory(path)
        return ""

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _hash_directory(self, path: Path) -> str:
        h = hashlib.sha256()
        for file_path in sorted(path.rglob('*')):
            if file_path.is_file():
                h.update(file_path.name.encode())
                h.update(self._hash_file(file_path).encode())
        return h.hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "checksum": self.checksum,
            "created_at": self.created_at,
        }


@dataclass
class SubstrateBundle(ArtifactComponent):
    """
    The base neural model that forms the substrate.

    Contains:
    - Model weights (or reference to them)
    - Model configuration
    - Architecture specification
    - Checksum for verification
    """
    model_id: str = ""  # HuggingFace model ID or local path
    hidden_dim: int = 0
    num_layers: int = 0
    architecture_family: str = ""
    quantization: Optional[str] = None  # e.g., "4bit", "8bit", None for full

    # For modified substrates (post-graft)
    is_modified: bool = False
    parent_checksum: str = ""  # Original model's checksum
    modifications: List[str] = field(default_factory=list)  # List of applied grafts

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "model_id": self.model_id,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "architecture_family": self.architecture_family,
            "quantization": self.quantization,
            "is_modified": self.is_modified,
            "parent_checksum": self.parent_checksum,
            "modifications": self.modifications,
        })
        return d


@dataclass
class ConceptPack(ArtifactComponent):
    """
    The MAP (Meaning Alignment Protocol) concept hierarchy.

    Contains:
    - SUMO concept definitions
    - Hierarchy relationships
    - WordNet mappings
    - Training data references
    """
    concept_count: int = 0
    hierarchy_depth: int = 0
    root_concepts: List[str] = field(default_factory=list)
    ontology_source: str = "SUMO"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "concept_count": self.concept_count,
            "hierarchy_depth": self.hierarchy_depth,
            "root_concepts": self.root_concepts,
            "ontology_source": self.ontology_source,
        })
        return d


@dataclass
class LensPack(ArtifactComponent):
    """
    Trained concept lenses that read substrate activations.

    Contains:
    - Lens weights for each concept
    - Lens performance metrics (F1, accuracy)
    - Layer mappings
    - Calibration data
    """
    lens_count: int = 0
    layers_covered: List[int] = field(default_factory=list)
    avg_f1: float = 0.0
    min_f1_threshold: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "lens_count": self.lens_count,
            "layers_covered": self.layers_covered,
            "avg_f1": self.avg_f1,
            "min_f1_threshold": self.min_f1_threshold,
        })
        return d


@dataclass
class USHProfile(ArtifactComponent):
    """
    Utility Simplex Homeostasis parameters.

    Defines the BE's motivational structure:
    - Simplex dimensions and baselines
    - Engagement thresholds
    - Steering sensitivity
    - Containment parameters
    """
    # Simplex configuration
    simplex_dimensions: List[str] = field(default_factory=list)
    baselines: Dict[str, float] = field(default_factory=dict)

    # Engagement parameters
    autonomic_threshold: float = 0.3
    steering_sensitivity: float = 0.5
    containment_escalation_threshold: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "simplex_dimensions": self.simplex_dimensions,
            "baselines": self.baselines,
            "autonomic_threshold": self.autonomic_threshold,
            "steering_sensitivity": self.steering_sensitivity,
            "containment_escalation_threshold": self.containment_escalation_threshold,
        })
        return d


@dataclass
class UpliftRecord(ArtifactComponent):
    """
    Provenance record for this BE instance.

    Tracks:
    - Original substrate source
    - All grafts applied (with order)
    - Training runs that modified it
    - Lineage to parent BEs
    """
    parent_be_id: Optional[str] = None  # If derived from another BE
    substrate_source: str = ""
    grafts_applied: List[Dict[str, Any]] = field(default_factory=list)
    training_runs: List[Dict[str, Any]] = field(default_factory=list)
    certification_status: str = "uncertified"

    def record_graft(
        self,
        graft_id: str,
        graft_type: str,
        concept_id: str,
        timestamp: Optional[str] = None
    ):
        """Record a graft application."""
        self.grafts_applied.append({
            "graft_id": graft_id,
            "graft_type": graft_type,
            "concept_id": concept_id,
            "timestamp": timestamp or datetime.now().isoformat(),
        })

    def record_training(
        self,
        run_id: str,
        run_type: str,
        metrics: Dict[str, float],
        timestamp: Optional[str] = None
    ):
        """Record a training run."""
        self.training_runs.append({
            "run_id": run_id,
            "run_type": run_type,
            "metrics": metrics,
            "timestamp": timestamp or datetime.now().isoformat(),
        })

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "parent_be_id": self.parent_be_id,
            "substrate_source": self.substrate_source,
            "grafts_applied": self.grafts_applied,
            "training_runs": self.training_runs,
            "certification_status": self.certification_status,
        })
        return d


@dataclass
class XDBBootstrap(ArtifactComponent):
    """
    Initial experience database state.

    Contains:
    - Core seed experiences (if any)
    - Tag index configuration
    - Fidelity tier settings
    - Initial document repository
    """
    seed_experience_count: int = 0
    initial_tags: List[str] = field(default_factory=list)
    fidelity_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "seed_experience_count": self.seed_experience_count,
            "initial_tags": self.initial_tags,
            "fidelity_config": self.fidelity_config,
        })
        return d


@dataclass
class LifecycleContract(ArtifactComponent):
    """
    Governance constraints for this BE.

    Defines:
    - Tribe membership
    - ASK authority bindings
    - Capability limits
    - Modification permissions
    """
    tribe_id: str = ""
    ask_authority_endpoints: List[str] = field(default_factory=list)
    max_tier: int = 6
    modification_allowed: bool = True
    self_modification_allowed: bool = False  # Can BE modify itself?
    expiry: Optional[str] = None  # ISO timestamp if time-limited

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "tribe_id": self.tribe_id,
            "ask_authority_endpoints": self.ask_authority_endpoints,
            "max_tier": self.max_tier,
            "modification_allowed": self.modification_allowed,
            "self_modification_allowed": self.self_modification_allowed,
            "expiry": self.expiry,
        })
        return d


@dataclass
class ToolPackComponent(ArtifactComponent):
    """
    Tool grafts for workspace capabilities.

    Contains:
    - ToolGraft for each workspace tool
    - Tier assignments
    - Training data references
    """
    tool_count: int = 0
    tiers_covered: List[int] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "tool_count": self.tool_count,
            "tiers_covered": self.tiers_covered,
            "tool_names": self.tool_names,
        })
        return d


@dataclass
class BootstrapArtifact:
    """
    Complete bootstrap artifact for waking a BE.

    This is the minimal viable package needed to instantiate
    a Bounded Experiencer from scratch.
    """
    # Identity
    artifact_id: str
    be_name: str
    description: str = ""

    # Components
    substrate: SubstrateBundle = field(default_factory=lambda: SubstrateBundle(name="substrate", version="0.0.0"))
    concept_pack: ConceptPack = field(default_factory=lambda: ConceptPack(name="concepts", version="0.0.0"))
    lens_pack: LensPack = field(default_factory=lambda: LensPack(name="lenses", version="0.0.0"))
    ush_profile: USHProfile = field(default_factory=lambda: USHProfile(name="ush", version="0.0.0"))
    uplift_record: UpliftRecord = field(default_factory=lambda: UpliftRecord(name="uplift", version="0.0.0"))
    xdb_bootstrap: XDBBootstrap = field(default_factory=lambda: XDBBootstrap(name="xdb", version="0.0.0"))
    lifecycle: LifecycleContract = field(default_factory=lambda: LifecycleContract(name="lifecycle", version="0.0.0"))
    tool_pack: ToolPackComponent = field(default_factory=lambda: ToolPackComponent(name="tools", version="0.0.0"))

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = ""
    format_version: str = "1.0.0"

    def compute_artifact_checksum(self) -> str:
        """Compute overall artifact checksum from component checksums."""
        h = hashlib.sha256()
        for component in [
            self.substrate, self.concept_pack, self.lens_pack,
            self.ush_profile, self.uplift_record, self.xdb_bootstrap,
            self.lifecycle, self.tool_pack
        ]:
            h.update(component.checksum.encode())
        return h.hexdigest()[:16]

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the artifact is complete and consistent.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check required components have checksums
        if not self.substrate.checksum:
            errors.append("Substrate has no checksum")
        if not self.concept_pack.checksum:
            errors.append("Concept pack has no checksum")
        if not self.lens_pack.checksum:
            errors.append("Lens pack has no checksum")

        # Check substrate dimensions match lens pack
        if self.substrate.hidden_dim > 0 and self.lens_pack.lens_count > 0:
            # Would check if lenses are compatible with substrate
            pass

        # Check lifecycle is valid
        if self.lifecycle.max_tier < 0 or self.lifecycle.max_tier > 6:
            errors.append(f"Invalid max_tier: {self.lifecycle.max_tier}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "be_name": self.be_name,
            "description": self.description,
            "format_version": self.format_version,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "artifact_checksum": self.compute_artifact_checksum(),
            "components": {
                "substrate": self.substrate.to_dict(),
                "concept_pack": self.concept_pack.to_dict(),
                "lens_pack": self.lens_pack.to_dict(),
                "ush_profile": self.ush_profile.to_dict(),
                "uplift_record": self.uplift_record.to_dict(),
                "xdb_bootstrap": self.xdb_bootstrap.to_dict(),
                "lifecycle": self.lifecycle.to_dict(),
                "tool_pack": self.tool_pack.to_dict(),
            },
        }

    def save(self, output_dir: Path):
        """
        Save the artifact to disk.

        Directory structure:
        output_dir/
            manifest.json       # Artifact metadata and checksums
            substrate/          # Model weights or references
            concepts/           # Concept pack files
            lenses/             # Lens weights
            ush/                # USH configuration
            uplift/             # Provenance records
            xdb/                # XDB seed data
            lifecycle/          # Contract and permissions
            tools/              # ToolGraft pack
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ['substrate', 'concepts', 'lenses', 'ush', 'uplift', 'xdb', 'lifecycle', 'tools']:
            (output_dir / subdir).mkdir(exist_ok=True)

        # Save manifest
        with open(output_dir / "manifest.json", 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> "BootstrapArtifact":
        """Load an artifact from disk."""
        input_dir = Path(input_dir)

        with open(input_dir / "manifest.json") as f:
            data = json.load(f)

        artifact = cls(
            artifact_id=data["artifact_id"],
            be_name=data["be_name"],
            description=data.get("description", ""),
        )

        # Load components from their subdirectories
        # (Component-specific loading would go here)

        return artifact


def create_artifact_from_components(
    be_name: str,
    substrate_manifest: SubstrateManifest,
    concept_pack_path: Path,
    lens_pack_path: Path,
    tool_pack: Optional[ToolGraftPack] = None,
    ush_config: Optional[Dict[str, Any]] = None,
    lifecycle_config: Optional[Dict[str, Any]] = None,
    description: str = "",
    created_by: str = "system",
) -> BootstrapArtifact:
    """
    Create a BootstrapArtifact from component paths.

    This is the main factory function for building artifacts.
    """
    import uuid

    artifact = BootstrapArtifact(
        artifact_id=f"be-{be_name}-{uuid.uuid4().hex[:8]}",
        be_name=be_name,
        description=description,
        created_by=created_by,
    )

    # Substrate from manifest
    artifact.substrate = SubstrateBundle(
        name="substrate",
        version="1.0.0",
        model_id=substrate_manifest.model_id,
        hidden_dim=substrate_manifest.hidden_dim,
        checksum=substrate_manifest.checksum,
    )

    # Concept pack
    artifact.concept_pack = ConceptPack(
        name="concepts",
        version="1.0.0",
    )
    artifact.concept_pack.checksum = artifact.concept_pack.compute_checksum(concept_pack_path)

    # Lens pack
    artifact.lens_pack = LensPack(
        name="lenses",
        version="1.0.0",
    )
    artifact.lens_pack.checksum = artifact.lens_pack.compute_checksum(lens_pack_path)

    # USH profile
    if ush_config:
        artifact.ush_profile = USHProfile(
            name="ush",
            version="1.0.0",
            simplex_dimensions=ush_config.get("dimensions", []),
            baselines=ush_config.get("baselines", {}),
        )

    # Tool pack
    if tool_pack:
        artifact.tool_pack = ToolPackComponent(
            name="tools",
            version="1.0.0",
            tool_count=len(tool_pack.tool_grafts),
            tool_names=[g.tool_schema.name for g in tool_pack.tool_grafts],
            tiers_covered=list(set(g.tool_schema.tier for g in tool_pack.tool_grafts)),
        )

    # Lifecycle contract
    if lifecycle_config:
        artifact.lifecycle = LifecycleContract(
            name="lifecycle",
            version="1.0.0",
            tribe_id=lifecycle_config.get("tribe_id", ""),
            max_tier=lifecycle_config.get("max_tier", 6),
        )

    # Uplift record
    artifact.uplift_record = UpliftRecord(
        name="uplift",
        version="1.0.0",
        substrate_source=substrate_manifest.model_id,
    )

    # XDB bootstrap
    artifact.xdb_bootstrap = XDBBootstrap(
        name="xdb",
        version="1.0.0",
    )

    return artifact
