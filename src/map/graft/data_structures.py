"""
Data structures for the Graft Protocol.

Simplified implementation for initial testing - excludes XDB integration and
full MELD governance. Focuses on core mechanics of region derivation, graft
training, and substrate expansion.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
from pathlib import Path


@dataclass
class LayerMask:
    """Dimensions in a specific layer that are relevant to a concept."""
    layer_index: int
    component: str  # "mlp", "attn", "residual"
    indices: List[int]  # Sparse list of important dimension indices
    total_dimensions: int  # Full dimension count for this layer

    def to_dict(self) -> Dict:
        return {
            "layer_index": self.layer_index,
            "component": self.component,
            "dimension_mask": {
                "format": "sparse_indices",
                "indices": self.indices,
                "total_dimensions": self.total_dimensions
            }
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LayerMask":
        mask = d.get("dimension_mask", d)
        return cls(
            layer_index=d["layer_index"],
            component=d.get("component", "residual"),
            indices=mask.get("indices", []),
            total_dimensions=mask.get("total_dimensions", 0)
        )


@dataclass
class ConceptRegion:
    """
    Identifies which substrate dimensions correlate with a concept.

    Derived from lens weight analysis - guides where biases should land
    during graft training and which auxiliary dimensions the lens should read.
    """
    region_id: str
    concept_id: str
    layers: List[LayerMask]
    derivation: Dict[str, Any]  # Method and parameters used
    source_lens_path: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_all_indices(self, layer_index: Optional[int] = None) -> List[int]:
        """Get all dimension indices, optionally filtered by layer."""
        if layer_index is not None:
            for layer in self.layers:
                if layer.layer_index == layer_index:
                    return layer.indices
            return []
        # Return union of all layer indices
        all_indices = set()
        for layer in self.layers:
            all_indices.update(layer.indices)
        return sorted(list(all_indices))

    def to_dict(self) -> Dict:
        return {
            "region_id": self.region_id,
            "concept_id": self.concept_id,
            "layers": [l.to_dict() for l in self.layers],
            "derivation": self.derivation,
            "source_lens_path": self.source_lens_path,
            "created_at": self.created_at
        }

    def save(self, path: Path):
        """Save region to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "ConceptRegion":
        return cls(
            region_id=d["region_id"],
            concept_id=d["concept_id"],
            layers=[LayerMask.from_dict(l) for l in d["layers"]],
            derivation=d["derivation"],
            source_lens_path=d.get("source_lens_path"),
            created_at=d.get("created_at", datetime.now().isoformat())
        )

    @classmethod
    def load(cls, path: Path) -> "ConceptRegion":
        """Load region from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class InjectionPoint:
    """Where a concept dimension is injected into the substrate."""
    layer: int
    component: str  # "residual", "mlp", "attn"
    projection_path: Optional[str] = None  # Path to projection weights

    def to_dict(self) -> Dict:
        return {
            "layer": self.layer,
            "component": self.component,
            "projection": self.projection_path
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "InjectionPoint":
        return cls(
            layer=d["layer"],
            component=d["component"],
            projection_path=d.get("projection")
        )


@dataclass
class SubstrateBias:
    """
    Sparse bias delta for a specific layer component.

    Represents the modifications to existing weights that encode how
    the concept relates to everything else in the substrate.
    """
    layer: int
    component: str  # "mlp.up_proj", "mlp.down_proj", "attn.v_proj", etc.
    bias_delta_path: str  # Path to sparse tensor file
    nnz: int  # Number of non-zero entries
    shape: List[int]  # Original tensor shape
    magnitude_stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "layer": self.layer,
            "component": self.component,
            "bias_delta": {
                "format": "sparse_coo",
                "location": self.bias_delta_path,
                "nnz": self.nnz,
                "shape": self.shape
            },
            "magnitude_stats": self.magnitude_stats
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SubstrateBias":
        bd = d["bias_delta"]
        return cls(
            layer=d["layer"],
            component=d["component"],
            bias_delta_path=bd["location"],
            nnz=bd["nnz"],
            shape=bd["shape"],
            magnitude_stats=d.get("magnitude_stats", {})
        )


@dataclass
class GraftConfig:
    """Configuration for graft training."""
    # Training hyperparameters
    learning_rate: float = 5e-5
    epochs: int = 3
    batch_size: int = 32

    # Dimension initialization
    dimension_init: str = "learned"  # "learned", "zero", "random"

    # Bias regularization
    bias_sparsity_target: float = 0.95  # Encourage sparse biases
    bias_magnitude_penalty: float = 0.01  # Regularize bias magnitudes
    sparsity_threshold: float = 1e-4  # Zero out biases below this

    # Layers to inject dimension
    injection_layers: List[int] = field(default_factory=lambda: [18, 20, 22])

    # Validation thresholds
    activation_auc_threshold: float = 0.85
    lens_f1_threshold: float = 0.85
    false_positive_threshold: float = 0.05

    def to_dict(self) -> Dict:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "dimension_init": self.dimension_init,
            "bias_sparsity_target": self.bias_sparsity_target,
            "bias_magnitude_penalty": self.bias_magnitude_penalty,
            "sparsity_threshold": self.sparsity_threshold,
            "injection_layers": self.injection_layers,
            "activation_auc_threshold": self.activation_auc_threshold,
            "lens_f1_threshold": self.lens_f1_threshold,
            "false_positive_threshold": self.false_positive_threshold
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GraftConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Graft:
    """
    Complete package for adding a concept to a substrate.

    Contains the new dimension allocation, substrate biases, and lens binding.
    """
    graft_id: str
    concept_id: str
    concept_version: str

    # The new dimension
    dimension_index: int  # Position in expanded substrate
    dimension_label: str  # Human-readable label
    injection_points: List[InjectionPoint]

    # Biases to existing weights
    substrate_biases: List[SubstrateBias]

    # Lens binding
    lens_path: str
    primary_dimension: int  # Same as dimension_index
    auxiliary_dimensions: List[int]  # From region analysis

    # Substrate compatibility
    substrate_id: str
    pre_graft_dim: int
    post_graft_dim: int

    # Provenance
    training_run_id: Optional[str] = None
    source_region_id: Optional[str] = None
    config: Optional[GraftConfig] = None

    # Metrics from training
    metrics: Dict[str, float] = field(default_factory=dict)

    # Lifecycle
    validated: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "graft_id": self.graft_id,
            "concept_id": self.concept_id,
            "concept_version": self.concept_version,
            "concept_dimension": {
                "dimension_index": self.dimension_index,
                "dimension_label": self.dimension_label,
                "injection_points": [ip.to_dict() for ip in self.injection_points]
            },
            "substrate_biases": [sb.to_dict() for sb in self.substrate_biases],
            "lens_binding": {
                "lens_path": self.lens_path,
                "primary_dimension": self.primary_dimension,
                "auxiliary_dimensions": self.auxiliary_dimensions
            },
            "applies_to": {
                "substrate_id": self.substrate_id,
                "pre_graft_dim": self.pre_graft_dim,
                "post_graft_dim": self.post_graft_dim
            },
            "training_run_id": self.training_run_id,
            "source_region_id": self.source_region_id,
            "config": self.config.to_dict() if self.config else None,
            "metrics": self.metrics,
            "validated": self.validated,
            "created_at": self.created_at
        }

    def save(self, path: Path):
        """Save graft metadata to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "Graft":
        cd = d["concept_dimension"]
        pb = d["lens_binding"]
        at = d["applies_to"]

        return cls(
            graft_id=d["graft_id"],
            concept_id=d["concept_id"],
            concept_version=d["concept_version"],
            dimension_index=cd["dimension_index"],
            dimension_label=cd["dimension_label"],
            injection_points=[InjectionPoint.from_dict(ip) for ip in cd["injection_points"]],
            substrate_biases=[SubstrateBias.from_dict(sb) for sb in d["substrate_biases"]],
            lens_path=pb["lens_path"],
            primary_dimension=pb["primary_dimension"],
            auxiliary_dimensions=pb["auxiliary_dimensions"],
            substrate_id=at["substrate_id"],
            pre_graft_dim=at["pre_graft_dim"],
            post_graft_dim=at["post_graft_dim"],
            training_run_id=d.get("training_run_id"),
            source_region_id=d.get("source_region_id"),
            config=GraftConfig.from_dict(d["config"]) if d.get("config") else None,
            metrics=d.get("metrics", {}),
            validated=d.get("validated", False),
            created_at=d.get("created_at", datetime.now().isoformat())
        )

    @classmethod
    def load(cls, path: Path) -> "Graft":
        """Load graft metadata from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class DimensionEntry:
    """Entry in the substrate manifest's dimension table."""
    dimension_index: int
    concept_id: str
    graft_id: str
    grafted_at: str

    def to_dict(self) -> Dict:
        return {
            "dimension_index": self.dimension_index,
            "concept_id": self.concept_id,
            "graft_id": self.graft_id,
            "grafted_at": self.grafted_at
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DimensionEntry":
        return cls(
            dimension_index=d["dimension_index"],
            concept_id=d["concept_id"],
            graft_id=d["graft_id"],
            grafted_at=d["grafted_at"]
        )


@dataclass
class SubstrateArchitecture:
    """
    Architecture specification for a substrate.

    Required for expand mode to know which weight matrices to modify.
    """
    family: str  # "llama", "gemma", "gpt2", "moe", etc.
    hidden_size: int
    intermediate_size: int  # MLP intermediate dimension
    num_attention_heads: int
    num_key_value_heads: int  # For GQA; equals num_attention_heads for MHA
    head_dim: int
    num_layers: int

    mlp_type: str = "glu"  # "glu" | "standard"
    attention_type: str = "gqa"  # "gqa" | "mha" | "mqa"
    norm_type: str = "rms_norm"  # "rms_norm" | "layer_norm"

    # MoE-specific
    is_moe: bool = False
    num_experts: int = 1

    # Component paths (optional - can be inferred from family)
    component_paths: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict:
        d = {
            "family": self.family,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "num_layers": self.num_layers,
            "mlp_type": self.mlp_type,
            "attention_type": self.attention_type,
            "norm_type": self.norm_type,
            "is_moe": self.is_moe,
            "num_experts": self.num_experts,
        }
        if self.component_paths:
            d["component_paths"] = self.component_paths
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "SubstrateArchitecture":
        return cls(
            family=d["family"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
            head_dim=d.get("head_dim", d["hidden_size"] // d["num_attention_heads"]),
            num_layers=d["num_layers"],
            mlp_type=d.get("mlp_type", "glu"),
            attention_type=d.get("attention_type", "gqa"),
            norm_type=d.get("norm_type", "rms_norm"),
            is_moe=d.get("is_moe", False),
            num_experts=d.get("num_experts", 1),
            component_paths=d.get("component_paths"),
        )

    @classmethod
    def from_model_config(cls, config) -> "SubstrateArchitecture":
        """Create architecture spec from HuggingFace model config."""
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)

        # Detect attention type
        if num_kv_heads == 1:
            attn_type = "mqa"
        elif num_kv_heads < num_heads:
            attn_type = "gqa"
        else:
            attn_type = "mha"

        # Detect MoE
        is_moe = hasattr(config, 'num_local_experts') or hasattr(config, 'num_experts')
        num_experts = getattr(config, 'num_local_experts', getattr(config, 'num_experts', 1))

        return cls(
            family=getattr(config, 'model_type', 'unknown'),
            hidden_size=hidden_size,
            intermediate_size=getattr(config, 'intermediate_size', hidden_size * 4),
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=hidden_size // num_heads,
            num_layers=config.num_hidden_layers,
            mlp_type="glu",  # Most modern models use GLU
            attention_type=attn_type,
            norm_type="rms_norm",  # Most modern models use RMSNorm
            is_moe=is_moe,
            num_experts=num_experts,
        )


@dataclass
class SubstrateManifest:
    """
    Tracks the current state of a substrate including all grafted dimensions.
    """
    manifest_id: str

    # Base substrate info
    substrate_id: str
    base_checksum: str
    base_hidden_dim: int

    # Architecture specification (required for expand mode)
    architecture: Optional[SubstrateArchitecture] = None

    # Current state
    current_checksum: str = ""
    current_hidden_dim: int = 0
    total_grafts_applied: int = 0

    # Dimension allocation table
    dimension_table: List[DimensionEntry] = field(default_factory=list)

    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_next_dimension_index(self) -> int:
        """Get the next available dimension index."""
        return self.current_hidden_dim

    def add_graft(self, graft: Graft):
        """Record a graft application in the manifest."""
        entry = DimensionEntry(
            dimension_index=graft.dimension_index,
            concept_id=graft.concept_id,
            graft_id=graft.graft_id,
            grafted_at=datetime.now().isoformat()
        )
        self.dimension_table.append(entry)
        self.current_hidden_dim = graft.post_graft_dim
        self.total_grafts_applied += 1
        self.updated_at = datetime.now().isoformat()

    def get_concept_for_dimension(self, dim_index: int) -> Optional[str]:
        """Look up which concept owns a dimension."""
        for entry in self.dimension_table:
            if entry.dimension_index == dim_index:
                return entry.concept_id
        return None

    def to_dict(self) -> Dict:
        d = {
            "manifest_id": self.manifest_id,
            "base_substrate": {
                "substrate_id": self.substrate_id,
                "base_checksum": self.base_checksum,
                "base_hidden_dim": self.base_hidden_dim
            },
            "current_state": {
                "checksum": self.current_checksum,
                "hidden_dim": self.current_hidden_dim,
                "total_grafts_applied": self.total_grafts_applied
            },
            "dimension_table": [e.to_dict() for e in self.dimension_table],
            "updated_at": self.updated_at
        }
        if self.architecture:
            d["architecture"] = self.architecture.to_dict()
        return d

    def save(self, path: Path):
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "SubstrateManifest":
        bs = d["base_substrate"]
        cs = d["current_state"]

        # Parse architecture if present
        architecture = None
        if "architecture" in d:
            architecture = SubstrateArchitecture.from_dict(d["architecture"])

        return cls(
            manifest_id=d["manifest_id"],
            substrate_id=bs["substrate_id"],
            base_checksum=bs["base_checksum"],
            base_hidden_dim=bs["base_hidden_dim"],
            architecture=architecture,
            current_checksum=cs["checksum"],
            current_hidden_dim=cs["hidden_dim"],
            total_grafts_applied=cs["total_grafts_applied"],
            dimension_table=[DimensionEntry.from_dict(e) for e in d.get("dimension_table", [])],
            updated_at=d.get("updated_at", datetime.now().isoformat())
        )

    @classmethod
    def load(cls, path: Path) -> "SubstrateManifest":
        """Load manifest from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def create_for_model(
        cls,
        model_id: str,
        hidden_dim: int,
        checksum: str = "",
        architecture: Optional[SubstrateArchitecture] = None,
        model_config: Optional[Any] = None
    ) -> "SubstrateManifest":
        """
        Create a new manifest for a fresh substrate.

        Args:
            model_id: HuggingFace model identifier or local path
            hidden_dim: Base hidden dimension of the model
            checksum: Optional checksum of the model weights
            architecture: Pre-built SubstrateArchitecture (takes precedence)
            model_config: HuggingFace model config to derive architecture from
        """
        # Derive architecture from config if not provided directly
        if architecture is None and model_config is not None:
            architecture = SubstrateArchitecture.from_model_config(model_config)

        return cls(
            manifest_id=f"substrate-{model_id.replace('/', '-')}-v0",
            substrate_id=model_id,
            base_checksum=checksum,
            base_hidden_dim=hidden_dim,
            architecture=architecture,
            current_checksum=checksum,
            current_hidden_dim=hidden_dim,
            total_grafts_applied=0
        )
