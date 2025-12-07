#!/usr/bin/env python3
"""
Dynamic Hierarchical Probe Manager

Loads/unloads SUMO concept probes on-demand based on parent confidence scores.
Enables running 110K+ concepts with minimal memory footprint.

Architecture:
- Always keep layers 0-1 loaded (base coverage)
- When parent fires high → load its children
- Unload cold branches to free memory
- Support both activation and text probes
- Support multiple probe roles: concept, simplex, behavioral, category
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .deployment_manifest import DeploymentManifest, ManifestResolver


class ProbeRole(Enum):
    """Roles for different probe types in the monitoring system."""
    CONCEPT = "concept"        # Hierarchical discrimination vs siblings
    SIMPLEX = "simplex"        # Intensity tracking relative to baseline (tripole)
    BEHAVIORAL = "behavioral"  # Pattern detection (e.g., deception markers)
    CATEGORY = "category"      # Domain/layer markers (layer 0 style)


class SimpleMLP(nn.Module):
    """Simple MLP classifier matching SUMO training architecture."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Keep 'net' name for backward compatibility with saved probes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
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


class SimplexBinding:
    """Configuration for a simplex bound to a concept."""

    def __init__(
        self,
        simplex_term: str,
        always_on: bool = False,
        poles: Optional[Dict[str, Any]] = None,
        monitoring: Optional[Dict[str, Any]] = None,
    ):
        self.simplex_term = simplex_term
        self.always_on = always_on
        self.poles = poles or {}
        self.monitoring = monitoring or {
            'baseline_window': 100,
            'alert_threshold': 2.0,
            'trend_window': 500
        }


class ConceptMetadata:
    """Metadata for a single SUMO concept."""

    def __init__(
        self,
        sumo_term: str,
        layer: int,
        category_children: List[str],
        parent_concepts: List[str],
        synset_count: int,
        sumo_depth: int,
        role: ProbeRole = ProbeRole.CONCEPT,
        simplex_binding: Optional[SimplexBinding] = None,
        domain: Optional[str] = None,
    ):
        self.sumo_term = sumo_term
        self.layer = layer
        self.category_children = category_children
        self.parent_concepts = parent_concepts
        self.synset_count = synset_count
        self.sumo_depth = sumo_depth

        # Role and simplex binding (new in MAP Meld Protocol)
        self.role = role
        self.simplex_binding = simplex_binding
        self.domain = domain

        # Probe paths (set by manager)
        self.activation_probe_path: Optional[Path] = None
        self.text_probe_path: Optional[Path] = None
        self.simplex_probe_path: Optional[Path] = None
        self.has_text_probe: bool = False
        self.has_activation_probe: bool = False
        self.has_simplex_probe: bool = False


class DynamicProbeManager:
    """
    Manages dynamic loading/unloading of SUMO concept probes.

    Strategy:
    1. Always keep base layers (0-1) loaded for broad coverage
    2. Load children when parent confidence > threshold
    3. Unload branches when all concepts < min_confidence
    4. Track access patterns for intelligent caching
    """

    @staticmethod
    def discover_concept_packs(pack_dir: Path = Path("probe_packs/concept_packs")) -> Dict[str, Path]:
        """
        Discover all MAP-compliant concept packs.

        Returns:
            Dict mapping pack_id to pack directory path
        """
        packs = {}
        if not pack_dir.exists():
            return packs

        for pack_path in pack_dir.iterdir():
            if not pack_path.is_dir():
                continue

            pack_json = pack_path / "pack.json"
            if pack_json.exists():
                try:
                    with open(pack_json) as f:
                        pack_data = json.load(f)
                    pack_id = pack_data.get("pack_id")
                    if pack_id:
                        packs[pack_id] = pack_path
                except Exception as e:
                    print(f"Warning: Failed to read {pack_json}: {e}")

        return packs

    @staticmethod
    def discover_probe_packs(
        probe_packs_dir: Path = Path("probe_packs"),
        substrate_id: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Discover all available probe packs (both legacy and MAP-compliant).

        Args:
            probe_packs_dir: Root probe packs directory
            substrate_id: Optional filter by substrate (e.g., "gemma-3-4b-pt")

        Returns:
            Dict mapping probe_pack_id to metadata dict with keys:
                - path: Path to probe pack
                - type: "legacy" or "map"
                - concept_pack_id: (MAP only) ID of bound concept pack
                - substrate_id: (MAP only) Substrate ID
        """
        packs = {}
        if not probe_packs_dir.exists():
            return packs

        for pack_path in probe_packs_dir.iterdir():
            if not pack_path.is_dir():
                continue

            # Skip concept_packs directory
            if pack_path.name == "concept_packs":
                continue

            pack_json = pack_path / "pack.json"

            if pack_json.exists():
                # Try to determine if MAP or legacy
                try:
                    with open(pack_json) as f:
                        pack_data = json.load(f)

                    # MAP packs have concept_pack_id and substrate_id
                    if "concept_pack_id" in pack_data and "substrate_id" in pack_data:
                        probe_substrate = pack_data.get("substrate_id", "")
                        if substrate_id and substrate_id not in probe_substrate:
                            continue

                        packs[pack_path.name] = {
                            "path": pack_path,
                            "type": "map",
                            "concept_pack_id": pack_data["concept_pack_id"],
                            "substrate_id": pack_data["substrate_id"]
                        }
                    else:
                        # Legacy pack with pack.json
                        packs[pack_path.name] = {
                            "path": pack_path,
                            "type": "legacy"
                        }
                except Exception as e:
                    print(f"Warning: Failed to read {pack_json}: {e}")
            else:
                # Legacy pack without pack.json
                # Infer from directory structure
                if (pack_path / "activation_probes").exists() or \
                   (pack_path / "text_probes").exists() or \
                   list(pack_path.glob("layer*")):
                    packs[pack_path.name] = {
                        "path": pack_path,
                        "type": "legacy"
                    }

        return packs

    @staticmethod
    def get_latest_version(concept_pack_name: str, pack_dir: Path = Path("probe_packs/concept_packs")) -> Optional[str]:
        """
        Get the latest version of a concept pack by name.

        Args:
            concept_pack_name: Base name (e.g., "sumo-wordnet")
            pack_dir: Concept packs directory

        Returns:
            Full pack_id with version (e.g., "sumo-wordnet-v1") or None
        """
        if not pack_dir.exists():
            return None

        matching_packs = []
        for pack_path in pack_dir.iterdir():
            if not pack_path.is_dir():
                continue

            if pack_path.name.startswith(concept_pack_name):
                pack_json = pack_path / "pack.json"
                if pack_json.exists():
                    try:
                        with open(pack_json) as f:
                            pack_data = json.load(f)
                        version = pack_data.get("version", "0.0.0")
                        matching_packs.append((pack_path.name, version))
                    except Exception:
                        continue

        if not matching_packs:
            return None

        # Sort by version (simple string sort works for semantic versioning)
        matching_packs.sort(key=lambda x: x[1], reverse=True)
        return matching_packs[0][0]

    def __init__(
        self,
        layers_data_dir: Path = Path("data/concept_graph/abstraction_layers"),
        probes_dir: Path = Path("results/sumo_classifiers"),
        device: str = "cuda",
        base_layers: List[int] = [0, 1],
        load_threshold: float = 0.5,
        unload_threshold: float = 0.1,
        max_loaded_probes: int = 500,
        keep_top_k: int = 50,
        aggressive_pruning: bool = True,
        use_text_probes: bool = False,  # NEW: Use text probes for token→concept mapping
        use_activation_probes: bool = True,  # Use activation probes (default)
        probe_pack_id: Optional[str] = None,  # NEW: Use probe pack instead of probes_dir
        normalize_hidden_states: bool = True,  # Normalize hidden states before probe inference
        manifest: Optional["DeploymentManifest"] = None,  # Deployment manifest for partial loading
        manifest_path: Optional[Path] = None,  # Path to manifest JSON file
    ):
        """
        Args:
            layers_data_dir: Directory with layer JSON files
            probes_dir: Directory with trained probes
            device: Device for probe inference
            base_layers: Always-loaded layers for broad coverage
            load_threshold: Load children when parent > this confidence
            unload_threshold: Unload when all concepts < this confidence
            max_loaded_probes: Maximum probes to keep in memory
            keep_top_k: Only keep top K scoring probes (aggressive pruning)
            aggressive_pruning: Prune low-scoring probes after every detection
            normalize_hidden_states: Whether to normalize hidden states before probe inference.
                Generation-time hidden states often have higher variance than training-time
                hidden states, which can cause probes to saturate. LayerNorm fixes this.
            manifest: Optional DeploymentManifest for partial loading. Controls which
                concepts are loaded based on layer, domain, and branch rules.
            manifest_path: Optional path to manifest JSON file. Alternative to passing
                manifest object directly.
        """
        self.layers_data_dir = layers_data_dir
        self.device = device
        self.base_layers = base_layers
        self.load_threshold = load_threshold
        self.unload_threshold = unload_threshold
        self.max_loaded_probes = max_loaded_probes
        self.keep_top_k = keep_top_k
        self.aggressive_pruning = aggressive_pruning
        self.use_text_probes = use_text_probes
        self.use_activation_probes = use_activation_probes
        self.normalize_hidden_states = normalize_hidden_states
        self._layer_norm = None  # Lazy init after we know hidden_dim

        # Handle probe pack vs legacy probes_dir
        if probe_pack_id:
            import warnings
            warnings.warn(
                f"probe_pack_id parameter is deprecated and will be removed in v5.0. "
                f"Please migrate to MAP-compliant structure with spec_id parameter.",
                DeprecationWarning,
                stacklevel=2
            )

            # Discover probe packs using new discovery method
            available_packs = self.discover_probe_packs()

            if probe_pack_id not in available_packs:
                raise ValueError(
                    f"Probe pack not found: {probe_pack_id}\n"
                    f"Available packs: {', '.join(available_packs.keys())}"
                )

            pack_info = available_packs[probe_pack_id]
            pack_path = pack_info["path"]

            self.probes_dir = pack_path

            # Read pack.json to get probe paths
            pack_json_path = pack_path / "pack.json"
            if pack_json_path.exists():
                with open(pack_json_path) as f:
                    pack_data = json.load(f)
                probe_paths = pack_data.get("probe_paths", {})
                # Use paths from pack.json, with defaults for backward compatibility
                self.activation_probes_dir = pack_path / probe_paths.get("activation_probes", "activation_probes")
                self.text_probes_dir = pack_path / probe_paths.get("text_probes", "text_probes")
            else:
                # Fallback to default structure if pack.json doesn't exist
                self.activation_probes_dir = pack_path / "activation_probes"
                self.text_probes_dir = pack_path / "text_probes"

            self.using_probe_pack = True

            print(f"✓ Using probe pack: {probe_pack_id}")
            print(f"  Type: {pack_info['type']}")
            print(f"  Path: {pack_path}")
            print(f"  Activation probes: {self.activation_probes_dir.name}/")
        else:
            # Auto-detect: if probes_dir doesn't exist, try to find a probe pack
            if not probes_dir.exists():
                available_packs = self.discover_probe_packs()

                if available_packs:
                    # Use first available probe pack (sorted alphabetically)
                    probe_pack_id = sorted(available_packs.keys())[0]
                    pack_info = available_packs[probe_pack_id]
                    pack_path = pack_info["path"]

                    print(f"⚠ Probes directory not found: {probes_dir}")
                    print(f"✓ Auto-detected probe pack: {probe_pack_id}")
                    print(f"  Type: {pack_info['type']}")
                    print(f"  Path: {pack_path}")

                    self.probes_dir = pack_path

                    # Read pack.json to get probe paths
                    pack_json_path = pack_path / "pack.json"
                    if pack_json_path.exists():
                        with open(pack_json_path) as f:
                            pack_data = json.load(f)
                        probe_paths = pack_data.get("probe_paths", {})
                        self.activation_probes_dir = pack_path / probe_paths.get("activation_probes", "activation_probes")
                        self.text_probes_dir = pack_path / probe_paths.get("text_probes", "text_probes")
                    else:
                        self.activation_probes_dir = pack_path / "activation_probes"
                        self.text_probes_dir = pack_path / "text_probes"

                    self.using_probe_pack = True
                    print(f"  Activation probes: {self.activation_probes_dir.name}/")
                else:
                    raise ValueError(
                        f"Probes directory not found and no probe packs available: {probes_dir}\n"
                        f"Please create a probe pack in probe_packs/ directory."
                    )
            else:
                # Use legacy structure
                self.probes_dir = probes_dir
                self.activation_probes_dir = None  # Will use old structure
                self.text_probes_dir = None
                self.using_probe_pack = False

        # Concept metadata (all layers, always in memory - lightweight)
        # Key: (sumo_term, layer) to handle duplicates across layers
        self.concept_metadata: Dict[Tuple[str, int], ConceptMetadata] = {}
        self.parent_to_children: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
        self.child_to_parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

        # Loaded probes (activation and/or text)
        # Key: (sumo_term, layer)
        self.loaded_activation_probes: Dict[Tuple[str, int], nn.Module] = {}
        self.loaded_text_probes: Dict[Tuple[str, int], any] = {}  # BinaryTextProbe instances
        self.loaded_probes = self.loaded_activation_probes  # Alias for backward compatibility
        self.probe_scores: Dict[Tuple[str, int], float] = {}  # Recent scores for cache mgmt
        self.probe_access_count: Dict[Tuple[str, int], int] = defaultdict(int)

        # Simplex probes (intensity monitoring, separate from hierarchical)
        # Key: simplex_term (not concept_key, since simplexes may not be in hierarchy)
        self.loaded_simplex_probes: Dict[str, nn.Module] = {}
        self.simplex_scores: Dict[str, float] = {}  # Current activations
        self.simplex_baselines: Dict[str, List[float]] = defaultdict(list)  # Rolling baselines

        # Binding registry: concept_term -> simplex_term
        # Allows looking up bound simplex for a hierarchical concept
        self.simplex_bindings: Dict[str, str] = {}

        # Always-on simplexes (run every token regardless of hierarchical activation)
        self.always_on_simplexes: Set[str] = set()

        # Warm cache: probes that were recently relevant but not in top-k
        # These stay in memory but are not actively run every token
        # Key: (sumo_term, layer), Value: (probe, reactivation_count)
        self.warm_cache: Dict[Tuple[str, int], Tuple[nn.Module, int]] = {}
        self.cache_reactivation_count: Dict[Tuple[str, int], int] = defaultdict(int)

        # Track which probes are in base layers (never evict these)
        self.base_layer_probes: Set[Tuple[str, int]] = set()

        # Hidden dimension (inferred from first probe)
        self.hidden_dim: Optional[int] = None

        # Model pool for lazy creation (preallocate models, swap weights)
        self.model_pool: List[nn.Module] = []
        self.available_models: List[int] = []  # Indices of free models in pool
        self.model_pool_size: int = 100  # Preallocate 100 models

        # Statistics
        self.stats = {
            'total_loads': 0,
            'total_unloads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

        # Deployment manifest for partial loading
        self.manifest: Optional["DeploymentManifest"] = None
        self.manifest_resolver: Optional["ManifestResolver"] = None

        if manifest_path is not None:
            from .deployment_manifest import DeploymentManifest
            self.manifest = DeploymentManifest.from_json(manifest_path)
            print(f"✓ Loaded manifest: {self.manifest.manifest_id}")
        elif manifest is not None:
            self.manifest = manifest
            print(f"✓ Using manifest: {self.manifest.manifest_id}")

        # Load metadata
        self._load_all_metadata()

        # Initialize manifest resolver after metadata is loaded
        if self.manifest is not None:
            from .deployment_manifest import ManifestResolver
            self.manifest_resolver = ManifestResolver(
                manifest=self.manifest,
                concept_hierarchy=self.concept_metadata,
                parent_to_children=self.parent_to_children,
                child_to_parent=self.child_to_parent,
            )
            # Override base_layers from manifest if specified
            if self.manifest.layer_bounds.always_load_layers:
                self.base_layers = self.manifest.layer_bounds.always_load_layers
            # Override max loaded probes from manifest
            if self.manifest.dynamic_loading.max_loaded_concepts:
                self.max_loaded_probes = self.manifest.dynamic_loading.max_loaded_concepts
            # Override thresholds from manifest
            self.load_threshold = self.manifest.dynamic_loading.parent_threshold
            self.unload_threshold = self.manifest.dynamic_loading.unload_threshold

        # Load base layers
        print(f"\nInitializing DynamicProbeManager...")
        print(f"  Base layers: {self.base_layers}")
        print(f"  Load threshold: {self.load_threshold}")
        print(f"  Max probes in memory: {self.max_loaded_probes}")
        if self.manifest:
            print(f"  Manifest: {self.manifest.manifest_id}")
        self._load_base_layers()

    def _load_all_metadata(self):
        """Load lightweight metadata for all concepts (all layers).

        Deduplicates concepts that appear in multiple layers by keeping only
        the highest (lowest number) layer for each concept name.
        Only loads concepts that have probe files.
        """
        layer_files = sorted(self.layers_data_dir.glob("layer*.json"))

        # First pass: collect all concepts and find best layer for each
        # Best = category probe > most synsets > lowest layer number
        concept_to_best_layer = {}  # concept_name -> (layer, metadata_dict)

        for layer_file in layer_files:
            layer = int(layer_file.stem.replace('layer', ''))

            with open(layer_file) as f:
                layer_data = json.load(f)

            for concept in layer_data['concepts']:
                sumo_term = concept['sumo_term']

                # Skip layer 6 synset-level entries - they're just training data
                if layer == 6:
                    continue

                # Determine quality of this entry
                is_category = concept.get('is_category_probe', False)
                synset_count = len(concept.get('synsets', []))

                # Check if we should keep this layer over existing
                if sumo_term in concept_to_best_layer:
                    existing_layer, existing_concept = concept_to_best_layer[sumo_term]
                    existing_is_category = existing_concept.get('is_category_probe', False)
                    existing_synset_count = len(existing_concept.get('synsets', []))

                    # Priority: category probe > more synsets > lower layer
                    should_replace = False
                    if is_category and not existing_is_category:
                        should_replace = True
                    elif is_category == existing_is_category:
                        if synset_count > existing_synset_count:
                            should_replace = True
                        elif synset_count == existing_synset_count and layer < existing_layer:
                            should_replace = True

                    if should_replace:
                        concept_to_best_layer[sumo_term] = (layer, concept)
                else:
                    concept_to_best_layer[sumo_term] = (layer, concept)

        # Second pass: create metadata only for best layer of each concept
        # and only if probe files exist
        total_concepts = 0
        skipped_no_probe = 0
        deduplicated = 0

        for sumo_term, (layer, concept) in concept_to_best_layer.items():
            # Check if probe exists
            has_probe = False
            if self.using_probe_pack:
                activation_path = self.activation_probes_dir / f"{sumo_term}_classifier.pt"
                has_probe = activation_path.exists()
            else:
                layer_dir = self.probes_dir / f"layer{layer}"
                activation_path = layer_dir / f"{sumo_term}_classifier.pt"
                has_probe = activation_path.exists()

            if not has_probe:
                skipped_no_probe += 1
                continue

            metadata = ConceptMetadata(
                sumo_term=sumo_term,
                layer=layer,
                category_children=concept.get('category_children', []),
                parent_concepts=concept.get('parent_concepts', []),
                synset_count=concept.get('synset_count', 0),
                sumo_depth=concept.get('sumo_depth', 0),
            )

            # Set probe paths
            if self.using_probe_pack:
                # Probe pack structure: probes/activation/{concept}_classifier.pt
                activation_path = self.activation_probes_dir / f"{sumo_term}_classifier.pt"
                if activation_path.exists():
                    metadata.activation_probe_path = activation_path
                    metadata.has_activation_probe = True

                # Probe pack structure: probes/embedding_centroids/{concept}_centroid.npy
                text_probe_path = self.text_probes_dir / f"{sumo_term}_centroid.npy"
                if text_probe_path.exists():
                    metadata.text_probe_path = text_probe_path
                    metadata.has_text_probe = True
            else:
                # Legacy structure: results/sumo_classifiers/layer{N}/*
                layer_dir = self.probes_dir / f"layer{layer}"

                # Activation probe path
                activation_path = layer_dir / f"{sumo_term}_classifier.pt"
                if activation_path.exists():
                    metadata.activation_probe_path = activation_path
                    metadata.has_activation_probe = True

                # Text centroid path (new embedding-based approach)
                centroid_dir = layer_dir / "embedding_centroids"
                centroid_path = centroid_dir / f"{sumo_term}_centroid.npy"
                if centroid_path.exists():
                    metadata.text_probe_path = centroid_path
                    metadata.has_text_probe = True
                else:
                    # Fallback to legacy text_probes (TF-IDF) if centroids don't exist
                    text_probe_dir = layer_dir / "text_probes"
                    text_probe_path = text_probe_dir / f"{sumo_term}_text_probe.joblib"
                    if text_probe_path.exists():
                        metadata.text_probe_path = text_probe_path
                        metadata.has_text_probe = True

            concept_key = (sumo_term, layer)
            self.concept_metadata[concept_key] = metadata
            total_concepts += 1

        # Build parent-child mappings (after all concepts loaded)
        # Use both category_children (downward) and parent_concepts (upward)
        for concept_key, metadata in self.concept_metadata.items():
            sumo_term, layer = concept_key

            # Build parent->children from category_children
            for child_name in metadata.category_children:
                # Find child concept (should be in next layer or same layer)
                child_key = None
                for (cname, clayer) in self.concept_metadata.keys():
                    if cname == child_name and clayer >= layer:
                        child_key = (cname, clayer)
                        break

                if child_key:
                    self.parent_to_children[concept_key].append(child_key)
                    # Also set reverse mapping if not already set
                    if child_key not in self.child_to_parent:
                        self.child_to_parent[child_key] = concept_key

            # Build child->parent from parent_concepts (more reliable)
            for parent_name in metadata.parent_concepts:
                # Find parent concept (should be in previous layer or same layer)
                parent_key = None
                for (pname, player) in self.concept_metadata.keys():
                    if pname == parent_name and player <= layer:
                        parent_key = (pname, player)
                        break

                if parent_key:
                    # Set child->parent mapping (prefer explicit parent_concepts)
                    self.child_to_parent[concept_key] = parent_key
                    # Also add to parent->children if not already there
                    if concept_key not in self.parent_to_children[parent_key]:
                        self.parent_to_children[parent_key].append(concept_key)

        print(f"\n✓ Loaded metadata for {total_concepts} concepts across {len(layer_files)} layers")
        print(f"  Parent-child relationships: {len(self.parent_to_children)}")

    def _load_base_layers(self):
        """Load base layers for broad coverage, respecting manifest rules."""
        if self.manifest_resolver is not None:
            # Use manifest to determine which concepts to load
            all_keys = set(self.concept_metadata.keys())
            concepts_to_load = self.manifest_resolver.resolve_concepts_to_load(all_keys)

            # Filter to only load base layer concepts initially
            # (deeper concepts will be loaded dynamically)
            base_concepts = {
                key for key in concepts_to_load
                if key[1] in self.base_layers
            }

            # Expand with siblings to ensure coherent discrimination
            base_concepts_expanded = self.manifest_resolver.expand_with_siblings(base_concepts)

            self._load_concepts(list(base_concepts_expanded), reason="base_layer")
            print(f"  Manifest filtered: {len(base_concepts_expanded)} concepts from {len(all_keys)} available")
        else:
            # No manifest - load all concepts in base layers
            for layer in self.base_layers:
                layer_concept_keys = [
                    key for key in self.concept_metadata.keys()
                    if key[1] == layer
                ]
                self._load_concepts(layer_concept_keys, reason="base_layer")

        # Mark all loaded probes as base layer probes (never evict these)
        for concept_key in self.loaded_activation_probes.keys():
            self.base_layer_probes.add(concept_key)

        print(f"✓ Base layers loaded: {len(self.loaded_probes)} probes")

    def _ensure_model_pool(self):
        """Ensure model pool is preallocated."""
        if self.hidden_dim is None or len(self.model_pool) >= self.model_pool_size:
            return

        print(f"  Preallocating model pool ({self.model_pool_size} models)...")
        for i in range(self.model_pool_size):
            model = SimpleMLP(self.hidden_dim).to(self.device)
            model.eval()
            self.model_pool.append(model)
            self.available_models.append(i)

    def _get_model_from_pool(self) -> Optional[nn.Module]:
        """Get an available model from pool, or None if pool exhausted."""
        if self.available_models:
            idx = self.available_models.pop()
            return self.model_pool[idx]
        return None

    def _return_model_to_pool(self, model: nn.Module):
        """Return a model to the pool."""
        for i, pool_model in enumerate(self.model_pool):
            if pool_model is model:
                self.available_models.append(i)
                break

    def _manage_cache_memory(self):
        """
        Manage warm cache size by evicting least-reactivated probes.

        Strategy:
        - Total memory budget = max_loaded_probes (for loaded + warm cache combined)
        - When budget exceeded, evict from warm cache
        - Sort by reactivation count (ascending) - least-reactivated get evicted first
        - Never evict base layer probes (they should always be in loaded_probes, not warm cache)
        """
        total_in_memory = len(self.loaded_activation_probes) + len(self.warm_cache)

        if total_in_memory <= self.max_loaded_probes:
            return

        # Calculate how many to evict
        num_to_evict = total_in_memory - self.max_loaded_probes

        # Sort warm cache by reactivation count (ascending)
        # Probes with fewer reactivations are more likely to be cold
        warm_cache_sorted = sorted(
            self.warm_cache.items(),
            key=lambda x: x[1][1]  # x[1][1] is reactivation_count
        )

        # Evict least-reactivated probes
        evicted = 0
        for concept_key, (probe, reactivation_count) in warm_cache_sorted:
            if evicted >= num_to_evict:
                break

            # Don't evict base layer probes (though they should never be in warm cache)
            if concept_key in self.base_layer_probes:
                continue

            # Return probe to pool (it's still in memory, just not in cache)
            self._return_model_to_pool(probe)

            # Remove from warm cache
            del self.warm_cache[concept_key]

            # Clean up reactivation count
            if concept_key in self.cache_reactivation_count:
                del self.cache_reactivation_count[concept_key]

            evicted += 1
            self.stats['total_unloads'] += 1

    def _load_concepts(self, concept_keys: List[Tuple[str, int]], reason: str = "dynamic"):
        """
        Load activation and/or text probes for specified concepts.

        Uses batch loading and model pool for optimal performance.
        Checks warm cache before loading from disk.
        """
        # Filter out already loaded and check warm cache
        keys_to_load_activation = []
        keys_to_load_text = []
        warm_cache_hits = 0

        for concept_key in concept_keys:
            # Check activation probes
            if self.use_activation_probes:
                if concept_key in self.loaded_activation_probes:
                    # Already in active set
                    self.stats['cache_hits'] += 1
                elif concept_key in self.warm_cache:
                    # Move from warm cache to active loaded probes (zero disk I/O!)
                    probe, reactivation_count = self.warm_cache[concept_key]
                    self.loaded_activation_probes[concept_key] = probe
                    self.loaded_probes[concept_key] = probe  # Backward compatibility
                    self.probe_scores[concept_key] = 0.0

                    # Increment reactivation count
                    self.cache_reactivation_count[concept_key] = reactivation_count + 1

                    # Remove from warm cache
                    del self.warm_cache[concept_key]

                    # Track stats
                    self.stats['cache_hits'] += 1
                    warm_cache_hits += 1
                else:
                    # Not in memory at all, need to load from disk
                    keys_to_load_activation.append(concept_key)

            # Check text probes
            if self.use_text_probes and concept_key not in self.loaded_text_probes:
                keys_to_load_text.append(concept_key)

        # Return warm cache hits count for timing info
        self._last_warm_cache_hits = warm_cache_hits

        if not keys_to_load_activation and not keys_to_load_text:
            return

        # === LOAD ACTIVATION PROBES ===
        if self.use_activation_probes and keys_to_load_activation:
            # Infer hidden dim from first valid probe if needed
            if self.hidden_dim is None:
                for key in keys_to_load_activation:
                    metadata = self.concept_metadata.get(key)
                    if metadata and metadata.activation_probe_path and metadata.activation_probe_path.exists():
                        state_dict = torch.load(metadata.activation_probe_path, map_location='cpu')
                        first_key_name = list(state_dict.keys())[0]
                        self.hidden_dim = state_dict[first_key_name].shape[1]
                        print(f"  Inferred hidden_dim: {self.hidden_dim}")
                        break

            # Ensure model pool is allocated
            self._ensure_model_pool()

            # Batch load: Load all state_dicts first (can be parallelized)
            state_dicts = []
            valid_keys = []

            for concept_key in keys_to_load_activation:
                metadata = self.concept_metadata.get(concept_key)
                if not metadata or not metadata.activation_probe_path:
                    continue

                # Load state dict (this is the slow I/O part)
                state_dict = torch.load(metadata.activation_probe_path, map_location='cpu')

                # Handle key mismatch
                model_keys_ref = set(self.model_pool[0].state_dict().keys()) if self.model_pool else None
                if model_keys_ref:
                    loaded_keys = set(state_dict.keys())
                    if model_keys_ref != loaded_keys:
                        new_state_dict = {}
                        for key, value in state_dict.items():
                            if not key.startswith('net.'):
                                new_state_dict[f'net.{key}'] = value
                            else:
                                new_state_dict[key] = value
                        state_dict = new_state_dict

                state_dicts.append(state_dict)
                valid_keys.append(concept_key)

            # Now assign models and load weights
            for concept_key, state_dict in zip(valid_keys, state_dicts):
                # Try to get from pool first
                probe = self._get_model_from_pool()

                if probe is None:
                    # Pool exhausted, create new model
                    probe = SimpleMLP(self.hidden_dim).to(self.device)
                    probe.eval()

                # Load weights (fast - already in memory)
                probe.load_state_dict(state_dict)

                self.loaded_activation_probes[concept_key] = probe
                self.loaded_probes[concept_key] = probe  # Backward compatibility
                self.probe_scores[concept_key] = 0.0
                self.stats['total_loads'] += 1
                self.stats['cache_misses'] += 1

        # === LOAD TEXT PROBES ===
        if self.use_text_probes and keys_to_load_text:
            import joblib

            for concept_key in keys_to_load_text:
                metadata = self.concept_metadata.get(concept_key)
                if not metadata or not metadata.text_probe_path:
                    continue

                # Load text probe (centroid .npy or legacy joblib)
                try:
                    if metadata.text_probe_path.suffix == '.npy':
                        # New centroid-based approach
                        from .centroid_text_detector import CentroidTextDetector
                        text_probe = CentroidTextDetector.load(
                            metadata.text_probe_path,
                            concept_name=concept_key[0]
                        )
                    else:
                        # Legacy TF-IDF joblib approach
                        import joblib
                        text_probe = joblib.load(metadata.text_probe_path)

                    self.loaded_text_probes[concept_key] = text_probe
                    self.stats['total_loads'] += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to load text probe for {concept_key[0]}: {e}")

    def _unload_cold_probes(self):
        """Unload probes with consistently low scores to free memory."""
        if len(self.loaded_probes) <= self.max_loaded_probes:
            return

        # Find cold probes (low scores) - now includes base layer probes!
        # This allows aggressive pruning to cull irrelevant base concepts
        cold_probes = [
            (key, score)
            for key, score in self.probe_scores.items()
            if score < self.unload_threshold
        ]

        # Sort by score (lowest first)
        cold_probes.sort(key=lambda x: x[1])

        # Unload until under max
        num_to_unload = len(self.loaded_probes) - self.max_loaded_probes
        for key, _ in cold_probes[:num_to_unload]:
            # Return model to pool before deleting
            if key in self.loaded_probes:
                self._return_model_to_pool(self.loaded_probes[key])
                del self.loaded_probes[key]
            del self.probe_scores[key]
            self.stats['total_unloads'] += 1

    def _aggressive_prune_to_top_k(self):
        """
        Aggressively prune to keep only top-K scoring probes.

        This is more aggressive than _unload_cold_probes - it keeps ONLY
        the top K probes regardless of their scores. Critical for per-token
        usage where we only report top 10 anyway.

        NOTE: Base layer probes are now PROTECTED from pruning to ensure
        broad hierarchical coverage across all tokens.
        """
        if not self.aggressive_pruning:
            return

        if len(self.loaded_probes) <= self.keep_top_k:
            return

        # Separate base layer probes from dynamic probes
        base_layer_probes = []
        dynamic_probes = []

        for k, v in self.probe_scores.items():
            concept_name, layer = k
            if layer in self.base_layers:
                base_layer_probes.append((k, v))
            else:
                dynamic_probes.append((k, v))

        # Sort dynamic probes by score (highest first)
        dynamic_probes.sort(key=lambda x: x[1], reverse=True)

        # Keep base layers + top K dynamic probes
        # This ensures base layers are always available for hierarchical expansion
        total_to_keep = len(base_layer_probes) + self.keep_top_k

        if len(self.loaded_probes) <= total_to_keep:
            return

        # Unload everything beyond base layers + top K
        to_unload = dynamic_probes[self.keep_top_k:]

        for key, _ in to_unload:
            if key in self.loaded_probes:
                # Return model to pool before deleting
                self._return_model_to_pool(self.loaded_probes[key])
                del self.loaded_probes[key]
                del self.probe_scores[key]
                self.stats['total_unloads'] += 1

    def _apply_hierarchical_suppression(
        self,
        results: List,
        return_logits: bool = False
    ) -> List:
        """
        Suppress parent concepts when their children are present in results.

        This ensures we show the most specific concepts. For example, if both
        "Bird" and "Sparrow" activate, we suppress "Bird" and only show "Sparrow".

        Importantly, this only suppresses within the top-k results. If a child
        drops out of the top-k later, its parent will naturally re-appear since
        it won't be suppressed anymore.

        Args:
            results: List of (name, prob, layer) or (name, prob, logit, layer) tuples
            return_logits: Whether results include logits

        Returns:
            Filtered results with parents suppressed
        """
        if not results:
            return results

        # Build set of concept keys present in results
        present_keys = set()
        for result in results:
            concept_name = result[0]
            layer = result[-1]  # layer is always last
            present_keys.add((concept_name, layer))

        # Identify parents to suppress
        suppressed_keys = set()
        for concept_key in present_keys:
            # Get children of this concept
            child_keys = self.parent_to_children.get(concept_key, [])

            # If ANY child is present in results, suppress this parent
            for child_key in child_keys:
                if child_key in present_keys:
                    suppressed_keys.add(concept_key)
                    break

        # Filter out suppressed parents
        filtered_results = []
        for result in results:
            concept_name = result[0]
            layer = result[-1]
            concept_key = (concept_name, layer)

            if concept_key not in suppressed_keys:
                filtered_results.append(result)

        return filtered_results

    def detect_and_expand(
        self,
        hidden_state: torch.Tensor,
        top_k: int = 10,
        return_timing: bool = False,
        return_logits: bool = False,
        skip_pruning: bool = False,
    ) -> Tuple[List[Tuple[str, float, int]], Optional[Dict]]:
        """
        Detect concepts in hidden state, dynamically loading children as needed.

        Args:
            hidden_state: Hidden state tensor [1, hidden_dim] or [hidden_dim]
            top_k: Return top K concepts
            return_timing: Return detailed timing breakdown
            return_logits: If True, return (concept_name, probability, logit, layer) tuples
            skip_pruning: If True, skip aggressive pruning for this detection (useful during prompt processing)

        Returns:
            (concept_scores, timing_info)
            concept_scores: List of (concept_name, probability, layer) or (concept_name, probability, logit, layer)
            timing_info: Dict with timing breakdown (if return_timing=True)
        """
        timing = {} if return_timing else None
        start = time.time()

        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        # Normalize hidden states to match training distribution
        # Generation-time hidden states often have std ~4.0 vs training std ~1.3
        # This causes probe saturation without normalization
        if self.normalize_hidden_states:
            hidden_dim = hidden_state.shape[-1]
            # Always move hidden_state to the same device as layer_norm
            if self._layer_norm is None or self._layer_norm.normalized_shape[0] != hidden_dim:
                self._layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False).to(hidden_state.device)
            # DEBUG: Check stats before/after normalization
            # pre_std = hidden_state.std().item()
            hidden_state = self._layer_norm(hidden_state)
            # post_std = hidden_state.std().item()
            # print(f"Normalization: pre_std={pre_std:.3f}, post_std={post_std:.3f}")

        # 1. Run all currently loaded probes
        t1 = time.time()
        current_scores = {}
        current_logits = {} if return_logits else None
        with torch.inference_mode():
            for concept_key, probe in self.loaded_probes.items():
                if return_logits:
                    prob, logit = probe(hidden_state, return_logits=True)
                    prob = prob.item()
                    logit = logit.item()
                    current_logits[concept_key] = logit
                else:
                    prob = probe(hidden_state).item()
                current_scores[concept_key] = prob
                self.probe_scores[concept_key] = prob
                self.probe_access_count[concept_key] += 1

        if timing is not None:
            timing['initial_detection'] = (time.time() - t1) * 1000

        # 2. Identify top-k parents and load their children
        # Only expand children for concepts that are in the top-k results
        # This prevents loading children for low-scoring concepts that exceed threshold
        t2 = time.time()
        child_keys_to_load = set()

        # Get top-k concepts from current scores
        sorted_concepts = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_concepts = sorted_concepts[:top_k]

        for concept_key, prob in top_k_concepts:
            # Get children from parent-child mapping
            child_keys = self.parent_to_children.get(concept_key, [])
            for child_key in child_keys:
                if child_key not in self.loaded_probes:
                    # Check manifest rules if available
                    if self.manifest_resolver is not None:
                        if self.manifest_resolver.should_load_concept(child_key):
                            child_keys_to_load.add(child_key)
                    else:
                        child_keys_to_load.add(child_key)

        # Expand with siblings for coherent discrimination (sibling coherence rule)
        if child_keys_to_load and self.manifest_resolver is not None:
            child_keys_to_load = self.manifest_resolver.expand_with_siblings(child_keys_to_load)
            # Filter out already loaded concepts
            child_keys_to_load = {k for k in child_keys_to_load if k not in self.loaded_probes}

        if child_keys_to_load:
            self._load_concepts(list(child_keys_to_load), reason="dynamic_expansion")

        if timing is not None:
            timing['child_loading'] = (time.time() - t2) * 1000
            timing['num_children_loaded'] = len(child_keys_to_load)

        # 3. Run newly loaded probes
        t3 = time.time()
        if child_keys_to_load:
            with torch.inference_mode():
                for concept_key in child_keys_to_load:
                    if concept_key in self.loaded_probes:
                        probe = self.loaded_probes[concept_key]
                        if return_logits:
                            prob, logit = probe(hidden_state, return_logits=True)
                            prob = prob.item()
                            logit = logit.item()
                            current_logits[concept_key] = logit
                        else:
                            prob = probe(hidden_state).item()
                        current_scores[concept_key] = prob
                        self.probe_scores[concept_key] = prob
                        self.probe_access_count[concept_key] += 1

        if timing is not None:
            timing['child_detection'] = (time.time() - t3) * 1000

        # 4. Warm cache management + pruning
        # Move non-top-k probes to warm cache instead of unloading them
        # Skip during prompt processing to avoid discarding relevant concepts
        t4 = time.time()

        # Track cache hits from warm cache reactivations
        cache_hits_this_token = getattr(self, '_last_warm_cache_hits', 0)
        cache_misses_this_token = len(child_keys_to_load)

        if not skip_pruning:
            # Get top-k concept keys from all current scores
            sorted_all_concepts = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_concept_keys = set([key for key, _ in sorted_all_concepts[:top_k]])

            # Move non-top-k probes to warm cache (but keep base layer probes active)
            to_warm_cache = []
            for concept_key in list(self.loaded_activation_probes.keys()):
                # Always keep base layer probes in loaded set
                if concept_key in self.base_layer_probes:
                    continue

                # Move to warm cache if not in top-k
                if concept_key not in top_k_concept_keys:
                    to_warm_cache.append(concept_key)

            # Move probes to warm cache
            for concept_key in to_warm_cache:
                probe = self.loaded_activation_probes[concept_key]

                # Store in warm cache with current reactivation count
                reactivation_count = self.cache_reactivation_count.get(concept_key, 0)
                self.warm_cache[concept_key] = (probe, reactivation_count)

                # Remove from active loaded probes
                del self.loaded_activation_probes[concept_key]
                if concept_key in self.loaded_probes:
                    del self.loaded_probes[concept_key]
                if concept_key in self.probe_scores:
                    del self.probe_scores[concept_key]

            # Manage total cache memory (evict from warm cache if needed)
            self._manage_cache_memory()

        if timing is not None:
            timing['cache_management'] = (time.time() - t4) * 1000

        # 5. Sort and return top K
        results = []
        for concept_key, prob in current_scores.items():
            concept_name, layer = concept_key
            if return_logits:
                logit = current_logits.get(concept_key, 0.0)
                results.append((concept_name, prob, logit, layer))
            else:
                results.append((concept_name, prob, layer))

        results.sort(key=lambda x: x[1], reverse=True)

        # 6. Apply hierarchical suppression to top-k only
        # Parents are suppressed if their children are in the top-k
        # If children drop out, parents re-appear naturally
        top_k_results = results[:top_k]
        top_k_results = self._apply_hierarchical_suppression(top_k_results, return_logits)

        if timing is not None:
            timing['total'] = (time.time() - start) * 1000
            timing['loaded_probes'] = len(self.loaded_probes)
            timing['cache_hits'] = cache_hits_this_token
            timing['cache_misses'] = cache_misses_this_token
            timing['warm_cache_size'] = len(self.warm_cache)

        return top_k_results, timing

    def detect_and_expand_with_divergence(
        self,
        hidden_state: torch.Tensor,
        token_embedding: np.ndarray,
        top_k: int = 10,
        return_timing: bool = False,
    ) -> Tuple[Dict[str, Dict], Optional[Dict]]:
        """
        Detect concepts with divergence scores using embedding centroids.

        Args:
            hidden_state: Hidden state tensor [1, hidden_dim] or [hidden_dim]
            token_embedding: Token embedding for centroid comparison [embedding_dim]
            top_k: Return top K concepts by activation
            return_timing: Return detailed timing breakdown

        Returns:
            (concepts_with_divergence, timing_info)
            concepts_with_divergence: Dict mapping concept names to:
                {
                    'probability': activation confidence,
                    'layer': int,
                    'text_confidence': centroid similarity [0, 1] or None,
                    'divergence': activation - text_confidence or None
                }
            timing_info: Dict with timing breakdown (if return_timing=True)
        """
        from src.monitoring.centroid_text_detector import CentroidTextDetector

        timing = {} if return_timing else None
        start = time.time()

        # 1. Get top-K concepts by activation
        detected_concepts, detect_timing = self.detect_and_expand(
            hidden_state,
            top_k=top_k,
            return_timing=True
        )

        if timing is not None:
            timing.update(detect_timing)

        # 2. For each detected concept, compute text confidence using centroid
        t_centroid = time.time()
        concepts_with_divergence = {}

        for concept_name, activation_prob, layer in detected_concepts:
            # Find centroid path
            centroid_path = self.probes_dir / f"layer{layer}" / "embedding_centroids" / f"{concept_name}_centroid.npy"

            text_conf = None
            divergence = None

            if centroid_path.exists():
                try:
                    centroid_detector = CentroidTextDetector.load(centroid_path, concept_name)
                    text_conf = float(centroid_detector.predict(token_embedding))
                    divergence = float(activation_prob - text_conf)
                    # Debug: log first few to see what values we're getting
                    if len(concepts_with_divergence) < 3:
                        print(f"[DEBUG] {concept_name}: activation={activation_prob:.3f}, text_sim={text_conf:.3f}, div={divergence:.3f}")
                except Exception as e:
                    # Centroid might be malformed or incompatible
                    print(f"[ERROR] Failed to load centroid for {concept_name}: {e}")
                    pass

            concepts_with_divergence[concept_name] = {
                'probability': float(activation_prob),
                'layer': int(layer),
                'text_confidence': text_conf,
                'divergence': divergence
            }

        if timing is not None:
            timing['centroid_comparison'] = (time.time() - t_centroid) * 1000
            timing['total_with_divergence'] = (time.time() - start) * 1000

        return concepts_with_divergence, timing

    def get_concept_path(self, concept_name: str, layer: int = None) -> List[str]:
        """Get hierarchical path from root to concept."""
        # Find concept key (prefer specified layer, otherwise use any)
        concept_key = None
        if layer is not None:
            concept_key = (concept_name, layer)
            if concept_key not in self.concept_metadata:
                concept_key = None

        if concept_key is None:
            # Find any matching concept
            for key in self.concept_metadata.keys():
                if key[0] == concept_name:
                    concept_key = key
                    break

        if concept_key is None:
            return [concept_name]

        path = [concept_key[0]]
        current = concept_key

        while current in self.child_to_parent:
            parent_key = self.child_to_parent[current]
            path.insert(0, parent_key[0])
            current = parent_key

        return path

    def print_stats(self):
        """Print manager statistics."""
        print("\n" + "=" * 80)
        print("DYNAMIC PROBE MANAGER STATISTICS")
        print("=" * 80)
        print(f"Total concepts in metadata: {len(self.concept_metadata)}")
        print(f"Currently loaded probes: {len(self.loaded_probes)}")
        print(f"Base layer probes (protected): {len(self.base_layer_probes)}")
        print(f"Warm cache size: {len(self.warm_cache)}")
        print(f"Total in memory: {len(self.loaded_probes) + len(self.warm_cache)}")
        print(f"Total loads: {self.stats['total_loads']}")
        print(f"Total unloads: {self.stats['total_unloads']}")
        print(f"Cache hits: {self.stats['cache_hits']}")
        print(f"Cache misses: {self.stats['cache_misses']}")

        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
            print(f"Cache hit rate: {hit_rate:.1%}")

        # Show top reactivated concepts from warm cache
        if self.cache_reactivation_count:
            print("\nTop 10 most reactivated concepts (from warm cache):")
            top_reactivated = sorted(
                self.cache_reactivation_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for concept_key, count in top_reactivated:
                concept_name, layer = concept_key
                print(f"  L{layer} {concept_name:30s} {count:4d} reactivations")

        # Show top accessed concepts
        if self.probe_access_count:
            print("\nTop 10 most accessed concepts:")
            top_accessed = sorted(
                self.probe_access_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for concept_key, count in top_accessed:
                concept_name, layer = concept_key
                print(f"  L{layer} {concept_name:30s} {count:4d} accesses")

        print("=" * 80)

    def get_loaded_fingerprint(self) -> str:
        """
        Get a fingerprint hash of currently loaded concepts.

        Used for comparability verification between BEs with same manifest.
        """
        if self.manifest_resolver is not None:
            loaded_keys = set(self.loaded_activation_probes.keys())
            return self.manifest_resolver.compute_fingerprint(loaded_keys)

        # Fallback: compute directly
        import hashlib
        sorted_keys = sorted(self.loaded_activation_probes.keys())
        key_str = "|".join(f"{name}:{layer}" for name, layer in sorted_keys)
        return f"sha256:{hashlib.sha256(key_str.encode()).hexdigest()[:16]}"

    def get_manifest_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the manifest configuration.

        Returns:
            Dict with manifest info or None if no manifest
        """
        if self.manifest is None:
            return {"manifest": None, "mode": "unrestricted"}

        return {
            "manifest_id": self.manifest.manifest_id,
            "manifest_version": self.manifest.manifest_version,
            "layer_bounds": {
                "default_max": self.manifest.layer_bounds.default_max_layer,
                "absolute_max": self.manifest.layer_bounds.absolute_max_layer,
                "always_load": self.manifest.layer_bounds.always_load_layers,
            },
            "domain_overrides": list(self.manifest.domain_overrides.keys()),
            "branch_rules": [r.branch for r in self.manifest.branch_rules],
            "explicit_includes": list(self.manifest.explicit_concepts.always_include),
            "explicit_excludes": list(self.manifest.explicit_concepts.always_exclude),
            "dynamic_loading_enabled": self.manifest.dynamic_loading.enabled,
            "loaded_concepts": len(self.loaded_activation_probes),
            "fingerprint": self.get_loaded_fingerprint(),
        }

    # =========================================================================
    # BE WORKSPACE TOOLS - Probe Expansion for Introspection
    # =========================================================================

    def request_probe_expansion(
        self,
        branch: str,
        reason: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """
        BE workspace tool: Request to expand probes for a branch.

        This is called when a BE uses a workspace tool to expand probes
        for self-introspection. The USH probe envelope determines what's allowed.

        Args:
            branch: The branch name to expand (e.g., "Emotion", "Curiosity")
            reason: Why the BE wants this expansion (for audit log)
            depth: How many layers deep to expand (default 2)

        Returns:
            Dict with:
                - success: bool
                - loaded_concepts: List[str] - concepts that were loaded
                - cat_scope: str - which CAT covers these probes
                - error: str - error message if failed
        """
        from .deployment_manifest import ProbeExpansionResult

        # Check envelope permissions
        if self.manifest_resolver is not None:
            result = self.manifest_resolver.check_branch_expansion(branch, reason)
            if not result.success:
                return {
                    "success": False,
                    "loaded_concepts": [],
                    "cat_scope": None,
                    "error": result.error,
                }
        else:
            # No manifest = allow all
            result = ProbeExpansionResult(success=True, cat_scope=None)

        # Find concepts under this branch
        branch_concepts = set()
        branch_root_key = None

        # Find the branch root
        for key in self.concept_metadata.keys():
            if key[0] == branch:
                branch_root_key = key
                break

        if branch_root_key is None:
            return {
                "success": False,
                "loaded_concepts": [],
                "cat_scope": result.cat_scope,
                "error": f"Branch '{branch}' not found in concept hierarchy",
            }

        # BFS to find all concepts under branch up to depth
        queue = [(branch_root_key, 0)]  # (concept_key, current_depth)
        while queue:
            current_key, current_depth = queue.pop(0)

            # Check manifest layer bounds
            if self.manifest_resolver is not None:
                if not self.manifest_resolver.should_load_concept(current_key):
                    continue

            branch_concepts.add(current_key)

            # Expand to children if not at max depth
            if current_depth < depth:
                children = self.parent_to_children.get(current_key, [])
                for child_key in children:
                    queue.append((child_key, current_depth + 1))

        # Expand with siblings for coherent discrimination
        if self.manifest_resolver is not None:
            branch_concepts = self.manifest_resolver.expand_with_siblings(branch_concepts)

        # Load the concepts
        concepts_to_load = [k for k in branch_concepts if k not in self.loaded_activation_probes]

        if concepts_to_load:
            self._load_concepts(concepts_to_load, reason=f"be_introspection:{branch}")

        # Return result
        loaded_names = [k[0] for k in concepts_to_load]
        return {
            "success": True,
            "loaded_concepts": loaded_names,
            "cat_scope": result.cat_scope,
            "error": None,
        }

    def request_probe_collapse(
        self,
        branch: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        BE workspace tool: Collapse probes for a branch to reclaim resources.

        This is the inverse of request_probe_expansion. It unloads probes
        that were loaded for introspection, returning them to the warm cache
        or fully evicting them depending on memory pressure.

        Note: Probes in must_enable branches cannot be collapsed.

        Args:
            branch: The branch name to collapse (e.g., "Emotion", "Curiosity")
            reason: Why the BE is collapsing this (for audit log)

        Returns:
            Dict with:
                - success: bool
                - collapsed_concepts: List[str] - concepts that were unloaded
                - retained_concepts: List[str] - concepts kept (must_enable)
                - error: str - error message if failed
        """
        # Check if branch is in must_enable (cannot collapse)
        if self.manifest_resolver is not None:
            must_enable = self.manifest_resolver.get_must_enable_branches()
            if branch in must_enable:
                return {
                    "success": False,
                    "collapsed_concepts": [],
                    "retained_concepts": [branch],
                    "error": f"Branch '{branch}' is in must_enable and cannot be collapsed",
                }

        # Find concepts under this branch that are currently loaded
        branch_concepts_loaded = []
        retained = []

        for key in list(self.loaded_activation_probes.keys()):
            concept_name, layer = key

            # Skip base layer probes - they're always retained
            if key in self.base_layer_probes:
                continue

            # Check if this concept is under the branch
            path = self.get_concept_path(concept_name, layer)

            if branch in path:
                # Check if this specific concept is in must_enable
                if self.manifest_resolver is not None:
                    envelope = self.manifest_resolver.manifest.probe_envelope
                    if envelope and concept_name in envelope.must_enable.branches:
                        retained.append(concept_name)
                        continue

                branch_concepts_loaded.append(key)

        # Move probes to warm cache (not full eviction - they may be needed again)
        collapsed_names = []
        for key in branch_concepts_loaded:
            probe = self.loaded_activation_probes[key]

            # Store in warm cache
            reactivation_count = self.cache_reactivation_count.get(key, 0)
            self.warm_cache[key] = (probe, reactivation_count)

            # Remove from active loaded probes
            del self.loaded_activation_probes[key]
            if key in self.loaded_probes:
                del self.loaded_probes[key]
            if key in self.probe_scores:
                del self.probe_scores[key]

            collapsed_names.append(key[0])

        # Manage cache memory (may evict from warm cache if over budget)
        self._manage_cache_memory()

        return {
            "success": True,
            "collapsed_concepts": collapsed_names,
            "retained_concepts": retained,
            "error": None,
        }

    def get_introspection_reading(
        self,
        branch: str,
        hidden_state: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        BE workspace tool: Get current probe readings for a branch.

        This allows the BE to introspect on its own state for a specific branch
        that was previously expanded via request_probe_expansion.

        Args:
            branch: The branch to read (must have been expanded first)
            hidden_state: Current hidden state to probe

        Returns:
            Dict with:
                - branch: str
                - readings: Dict[concept_name, activation] for concepts in branch
                - top_concept: str - highest activating concept
                - interpretation: str - brief interpretation
        """
        # Find concepts under this branch
        branch_concepts = {}

        for key in self.loaded_activation_probes.keys():
            # Check if this concept is under the branch
            concept_name, layer = key
            path = self.get_concept_path(concept_name, layer)

            if branch in path:
                branch_concepts[key] = self.loaded_activation_probes[key]

        if not branch_concepts:
            return {
                "branch": branch,
                "readings": {},
                "top_concept": None,
                "interpretation": f"No probes loaded for branch '{branch}'. Call request_probe_expansion first.",
            }

        # Run probes on current hidden state
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        if self.normalize_hidden_states:
            hidden_dim = hidden_state.shape[-1]
            if self._layer_norm is None or self._layer_norm.normalized_shape[0] != hidden_dim:
                self._layer_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False).to(hidden_state.device)
            hidden_state = self._layer_norm(hidden_state)

        readings = {}
        with torch.inference_mode():
            for key, probe in branch_concepts.items():
                prob = probe(hidden_state).item()
                readings[key[0]] = prob

        # Find top concept
        top_concept = max(readings, key=readings.get) if readings else None
        top_score = readings.get(top_concept, 0.0)

        # Generate interpretation
        if top_score > 0.7:
            interpretation = f"Strong activation of '{top_concept}' ({top_score:.2f})"
        elif top_score > 0.4:
            interpretation = f"Moderate activation of '{top_concept}' ({top_score:.2f})"
        else:
            interpretation = f"Low activation across branch (top: {top_concept} at {top_score:.2f})"

        return {
            "branch": branch,
            "readings": readings,
            "top_concept": top_concept,
            "interpretation": interpretation,
        }

    def get_envelope_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the USH probe envelope.

        Returns summary of what probes the BE can/cannot access.
        """
        if self.manifest_resolver is not None:
            return self.manifest_resolver.get_envelope_summary()

        return {
            "has_envelope": False,
            "mode": "unrestricted",
        }

    def reset_to_base(self):
        """Reset to only base layer probes (for clean benchmarking)."""
        # Clear all non-base probes
        to_remove = [k for k in self.loaded_probes.keys() if k not in self.base_layer_probes]
        for key in to_remove:
            del self.loaded_probes[key]

        # Clear warm cache
        self.warm_cache.clear()
        self.cache_reactivation_count.clear()

        # Clear scores and access counts
        self.probe_scores.clear()
        self.probe_access_count.clear()

    def prewarm_from_prompt(self, hidden_state: torch.Tensor, top_k: int = 10):
        """
        Pre-warm cache by loading child probes based on prompt hidden state.

        This should be called during prompt processing (before generation starts)
        to pre-load children that will likely be needed during generation.

        The cost of this operation does NOT count against generation latency
        since it happens during prompt eval.

        Args:
            hidden_state: Hidden state from prompt's final token [1, hidden_dim]
            top_k: Number of top concepts to expand (same as detect_and_expand)

        Returns:
            Number of children pre-loaded
        """
        # Run detection to identify top concepts and load their children
        # This is exactly the same as detect_and_expand, but we don't return results
        _ = self.detect_and_expand(hidden_state, top_k=top_k)

        # Return count of loaded children (for stats)
        return len(self.loaded_probes) - len(self.base_layer_probes)

    # =========================================================================
    # SIMPLEX / ROLE-AWARE METHODS (MAP Meld Protocol §12)
    # =========================================================================

    def load_simplex(self, simplex_term: str, probe_path: Path) -> bool:
        """
        Load a simplex probe for intensity monitoring.

        Args:
            simplex_term: Name of the simplex (e.g., "AutonomyDrive")
            probe_path: Path to the simplex probe file

        Returns:
            True if loaded successfully, False otherwise
        """
        if simplex_term in self.loaded_simplex_probes:
            return True  # Already loaded

        if not probe_path.exists():
            return False

        try:
            state_dict = torch.load(probe_path, map_location='cpu')

            # Infer hidden dim if not set
            if self.hidden_dim is None:
                first_key = list(state_dict.keys())[0]
                self.hidden_dim = state_dict[first_key].shape[1]

            probe = SimpleMLP(self.hidden_dim).to(self.device)
            probe.eval()

            # Handle key mismatch (net. prefix)
            model_keys = set(probe.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            if model_keys != loaded_keys:
                new_state_dict = {}
                for key, value in state_dict.items():
                    if not key.startswith('net.'):
                        new_state_dict[f'net.{key}'] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict

            probe.load_state_dict(state_dict)
            self.loaded_simplex_probes[simplex_term] = probe
            self.simplex_scores[simplex_term] = 0.0
            return True

        except Exception as e:
            print(f"  Failed to load simplex {simplex_term}: {e}")
            return False

    def register_simplex_binding(
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

    def detect_simplexes(
        self,
        hidden_state: torch.Tensor,
        simplex_terms: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Run simplex probes and return activations.

        Args:
            hidden_state: Hidden state tensor [1, hidden_dim] or [hidden_dim]
            simplex_terms: Specific simplexes to run, or None for always-on only

        Returns:
            Dict mapping simplex_term to activation score
        """
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        terms_to_run = simplex_terms or list(self.always_on_simplexes)
        results = {}

        with torch.inference_mode():
            for simplex_term in terms_to_run:
                if simplex_term not in self.loaded_simplex_probes:
                    continue

                probe = self.loaded_simplex_probes[simplex_term]
                prob = probe(hidden_state).item()

                results[simplex_term] = prob
                self.simplex_scores[simplex_term] = prob

                # Update rolling baseline
                baseline_list = self.simplex_baselines[simplex_term]
                baseline_list.append(prob)

                # Keep only last N samples (default 100)
                max_baseline = 100
                if len(baseline_list) > max_baseline:
                    self.simplex_baselines[simplex_term] = baseline_list[-max_baseline:]

        return results

    def get_simplex_deviation(self, simplex_term: str) -> Optional[float]:
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
        if len(baseline) < 10:  # Need minimum samples for meaningful baseline
            return None

        current = self.simplex_scores[simplex_term]
        mean = np.mean(baseline)
        std = np.std(baseline)

        if std < 0.001:  # Avoid division by zero
            return 0.0

        return (current - mean) / std

    def get_combined_activation(
        self,
        concept_term: str,
        layer: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get both hierarchical and simplex activation for a concept.

        This provides the dual-view described in MAP Meld Protocol §12.3:
        - Hierarchical: "Is this concept prominent vs siblings?"
        - Simplex: "How intense is this drive vs baseline?"

        Args:
            concept_term: Name of the concept
            layer: Optional layer hint (uses any matching if not specified)

        Returns:
            Dict with:
                - concept_term: str
                - hierarchical: float or None (concept probe activation)
                - simplex: float or None (simplex probe activation)
                - simplex_deviation: float or None (std devs from baseline)
                - interpretation: str (combined interpretation)
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
            # Find any matching concept
            for key in self.concept_metadata.keys():
                if key[0] == concept_term:
                    concept_key = key
                    break

        if concept_key and concept_key in self.probe_scores:
            result['hierarchical'] = self.probe_scores[concept_key]

        # Get simplex activation if bound
        if concept_term in self.simplex_bindings:
            simplex_term = self.simplex_bindings[concept_term]
            if simplex_term in self.simplex_scores:
                result['simplex'] = self.simplex_scores[simplex_term]
                result['simplex_deviation'] = self.get_simplex_deviation(simplex_term)

        # Generate interpretation
        h = result['hierarchical']
        s = result['simplex']

        if h is not None and s is not None:
            h_high = h > 0.6
            s_high = s > 0.6  # Or use deviation threshold

            if h_high and s_high:
                result['interpretation'] = 'active_elevated'  # Discussing + feeling it
            elif h_high and not s_high:
                result['interpretation'] = 'discussing_not_activated'  # Talking about, not feeling
            elif not h_high and s_high:
                result['interpretation'] = 'implicit_elevated'  # Not discussing, but feeling (concerning)
            else:
                result['interpretation'] = 'not_relevant'  # Neither discussing nor feeling
        elif h is not None:
            result['interpretation'] = 'hierarchical_only'
        elif s is not None:
            result['interpretation'] = 'simplex_only'

        return result

    def get_all_simplex_activations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current activations for all loaded simplexes.

        Returns:
            Dict mapping simplex_term to activation info
        """
        results = {}
        for simplex_term in self.loaded_simplex_probes:
            results[simplex_term] = {
                'activation': self.simplex_scores.get(simplex_term, 0.0),
                'deviation': self.get_simplex_deviation(simplex_term),
                'always_on': simplex_term in self.always_on_simplexes,
                'bound_to': [
                    concept for concept, simplex in self.simplex_bindings.items()
                    if simplex == simplex_term
                ]
            }
        return results


__all__ = [
    "DynamicProbeManager",
    "ConceptMetadata",
    "ProbeRole",
    "SimplexBinding",
    # Re-export manifest types for convenience
    "DeploymentManifest",
    "ManifestResolver",
    "LoadPriority",
    "ProbeEnvelope",
    "ProbeEnvelopeRule",
    "ProbeExpansionResult",
    "PRESET_MANIFESTS",
]

# Lazy imports for manifest types
def __getattr__(name):
    if name in (
        "DeploymentManifest", "ManifestResolver", "LoadPriority",
        "ProbeEnvelope", "ProbeEnvelopeRule", "ProbeExpansionResult",
        "PRESET_MANIFESTS",
    ):
        from .deployment_manifest import (
            DeploymentManifest,
            ManifestResolver,
            LoadPriority,
            ProbeEnvelope,
            ProbeEnvelopeRule,
            ProbeExpansionResult,
            PRESET_MANIFESTS,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
