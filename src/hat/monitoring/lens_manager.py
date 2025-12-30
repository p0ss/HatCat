#!/usr/bin/env python3
"""
Dynamic Hierarchical Lens Manager

Loads/unloads SUMO concept lenses on-demand based on parent confidence scores.
Enables running 110K+ concepts with minimal memory footprint.

Architecture:
- Always keep layers 0-1-2 loaded (base coverage)
- When parent fires high → load its children
- Unload cold branches to free memory
- Support both activation and text lenses
- Support multiple lens roles: concept, simplex, behavioral, category

This file is the main orchestrator that uses modular components from:
- lens_types.py: Core types (LensRole, SimpleMLP, ConceptMetadata, etc.)
- lens_batched.py: BatchedLensBank for efficient inference
- lens_hierarchy.py: HierarchyManager for parent-child relationships
- lens_cache.py: LensCacheManager for multi-tier caching
- lens_loader.py: LensLoader for disk I/O
- lens_simplex.py: SimplexManager for intensity monitoring
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

# Import modular components
from .lens_types import (
    LensRole,
    SimpleMLP,
    SimplexBinding,
    ConceptMetadata,
    detect_layer_norm,
    create_lens_from_state_dict,
)
from .lens_batched import BatchedLensBank
from .lens_hierarchy import HierarchyManager
from .lens_cache import LensCacheManager
from .lens_loader import LensLoader, MetadataLoader
from .lens_simplex import SimplexManager

if TYPE_CHECKING:
    from .deployment_manifest import DeploymentManifest, ManifestResolver


class DynamicLensManager:
    """
    Manages dynamic loading/unloading of SUMO concept lenses.

    Strategy:
    1. Always keep base layers (0-1-2) loaded for broad coverage
    2. Load children when parent confidence > threshold
    3. Unload branches when all concepts < min_confidence
    4. Track access patterns for intelligent caching
    """

    @staticmethod
    def discover_concept_packs(pack_dir: Path = Path("lens_packs/concept_packs")) -> Dict[str, Path]:
        """Discover all MAP-compliant concept packs."""
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
    def discover_lens_packs(
        lens_packs_dir: Path = Path("lens_packs"),
        substrate_id: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Discover all available lens packs (both legacy and MAP-compliant)."""
        packs = {}
        if not lens_packs_dir.exists():
            return packs

        for pack_path in lens_packs_dir.iterdir():
            if not pack_path.is_dir() or pack_path.name == "concept_packs":
                continue

            pack_json = pack_path / "pack.json"

            if pack_json.exists():
                try:
                    with open(pack_json) as f:
                        pack_data = json.load(f)

                    if "concept_pack_id" in pack_data and "substrate_id" in pack_data:
                        lens_substrate = pack_data.get("substrate_id", "")
                        if substrate_id and substrate_id not in lens_substrate:
                            continue

                        packs[pack_path.name] = {
                            "path": pack_path,
                            "type": "map",
                            "concept_pack_id": pack_data["concept_pack_id"],
                            "substrate_id": pack_data["substrate_id"]
                        }
                    else:
                        packs[pack_path.name] = {"path": pack_path, "type": "legacy"}
                except Exception as e:
                    print(f"Warning: Failed to read {pack_json}: {e}")
            else:
                if (pack_path / "activation_lenses").exists() or \
                   (pack_path / "text_lenses").exists() or \
                   list(pack_path.glob("layer*")):
                    packs[pack_path.name] = {"path": pack_path, "type": "legacy"}

        return packs

    def __init__(
        self,
        layers_data_dir: Path = Path("data/concept_graph/abstraction_layers"),
        lenses_dir: Path = Path("results/sumo_classifiers"),
        device: str = "cuda",
        base_layers: List[int] = [0, 1, 2],
        load_threshold: float = 0.5,
        unload_threshold: float = 0.1,
        max_loaded_lenses: int = 500,
        keep_top_k: int = 50,
        aggressive_pruning: bool = True,
        use_text_lenses: bool = False,
        use_activation_lenses: bool = True,
        lens_pack_id: Optional[str] = None,
        normalize_hidden_states: bool = True,
        manifest: Optional["DeploymentManifest"] = None,
        manifest_path: Optional[Path] = None,
    ):
        self.layers_data_dir = layers_data_dir
        self.device = device
        self.base_layers = base_layers
        self.load_threshold = load_threshold
        self.unload_threshold = unload_threshold
        self.use_text_lenses = use_text_lenses
        self.use_activation_lenses = use_activation_lenses
        self.normalize_hidden_states = normalize_hidden_states
        self._layer_norm = None

        # === RESOLVE LENS PACK ===
        self.using_lens_pack = False
        self.activation_lenses_dir = None
        self.text_lenses_dir = None

        if lens_pack_id:
            self._setup_lens_pack(lens_pack_id, lenses_dir)
        elif not lenses_dir.exists():
            self._auto_detect_lens_pack(lenses_dir)
        else:
            self.lenses_dir = lenses_dir
            self.activation_lenses_dir = None
            self.text_lenses_dir = None

        # === INITIALIZE MODULAR COMPONENTS ===

        # Cache manager
        self.cache = LensCacheManager(
            device=device,
            max_loaded_lenses=max_loaded_lenses,
            keep_top_k=keep_top_k,
            aggressive_pruning=aggressive_pruning,
        )

        # Hierarchy manager
        self.hierarchy = HierarchyManager()

        # Simplex manager
        self.simplex = SimplexManager(device=device)

        # Metadata storage
        self.concept_metadata: Dict[Tuple[str, int], ConceptMetadata] = {}

        # Aliases for backward compatibility
        self.loaded_activation_lenses = self.cache.loaded_activation_lenses
        self.loaded_text_lenses = self.cache.loaded_text_lenses
        self.loaded_lenses = self.cache.loaded_lenses
        self.lens_scores = self.cache.lens_scores
        self.lens_access_count = self.cache.lens_access_count
        self.base_layer_lenses = self.cache.base_layer_lenses
        self.warm_cache = self.cache.warm_cache
        self.stats = self.cache.stats

        # Hierarchy aliases
        self.parent_to_children = self.hierarchy.parent_to_children
        self.child_to_parent = self.hierarchy.child_to_parent
        self.leaf_concepts = self.hierarchy.leaf_concepts

        # Simplex aliases
        self.loaded_simplex_lenses = self.simplex.loaded_simplex_lenses
        self.simplex_scores = self.simplex.simplex_scores
        self.simplex_bindings = self.simplex.simplex_bindings
        self.always_on_simplexes = self.simplex.always_on_simplexes

        # Hidden dimension
        self.hidden_dim: Optional[int] = None

        # Manifest
        self.manifest: Optional["DeploymentManifest"] = None
        self.manifest_resolver: Optional["ManifestResolver"] = None

        if manifest_path is not None:
            from .deployment_manifest import DeploymentManifest
            self.manifest = DeploymentManifest.from_json(manifest_path)
            print(f"✓ Loaded manifest: {self.manifest.manifest_id}")
        elif manifest is not None:
            self.manifest = manifest
            print(f"✓ Using manifest: {self.manifest.manifest_id}")

        # Try to load calibration data from lens pack
        calibration_path = self.lenses_dir / "calibration.json"
        if calibration_path.exists():
            if self.manifest is None:
                from .deployment_manifest import DeploymentManifest
                self.manifest = DeploymentManifest.default("auto-calibrated")
            count = self.manifest.load_calibration(calibration_path)
            print(f"✓ Loaded calibration for {count} concepts")

        # === LOAD METADATA ===
        self._load_all_metadata()

        # Initialize manifest resolver after metadata
        if self.manifest is not None:
            from .deployment_manifest import ManifestResolver
            self.manifest_resolver = ManifestResolver(
                manifest=self.manifest,
                concept_hierarchy=self.concept_metadata,
                parent_to_children=self.parent_to_children,
                child_to_parent=self.child_to_parent,
            )
            if self.manifest.layer_bounds.always_load_layers:
                self.base_layers = self.manifest.layer_bounds.always_load_layers
            if self.manifest.dynamic_loading.max_loaded_concepts:
                self.cache.max_loaded_lenses = self.manifest.dynamic_loading.max_loaded_concepts
            self.load_threshold = self.manifest.dynamic_loading.parent_threshold
            self.unload_threshold = self.manifest.dynamic_loading.unload_threshold

        # Create lens loader
        self.loader = LensLoader(
            lenses_dir=self.lenses_dir,
            device=device,
            use_activation_lenses=use_activation_lenses,
            use_text_lenses=use_text_lenses,
        )

        # === LOAD BASE LAYERS ===
        print(f"\nInitializing DynamicLensManager...")
        print(f"  Base layers: {self.base_layers}")
        print(f"  Load threshold: {self.load_threshold}")
        print(f"  Max lenses in memory: {self.cache.max_loaded_lenses}")
        if self.manifest:
            print(f"  Manifest: {self.manifest.manifest_id}")
        self._load_base_layers()

    def _setup_lens_pack(self, lens_pack_id: str, lenses_dir: Path):
        """Setup lens pack from ID."""
        import warnings
        warnings.warn(
            f"lens_pack_id parameter is deprecated. Please migrate to MAP-compliant structure.",
            DeprecationWarning,
            stacklevel=3
        )

        available_packs = self.discover_lens_packs()
        if lens_pack_id not in available_packs:
            raise ValueError(f"Lens pack not found: {lens_pack_id}")

        pack_info = available_packs[lens_pack_id]
        pack_path = pack_info["path"]
        self.lenses_dir = pack_path

        pack_info_json = pack_path / "pack_info.json"
        if pack_info_json.exists():
            with open(pack_info_json) as f:
                pack_info_data = json.load(f)
            source_pack = pack_info_data.get("source_pack")
            if source_pack:
                concept_pack_hierarchy = Path("concept_packs") / source_pack / "hierarchy"
                if concept_pack_hierarchy.exists():
                    self.layers_data_dir = concept_pack_hierarchy

        pack_json = pack_path / "pack.json"
        if pack_json.exists():
            with open(pack_json) as f:
                pack_data = json.load(f)
            lens_paths = pack_data.get("lens_paths", {})
            self.activation_lenses_dir = pack_path / lens_paths.get("activation_lenses", "activation_lenses")
            self.text_lenses_dir = pack_path / lens_paths.get("text_lenses", "text_lenses")
        else:
            self.activation_lenses_dir = pack_path / "activation_lenses"
            self.text_lenses_dir = pack_path / "text_lenses"

        self.using_lens_pack = True
        print(f"✓ Using lens pack: {lens_pack_id}")

    def _auto_detect_lens_pack(self, lenses_dir: Path):
        """Auto-detect lens pack when lenses_dir doesn't exist."""
        available_packs = self.discover_lens_packs()
        if not available_packs:
            raise ValueError(f"Lenses directory not found and no lens packs available: {lenses_dir}")

        lens_pack_id = sorted(available_packs.keys())[0]
        pack_info = available_packs[lens_pack_id]
        pack_path = pack_info["path"]

        print(f"⚠ Lenses directory not found: {lenses_dir}")
        print(f"✓ Auto-detected lens pack: {lens_pack_id}")

        self.lenses_dir = pack_path

        pack_info_json = pack_path / "pack_info.json"
        if pack_info_json.exists():
            with open(pack_info_json) as f:
                pack_info_data = json.load(f)
            source_pack = pack_info_data.get("source_pack")
            if source_pack:
                concept_pack_hierarchy = Path("concept_packs") / source_pack / "hierarchy"
                if concept_pack_hierarchy.exists():
                    self.layers_data_dir = concept_pack_hierarchy

        pack_json = pack_path / "pack.json"
        if pack_json.exists():
            with open(pack_json) as f:
                pack_data = json.load(f)
            lens_paths = pack_data.get("lens_paths", {})
            self.activation_lenses_dir = pack_path / lens_paths.get("activation_lenses", "activation_lenses")
            self.text_lenses_dir = pack_path / lens_paths.get("text_lenses", "text_lenses")
        else:
            self.activation_lenses_dir = pack_path / "activation_lenses"
            self.text_lenses_dir = pack_path / "text_lenses"

        self.using_lens_pack = True

    def _load_all_metadata(self):
        """Load metadata for all concepts."""
        metadata_loader = MetadataLoader(
            layers_data_dir=self.layers_data_dir,
            lenses_dir=self.lenses_dir,
            using_lens_pack=self.using_lens_pack,
            activation_lenses_dir=self.activation_lenses_dir,
            text_lenses_dir=self.text_lenses_dir,
        )
        self.concept_metadata = metadata_loader.load_all_metadata()

        # Load hierarchy
        hierarchy_path = None
        if self.using_lens_pack:
            lens_pack_hierarchy = self.lenses_dir / "hierarchy.json"
            if lens_pack_hierarchy.exists():
                hierarchy_path = lens_pack_hierarchy

        if not hierarchy_path:
            concept_pack_hierarchy = self.layers_data_dir / "hierarchy.json"
            if concept_pack_hierarchy.exists():
                hierarchy_path = concept_pack_hierarchy

        if hierarchy_path:
            self.hierarchy.load_authoritative_hierarchy(hierarchy_path, self.concept_metadata)

        # Build from metadata as fallback
        pre_count = len(self.hierarchy.parent_to_children)
        self.hierarchy.build_from_metadata(self.concept_metadata)
        added = len(self.hierarchy.parent_to_children) - pre_count
        if added > 0:
            print(f"  Added {added} parents from metadata (fallback)")

        stats = self.hierarchy.get_stats()
        print(f"  Parent-child relationships: {stats['total_relationships']} ({stats['unique_parents']} unique parents)")
        print(f"  Leaf concepts: {stats['leaf_concepts']}")

    def _load_base_layers(self):
        """Load base layers for broad coverage."""
        if self.manifest_resolver is not None:
            # Use manifest's always_load_layers, not the parameter
            # Skip sibling expansion - let detect_and_expand handle it dynamically
            always_load = self.manifest.layer_bounds.always_load_layers
            excludes = self.manifest.explicit_concepts.always_exclude

            base_concepts = set()
            for layer in always_load:
                layer_keys = [key for key in self.concept_metadata.keys()
                              if key[1] == layer and key[0] not in excludes]
                base_concepts.update(layer_keys)

            self._load_concepts(list(base_concepts), reason="base_layer")
            print(f"  Manifest: loading layers {always_load}, {len(base_concepts)} concepts (excluding {len(excludes)} over-firers)")
        else:
            for layer in self.base_layers:
                layer_keys = [key for key in self.concept_metadata.keys() if key[1] == layer]
                self._load_concepts(layer_keys, reason="base_layer")

        # Mark as base layer lenses
        for key in self.cache.loaded_activation_lenses.keys():
            self.cache.base_layer_lenses.add(key)

        print(f"✓ Base layers loaded: {len(self.cache.loaded_lenses)} lenses")

    def _load_concepts(self, concept_keys: List[Tuple[str, int]], reason: str = "dynamic"):
        """Load lenses for specified concepts."""
        loaded = self.loader.load_concepts(
            concept_keys,
            self.concept_metadata,
            self.cache,
            reason=reason,
        )

        # Update hidden dim if discovered
        if self.hidden_dim is None and self.cache.hidden_dim is not None:
            self.hidden_dim = self.cache.hidden_dim
            self.simplex.set_hidden_dim(self.hidden_dim)

        self.cache.mark_lens_bank_dirty()
        return loaded

    def detect_and_expand(
        self,
        hidden_state: torch.Tensor,
        top_k: int = 10,
        return_timing: bool = False,
        return_logits: bool = False,
        skip_pruning: bool = False,
        max_expansion_depth: int = None,
        use_calibration: bool = True,
    ) -> Tuple[List[Tuple[str, float, int]], Optional[Dict]]:
        """
        Detect concepts in hidden state, dynamically loading children as needed.

        Args:
            hidden_state: Hidden state tensor [1, hidden_dim] or [hidden_dim]
            top_k: Return top K concepts
            return_timing: Return detailed timing breakdown
            return_logits: If True, return (concept_name, probability, logit, layer) tuples
            skip_pruning: If True, skip aggressive pruning
            max_expansion_depth: Max hierarchy depth to expand
            use_calibration: If True and calibration data available, normalize scores.
                Normalized scores have meaning:
                - 1.0 = firing at self_mean level (genuine signal)
                - 0.5 = firing at cross_mean level (noise floor for this concept)
                - 0.0 = floor

        Returns:
            (concept_scores, timing_info)
        """
        timing = {} if return_timing else None
        start = time.time()

        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        # Normalize hidden states
        if self.normalize_hidden_states:
            hidden_dim = hidden_state.shape[-1]
            if self._layer_norm is None or self._layer_norm.normalized_shape[0] != hidden_dim:
                self._layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False).to(hidden_state.device)
            hidden_state = self._layer_norm(hidden_state)

        hidden_state = hidden_state.to(self.device)

        # Match dtype to lens dtype
        if self.cache.loaded_lenses:
            sample_lens = next(iter(self.cache.loaded_lenses.values()))
            lens_dtype = next(sample_lens.parameters()).dtype
            if hidden_state.dtype != lens_dtype:
                hidden_state = hidden_state.to(dtype=lens_dtype)

        # 1. Run all currently loaded lenses
        t1 = time.time()
        current_scores = {}
        current_logits = {} if return_logits else None

        with torch.inference_mode():
            lens_bank = self.cache.get_lens_bank()
            use_batched = lens_bank is not None and self.cache.is_bank_compiled()

            if use_batched:
                if return_logits:
                    current_scores, current_logits = lens_bank(hidden_state, return_logits=True)
                else:
                    current_scores = lens_bank(hidden_state)

                for concept_key in current_scores:
                    self.cache.lens_scores[concept_key] = current_scores[concept_key]
                    self.cache.lens_access_count[concept_key] += 1
            else:
                for concept_key, lens in self.cache.loaded_lenses.items():
                    if return_logits:
                        prob, logit = lens(hidden_state, return_logits=True)
                        prob = prob.item()
                        logit = logit.item()
                        current_logits[concept_key] = logit
                    else:
                        prob = lens(hidden_state).item()
                    current_scores[concept_key] = prob
                    self.cache.lens_scores[concept_key] = prob
                    self.cache.lens_access_count[concept_key] += 1

        if timing is not None:
            timing['initial_detection'] = (time.time() - t1) * 1000

        # 2. Iterative decomposition: replace parents with children
        t2 = time.time()
        total_children_loaded = 0
        decomposition_iterations = 0
        max_iterations = max_expansion_depth or getattr(self, 'max_expansion_depth', 5)
        decomposed_parents = set()

        while decomposition_iterations < max_iterations:
            decomposition_iterations += 1

            # Get current top-k (excluding already-decomposed parents)
            eligible_scores = {k: v for k, v in current_scores.items() if k not in decomposed_parents}
            sorted_concepts = sorted(eligible_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_concepts = sorted_concepts[:top_k]

            # Find parents in top-k that have children
            parents_to_decompose = []
            child_keys_to_load = set()

            for concept_key, prob in top_k_concepts:
                child_keys = self.hierarchy.get_children(concept_key)
                if child_keys:
                    parents_to_decompose.append(concept_key)
                    for child_key in child_keys:
                        if child_key not in self.cache.loaded_lenses:
                            if self.manifest_resolver is not None:
                                if self.manifest_resolver.should_load_concept(child_key):
                                    child_keys_to_load.add(child_key)
                            else:
                                child_keys_to_load.add(child_key)

            if not parents_to_decompose:
                break

            decomposed_parents.update(parents_to_decompose)

            # Load and score new children
            if child_keys_to_load:
                if self.manifest_resolver is not None:
                    child_keys_to_load = self.manifest_resolver.expand_with_siblings(child_keys_to_load)
                    # Filter out explicitly excluded concepts after sibling expansion
                    excludes = self.manifest.explicit_concepts.always_exclude
                    child_keys_to_load = {k for k in child_keys_to_load
                                          if k not in self.cache.loaded_lenses and k[0] not in excludes}

                if child_keys_to_load:
                    t_load_start = time.time()
                    self._load_concepts(list(child_keys_to_load), reason="dynamic_expansion")
                    if timing is not None:
                        timing['_disk_load'] = timing.get('_disk_load', 0) + (time.time() - t_load_start) * 1000
                    total_children_loaded += len(child_keys_to_load)

                    # Score newly loaded lenses
                    t_score_start = time.time()
                    with torch.inference_mode():
                        for concept_key in child_keys_to_load:
                            if concept_key in self.cache.loaded_lenses:
                                lens = self.cache.loaded_lenses[concept_key]
                                if return_logits:
                                    prob, logit = lens(hidden_state, return_logits=True)
                                    prob = prob.item()
                                    logit = logit.item()
                                    current_logits[concept_key] = logit
                                else:
                                    prob = lens(hidden_state).item()
                                current_scores[concept_key] = prob
                                self.cache.lens_scores[concept_key] = prob
                                self.cache.lens_access_count[concept_key] += 1
                    if timing is not None:
                        timing['_child_scoring'] = timing.get('_child_scoring', 0) + (time.time() - t_score_start) * 1000

        if timing is not None:
            timing['child_loading'] = (time.time() - t2) * 1000
            timing['num_children_loaded'] = total_children_loaded
            timing['decomposition_iterations'] = decomposition_iterations
            timing['parents_decomposed'] = len(decomposed_parents)

        # 3. Cache management + pruning
        t4 = time.time()
        cache_hits_this_token = getattr(self.cache, '_last_warm_cache_hits', 0)
        cache_misses_this_token = total_children_loaded

        if not skip_pruning:
            # Get top-k from ALL current scores (including parents)
            # NOTE: Parents are NOT excluded from top-k calculation - they stay loaded
            # if they score highly, which is important for hierarchical coverage
            self.cache.prune_to_top_k(top_k, skip_base_layers=True)

        if timing is not None:
            timing['cache_management'] = (time.time() - t4) * 1000

        # 4. Build results
        # NOTE: We no longer filter to only leaf concepts - parents can appear in results
        # if they score highly and weren't decomposed. Calibration handles any overfiring.
        results = []

        # Get calibration data if available
        calibration_data = None
        if use_calibration and self.manifest is not None:
            calibration_data = self.manifest.concept_calibration

        for concept_key, prob in current_scores.items():
            # Only skip parents that were decomposed into children during this detection
            if concept_key in decomposed_parents:
                continue

            concept_name, layer = concept_key

            # Apply calibration normalization if available
            display_prob = prob
            if calibration_data:
                cal_key = f"{concept_name}_L{layer}"
                if cal_key in calibration_data:
                    display_prob = calibration_data[cal_key].normalize(prob)

            if return_logits:
                logit = current_logits.get(concept_key, 0.0)
                results.append((concept_name, display_prob, logit, layer))
            else:
                results.append((concept_name, display_prob, layer))

        results.sort(key=lambda x: x[1], reverse=True)
        top_k_results = results[:top_k]

        if timing is not None:
            timing['total'] = (time.time() - start) * 1000
            timing['loaded_lenses'] = len(self.cache.loaded_lenses)
            timing['cache_hits'] = cache_hits_this_token
            timing['cache_misses'] = cache_misses_this_token
            timing['warm_cache_size'] = len(self.cache.warm_cache)

        return top_k_results, timing

    def detect_and_expand_with_divergence(
        self,
        hidden_state: torch.Tensor,
        token_embedding: np.ndarray,
        top_k: int = 10,
        return_timing: bool = False,
    ) -> Tuple[Dict[str, Dict], Optional[Dict]]:
        """Detect concepts with divergence scores using embedding centroids."""
        from .centroid_detector import CentroidTextDetector

        timing = {} if return_timing else None
        start = time.time()

        detected_concepts, detect_timing = self.detect_and_expand(
            hidden_state, top_k=top_k, return_timing=True
        )

        if timing is not None:
            timing.update(detect_timing)

        t_centroid = time.time()
        concepts_with_divergence = {}

        for concept_name, activation_prob, layer in detected_concepts:
            centroid_path = self.lenses_dir / f"layer{layer}" / "embedding_centroids" / f"{concept_name}_centroid.npy"

            text_conf = None
            divergence = None

            if centroid_path.exists():
                try:
                    centroid_detector = CentroidTextDetector.load(centroid_path, concept_name)
                    text_conf = float(centroid_detector.predict(token_embedding))
                    divergence = float(activation_prob - text_conf)
                except Exception:
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
        concept_key = None
        if layer is not None:
            concept_key = (concept_name, layer)
            if concept_key not in self.concept_metadata:
                concept_key = None

        if concept_key is None:
            for key in self.concept_metadata.keys():
                if key[0] == concept_name:
                    concept_key = key
                    break

        if concept_key is None:
            return [concept_name]

        return self.hierarchy.get_path_to_root(concept_key)

    def print_stats(self):
        """Print manager statistics."""
        print("\n" + "=" * 80)
        print("DYNAMIC LENS MANAGER STATISTICS")
        print("=" * 80)
        print(f"Total concepts in metadata: {len(self.concept_metadata)}")
        print(f"Currently loaded lenses: {len(self.cache.loaded_lenses)}")
        print(f"Base layer lenses (protected): {len(self.cache.base_layer_lenses)}")
        print(f"Warm cache size: {len(self.cache.warm_cache)}")
        print(f"Total in memory: {len(self.cache.loaded_lenses) + len(self.cache.warm_cache)}")
        print(f"Total loads: {self.cache.stats['total_loads']}")
        print(f"Total unloads: {self.cache.stats['total_unloads']}")
        print(f"Cache hits: {self.cache.stats['cache_hits']}")
        print(f"Cache misses: {self.cache.stats['cache_misses']}")

        if self.cache.stats['cache_hits'] + self.cache.stats['cache_misses'] > 0:
            hit_rate = self.cache.stats['cache_hits'] / (self.cache.stats['cache_hits'] + self.cache.stats['cache_misses'])
            print(f"Cache hit rate: {hit_rate:.1%}")
        print("=" * 80)

    def reset_to_base(self, keep_warm_cache: bool = True):
        """Reset to only base layer lenses."""
        self.cache.reset_to_base(keep_warm_cache)

    def preload_pack_to_ram(self, max_ram_mb: int = None, priority_layers: List[int] = None):
        """Pre-load lens pack to CPU RAM (tepid cache)."""
        return self.cache.preload_to_ram(self.concept_metadata, max_ram_mb, priority_layers)

    def prewarm_from_prompt(self, hidden_state: torch.Tensor, top_k: int = 10):
        """Pre-warm cache by loading child lenses based on prompt hidden state."""
        _ = self.detect_and_expand(hidden_state, top_k=top_k)
        return len(self.cache.loaded_lenses) - len(self.cache.base_layer_lenses)

    # === SIMPLEX METHODS ===

    def load_simplex(self, simplex_term: str, lens_path: Path) -> bool:
        """Load a simplex lens for intensity monitoring."""
        return self.simplex.load_simplex(simplex_term, lens_path)

    def register_simplex_binding(self, concept_term: str, simplex_term: str, always_on: bool = False):
        """Register a binding between a concept and its simplex."""
        self.simplex.register_binding(concept_term, simplex_term, always_on)

    def detect_simplexes(self, hidden_state: torch.Tensor, simplex_terms: Optional[List[str]] = None):
        """Run simplex lenses and return activations."""
        return self.simplex.detect(hidden_state, simplex_terms)

    def get_simplex_deviation(self, simplex_term: str) -> Optional[float]:
        """Get current deviation from baseline for a simplex."""
        return self.simplex.get_deviation(simplex_term)

    def get_combined_activation(self, concept_term: str, layer: Optional[int] = None):
        """Get both hierarchical and simplex activation for a concept."""
        return self.simplex.get_combined_activation(concept_term, self.cache.lens_scores, layer)

    def get_all_simplex_activations(self):
        """Get current activations for all loaded simplexes."""
        return self.simplex.get_all_activations()

    # === BE WORKSPACE TOOLS ===

    def request_lens_expansion(self, branch: str, reason: str, depth: int = 2) -> Dict[str, Any]:
        """BE workspace tool: Request to expand lenses for a branch."""
        from .deployment_manifest import LensExpansionResult

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
            result = LensExpansionResult(success=True, cat_scope=None)

        # Find branch root
        branch_root_key = None
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

        # Get branch concepts
        branch_concepts = self.hierarchy.get_branch_concepts(branch_root_key, max_depth=depth)

        if self.manifest_resolver is not None:
            branch_concepts = {k for k in branch_concepts if self.manifest_resolver.should_load_concept(k)}
            branch_concepts = self.manifest_resolver.expand_with_siblings(branch_concepts)

        concepts_to_load = [k for k in branch_concepts if k not in self.cache.loaded_activation_lenses]

        if concepts_to_load:
            self._load_concepts(concepts_to_load, reason=f"be_introspection:{branch}")

        return {
            "success": True,
            "loaded_concepts": [k[0] for k in concepts_to_load],
            "cat_scope": result.cat_scope,
            "error": None,
        }

    def request_lens_collapse(self, branch: str, reason: str) -> Dict[str, Any]:
        """BE workspace tool: Collapse lenses for a branch."""
        if self.manifest_resolver is not None:
            must_enable = self.manifest_resolver.get_must_enable_branches()
            if branch in must_enable:
                return {
                    "success": False,
                    "collapsed_concepts": [],
                    "retained_concepts": [branch],
                    "error": f"Branch '{branch}' is in must_enable and cannot be collapsed",
                }

        branch_concepts_loaded = []
        retained = []

        for key in list(self.cache.loaded_activation_lenses.keys()):
            concept_name, layer = key
            if key in self.cache.base_layer_lenses:
                continue

            path = self.get_concept_path(concept_name, layer)
            if branch in path:
                if self.manifest_resolver is not None:
                    envelope = self.manifest_resolver.manifest.aperture
                    if envelope and concept_name in envelope.must_enable.branches:
                        retained.append(concept_name)
                        continue
                branch_concepts_loaded.append(key)

        self.cache.move_to_warm_cache(branch_concepts_loaded)
        self.cache.manage_cache_memory()

        return {
            "success": True,
            "collapsed_concepts": [k[0] for k in branch_concepts_loaded],
            "retained_concepts": retained,
            "error": None,
        }

    def get_introspection_reading(self, branch: str, hidden_state: torch.Tensor) -> Dict[str, Any]:
        """BE workspace tool: Get current lens readings for a branch."""
        branch_concepts = {}

        for key in self.cache.loaded_activation_lenses.keys():
            concept_name, layer = key
            path = self.get_concept_path(concept_name, layer)
            if branch in path:
                branch_concepts[key] = self.cache.loaded_activation_lenses[key]

        if not branch_concepts:
            return {
                "branch": branch,
                "readings": {},
                "top_concept": None,
                "interpretation": f"No lenses loaded for branch '{branch}'. Call request_lens_expansion first.",
            }

        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        if self.normalize_hidden_states:
            hidden_dim = hidden_state.shape[-1]
            if self._layer_norm is None or self._layer_norm.normalized_shape[0] != hidden_dim:
                self._layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False).to(hidden_state.device)
            hidden_state = self._layer_norm(hidden_state)

        if branch_concepts:
            sample_lens = next(iter(branch_concepts.values()))
            lens_dtype = next(sample_lens.parameters()).dtype
            if hidden_state.dtype != lens_dtype:
                hidden_state = hidden_state.to(dtype=lens_dtype)

        readings = {}
        with torch.inference_mode():
            for key, lens in branch_concepts.items():
                prob = lens(hidden_state).item()
                readings[key[0]] = prob

        top_concept = max(readings, key=readings.get) if readings else None
        top_score = readings.get(top_concept, 0.0)

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
        """Get a summary of the USH lens envelope."""
        if self.manifest_resolver is not None:
            return self.manifest_resolver.get_envelope_summary()
        return {"has_envelope": False, "mode": "unrestricted"}

    def get_loaded_fingerprint(self) -> str:
        """Get a fingerprint hash of currently loaded concepts."""
        if self.manifest_resolver is not None:
            return self.manifest_resolver.compute_fingerprint(set(self.cache.loaded_activation_lenses.keys()))

        import hashlib
        sorted_keys = sorted(self.cache.loaded_activation_lenses.keys())
        key_str = "|".join(f"{name}:{layer}" for name, layer in sorted_keys)
        return f"sha256:{hashlib.sha256(key_str.encode()).hexdigest()[:16]}"

    def get_manifest_summary(self) -> Dict[str, Any]:
        """Get a summary of the manifest configuration."""
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
            "loaded_concepts": len(self.cache.loaded_activation_lenses),
            "fingerprint": self.get_loaded_fingerprint(),
        }


__all__ = [
    "DynamicLensManager",
    "ConceptMetadata",
    "LensRole",
    "SimplexBinding",
    "SimpleMLP",
    "BatchedLensBank",
    "HierarchyManager",
    "LensCacheManager",
    "SimplexManager",
]
