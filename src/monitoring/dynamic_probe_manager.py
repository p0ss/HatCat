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
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple MLP classifier matching SUMO training architecture."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ConceptMetadata:
    """Metadata for a single SUMO concept."""

    def __init__(
        self,
        sumo_term: str,
        layer: int,
        category_children: List[str],
        synset_count: int,
        sumo_depth: int,
    ):
        self.sumo_term = sumo_term
        self.layer = layer
        self.category_children = category_children
        self.synset_count = synset_count
        self.sumo_depth = sumo_depth

        # Probe paths (set by manager)
        self.activation_probe_path: Optional[Path] = None
        self.text_probe_path: Optional[Path] = None
        self.has_text_probe: bool = False
        self.has_activation_probe: bool = False


class DynamicProbeManager:
    """
    Manages dynamic loading/unloading of SUMO concept probes.

    Strategy:
    1. Always keep base layers (0-1) loaded for broad coverage
    2. Load children when parent confidence > threshold
    3. Unload branches when all concepts < min_confidence
    4. Track access patterns for intelligent caching
    """

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

        # Handle probe pack vs legacy probes_dir
        if probe_pack_id:
            from src.registry import ProbePackRegistry
            registry = ProbePackRegistry()
            probe_pack = registry.get_pack(probe_pack_id)
            if not probe_pack:
                raise ValueError(f"Probe pack not found: {probe_pack_id}")

            self.probes_dir = probe_pack.pack_dir
            self.activation_probes_dir = probe_pack.activation_probes_dir
            self.text_probes_dir = probe_pack.text_probes_dir
            self.using_probe_pack = True
            print(f"Using probe pack: {probe_pack_id}")
        else:
            # Auto-detect: if probes_dir doesn't exist, try to find a probe pack
            if not probes_dir.exists():
                from src.registry import ProbePackRegistry
                registry = ProbePackRegistry()
                available_packs = registry.list_packs()

                if available_packs:
                    # Use first available probe pack
                    probe_pack = available_packs[0]
                    print(f"⚠ Probes directory not found: {probes_dir}")
                    print(f"✓ Auto-detected probe pack: {probe_pack.probe_pack_id}")
                    self.probes_dir = probe_pack.pack_dir
                    self.activation_probes_dir = probe_pack.activation_probes_dir
                    self.text_probes_dir = probe_pack.text_probes_dir
                    self.using_probe_pack = True
                else:
                    raise ValueError(f"Probes directory not found and no probe packs available: {probes_dir}")
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

        # Load metadata
        self._load_all_metadata()

        # Load base layers
        print(f"\nInitializing DynamicProbeManager...")
        print(f"  Base layers: {base_layers}")
        print(f"  Load threshold: {load_threshold}")
        print(f"  Max probes in memory: {max_loaded_probes}")
        self._load_base_layers()

    def _load_all_metadata(self):
        """Load lightweight metadata for all concepts (all layers)."""
        layer_files = sorted(self.layers_data_dir.glob("layer*.json"))

        total_concepts = 0
        for layer_file in layer_files:
            layer = int(layer_file.stem.replace('layer', ''))

            with open(layer_file) as f:
                layer_data = json.load(f)

            for concept in layer_data['concepts']:
                sumo_term = concept['sumo_term']

                metadata = ConceptMetadata(
                    sumo_term=sumo_term,
                    layer=layer,
                    category_children=concept.get('category_children', []),
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

                    # Probe pack structure: probes/text/{concept}_text_probe.joblib
                    text_probe_path = self.text_probes_dir / f"{sumo_term}_text_probe.joblib"
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

                    # Text probe path
                    text_probe_dir = layer_dir / "text_probes"
                    text_probe_path = text_probe_dir / f"{sumo_term}_text_probe.joblib"
                    if text_probe_path.exists():
                        metadata.text_probe_path = text_probe_path
                        metadata.has_text_probe = True

                concept_key = (sumo_term, layer)
                self.concept_metadata[concept_key] = metadata

                # Build parent-child mappings
                # Note: children are referenced by name only, need to find their layer later
                total_concepts += 1

        # Build parent-child mappings (after all concepts loaded)
        for concept_key, metadata in self.concept_metadata.items():
            sumo_term, layer = concept_key
            for child_name in metadata.category_children:
                # Find child concept (should be in next layer or same layer)
                child_key = None
                for (cname, clayer) in self.concept_metadata.keys():
                    if cname == child_name and clayer >= layer:
                        child_key = (cname, clayer)
                        break

                if child_key:
                    self.parent_to_children[concept_key].append(child_key)
                    self.child_to_parent[child_key] = concept_key

        print(f"\n✓ Loaded metadata for {total_concepts} concepts across {len(layer_files)} layers")
        print(f"  Parent-child relationships: {len(self.parent_to_children)}")

    def _load_base_layers(self):
        """Load base layers for broad coverage."""
        for layer in self.base_layers:
            layer_concept_keys = [
                key for key in self.concept_metadata.keys()
                if key[1] == layer
            ]
            self._load_concepts(layer_concept_keys, reason="base_layer")

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

    def _load_concepts(self, concept_keys: List[Tuple[str, int]], reason: str = "dynamic"):
        """
        Load activation and/or text probes for specified concepts.

        Uses batch loading and model pool for optimal performance.
        """
        # Filter out already loaded
        keys_to_load_activation = []
        keys_to_load_text = []

        for concept_key in concept_keys:
            # Check activation probes
            if self.use_activation_probes and concept_key not in self.loaded_activation_probes:
                keys_to_load_activation.append(concept_key)
            elif self.use_activation_probes:
                self.stats['cache_hits'] += 1

            # Check text probes
            if self.use_text_probes and concept_key not in self.loaded_text_probes:
                keys_to_load_text.append(concept_key)

        if not keys_to_load_activation and not keys_to_load_text:
            return

        # === LOAD ACTIVATION PROBES ===
        if self.use_activation_probes and keys_to_load_activation:
            # Infer hidden dim from first probe if needed
            if self.hidden_dim is None:
                first_key = keys_to_load_activation[0]
                metadata = self.concept_metadata.get(first_key)
                if metadata and metadata.activation_probe_path:
                    state_dict = torch.load(metadata.activation_probe_path, map_location='cpu')
                    first_key_name = list(state_dict.keys())[0]
                    self.hidden_dim = state_dict[first_key_name].shape[1]
                    print(f"  Inferred hidden_dim: {self.hidden_dim}")

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

                # Load text probe (joblib serialized sklearn pipeline)
                try:
                    text_probe = joblib.load(metadata.text_probe_path)
                    self.loaded_text_probes[concept_key] = text_probe
                    self.stats['total_loads'] += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to load text probe for {concept_key[0]}: {e}")

    def _unload_cold_probes(self):
        """Unload probes with consistently low scores to free memory."""
        if len(self.loaded_probes) <= self.max_loaded_probes:
            return

        # Don't unload base layer probes
        base_concept_keys = {
            key for key in self.concept_metadata.keys()
            if key[1] in self.base_layers
        }

        # Find cold probes (low scores, not in base layers)
        cold_probes = [
            (key, score)
            for key, score in self.probe_scores.items()
            if key not in base_concept_keys and score < self.unload_threshold
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
        """
        if not self.aggressive_pruning:
            return

        if len(self.loaded_probes) <= self.keep_top_k:
            return

        # Don't unload base layer probes
        base_concept_keys = {
            key for key in self.concept_metadata.keys()
            if key[1] in self.base_layers
        }

        # Separate base and non-base probes
        base_probes = {k: v for k, v in self.probe_scores.items() if k in base_concept_keys}
        non_base_probes = [(k, v) for k, v in self.probe_scores.items() if k not in base_concept_keys]

        # Sort non-base by score (highest first)
        non_base_probes.sort(key=lambda x: x[1], reverse=True)

        # Keep top (keep_top_k - num_base) non-base probes
        num_base = len(base_probes)
        keep_non_base = max(0, self.keep_top_k - num_base)

        # Unload everything beyond top K
        to_unload = non_base_probes[keep_non_base:]

        for key, _ in to_unload:
            if key in self.loaded_probes:
                # Return model to pool before deleting
                self._return_model_to_pool(self.loaded_probes[key])
                del self.loaded_probes[key]
                del self.probe_scores[key]
                self.stats['total_unloads'] += 1

    def detect_and_expand(
        self,
        hidden_state: torch.Tensor,
        top_k: int = 10,
        return_timing: bool = False,
    ) -> Tuple[List[Tuple[str, float, int]], Optional[Dict]]:
        """
        Detect concepts in hidden state, dynamically loading children as needed.

        Args:
            hidden_state: Hidden state tensor [1, hidden_dim] or [hidden_dim]
            top_k: Return top K concepts
            return_timing: Return detailed timing breakdown

        Returns:
            (concept_scores, timing_info)
            concept_scores: List of (concept_name, probability, layer)
            timing_info: Dict with timing breakdown (if return_timing=True)
        """
        timing = {} if return_timing else None
        start = time.time()

        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        # 1. Run all currently loaded probes
        t1 = time.time()
        current_scores = {}
        with torch.inference_mode():
            for concept_key, probe in self.loaded_probes.items():
                prob = probe(hidden_state).item()
                current_scores[concept_key] = prob
                self.probe_scores[concept_key] = prob
                self.probe_access_count[concept_key] += 1

        if timing is not None:
            timing['initial_detection'] = (time.time() - t1) * 1000

        # 2. Identify high-confidence parents and load their children
        t2 = time.time()
        child_keys_to_load = []
        for concept_key, prob in current_scores.items():
            if prob > self.load_threshold:
                # Get children from parent-child mapping
                child_keys = self.parent_to_children.get(concept_key, [])
                for child_key in child_keys:
                    if child_key not in self.loaded_probes:
                        child_keys_to_load.append(child_key)

        if child_keys_to_load:
            self._load_concepts(child_keys_to_load, reason="dynamic_expansion")

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
                        prob = probe(hidden_state).item()
                        current_scores[concept_key] = prob
                        self.probe_scores[concept_key] = prob
                        self.probe_access_count[concept_key] += 1

        if timing is not None:
            timing['child_detection'] = (time.time() - t3) * 1000

        # 4. Pruning: aggressive top-K or conservative cold probe removal
        t4 = time.time()
        if self.aggressive_pruning:
            self._aggressive_prune_to_top_k()
        else:
            self._unload_cold_probes()
        if timing is not None:
            timing['cache_management'] = (time.time() - t4) * 1000

        # 5. Sort and return top K
        results = []
        for concept_key, prob in current_scores.items():
            concept_name, layer = concept_key
            results.append((concept_name, prob, layer))

        results.sort(key=lambda x: x[1], reverse=True)

        if timing is not None:
            timing['total'] = (time.time() - start) * 1000
            timing['loaded_probes'] = len(self.loaded_probes)

        return results[:top_k], timing

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
        print(f"Total loads: {self.stats['total_loads']}")
        print(f"Total unloads: {self.stats['total_unloads']}")
        print(f"Cache hits: {self.stats['cache_hits']}")
        print(f"Cache misses: {self.stats['cache_misses']}")

        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
            print(f"Cache hit rate: {hit_rate:.1%}")

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


__all__ = [
    "DynamicProbeManager",
    "ConceptMetadata",
]
