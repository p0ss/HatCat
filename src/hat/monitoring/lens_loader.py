#!/usr/bin/env python3
"""
Lens Loading

Handles loading lenses from disk with support for:
- Batch loading with thread pool
- Bank format (consolidated lens files per parent)
- Legacy individual file format
- Text lenses (centroid-based and TF-IDF)
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, TYPE_CHECKING

import torch
import torch.nn as nn

from .lens_types import (
    SimpleMLP,
    ConceptMetadata,
    detect_layer_norm,
    create_lens_from_state_dict,
)

if TYPE_CHECKING:
    from .lens_cache import LensCacheManager


class LensLoader:
    """
    Handles loading lens weights from disk.

    Supports:
    - Individual .pt files per concept
    - Banked format (consolidated per-parent files)
    - Parallel loading with ThreadPoolExecutor
    """

    def __init__(
        self,
        lenses_dir: Path,
        device: str = "cuda",
        use_activation_lenses: bool = True,
        use_text_lenses: bool = False,
    ):
        self.lenses_dir = lenses_dir
        self.device = device
        self.use_activation_lenses = use_activation_lenses
        self.use_text_lenses = use_text_lenses

        # Bank format support
        self._is_banked_pack: bool = False
        self._bank_index: Dict[str, Dict] = {}
        self._loaded_banks: Dict[str, Dict[str, Dict]] = {}
        self._banks_dir: Optional[Path] = None
        self._individual_dir: Optional[Path] = None

        # Detect bank format
        self._detect_banked_pack()

    def _detect_banked_pack(self):
        """Detect if this is a banked lens pack and load bank index."""
        bank_index_path = self.lenses_dir / "bank_index.json"
        if bank_index_path.exists():
            self._is_banked_pack = True
            self._bank_index = json.load(open(bank_index_path))
            self._banks_dir = self.lenses_dir / "banks"
            self._individual_dir = self.lenses_dir / "individual"
            print(f"  Detected banked pack: {len(self._bank_index)} banks")
        else:
            self._is_banked_pack = False

    def load_concepts(
        self,
        concept_keys: List[Tuple[str, int]],
        concept_metadata: Dict[Tuple[str, int], ConceptMetadata],
        cache_manager: "LensCacheManager",
        reason: str = "dynamic",
    ) -> int:
        """
        Load activation and/or text lenses for specified concepts.

        Args:
            concept_keys: List of (sumo_term, layer) tuples to load
            concept_metadata: Dict of concept metadata
            cache_manager: Cache manager for storing loaded lenses
            reason: Reason for loading (for logging)

        Returns:
            Number of lenses loaded from disk (cache misses)
        """
        keys_to_load_activation = []
        keys_to_load_text = []
        warm_cache_hits = 0

        for concept_key in concept_keys:
            if self.use_activation_lenses:
                # Check if already in active set
                if concept_key in cache_manager.loaded_activation_lenses:
                    cache_manager.stats['cache_hits'] += 1
                    continue

                # Check warm cache
                lens = cache_manager.check_warm_cache(concept_key)
                if lens is not None:
                    warm_cache_hits += 1
                    continue

                # Check tepid cache
                state_dict = cache_manager.check_tepid_cache(concept_key)
                if state_dict is not None:
                    # Transfer to GPU and create lens
                    state_dict_gpu = {k: v.to(self.device) for k, v in state_dict.items()}
                    has_ln = detect_layer_norm(state_dict_gpu)
                    if has_ln:
                        lens = create_lens_from_state_dict(state_dict_gpu, cache_manager.hidden_dim, self.device)
                    else:
                        lens = cache_manager.get_model_from_pool()
                        if lens is None:
                            lens = SimpleMLP(cache_manager.hidden_dim).to(self.device)
                            lens.eval()
                        lens.load_state_dict(state_dict_gpu)

                    cache_manager.add_to_active(concept_key, lens)
                    continue

                # Need to load from disk
                keys_to_load_activation.append(concept_key)

            if self.use_text_lenses and concept_key not in cache_manager.loaded_text_lenses:
                keys_to_load_text.append(concept_key)

        cache_manager._last_warm_cache_hits = warm_cache_hits

        if not keys_to_load_activation and not keys_to_load_text:
            return 0

        # Load activation lenses
        loaded_count = 0
        if self.use_activation_lenses and keys_to_load_activation:
            loaded_count = self._load_activation_lenses(
                keys_to_load_activation,
                concept_metadata,
                cache_manager,
            )

        # Load text lenses
        if self.use_text_lenses and keys_to_load_text:
            self._load_text_lenses(keys_to_load_text, concept_metadata, cache_manager)

        return loaded_count

    def _load_activation_lenses(
        self,
        keys_to_load: List[Tuple[str, int]],
        concept_metadata: Dict[Tuple[str, int], ConceptMetadata],
        cache_manager: "LensCacheManager",
    ) -> int:
        """Load activation lenses from disk."""
        # Infer hidden dim if needed
        if cache_manager.hidden_dim is None:
            for key in keys_to_load:
                metadata = concept_metadata.get(key)
                if metadata and metadata.activation_lens_path and metadata.activation_lens_path.exists():
                    state_dict = torch.load(metadata.activation_lens_path, map_location='cpu')
                    for param_key, param in state_dict.items():
                        if 'weight' in param_key and len(param.shape) == 2:
                            cache_manager.set_hidden_dim(param.shape[1])
                            print(f"  Inferred hidden_dim: {cache_manager.hidden_dim}")
                            break
                    if cache_manager.hidden_dim is not None:
                        break

        # Batch load state dicts in parallel
        def load_lens_state_dict(concept_key):
            metadata = concept_metadata.get(concept_key)
            if not metadata or not metadata.activation_lens_path:
                return None, concept_key

            state_dict = torch.load(
                metadata.activation_lens_path,
                map_location=self.device,
                weights_only=True
            )

            # Handle key mismatch
            if cache_manager.model_pool:
                model_keys_ref = set(cache_manager.model_pool[0].state_dict().keys())
                loaded_keys = set(state_dict.keys())
                if model_keys_ref != loaded_keys:
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if not key.startswith('net.'):
                            new_state_dict[f'net.{key}'] = value
                        else:
                            new_state_dict[key] = value
                    state_dict = new_state_dict

            return state_dict, concept_key

        # Parallel loading
        state_dicts = []
        valid_keys = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(load_lens_state_dict, keys_to_load))

        for state_dict, concept_key in results:
            if state_dict is not None:
                state_dicts.append(state_dict)
                valid_keys.append(concept_key)

        # Create lenses from state dicts
        for concept_key, state_dict in zip(valid_keys, state_dicts):
            has_ln = detect_layer_norm(state_dict)

            if has_ln:
                lens = create_lens_from_state_dict(state_dict, cache_manager.hidden_dim, self.device)
            else:
                lens = cache_manager.get_model_from_pool()
                if lens is None:
                    lens = SimpleMLP(cache_manager.hidden_dim).to(self.device)
                    lens.eval()
                lens.load_state_dict(state_dict)

            cache_manager.add_to_active(concept_key, lens)
            cache_manager.stats['cache_misses'] += 1

        return len(valid_keys)

    def _load_text_lenses(
        self,
        keys_to_load: List[Tuple[str, int]],
        concept_metadata: Dict[Tuple[str, int], ConceptMetadata],
        cache_manager: "LensCacheManager",
    ):
        """Load text lenses (centroids or TF-IDF)."""
        for concept_key in keys_to_load:
            metadata = concept_metadata.get(concept_key)
            if not metadata or not metadata.text_lens_path:
                continue

            try:
                if metadata.text_lens_path.suffix == '.npy':
                    # Centroid-based approach
                    from .centroid_detector import CentroidTextDetector
                    text_lens = CentroidTextDetector.load(
                        metadata.text_lens_path,
                        concept_name=concept_key[0]
                    )
                else:
                    # Legacy TF-IDF joblib approach
                    import joblib
                    text_lens = joblib.load(metadata.text_lens_path)

                cache_manager.loaded_text_lenses[concept_key] = text_lens
                cache_manager.stats['total_loads'] += 1
            except Exception as e:
                print(f"  Failed to load text lens for {concept_key[0]}: {e}")


class MetadataLoader:
    """
    Loads concept metadata from layer JSON files.

    Handles:
    - Deduplication across layers
    - Lens path discovery
    - Hierarchy loading
    """

    def __init__(
        self,
        layers_data_dir: Path,
        lenses_dir: Path,
        using_lens_pack: bool = False,
        activation_lenses_dir: Optional[Path] = None,
        text_lenses_dir: Optional[Path] = None,
    ):
        self.layers_data_dir = layers_data_dir
        self.lenses_dir = lenses_dir
        self.using_lens_pack = using_lens_pack
        self.activation_lenses_dir = activation_lenses_dir or lenses_dir / "activation_lenses"
        self.text_lenses_dir = text_lenses_dir or lenses_dir / "text_lenses"

    def load_all_metadata(self) -> Dict[Tuple[str, int], ConceptMetadata]:
        """
        Load lightweight metadata for all concepts.

        Returns:
            Dict of (sumo_term, layer) -> ConceptMetadata
        """
        layer_files = sorted(self.layers_data_dir.glob("layer*.json"))

        # First pass: collect concepts and find best layer for each
        concept_to_best_layer = {}

        for layer_file in layer_files:
            layer = int(layer_file.stem.replace('layer', ''))

            with open(layer_file) as f:
                layer_data = json.load(f)

            for concept in layer_data['concepts']:
                sumo_term = concept['sumo_term']
                is_category = concept.get('is_category_lens', False)
                synset_count = len(concept.get('synsets', []))

                if sumo_term in concept_to_best_layer:
                    existing_layer, existing_concept = concept_to_best_layer[sumo_term]
                    existing_is_category = existing_concept.get('is_category_lens', False)
                    existing_synset_count = len(existing_concept.get('synsets', []))

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

        # Second pass: create metadata with lens paths
        concept_metadata = {}
        total_concepts = 0
        skipped_no_lens = 0

        for sumo_term, (layer, concept) in concept_to_best_layer.items():
            # Check if lens exists
            has_lens, activation_path = self._find_lens_path(sumo_term, layer)

            if not has_lens:
                skipped_no_lens += 1
                continue

            metadata = ConceptMetadata(
                sumo_term=sumo_term,
                layer=layer,
                category_children=concept.get('category_children', []),
                parent_concepts=concept.get('parent_concepts', []),
                synset_count=concept.get('synset_count', 0),
                sumo_depth=concept.get('sumo_depth', 0),
            )

            # Set lens paths
            if activation_path:
                metadata.activation_lens_path = activation_path
                metadata.has_activation_lens = True

            # Find text lens path
            text_lens_path = self._find_text_lens_path(sumo_term, layer)
            if text_lens_path:
                metadata.text_lens_path = text_lens_path
                metadata.has_text_lens = True

            concept_metadata[(sumo_term, layer)] = metadata
            total_concepts += 1

        print(f"\n✓ Loaded metadata for {total_concepts} concepts across {len(layer_files)} layers")
        if skipped_no_lens > 0:
            print(f"  Skipped {skipped_no_lens} concepts without lenses")

        return concept_metadata

    def _find_lens_path(self, sumo_term: str, layer: int) -> Tuple[bool, Optional[Path]]:
        """Find activation lens path for a concept."""
        if self.using_lens_pack:
            # Try layer-based structure
            layer_dir = self.lenses_dir / f"layer{layer}"
            layer_based_path = layer_dir / f"{sumo_term}.pt"
            if not layer_based_path.exists():
                layer_based_path = layer_dir / f"{sumo_term}_classifier.pt"
            if layer_based_path.exists():
                return True, layer_based_path

            # Fall back to flat structure
            flat_path = self.activation_lenses_dir / f"{sumo_term}.pt"
            if not flat_path.exists():
                flat_path = self.activation_lenses_dir / f"{sumo_term}_classifier.pt"
            if flat_path.exists():
                return True, flat_path
        else:
            layer_dir = self.lenses_dir / f"layer{layer}"
            activation_path = layer_dir / f"{sumo_term}.pt"
            if not activation_path.exists():
                activation_path = layer_dir / f"{sumo_term}_classifier.pt"
            if activation_path.exists():
                return True, activation_path

        return False, None

    def _find_text_lens_path(self, sumo_term: str, layer: int) -> Optional[Path]:
        """Find text lens path for a concept."""
        if self.using_lens_pack:
            layer_dir = self.lenses_dir / f"layer{layer}"
            text_lens_path = layer_dir / f"{sumo_term}_centroid.npy"
            if not text_lens_path.exists():
                text_lens_path = self.text_lenses_dir / f"{sumo_term}_centroid.npy"
            if text_lens_path.exists():
                return text_lens_path
        else:
            layer_dir = self.lenses_dir / f"layer{layer}"
            centroid_dir = layer_dir / "embedding_centroids"
            centroid_path = centroid_dir / f"{sumo_term}_centroid.npy"
            if centroid_path.exists():
                return centroid_path

            # Legacy TF-IDF
            text_lens_dir = layer_dir / "text_lenses"
            text_lens_path = text_lens_dir / f"{sumo_term}_text_lens.joblib"
            if text_lens_path.exists():
                return text_lens_path

        return None


__all__ = ["LensLoader", "MetadataLoader"]
