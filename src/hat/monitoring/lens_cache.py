#!/usr/bin/env python3
"""
Lens Cache Management

Manages multi-tier caching for lens management:
- Hot VRAM: BatchedLensBank (active inference)
- Warm VRAM: GPU tensors waiting for parent activation
- Tepid RAM: CPU tensors pre-loaded (fast .to(device))
- Cold Disk: Only when RAM can't fit pack
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from .lens_types import SimpleMLP, ConceptMetadata
from .lens_batched import BatchedLensBank

if TYPE_CHECKING:
    pass


class LensCacheManager:
    """
    Manages lens caching across multiple tiers for optimal performance.

    Tiers:
    1. Hot (loaded_lenses): Active in BatchedLensBank, run every token
    2. Warm (warm_cache): In VRAM but not in bank, fast reactivation
    3. Tepid (tepid_cache): In CPU RAM, avoids torch.load() deserialize
    4. Cold: On disk, requires full load
    """

    def __init__(
        self,
        device: str = "cuda",
        max_loaded_lenses: int = 500,
        keep_top_k: int = 50,
        aggressive_pruning: bool = True,
        model_pool_size: int = 100,
    ):
        self.device = device
        self.max_loaded_lenses = max_loaded_lenses
        self.keep_top_k = keep_top_k
        self.aggressive_pruning = aggressive_pruning
        self.model_pool_size = model_pool_size

        # Hot tier: actively loaded lenses
        self.loaded_activation_lenses: Dict[Tuple[str, int], nn.Module] = {}
        self.loaded_lenses = self.loaded_activation_lenses  # Alias for backward compat
        self.lens_scores: Dict[Tuple[str, int], float] = {}
        self.lens_access_count: Dict[Tuple[str, int], int] = defaultdict(int)

        # Warm tier: VRAM cache for fast reactivation
        # Key: (sumo_term, layer), Value: (lens, reactivation_count)
        self.warm_cache: Dict[Tuple[str, int], Tuple[nn.Module, int]] = {}
        self.cache_reactivation_count: Dict[Tuple[str, int], int] = defaultdict(int)

        # Tepid tier: CPU RAM cache (pre-loaded state_dicts)
        self.tepid_cache: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}
        self._tepid_cache_loaded: bool = False

        # Protected lenses (never evict)
        self.base_layer_lenses: Set[Tuple[str, int]] = set()

        # Model pool for efficient weight swapping
        self.model_pool: List[nn.Module] = []
        self.available_models: List[int] = []
        self.hidden_dim: Optional[int] = None

        # Batched inference bank
        self._lens_bank: Optional[BatchedLensBank] = None
        self._lens_bank_dirty: bool = True
        self._use_batched_inference: bool = True

        # Stats
        self.stats = {
            'total_loads': 0,
            'total_unloads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'tepid_hits': 0,
        }

    def set_hidden_dim(self, hidden_dim: int):
        """Set hidden dimension and initialize model pool."""
        self.hidden_dim = hidden_dim
        self._ensure_model_pool()

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

    def get_model_from_pool(self) -> Optional[nn.Module]:
        """Get an available model from pool, or None if pool exhausted."""
        if self.available_models:
            idx = self.available_models.pop()
            return self.model_pool[idx]
        return None

    def return_model_to_pool(self, model: nn.Module):
        """Return a model to the pool."""
        for i, pool_model in enumerate(self.model_pool):
            if pool_model is model:
                self.available_models.append(i)
                break

    def check_warm_cache(self, concept_key: Tuple[str, int]) -> Optional[nn.Module]:
        """
        Check if concept is in warm cache and promote to active if so.

        Returns:
            Lens if found in warm cache (now promoted to active), None otherwise
        """
        if concept_key in self.warm_cache:
            lens, reactivation_count = self.warm_cache[concept_key]

            # Promote to active
            self.loaded_activation_lenses[concept_key] = lens
            self.loaded_lenses[concept_key] = lens
            self.lens_scores[concept_key] = 0.0

            # Increment reactivation count
            self.cache_reactivation_count[concept_key] = reactivation_count + 1

            # Remove from warm cache
            del self.warm_cache[concept_key]

            self.stats['cache_hits'] += 1
            return lens

        return None

    def check_tepid_cache(self, concept_key: Tuple[str, int]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Check if concept state_dict is in tepid cache.

        Returns:
            State dict if found, None otherwise
        """
        if concept_key in self.tepid_cache:
            self.stats['cache_hits'] += 1
            self.stats['tepid_hits'] += 1
            return self.tepid_cache[concept_key]
        return None

    def add_to_active(self, concept_key: Tuple[str, int], lens: nn.Module, is_base_layer: bool = False):
        """Add a lens to the active set."""
        self.loaded_activation_lenses[concept_key] = lens
        self.loaded_lenses[concept_key] = lens
        self.lens_scores[concept_key] = 0.0
        self.stats['total_loads'] += 1

        if is_base_layer:
            self.base_layer_lenses.add(concept_key)

        self.mark_lens_bank_dirty()

    def move_to_warm_cache(self, concept_keys: List[Tuple[str, int]]):
        """Move lenses from active to warm cache."""
        for concept_key in concept_keys:
            if concept_key in self.base_layer_lenses:
                continue  # Never evict base layers

            if concept_key in self.loaded_activation_lenses:
                lens = self.loaded_activation_lenses[concept_key]

                # Store in warm cache with reactivation count
                reactivation_count = self.cache_reactivation_count.get(concept_key, 0)
                self.warm_cache[concept_key] = (lens, reactivation_count)

                # Remove from active
                del self.loaded_activation_lenses[concept_key]
                if concept_key in self.loaded_lenses:
                    del self.loaded_lenses[concept_key]
                if concept_key in self.lens_scores:
                    del self.lens_scores[concept_key]

        self.mark_lens_bank_dirty()

    def manage_cache_memory(self):
        """
        Manage warm cache size by evicting least-reactivated lenses.

        Strategy:
        - Total memory budget = max_loaded_lenses (for loaded + warm cache combined)
        - When budget exceeded, evict from warm cache
        - Sort by reactivation count (ascending) - least-reactivated get evicted first
        """
        total_in_memory = len(self.loaded_activation_lenses) + len(self.warm_cache)

        if total_in_memory <= self.max_loaded_lenses:
            return

        num_to_evict = total_in_memory - self.max_loaded_lenses

        # Sort warm cache by reactivation count (ascending)
        warm_cache_sorted = sorted(
            self.warm_cache.items(),
            key=lambda x: x[1][1]  # x[1][1] is reactivation_count
        )

        evicted = 0
        for concept_key, (lens, reactivation_count) in warm_cache_sorted:
            if evicted >= num_to_evict:
                break

            if concept_key in self.base_layer_lenses:
                continue

            # Return lens to pool
            self.return_model_to_pool(lens)

            # Remove from warm cache
            del self.warm_cache[concept_key]

            if concept_key in self.cache_reactivation_count:
                del self.cache_reactivation_count[concept_key]

            evicted += 1
            self.stats['total_unloads'] += 1

    def aggressive_prune_to_top_k(self, base_layers: List[int]):
        """
        Aggressively prune to keep only top-K scoring lenses.

        NOTE: Base layer lenses are PROTECTED from pruning.

        Args:
            base_layers: List of layer indices that are protected
        """
        if not self.aggressive_pruning:
            return

        if len(self.loaded_lenses) <= self.keep_top_k:
            return

        # Separate base layer lenses from dynamic lenses
        base_layer_lenses = []
        dynamic_lenses = []

        for k, v in self.lens_scores.items():
            concept_name, layer = k
            if layer in base_layers:
                base_layer_lenses.append((k, v))
            else:
                dynamic_lenses.append((k, v))

        # Sort dynamic lenses by score (highest first)
        dynamic_lenses.sort(key=lambda x: x[1], reverse=True)

        # Keep base layers + top K dynamic lenses
        total_to_keep = len(base_layer_lenses) + self.keep_top_k

        if len(self.loaded_lenses) <= total_to_keep:
            return

        # Unload everything beyond base layers + top K
        to_unload = dynamic_lenses[self.keep_top_k:]

        for key, _ in to_unload:
            if key in self.loaded_lenses:
                self.return_model_to_pool(self.loaded_lenses[key])
                del self.loaded_lenses[key]
                del self.lens_scores[key]
                self.stats['total_unloads'] += 1

        self.mark_lens_bank_dirty()

    def compute_top_k_keys(self, top_k: int) -> Set[Tuple[str, int]]:
        """
        Compute top-k concept keys from current scores.

        Returns:
            Set of concept keys that are in the top-k by score
        """
        sorted_concepts = sorted(self.lens_scores.items(), key=lambda x: x[1], reverse=True)
        return set(key for key, _ in sorted_concepts[:top_k])

    def prune_to_top_k(self, top_k: int, skip_base_layers: bool = True):
        """
        Move non-top-k lenses to warm cache.

        Args:
            top_k: Number of top lenses to keep active
            skip_base_layers: If True, base layer lenses are always kept active
        """
        top_k_keys = self.compute_top_k_keys(top_k)

        to_warm_cache = []
        for concept_key in list(self.loaded_activation_lenses.keys()):
            if skip_base_layers and concept_key in self.base_layer_lenses:
                continue

            if concept_key not in top_k_keys:
                to_warm_cache.append(concept_key)

        if to_warm_cache:
            self.move_to_warm_cache(to_warm_cache)
            self.manage_cache_memory()

    # Lens bank management

    def mark_lens_bank_dirty(self):
        """Mark lens bank as needing rebuild."""
        self._lens_bank_dirty = True

    def rebuild_lens_bank(self):
        """Rebuild the batched lens bank from currently loaded lenses."""
        if not self._use_batched_inference:
            return

        if self._lens_bank is None:
            self._lens_bank = BatchedLensBank(device=self.device)
        else:
            self._lens_bank.clear()

        if self.loaded_lenses:
            self._lens_bank.add_lenses(self.loaded_lenses)

        self._lens_bank_dirty = False

    def get_lens_bank(self) -> Optional[BatchedLensBank]:
        """Get the lens bank, rebuilding if necessary."""
        if self._lens_bank_dirty:
            self.rebuild_lens_bank()
        return self._lens_bank

    def is_bank_compiled(self) -> bool:
        """Check if lens bank is compiled and ready."""
        if self._lens_bank is None:
            return False
        return self._lens_bank.is_compiled

    # Tepid cache preloading

    def preload_to_ram(
        self,
        concept_metadata: Dict[Tuple[str, int], ConceptMetadata],
        max_ram_mb: int = None,
        priority_layers: List[int] = None
    ) -> Dict[str, any]:
        """
        Pre-load lens pack to CPU RAM (tepid cache).

        Args:
            concept_metadata: Dict of concept_key -> ConceptMetadata
            max_ram_mb: Max RAM to use (MB). None = no limit.
            priority_layers: Load these layers first.

        Returns:
            Dict with loading stats
        """
        if self._tepid_cache_loaded:
            return {"status": "already_loaded", "concepts": len(self.tepid_cache)}

        if priority_layers is None:
            priority_layers = [3, 4, 5, 6, 2, 1, 0]

        start = time.time()
        loaded_count = 0
        loaded_bytes = 0
        max_bytes = max_ram_mb * 1024 * 1024 if max_ram_mb else float('inf')

        # Collect concepts by layer
        concepts_by_layer: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
        for concept_key, metadata in concept_metadata.items():
            if metadata.activation_lens_path and metadata.activation_lens_path.exists():
                concepts_by_layer[metadata.layer].append(concept_key)

        # Load in priority order
        for layer in priority_layers:
            if layer not in concepts_by_layer:
                continue

            for concept_key in concepts_by_layer[layer]:
                if loaded_bytes >= max_bytes:
                    break

                if concept_key in self.tepid_cache:
                    continue

                metadata = concept_metadata[concept_key]
                try:
                    state_dict = torch.load(
                        metadata.activation_lens_path,
                        map_location='cpu',
                        weights_only=True
                    )

                    # Handle key mismatch
                    if self.model_pool:
                        model_keys = set(self.model_pool[0].state_dict().keys())
                        loaded_keys = set(state_dict.keys())
                        if model_keys != loaded_keys and not any(k.startswith('net.') for k in loaded_keys):
                            state_dict = {f'net.{k}': v for k, v in state_dict.items()}

                    self.tepid_cache[concept_key] = state_dict
                    loaded_count += 1

                    for v in state_dict.values():
                        loaded_bytes += v.numel() * v.element_size()

                except Exception as e:
                    print(f"  Warning: Failed to preload {concept_key[0]}: {e}")

            if loaded_bytes >= max_bytes:
                print(f"  RAM budget reached at layer {layer}")
                break

        # Load remaining layers
        all_layers = set(concepts_by_layer.keys())
        remaining_layers = sorted(all_layers - set(priority_layers))
        for layer in remaining_layers:
            if loaded_bytes >= max_bytes:
                break
            for concept_key in concepts_by_layer[layer]:
                if loaded_bytes >= max_bytes:
                    break
                if concept_key in self.tepid_cache:
                    continue

                metadata = concept_metadata[concept_key]
                try:
                    state_dict = torch.load(
                        metadata.activation_lens_path,
                        map_location='cpu',
                        weights_only=True
                    )
                    if self.model_pool:
                        model_keys = set(self.model_pool[0].state_dict().keys())
                        loaded_keys = set(state_dict.keys())
                        if model_keys != loaded_keys and not any(k.startswith('net.') for k in loaded_keys):
                            state_dict = {f'net.{k}': v for k, v in state_dict.items()}

                    self.tepid_cache[concept_key] = state_dict
                    loaded_count += 1
                    for v in state_dict.values():
                        loaded_bytes += v.numel() * v.element_size()
                except Exception:
                    pass

        self._tepid_cache_loaded = True
        elapsed = time.time() - start

        return {
            "status": "loaded",
            "concepts": loaded_count,
            "ram_mb": loaded_bytes / (1024 * 1024),
            "elapsed_s": elapsed,
        }

    def reset_to_base(self, keep_warm_cache: bool = True):
        """Reset to only base layer lenses."""
        to_remove = [k for k in self.loaded_lenses.keys() if k not in self.base_layer_lenses]
        for key in to_remove:
            if keep_warm_cache and key in self.loaded_activation_lenses:
                lens = self.loaded_activation_lenses[key]
                reactivation_count = self.cache_reactivation_count.get(key, 0)
                self.warm_cache[key] = (lens, reactivation_count)
            del self.loaded_lenses[key]
            if key in self.loaded_activation_lenses:
                del self.loaded_activation_lenses[key]

        if not keep_warm_cache:
            self.warm_cache.clear()
            self.cache_reactivation_count.clear()

        self.lens_scores.clear()
        self.lens_access_count.clear()
        self.mark_lens_bank_dirty()


__all__ = ["LensCacheManager"]
