"""
Lens Pool - Shared lens manager instances for Inspect integration.

Critical for performance: lens manager initialization takes ~60s.
This pool ensures we reuse managers across samples/tasks.
"""

from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple, Any
import weakref

import torch


class LensPool:
    """
    Singleton pool for DynamicLensManager instances.

    Ensures lens managers are reused across Inspect samples,
    avoiding the 60s+ initialization cost per sample.
    """

    _instance: Optional["LensPool"] = None
    _lock = Lock()

    def __new__(cls) -> "LensPool":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._managers: Dict[str, Any] = {}
                    cls._instance._manager_lock = Lock()
        return cls._instance

    def _make_key(
        self,
        lens_pack: str,
        device: str,
        hidden_dim: int,
    ) -> str:
        """Create cache key for lens manager lookup."""
        return f"{lens_pack}:{device}:{hidden_dim}"

    def get_manager(
        self,
        lens_pack: str,
        device: str = "cuda",
        hidden_dim: int = 2048,
        load_threshold: float = 0.5,
        unload_threshold: float = 0.1,
        max_loaded_lenses: int = 500,
        normalize_hidden_states: bool = True,
    ) -> Any:
        """
        Get or create a lens manager for the given configuration.

        Args:
            lens_pack: Path to lens pack directory
            device: Device for lens inference
            hidden_dim: Model hidden dimension (for lens compatibility check)
            load_threshold: Confidence threshold to load child lenses
            unload_threshold: Confidence threshold to unload lenses
            max_loaded_lenses: Maximum lenses to keep loaded
            normalize_hidden_states: Whether to normalize inputs

        Returns:
            DynamicLensManager instance (cached or newly created)
        """
        key = self._make_key(lens_pack, device, hidden_dim)

        with self._manager_lock:
            if key in self._managers:
                return self._managers[key]

            # Import here to avoid circular imports
            from src.hat.monitoring.lens_manager import DynamicLensManager

            lens_pack_path = Path(lens_pack)

            # Determine lens pack structure
            if (lens_pack_path / "activation_lenses").exists():
                lenses_dir = lens_pack_path / "activation_lenses"
            elif list(lens_pack_path.glob("layer*")):
                lenses_dir = lens_pack_path
            else:
                lenses_dir = lens_pack_path

            # Check for hierarchy data
            layers_data_dir = None
            if (lens_pack_path / "layers").exists():
                layers_data_dir = lens_pack_path / "layers"
            elif Path("data/concept_graph/abstraction_layers").exists():
                layers_data_dir = Path("data/concept_graph/abstraction_layers")

            manager = DynamicLensManager(
                layers_data_dir=layers_data_dir,
                lenses_dir=lenses_dir,
                device=device,
                load_threshold=load_threshold,
                unload_threshold=unload_threshold,
                max_loaded_lenses=max_loaded_lenses,
                normalize_hidden_states=normalize_hidden_states,
            )

            self._managers[key] = manager
            return manager

    def clear(self):
        """Clear all cached managers (for testing/cleanup)."""
        with self._manager_lock:
            self._managers.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._manager_lock:
            return {
                "num_managers": len(self._managers),
                "keys": list(self._managers.keys()),
            }


# Global pool instance
_lens_pool: Optional[LensPool] = None


def get_lens_pool() -> LensPool:
    """Get the global lens pool instance."""
    global _lens_pool
    if _lens_pool is None:
        _lens_pool = LensPool()
    return _lens_pool


class HushControllerPool:
    """
    Pool for HushController instances paired with lens managers.

    Separate from LensPool because HushController needs additional
    configuration (CSH profiles, etc.) that may vary per task.
    """

    _instance: Optional["HushControllerPool"] = None
    _lock = Lock()

    def __new__(cls) -> "HushControllerPool":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._controllers: Dict[str, Any] = {}
                    cls._instance._controller_lock = Lock()
        return cls._instance

    def _make_key(
        self,
        lens_pack: str,
        csh_profile: Optional[str],
    ) -> str:
        """Create cache key for controller lookup."""
        return f"{lens_pack}:{csh_profile or 'default'}"

    def get_controller(
        self,
        lens_manager: Any,
        csh_profile: Optional[str] = None,
        lens_pack: str = "",
    ) -> Any:
        """
        Get or create a HushController for the given configuration.

        Args:
            lens_manager: DynamicLensManager to pair with
            csh_profile: Optional path to CSH profile YAML
            lens_pack: Lens pack identifier for caching

        Returns:
            HushController instance
        """
        key = self._make_key(lens_pack, csh_profile)

        with self._controller_lock:
            if key in self._controllers:
                return self._controllers[key]

            # Import here to avoid circular imports
            from src.hush.hush_controller import HushController, SafetyHarnessProfile

            # Load CSH profile if provided
            profile = None
            if csh_profile:
                profile = SafetyHarnessProfile.from_yaml(csh_profile)

            controller = HushController(
                lens_manager=lens_manager,
                csh_profile=profile,
            )

            self._controllers[key] = controller
            return controller

    def clear(self):
        """Clear all cached controllers."""
        with self._controller_lock:
            self._controllers.clear()


_hush_pool: Optional[HushControllerPool] = None


def get_hush_pool() -> HushControllerPool:
    """Get the global HushController pool instance."""
    global _hush_pool
    if _hush_pool is None:
        _hush_pool = HushControllerPool()
    return _hush_pool
