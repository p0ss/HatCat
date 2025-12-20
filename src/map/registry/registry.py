"""
MAP Pack Registry

Manages local pack directories and synchronization with HuggingFace Hub.
Tracks which packs are synced from remote vs locally created/modified.
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Literal

# HuggingFace Hub - optional dependency
try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


REGISTRY_SCHEMA_VERSION = "1.0"
DEFAULT_HF_ORG = "HatCatFTW"


@dataclass
class PackInfo:
    """Information about an installed pack."""
    name: str
    source: str  # "local" or "hf://org/repo"
    version: str
    revision: Optional[str] = None  # Git commit hash for HF packs
    synced_at: Optional[str] = None
    created_at: Optional[str] = None
    modified: bool = False
    based_on: Optional[str] = None  # For local packs derived from remote
    size_bytes: Optional[int] = None
    # For lens packs: track which layers are installed
    layers_installed: Optional[List[int]] = None
    layers_available: Optional[List[int]] = None  # All layers known to exist remotely

    def to_dict(self) -> Dict:
        d = {
            "source": self.source,
            "version": self.version,
            "revision": self.revision,
            "synced_at": self.synced_at,
            "created_at": self.created_at,
            "modified": self.modified,
            "based_on": self.based_on,
            "size_bytes": self.size_bytes,
        }
        if self.layers_installed is not None:
            d["layers_installed"] = self.layers_installed
        if self.layers_available is not None:
            d["layers_available"] = self.layers_available
        return d

    @classmethod
    def from_dict(cls, name: str, data: Dict) -> "PackInfo":
        return cls(
            name=name,
            source=data.get("source", "local"),
            version=data.get("version", "unknown"),
            revision=data.get("revision"),
            synced_at=data.get("synced_at"),
            created_at=data.get("created_at"),
            modified=data.get("modified", False),
            based_on=data.get("based_on"),
            size_bytes=data.get("size_bytes"),
            layers_installed=data.get("layers_installed"),
            layers_available=data.get("layers_available"),
        )

    @property
    def is_remote(self) -> bool:
        return self.source.startswith("hf://")

    @property
    def is_local(self) -> bool:
        return self.source == "local"


@dataclass
class RegistryState:
    """State of a pack registry (.registry.json)."""
    schema_version: str = REGISTRY_SCHEMA_VERSION
    packs: Dict[str, PackInfo] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "schema_version": self.schema_version,
            "packs": {name: info.to_dict() for name, info in self.packs.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RegistryState":
        packs = {}
        for name, info in data.get("packs", {}).items():
            packs[name] = PackInfo.from_dict(name, info)
        return cls(
            schema_version=data.get("schema_version", REGISTRY_SCHEMA_VERSION),
            packs=packs
        )

    @classmethod
    def load(cls, path: Path) -> "RegistryState":
        if path.exists():
            with open(path) as f:
                return cls.from_dict(json.load(f))
        return cls()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class PackRegistry:
    """
    Registry for managing concept packs and lens packs.

    Handles:
    - Local pack discovery and loading
    - HuggingFace Hub synchronization (pull/push)
    - Tracking synced vs local packs
    - Version management and updates
    """

    def __init__(
        self,
        concept_packs_dir: Optional[Path] = None,
        lens_packs_dir: Optional[Path] = None,
        hf_org: str = DEFAULT_HF_ORG,
    ):
        # Find project root
        project_root = Path(__file__).parent.parent.parent

        self.concept_packs_dir = Path(concept_packs_dir or project_root / "concept_packs")
        self.lens_packs_dir = Path(lens_packs_dir or project_root / "lens_packs")
        self.hf_org = hf_org

        # Ensure directories exist
        self.concept_packs_dir.mkdir(parents=True, exist_ok=True)
        self.lens_packs_dir.mkdir(parents=True, exist_ok=True)

        # Load registry states
        self._concept_registry = RegistryState.load(
            self.concept_packs_dir / ".registry.json"
        )
        self._lens_registry = RegistryState.load(
            self.lens_packs_dir / ".registry.json"
        )

        # HF API (lazy init)
        self._hf_api = None

    @property
    def hf_api(self) -> "HfApi":
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required for remote sync. "
                "Install with: pip install huggingface_hub"
            )
        if self._hf_api is None:
            self._hf_api = HfApi()
        return self._hf_api

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def discover_packs(self, pack_type: Literal["concept", "lens"] = "concept"):
        """Scan pack directory and update registry with any untracked packs."""
        if pack_type == "concept":
            packs_dir = self.concept_packs_dir
            registry_state = self._concept_registry
        else:
            packs_dir = self.lens_packs_dir
            registry_state = self._lens_registry

        for pack_dir in packs_dir.iterdir():
            if not pack_dir.is_dir() or pack_dir.name.startswith("."):
                continue

            pack_json_path = pack_dir / "pack.json"
            if not pack_json_path.exists():
                # Check for pack_info.json (lens packs use this)
                pack_json_path = pack_dir / "pack_info.json"
                if not pack_json_path.exists():
                    continue

            pack_name = pack_dir.name

            # Already tracked?
            if pack_name in registry_state.packs:
                continue

            # Load pack metadata
            try:
                with open(pack_json_path) as f:
                    pack_json = json.load(f)

                version = pack_json.get("version") or pack_json.get("pack_version", "unknown")

                # Calculate size
                size = sum(
                    f.stat().st_size
                    for f in pack_dir.rglob("*")
                    if f.is_file()
                )

                # Add as local pack
                registry_state.packs[pack_name] = PackInfo(
                    name=pack_name,
                    source="local",
                    version=version,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    size_bytes=size,
                )
                print(f"Discovered local {pack_type} pack: {pack_name} v{version}")

            except Exception as e:
                print(f"Error scanning pack {pack_dir}: {e}")

        # Save updated registry
        registry_path = packs_dir / ".registry.json"
        registry_state.save(registry_path)

    def discover_all(self):
        """Discover all untracked packs."""
        self.discover_packs("concept")
        self.discover_packs("lens")

    # -------------------------------------------------------------------------
    # Listing
    # -------------------------------------------------------------------------

    def list_concept_packs(self) -> List[PackInfo]:
        """List all installed concept packs."""
        self.discover_packs("concept")
        return list(self._concept_registry.packs.values())

    def list_lens_packs(self) -> List[PackInfo]:
        """List all installed lens packs."""
        self.discover_packs("lens")
        return list(self._lens_registry.packs.values())

    def status(self) -> Dict:
        """Get status of all packs (installed, outdated, modified)."""
        self.discover_all()

        return {
            "concept_packs": [
                {
                    "name": p.name,
                    "version": p.version,
                    "source": p.source,
                    "modified": p.modified,
                    "synced_at": p.synced_at,
                }
                for p in self._concept_registry.packs.values()
            ],
            "lens_packs": [
                {
                    "name": p.name,
                    "version": p.version,
                    "source": p.source,
                    "modified": p.modified,
                    "synced_at": p.synced_at,
                }
                for p in self._lens_registry.packs.values()
            ],
        }

    # -------------------------------------------------------------------------
    # Pull from HuggingFace
    # -------------------------------------------------------------------------

    def pull_concept_pack(
        self,
        name: str,
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        """
        Pull a concept pack from HuggingFace Hub.

        Args:
            name: Pack name (used as local directory name)
            repo_id: HF repo ID (default: {hf_org}/concept-pack-{name})
            revision: Specific git revision/tag/branch
            force: Overwrite existing pack

        Returns:
            Path to the downloaded pack directory
        """
        return self._pull_pack(
            pack_type="concept",
            name=name,
            repo_id=repo_id or f"{self.hf_org}/concept-pack-{name}",
            revision=revision,
            force=force,
        )

    def pull_lens_pack(
        self,
        name: str,
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        """
        Pull a lens pack from HuggingFace Hub.

        Args:
            name: Pack name (used as local directory name)
            repo_id: HF repo ID (default: {hf_org}/lens-{name})
            revision: Specific git revision/tag/branch
            force: Overwrite existing pack

        Returns:
            Path to the downloaded pack directory
        """
        return self._pull_pack(
            pack_type="lens",
            name=name,
            repo_id=repo_id or f"{self.hf_org}/lens-{name}",
            revision=revision,
            force=force,
        )

    def _pull_pack(
        self,
        pack_type: Literal["concept", "lens"],
        name: str,
        repo_id: str,
        revision: Optional[str],
        force: bool,
    ) -> Path:
        """Internal method to pull a pack from HF."""
        if pack_type == "concept":
            packs_dir = self.concept_packs_dir
            registry_state = self._concept_registry
        else:
            packs_dir = self.lens_packs_dir
            registry_state = self._lens_registry

        pack_dir = packs_dir / name

        # Check if already exists
        if pack_dir.exists() and not force:
            existing = registry_state.packs.get(name)
            if existing and existing.modified:
                raise ValueError(
                    f"Pack {name} has local modifications. "
                    "Use force=True to overwrite."
                )
            print(f"Pack {name} already exists. Use force=True to re-download.")
            return pack_dir

        # Download from HF
        print(f"Pulling {pack_type} pack from {repo_id}...")

        try:
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=pack_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download {repo_id}: {e}")

        # Get revision info
        try:
            repo_info = self.hf_api.repo_info(repo_id, revision=revision)
            actual_revision = repo_info.sha
        except Exception:
            actual_revision = revision

        # Load version from pack.json
        pack_json_path = pack_dir / "pack.json"
        if not pack_json_path.exists():
            pack_json_path = pack_dir / "pack_info.json"

        version = "unknown"
        if pack_json_path.exists():
            with open(pack_json_path) as f:
                pack_json = json.load(f)
                version = pack_json.get("version") or pack_json.get("pack_version", "unknown")

        # Calculate size
        size = sum(
            f.stat().st_size
            for f in pack_dir.rglob("*")
            if f.is_file()
        )

        # Update registry
        registry_state.packs[name] = PackInfo(
            name=name,
            source=f"hf://{repo_id}",
            version=version,
            revision=actual_revision,
            synced_at=datetime.now(timezone.utc).isoformat(),
            modified=False,
            size_bytes=size,
        )
        registry_state.save(packs_dir / ".registry.json")

        print(f"✓ Pulled {name} v{version} ({size / 1024 / 1024:.1f} MB)")
        return pack_dir

    # -------------------------------------------------------------------------
    # Push to HuggingFace
    # -------------------------------------------------------------------------

    def push_lens_pack(
        self,
        name: str,
        repo_id: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Push a lens pack to HuggingFace Hub.

        Args:
            name: Local pack name
            repo_id: HF repo ID (default: {hf_org}/lens-{name})
            private: Create as private repo
            commit_message: Commit message

        Returns:
            URL of the uploaded repo
        """
        return self._push_pack(
            pack_type="lens",
            name=name,
            repo_id=repo_id or f"{self.hf_org}/lens-{name}",
            private=private,
            commit_message=commit_message,
        )

    def push_concept_pack(
        self,
        name: str,
        repo_id: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Push a concept pack to HuggingFace Hub.

        Args:
            name: Local pack name
            repo_id: HF repo ID (default: {hf_org}/concept-pack-{name})
            private: Create as private repo
            commit_message: Commit message

        Returns:
            URL of the uploaded repo
        """
        return self._push_pack(
            pack_type="concept",
            name=name,
            repo_id=repo_id or f"{self.hf_org}/concept-pack-{name}",
            private=private,
            commit_message=commit_message,
        )

    def _push_pack(
        self,
        pack_type: Literal["concept", "lens"],
        name: str,
        repo_id: str,
        private: bool,
        commit_message: Optional[str],
    ) -> str:
        """Internal method to push a pack to HF."""
        if pack_type == "concept":
            packs_dir = self.concept_packs_dir
            registry_state = self._concept_registry
        else:
            packs_dir = self.lens_packs_dir
            registry_state = self._lens_registry

        pack_dir = packs_dir / name
        if not pack_dir.exists():
            raise ValueError(f"Pack {name} not found at {pack_dir}")

        # Load version
        pack_json_path = pack_dir / "pack.json"
        if not pack_json_path.exists():
            pack_json_path = pack_dir / "pack_info.json"

        version = "unknown"
        if pack_json_path.exists():
            with open(pack_json_path) as f:
                pack_json = json.load(f)
                version = pack_json.get("version") or pack_json.get("pack_version", "unknown")

        # Create repo if needed
        try:
            self.hf_api.create_repo(
                repo_id=repo_id,
                repo_type="model",  # Using model type for now
                private=private,
                exist_ok=True,
            )
        except Exception as e:
            print(f"Note: {e}")

        # Upload
        print(f"Pushing {name} to {repo_id}...")

        commit_message = commit_message or f"Upload {pack_type} pack {name} v{version}"

        # Patterns to exclude from upload
        ignore_patterns = [
            "logs/*",
            "*.log",
            "__pycache__/*",
            ".DS_Store",
            "*.pyc",
        ]

        self.hf_api.upload_folder(
            folder_path=pack_dir,
            repo_id=repo_id,
            commit_message=commit_message,
            ignore_patterns=ignore_patterns,
        )

        # Update registry
        registry_state.packs[name] = PackInfo(
            name=name,
            source=f"hf://{repo_id}",
            version=version,
            synced_at=datetime.now(timezone.utc).isoformat(),
            modified=False,
        )
        registry_state.save(packs_dir / ".registry.json")

        url = f"https://huggingface.co/{repo_id}"
        print(f"✓ Pushed to {url}")
        return url

    # -------------------------------------------------------------------------
    # Loading with auto-pull
    # -------------------------------------------------------------------------

    def get_concept_pack_path(
        self,
        name: str,
        auto_pull: bool = True,
    ) -> Path:
        """
        Get path to a concept pack, optionally pulling if not present.

        Args:
            name: Pack name
            auto_pull: If True, pull from HF if not installed

        Returns:
            Path to pack directory
        """
        pack_dir = self.concept_packs_dir / name

        if not pack_dir.exists() and auto_pull:
            print(f"Concept pack {name} not found locally, pulling from HF...")
            return self.pull_concept_pack(name)

        if not pack_dir.exists():
            raise ValueError(f"Concept pack {name} not found at {pack_dir}")

        return pack_dir

    def get_lens_pack_path(
        self,
        name: str,
        auto_pull: bool = True,
    ) -> Path:
        """
        Get path to a lens pack, optionally pulling if not present.

        Args:
            name: Pack name
            auto_pull: If True, pull from HF if not installed

        Returns:
            Path to pack directory
        """
        pack_dir = self.lens_packs_dir / name

        if not pack_dir.exists() and auto_pull:
            print(f"Lens pack {name} not found locally, pulling from HF...")
            return self.pull_lens_pack(name)

        if not pack_dir.exists():
            raise ValueError(f"Lens pack {name} not found at {pack_dir}")

        return pack_dir

    # -------------------------------------------------------------------------
    # Mark modified
    # -------------------------------------------------------------------------

    def mark_modified(
        self,
        name: str,
        pack_type: Literal["concept", "lens"],
    ):
        """Mark a pack as locally modified."""
        if pack_type == "concept":
            packs_dir = self.concept_packs_dir
            registry_state = self._concept_registry
        else:
            packs_dir = self.lens_packs_dir
            registry_state = self._lens_registry

        if name in registry_state.packs:
            registry_state.packs[name].modified = True
            registry_state.save(packs_dir / ".registry.json")

    # -------------------------------------------------------------------------
    # Per-layer operations for lens packs
    # -------------------------------------------------------------------------

    def get_installed_layers(self, name: str) -> List[int]:
        """Get list of layers installed locally for a lens pack."""
        pack_dir = self.lens_packs_dir / name
        if not pack_dir.exists():
            return []

        layers = []
        for item in pack_dir.iterdir():
            if item.is_dir() and item.name.startswith("layer"):
                try:
                    layer_num = int(item.name.replace("layer", ""))
                    # Check if layer has actual classifier files
                    pt_files = list(item.glob("*.pt"))
                    if pt_files:
                        layers.append(layer_num)
                except ValueError:
                    continue

        return sorted(layers)

    def list_remote_layers(self, name: str) -> List[int]:
        """List layers available on HuggingFace for a lens pack."""
        layers = []

        # Check for layer-specific repos
        for layer in range(10):  # Check layers 0-9
            repo_id = f"{self.hf_org}/lens-{name}-layer{layer}"
            try:
                self.hf_api.repo_info(repo_id)
                layers.append(layer)
            except Exception:
                continue

        return layers

    def push_lens_layer(
        self,
        name: str,
        layer: int,
        repo_id: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Push a single layer of a lens pack to HuggingFace Hub.

        Args:
            name: Local pack name
            layer: Layer number to push
            repo_id: HF repo ID (default: {hf_org}/lens-{name}-layer{layer})
            private: Create as private repo
            commit_message: Commit message

        Returns:
            URL of the uploaded repo
        """
        pack_dir = self.lens_packs_dir / name
        layer_dir = pack_dir / f"layer{layer}"

        if not layer_dir.exists():
            raise ValueError(f"Layer {layer} not found at {layer_dir}")

        repo_id = repo_id or f"{self.hf_org}/lens-{name}-layer{layer}"

        # Load version from pack metadata
        pack_json_path = pack_dir / "pack_info.json"
        if not pack_json_path.exists():
            pack_json_path = pack_dir / "pack.json"

        version = "unknown"
        if pack_json_path.exists():
            with open(pack_json_path) as f:
                pack_json = json.load(f)
                version = pack_json.get("version") or pack_json.get("pack_version", "unknown")

        # Create repo if needed
        try:
            self.hf_api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
            )
        except Exception as e:
            print(f"Note: {e}")

        print(f"Pushing layer {layer} of {name} to {repo_id}...")

        commit_message = commit_message or f"Upload lens pack {name} layer {layer} v{version}"

        # Patterns to exclude
        ignore_patterns = [
            "logs/*",
            "*.log",
            "__pycache__/*",
            ".DS_Store",
            "*.pyc",
        ]

        self.hf_api.upload_folder(
            folder_path=layer_dir,
            repo_id=repo_id,
            commit_message=commit_message,
            ignore_patterns=ignore_patterns,
        )

        # Also upload pack metadata to each layer repo for self-description
        if pack_json_path.exists():
            self.hf_api.upload_file(
                path_or_fileobj=str(pack_json_path),
                path_in_repo="pack_info.json",
                repo_id=repo_id,
                commit_message="Add pack metadata",
            )

        url = f"https://huggingface.co/{repo_id}"
        print(f"✓ Pushed layer {layer} to {url}")

        # Update registry
        self._update_lens_registry_layers(name, version)

        return url

    def push_all_lens_layers(
        self,
        name: str,
        private: bool = False,
    ) -> List[str]:
        """
        Push all layers of a lens pack to HuggingFace Hub.

        Args:
            name: Local pack name
            private: Create as private repos

        Returns:
            List of URLs for uploaded layer repos
        """
        layers = self.get_installed_layers(name)
        if not layers:
            raise ValueError(f"No layers found for lens pack {name}")

        urls = []
        for layer in layers:
            url = self.push_lens_layer(name, layer, private=private)
            urls.append(url)

        return urls

    def pull_lens_layer(
        self,
        name: str,
        layer: int,
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        """
        Pull a single layer of a lens pack from HuggingFace Hub.

        Args:
            name: Pack name (used as local directory name)
            layer: Layer number to pull
            repo_id: HF repo ID (default: {hf_org}/lens-{name}-layer{layer})
            revision: Specific git revision/tag/branch
            force: Overwrite existing layer

        Returns:
            Path to the downloaded layer directory
        """
        pack_dir = self.lens_packs_dir / name
        layer_dir = pack_dir / f"layer{layer}"

        repo_id = repo_id or f"{self.hf_org}/lens-{name}-layer{layer}"

        # Check if already exists
        if layer_dir.exists() and not force:
            print(f"Layer {layer} already exists. Use force=True to re-download.")
            return layer_dir

        # Ensure pack directory exists
        pack_dir.mkdir(parents=True, exist_ok=True)

        print(f"Pulling layer {layer} from {repo_id}...")

        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=layer_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download {repo_id}: {e}")

        # Copy pack_info.json to pack root if not present
        layer_pack_info = layer_dir / "pack_info.json"
        pack_pack_info = pack_dir / "pack_info.json"
        if layer_pack_info.exists() and not pack_pack_info.exists():
            shutil.copy(layer_pack_info, pack_pack_info)

        # Update registry
        version = "unknown"
        if pack_pack_info.exists():
            with open(pack_pack_info) as f:
                pack_json = json.load(f)
                version = pack_json.get("version") or pack_json.get("pack_version", "unknown")

        self._update_lens_registry_layers(name, version)

        print(f"✓ Pulled layer {layer}")
        return layer_dir

    def pull_lens_layers(
        self,
        name: str,
        layers: List[int],
        force: bool = False,
    ) -> Path:
        """
        Pull multiple layers of a lens pack.

        Args:
            name: Pack name
            layers: List of layer numbers to pull
            force: Overwrite existing layers

        Returns:
            Path to pack directory
        """
        for layer in layers:
            self.pull_lens_layer(name, layer, force=force)

        return self.lens_packs_dir / name

    def _update_lens_registry_layers(self, name: str, version: str):
        """Update registry with current layer state."""
        installed = self.get_installed_layers(name)

        if name in self._lens_registry.packs:
            self._lens_registry.packs[name].layers_installed = installed
            self._lens_registry.packs[name].version = version
        else:
            self._lens_registry.packs[name] = PackInfo(
                name=name,
                source=f"hf://{self.hf_org}/lens-{name}-layer*",
                version=version,
                synced_at=datetime.now(timezone.utc).isoformat(),
                layers_installed=installed,
            )

        self._lens_registry.save(self.lens_packs_dir / ".registry.json")


# Global singleton instance
_registry: Optional[PackRegistry] = None


def registry() -> PackRegistry:
    """Get the global pack registry instance."""
    global _registry
    if _registry is None:
        _registry = PackRegistry()
    return _registry
