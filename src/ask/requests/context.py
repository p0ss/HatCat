"""
Deployment context registry for ASK audit entries.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading


@dataclass
class DeploymentContext:
    """
    Registered once per deployment, referenced by ID in audit entries.

    Contains regulatory identification, model provenance, and configuration
    that applies to all requests within a deployment.
    """

    deployment_id: str
    schema_version: str = "ftw.deployment.v0.1"

    # Regulatory identification
    system_id: str = ""  # e.g., "annexIII.recruitment.screening"
    provider: str = ""  # Organization operating the system
    deployer: str = ""  # Specific deployment instance
    jurisdiction: str = ""  # e.g., "AU", "EU", "US-CA"
    use_case: str = ""  # e.g., "candidate_shortlisting"

    # Model provenance
    model_id: str = ""  # e.g., "google/gemma-3-4b-pt"
    model_version: str = ""  # Commit hash or version tag
    runtime_version: str = ""  # e.g., "hatcat.v0.3"
    weights_hash: str = ""  # SHA256 of model weights

    # Configuration
    lens_pack_ids: List[str] = field(default_factory=list)
    policy_profiles: List[str] = field(default_factory=list)  # USH/CSH profile IDs

    # Timestamps
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "schema_version": self.schema_version,
            "deployment_id": self.deployment_id,
            "system_id": self.system_id,
            "provider": self.provider,
            "deployer": self.deployer,
            "jurisdiction": self.jurisdiction,
            "use_case": self.use_case,
            "model": {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "runtime_version": self.runtime_version,
                "weights_hash": self.weights_hash,
            },
            "lens_pack_ids": self.lens_pack_ids,
            "policy_profiles": self.policy_profiles,
            "registered_at": self.registered_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentContext":
        """Deserialize from dictionary."""
        model = data.get("model", {})
        registered_at = data.get("registered_at")
        if isinstance(registered_at, str):
            registered_at = datetime.fromisoformat(registered_at.replace("Z", "+00:00"))

        return cls(
            deployment_id=data["deployment_id"],
            schema_version=data.get("schema_version", "ftw.deployment.v0.1"),
            system_id=data.get("system_id", ""),
            provider=data.get("provider", ""),
            deployer=data.get("deployer", ""),
            jurisdiction=data.get("jurisdiction", ""),
            use_case=data.get("use_case", ""),
            model_id=model.get("model_id", data.get("model_id", "")),
            model_version=model.get("model_version", data.get("model_version", "")),
            runtime_version=model.get("runtime_version", data.get("runtime_version", "")),
            weights_hash=model.get("weights_hash", data.get("weights_hash", "")),
            lens_pack_ids=data.get("lens_pack_ids", []),
            policy_profiles=data.get("policy_profiles", []),
            registered_at=registered_at or datetime.now(timezone.utc),
        )


class DeploymentContextRegistry:
    """
    Registry for deployment contexts.

    Manages registration and lookup of deployment contexts.
    Thread-safe for concurrent access.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the registry.

        Args:
            storage_path: Optional path for persistent storage of contexts
        """
        self._contexts: Dict[str, DeploymentContext] = {}
        self._lock = threading.RLock()
        self._storage_path = Path(storage_path) if storage_path else None

        if self._storage_path and self._storage_path.exists():
            self._load_contexts()

    def _load_contexts(self) -> None:
        """Load contexts from persistent storage."""
        if not self._storage_path:
            return

        contexts_file = self._storage_path / "deployment_contexts.json"
        if contexts_file.exists():
            with open(contexts_file, "r") as f:
                data = json.load(f)
                for ctx_data in data.get("contexts", []):
                    ctx = DeploymentContext.from_dict(ctx_data)
                    self._contexts[ctx.deployment_id] = ctx

    def _save_contexts(self) -> None:
        """Save contexts to persistent storage."""
        if not self._storage_path:
            return

        self._storage_path.mkdir(parents=True, exist_ok=True)
        contexts_file = self._storage_path / "deployment_contexts.json"

        data = {
            "contexts": [ctx.to_dict() for ctx in self._contexts.values()]
        }
        with open(contexts_file, "w") as f:
            json.dump(data, f, indent=2)

    def register(self, context: DeploymentContext) -> DeploymentContext:
        """
        Register a deployment context.

        Args:
            context: DeploymentContext to register

        Returns:
            The registered context
        """
        with self._lock:
            self._contexts[context.deployment_id] = context
            self._save_contexts()
            return context

    def get(self, deployment_id: str) -> Optional[DeploymentContext]:
        """
        Get a deployment context by ID.

        Args:
            deployment_id: ID of the context to retrieve

        Returns:
            DeploymentContext if found, None otherwise
        """
        with self._lock:
            return self._contexts.get(deployment_id)

    def get_or_create(
        self,
        deployment_id: str,
        **kwargs,
    ) -> DeploymentContext:
        """
        Get existing context or create a new one.

        Args:
            deployment_id: ID of the context
            **kwargs: Additional fields for new context

        Returns:
            Existing or newly created DeploymentContext
        """
        with self._lock:
            existing = self._contexts.get(deployment_id)
            if existing:
                return existing

            context = DeploymentContext(deployment_id=deployment_id, **kwargs)
            return self.register(context)

    def list_all(self) -> List[DeploymentContext]:
        """List all registered contexts."""
        with self._lock:
            return list(self._contexts.values())

    def remove(self, deployment_id: str) -> bool:
        """
        Remove a deployment context.

        Args:
            deployment_id: ID of the context to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if deployment_id in self._contexts:
                del self._contexts[deployment_id]
                self._save_contexts()
                return True
            return False


# Global registry instance (can be overridden for testing)
_global_registry: Optional[DeploymentContextRegistry] = None


def get_registry(storage_path: Optional[Path] = None) -> DeploymentContextRegistry:
    """
    Get the global deployment context registry.

    Args:
        storage_path: Optional path for persistent storage (only used on first call)

    Returns:
        The global DeploymentContextRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = DeploymentContextRegistry(storage_path)
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (primarily for testing)."""
    global _global_registry
    _global_registry = None
