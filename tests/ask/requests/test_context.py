"""Tests for DeploymentContext and registry."""

import pytest
import tempfile
from pathlib import Path

from src.ask.requests.context import (
    DeploymentContext,
    DeploymentContextRegistry,
    get_registry,
    reset_registry,
)


class TestDeploymentContext:
    """Tests for DeploymentContext dataclass."""

    def test_create_context(self):
        """Should create context with all fields."""
        ctx = DeploymentContext(
            deployment_id="dep_test_001",
            system_id="annexIII.recruitment",
            provider="TestCorp",
            deployer="au-east-1.prod",
            jurisdiction="AU",
            use_case="candidate_screening",
            model_id="google/gemma-3-4b-pt",
            model_version="v1.0",
            runtime_version="hatcat.v0.3",
            weights_hash="sha256:abc123",
            lens_pack_ids=["first-light"],
            policy_profiles=["org.hatcat/au-recruit-ush@0.1.0"],
        )

        assert ctx.deployment_id == "dep_test_001"
        assert ctx.jurisdiction == "AU"
        assert ctx.registered_at is not None

    def test_to_dict_serializes(self):
        """to_dict() should produce valid dict."""
        ctx = DeploymentContext(
            deployment_id="test",
            jurisdiction="EU",
            model_id="test-model",
        )

        data = ctx.to_dict()

        assert data["deployment_id"] == "test"
        assert data["jurisdiction"] == "EU"
        assert data["model"]["model_id"] == "test-model"
        assert "registered_at" in data

    def test_from_dict_deserializes(self):
        """from_dict() should reconstruct context."""
        original = DeploymentContext(
            deployment_id="test",
            jurisdiction="AU",
            model_id="test-model",
            lens_pack_ids=["pack1", "pack2"],
        )

        data = original.to_dict()
        restored = DeploymentContext.from_dict(data)

        assert restored.deployment_id == original.deployment_id
        assert restored.jurisdiction == original.jurisdiction
        assert restored.lens_pack_ids == original.lens_pack_ids


class TestDeploymentContextRegistry:
    """Tests for DeploymentContextRegistry."""

    def test_register_and_get(self):
        """Should register and retrieve contexts."""
        registry = DeploymentContextRegistry()

        ctx = DeploymentContext(
            deployment_id="test-1",
            jurisdiction="AU",
        )
        registry.register(ctx)

        retrieved = registry.get("test-1")
        assert retrieved is not None
        assert retrieved.deployment_id == "test-1"

    def test_get_nonexistent_returns_none(self):
        """get() should return None for unknown ID."""
        registry = DeploymentContextRegistry()

        result = registry.get("nonexistent")
        assert result is None

    def test_get_or_create_returns_existing(self):
        """get_or_create() should return existing context."""
        registry = DeploymentContextRegistry()

        ctx = DeploymentContext(
            deployment_id="test",
            jurisdiction="EU",
        )
        registry.register(ctx)

        result = registry.get_or_create("test", jurisdiction="AU")

        # Should return existing, not create new with AU
        assert result.jurisdiction == "EU"

    def test_get_or_create_creates_new(self):
        """get_or_create() should create new if not exists."""
        registry = DeploymentContextRegistry()

        result = registry.get_or_create(
            "new-deployment",
            jurisdiction="AU",
            model_id="test-model",
        )

        assert result.deployment_id == "new-deployment"
        assert result.jurisdiction == "AU"

    def test_list_all(self):
        """list_all() should return all contexts."""
        registry = DeploymentContextRegistry()

        registry.register(DeploymentContext(deployment_id="ctx-1"))
        registry.register(DeploymentContext(deployment_id="ctx-2"))

        all_contexts = registry.list_all()
        assert len(all_contexts) == 2

    def test_remove(self):
        """remove() should delete context."""
        registry = DeploymentContextRegistry()

        registry.register(DeploymentContext(deployment_id="to-remove"))
        assert registry.get("to-remove") is not None

        result = registry.remove("to-remove")
        assert result is True
        assert registry.get("to-remove") is None

    def test_remove_nonexistent_returns_false(self):
        """remove() should return False for unknown ID."""
        registry = DeploymentContextRegistry()

        result = registry.remove("nonexistent")
        assert result is False

    def test_persistent_storage(self):
        """Registry should persist to and load from storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            # Create and populate registry
            registry1 = DeploymentContextRegistry(storage_path)
            registry1.register(DeploymentContext(
                deployment_id="persistent-ctx",
                jurisdiction="AU",
            ))

            # Create new registry from same storage
            registry2 = DeploymentContextRegistry(storage_path)

            # Should find the persisted context
            ctx = registry2.get("persistent-ctx")
            assert ctx is not None
            assert ctx.jurisdiction == "AU"


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_registry()

    def teardown_method(self):
        """Reset global registry after each test."""
        reset_registry()

    def test_get_registry_returns_singleton(self):
        """get_registry() should return same instance."""
        reg1 = get_registry()
        reg2 = get_registry()

        assert reg1 is reg2

    def test_reset_registry_clears_singleton(self):
        """reset_registry() should clear the singleton."""
        reg1 = get_registry()
        reg1.register(DeploymentContext(deployment_id="test"))

        reset_registry()

        reg2 = get_registry()
        assert reg2.get("test") is None
