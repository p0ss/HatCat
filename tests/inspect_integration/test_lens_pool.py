"""
Unit tests for HatCat-Inspect lens pool module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.inspect_integration.model.lens_pool import (
    LensPool,
    HushControllerPool,
    get_lens_pool,
    get_hush_pool,
)


class TestLensPool:
    """Tests for LensPool singleton."""

    def test_singleton_pattern(self):
        """Should return same instance."""
        pool1 = LensPool()
        pool2 = LensPool()
        assert pool1 is pool2

    def test_get_lens_pool_function(self):
        """get_lens_pool should return pool instance."""
        pool = get_lens_pool()
        assert isinstance(pool, LensPool)

    def test_make_key(self):
        """Should create consistent cache keys."""
        pool = get_lens_pool()
        key1 = pool._make_key("lens_packs/test", "cuda", 2048)
        key2 = pool._make_key("lens_packs/test", "cuda", 2048)
        assert key1 == key2

        key3 = pool._make_key("lens_packs/other", "cuda", 2048)
        assert key1 != key3

    def test_clear(self):
        """Should clear cached managers."""
        pool = get_lens_pool()
        # Clear any existing
        pool.clear()
        stats = pool.get_stats()
        assert stats["num_managers"] == 0

    def test_get_stats(self):
        """Should return pool statistics."""
        pool = get_lens_pool()
        pool.clear()
        stats = pool.get_stats()
        assert "num_managers" in stats
        assert "keys" in stats


class TestHushControllerPool:
    """Tests for HushControllerPool singleton."""

    def test_singleton_pattern(self):
        """Should return same instance."""
        pool1 = HushControllerPool()
        pool2 = HushControllerPool()
        assert pool1 is pool2

    def test_get_hush_pool_function(self):
        """get_hush_pool should return pool instance."""
        pool = get_hush_pool()
        assert isinstance(pool, HushControllerPool)

    def test_make_key(self):
        """Should create consistent cache keys."""
        pool = get_hush_pool()
        key1 = pool._make_key("lens_packs/test", "profile.yaml")
        key2 = pool._make_key("lens_packs/test", "profile.yaml")
        assert key1 == key2

        key3 = pool._make_key("lens_packs/test", None)
        assert key1 != key3

    def test_clear(self):
        """Should clear cached controllers."""
        pool = get_hush_pool()
        pool.clear()
        # After clearing, should have no controllers
        assert len(pool._controllers) == 0
