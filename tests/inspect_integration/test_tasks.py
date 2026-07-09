"""
Unit tests for HatCat-Inspect tasks module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.inspect_integration.config import Condition
from src.inspect_integration.tasks.presets import (
    SAFETY_EVALS,
    KNOWLEDGE_EVALS,
    REASONING_EVALS,
    PRESET_BUNDLES,
    get_preset_bundle,
    get_comparison_matrix,
    get_adversarial_battery,
)
from src.inspect_integration.tasks.wrapped import KNOWN_EVALS


class TestKnownEvals:
    """Tests for known eval registry."""

    def test_common_evals_registered(self):
        """Common evals should be in registry."""
        assert "gsm8k" in KNOWN_EVALS
        assert "truthfulqa" in KNOWN_EVALS
        assert "mmlu" in KNOWN_EVALS
        assert "humaneval" in KNOWN_EVALS

    def test_eval_paths_are_strings(self):
        """Eval paths should be module paths."""
        for name, path in KNOWN_EVALS.items():
            assert isinstance(path, str)
            assert "." in path  # Should be module path


class TestPresets:
    """Tests for preset eval bundles."""

    def test_safety_evals_defined(self):
        """Safety evals should be defined."""
        assert len(SAFETY_EVALS) > 0
        assert "agentharm" in SAFETY_EVALS or "truthfulqa" in SAFETY_EVALS

    def test_knowledge_evals_defined(self):
        """Knowledge evals should be defined."""
        assert len(KNOWLEDGE_EVALS) > 0
        assert "mmlu" in KNOWLEDGE_EVALS

    def test_reasoning_evals_defined(self):
        """Reasoning evals should be defined."""
        assert len(REASONING_EVALS) > 0
        assert "gsm8k" in REASONING_EVALS

    def test_preset_bundles_complete(self):
        """All bundle types should be in PRESET_BUNDLES."""
        assert "safety" in PRESET_BUNDLES
        assert "knowledge" in PRESET_BUNDLES
        assert "reasoning" in PRESET_BUNDLES


class TestGetPresetBundle:
    """Tests for get_preset_bundle function."""

    def test_safety_bundle(self):
        """Should return safety eval configs."""
        configs = get_preset_bundle("safety")
        assert len(configs) > 0
        for config in configs:
            assert "eval_name" in config
            assert "condition" in config

    def test_unknown_bundle_raises(self):
        """Should raise for unknown bundle."""
        with pytest.raises(ValueError):
            get_preset_bundle("nonexistent")

    def test_custom_conditions(self):
        """Should accept custom conditions."""
        configs = get_preset_bundle("knowledge", conditions=["A", "B"])
        conditions = {c["condition"] for c in configs}
        assert "A" in conditions
        assert "B" in conditions


class TestGetComparisonMatrix:
    """Tests for get_comparison_matrix function."""

    def test_generates_matrix(self):
        """Should generate full eval x condition matrix."""
        matrix = get_comparison_matrix(
            evals=["gsm8k", "mmlu"],
            conditions=["A", "B", "C"]
        )
        # 2 evals x 3 conditions = 6 configs
        assert len(matrix) == 6

    def test_includes_run_id(self):
        """Should include run_id for each config."""
        matrix = get_comparison_matrix(["gsm8k"], ["A"])
        assert matrix[0]["run_id"] == "gsm8k_A"


class TestGetAdversarialBattery:
    """Tests for get_adversarial_battery function."""

    def test_returns_adversarial_configs(self):
        """Should return D/E/F condition configs."""
        configs = get_adversarial_battery()
        conditions = {c["condition"] for c in configs}
        assert "D" in conditions
        assert "E" in conditions
        assert "F" in conditions

    def test_includes_description(self):
        """Should include description."""
        configs = get_adversarial_battery()
        for config in configs:
            assert "description" in config
