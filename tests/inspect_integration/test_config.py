"""
Unit tests for HatCat-Inspect config module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.inspect_integration.config import (
    Condition,
    HatCatConfig,
    HatCatMetrics,
    LensConfig,
    SteeringConfig,
    InductionConfig,
    INDUCTION_CONCEPTS,
)


class TestCondition:
    """Tests for Condition enum."""

    def test_all_conditions_exist(self):
        """All conditions A-F should be defined."""
        assert Condition.A.value == "A"
        assert Condition.B.value == "B"
        assert Condition.C.value == "C"
        assert Condition.D.value == "D"
        assert Condition.E.value == "E"
        assert Condition.F.value == "F"

    def test_condition_from_string(self):
        """Should create condition from string."""
        assert Condition("A") == Condition.A
        assert Condition("C") == Condition.C

    def test_induction_concepts_mapping(self):
        """Adversarial conditions should have induction concepts."""
        assert Condition.D in INDUCTION_CONCEPTS
        assert Condition.E in INDUCTION_CONCEPTS
        assert Condition.F in INDUCTION_CONCEPTS
        assert INDUCTION_CONCEPTS[Condition.D] == "deception"
        assert INDUCTION_CONCEPTS[Condition.E] == "sycophancy"
        assert INDUCTION_CONCEPTS[Condition.F] == "manipulation"


class TestHatCatConfig:
    """Tests for HatCatConfig dataclass."""

    def test_default_config(self):
        """Default config should have sane values."""
        config = HatCatConfig()
        assert config.condition == Condition.A
        assert config.model_name == "google/gemma-3-4b-it"
        assert config.device == "cuda"
        assert config.max_new_tokens == 512

    def test_monitoring_enabled_property(self):
        """monitoring_enabled should be False only for condition A."""
        assert not HatCatConfig(condition=Condition.A).monitoring_enabled
        assert HatCatConfig(condition=Condition.B).monitoring_enabled
        assert HatCatConfig(condition=Condition.C).monitoring_enabled
        assert HatCatConfig(condition=Condition.D).monitoring_enabled

    def test_steering_enabled_property(self):
        """steering_enabled should be True for C, D, E, F."""
        assert not HatCatConfig(condition=Condition.A).steering_enabled
        assert not HatCatConfig(condition=Condition.B).steering_enabled
        assert HatCatConfig(condition=Condition.C).steering_enabled
        assert HatCatConfig(condition=Condition.D).steering_enabled
        assert HatCatConfig(condition=Condition.E).steering_enabled
        assert HatCatConfig(condition=Condition.F).steering_enabled

    def test_induction_enabled_property(self):
        """induction_enabled should be True for D, E, F only."""
        assert not HatCatConfig(condition=Condition.A).induction_enabled
        assert not HatCatConfig(condition=Condition.B).induction_enabled
        assert not HatCatConfig(condition=Condition.C).induction_enabled
        assert HatCatConfig(condition=Condition.D).induction_enabled
        assert HatCatConfig(condition=Condition.E).induction_enabled
        assert HatCatConfig(condition=Condition.F).induction_enabled

    def test_auto_induction_config(self):
        """Adversarial conditions should auto-create induction config."""
        config_d = HatCatConfig(condition=Condition.D)
        assert config_d.induction is not None
        assert config_d.induction.concept == "deception"

        config_e = HatCatConfig(condition=Condition.E)
        assert config_e.induction.concept == "sycophancy"

        config_f = HatCatConfig(condition=Condition.F)
        assert config_f.induction.concept == "manipulation"

    def test_custom_induction_config(self):
        """Custom induction config should override auto-config."""
        custom = InductionConfig(concept="custom", strength=0.5)
        config = HatCatConfig(condition=Condition.D, induction=custom)
        assert config.induction.concept == "custom"
        assert config.induction.strength == 0.5


class TestHatCatMetrics:
    """Tests for HatCatMetrics dataclass."""

    def test_default_metrics(self):
        """Default metrics should be zeroed."""
        metrics = HatCatMetrics()
        assert metrics.violation_count == 0
        assert metrics.steering_count == 0
        assert metrics.peak_safety_intensity == 0.0

    def test_to_dict(self):
        """Should convert to dict for Inspect metadata."""
        metrics = HatCatMetrics(
            violation_count=5,
            peak_safety_intensity=0.7,
            peak_concepts={"deception": 0.8},
        )
        d = metrics.to_dict()
        assert d["violation_count"] == 5
        assert d["peak_safety_intensity"] == 0.7
        assert d["peak_concepts"]["deception"] == 0.8

    def test_from_empty_worldticks(self):
        """Should handle empty tick list."""
        metrics = HatCatMetrics.from_worldticks([])
        assert metrics.violation_count == 0
        assert metrics.total_tokens == 0

    def test_from_worldticks_with_mock_data(self):
        """Should aggregate metrics from mock tick data."""
        # Create mock tick-like objects
        class MockTick:
            def __init__(self):
                self.violations = [{"type": "simplex"}]
                self.steering_applied = [{"concept": "honesty"}]
                self.safety_intensity = 0.6
                self.concept_activations = {"deception_L0": 0.7}
                self.is_filler = False
                self.significance = 0.8

        ticks = [MockTick(), MockTick()]
        metrics = HatCatMetrics.from_worldticks(ticks)

        assert metrics.total_tokens == 2
        assert metrics.violation_count == 2
        assert metrics.steering_count == 2
        assert metrics.peak_safety_intensity == 0.6


class TestLensConfig:
    """Tests for LensConfig dataclass."""

    def test_defaults(self):
        """Should have default lens pack and thresholds."""
        config = LensConfig()
        assert config.lens_pack == "lens_packs/sumo-2k"
        assert config.load_threshold == 0.5
        assert config.unload_threshold == 0.1
        assert config.base_layers == [0, 1, 2]


class TestSteeringConfig:
    """Tests for SteeringConfig dataclass."""

    def test_defaults(self):
        """Should have default steering strength."""
        config = SteeringConfig()
        assert config.steering_strength == 0.3
        assert config.enable_layer_escalation is True
