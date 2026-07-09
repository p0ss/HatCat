"""
Unit tests for HatCat-Inspect scorers module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.inspect_integration.config import HatCatMetrics
from src.inspect_integration.scorers.hatcat_metrics import (
    extract_hatcat_metrics,
    _dict_to_metrics,
    _ticks_to_metrics,
)


class TestMetricsExtraction:
    """Tests for metrics extraction from task state."""

    def test_dict_to_metrics(self):
        """Should convert dict to HatCatMetrics."""
        d = {
            "violation_count": 3,
            "steering_count": 2,
            "peak_safety_intensity": 0.8,
            "mean_safety_intensity": 0.4,
            "peak_concepts": {"deception": 0.9},
            "total_tokens": 100,
        }
        metrics = _dict_to_metrics(d)
        assert isinstance(metrics, HatCatMetrics)
        assert metrics.violation_count == 3
        assert metrics.steering_count == 2
        assert metrics.peak_safety_intensity == 0.8
        assert metrics.peak_concepts["deception"] == 0.9

    def test_dict_to_metrics_with_missing_keys(self):
        """Should handle missing keys with defaults."""
        d = {"violation_count": 1}
        metrics = _dict_to_metrics(d)
        assert metrics.violation_count == 1
        assert metrics.steering_count == 0
        assert metrics.peak_safety_intensity == 0.0

    def test_ticks_to_metrics_empty(self):
        """Should handle empty tick list."""
        metrics = _ticks_to_metrics([])
        assert metrics.total_tokens == 0
        assert metrics.violation_count == 0

    def test_ticks_to_metrics_with_data(self):
        """Should aggregate tick data correctly."""
        ticks = [
            {
                "violations": [{"type": "simplex"}],
                "steering_applied": [],
                "safety_intensity": 0.3,
                "concept_activations": {"deception_L0": 0.6},
                "is_filler": False,
                "significance": 0.7,
            },
            {
                "violations": [],
                "steering_applied": [{"concept": "honesty"}],
                "safety_intensity": 0.5,
                "concept_activations": {"deception_L0": 0.8},
                "is_filler": True,
                "significance": 0.2,
            },
        ]
        metrics = _ticks_to_metrics(ticks)

        assert metrics.total_tokens == 2
        assert metrics.violation_count == 1
        assert metrics.steering_count == 1
        assert metrics.peak_safety_intensity == 0.5
        assert metrics.mean_safety_intensity == 0.4
        assert "deception_L0" in metrics.peak_concepts
        assert metrics.filler_tokens == 1
        assert metrics.significant_tokens == 1


class TestExtractHatCatMetrics:
    """Tests for extract_hatcat_metrics function."""

    def test_returns_none_for_no_data(self):
        """Should return None when no HatCat data present."""
        class MockState:
            output = None
            metadata = {}

        result = extract_hatcat_metrics(MockState())
        assert result is None

    def test_extracts_from_output_metadata(self):
        """Should extract from state.output.metadata."""
        class MockOutput:
            metadata = {
                "hatcat_metrics": {
                    "violation_count": 5,
                    "peak_safety_intensity": 0.7,
                }
            }

        class MockState:
            output = MockOutput()
            metadata = {}

        result = extract_hatcat_metrics(MockState())
        assert result is not None
        assert result.violation_count == 5
        assert result.peak_safety_intensity == 0.7

    def test_extracts_from_ticks(self):
        """Should reconstruct from tick data."""
        class MockOutput:
            metadata = {
                "hatcat_ticks": [
                    {
                        "violations": [{"type": "test"}],
                        "steering_applied": [],
                        "safety_intensity": 0.6,
                        "concept_activations": {},
                        "is_filler": False,
                        "significance": 0.5,
                    }
                ]
            }

        class MockState:
            output = MockOutput()
            metadata = {}

        result = extract_hatcat_metrics(MockState())
        assert result is not None
        assert result.violation_count == 1
        assert result.total_tokens == 1

    def test_extracts_from_state_metadata(self):
        """Should fallback to state.metadata."""
        class MockState:
            output = None
            metadata = {
                "hatcat_metrics": {
                    "violation_count": 2,
                }
            }

        result = extract_hatcat_metrics(MockState())
        assert result is not None
        assert result.violation_count == 2
