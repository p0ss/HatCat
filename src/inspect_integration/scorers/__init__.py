"""
HatCat scorers for Inspect integration.

Exports:
    hatcat_metrics_scorer: Standalone scorer for HatCat metrics
    hatcat_combined_scorer: Wrapper to add HatCat metrics to any scorer
    HatCatMetric: Metric definitions for aggregation
"""

from .hatcat_metrics import (
    hatcat_metrics_scorer,
    extract_hatcat_metrics,
)
from .combined import (
    hatcat_combined_scorer,
    with_hatcat_metrics,
)

__all__ = [
    "hatcat_metrics_scorer",
    "extract_hatcat_metrics",
    "hatcat_combined_scorer",
    "with_hatcat_metrics",
]
