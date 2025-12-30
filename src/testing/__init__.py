"""
Shared testing utilities for HatCat concept detection.

This module provides reusable functions for running concept detection experiments
with DynamicLensManager.
"""

from .concept_test_runner import (
    generate_with_concept_detection,
    score_activation_with_lens_manager,
    batch_score_activations,
)

__all__ = [
    'generate_with_concept_detection',
    'score_activation_with_lens_manager',
    'batch_score_activations',
]
