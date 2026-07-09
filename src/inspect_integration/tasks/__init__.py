"""
HatCat task wrappers for Inspect integration.

Exports:
    hatcat_wrapped: Generic wrapper for any Inspect eval
    create_hatcat_task: Factory for creating wrapped tasks
    Preset task bundles for common eval scenarios
"""

from .wrapped import (
    hatcat_wrapped,
    create_hatcat_task,
    wrap_existing_task,
)
from .presets import (
    SAFETY_EVALS,
    KNOWLEDGE_EVALS,
    REASONING_EVALS,
    get_preset_bundle,
)

__all__ = [
    "hatcat_wrapped",
    "create_hatcat_task",
    "wrap_existing_task",
    "SAFETY_EVALS",
    "KNOWLEDGE_EVALS",
    "REASONING_EVALS",
    "get_preset_bundle",
]
