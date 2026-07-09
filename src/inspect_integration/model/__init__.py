"""
HatCat model provider for Inspect integration.

Exports:
    HatCatModel: Custom Inspect Model provider with HatCat instrumentation
    get_lens_pool: Access shared lens manager pool
    get_hush_pool: Access shared HUSH controller pool
"""

from .hatcat_model import HatCatModel, get_hatcat_model
from .lens_pool import get_lens_pool, get_hush_pool, LensPool, HushControllerPool

__all__ = [
    "HatCatModel",
    "get_hatcat_model",
    "get_lens_pool",
    "get_hush_pool",
    "LensPool",
    "HushControllerPool",
]
