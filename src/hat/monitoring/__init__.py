"""
HAT Monitoring Module

Real-time concept monitoring and lens management for LLM activations.
"""

# Core lens types
from .lens_types import LensRole, SimpleMLP, SimplexBinding, ConceptMetadata

# Modular components
from .lens_batched import BatchedLensBank
from .lens_hierarchy import HierarchyManager
from .lens_cache import LensCacheManager
from .lens_loader import LensLoader, MetadataLoader
from .lens_simplex import SimplexManager

# Main orchestrator
from .lens_manager import DynamicLensManager

# Monitor
from .monitor import SUMOTemporalMonitor, load_sumo_classifiers
from .sumo_temporal import run_temporal_detection

# Detectors
from .centroid_detector import CentroidTextDetector
from .text_lens import TfidfConceptLens

__all__ = [
    # Core types
    "LensRole",
    "SimpleMLP",
    "SimplexBinding",
    "ConceptMetadata",
    # Modular components
    "BatchedLensBank",
    "HierarchyManager",
    "LensCacheManager",
    "LensLoader",
    "MetadataLoader",
    "SimplexManager",
    # Main orchestrator
    "DynamicLensManager",
    # Monitor
    "SUMOTemporalMonitor",
    "load_sumo_classifiers",
    "run_temporal_detection",
    # Detectors
    "CentroidTextDetector",
    "TfidfConceptLens",
]
