"""
HAT Monitoring Module

Real-time concept monitoring and lens management for LLM activations.
"""

from .lens_manager import DynamicLensManager, SimpleMLP, ConceptMetadata, LensRole
from .monitor import SUMOTemporalMonitor, load_sumo_classifiers
from .sumo_temporal import run_temporal_detection
from .centroid_detector import CentroidTextDetector
from .text_lens import TfidfConceptLens

__all__ = [
    # Lens Manager
    "DynamicLensManager",
    "SimpleMLP",
    "ConceptMetadata",
    "LensRole",
    # Monitor
    "SUMOTemporalMonitor",
    "load_sumo_classifiers",
    "run_temporal_detection",
    # Detectors
    "CentroidTextDetector",
    "TfidfConceptLens",
]
