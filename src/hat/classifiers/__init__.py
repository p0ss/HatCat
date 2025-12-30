"""
HAT Classifiers Module

Classifier infrastructure for concept detection in LLM activations.
"""

from .classifier import MLPClassifier, LinearProbe, load_classifier, save_classifier
from .lens import Lens, ClassifierInfo, LensNotFoundError
from .capture import ActivationCapture, ActivationConfig, BaselineGenerator

__all__ = [
    # Classifiers
    "MLPClassifier",
    "LinearProbe",
    "load_classifier",
    "save_classifier",
    # Lens abstraction
    "Lens",
    "ClassifierInfo",
    "LensNotFoundError",
    # Capture
    "ActivationCapture",
    "ActivationConfig",
    "BaselineGenerator",
]
