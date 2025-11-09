"""Activation capture module for neural network interpretability."""

from .hooks import ActivationCapture, ActivationConfig, BaselineGenerator

__all__ = ['ActivationCapture', 'ActivationConfig', 'BaselineGenerator']
