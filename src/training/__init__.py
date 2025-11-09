"""
Training module: Binary classifier training with neutral negative examples.

This module provides tools for:
- Training binary classifiers to detect concepts in activations
- Training text probes for fast token→concept mapping
- Generating training prompts (definitions, relationships, negatives)
- Extracting activations from language models
- Phase 4 neutral training methodology
"""

from .data_generation import (
    generate_definition_prompt,
    generate_relationship_prompts,
    generate_negative_prompts
)
from .classifier import train_binary_classifier, BinaryClassifier
from .activations import get_mean_activation
from .sumo_classifiers import train_sumo_classifiers
from .text_probes import (
    BinaryTextProbe,
    train_text_probe_for_concept,
    train_text_probes_for_layer,
)

__all__ = [
    "generate_definition_prompt",
    "generate_relationship_prompts",
    "generate_negative_prompts",
    "train_binary_classifier",
    "BinaryClassifier",
    "get_mean_activation",
    "train_sumo_classifiers",
    "BinaryTextProbe",
    "train_text_probe_for_concept",
    "train_text_probes_for_layer",
]
