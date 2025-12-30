"""
CAT - Conjoined Adversarial Tomograph

Layer 2.5 of the FTW architecture. CAT detects concept divergence
and adversarial patterns by comparing concept activations.
"""

from .divergence import (
    SUMOConceptGraph,
    TokenToSUMOMapper,
    concept_divergence,
    batch_divergence,
    compute_temporal_lag,
)

__all__ = [
    'SUMOConceptGraph',
    'TokenToSUMOMapper',
    'concept_divergence',
    'batch_divergence',
    'compute_temporal_lag',
]
