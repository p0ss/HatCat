"""Encyclopedia building module for concept activation patterns."""

from .concept_loader import (
    load_concepts,
    load_test_concepts,
    load_convergence_test_concepts,
    get_concept_category
)

__all__ = [
    'load_concepts',
    'load_test_concepts',
    'load_convergence_test_concepts',
    'get_concept_category'
]
