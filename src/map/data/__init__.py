"""
Data utilities for concept packs and lens manifests.

Provides:
- LensManifest: Version tracking for lens packs
- ConceptEmbeddingIndex: Embedding-based concept lookup
- WordNetPatchLoader: Custom WordNet extensions
- Meld validation and application utilities
"""

from .version_manifest import LensManifest, LensEntry, ClassifierEntry

__all__ = [
    'LensManifest',
    'LensEntry',
    'ClassifierEntry',
]
