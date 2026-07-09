"""
Activation Space Octree

Geometric partitioning of model activation space for:
- Concept localization (mapping semantic concepts to regions)
- Cleft identification (which regions to modify for grafting)
- Coverage analysis (mapped vs unmapped activation space)
- Exploration (steering toward unmapped regions)

The octree subdivides high-dimensional activation space recursively,
creating a deterministic mapping from activations to cell addresses.
"""

from .tree import ActivationOctree, OctreeNode, CellAddress
from .builder import OctreeBuilder
from .query import OctreeQuery

__all__ = [
    "ActivationOctree",
    "OctreeNode",
    "CellAddress",
    "OctreeBuilder",
    "OctreeQuery",
]
