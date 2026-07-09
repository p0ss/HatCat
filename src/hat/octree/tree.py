"""
Activation Space Octree Structure

A generalized space-partitioning tree for high-dimensional activation space.
Uses recursive bisection along principal variance dimensions.

Unlike a traditional 3D octree (8 children per node), this adapts to
high-dimensional space by splitting on the dimension with highest variance,
similar to a k-d tree but with deterministic addressing.

Cell addresses are bit strings encoding the path from root to leaf:
- Each bit represents which side of a split the point falls on
- Address length indicates depth in the tree
- Same activation always maps to same address (deterministic)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import json
from pathlib import Path


@dataclass
class CellAddress:
    """
    Deterministic address for an octree cell.

    Encoded as a bit string where each bit represents a binary split decision.
    The address uniquely identifies a region of activation space.
    """
    bits: str  # Binary string, e.g., "01101"

    @property
    def depth(self) -> int:
        return len(self.bits)

    def parent(self) -> Optional["CellAddress"]:
        if self.depth == 0:
            return None
        return CellAddress(self.bits[:-1])

    def children(self) -> Tuple["CellAddress", "CellAddress"]:
        return (
            CellAddress(self.bits + "0"),
            CellAddress(self.bits + "1"),
        )

    def is_ancestor_of(self, other: "CellAddress") -> bool:
        return other.bits.startswith(self.bits) and len(other.bits) > len(self.bits)

    def is_descendant_of(self, other: "CellAddress") -> bool:
        return self.bits.startswith(other.bits) and len(self.bits) > len(other.bits)

    def distance_to(self, other: "CellAddress") -> int:
        """Number of bits that differ (structural distance in tree)."""
        # Find common prefix length
        common = 0
        for a, b in zip(self.bits, other.bits):
            if a == b:
                common += 1
            else:
                break
        # Distance = steps up to common ancestor + steps down to other
        return (self.depth - common) + (other.depth - common)

    def __hash__(self):
        return hash(self.bits)

    def __eq__(self, other):
        return isinstance(other, CellAddress) and self.bits == other.bits

    def __str__(self):
        return f"Cell({self.bits})" if self.bits else "Cell(root)"

    def __repr__(self):
        return self.__str__()


@dataclass
class OctreeNode:
    """
    A node in the activation octree.

    Internal nodes have a split plane (dimension + threshold).
    Leaf nodes have a cell address and sample indices.
    """
    address: CellAddress

    # Split plane (for internal nodes)
    split_dim: Optional[int] = None
    split_threshold: Optional[float] = None

    # Children (for internal nodes)
    left: Optional["OctreeNode"] = None   # Points where x[split_dim] < threshold
    right: Optional["OctreeNode"] = None  # Points where x[split_dim] >= threshold

    # Samples (for leaf nodes)
    sample_indices: List[int] = field(default_factory=list)

    # Statistics
    centroid: Optional[np.ndarray] = None
    n_samples: int = 0

    # Semantic labels (filled in during alignment phase)
    concept_labels: List[str] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def to_dict(self) -> Dict:
        """Serialize for JSON storage."""
        d = {
            "address": self.address.bits,
            "n_samples": self.n_samples,
            "is_leaf": self.is_leaf,
        }
        if self.split_dim is not None:
            d["split_dim"] = self.split_dim
            d["split_threshold"] = float(self.split_threshold)
        if self.centroid is not None:
            d["centroid"] = self.centroid.tolist()
        if self.concept_labels:
            d["concept_labels"] = self.concept_labels
        if self.is_leaf:
            d["sample_indices"] = self.sample_indices
        return d

    @classmethod
    def from_dict(cls, d: Dict, left: Optional["OctreeNode"] = None,
                  right: Optional["OctreeNode"] = None) -> "OctreeNode":
        """Deserialize from JSON."""
        node = cls(
            address=CellAddress(d["address"]),
            split_dim=d.get("split_dim"),
            split_threshold=d.get("split_threshold"),
            left=left,
            right=right,
            sample_indices=d.get("sample_indices", []),
            n_samples=d["n_samples"],
            concept_labels=d.get("concept_labels", []),
        )
        if "centroid" in d:
            node.centroid = np.array(d["centroid"])
        return node


@dataclass
class ActivationOctree:
    """
    Complete octree structure for activation space.

    Provides:
    - Deterministic mapping from activations to cell addresses
    - Hierarchical organization of activation space
    - Statistics per cell (centroid, sample count, labels)
    """
    root: OctreeNode
    n_dimensions: int
    max_depth: int
    min_samples: int

    # Index for fast lookup
    _leaves: Dict[str, OctreeNode] = field(default_factory=dict)
    _all_nodes: Dict[str, OctreeNode] = field(default_factory=dict)

    def __post_init__(self):
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild leaf and node indices."""
        self._leaves = {}
        self._all_nodes = {}
        self._index_node(self.root)

    def _index_node(self, node: OctreeNode):
        """Recursively index a node and its children."""
        self._all_nodes[node.address.bits] = node
        if node.is_leaf:
            self._leaves[node.address.bits] = node
        else:
            if node.left:
                self._index_node(node.left)
            if node.right:
                self._index_node(node.right)

    def query(self, activation: np.ndarray) -> CellAddress:
        """
        Find the cell address for an activation vector.

        Traverses tree from root, following split decisions.
        Returns the leaf cell address.
        """
        node = self.root
        while not node.is_leaf:
            if activation[node.split_dim] < node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node.address

    def query_batch(self, activations: np.ndarray) -> List[CellAddress]:
        """Query multiple activations efficiently."""
        return [self.query(act) for act in activations]

    def get_node(self, address: CellAddress) -> Optional[OctreeNode]:
        """Get node by address."""
        return self._all_nodes.get(address.bits)

    def get_leaf(self, address: CellAddress) -> Optional[OctreeNode]:
        """Get leaf node by address."""
        return self._leaves.get(address.bits)

    @property
    def leaves(self) -> List[OctreeNode]:
        """All leaf nodes."""
        return list(self._leaves.values())

    @property
    def n_leaves(self) -> int:
        return len(self._leaves)

    @property
    def n_nodes(self) -> int:
        return len(self._all_nodes)

    def get_cells_at_depth(self, depth: int) -> List[OctreeNode]:
        """Get all nodes at a specific depth."""
        return [n for n in self._all_nodes.values() if n.address.depth == depth]

    def get_unmapped_leaves(self) -> List[OctreeNode]:
        """Get leaves with no concept labels."""
        return [n for n in self.leaves if not n.concept_labels]

    def get_mapped_leaves(self) -> List[OctreeNode]:
        """Get leaves with concept labels."""
        return [n for n in self.leaves if n.concept_labels]

    def coverage_stats(self) -> Dict:
        """Compute coverage statistics."""
        total_samples = sum(n.n_samples for n in self.leaves)
        mapped_samples = sum(n.n_samples for n in self.get_mapped_leaves())

        return {
            "total_leaves": self.n_leaves,
            "mapped_leaves": len(self.get_mapped_leaves()),
            "unmapped_leaves": len(self.get_unmapped_leaves()),
            "total_samples": total_samples,
            "mapped_samples": mapped_samples,
            "coverage_by_leaves": len(self.get_mapped_leaves()) / max(self.n_leaves, 1),
            "coverage_by_samples": mapped_samples / max(total_samples, 1),
        }

    def save(self, path: Path):
        """Save octree to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save structure as JSON
        def node_to_full_dict(node: OctreeNode) -> Dict:
            d = node.to_dict()
            if node.left:
                d["left"] = node_to_full_dict(node.left)
            if node.right:
                d["right"] = node_to_full_dict(node.right)
            return d

        structure = {
            "n_dimensions": self.n_dimensions,
            "max_depth": self.max_depth,
            "min_samples": self.min_samples,
            "root": node_to_full_dict(self.root),
        }

        with open(path / "structure.json", "w") as f:
            json.dump(structure, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ActivationOctree":
        """Load octree from disk."""
        path = Path(path)

        with open(path / "structure.json") as f:
            structure = json.load(f)

        def dict_to_node(d: Dict) -> OctreeNode:
            left = dict_to_node(d["left"]) if "left" in d else None
            right = dict_to_node(d["right"]) if "right" in d else None
            return OctreeNode.from_dict(d, left=left, right=right)

        root = dict_to_node(structure["root"])

        return cls(
            root=root,
            n_dimensions=structure["n_dimensions"],
            max_depth=structure["max_depth"],
            min_samples=structure["min_samples"],
        )

    def __repr__(self):
        stats = self.coverage_stats()
        return (
            f"ActivationOctree(dims={self.n_dimensions}, "
            f"leaves={stats['total_leaves']}, "
            f"mapped={stats['mapped_leaves']}, "
            f"coverage={stats['coverage_by_samples']:.1%})"
        )
