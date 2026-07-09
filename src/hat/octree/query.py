"""
Octree Query and Analysis

Tools for querying the octree and analyzing coverage.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import logging

from .tree import ActivationOctree, OctreeNode, CellAddress

logger = logging.getLogger(__name__)


@dataclass
class CoverageReport:
    """Report on octree coverage by semantic concepts."""
    total_leaves: int
    mapped_leaves: int
    unmapped_leaves: int

    total_samples: int
    mapped_samples: int
    unmapped_samples: int

    concepts_mapped: int
    concept_to_cells: Dict[str, List[str]]  # concept -> list of cell addresses
    cell_to_concepts: Dict[str, List[str]]  # cell address -> list of concepts

    # Overlap analysis
    overlapping_cells: int  # Cells with multiple concepts
    unique_cells: int  # Cells with exactly one concept

    def coverage_by_leaves(self) -> float:
        return self.mapped_leaves / max(self.total_leaves, 1)

    def coverage_by_samples(self) -> float:
        return self.mapped_samples / max(self.total_samples, 1)

    def to_dict(self) -> Dict:
        return {
            "total_leaves": self.total_leaves,
            "mapped_leaves": self.mapped_leaves,
            "unmapped_leaves": self.unmapped_leaves,
            "coverage_by_leaves": self.coverage_by_leaves(),
            "total_samples": self.total_samples,
            "mapped_samples": self.mapped_samples,
            "unmapped_samples": self.unmapped_samples,
            "coverage_by_samples": self.coverage_by_samples(),
            "concepts_mapped": self.concepts_mapped,
            "overlapping_cells": self.overlapping_cells,
            "unique_cells": self.unique_cells,
        }


class OctreeQuery:
    """
    Query interface for activation octree.

    Provides:
    - Activation to cell mapping
    - Coverage analysis
    - Nearest cell search
    - Concept-to-cell mapping
    """

    def __init__(self, octree: ActivationOctree):
        self.octree = octree

    def locate(self, activation: np.ndarray) -> CellAddress:
        """Find cell address for an activation."""
        return self.octree.query(activation)

    def locate_batch(self, activations: np.ndarray) -> List[CellAddress]:
        """Find cell addresses for multiple activations."""
        return self.octree.query_batch(activations)

    def get_cell_info(self, address: CellAddress) -> Optional[Dict]:
        """Get information about a cell."""
        node = self.octree.get_node(address)
        if node is None:
            return None

        return {
            "address": address.bits,
            "depth": address.depth,
            "n_samples": node.n_samples,
            "is_leaf": node.is_leaf,
            "concept_labels": node.concept_labels,
            "has_concepts": len(node.concept_labels) > 0,
        }

    def get_neighbors(self, address: CellAddress, max_distance: int = 2) -> List[CellAddress]:
        """
        Find neighboring cells (cells within tree distance).

        Tree distance = steps up to common ancestor + steps down to neighbor.
        """
        neighbors = []
        for leaf in self.octree.leaves:
            dist = address.distance_to(leaf.address)
            if 0 < dist <= max_distance:
                neighbors.append(leaf.address)
        return neighbors

    def analyze_coverage(self) -> CoverageReport:
        """Analyze concept coverage of the octree."""
        leaves = self.octree.leaves

        mapped_leaves = [l for l in leaves if l.concept_labels]
        unmapped_leaves = [l for l in leaves if not l.concept_labels]

        mapped_samples = sum(l.n_samples for l in mapped_leaves)
        unmapped_samples = sum(l.n_samples for l in unmapped_leaves)

        # Build concept-to-cell mapping
        concept_to_cells: Dict[str, List[str]] = {}
        cell_to_concepts: Dict[str, List[str]] = {}

        for leaf in leaves:
            cell_to_concepts[leaf.address.bits] = leaf.concept_labels.copy()
            for concept in leaf.concept_labels:
                if concept not in concept_to_cells:
                    concept_to_cells[concept] = []
                concept_to_cells[concept].append(leaf.address.bits)

        # Count overlaps
        overlapping = sum(1 for c in cell_to_concepts.values() if len(c) > 1)
        unique = sum(1 for c in cell_to_concepts.values() if len(c) == 1)

        return CoverageReport(
            total_leaves=len(leaves),
            mapped_leaves=len(mapped_leaves),
            unmapped_leaves=len(unmapped_leaves),
            total_samples=mapped_samples + unmapped_samples,
            mapped_samples=mapped_samples,
            unmapped_samples=unmapped_samples,
            concepts_mapped=len(concept_to_cells),
            concept_to_cells=concept_to_cells,
            cell_to_concepts=cell_to_concepts,
            overlapping_cells=overlapping,
            unique_cells=unique,
        )

    def find_unmapped_regions(self, min_samples: int = 10) -> List[OctreeNode]:
        """
        Find unmapped regions worth exploring.

        Returns leaves with no concept labels but sufficient samples
        to be interesting.
        """
        candidates = []
        for leaf in self.octree.leaves:
            if not leaf.concept_labels and leaf.n_samples >= min_samples:
                candidates.append(leaf)

        # Sort by sample count (most populated first)
        candidates.sort(key=lambda n: n.n_samples, reverse=True)
        return candidates

    def find_concept_cells(self, concept: str) -> List[OctreeNode]:
        """Find all cells associated with a concept."""
        cells = []
        for leaf in self.octree.leaves:
            if concept in leaf.concept_labels:
                cells.append(leaf)
        return cells

    def compute_concept_spread(self, concept: str) -> Dict:
        """
        Analyze how spread out a concept is across cells.

        Returns stats about the concept's distribution in activation space.
        """
        cells = self.find_concept_cells(concept)

        if not cells:
            return {
                "concept": concept,
                "n_cells": 0,
                "total_samples": 0,
                "spread": 0,
            }

        # Compute centroid of all concept cells
        centroids = [c.centroid for c in cells if c.centroid is not None]
        if not centroids:
            return {
                "concept": concept,
                "n_cells": len(cells),
                "total_samples": sum(c.n_samples for c in cells),
                "spread": 0,
            }

        centroids = np.stack(centroids)
        mean_centroid = centroids.mean(axis=0)

        # Compute spread as mean distance from concept centroid
        distances = np.linalg.norm(centroids - mean_centroid, axis=1)
        spread = float(distances.mean())

        return {
            "concept": concept,
            "n_cells": len(cells),
            "total_samples": sum(c.n_samples for c in cells),
            "spread": spread,
            "mean_centroid": mean_centroid,
        }

    def steering_vector_to_cell(self, target: CellAddress) -> Optional[np.ndarray]:
        """
        Compute a steering vector toward a target cell.

        Returns the centroid of the target cell, which can be used
        as a direction for steering.
        """
        node = self.octree.get_node(target)
        if node is None or node.centroid is None:
            return None
        return node.centroid

    def compare_cells(self, addr1: CellAddress, addr2: CellAddress) -> Dict:
        """Compare two cells."""
        node1 = self.octree.get_node(addr1)
        node2 = self.octree.get_node(addr2)

        if node1 is None or node2 is None:
            return {"error": "One or both cells not found"}

        result = {
            "tree_distance": addr1.distance_to(addr2),
            "cell1_samples": node1.n_samples,
            "cell2_samples": node2.n_samples,
            "cell1_concepts": node1.concept_labels,
            "cell2_concepts": node2.concept_labels,
        }

        if node1.centroid is not None and node2.centroid is not None:
            result["centroid_distance"] = float(
                np.linalg.norm(node1.centroid - node2.centroid)
            )

        return result
