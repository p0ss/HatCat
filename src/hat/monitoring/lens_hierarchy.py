#!/usr/bin/env python3
"""
Lens Hierarchy Management

Manages parent-child relationships between concepts for hierarchical
lens expansion and decomposition.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from .lens_types import ConceptMetadata


class HierarchyManager:
    """
    Manages the concept hierarchy for dynamic lens expansion.

    Tracks parent-child relationships to enable:
    - Hierarchical decomposition (parent → children when parent fires)
    - Path tracing (leaf → root for context)
    - Leaf detection (concepts with no children)
    """

    def __init__(self):
        # Parent-child mappings
        # Key: (sumo_term, layer)
        self.parent_to_children: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
        self.child_to_parent: Dict[Tuple[str, int], Tuple[str, int]] = {}
        self.leaf_concepts: Set[Tuple[str, int]] = set()

    def load_authoritative_hierarchy(
        self,
        hierarchy_path: Path,
        concept_metadata: Dict[Tuple[str, int], ConceptMetadata]
    ):
        """
        Load parent-child mappings from authoritative hierarchy.json file.

        Args:
            hierarchy_path: Path to hierarchy.json
            concept_metadata: Dict of concept_key -> ConceptMetadata (for filtering)
        """
        with open(hierarchy_path) as f:
            hierarchy_data = json.load(f)

        # Parse parent_to_children
        for parent_str, children_list in hierarchy_data.get("parent_to_children", {}).items():
            name, layer = parent_str.rsplit(":", 1)
            parent_key = (name, int(layer))
            # Only include if parent is in our concept_metadata (has a lens)
            if parent_key not in concept_metadata:
                continue
            for child_str in children_list:
                child_name, child_layer = child_str.rsplit(":", 1)
                child_key = (child_name, int(child_layer))
                # Only include if child is in our concept_metadata (has a lens)
                if child_key in concept_metadata:
                    self.parent_to_children[parent_key].append(child_key)

        # Parse child_to_parent
        for child_str, parent_str in hierarchy_data.get("child_to_parent", {}).items():
            child_name, child_layer = child_str.rsplit(":", 1)
            child_key = (child_name, int(child_layer))
            parent_name, parent_layer = parent_str.rsplit(":", 1)
            parent_key = (parent_name, int(parent_layer))
            # Only include if both are in concept_metadata
            if child_key in concept_metadata and parent_key in concept_metadata:
                self.child_to_parent[child_key] = parent_key

        # Parse leaf_concepts (concepts with no children)
        for leaf_str in hierarchy_data.get("leaf_concepts", []):
            leaf_name, leaf_layer = leaf_str.rsplit(":", 1)
            leaf_key = (leaf_name, int(leaf_layer))
            if leaf_key in concept_metadata:
                self.leaf_concepts.add(leaf_key)

        print(f"  Loaded authoritative hierarchy from: {hierarchy_path.name}")

    def build_from_metadata(
        self,
        concept_metadata: Dict[Tuple[str, int], ConceptMetadata]
    ):
        """
        Fallback: build parent-child mappings from concept metadata fields.

        Args:
            concept_metadata: Dict of concept_key -> ConceptMetadata
        """
        # Build from category_children (downward) and parent_concepts (upward)
        for concept_key, metadata in concept_metadata.items():
            sumo_term, layer = concept_key

            # Build parent->children from category_children
            for child_name in metadata.category_children:
                child_key = None
                for (cname, clayer) in concept_metadata.keys():
                    if cname == child_name and clayer >= layer:
                        child_key = (cname, clayer)
                        break
                if child_key:
                    self.parent_to_children[concept_key].append(child_key)
                    if child_key not in self.child_to_parent:
                        self.child_to_parent[child_key] = concept_key

            # Build child->parent from parent_concepts
            for parent_name in metadata.parent_concepts:
                parent_key = None
                for (pname, player) in concept_metadata.keys():
                    if pname == parent_name and player <= layer:
                        parent_key = (pname, player)
                        break
                if parent_key:
                    self.child_to_parent[concept_key] = parent_key
                    if concept_key not in self.parent_to_children[parent_key]:
                        self.parent_to_children[parent_key].append(concept_key)

        # Compute leaf concepts (concepts with no children)
        all_concepts = set(concept_metadata.keys())
        parent_concepts = set(self.parent_to_children.keys())
        self.leaf_concepts = all_concepts - parent_concepts

        print(f"  Built hierarchy from metadata (fallback mode)")

    def get_children(self, concept_key: Tuple[str, int]) -> List[Tuple[str, int]]:
        """Get children of a concept."""
        return self.parent_to_children.get(concept_key, [])

    def get_parent(self, concept_key: Tuple[str, int]) -> Optional[Tuple[str, int]]:
        """Get parent of a concept."""
        return self.child_to_parent.get(concept_key)

    def is_leaf(self, concept_key: Tuple[str, int]) -> bool:
        """Check if concept is a leaf (has no children)."""
        return concept_key in self.leaf_concepts

    def is_parent(self, concept_key: Tuple[str, int]) -> bool:
        """Check if concept is a parent (has children)."""
        return concept_key in self.parent_to_children

    def get_path_to_root(self, concept_key: Tuple[str, int]) -> List[str]:
        """
        Get hierarchical path from concept to root.

        Args:
            concept_key: (sumo_term, layer) tuple

        Returns:
            List of concept names from root to this concept
        """
        path = [concept_key[0]]
        current = concept_key

        while current in self.child_to_parent:
            parent_key = self.child_to_parent[current]
            path.insert(0, parent_key[0])
            current = parent_key

        return path

    def get_branch_concepts(
        self,
        branch_root: Tuple[str, int],
        max_depth: int = None
    ) -> Set[Tuple[str, int]]:
        """
        Get all concepts under a branch up to max_depth.

        Args:
            branch_root: Root concept key
            max_depth: Maximum depth to traverse (None = unlimited)

        Returns:
            Set of concept keys under this branch
        """
        branch_concepts = set()
        queue = [(branch_root, 0)]  # (concept_key, current_depth)

        while queue:
            current_key, current_depth = queue.pop(0)
            branch_concepts.add(current_key)

            # Expand to children if not at max depth
            if max_depth is None or current_depth < max_depth:
                children = self.parent_to_children.get(current_key, [])
                for child_key in children:
                    queue.append((child_key, current_depth + 1))

        return branch_concepts

    def get_stats(self) -> Dict[str, int]:
        """Get hierarchy statistics."""
        total_relationships = sum(len(children) for children in self.parent_to_children.values())
        return {
            "unique_parents": len(self.parent_to_children),
            "total_relationships": total_relationships,
            "leaf_concepts": len(self.leaf_concepts),
        }


__all__ = ["HierarchyManager"]
