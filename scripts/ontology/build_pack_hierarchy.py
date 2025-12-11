#!/usr/bin/env python3
"""
Build authoritative parent-child hierarchy for a concept pack.

This script generates a single hierarchy.json file that contains:
1. For each concept: its canonical parent and all direct children
2. Leaf nodes (concepts with no children)
3. Root nodes (concepts with no parents in the hierarchy)

This hierarchy is built once at pack-build time and is authoritative -
the DynamicLensManager will load this directly instead of reconstructing
parent-child relationships at runtime from scattered fields.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def build_hierarchy(concept_pack_dir: Path) -> Dict:
    """
    Build authoritative hierarchy from concept pack layer files.

    Returns a dict with:
    - parent_to_children: {(concept, layer): [(child, layer), ...]}
    - child_to_parent: {(concept, layer): (parent, layer)}
    - leaf_concepts: set of (concept, layer) tuples with no children
    - root_concepts: set of (concept, layer) tuples with no parents
    - concept_metadata: {(concept, layer): {sumo_term, layer, ...}}
    """
    hierarchy_dir = concept_pack_dir / "hierarchy"

    # Load all concepts from layer files
    concept_metadata = {}

    layer_files = sorted(hierarchy_dir.glob("layer*.json"))
    for layer_file in layer_files:
        layer = int(layer_file.stem.replace("layer", ""))
        with open(layer_file) as f:
            data = json.load(f)

        for concept in data.get("concepts", []):
            sumo_term = concept["sumo_term"]
            concept_key = (sumo_term, layer)
            concept_metadata[concept_key] = {
                "sumo_term": sumo_term,
                "layer": layer,
                "parent_concepts": concept.get("parent_concepts", []),
                "category_children": concept.get("category_children", []),
            }

    # Build parent-child relationships
    # We build child_to_parent first (each child has exactly ONE canonical parent)
    # Then derive parent_to_children from it (ensuring tree structure, not DAG)
    child_to_parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

    # First pass: use category_children (downward links, primarily for L0)
    for concept_key, meta in concept_metadata.items():
        sumo_term, layer = concept_key

        for child_name in meta["category_children"]:
            # Find the child in next layer or same layer
            child_key = None
            for (cname, clayer) in concept_metadata.keys():
                if cname == child_name and clayer >= layer:
                    child_key = (cname, clayer)
                    break

            if child_key and child_key not in child_to_parent:
                child_to_parent[child_key] = concept_key

    # Second pass: use parent_concepts (upward links, for all layers)
    # This is more reliable for L1+ concepts and OVERRIDES category_children
    # NOTE: Don't restrict by layer number - parent_concepts can reference any layer
    # (e.g., Tool at layer 1 can have parent HarnessComponent at layer 2)
    for concept_key, meta in concept_metadata.items():
        sumo_term, layer = concept_key

        for parent_name in meta["parent_concepts"]:
            # Find parent by name in any layer
            parent_key = None
            for (pname, player) in concept_metadata.keys():
                if pname == parent_name:
                    parent_key = (pname, player)
                    break

            if parent_key:
                # Set child->parent (prefer explicit parent_concepts, overwrite if exists)
                child_to_parent[concept_key] = parent_key

    # Now derive parent_to_children FROM child_to_parent (ensures tree, not DAG)
    parent_to_children: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    for child, parent in child_to_parent.items():
        parent_to_children[parent].append(child)

    # Identify leaf concepts (no children) and root concepts (no parents)
    all_concepts = set(concept_metadata.keys())
    leaf_concepts = all_concepts - set(parent_to_children.keys())
    root_concepts = all_concepts - set(child_to_parent.keys())

    return {
        "parent_to_children": {
            f"{k[0]}:{k[1]}": [f"{c[0]}:{c[1]}" for c in v]
            for k, v in parent_to_children.items()
        },
        "child_to_parent": {
            f"{k[0]}:{k[1]}": f"{v[0]}:{v[1]}"
            for k, v in child_to_parent.items()
        },
        "leaf_concepts": [f"{c[0]}:{c[1]}" for c in leaf_concepts],
        "root_concepts": [f"{c[0]}:{c[1]}" for c in root_concepts],
        "total_concepts": len(all_concepts),
        "total_parents": len(parent_to_children),
        "total_leaves": len(leaf_concepts),
        "total_roots": len(root_concepts),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build authoritative hierarchy for a concept pack"
    )
    parser.add_argument(
        "--concept-pack",
        type=str,
        required=True,
        help="Concept pack name (e.g., sumo-wordnet-v4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: concept_packs/{pack}/hierarchy.json)",
    )

    args = parser.parse_args()

    concept_pack_dir = Path(f"concept_packs/{args.concept_pack}")
    if not concept_pack_dir.exists():
        print(f"Error: Concept pack not found: {concept_pack_dir}")
        return 1

    print(f"Building hierarchy for: {args.concept_pack}")
    print(f"  Source: {concept_pack_dir}")

    hierarchy = build_hierarchy(concept_pack_dir)

    output_path = Path(args.output) if args.output else concept_pack_dir / "hierarchy.json"

    with open(output_path, "w") as f:
        json.dump(hierarchy, f, indent=2, sort_keys=True)

    print(f"\n✓ Hierarchy written to: {output_path}")
    print(f"  Total concepts: {hierarchy['total_concepts']}")
    print(f"  Parent concepts: {hierarchy['total_parents']}")
    print(f"  Leaf concepts: {hierarchy['total_leaves']}")
    print(f"  Root concepts: {hierarchy['total_roots']}")

    # Show some examples
    print("\n  Sample parents with most children:")
    parent_counts = [(k, len(v)) for k, v in hierarchy["parent_to_children"].items()]
    parent_counts.sort(key=lambda x: x[1], reverse=True)
    for parent, count in parent_counts[:5]:
        print(f"    {parent}: {count} children")

    return 0


if __name__ == "__main__":
    exit(main())
