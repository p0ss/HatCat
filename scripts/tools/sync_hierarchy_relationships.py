#!/usr/bin/env python3
"""
Sync parent_concepts and category_children in layer files.

The hierarchy stores parent-child relationships in two places:
- parent_concepts on children (child says "my parent is X")
- category_children on parents (parent says "my children are Y, Z")

These can get out of sync. This script merges both directions so that
if A->B exists in either direction, it exists in both.

Usage:
    python scripts/tools/sync_hierarchy_relationships.py concept_packs/first-light
    python scripts/tools/sync_hierarchy_relationships.py concept_packs/first-light --dry-run
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_all_concepts(hierarchy_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    """Load all concepts from layer files."""
    all_concepts = []
    for layer in range(7):
        try:
            with open(hierarchy_dir / f"layer{layer}.json") as f:
                data = json.load(f)
                all_concepts.extend(data["concepts"])
        except FileNotFoundError:
            continue
    concept_map = {c["sumo_term"]: c for c in all_concepts}
    return all_concepts, concept_map


def analyze_asymmetry(all_concepts: list[dict], concept_map: dict[str, dict]) -> dict:
    """Analyze the current state of parent-child relationships."""
    from_parent_concepts = set()  # (parent, child) from child's parent_concepts
    from_category_children = set()  # (parent, child) from parent's category_children

    for concept in all_concepts:
        term = concept["sumo_term"]
        for parent in concept.get("parent_concepts", []):
            from_parent_concepts.add((parent, term))
        for child in concept.get("category_children", []):
            from_category_children.add((term, child))

    both = from_parent_concepts & from_category_children
    only_in_parent = from_parent_concepts - from_category_children
    only_in_children = from_category_children - from_parent_concepts

    # Filter to only relationships where both concepts exist
    only_in_parent_valid = {(p, c) for p, c in only_in_parent if p in concept_map}
    only_in_children_valid = {(p, c) for p, c in only_in_children if c in concept_map}

    return {
        "symmetric": len(both),
        "only_in_parent_concepts": len(only_in_parent),
        "only_in_parent_concepts_valid": len(only_in_parent_valid),
        "only_in_category_children": len(only_in_children),
        "only_in_category_children_valid": len(only_in_children_valid),
        "to_add_to_category_children": only_in_parent_valid,
        "to_add_to_parent_concepts": only_in_children_valid,
    }


def sync_relationships(hierarchy_dir: Path, dry_run: bool = False) -> dict:
    """Sync parent_concepts and category_children in all layer files."""

    # First load all concepts to build complete picture
    all_concepts, concept_map = load_all_concepts(hierarchy_dir)

    # Analyze current state
    analysis = analyze_asymmetry(all_concepts, concept_map)

    print(f"Current state:")
    print(f"  Symmetric relationships: {analysis['symmetric']}")
    print(f"  Only in parent_concepts: {analysis['only_in_parent_concepts']} ({analysis['only_in_parent_concepts_valid']} fixable)")
    print(f"  Only in category_children: {analysis['only_in_category_children']} ({analysis['only_in_category_children_valid']} fixable)")

    if dry_run:
        print(f"\n[DRY RUN] Would add:")
        print(f"  {len(analysis['to_add_to_category_children'])} entries to category_children")
        print(f"  {len(analysis['to_add_to_parent_concepts'])} entries to parent_concepts")
        return analysis

    # Build maps of what to add
    add_to_category_children = defaultdict(set)  # parent -> {children to add}
    add_to_parent_concepts = defaultdict(set)  # child -> {parents to add}

    for parent, child in analysis["to_add_to_category_children"]:
        add_to_category_children[parent].add(child)

    for parent, child in analysis["to_add_to_parent_concepts"]:
        add_to_parent_concepts[child].add(parent)

    # Process each layer file
    total_category_children_added = 0
    total_parent_concepts_added = 0

    for layer in range(7):
        layer_path = hierarchy_dir / f"layer{layer}.json"
        if not layer_path.exists():
            continue

        with open(layer_path) as f:
            layer_data = json.load(f)

        modified = False

        for concept in layer_data["concepts"]:
            term = concept["sumo_term"]

            # Add missing category_children
            if term in add_to_category_children:
                if "category_children" not in concept:
                    concept["category_children"] = []
                for child in add_to_category_children[term]:
                    if child not in concept["category_children"]:
                        concept["category_children"].append(child)
                        total_category_children_added += 1
                        modified = True
                concept["category_children"].sort()

            # Add missing parent_concepts
            if term in add_to_parent_concepts:
                if "parent_concepts" not in concept:
                    concept["parent_concepts"] = []
                for parent in add_to_parent_concepts[term]:
                    if parent not in concept["parent_concepts"]:
                        concept["parent_concepts"].append(parent)
                        total_parent_concepts_added += 1
                        modified = True
                concept["parent_concepts"].sort()

        if modified:
            with open(layer_path, "w") as f:
                json.dump(layer_data, f, indent=2)
            print(f"  Updated {layer_path.name}")

    print(f"\nSync complete:")
    print(f"  Added {total_category_children_added} entries to category_children")
    print(f"  Added {total_parent_concepts_added} entries to parent_concepts")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Sync parent-child relationships in layer files")
    parser.add_argument("pack_dir", type=Path, help="Path to concept pack directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    args = parser.parse_args()

    hierarchy_dir = args.pack_dir / "hierarchy"
    if not hierarchy_dir.exists():
        print(f"Error: {hierarchy_dir} does not exist")
        return 1

    sync_relationships(hierarchy_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    exit(main())
