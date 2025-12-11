#!/usr/bin/env python3
"""
Rebuild concept pack layer files using TRUE hierarchy depth.

The layer files should reflect actual parent-child depth, not arbitrary domain groupings.
This script reads hierarchy_tree_v5.json (the pruned runtime hierarchy) which has the
true depth encoded as integer leaf values.

Usage:
    python scripts/rebuild_layer_files_true_depth.py --concept-pack first-light
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_hierarchy_tree_v5(pack_dir: Path) -> dict[str, int]:
    """Load hierarchy_tree_v5.json and extract depth for all concepts.

    The format is nested dicts where leaf values are depth integers.
    Branch nodes (dicts) need their depth computed from position in tree.
    """
    hierarchy_path = pack_dir / "hierarchy" / "hierarchy_tree_v5.json"

    with open(hierarchy_path) as f:
        tree = json.load(f)

    depths = {}

    def walk(obj, current_depth=0, parent_name=None):
        """Walk the tree and collect depths.

        - If value is an int, that's a leaf with explicit depth
        - If value is a dict, it's a branch node at current_depth
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Clean up concept name (remove prefixes like MOVED:, NEW:, etc.)
                clean_name = key
                for prefix in ['MOVED:', 'NEW:', 'ELEVATED:', 'ABSORBED:', 'PRUNED:']:
                    if clean_name.startswith(prefix):
                        clean_name = clean_name[len(prefix):]
                        break

                if isinstance(value, int):
                    # Leaf node with explicit depth
                    depths[clean_name] = value
                elif isinstance(value, dict):
                    # Branch node - its depth is current_depth
                    # But we need to figure out the depth from the tree structure
                    # The root is depth 0, each level adds 1
                    depths[clean_name] = current_depth
                    walk(value, current_depth + 1, clean_name)

    # Start with depth 0 for root level
    walk(tree, current_depth=0)

    return depths


def load_existing_layer_files(pack_dir: Path) -> dict:
    """Load all existing layer files to get full concept metadata."""
    concepts = {}
    hierarchy_dir = pack_dir / "hierarchy"

    for layer_file in hierarchy_dir.glob("layer*.json"):
        with open(layer_file) as f:
            layer_data = json.load(f)

        for concept in layer_data.get("concepts", []):
            term = concept.get("sumo_term")
            if term:
                concepts[term] = concept

    return concepts


def parse_concept_depth(key: str) -> tuple[str, int]:
    """Parse 'ConceptName:depth' format."""
    if ':' in key:
        parts = key.rsplit(':', 1)
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass
    return key, -1


def load_hierarchy_json_depths(pack_dir: Path) -> dict[str, int]:
    """Load depths from hierarchy.json as fallback for concepts not in v5."""
    hierarchy_path = pack_dir / "hierarchy" / "hierarchy.json"

    if not hierarchy_path.exists():
        return {}

    with open(hierarchy_path) as f:
        data = json.load(f)

    depths = {}

    # The child_to_parent dict has format "Child:depth": "Parent:depth"
    child_to_parent = data.get("child_to_parent", {})
    for child_key, parent_key in child_to_parent.items():
        child_name, child_depth = parse_concept_depth(child_key)
        parent_name, parent_depth = parse_concept_depth(parent_key)

        if child_depth >= 0:
            depths[child_name] = child_depth
        if parent_depth >= 0:
            depths[parent_name] = parent_depth

    # Also check parent_to_children if present
    parent_to_children = data.get("parent_to_children", {})
    for parent_key, children in parent_to_children.items():
        parent_name, parent_depth = parse_concept_depth(parent_key)
        if parent_depth >= 0:
            depths[parent_name] = parent_depth

        for child_key in children:
            child_name, child_depth = parse_concept_depth(child_key)
            if child_depth >= 0:
                depths[child_name] = child_depth

    return depths


def compute_depths_from_parent_concepts(concepts: dict) -> dict[str, int]:
    """Compute depth by walking parent_concepts from roots via BFS."""
    # Build parent->children mapping
    children_of = defaultdict(set)
    for term, c in concepts.items():
        for parent in c.get("parent_concepts", []):
            children_of[parent].add(term)

    # Find roots (concepts with no parents or parents not in our set)
    roots = []
    for term, c in concepts.items():
        parents = c.get("parent_concepts", [])
        if not parents or all(p not in concepts for p in parents):
            roots.append(term)

    # Compute depth by BFS from roots
    depths = {}
    queue = [(r, 0) for r in roots]
    while queue:
        term, depth = queue.pop(0)
        if term in depths:
            continue
        depths[term] = depth
        for child in children_of.get(term, []):
            if child not in depths:
                queue.append((child, depth + 1))

    return depths


def rebuild_layer_files(pack_dir: Path, dry_run: bool = False):
    """Rebuild layer files with true depth assignments."""
    print(f"Loading hierarchy from {pack_dir}")

    # Load existing concepts with full metadata
    concepts = load_existing_layer_files(pack_dir)
    print(f"  Found {len(concepts)} concepts in existing layer files")

    # Compute depths from parent_concepts (ground truth)
    depths = compute_depths_from_parent_concepts(concepts)
    print(f"  Computed depths for {len(depths)} concepts from parent_concepts")

    # Group concepts by true depth
    by_depth = defaultdict(list)
    unmapped = []

    for term, concept_data in concepts.items():
        true_depth = depths.get(term)
        if true_depth is not None:
            # Update the layer field to match true depth
            concept_data = concept_data.copy()
            concept_data["layer"] = true_depth
            by_depth[true_depth].append(concept_data)
        else:
            unmapped.append(term)

    if unmapped:
        print(f"\n  WARNING: {len(unmapped)} concepts without depth mapping:")
        for term in unmapped[:10]:
            print(f"    - {term}")
        if len(unmapped) > 10:
            print(f"    ... and {len(unmapped) - 10} more")

    # Report distribution
    print(f"\n  Depth distribution:")
    total = 0
    max_depth = max(by_depth.keys()) if by_depth else 0
    for depth in range(max_depth + 1):
        count = len(by_depth.get(depth, []))
        print(f"    Layer {depth}: {count} concepts")
        total += count
    print(f"    Total: {total}")

    if dry_run:
        print("\n  DRY RUN - no files written")
        return

    # Write new layer files
    hierarchy_dir = pack_dir / "hierarchy"

    # Backup old files
    backup_dir = hierarchy_dir / "backup_old_layers"
    backup_dir.mkdir(exist_ok=True)
    for old_file in hierarchy_dir.glob("layer*.json"):
        old_file.rename(backup_dir / old_file.name)
    print(f"\n  Backed up old layer files to {backup_dir}")

    # Write new layer files
    for depth in range(max_depth + 1):
        layer_concepts = by_depth.get(depth, [])
        if not layer_concepts:
            continue

        # Sort concepts by name for consistency
        layer_concepts.sort(key=lambda c: c.get("sumo_term", ""))

        layer_data = {
            "layer": depth,
            "concepts": layer_concepts
        }

        layer_path = hierarchy_dir / f"layer{depth}.json"
        with open(layer_path, 'w') as f:
            json.dump(layer_data, f, indent=2)

        print(f"  Wrote {layer_path.name}: {len(layer_concepts)} concepts")

    return max_depth, by_depth


def update_pack_json(pack_dir: Path, max_depth: int, by_depth: dict):
    """Update pack.json with new layer metadata."""
    pack_json = pack_dir / "pack.json"

    with open(pack_json) as f:
        pack_data = json.load(f)

    # Update layer list and distribution
    layers = list(range(max_depth + 1))
    distribution = {str(d): len(by_depth.get(d, [])) for d in layers}

    pack_data["concept_metadata"]["layers"] = layers
    pack_data["concept_metadata"]["layer_distribution"] = distribution

    with open(pack_json, 'w') as f:
        json.dump(pack_data, f, indent=2)

    print(f"\n  Updated {pack_json}")
    print(f"    Layers: {layers}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild layer files with true depth")
    parser.add_argument("--concept-pack", required=True, help="Concept pack name")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")
    args = parser.parse_args()

    pack_dir = Path("concept_packs") / args.concept_pack
    if not pack_dir.exists():
        print(f"Error: Concept pack not found at {pack_dir}")
        return 1

    print("=" * 70)
    print("REBUILDING LAYER FILES WITH TRUE DEPTH")
    print("=" * 70)

    result = rebuild_layer_files(pack_dir, dry_run=args.dry_run)

    if result and not args.dry_run:
        max_depth, by_depth = result
        update_pack_json(pack_dir, max_depth, by_depth)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
