#!/usr/bin/env python3
"""
Regenerate layer files from hierarchy_tree.

This takes the hierarchy_tree as source of truth and rebuilds layer0.json through layerN.json
to match the tree structure.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_hierarchy_tree(tree_path: Path) -> dict:
    """Load hierarchy tree JSON."""
    with open(tree_path) as f:
        return json.load(f)


def load_existing_layer_files(hierarchy_dir: Path) -> dict:
    """Load all existing layer files to get concept metadata."""
    concepts = {}
    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        with open(layer_file) as f:
            data = json.load(f)
        for c in data.get("concepts", []):
            term = c.get("sumo_term")
            if term:
                concepts[term] = c
    return concepts


def extract_tree_structure(tree: dict) -> tuple[dict, dict]:
    """
    Extract depth and parent info from tree.

    Returns:
        depths: {concept: depth}
        parents: {concept: parent_concept}
    """
    depths = {}
    parents = {}

    def walk(node, depth=0, parent=None):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            # Clean concept name
            clean_k = k
            for prefix in ['NEW:', 'MOVED:', 'ELEVATED:', 'ABSORBED:', 'RENAMED:', 'ORPHAN:']:
                clean_k = clean_k.replace(prefix, '')

            depths[clean_k] = depth
            if parent:
                parents[clean_k] = parent

            if isinstance(v, dict):
                walk(v, depth + 1, clean_k)

    walk(tree)
    return depths, parents


def rebuild_layer_files(
    tree_path: Path,
    hierarchy_dir: Path,
    dry_run: bool = False
):
    """Rebuild layer files from hierarchy tree."""

    print(f"Loading tree from {tree_path}")
    tree = load_hierarchy_tree(tree_path)

    # Extract structure from tree
    tree_depths, tree_parents = extract_tree_structure(tree)
    print(f"  Tree has {len(tree_depths)} concepts")

    # Load existing metadata
    existing = load_existing_layer_files(hierarchy_dir)
    print(f"  Existing layer files have {len(existing)} concepts with metadata")

    # Build new layer data
    by_layer = defaultdict(list)
    missing_metadata = []

    for concept, depth in tree_depths.items():
        if concept in existing:
            # Update existing concept with new depth and parent
            c = existing[concept].copy()
            c["layer"] = depth

            # Update parent_concepts if we have tree parent info
            if concept in tree_parents:
                c["parent_concepts"] = [tree_parents[concept]]

            by_layer[depth].append(c)
        else:
            # New concept without metadata
            missing_metadata.append(concept)
            by_layer[depth].append({
                "sumo_term": concept,
                "layer": depth,
                "parent_concepts": [tree_parents[concept]] if concept in tree_parents else [],
            })

    if missing_metadata:
        print(f"\n  WARNING: {len(missing_metadata)} concepts without full metadata:")
        for c in missing_metadata[:10]:
            print(f"    - {c}")
        if len(missing_metadata) > 10:
            print(f"    ... and {len(missing_metadata) - 10} more")

    # Find concepts in existing but not in tree (removed)
    removed = set(existing.keys()) - set(tree_depths.keys())
    if removed:
        print(f"\n  Removed concepts (in old layers but not in tree): {len(removed)}")
        for c in sorted(removed)[:10]:
            print(f"    - {c}")
        if len(removed) > 10:
            print(f"    ... and {len(removed) - 10} more")

    # Report distribution
    max_layer = max(by_layer.keys()) if by_layer else 0
    print(f"\n  New layer distribution:")
    total = 0
    for layer in range(max_layer + 1):
        count = len(by_layer.get(layer, []))
        print(f"    Layer {layer}: {count} concepts")
        total += count
    print(f"    Total: {total}")

    if dry_run:
        print("\n  [DRY RUN] No files written")
        return

    # Backup old files
    backup_dir = hierarchy_dir / "backup_pre_v8"
    backup_dir.mkdir(exist_ok=True)
    for old_file in hierarchy_dir.glob("layer*.json"):
        backup_path = backup_dir / old_file.name
        if not backup_path.exists():  # Don't overwrite existing backups
            old_file.rename(backup_path)
    print(f"\n  Backed up old layer files to {backup_dir}")

    # Write new layer files
    for layer in range(max_layer + 1):
        concepts = by_layer.get(layer, [])
        if not concepts:
            continue

        # Sort by name
        concepts.sort(key=lambda c: c.get("sumo_term", ""))

        layer_data = {
            "layer": layer,
            "concepts": concepts
        }

        layer_path = hierarchy_dir / f"layer{layer}.json"
        with open(layer_path, 'w') as f:
            json.dump(layer_data, f, indent=2)

        print(f"  Wrote {layer_path.name}: {len(concepts)} concepts")

    return max_layer, by_layer


def main():
    parser = argparse.ArgumentParser(description="Regenerate layer files from hierarchy tree")
    parser.add_argument("--tree", default="concept_packs/first-light/hierarchy/hierarchy_tree_v8.json",
                        help="Path to hierarchy tree JSON")
    parser.add_argument("--hierarchy-dir", default="concept_packs/first-light/hierarchy",
                        help="Directory containing layer files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    tree_path = Path(args.tree)
    hierarchy_dir = Path(args.hierarchy_dir)

    if not tree_path.exists():
        print(f"Error: Tree file not found: {tree_path}")
        return 1

    print("=" * 60)
    print("REGENERATING LAYER FILES FROM HIERARCHY TREE")
    print("=" * 60)

    rebuild_layer_files(tree_path, hierarchy_dir, dry_run=args.dry_run)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
