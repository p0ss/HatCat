#!/usr/bin/env python3
"""
Apply hierarchy flattening melds - Round 3.

Operations:
1. Delete PotOrPan (promote children to FoodContainers)
2. Delete grape varieties (children of Grape)
3. Fix incorrect layer numbers after hierarchy changes
4. Delete RollingStock, move children to parent (RailVehicle), delete grandchildren
5. Delete grandchildren of BodyVessel
"""

import json
import argparse
from pathlib import Path


def strip_markers(key):
    """Remove all marker prefixes from a key."""
    markers = ['NEW:', 'MOVED:', 'ORPHAN:', 'RENAMED:', 'ELEVATED:', 'ABSORBED:']
    result = key
    for m in markers:
        result = result.replace(m, '')
    return result


def find_node(node, target):
    """Find a node by name, return (parent_dict, key, value) or None."""
    if not isinstance(node, dict):
        return None

    for k, v in node.items():
        clean_k = strip_markers(k)
        if clean_k == target:
            return (node, k, v)

        if isinstance(v, dict):
            result = find_node(v, target)
            if result:
                return result

    return None


def delete_children(tree, target):
    """Delete all children of target, making it a leaf node."""
    result = find_node(tree, target)
    if not result:
        return False, 0

    parent, key, value = result
    if isinstance(value, dict):
        count = len(value)
        parent[key] = count  # Store count as leaf value
        return True, count
    return False, 0


def delete_and_promote(tree, target):
    """Delete target node and promote its children to parent level."""
    result = find_node(tree, target)
    if not result:
        return False

    parent, key, value = result

    if isinstance(value, dict):
        for child_k, child_v in value.items():
            clean_child = strip_markers(child_k)
            new_key = f"ELEVATED:{clean_child}"
            parent[new_key] = child_v

    del parent[key]
    return True


def delete_grandchildren(tree, target):
    """Delete all grandchildren of target (make children into leaves)."""
    result = find_node(tree, target)
    if not result:
        return False, 0

    parent, key, value = result
    if not isinstance(value, dict):
        return False, 0

    total_deleted = 0
    for child_k, child_v in value.items():
        if isinstance(child_v, dict):
            count = len(child_v)
            value[child_k] = count
            total_deleted += count

    return True, total_deleted


def delete_promote_and_delete_grandchildren(tree, target):
    """Delete target, promote children to parent, then delete those children's children."""
    result = find_node(tree, target)
    if not result:
        return False, 0

    parent, key, value = result
    if not isinstance(value, dict):
        return False, 0

    total_deleted = 0
    for child_k, child_v in value.items():
        clean_child = strip_markers(child_k)
        new_key = f"ELEVATED:{clean_child}"

        # If child has grandchildren, count and convert to leaf
        if isinstance(child_v, dict):
            total_deleted += len(child_v)
            parent[new_key] = len(child_v)  # Make leaf with count
        else:
            parent[new_key] = child_v

    del parent[key]
    return True, total_deleted


def recalculate_depths(node, depth=0):
    """Recalculate all depth values in the tree."""
    if not isinstance(node, dict):
        return node

    new_node = {}
    for k, v in node.items():
        if isinstance(v, dict):
            new_node[k] = recalculate_depths(v, depth + 1)
        elif isinstance(v, int):
            # Leaf node - set to current depth + 1
            new_node[k] = depth + 1
        else:
            new_node[k] = v

    return new_node


def clean_elevated_prefix(node):
    """Remove ELEVATED: prefix from keys."""
    if not isinstance(node, dict):
        return node

    new_node = {}
    for k, v in node.items():
        clean_k = k.replace('ELEVATED:', '')
        new_node[clean_k] = clean_elevated_prefix(v)

    return new_node


def apply_melds(tree):
    """Apply all round 3 flattening melds."""

    print("=== Round 3 Hierarchy Flattening ===\n")

    # 1. Delete PotOrPan
    if delete_and_promote(tree, 'PotOrPan'):
        print("✓ Deleted PotOrPan, promoted children to FoodContainers")
    else:
        print("✗ Could not find PotOrPan")

    # 2. Delete grape varieties
    success, count = delete_children(tree, 'Grape')
    if success:
        print(f"✓ Deleted {count} grape varieties")
    else:
        print("✗ Could not find Grape")

    # 3. Fix layer numbers - done at end with recalculate_depths

    # 4. Delete RollingStock, promote children, delete grandchildren
    success, count = delete_promote_and_delete_grandchildren(tree, 'RollingStock')
    if success:
        print(f"✓ Deleted RollingStock, promoted children to RailVehicle, deleted {count} grandchildren")
    else:
        print("✗ Could not find RollingStock")

    # 5. Delete grandchildren of BodyVessel
    success, count = delete_grandchildren(tree, 'BodyVessel')
    if success:
        print(f"✓ Deleted {count} grandchildren of BodyVessel")
    else:
        print("✗ Could not find BodyVessel")

    return tree


def main():
    parser = argparse.ArgumentParser(description='Flatten hierarchy - Round 3')
    parser.add_argument('--input', default='concept_packs/first-light/hierarchy/hierarchy_tree_v7.json')
    parser.add_argument('--output', default='concept_packs/first-light/hierarchy/hierarchy_tree_v8.json')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without writing')
    parser.add_argument('--keep-markers', action='store_true', help='Keep ELEVATED: markers')
    parser.add_argument('--no-recalc', action='store_true', help='Skip depth recalculation')
    args = parser.parse_args()

    with open(args.input) as f:
        tree = json.load(f)

    print(f"Loaded tree from {args.input}\n")

    tree = apply_melds(tree)

    if not args.keep_markers:
        tree = clean_elevated_prefix(tree)
        print("\n✓ Cleaned ELEVATED: markers")

    if not args.no_recalc:
        tree = recalculate_depths(tree)
        print("✓ Recalculated all depth values")

    if args.dry_run:
        print("\n[DRY RUN] Would write to:", args.output)
    else:
        with open(args.output, 'w') as f:
            json.dump(tree, f, indent=2)
        print(f"\n✓ Wrote flattened tree to {args.output}")


if __name__ == '__main__':
    main()
