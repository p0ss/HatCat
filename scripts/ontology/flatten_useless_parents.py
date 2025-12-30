#!/usr/bin/env python3
"""
Flatten useless intermediary nodes in hierarchy tree.

A "useless" parent is one with only 1 child - it adds latency without
providing meaningful categorization. This script promotes the child
directly to the grandparent, eliminating the useless intermediate level.

For parents with 2 children, we preserve them as they provide minimal
categorization value, but flag them for review.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def clean_name(name: str) -> str:
    """Remove NEW:/MOVED:/ORPHAN: prefixes."""
    for p in ['NEW:', 'MOVED:', 'ORPHAN:']:
        if name.startswith(p):
            return name[len(p):]
    return name


def count_useless(node: dict) -> dict:
    """Count useless parents in a tree."""
    counts = {'single_child': 0, 'double_child': 0, 'total_parents': 0}

    if not isinstance(node, dict):
        return counts

    for k, v in node.items():
        if isinstance(v, dict):
            child_count = len(v)
            counts['total_parents'] += 1
            if child_count == 1:
                counts['single_child'] += 1
            elif child_count == 2:
                counts['double_child'] += 1

            # Recurse
            sub_counts = count_useless(v)
            for key in counts:
                counts[key] += sub_counts[key]

    return counts


def flatten_single_child_parents(node: dict, flattened: list = None, depth: int = 0, min_depth: int = 2) -> dict:
    """
    Recursively flatten single-child parents by promoting their child.

    Args:
        node: The tree node to process
        flattened: List to record flattening operations
        depth: Current depth in tree
        min_depth: Don't flatten at depths less than this (preserves top-level structure)

    Returns modified tree and list of flattening operations performed.
    """
    if flattened is None:
        flattened = []

    if not isinstance(node, dict):
        return node

    result = {}

    for k, v in node.items():
        clean_k = clean_name(k)

        if not isinstance(v, dict):
            # Leaf node - keep as is
            result[k] = v
            continue

        # Parent node - check if it has single child AND we're deep enough to flatten
        if len(v) == 1 and depth >= min_depth:
            # Get the single child
            child_key = list(v.keys())[0]
            child_value = v[child_key]
            clean_child = clean_name(child_key)

            # Record the flattening
            flattened.append({
                'eliminated': clean_k,
                'child_promoted': clean_child,
                'depth': depth
            })

            # Promote child to this level (skip the useless parent)
            # But first, recursively flatten the child's subtree
            flattened_child_value = flatten_single_child_parents(child_value, flattened, depth, min_depth)

            # Mark as moved if it was a meaningful concept
            if child_key.startswith('NEW:') or child_key.startswith('MOVED:'):
                result[child_key] = flattened_child_value
            else:
                result[f'MOVED:{clean_child}'] = flattened_child_value
        else:
            # Multiple children OR too shallow to flatten - keep parent, recurse to flatten children
            result[k] = flatten_single_child_parents(v, flattened, depth + 1, min_depth)

    return result


def analyze_tree(tree: dict) -> None:
    """Print analysis of tree structure."""

    def get_depth_stats(node, depth=0, stats=None):
        if stats is None:
            stats = defaultdict(lambda: {'count': 0, 'children_sum': 0})

        if not isinstance(node, dict):
            return stats

        for k, v in node.items():
            if isinstance(v, dict):
                child_count = len(v)
                stats[depth]['count'] += 1
                stats[depth]['children_sum'] += child_count
                get_depth_stats(v, depth + 1, stats)

        return stats

    stats = get_depth_stats(tree)

    print("\nTree structure by depth:")
    print(f"{'Depth':<8} {'Parents':<10} {'Avg Children':<15}")
    print("-" * 35)

    for depth in sorted(stats.keys()):
        avg = stats[depth]['children_sum'] / stats[depth]['count'] if stats[depth]['count'] > 0 else 0
        print(f"{depth:<8} {stats[depth]['count']:<10} {avg:<15.1f}")


def main():
    parser = argparse.ArgumentParser(description='Flatten single-child parents in hierarchy')
    parser.add_argument('--input', type=str, required=True, help='Input hierarchy JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: input with _flattened suffix)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying')
    parser.add_argument('--min-depth', type=int, default=2, help='Minimum depth to start flattening (default: 2)')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_stem(input_path.stem + '_flattened')

    print(f"Loading: {input_path}")
    with open(input_path) as f:
        tree = json.load(f)

    # Before stats
    print("\n" + "=" * 60)
    print("BEFORE FLATTENING")
    print("=" * 60)
    before_counts = count_useless(tree)
    print(f"Total parents: {before_counts['total_parents']}")
    print(f"Single-child parents (will be eliminated): {before_counts['single_child']}")
    print(f"Double-child parents (preserved): {before_counts['double_child']}")
    analyze_tree(tree)

    # Flatten
    print(f"\nFlattening with min_depth={args.min_depth} (preserving depths 0-{args.min_depth - 1})")
    flattened_ops = []
    new_tree = flatten_single_child_parents(tree, flattened_ops, depth=0, min_depth=args.min_depth)

    # After stats
    print("\n" + "=" * 60)
    print("AFTER FLATTENING")
    print("=" * 60)
    after_counts = count_useless(new_tree)
    print(f"Total parents: {after_counts['total_parents']}")
    print(f"Single-child parents: {after_counts['single_child']}")
    print(f"Double-child parents: {after_counts['double_child']}")
    analyze_tree(new_tree)

    # Summary
    print("\n" + "=" * 60)
    print("FLATTENING SUMMARY")
    print("=" * 60)
    print(f"Parents eliminated: {len(flattened_ops)}")
    print(f"Depth reduction potential: {before_counts['single_child']} levels removed")

    # Show sample operations
    if flattened_ops:
        print("\nSample flattenings:")
        for op in flattened_ops[:20]:
            print(f"  {op['eliminated']} → {op['child_promoted']} (depth {op['depth']})")
        if len(flattened_ops) > 20:
            print(f"  ... and {len(flattened_ops) - 20} more")

    if args.dry_run:
        print("\n[DRY RUN - no files modified]")
    else:
        print(f"\nWriting to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(new_tree, f, indent=2)

        # Also save the operations log
        ops_path = output_path.with_stem(output_path.stem + '_operations')
        with open(ops_path, 'w') as f:
            json.dump({
                'before': before_counts,
                'after': after_counts,
                'operations': flattened_ops
            }, f, indent=2)
        print(f"Operations log: {ops_path}")


if __name__ == '__main__':
    main()
