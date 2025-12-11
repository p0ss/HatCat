#!/usr/bin/env python3
"""
Fix hierarchy tree layer values to match actual tree depth.

The old system had artificial "layer" values (0-4) that didn't correspond
to actual tree position. This script updates all values to reflect the
true depth of each concept in the hierarchy.
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


def update_depths(node: dict, depth: int = 0) -> dict:
    """
    Recursively update all layer values to match actual tree depth.

    - Parent nodes (dicts) get their children recursively updated
    - Leaf nodes (integers) get replaced with their actual depth
    """
    if not isinstance(node, dict):
        # Leaf node - return actual depth
        return depth

    result = {}
    for k, v in node.items():
        if isinstance(v, dict):
            # Parent node - recurse
            result[k] = update_depths(v, depth + 1)
        else:
            # Leaf node - set to actual depth
            result[k] = depth

    return result


def analyze_depths(node: dict, depth: int = 0) -> dict:
    """Analyze depth distribution."""
    stats = defaultdict(int)

    def count(n, d):
        if not isinstance(n, dict):
            stats[d] += 1
            return
        stats[d] += 1
        for v in n.values():
            count(v, d + 1)

    count(node, depth)
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description='Fix hierarchy layer values to match actual depth')
    parser.add_argument('--input', type=str, required=True, help='Input hierarchy JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: overwrite input)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change without modifying')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    print(f"Loading: {input_path}")
    with open(input_path) as f:
        tree = json.load(f)

    # Analyze before
    print("\n" + "=" * 60)
    print("BEFORE - Depth distribution:")
    print("=" * 60)
    before_stats = analyze_depths(tree)
    for d in sorted(before_stats.keys()):
        print(f"  Depth {d}: {before_stats[d]} concepts")

    # Check old layer values
    old_layers = defaultdict(int)
    def count_old_layers(n):
        if not isinstance(n, dict):
            old_layers[n] += 1
            return
        for v in n.values():
            count_old_layers(v)
    count_old_layers(tree)

    print("\nOLD layer values found:")
    for layer in sorted(old_layers.keys()):
        print(f"  Layer {layer}: {old_layers[layer]} concepts")

    # Update depths
    new_tree = update_depths(tree)

    # Analyze after
    print("\n" + "=" * 60)
    print("AFTER - New depth values:")
    print("=" * 60)
    new_layers = defaultdict(int)
    def count_new_layers(n):
        if not isinstance(n, dict):
            new_layers[n] += 1
            return
        for v in n.values():
            count_new_layers(v)
    count_new_layers(new_tree)

    for layer in sorted(new_layers.keys()):
        print(f"  Depth {layer}: {new_layers[layer]} concepts")

    # Show sample
    print("\n" + "=" * 60)
    print("SAMPLE - First few entries:")
    print("=" * 60)

    def show_sample(node, depth=0, shown=[0], max_show=15):
        if shown[0] >= max_show:
            return
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if shown[0] >= max_show:
                break
            clean = clean_name(k)
            indent = "  " * depth
            if isinstance(v, dict):
                print(f"{indent}{clean} (parent, depth={depth})")
            else:
                print(f"{indent}{clean} = depth {v}")
                shown[0] += 1
            show_sample(v, depth + 1, shown, max_show)

    show_sample(new_tree)

    if args.dry_run:
        print("\n[DRY RUN - no files modified]")
    else:
        print(f"\nWriting to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(new_tree, f, indent=2)
        print("Done!")


if __name__ == '__main__':
    main()
