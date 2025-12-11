#!/usr/bin/env python3
"""
Collapse intermediate parents when it would improve branching factor.

If a parent has children where collapsing them (promoting grandchildren)
would result in a reasonable branching factor (≤ max_children), do it.

The key insight: an intermediate node that splits a parent's children
into small groups adds latency without providing much value. Better to
have one parent with 8 children than one parent with 2 children that
each have 4 grandchildren.

This complements flatten_useless_parents.py which handles single-child cases.
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict


def clean_name(name: str) -> str:
    """Remove NEW:/MOVED:/ORPHAN: prefixes."""
    for p in ['NEW:', 'MOVED:', 'ORPHAN:']:
        if name.startswith(p):
            return name[len(p):]
    return name


def normalize_name(name: str) -> str:
    """Normalize name for comparison (lowercase, remove common suffixes)."""
    name = clean_name(name).lower()
    # Remove common suffixes that indicate the same concept
    for suffix in ['s', 'es', 'device', 'devices', 'tool', 'tools',
                   'artifact', 'artifacts', 'thing', 'things',
                   'object', 'objects', 'item', 'items']:
        if name.endswith(suffix) and len(name) > len(suffix) + 2:
            name = name[:-len(suffix)]
    return name


def names_are_redundant(parent_name: str, child_name: str) -> bool:
    """Check if child name is semantically redundant with parent."""
    parent_norm = normalize_name(parent_name)
    child_norm = normalize_name(child_name)

    # Direct match after normalization
    if parent_norm == child_norm:
        return True

    # One contains the other
    if parent_norm in child_norm or child_norm in parent_norm:
        return True

    # Check for common patterns like "XTools" -> "XDevice" or "X" -> "XDevice"
    # Strip common prefixes/suffixes and compare cores
    return False


def count_descendants(node: dict) -> int:
    """Count total descendants (children + grandchildren + ...)."""
    if not isinstance(node, dict):
        return 0
    count = len(node)
    for v in node.values():
        if isinstance(v, dict):
            count += count_descendants(v)
    return count


def count_direct_grandchildren(node: dict) -> int:
    """Count how many nodes would exist if we collapsed all children."""
    if not isinstance(node, dict):
        return 0

    total = 0
    for v in node.values():
        if isinstance(v, dict):
            total += len(v)
        else:
            total += 1  # Leaf counts as 1
    return total


def find_collapse_candidates(node: dict, parent_name: str = "ROOT", depth: int = 0,
                             max_children: int = 10, min_depth: int = 2) -> list:
    """
    Find parents where collapsing ALL children would keep us under max_children.

    The rule: if promoting all grandchildren directly under a parent would
    result in ≤ max_children, the intermediate layer is wasteful.

    Returns list of collapse opportunities.
    """
    candidates = []

    if not isinstance(node, dict):
        return candidates

    for k, v in node.items():
        clean_k = clean_name(k)

        if not isinstance(v, dict):
            continue

        # Check if collapsing ALL children of this node would be beneficial
        if depth >= min_depth:
            # Count how many grandchildren we'd have if we collapsed all children
            total_grandchildren = 0
            children_to_collapse = []

            for child_k, child_v in v.items():
                clean_child = clean_name(child_k)
                if isinstance(child_v, dict):
                    grandchild_count = len(child_v)
                    total_grandchildren += grandchild_count
                    children_to_collapse.append({
                        'name': clean_child,
                        'grandchildren': grandchild_count,
                        'grandchildren_names': [clean_name(gk) for gk in list(child_v.keys())[:3]]
                    })
                else:
                    # Leaf child - counts as 1
                    total_grandchildren += 1

            # If collapsing would keep us under max_children, it's a candidate
            if total_grandchildren <= max_children and len(children_to_collapse) > 0:
                candidates.append({
                    'parent': clean_k,
                    'depth': depth,
                    'current_children': len(v),
                    'total_grandchildren': total_grandchildren,
                    'children_to_collapse': children_to_collapse
                })

        # Recurse into children
        for child_k, child_v in v.items():
            if isinstance(child_v, dict):
                candidates.extend(find_collapse_candidates(
                    {child_k: child_v}, clean_k, depth + 1, max_children, min_depth
                ))

    return candidates


def collapse_all_children(node: dict, depth: int = 0,
                          max_children: int = 10, min_depth: int = 2,
                          collapsed: list = None) -> dict:
    """
    Recursively collapse intermediate layers where it improves branching factor.

    For each parent, if collapsing ALL its children (promoting grandchildren)
    would keep us under max_children, do it.
    """
    if collapsed is None:
        collapsed = []

    if not isinstance(node, dict):
        return node

    result = {}

    for k, v in node.items():
        clean_k = clean_name(k)

        if not isinstance(v, dict):
            result[k] = v
            continue

        # First, recursively process children (bottom-up)
        processed_v = collapse_all_children(v, depth + 1, max_children, min_depth, collapsed)

        # Now check if we should collapse ALL children of this node
        if depth >= min_depth:
            # Count total grandchildren
            total_grandchildren = 0
            has_intermediate_children = False

            for child_k, child_v in processed_v.items():
                if isinstance(child_v, dict):
                    total_grandchildren += len(child_v)
                    has_intermediate_children = True
                else:
                    total_grandchildren += 1

            # If collapsing would keep us under max_children, do it
            if has_intermediate_children and total_grandchildren <= max_children:
                new_v = {}
                eliminated_children = []

                for child_k, child_v in processed_v.items():
                    clean_child = clean_name(child_k)

                    if isinstance(child_v, dict):
                        # Collapse: promote grandchildren
                        eliminated_children.append(clean_child)
                        for gc_k, gc_v in child_v.items():
                            gc_clean = clean_name(gc_k)
                            if gc_k.startswith('NEW:') or gc_k.startswith('MOVED:'):
                                new_v[gc_k] = gc_v
                            else:
                                new_v[f'MOVED:{gc_clean}'] = gc_v
                    else:
                        # Keep leaf children as-is
                        new_v[child_k] = child_v

                collapsed.append({
                    'parent': clean_k,
                    'depth': depth,
                    'eliminated_children': eliminated_children,
                    'new_children_count': len(new_v)
                })

                result[k] = new_v
            else:
                result[k] = processed_v
        else:
            result[k] = processed_v

    return result


def main():
    parser = argparse.ArgumentParser(description='Collapse redundant parent-child relationships')
    parser.add_argument('--input', type=str, required=True, help='Input hierarchy JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--max-children', type=int, default=10,
                        help='Max children after collapse (default: 10)')
    parser.add_argument('--min-depth', type=int, default=2,
                        help='Min depth to start collapsing (default: 2)')
    parser.add_argument('--dry-run', action='store_true', help='Show candidates without modifying')
    parser.add_argument('--scan-only', action='store_true', help='Only scan for candidates, no output')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_stem(input_path.stem + '_collapsed')

    print(f"Loading: {input_path}")
    with open(input_path) as f:
        tree = json.load(f)

    # Find candidates
    print(f"\nScanning for redundant parent-child relationships...")
    print(f"  Max children after collapse: {args.max_children}")
    print(f"  Min depth: {args.min_depth}")

    candidates = find_collapse_candidates(tree, max_children=args.max_children, min_depth=args.min_depth)

    print(f"\n{'='*70}")
    print(f"FOUND {len(candidates)} COLLAPSE CANDIDATES")
    print(f"{'='*70}")

    for c in candidates:
        print(f"\n  {c['parent']} (depth {c['depth']})")
        print(f"    Current children: {c['current_children']}")
        print(f"    Total grandchildren: {c['total_grandchildren']}")
        print(f"    Children to eliminate: {[ch['name'] for ch in c['children_to_collapse']]}")

    if args.scan_only:
        return

    if args.dry_run:
        print(f"\n[DRY RUN - no files modified]")
        return

    # Apply collapses
    print(f"\nApplying collapses...")
    collapsed_ops = []
    new_tree = collapse_all_children(tree, max_children=args.max_children,
                                     min_depth=args.min_depth, collapsed=collapsed_ops)

    print(f"\n{'='*70}")
    print(f"COLLAPSE SUMMARY")
    print(f"{'='*70}")
    print(f"Parents that had children collapsed: {len(collapsed_ops)}")

    for op in collapsed_ops[:20]:
        print(f"  {op['parent']}: eliminated {op['eliminated_children']}, now has {op['new_children_count']} children")
    if len(collapsed_ops) > 20:
        print(f"  ... and {len(collapsed_ops) - 20} more")

    print(f"\nWriting to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(new_tree, f, indent=2)
    print("Done!")


if __name__ == '__main__':
    main()
