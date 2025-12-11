#!/usr/bin/env python3
"""
Absorb single children into their parents.

When a parent has exactly one child, the child is absorbed into the parent:
- Parent keeps its name
- Child's subtree becomes parent's subtree
- Child is marked with ABSORBED: prefix to indicate the meld operation needed

This differs from flatten_useless_parents.py which promotes the child's name.
Here we keep the parent's name, which means at meld time we use the child's
probe for the parent concept.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def clean_name(name: str) -> str:
    """Remove prefixes."""
    for p in ['NEW:', 'MOVED:', 'ORPHAN:', 'RENAMED:', 'ELEVATED:', 'ABSORBED:']:
        if name.startswith(p):
            return name[len(p):]
    return name


def count_single_child_parents(node: dict, depth: int = 0, min_depth: int = 2) -> list:
    """Find all single-child parents at or below min_depth."""
    candidates = []

    if not isinstance(node, dict):
        return candidates

    for k, v in node.items():
        if not isinstance(v, dict):
            continue

        clean_k = clean_name(k)

        # Check if this is a single-child parent
        if len(v) == 1 and depth >= min_depth:
            child_key = list(v.keys())[0]
            child_value = v[child_key]
            clean_child = clean_name(child_key)

            candidates.append({
                'parent': clean_k,
                'child': clean_child,
                'depth': depth,
                'child_has_children': isinstance(child_value, dict)
            })

        # Recurse
        candidates.extend(count_single_child_parents(v, depth + 1, min_depth))

    return candidates


def absorb_single_children(node: dict, depth: int = 0, min_depth: int = 2,
                           absorbed: list = None) -> dict:
    """
    Recursively absorb single children into parents.

    Parent keeps its name, child's subtree becomes parent's subtree.
    Child key gets ABSORBED: prefix to track the operation.
    """
    if absorbed is None:
        absorbed = []

    if not isinstance(node, dict):
        return node

    result = {}

    for k, v in node.items():
        if not isinstance(v, dict):
            result[k] = v
            continue

        clean_k = clean_name(k)

        # First recursively process children
        processed_v = absorb_single_children(v, depth + 1, min_depth, absorbed)

        # Now check if this processed result has single child
        if len(processed_v) == 1 and depth >= min_depth:
            child_key = list(processed_v.keys())[0]
            child_value = processed_v[child_key]
            clean_child = clean_name(child_key)

            absorbed.append({
                'parent': clean_k,
                'absorbed_child': clean_child,
                'depth': depth
            })

            # Parent keeps its key, but takes child's subtree
            # Mark child as absorbed in the new structure
            if isinstance(child_value, dict):
                # Child was a parent - its children become our children
                new_v = {}
                for gc_k, gc_v in child_value.items():
                    gc_clean = clean_name(gc_k)
                    # Preserve existing prefixes or add MOVED
                    if gc_k.startswith(('NEW:', 'MOVED:', 'ABSORBED:', 'RENAMED:', 'ELEVATED:')):
                        new_v[gc_k] = gc_v
                    else:
                        new_v[f'MOVED:{gc_clean}'] = gc_v

                # Add marker for the absorbed child
                new_v[f'ABSORBED:{clean_child}'] = depth + 1
                result[k] = new_v
            else:
                # Child was a leaf - parent becomes leaf with absorbed marker
                result[k] = {f'ABSORBED:{clean_child}': child_value}
        else:
            result[k] = processed_v

    return result


def main():
    parser = argparse.ArgumentParser(description='Absorb single children into parents')
    parser.add_argument('--input', type=str, required=True, help='Input hierarchy JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--min-depth', type=int, default=2,
                        help='Min depth to start absorbing (default: 2)')
    parser.add_argument('--dry-run', action='store_true', help='Show candidates without modifying')
    parser.add_argument('--scan-only', action='store_true', help='Only scan for candidates')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_stem(input_path.stem + '_absorbed')

    print(f"Loading: {input_path}")
    with open(input_path) as f:
        tree = json.load(f)

    # Find candidates
    print(f"\nScanning for single-child parents (min_depth={args.min_depth})...")
    candidates = count_single_child_parents(tree, min_depth=args.min_depth)

    print(f"\n{'='*70}")
    print(f"FOUND {len(candidates)} SINGLE-CHILD PARENTS")
    print(f"{'='*70}")

    # Group by depth
    by_depth = defaultdict(list)
    for c in candidates:
        by_depth[c['depth']].append(c)

    for d in sorted(by_depth.keys()):
        print(f"\n  Depth {d}: {len(by_depth[d])} candidates")
        for c in by_depth[d][:5]:
            child_type = "parent" if c['child_has_children'] else "leaf"
            print(f"    {c['parent']} -> {c['child']} ({child_type})")
        if len(by_depth[d]) > 5:
            print(f"    ... and {len(by_depth[d]) - 5} more")

    if args.scan_only:
        return

    if args.dry_run:
        print(f"\n[DRY RUN - no files modified]")
        return

    # Apply absorptions
    print(f"\nApplying absorptions...")
    absorbed_ops = []
    new_tree = absorb_single_children(tree, min_depth=args.min_depth, absorbed=absorbed_ops)

    print(f"\n{'='*70}")
    print(f"ABSORPTION SUMMARY")
    print(f"{'='*70}")
    print(f"Parents that absorbed children: {len(absorbed_ops)}")

    for op in absorbed_ops[:20]:
        print(f"  {op['parent']} absorbed {op['absorbed_child']} (depth {op['depth']})")
    if len(absorbed_ops) > 20:
        print(f"  ... and {len(absorbed_ops) - 20} more")

    print(f"\nWriting to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(new_tree, f, indent=2)
    print("Done!")


if __name__ == '__main__':
    main()
