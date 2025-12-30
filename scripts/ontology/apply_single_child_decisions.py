#!/usr/bin/env python3
"""
Apply decisions from single_child_review.txt to the hierarchy tree.

Decisions:
  P = Keep Parent only (delete child)
  C = Keep Child only (delete parent, promote child up)
  K = Keep Both (skip - will add children later)
  X = Delete Both
"""

import json
import argparse
import re
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


def delete_node(tree, target):
    """Delete a node entirely."""
    result = find_node(tree, target)
    if not result:
        return False
    parent, key, value = result
    del parent[key]
    return True


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


def parse_review_file(filepath):
    """Parse the review file and extract decisions."""
    decisions = {'P': [], 'C': [], 'X': [], 'K': []}

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Find the decision letter (P, C, K, X) - it's near the end after |
            # Handle formats like "| P", "|P", "P|", "  P"
            decision_match = re.search(r'\|\s*([PCKX])\s*\|?\s*$|([PCKX])\s*\|\s*$', line)
            if not decision_match:
                continue

            decision = decision_match.group(1) or decision_match.group(2)

            # Parse the structure part: [L2] Grandparent -> Parent -> Child
            struct_match = re.match(r'\[L\d+\]\s+(.+?)\s+->\s+(.+?)\s+->\s+(.+?)\s*\|', line)
            if struct_match:
                grandparent, parent, child = struct_match.groups()
                if decision in decisions:
                    decisions[decision].append({
                        'grandparent': grandparent.strip(),
                        'parent': parent.strip(),
                        'child': child.strip()
                    })

    return decisions


def clean_elevated_prefix(node):
    """Remove ELEVATED: prefix from keys."""
    if not isinstance(node, dict):
        return node

    new_node = {}
    for k, v in node.items():
        clean_k = k.replace('ELEVATED:', '')
        new_node[clean_k] = clean_elevated_prefix(v)

    return new_node


def fix_depths(node, depth=0):
    """Set leaf values to their actual depth."""
    if not isinstance(node, dict):
        return node

    new_node = {}
    for k, v in node.items():
        if isinstance(v, dict):
            new_node[k] = fix_depths(v, depth + 1)
        elif isinstance(v, int):
            new_node[k] = depth
        else:
            new_node[k] = v
    return new_node


def apply_decisions(tree, decisions, dry_run=False):
    """Apply all decisions to the tree."""

    stats = {'P': 0, 'C': 0, 'X': 0, 'K': len(decisions['K']), 'errors': []}

    # P = Keep Parent, delete Child
    print(f"\n=== Applying P decisions (keep parent, delete child): {len(decisions['P'])} ===")
    for item in decisions['P']:
        child = item['child']
        if delete_node(tree, child):
            stats['P'] += 1
            if not dry_run:
                print(f"  ✓ Deleted {child}")
        else:
            stats['errors'].append(f"P: Could not find {child}")
            print(f"  ✗ Could not find {child}")

    # C = Keep Child, delete Parent (promote child)
    print(f"\n=== Applying C decisions (keep child, delete parent): {len(decisions['C'])} ===")
    for item in decisions['C']:
        parent = item['parent']
        if delete_and_promote(tree, parent):
            stats['C'] += 1
            if not dry_run:
                print(f"  ✓ Deleted {parent}, promoted {item['child']}")
        else:
            stats['errors'].append(f"C: Could not find {parent}")
            print(f"  ✗ Could not find {parent}")

    # X = Delete Both
    print(f"\n=== Applying X decisions (delete both): {len(decisions['X'])} ===")
    for item in decisions['X']:
        parent = item['parent']
        # Delete parent (which includes the child)
        if delete_node(tree, parent):
            stats['X'] += 1
            if not dry_run:
                print(f"  ✓ Deleted {parent} and {item['child']}")
        else:
            stats['errors'].append(f"X: Could not find {parent}")
            print(f"  ✗ Could not find {parent}")

    # K = Keep both (just report)
    print(f"\n=== K decisions (keeping both, need children later): {len(decisions['K'])} ===")
    for item in decisions['K']:
        print(f"  → {item['parent']} (has child: {item['child']})")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Apply single-child decisions')
    parser.add_argument('--input', default='concept_packs/first-light/hierarchy/hierarchy_tree_v10.json')
    parser.add_argument('--output', default='concept_packs/first-light/hierarchy/hierarchy_tree_v11.json')
    parser.add_argument('--review-file', default='scripts/ontology/single_child_review.txt')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    args = parser.parse_args()

    # Parse review file
    print(f"Parsing {args.review_file}...")
    decisions = parse_review_file(args.review_file)

    print(f"\nDecisions found:")
    print(f"  P (keep parent): {len(decisions['P'])}")
    print(f"  C (keep child): {len(decisions['C'])}")
    print(f"  X (delete both): {len(decisions['X'])}")
    print(f"  K (keep both): {len(decisions['K'])}")

    # Load tree
    with open(args.input) as f:
        tree = json.load(f)
    print(f"\nLoaded tree from {args.input}")

    # Apply decisions
    stats = apply_decisions(tree, decisions, dry_run=args.dry_run)

    # Clean up
    tree = clean_elevated_prefix(tree)
    print("\n✓ Cleaned ELEVATED: markers")

    tree = fix_depths(tree)
    print("✓ Fixed depth values")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  P applied: {stats['P']}/{len(decisions['P'])}")
    print(f"  C applied: {stats['C']}/{len(decisions['C'])}")
    print(f"  X applied: {stats['X']}/{len(decisions['X'])}")
    print(f"  K skipped: {stats['K']} (need children)")

    if stats['errors']:
        print(f"\n  Errors: {len(stats['errors'])}")
        for err in stats['errors']:
            print(f"    - {err}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would write to: {args.output}")
    else:
        with open(args.output, 'w') as f:
            json.dump(tree, f, indent=2)
        print(f"\n✓ Wrote to {args.output}")

    # Output K decisions for next step
    if decisions['K']:
        k_file = Path(args.review_file).parent / 'keep_both_nodes.json'
        with open(k_file, 'w') as f:
            json.dump(decisions['K'], f, indent=2)
        print(f"✓ Wrote K decisions to {k_file} for child generation")


if __name__ == '__main__':
    main()
