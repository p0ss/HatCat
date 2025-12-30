#!/usr/bin/env python3
"""
Apply hierarchy flattening melds - Round 2.

Operations:
1. Delete all children under ShipContainer (keep ShipContainer as leaf)
2. Move FluidContainer up under Containers
3. Delete ComputerKeyboardKey, raise children under ComputerInputButton
4. Delete all children of Cargoship (keep Cargoship as leaf)
5. Delete CorporateBond, advance JunkBond and MortgageBond under Bond
6. Delete children of CurrencyCoin and UnitedStatesDollarBill
7. Delete children of AegilopsGrass
8. Promote CerealGrass to child of Plant
9. Promote Sport to same level as Game
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
        return False

    parent, key, value = result
    if isinstance(value, dict):
        # Count children being deleted
        count = len(value)
        # Make it a leaf (use layer number or just keep as empty dict)
        parent[key] = count  # Store count as the leaf value (indicates depth)
        return True
    return False


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


def move_node_to_target(tree, source, dest):
    """Move source node to be a child of dest."""
    source_result = find_node(tree, source)
    dest_result = find_node(tree, dest)

    if not source_result:
        print(f"  Source '{source}' not found")
        return False
    if not dest_result:
        print(f"  Destination '{dest}' not found")
        return False

    source_parent, source_key, source_value = source_result
    dest_parent, dest_key, dest_value = dest_result

    if not isinstance(dest_value, dict):
        dest_value = {}
        dest_parent[dest_key] = dest_value

    clean_source = strip_markers(source_key)
    new_key = f"ELEVATED:{clean_source}"
    dest_value[new_key] = source_value

    del source_parent[source_key]
    return True


def move_node_to_sibling_of(tree, source, sibling):
    """Move source node to be a sibling of the specified node."""
    source_result = find_node(tree, source)
    sibling_result = find_node(tree, sibling)

    if not source_result:
        print(f"  Source '{source}' not found")
        return False
    if not sibling_result:
        print(f"  Sibling '{sibling}' not found")
        return False

    source_parent, source_key, source_value = source_result
    sibling_parent, sibling_key, sibling_value = sibling_result

    # Add source as sibling (to sibling's parent)
    clean_source = strip_markers(source_key)
    new_key = f"ELEVATED:{clean_source}"
    sibling_parent[new_key] = source_value

    # Delete from original location
    del source_parent[source_key]
    return True


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
    """Apply all round 2 flattening melds."""

    print("=== Round 2 Hierarchy Flattening ===\n")

    # 1. Delete all children under ShipContainer
    if delete_children(tree, 'ShipContainer'):
        print("✓ Deleted children of ShipContainer")
    else:
        print("✗ Could not find ShipContainer")

    # 2. Move FluidContainer up under Containers
    if move_node_to_target(tree, 'FluidContainer', 'Containers'):
        print("✓ Moved FluidContainer under Containers")
    else:
        print("✗ Failed to move FluidContainer")

    # 3. Delete ComputerKeyboardKey, raise children under ComputerInputButton
    if delete_and_promote(tree, 'ComputerKeyboardKey'):
        print("✓ Deleted ComputerKeyboardKey, promoted children to ComputerInputButton")
    else:
        print("✗ Could not find ComputerKeyboardKey")

    # 4. Delete all children of Cargoship
    if delete_children(tree, 'CargoShip'):
        print("✓ Deleted children of CargoShip")
    else:
        # Try alternate spelling
        if delete_children(tree, 'Cargoship'):
            print("✓ Deleted children of Cargoship")
        else:
            print("✗ Could not find CargoShip/Cargoship")

    # 5. Delete CorporateBond, advance JunkBond and MortgageBond under Bond
    if delete_and_promote(tree, 'CorporateBond'):
        print("✓ Deleted CorporateBond, promoted JunkBond/MortgageBond to Bond")
    else:
        print("✗ Could not find CorporateBond")

    # 6. Delete children of CurrencyCoin and CurrencyBill (which contains UnitedStatesDollarBill)
    if delete_children(tree, 'CurrencyCoin'):
        print("✓ Deleted children of CurrencyCoin")
    else:
        print("✗ Could not find CurrencyCoin")

    if delete_children(tree, 'CurrencyBill'):
        print("✓ Deleted children of CurrencyBill (UnitedStatesDollarBill)")
    else:
        print("✗ Could not find CurrencyBill")

    # 7. Delete children of AegilopsGrass
    if delete_children(tree, 'AegilopsGrass'):
        print("✓ Deleted children of AegilopsGrass")
    else:
        print("✗ Could not find AegilopsGrass")

    # 8. Promote CerealGrass to child of Plant
    if move_node_to_target(tree, 'CerealGrass', 'Plant'):
        print("✓ Moved CerealGrass under Plant")
    else:
        print("✗ Failed to move CerealGrass under Plant")

    # 9. Promote Sport to same level as Game
    if move_node_to_sibling_of(tree, 'Sport', 'Game'):
        print("✓ Moved Sport to same level as Game")
    else:
        print("✗ Failed to move Sport")

    return tree


def main():
    parser = argparse.ArgumentParser(description='Flatten hierarchy - Round 2')
    parser.add_argument('--input', default='concept_packs/first-light/hierarchy/hierarchy_tree_v6.json')
    parser.add_argument('--output', default='concept_packs/first-light/hierarchy/hierarchy_tree_v7.json')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without writing')
    parser.add_argument('--keep-markers', action='store_true', help='Keep ELEVATED: markers')
    args = parser.parse_args()

    with open(args.input) as f:
        tree = json.load(f)

    print(f"Loaded tree from {args.input}\n")

    tree = apply_melds(tree)

    if not args.keep_markers:
        tree = clean_elevated_prefix(tree)
        print("\n✓ Cleaned ELEVATED: markers")

    if args.dry_run:
        print("\n[DRY RUN] Would write to:", args.output)
    else:
        with open(args.output, 'w') as f:
            json.dump(tree, f, indent=2)
        print(f"\n✓ Wrote flattened tree to {args.output}")


if __name__ == '__main__':
    main()
