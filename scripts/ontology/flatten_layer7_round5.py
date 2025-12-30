#!/usr/bin/env python3
"""
Apply hierarchy flattening melds - Round 5 (final layer 7 cleanup).

Operations:
1. Promote SemiconductorComponent and Terminal to same level as ElectricalComponent
2. Delete ComputingElectronics and DigitalDataStorageDevice intermediate nodes
3. Delete RefrigerationAppliance and promote children
4. Delete AppraisalAsExpected
5. Rename LiabilityAccount to CreditAccount and delete CreditCardAccount
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


def delete_node(tree, target):
    """Delete a node entirely (no promotion of children)."""
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


def move_node_to_sibling_of(tree, source, sibling):
    """Move source node to be a sibling of the specified node."""
    source_result = find_node(tree, source)
    sibling_result = find_node(tree, sibling)

    if not source_result:
        return False, f"Source '{source}' not found"
    if not sibling_result:
        return False, f"Sibling '{sibling}' not found"

    source_parent, source_key, source_value = source_result
    sibling_parent, sibling_key, sibling_value = sibling_result

    clean_source = strip_markers(source_key)
    new_key = f"ELEVATED:{clean_source}"
    sibling_parent[new_key] = source_value

    del source_parent[source_key]
    return True, None


def rename_node(tree, old_name, new_name):
    """Rename a node."""
    result = find_node(tree, old_name)
    if not result:
        return False, f"Node '{old_name}' not found"

    parent, key, value = result
    del parent[key]
    parent[new_name] = value
    return True, None


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


def apply_melds(tree):
    """Apply all round 5 flattening melds."""

    print("=== Round 5 Hierarchy Flattening (Final Layer 7 Cleanup) ===\n")

    # 1. Promote SemiconductorComponent and Terminal to same level as ElectricalComponent
    ok, err = move_node_to_sibling_of(tree, 'SemiconductorComponent', 'ElectricalComponent')
    if ok:
        print("✓ 1a. Moved SemiconductorComponent to same level as ElectricalComponent")
    else:
        print(f"✗ 1a. {err}")

    ok, err = move_node_to_sibling_of(tree, 'Terminal', 'ElectricalComponent')
    if ok:
        print("✓ 1b. Moved Terminal to same level as ElectricalComponent")
    else:
        print(f"✗ 1b. {err}")

    # 2. Delete ComputingElectronics - promote DigitalDataStorageDevice
    #    Then delete DigitalDataStorageDevice intermediate containers
    if delete_and_promote(tree, 'ComputingElectronics'):
        print("✓ 2a. Deleted ComputingElectronics, promoted children")
    else:
        print("✗ 2a. Could not find ComputingElectronics")

    # Delete intermediate storage device containers
    storage_containers = [
        'DigitalDataStorageDevice',
        'ComputerDisk',
        'InternalDigitalDataStorageDevice',
        'OpticalDisc',
        'ReadOnlyMemoryDataStorage',
        'RemovableDigitalDataStorageDevice',
        'RewritableDataStorage'
    ]
    for container in storage_containers:
        if delete_and_promote(tree, container):
            print(f"✓ 2. Deleted {container}, promoted children")
        else:
            print(f"✗ 2. Could not find {container}")

    # 3. Delete RefrigerationAppliance and promote children
    if delete_and_promote(tree, 'RefrigerationAppliance'):
        print("✓ 3. Deleted RefrigerationAppliance, promoted children")
    else:
        print("✗ 3. Could not find RefrigerationAppliance")

    # 4. Delete AppraisalAsExpected
    if delete_node(tree, 'AppraisalAsExpected'):
        print("✓ 4. Deleted AppraisalAsExpected")
    else:
        print("✗ 4. Could not find AppraisalAsExpected")

    # 5. Replace LiabilityAccount with CreditAccount and delete CreditCardAccount
    # First delete CreditCardAccount
    if delete_node(tree, 'CreditCardAccount'):
        print("✓ 5a. Deleted CreditCardAccount")
    else:
        print("✗ 5a. Could not find CreditCardAccount")

    # Delete CreditAccount (intermediate node)
    if delete_and_promote(tree, 'CreditAccount'):
        print("✓ 5b. Deleted CreditAccount")
    else:
        print("✗ 5b. Could not find CreditAccount")

    # Rename LiabilityAccount to CreditAccount
    ok, err = rename_node(tree, 'LiabilityAccount', 'CreditAccount')
    if ok:
        print("✓ 5c. Renamed LiabilityAccount to CreditAccount")
    else:
        print(f"✗ 5c. {err}")

    return tree


def main():
    parser = argparse.ArgumentParser(description='Flatten hierarchy - Round 5')
    parser.add_argument('--input', default='concept_packs/first-light/hierarchy/hierarchy_tree_v9.json')
    parser.add_argument('--output', default='concept_packs/first-light/hierarchy/hierarchy_tree_v10.json')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without writing')
    args = parser.parse_args()

    with open(args.input) as f:
        tree = json.load(f)

    print(f"Loaded tree from {args.input}\n")

    tree = apply_melds(tree)

    tree = clean_elevated_prefix(tree)
    print("\n✓ Cleaned ELEVATED: markers")

    tree = fix_depths(tree)
    print("✓ Fixed depth values")

    if args.dry_run:
        print("\n[DRY RUN] Would write to:", args.output)
    else:
        with open(args.output, 'w') as f:
            json.dump(tree, f, indent=2)
        print(f"\n✓ Wrote flattened tree to {args.output}")


if __name__ == '__main__':
    main()
