#!/usr/bin/env python3
"""
Apply hierarchy flattening melds - Round 4.

Operations:
1. Delete NaturalWaterBodies, promote children
2. Merge BodyOfWater into SaltWaterArea
3. Merge StaticWaterArea into FreshWaterArea
4. Move children of Ocean into WaterZones
5. Move Swamp up a level
6. Move AlcoholicBeverage under Beverage
7. Move Wine, Beer, Cider, Sake under AlcoholicBeverage
8. Delete children of CommunicationSatellite
9. Delete children of DataDisplayDevice
10. Promote Transducer under Machines
11. Promote children of SmallKitchenAppliance under FoodPrepAppliances
12. Promote Loan under FinancialAccount
13. Delete BodyPart and promote children
14. Promote Nutrient under BiologicalSubstance
15. Delete WoodProduct and promote children
16. Promote Seafood to same level as Meat
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
        parent[key] = count
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


def move_node_to_target(tree, source, dest):
    """Move source node to be a child of dest."""
    source_result = find_node(tree, source)
    dest_result = find_node(tree, dest)

    if not source_result:
        return False, f"Source '{source}' not found"
    if not dest_result:
        return False, f"Destination '{dest}' not found"

    source_parent, source_key, source_value = source_result
    dest_parent, dest_key, dest_value = dest_result

    if not isinstance(dest_value, dict):
        dest_value = {}
        dest_parent[dest_key] = dest_value

    clean_source = strip_markers(source_key)
    new_key = f"ELEVATED:{clean_source}"
    dest_value[new_key] = source_value

    del source_parent[source_key]
    return True, None


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


def merge_into(tree, source, dest):
    """Merge source's children into dest, then delete source."""
    source_result = find_node(tree, source)
    dest_result = find_node(tree, dest)

    if not source_result:
        return False, f"Source '{source}' not found"
    if not dest_result:
        return False, f"Destination '{dest}' not found"

    source_parent, source_key, source_value = source_result
    dest_parent, dest_key, dest_value = dest_result

    if not isinstance(dest_value, dict):
        dest_value = {}
        dest_parent[dest_key] = dest_value

    # Move source's children to dest
    if isinstance(source_value, dict):
        for child_k, child_v in source_value.items():
            clean_child = strip_markers(child_k)
            new_key = f"ELEVATED:{clean_child}"
            dest_value[new_key] = child_v

    # Delete source
    del source_parent[source_key]
    return True, None


def move_children_to_target(tree, source, dest):
    """Move children of source to dest, then delete source."""
    source_result = find_node(tree, source)
    dest_result = find_node(tree, dest)

    if not source_result:
        return False, f"Source '{source}' not found"
    if not dest_result:
        return False, f"Destination '{dest}' not found"

    source_parent, source_key, source_value = source_result
    dest_parent, dest_key, dest_value = dest_result

    if not isinstance(dest_value, dict):
        dest_value = {}
        dest_parent[dest_key] = dest_value

    if isinstance(source_value, dict):
        for child_k, child_v in source_value.items():
            clean_child = strip_markers(child_k)
            new_key = f"ELEVATED:{clean_child}"
            dest_value[new_key] = child_v

    del source_parent[source_key]
    return True, None


def move_up_one_level(tree, target):
    """Move target up to its grandparent level."""
    # Find target and its parent chain
    def find_with_parent(node, target, parent=None, grandparent=None):
        if not isinstance(node, dict):
            return None
        for k, v in node.items():
            clean_k = strip_markers(k)
            if clean_k == target:
                return (node, k, v, parent, grandparent)
            if isinstance(v, dict):
                result = find_with_parent(v, target, node, parent)
                if result:
                    return result
        return None

    result = find_with_parent(tree, target)
    if not result:
        return False, f"Target '{target}' not found"

    current_parent, key, value, grandparent, _ = result

    if grandparent is None:
        return False, f"Target '{target}' has no grandparent to move to"

    # Move to grandparent
    clean_key = strip_markers(key)
    new_key = f"ELEVATED:{clean_key}"
    grandparent[new_key] = value

    # Remove from current location
    del current_parent[key]
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
    """Apply all round 4 flattening melds."""

    print("=== Round 4 Hierarchy Flattening ===\n")

    # 1. Delete NaturalWaterBodies, promote children
    if delete_and_promote(tree, 'NaturalWaterBodies'):
        print("✓ 1. Deleted NaturalWaterBodies, promoted children")
    else:
        print("✗ 1. Could not find NaturalWaterBodies")

    # 2. Merge BodyOfWater into SaltWaterArea
    ok, err = merge_into(tree, 'BodyOfWater', 'SaltWaterArea')
    if ok:
        print("✓ 2. Merged BodyOfWater into SaltWaterArea")
    else:
        print(f"✗ 2. {err}")

    # 3. Merge StaticWaterArea into FreshWaterArea
    ok, err = merge_into(tree, 'StaticWaterArea', 'FreshWaterArea')
    if ok:
        print("✓ 3. Merged StaticWaterArea into FreshWaterArea")
    else:
        print(f"✗ 3. {err}")

    # 4. Move children of Ocean into WaterZones
    ok, err = move_children_to_target(tree, 'Ocean', 'WaterZones')
    if ok:
        print("✓ 4. Moved Ocean children into WaterZones")
    else:
        print(f"✗ 4. {err}")

    # 5. Move Swamp up a level
    ok, err = move_up_one_level(tree, 'Swamp')
    if ok:
        print("✓ 5. Moved Swamp up a level")
    else:
        print(f"✗ 5. {err}")

    # 6. Move AlcoholicBeverage under Beverage
    ok, err = move_node_to_target(tree, 'AlcoholicBeverage', 'Beverage')
    if ok:
        print("✓ 6. Moved AlcoholicBeverage under Beverage")
    else:
        print(f"✗ 6. {err}")

    # 7. Move Wine, Beer, Cider, Sake under AlcoholicBeverage
    drinks = ['Wine', 'Beer', 'Cider', 'Sake']
    for drink in drinks:
        ok, err = move_node_to_target(tree, drink, 'AlcoholicBeverage')
        if ok:
            print(f"✓ 7. Moved {drink} under AlcoholicBeverage")
        else:
            print(f"✗ 7. {drink}: {err}")

    # 8. Delete children of CommunicationSatellite
    ok, count = delete_children(tree, 'CommunicationSatellite')
    if ok:
        print(f"✓ 8. Deleted {count} children of CommunicationSatellite")
    else:
        print("✗ 8. Could not find CommunicationSatellite")

    # 9. Delete children of DataDisplayDevice
    ok, count = delete_children(tree, 'DataDisplayDevice')
    if ok:
        print(f"✓ 9. Deleted {count} children of DataDisplayDevice")
    else:
        print("✗ 9. Could not find DataDisplayDevice")

    # 10. Promote Transducer under Machines
    ok, err = move_node_to_target(tree, 'Transducer', 'Machines')
    if ok:
        print("✓ 10. Moved Transducer under Machines")
    else:
        print(f"✗ 10. {err}")

    # 11. Promote children of SmallKichenAppliance under FoodPrepAppliances (note: typo in source)
    ok, err = move_children_to_target(tree, 'SmallKichenAppliance', 'FoodPrepAppliances')
    if ok:
        print("✓ 11. Promoted SmallKichenAppliance children under FoodPrepAppliances")
    else:
        print(f"✗ 11. {err}")

    # 12. Promote Loan under FinancialAccount
    ok, err = move_node_to_target(tree, 'Loan', 'FinancialAccount')
    if ok:
        print("✓ 12. Moved Loan under FinancialAccount")
    else:
        print(f"✗ 12. {err}")

    # 13. Delete BodyPart and promote children
    if delete_and_promote(tree, 'BodyPart'):
        print("✓ 13. Deleted BodyPart, promoted children")
    else:
        print("✗ 13. Could not find BodyPart")

    # 14. Promote Nutrient under BiologicalSubstance
    ok, err = move_node_to_target(tree, 'Nutrient', 'BiologicalSubstance')
    if ok:
        print("✓ 14. Moved Nutrient under BiologicalSubstance")
    else:
        print(f"✗ 14. {err}")

    # 15. Delete WoodProduct and promote children
    if delete_and_promote(tree, 'WoodProduct'):
        print("✓ 15. Deleted WoodProduct, promoted children")
    else:
        print("✗ 15. Could not find WoodProduct")

    # 16. Promote Seafood to same level as Meat
    ok, err = move_node_to_sibling_of(tree, 'Seafood', 'Meat')
    if ok:
        print("✓ 16. Moved Seafood to same level as Meat")
    else:
        print(f"✗ 16. {err}")

    return tree


def main():
    parser = argparse.ArgumentParser(description='Flatten hierarchy - Round 4')
    parser.add_argument('--input', default='concept_packs/first-light/hierarchy/hierarchy_tree_v8.json')
    parser.add_argument('--output', default='concept_packs/first-light/hierarchy/hierarchy_tree_v9.json')
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
