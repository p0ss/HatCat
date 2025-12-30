#!/usr/bin/env python3
"""
Apply hierarchy flattening melds to eliminate layer 7.

Operations:
1. Remove OilFromPlant - bring TeaTreeOil and VegetableOil up under EdibleOil
2. Delete WheatGrain - bring wheat varieties up under CerealGrain
3. Delete FruitOrVegetable - place children directly under FoodIngredient
4. Delete EdibleTuber - move tubers under RootVegetable
5. Delete Oil (chemical) - place PetroleumProduct under CompoundSubstance
6. Delete ElementsAndCompounds - move CompoundSubstance/ElementalSubstance to Substance
7. Move TimeMeasure up directly under Quantity
8. Delete EmotionalVoiceUtterances - move children under EmotionalSpeakingBehavior
9. Delete ContentDevelopment - place children under CreativeActivities
"""

import json
import argparse
from pathlib import Path
from copy import deepcopy


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


def delete_and_promote(tree, target):
    """Delete target node and promote its children to parent level."""
    result = find_node(tree, target)
    if not result:
        return False

    parent, key, value = result

    if isinstance(value, dict):
        # Promote children to parent level
        for child_k, child_v in value.items():
            clean_child = strip_markers(child_k)
            new_key = f"ELEVATED:{clean_child}"
            parent[new_key] = child_v

    del parent[key]
    return True


def move_children_to_target(tree, source, dest):
    """Move children of source node to dest node, then delete source."""
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

    if not isinstance(source_value, dict):
        print(f"  Source '{source}' has no children")
        return False

    if not isinstance(dest_value, dict):
        # Convert to dict if it's a number (leaf node)
        dest_value = {}
        dest_parent[dest_key] = dest_value

    # Move children to destination
    for child_k, child_v in source_value.items():
        clean_child = strip_markers(child_k)
        new_key = f"ELEVATED:{clean_child}"
        dest_value[new_key] = child_v

    # Delete source
    del source_parent[source_key]
    return True


def move_node_to_target(tree, source, dest):
    """Move source node (not just children) to be a child of dest."""
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
        # Convert to dict if it's a number (leaf node)
        dest_value = {}
        dest_parent[dest_key] = dest_value

    # Move the whole node to destination
    clean_source = strip_markers(source_key)
    new_key = f"ELEVATED:{clean_source}"
    dest_value[new_key] = source_value

    # Delete from original location
    del source_parent[source_key]
    return True


def apply_melds(tree):
    """Apply all flattening melds."""
    # Simple delete_promote operations
    simple_ops = [
        ('OilFromPlant', 'Remove OilFromPlant, promote children to EdibleOil'),
        ('WheatGrain', 'Remove WheatGrain, promote wheats to CerealGrain'),
        ('FruitOrVegetable', 'Remove FruitOrVegetable, promote to FoodIngredient'),
        ('EdibleTuber', 'Remove EdibleTuber, promote to RootVegetable'),
        ('Oil', 'Remove Oil, promote PetroleumProduct to CompoundSubstance'),
        ('EmotionalVoiceUtterances', 'Remove EmotionalVoiceUtterances'),
        ('ContentDevelopment', 'Remove ContentDevelopment'),
    ]

    for target, desc in simple_ops:
        if delete_and_promote(tree, target):
            print(f"✓ {desc}")
        else:
            print(f"✗ Could not find {target}")

    # Special operation 6: Move ElementsAndCompounds children to Substance
    # (not ChemicalSubstance where they currently are)
    print("\n--- Special operations ---")
    if move_children_to_target(tree, 'ElementsAndCompounds', 'Substance'):
        print("✓ Moved CompoundSubstance/ElementalSubstance to Substance")
    else:
        print("✗ Failed to move ElementsAndCompounds children to Substance")

    # Special operation 7: Move TimeMeasure up to Quantity
    if move_node_to_target(tree, 'TimeMeasure', 'Quantity'):
        print("✓ Moved TimeMeasure directly under Quantity")
    else:
        print("✗ Failed to move TimeMeasure to Quantity")

    return tree


def clean_elevated_prefix(node):
    """Remove ELEVATED: prefix from keys (final cleanup)."""
    if not isinstance(node, dict):
        return node

    new_node = {}
    for k, v in node.items():
        clean_k = k.replace('ELEVATED:', '')
        new_node[clean_k] = clean_elevated_prefix(v)

    return new_node


def main():
    parser = argparse.ArgumentParser(description='Flatten hierarchy to eliminate layer 7')
    parser.add_argument('--input', default='concept_packs/first-light/hierarchy/hierarchy_tree_v5.json')
    parser.add_argument('--output', default='concept_packs/first-light/hierarchy/hierarchy_tree_v6.json')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without writing')
    parser.add_argument('--keep-markers', action='store_true', help='Keep ELEVATED: markers')
    args = parser.parse_args()

    with open(args.input) as f:
        tree = json.load(f)

    print(f"Loaded tree from {args.input}")

    # Apply melds
    tree = apply_melds(tree)

    # Clean markers unless keeping
    if not args.keep_markers:
        tree = clean_elevated_prefix(tree)
        print("✓ Cleaned ELEVATED: markers")

    if args.dry_run:
        print("\n[DRY RUN] Would write to:", args.output)
    else:
        with open(args.output, 'w') as f:
            json.dump(tree, f, indent=2)
        print(f"\n✓ Wrote flattened tree to {args.output}")


if __name__ == '__main__':
    main()
