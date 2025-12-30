#!/usr/bin/env python3
"""
Generate meld operations for hierarchy restructuring.

This script analyzes the changes between the original layer files and the
restructured hierarchy tree, then generates all operations needed:
1. CREATE_MELD - new intermediate nodes need melds from children
2. UPDATE_MELD - existing parent nodes whose children changed need re-melding
3. REMOVE - nodes removed from tree (orphaned probes)

A meld combines the probes of child concepts to create a parent probe.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def strip_prefix(name: str) -> tuple[str, str | None]:
    """Strip NEW:/MOVED:/ORPHAN: prefix from name."""
    for p in ['NEW:', 'MOVED:', 'ORPHAN:']:
        if name.startswith(p):
            return name[len(p):], p.rstrip(':')
    return name, None


def get_tree_structure(node: dict, parent_name: str = None, results: dict = None) -> dict:
    """Extract all nodes with their children and prefix status."""
    if results is None:
        results = {}
    if not isinstance(node, dict):
        return results

    for k, v in node.items():
        clean_name, prefix = strip_prefix(k)
        children = []
        if isinstance(v, dict):
            children = [strip_prefix(ck)[0] for ck in v.keys()]
            get_tree_structure(v, clean_name, results)

        results[clean_name] = {
            'prefix': prefix,
            'parent': parent_name,
            'children': children,
            'is_leaf': not isinstance(v, dict)
        }
    return results


def load_layer_concepts(lens_pack_dir: Path) -> dict:
    """Load all concepts from layer files."""
    concepts = {}
    for layer in range(5):
        layer_file = lens_pack_dir / f'layer{layer}' / f'layer{layer}_concepts.json'
        if layer_file.exists():
            with open(layer_file) as f:
                data = json.load(f)
            for c in data['concepts']:
                concepts[c['name']] = {
                    'layer': layer,
                    'is_leaf': c.get('is_leaf', False),
                    'parent': c.get('parent'),
                    'children_count': c.get('children_count', 0),
                    'concept_key': c.get('concept_key')
                }
    return concepts


def find_meldable_children(node_name: str, tree_structure: dict, layer_concepts: dict) -> list[str]:
    """
    Find children that can be melded (have existing probes).
    Recursively descends if a child is also a NEW node without a probe.
    """
    node = tree_structure.get(node_name, {})
    meldable = []

    for child_name in node.get('children', []):
        child_info = tree_structure.get(child_name, {})

        # If child has a probe, we can use it
        if child_name in layer_concepts:
            meldable.append(child_name)
        # If child is a NEW node without probe, recurse to find its meldable children
        elif child_info.get('prefix') == 'NEW' and child_name not in layer_concepts:
            meldable.extend(find_meldable_children(child_name, tree_structure, layer_concepts))
        # MOVED nodes should have probes
        elif child_info.get('prefix') == 'MOVED' and child_name in layer_concepts:
            meldable.append(child_name)

    return meldable


def get_original_children(concept_name: str, layer_concepts: dict) -> set[str]:
    """Get the original children of a concept from layer files."""
    children = set()
    for name, info in layer_concepts.items():
        # Parent field format is "ParentName:layer"
        parent_field = info.get('parent', '')
        if parent_field:
            parent_name = parent_field.split(':')[0] if ':' in parent_field else parent_field
            if parent_name == concept_name:
                children.add(name)
    return children


def generate_all_operations(lens_pack_dir: Path) -> dict:
    """Generate all meld operations needed for the restructured hierarchy."""

    # Load data
    tree_file = lens_pack_dir / 'hierarchy_tree_v4.json'
    with open(tree_file) as f:
        tree = json.load(f)

    layer_concepts = load_layer_concepts(lens_pack_dir)
    tree_structure = get_tree_structure(tree)

    operations = {
        'create_meld': [],   # New nodes that need melds created
        'update_meld': [],   # Existing nodes whose children changed
        'remove': [],        # Nodes removed from tree
    }

    # Track which parents need updates
    affected_parents = set()

    # 1. Find NEW nodes that need melds created
    for name, info in tree_structure.items():
        if info['prefix'] == 'NEW' and name not in layer_concepts:
            meldable_children = find_meldable_children(name, tree_structure, layer_concepts)

            if meldable_children:
                child_layers = [layer_concepts[c]['layer'] for c in meldable_children if c in layer_concepts]
                new_layer = max(0, min(child_layers) - 1) if child_layers else 2

                operations['create_meld'].append({
                    'operation': 'create_meld',
                    'target': name,
                    'target_layer': new_layer,
                    'source_concepts': meldable_children,
                    'source_count': len(meldable_children),
                    'new_parent': info['parent']
                })

    # 2. Find MOVED nodes - their old and new parents need updates
    for name, info in tree_structure.items():
        if info['prefix'] == 'MOVED' and name in layer_concepts:
            old_parent_field = layer_concepts[name].get('parent', '')
            old_parent = old_parent_field.split(':')[0] if old_parent_field and ':' in old_parent_field else old_parent_field
            new_parent = info['parent']

            if old_parent != new_parent:
                # Track both old and new parents as needing updates
                if old_parent and old_parent in layer_concepts:
                    affected_parents.add(old_parent)
                if new_parent and new_parent in layer_concepts:
                    affected_parents.add(new_parent)

                operations['update_meld'].append({
                    'operation': 'move',
                    'target': name,
                    'old_parent': old_parent,
                    'new_parent': new_parent,
                    'layer': layer_concepts[name]['layer']
                })

    # 3. Find removed nodes (in layer files but not in tree)
    tree_names = set(tree_structure.keys())
    for name, info in layer_concepts.items():
        if name not in tree_names:
            operations['remove'].append({
                'operation': 'remove',
                'target': name,
                'layer': info['layer'],
                'was_leaf': info['is_leaf'],
                'old_parent': info.get('parent', '').split(':')[0] if info.get('parent') else None
            })

            # Old parent needs update
            old_parent = info.get('parent', '').split(':')[0] if info.get('parent') else None
            if old_parent and old_parent in layer_concepts:
                affected_parents.add(old_parent)

    # 4. Generate update operations for affected existing parents
    parent_updates = []
    for parent_name in affected_parents:
        if parent_name in tree_structure:
            # Get new children from tree
            new_children = set(tree_structure[parent_name].get('children', []))
            # Get old children from layer files
            old_children = get_original_children(parent_name, layer_concepts)

            if new_children != old_children:
                added = new_children - old_children
                removed = old_children - new_children

                parent_updates.append({
                    'operation': 'update_parent_meld',
                    'target': parent_name,
                    'layer': layer_concepts[parent_name]['layer'],
                    'old_children_count': len(old_children),
                    'new_children_count': len(new_children),
                    'children_added': list(added),
                    'children_removed': list(removed),
                    'new_meld_sources': find_meldable_children(parent_name, tree_structure, layer_concepts)
                })

    operations['update_meld'].extend(parent_updates)

    # Sort create operations by layer (bottom-up)
    operations['create_meld'].sort(key=lambda x: -x['target_layer'])

    return operations


def main():
    parser = argparse.ArgumentParser(description='Generate meld operations for hierarchy restructuring')
    parser.add_argument('--lens-pack', type=str, default='lens_packs/apertus-8b_first-light-v1',
                        help='Path to lens pack directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for meld operations (default: <lens-pack>/meld_operations.json)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary only')

    args = parser.parse_args()

    lens_pack_dir = Path(args.lens_pack)

    if not lens_pack_dir.exists():
        print(f"Error: Lens pack directory not found: {lens_pack_dir}")
        return 1

    print(f"Analyzing hierarchy changes in: {lens_pack_dir}")

    operations = generate_all_operations(lens_pack_dir)

    # Summary
    print(f"\n{'='*60}")
    print("HIERARCHY OPERATIONS SUMMARY")
    print(f"{'='*60}")

    print(f"\n## CREATE_MELD (new intermediate nodes)")
    print(f"   {len(operations['create_meld'])} new nodes need melds created")

    # Group creates by layer
    creates_by_layer = defaultdict(list)
    for op in operations['create_meld']:
        creates_by_layer[op['target_layer']].append(op['target'])
    for layer in sorted(creates_by_layer.keys()):
        print(f"     Layer {layer}: {len(creates_by_layer[layer])} melds")

    print(f"\n## UPDATE_MELD (existing nodes with changed children)")
    moves = [op for op in operations['update_meld'] if op['operation'] == 'move']
    parent_updates = [op for op in operations['update_meld'] if op['operation'] == 'update_parent_meld']
    print(f"   {len(moves)} nodes moved to new parents")
    print(f"   {len(parent_updates)} existing parents need re-melding")

    print(f"\n## REMOVE (orphaned probes)")
    print(f"   {len(operations['remove'])} nodes removed from tree")
    leaves_removed = sum(1 for op in operations['remove'] if op['was_leaf'])
    parents_removed = len(operations['remove']) - leaves_removed
    print(f"     {leaves_removed} leaves, {parents_removed} parents")

    if not args.summary:
        # Show some examples
        print(f"\n{'='*60}")
        print("SAMPLE OPERATIONS")
        print(f"{'='*60}")

        print("\n## Sample CREATE_MELD operations:")
        for op in operations['create_meld'][:10]:
            print(f"\n  {op['target']} (layer {op['target_layer']}):")
            print(f"    Parent: {op['new_parent']}")
            print(f"    Meld from {op['source_count']} concepts: {op['source_concepts'][:3]}...")

        print("\n## Sample MOVE operations:")
        for op in moves[:10]:
            print(f"  {op['target']}: {op['old_parent']} -> {op['new_parent']}")

        print("\n## Sample PARENT UPDATE operations:")
        for op in parent_updates[:10]:
            print(f"\n  {op['target']} (layer {op['layer']}):")
            print(f"    Children: {op['old_children_count']} -> {op['new_children_count']}")
            if op['children_added']:
                print(f"    Added: {op['children_added'][:3]}...")
            if op['children_removed']:
                print(f"    Removed: {op['children_removed'][:3]}...")

        print("\n## REMOVED nodes:")
        for op in operations['remove']:
            leaf_str = "leaf" if op['was_leaf'] else "parent"
            print(f"  {op['target']} (layer {op['layer']}, {leaf_str})")

    # Save full report
    output_file = Path(args.output) if args.output else lens_pack_dir / 'meld_operations.json'

    # Convert to serializable format
    result = {
        'summary': {
            'create_meld': len(operations['create_meld']),
            'moves': len(moves),
            'parent_updates': len(parent_updates),
            'removed': len(operations['remove']),
        },
        'operations': operations
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n\nFull operations saved to: {output_file}")

    return 0


if __name__ == '__main__':
    exit(main())
