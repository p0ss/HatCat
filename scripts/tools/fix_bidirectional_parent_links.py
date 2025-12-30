#!/usr/bin/env python3
"""
Fix bidirectional parent links in the concept hierarchy.

Problem: Parent concepts have 'category_children' arrays, but child concepts
don't have their 'parent' field set, breaking upward traversal.

Solution: Rebuild all 'parent' fields from 'category_children' relationships.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
LAYER_DIR = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers"


def fix_parent_links():
    """Fix parent links across all layers."""

    print("="*80)
    print("FIXING BIDIRECTIONAL PARENT LINKS")
    print("="*80)

    # Load all layers
    all_layers = {}
    for layer_num in range(7):
        layer_file = LAYER_DIR / f"layer{layer_num}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            all_layers[layer_num] = json.load(f)

        print(f"✓ Loaded Layer {layer_num}: {len(all_layers[layer_num]['concepts'])} concepts")

    # Build child → parent mapping from category_children
    child_to_parent = {}

    print("\nBuilding parent mappings from category_children...")
    for layer_num, layer_data in all_layers.items():
        for concept in layer_data['concepts']:
            parent_name = concept['sumo_term']
            children = concept.get('category_children', [])

            for child_name in children:
                if child_name in child_to_parent:
                    print(f"  ⚠️  {child_name} has multiple parents: {child_to_parent[child_name]} and {parent_name}")
                child_to_parent[child_name] = parent_name

    print(f"✓ Built mappings for {len(child_to_parent)} child concepts")

    # Apply parent links
    fixes_by_layer = {}
    print("\nApplying parent links...")

    for layer_num, layer_data in all_layers.items():
        fixes = 0
        for concept in layer_data['concepts']:
            concept_name = concept['sumo_term']

            # Layer 0 concepts should have no parent
            if layer_num == 0:
                if concept.get('parent') and concept.get('parent') != 'None':
                    print(f"  Layer 0 concept {concept_name} has parent {concept['parent']}, removing")
                    concept['parent'] = None
                    fixes += 1
                continue

            # Other layers: set parent from mapping
            expected_parent = child_to_parent.get(concept_name)
            current_parent = concept.get('parent')

            # Normalize empty/None values
            if current_parent in [None, '', 'None']:
                current_parent = None

            if expected_parent != current_parent:
                old_parent = current_parent or 'NONE'
                new_parent = expected_parent or 'NONE'
                concept['parent'] = expected_parent
                fixes += 1

        fixes_by_layer[layer_num] = fixes
        print(f"  Layer {layer_num}: {fixes} fixes")

    # Backup and save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for layer_num, layer_data in all_layers.items():
        if fixes_by_layer[layer_num] == 0:
            continue

        layer_file = LAYER_DIR / f"layer{layer_num}.json"
        backup_dir = LAYER_DIR / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_file = backup_dir / f"layer{layer_num}_pre_parent_fix_{timestamp}.json"

        # Backup
        with open(layer_file) as f:
            original = f.read()
        with open(backup_file, 'w') as f:
            f.write(original)

        # Save fixed version
        with open(layer_file, 'w') as f:
            json.dump(layer_data, f, indent=2)

        print(f"  ✓ Saved Layer {layer_num} (backup: {backup_file.name})")

    # Verification
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    total_concepts = 0
    concepts_with_parents = 0
    concepts_without_parents = 0

    for layer_num, layer_data in all_layers.items():
        for concept in layer_data['concepts']:
            total_concepts += 1
            parent = concept.get('parent')

            if layer_num == 0:
                # Layer 0 should have no parents
                if parent and parent != 'None':
                    print(f"  ✗ Layer 0 concept {concept['sumo_term']} still has parent: {parent}")
            else:
                # Other layers should have parents
                if parent and parent != 'None' and parent != '':
                    concepts_with_parents += 1
                else:
                    concepts_without_parents += 1

    layer0_count = len(all_layers[0]['concepts'])
    non_layer0_count = total_concepts - layer0_count

    print(f"\nTotal concepts: {total_concepts}")
    print(f"  Layer 0: {layer0_count} (should have no parents)")
    print(f"  Layers 1+: {non_layer0_count}")
    print(f"    With parent: {concepts_with_parents}")
    print(f"    Without parent: {concepts_without_parents}")

    if concepts_without_parents > 0:
        print(f"\n⚠️  {concepts_without_parents} concepts in Layers 1+ still lack parents")
        print("These may be orphaned concepts that need manual review.")
    else:
        print("\n✓ All non-Layer0 concepts have parents!")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    total_fixes = sum(fixes_by_layer.values())
    print(f"Fixed {total_fixes} parent links across {len(all_layers)} layers")


if __name__ == '__main__':
    fix_parent_links()
