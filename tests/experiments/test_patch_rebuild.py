#!/usr/bin/env python3
"""
Test script to rebuild layers with patches and compare to known-good layers.

This script:
1. Loads SUMO KIF files + patch files
2. Rebuilds the layer structure
3. Compares to existing layer files
4. Reports any differences
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

def load_sumo_with_patches():
    """Load SUMO KIF files plus patches.

    IMPORTANT: Patches OVERRIDE SUMO parents, they don't add to them.
    If a concept appears in a patch file, we ignore its SUMO parent entirely.
    """
    kif_dir = Path('data/concept_graph/sumo_source')
    patch_dir = Path('data/concept_graph/sumo_patches')

    # Maps: child -> set of parents
    sumo_parents = defaultdict(set)
    patch_parents = defaultdict(set)

    # Track which concepts have patches (to know what to override)
    patched_concepts = set()

    print("Loading SUMO KIF files...")
    for kif_file in kif_dir.glob('*.kif'):
        with open(kif_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('(subclass '):
                    parts = line.replace('(subclass ', '').replace(')', '').split()
                    if len(parts) >= 2:
                        child, parent = parts[0], parts[1]
                        sumo_parents[child].add(parent)
                elif line.startswith('(instance '):
                    parts = line.replace('(instance ', '').replace(')', '').split()
                    if len(parts) >= 2:
                        child, parent = parts[0], parts[1]
                        sumo_parents[child].add(parent)

    sumo_count = sum(len(parents) for parents in sumo_parents.values())
    print(f"  Loaded {sumo_count} SUMO relationships")

    print("Loading patch files (patches OVERRIDE SUMO)...")
    for patch_file in patch_dir.glob('*.patch.kif'):
        with open(patch_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('(subclass '):
                    parts = line.replace('(subclass ', '').replace(')', '').split()
                    if len(parts) >= 2:
                        child, parent = parts[0], parts[1]
                        patch_parents[child].add(parent)
                        patched_concepts.add(child)

    patch_count = sum(len(parents) for parents in patch_parents.values())
    print(f"  Loaded {patch_count} patch relationships")
    print(f"  Patches override {len(patched_concepts)} concepts")

    # Combine: Use patches for patched concepts, SUMO for everything else
    final_parents = defaultdict(set)

    for child in patched_concepts:
        final_parents[child] = patch_parents[child]

    for child, parents in sumo_parents.items():
        if child not in patched_concepts:
            final_parents[child] = parents

    total_count = sum(len(parents) for parents in final_parents.values())
    print(f"  Final: {total_count} relationships ({len(patched_concepts)} overridden by patches)")

    return final_parents

def load_existing_layers():
    """Load existing layer files."""
    layer_dir = Path('data/concept_graph/abstraction_layers')
    layer_data = {}

    print("\nLoading existing layers...")
    for layer_file in sorted(layer_dir.glob('layer*.json')):
        with open(layer_file) as f:
            data = json.load(f)

        layer_num = int(layer_file.stem.replace('layer', ''))
        layer_data[layer_num] = data
        print(f"  Layer {layer_num}: {len(data['concepts'])} concepts")

    return layer_data

def rebuild_layers_from_patches(parent_map):
    """Rebuild layer structure from SUMO + patches."""
    print("\nRebuilding layers from SUMO + patches...")

    # Create output directory
    output_dir = Path('data/concept_graph/abstraction_layers_rebuilt')
    output_dir.mkdir(exist_ok=True)

    # For now, just copy the existing structure but update parent relationships
    # This is a simplified rebuild - the full rebuild would require the original
    # build script logic, but for testing we can verify parent relationships match

    existing_layers = load_existing_layers()

    rebuilt_layers = {}
    for layer_num, layer_data in existing_layers.items():
        rebuilt_concepts = []

        for concept in layer_data['concepts']:
            sumo_term = concept['sumo_term']

            # Get parents from our combined SUMO + patches
            new_parents = sorted(list(parent_map.get(sumo_term, set())))

            # Create updated concept
            updated_concept = concept.copy()
            updated_concept['parent_concepts'] = new_parents

            rebuilt_concepts.append(updated_concept)

        rebuilt_layers[layer_num] = {
            'layer': layer_num,
            'concepts': rebuilt_concepts
        }

        # Write to test directory
        output_file = output_dir / f'layer{layer_num}.json'
        with open(output_file, 'w') as f:
            json.dump(rebuilt_layers[layer_num], f, indent=2)

        print(f"  Rebuilt layer {layer_num}: {len(rebuilt_concepts)} concepts")

    return rebuilt_layers

def compare_layers(existing_layers, rebuilt_layers):
    """Compare existing and rebuilt layers."""
    print("\nComparing layers...")

    total_diffs = 0

    for layer_num in sorted(existing_layers.keys()):
        existing = existing_layers[layer_num]
        rebuilt = rebuilt_layers[layer_num]

        # Build concept maps
        existing_map = {c['sumo_term']: c for c in existing['concepts']}
        rebuilt_map = {c['sumo_term']: c for c in rebuilt['concepts']}

        # Check for differences
        layer_diffs = []

        for sumo_term in existing_map:
            existing_parents = set(existing_map[sumo_term].get('parent_concepts', []))
            rebuilt_parents = set(rebuilt_map.get(sumo_term, {}).get('parent_concepts', []))

            if existing_parents != rebuilt_parents:
                layer_diffs.append({
                    'concept': sumo_term,
                    'existing_parents': sorted(existing_parents),
                    'rebuilt_parents': sorted(rebuilt_parents),
                    'missing': sorted(existing_parents - rebuilt_parents),
                    'added': sorted(rebuilt_parents - existing_parents)
                })

        if layer_diffs:
            print(f"\n  Layer {layer_num}: {len(layer_diffs)} differences found")
            for diff in layer_diffs[:5]:  # Show first 5
                print(f"    {diff['concept']}:")
                if diff['missing']:
                    print(f"      Missing: {diff['missing']}")
                if diff['added']:
                    print(f"      Added: {diff['added']}")

            if len(layer_diffs) > 5:
                print(f"    ... and {len(layer_diffs) - 5} more")

            total_diffs += len(layer_diffs)
        else:
            print(f"  Layer {layer_num}: ✅ Perfect match!")

    print(f"\nTotal differences: {total_diffs}")

    if total_diffs == 0:
        print("\n✅ SUCCESS: Rebuilt layers match existing layers perfectly!")
        print("The patch system is working correctly.")
    else:
        print(f"\n⚠️  WARNING: Found {total_diffs} differences")
        print("Review the differences above.")

    return total_diffs == 0

def main():
    print("=" * 80)
    print("PATCH SYSTEM TEST: Rebuild and Compare")
    print("=" * 80)

    # Load SUMO + patches
    parent_map = load_sumo_with_patches()

    # Load existing layers
    existing_layers = load_existing_layers()

    # Rebuild layers
    rebuilt_layers = rebuild_layers_from_patches(parent_map)

    # Compare
    success = compare_layers(existing_layers, rebuilt_layers)

    print("\n" + "=" * 80)
    if success:
        print("RESULT: ✅ Test PASSED")
    else:
        print("RESULT: ⚠️  Test found differences (review above)")
    print("=" * 80)

    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
