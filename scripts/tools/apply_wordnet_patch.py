#!/usr/bin/env python3
"""
Apply WordNet synset suggestions to SUMO concept layer files.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def apply_patch(suggestions_file: str, layer_dir: str, dry_run: bool = False):
    """Apply synset suggestions to layer files."""

    # Load suggestions
    with open(suggestions_file, 'r') as f:
        suggestions = json.load(f)

    print(f"Loaded {len(suggestions)} concept suggestions")
    print("=" * 80)

    # Group by layer
    by_layer = defaultdict(list)
    for concept in suggestions:
        by_layer[concept['layer']].append(concept)

    stats = {
        'concepts_updated': 0,
        'synsets_added': 0,
        'by_layer': {}
    }

    # Process each layer
    for layer_num in sorted(by_layer.keys()):
        layer_file = Path(layer_dir) / f"layer{layer_num}.json"

        if not layer_file.exists():
            print(f"Warning: Layer file not found: {layer_file}")
            continue

        print(f"\nProcessing Layer {layer_num}: {len(by_layer[layer_num])} concepts")
        print("-" * 80)

        # Load layer data
        with open(layer_file, 'r') as f:
            layer_data = json.load(f)

        layer_stats = {
            'concepts_updated': 0,
            'synsets_added': 0
        }

        # Build index of concepts in layer
        concept_index = {}
        for i, concept in enumerate(layer_data['concepts']):
            concept_index[concept['sumo_term']] = i

        # Apply patches
        for suggestion in by_layer[layer_num]:
            sumo_term = suggestion['sumo_term']

            if sumo_term not in concept_index:
                print(f"  Warning: {sumo_term} not found in layer {layer_num}")
                continue

            idx = concept_index[sumo_term]
            concept = layer_data['concepts'][idx]

            # Get suggested synsets
            suggested_synsets = [s['synset'] for s in suggestion['suggested_synsets']]

            # Get current synsets
            current_synsets = set(concept.get('synsets', []))
            synsets_to_add = [s for s in suggested_synsets if s not in current_synsets]

            if not synsets_to_add:
                print(f"  ✓ {sumo_term}: All suggested synsets already present ({len(current_synsets)} synsets)")
                continue

            # Add new synsets
            if 'synsets' not in concept:
                concept['synsets'] = []

            concept['synsets'].extend(synsets_to_add)

            # Update counts
            old_count = concept.get('synset_count', 0)
            new_count = len(concept['synsets'])
            concept['synset_count'] = new_count
            concept['direct_synset_count'] = new_count

            # Update metadata
            if 'mapping_metadata' not in concept:
                concept['mapping_metadata'] = {}

            concept['mapping_metadata'].update({
                'wordnet_patch_applied': True,
                'wordnet_patch_date': datetime.now().isoformat(),
                'synsets_added_by_patch': len(synsets_to_add),
                'previous_synset_count': old_count
            })

            print(f"  ✓ {sumo_term}: Added {len(synsets_to_add)} synsets ({old_count} → {new_count})")
            print(f"    New synsets: {', '.join(synsets_to_add)}")

            layer_stats['concepts_updated'] += 1
            layer_stats['synsets_added'] += len(synsets_to_add)

        # Update layer metadata
        layer_data['metadata']['wordnet_patch_applied'] = True
        layer_data['metadata']['wordnet_patch_date'] = datetime.now().isoformat()
        layer_data['metadata']['patch_stats'] = layer_stats

        # Save updated layer file
        if not dry_run:
            with open(layer_file, 'w') as f:
                json.dump(layer_data, f, indent=2)
            print(f"\n✓ Saved updated layer file: {layer_file}")
        else:
            print(f"\n[DRY RUN] Would save updated layer file: {layer_file}")

        stats['by_layer'][layer_num] = layer_stats
        stats['concepts_updated'] += layer_stats['concepts_updated']
        stats['synsets_added'] += layer_stats['synsets_added']

    # Print summary
    print("\n" + "=" * 80)
    print("PATCH APPLICATION SUMMARY")
    print("=" * 80)
    print(f"Total concepts updated: {stats['concepts_updated']}")
    print(f"Total synsets added: {stats['synsets_added']}")
    print("\nBy layer:")
    for layer_num in sorted(stats['by_layer'].keys()):
        layer_stats = stats['by_layer'][layer_num]
        print(f"  Layer {layer_num}: {layer_stats['concepts_updated']} concepts, "
              f"{layer_stats['synsets_added']} synsets")

    if dry_run:
        print("\n[DRY RUN MODE] No files were modified")

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Apply WordNet synset patch to layer files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    args = parser.parse_args()

    suggestions_file = Path(__file__).parent.parent / 'results' / 'wordnet_patch_suggestions.json'
    layer_dir = Path(__file__).parent.parent / 'data' / 'concept_graph' / 'abstraction_layers'

    print(f"Suggestions file: {suggestions_file}")
    print(f"Layer directory: {layer_dir}")
    print()

    if args.dry_run:
        print("=== DRY RUN MODE ===\n")

    stats = apply_patch(suggestions_file, layer_dir, dry_run=args.dry_run)
