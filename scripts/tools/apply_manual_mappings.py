#!/usr/bin/env python3
"""Apply manually curated synset mappings."""

import json
import shutil
from pathlib import Path
from datetime import datetime

def apply_manual_mappings():
    # Load curated mappings
    mappings_file = Path("results/manual_synset_mappings_curated.json")
    with open(mappings_file) as f:
        mappings = json.load(f)

    print("=" * 80)
    print("APPLYING MANUAL SYNSET MAPPINGS")
    print("=" * 80)
    print(f"\nLoading {len(mappings)} manual mappings...")

    # Group by layer
    by_layer = {}
    for mapping in mappings:
        layer = mapping['layer']
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(mapping)

    # Apply to each layer
    total_applied = 0

    for layer_num in sorted(by_layer.keys()):
        layer_mappings = by_layer[layer_num]
        layer_path = Path(f"data/concept_graph/abstraction_layers/layer{layer_num}.json")

        print(f"\nLayer {layer_num}: Applying {len(layer_mappings)} mappings...")

        # Backup
        backup_dir = Path("data/concept_graph/abstraction_layers/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"layer{layer_num}_manual_{timestamp}.json"
        shutil.copy2(layer_path, backup_path)
        print(f"  ✓ Backup: {backup_path}")

        # Load layer
        with open(layer_path) as f:
            layer_data = json.load(f)

        # Apply mappings
        for mapping in layer_mappings:
            concept_name = mapping['concept']
            synset_id = mapping['suggested_synset']

            # Find concept
            for concept in layer_data.get('concepts', []):
                if concept['sumo_term'] == concept_name:
                    concept['canonical_synset'] = synset_id

                    # Add metadata
                    if 'mapping_metadata' not in concept:
                        concept['mapping_metadata'] = {}
                    concept['mapping_metadata']['manually_curated'] = True
                    concept['mapping_metadata']['mapped_date'] = datetime.now().isoformat()
                    concept['mapping_metadata']['note'] = mapping.get('note', '')

                    total_applied += 1
                    print(f"  ✓ {concept_name} → {synset_id}")
                    break

        # Save
        with open(layer_path, 'w') as f:
            json.dump(layer_data, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal mappings applied: {total_applied}")
    print("\nBackups created in: data/concept_graph/abstraction_layers/backups/")
    print("\nNext steps:")
    print("  1. Verify coverage: python scripts/analyze_sumo_concept_coverage.py")
    print("  2. Train with new mappings")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(apply_manual_mappings())
