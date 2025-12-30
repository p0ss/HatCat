#!/usr/bin/env python3
"""
Migrate existing lens results to concept pack + lens pack structure.

This script:
1. Creates concept pack from existing SUMO structure
2. Creates lens pack from existing trained lenses
3. Migrates files to new structure
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


def migrate_to_packs():
    print("=" * 80)
    print("MIGRATING TO CONCEPT PACK + LENS PACK STRUCTURE")
    print("=" * 80)
    print()

    # Paths
    project_root = Path(__file__).parent.parent
    old_lenses_dir = project_root / 'results' / 'sumo_classifiers_adaptive_l0_5'
    concept_pack_dir = project_root / 'concept_packs' / 'sumo-wordnet-v1'
    lens_pack_dir = project_root / 'lens_packs' / 'gemma-3-4b-pt_sumo-wordnet-v1'

    # Check source exists
    if not old_lenses_dir.exists():
        print(f"Error: Old lenses directory not found: {old_lenses_dir}")
        return

    print(f"Source: {old_lenses_dir}")
    print(f"Concept pack destination: {concept_pack_dir}")
    print(f"Lens pack destination: {lens_pack_dir}")
    print()

    # Create directories
    concept_pack_dir.mkdir(parents=True, exist_ok=True)
    lens_pack_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Create concept pack structure
    # ========================================================================
    print("STEP 1: Creating concept pack structure...")
    print()

    # Concept pack already has pack.json created
    # Just need to create hierarchy directory
    hierarchy_dir = concept_pack_dir / 'hierarchy'
    hierarchy_dir.mkdir(exist_ok=True)

    # Copy concept metadata if exists
    old_metadata = old_lenses_dir / 'concept_metadata.json'
    if old_metadata.exists():
        new_metadata = hierarchy_dir / 'concept_hierarchy.json'
        shutil.copy2(old_metadata, new_metadata)
        print(f"✓ Copied concept hierarchy metadata")

    print(f"✓ Concept pack structure ready: {concept_pack_dir}")
    print()

    # ========================================================================
    # STEP 2: Create lens pack structure
    # ========================================================================
    print("STEP 2: Creating lens pack structure...")
    print()

    # Create lens directories
    lenses_dir = lens_pack_dir / 'lenses'
    activation_dir = lenses_dir / 'activation'
    text_dir = lenses_dir / 'text'
    metadata_dir = lens_pack_dir / 'metadata'

    activation_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 3: Migrate lenses
    # ========================================================================
    print("STEP 3: Migrating lenses...")
    print()

    activation_count = 0
    text_count = 0

    # Migrate activation lenses (named *_classifier.pt)
    for lens_file in old_lenses_dir.glob('**/*_classifier.pt'):
        dest = activation_dir / lens_file.name
        shutil.copy2(lens_file, dest)
        activation_count += 1

    print(f"✓ Migrated {activation_count} activation lenses")

    # Migrate text lenses
    for lens_file in old_lenses_dir.glob('**/*_text_lens.joblib'):
        dest = text_dir / lens_file.name
        shutil.copy2(lens_file, dest)
        text_count += 1

    print(f"✓ Migrated {text_count} text lenses")

    # Migrate metadata
    metadata_count = 0
    for meta_file in old_lenses_dir.glob('**/*_metadata.json'):
        dest = metadata_dir / meta_file.name
        shutil.copy2(meta_file, dest)
        metadata_count += 1

    print(f"✓ Migrated {metadata_count} metadata files")
    print()

    # ========================================================================
    # STEP 4: Copy training results
    # ========================================================================
    print("STEP 4: Copying training results...")
    print()

    old_results = old_lenses_dir / 'training_results.json'
    if old_results.exists():
        new_results = lens_pack_dir / 'training_results.json'
        shutil.copy2(old_results, new_results)
        print(f"✓ Copied training results")

    # ========================================================================
    # STEP 5: Copy sunburst visualization
    # ========================================================================
    print("STEP 5: Copying sunburst visualization...")
    print()

    old_sunburst = project_root / 'results' / 'concept_sunburst_positions.json'
    if old_sunburst.exists():
        new_sunburst = lens_pack_dir / 'concept_sunburst_positions.json'
        shutil.copy2(old_sunburst, new_sunburst)
        print(f"✓ Copied sunburst positions")

    old_sunburst_html = project_root / 'results' / 'concept_sunburst_visualization.html'
    if old_sunburst_html.exists():
        new_sunburst_html = lens_pack_dir / 'concept_sunburst_visualization.html'
        shutil.copy2(old_sunburst_html, new_sunburst_html)
        print(f"✓ Copied sunburst visualization")

    print()

    # ========================================================================
    # STEP 6: Update lens pack.json with actual counts
    # ========================================================================
    print("STEP 6: Updating lens pack metadata...")
    print()

    pack_json_path = lens_pack_dir / 'pack.json'
    if pack_json_path.exists():
        with open(pack_json_path) as f:
            pack_json = json.load(f)

        # Update with actual training results if available
        if old_results.exists():
            with open(old_results) as f:
                training_results = json.load(f)

            # Calculate averages
            if 'concept_results' in training_results:
                concept_results = training_results['concept_results']

                activation_f1s = []
                text_f1s = []

                for concept_data in concept_results.values():
                    if 'activation_lens' in concept_data:
                        activation_f1s.append(concept_data['activation_lens'].get('final_f1', 0))
                    if 'text_lens' in concept_data:
                        text_f1s.append(concept_data['text_lens'].get('final_f1', 0))

                if activation_f1s:
                    pack_json['performance']['activation_lenses']['avg_f1'] = sum(activation_f1s) / len(activation_f1s)
                if text_f1s:
                    pack_json['performance']['text_lenses']['avg_f1'] = sum(text_f1s) / len(text_f1s)

        # Write updated pack.json
        with open(pack_json_path, 'w') as f:
            json.dump(pack_json, f, indent=2)

        print(f"✓ Updated lens pack metadata")

    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("MIGRATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"Concept pack: {concept_pack_dir}")
    print(f"  - Ontology definitions: SUMO + WordNet")
    print()
    print(f"Lens pack: {lens_pack_dir}")
    print(f"  - Activation lenses: {activation_count}")
    print(f"  - Text lenses: {text_count}")
    print(f"  - Metadata files: {metadata_count}")
    print()
    print("Next steps:")
    print("  1. Verify lens pack structure")
    print("  2. Test loading with updated DynamicLensManager")
    print("  3. Update server to use lens pack registry")
    print()


if __name__ == "__main__":
    migrate_to_packs()
