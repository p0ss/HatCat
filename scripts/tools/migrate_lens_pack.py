#!/usr/bin/env python3
"""
Migrate lens pack from v4.1 to v4.2.

This script:
1. Creates v4.2 lens pack directory structure
2. Copies unchanged base lenses from 4.1
3. Generates training manifest for lenses that need training
4. Clears sibling refinement manifests (all need re-refinement)

Usage:
    python scripts/migrate_lens_pack.py \
        --source lens_packs/apertus-8b_sumo-wordnet-v4.1 \
        --target lens_packs/apertus-8b_sumo-wordnet-v4.2 \
        --migration-plan results/meld_migration_plan.json
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate lens pack between concept pack versions"
    )
    parser.add_argument('--source', required=True,
                        help='Source lens pack directory (e.g., lens_packs/apertus-8b_sumo-wordnet-v4.1)')
    parser.add_argument('--target', required=True,
                        help='Target lens pack directory (e.g., lens_packs/apertus-8b_sumo-wordnet-v4.2)')
    parser.add_argument('--migration-plan', required=True,
                        help='Migration plan JSON from apply_meld.py')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without copying')
    return parser.parse_args()


def main():
    args = parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)

    # Load migration plan
    with open(args.migration_plan) as f:
        migration_plan = json.load(f)

    print("=" * 70)
    print("LENS PACK MIGRATION")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print()
    print("Migration Summary:")
    summary = migration_plan.get('summary', {})
    print(f"  Lenses to copy (base weights):  {summary.get('copy_count', 0)}")
    print(f"  Lenses to delete (obsolete):    {summary.get('delete_count', 0)}")
    print(f"  New concepts to train:          {summary.get('new_count', 0)}")
    print(f"  Total needing training:         {summary.get('total_to_train', 0)}")
    print()
    print("NOTE: ALL lenses will need sibling refinement (sibling groups changed)")
    print()

    if args.dry_run:
        print("[DRY RUN - no changes will be made]")
        print()

    # Create target directory structure
    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "logs").mkdir(exist_ok=True)

    # Get lenses to copy by layer
    lenses_to_copy = migration_plan.get('lenses_to_copy', {})

    copied_count = 0
    skipped_count = 0

    for layer_str, concepts in lenses_to_copy.items():
        # Convert "0" to "layer0", etc.
        layer_name = f"layer{layer_str}" if not layer_str.startswith("layer") else layer_str
        layer_dir = target_dir / layer_name
        source_layer_dir = source_dir / layer_name

        if not source_layer_dir.exists():
            print(f"  Warning: Source layer not found: {source_layer_dir}")
            continue

        if not args.dry_run:
            layer_dir.mkdir(exist_ok=True)

        print(f"Layer {layer_name}: {len(concepts)} lenses to copy")

        for lens_path_or_concept in concepts:
            # Handle both full paths and concept names
            if isinstance(lens_path_or_concept, str) and '/' in lens_path_or_concept:
                # Full path - extract just the filename
                source_lens = Path(lens_path_or_concept)
                target_lens = layer_dir / source_lens.name
            else:
                # Just concept name
                source_lens = source_layer_dir / f"{lens_path_or_concept}_classifier.pt"
                target_lens = layer_dir / f"{lens_path_or_concept}_classifier.pt"

            if source_lens.exists():
                if not args.dry_run:
                    shutil.copy2(source_lens, target_lens)
                copied_count += 1
            else:
                print(f"    Warning: Source lens not found: {source_lens.name}")
                skipped_count += 1

    print()
    print(f"Copied: {copied_count} lenses")
    if skipped_count > 0:
        print(f"Skipped (not found in source): {skipped_count} lenses")

    # Generate training manifest for lenses that need training
    needs_training = []

    # Add new concepts
    for concept in migration_plan.get('new_concepts', []):
        needs_training.append({
            'concept': concept,
            'reason': 'new_concept',
            'priority': 'high'
        })

    # Add concepts that must be retrained (parents of splits)
    for concept in migration_plan.get('must_retrain', []):
        needs_training.append({
            'concept': concept,
            'reason': 'parent_of_split',
            'priority': 'high'
        })

    # Add concepts that should be retrained (siblings of splits)
    for concept in migration_plan.get('should_retrain', []):
        needs_training.append({
            'concept': concept,
            'reason': 'sibling_of_split',
            'priority': 'medium'
        })

    # Write training manifest
    training_manifest = {
        'created': datetime.now().isoformat(),
        'source_pack': str(source_dir),
        'target_pack': str(target_dir),
        'migration_plan': args.migration_plan,
        'lenses_to_train': needs_training,
        'sibling_refinement_required': True,
        'sibling_refinement_note': 'ALL lenses need sibling refinement after base training is complete',
        'summary': {
            'total_lenses_to_train': len(needs_training),
            'new_concepts': len(migration_plan.get('new_concepts', [])),
            'must_retrain': len(migration_plan.get('must_retrain', [])),
            'should_retrain': len(migration_plan.get('should_retrain', [])),
            'copied_from_source': copied_count,
        }
    }

    manifest_path = target_dir / "training_manifest.json"
    if not args.dry_run:
        with open(manifest_path, 'w') as f:
            json.dump(training_manifest, f, indent=2)
        print(f"\nWrote training manifest to: {manifest_path}")
    else:
        print(f"\n[Would write training manifest to: {manifest_path}]")

    # Clear sibling refinement manifests (they're now invalid)
    if not args.dry_run:
        for layer_dir in target_dir.glob("layer*"):
            refinement_manifest = layer_dir / "sibling_refinement.json"
            if refinement_manifest.exists():
                refinement_manifest.unlink()
                print(f"  Cleared sibling refinement manifest: {refinement_manifest}")

    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Train lenses for concepts that need it:")
    print(f"   python src/training/train_pending_lenses.py \\")
    print(f"       --lens-pack {target_dir} \\")
    print(f"       --concept-pack sumo-wordnet-v4 \\")
    print(f"       --model swiss-ai/Apertus-8B-2509")
    print()
    print("2. Run sibling refinement on ALL lenses:")
    print(f"   python src/training/train_concept_pack_lenses.py \\")
    print(f"       --concept-pack sumo-wordnet-v4 \\")
    print(f"       --output-dir {target_dir} \\")
    print(f"       --model swiss-ai/Apertus-8B-2509 \\")
    print(f"       --refine-only")
    print()


if __name__ == "__main__":
    main()
