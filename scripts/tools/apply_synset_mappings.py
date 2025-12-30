#!/usr/bin/env python3
"""
Apply suggested synset mappings to layer JSON files.

This script:
1. Loads synset mapping suggestions
2. Applies top-quality mappings (with quality threshold)
3. Updates layer JSON files with canonical_synset fields
4. Creates backup before modification
5. Generates report of changes
"""

import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_mapping_suggestions() -> Dict:
    """Load synset mapping suggestions from previous analysis."""
    suggestions_path = Path("results/synset_mapping_suggestions.json")
    if not suggestions_path.exists():
        print("Error: Run suggest_synset_mappings.py first")
        sys.exit(1)

    with open(suggestions_path) as f:
        return json.load(f)


def backup_layer_file(layer_num: int) -> Path:
    """Create timestamped backup of layer file."""
    layer_path = Path(f"data/concept_graph/abstraction_layers/layer{layer_num}.json")
    if not layer_path.exists():
        return None

    backup_dir = Path("data/concept_graph/abstraction_layers/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"layer{layer_num}_{timestamp}.json"

    shutil.copy2(layer_path, backup_path)
    return backup_path


def apply_mapping(
    layer_num: int,
    concept_name: str,
    synset_id: str,
    quality_score: int
) -> bool:
    """
    Apply a synset mapping to a concept in a layer file.

    Returns:
        True if successful, False otherwise
    """
    layer_path = Path(f"data/concept_graph/abstraction_layers/layer{layer_num}.json")
    if not layer_path.exists():
        return False

    # Load layer data
    with open(layer_path) as f:
        data = json.load(f)

    # Find and update concept
    found = False
    for concept in data.get('concepts', []):
        if concept['sumo_term'] == concept_name:
            # Store old value for reporting
            old_synset = concept.get('canonical_synset')

            # Apply new mapping
            concept['canonical_synset'] = synset_id

            # Add metadata about mapping
            if 'mapping_metadata' not in concept:
                concept['mapping_metadata'] = {}

            concept['mapping_metadata']['auto_mapped'] = True
            concept['mapping_metadata']['quality_score'] = quality_score
            concept['mapping_metadata']['previous_synset'] = old_synset
            concept['mapping_metadata']['mapped_date'] = datetime.now().isoformat()

            found = True
            break

    if not found:
        return False

    # Write updated data
    with open(layer_path, 'w') as f:
        json.dump(data, f, indent=2)

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Apply synset mappings to layer files")
    parser.add_argument('--auto-approve', action='store_true',
                        help='Skip confirmation prompt and apply automatically')
    parser.add_argument('--quality-threshold', type=int, default=30,
                        help='Minimum quality score for mappings (default: 30)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be changed without making changes')
    args = parser.parse_args()

    print("=" * 80)
    print("APPLYING SYNSET MAPPINGS")
    print("=" * 80)

    # Load suggestions
    print("\nLoading mapping suggestions...")
    data = load_mapping_suggestions()
    suggestions = data['suggestions']

    print(f"✓ Loaded {len(suggestions)} mapping suggestions")

    # Filter by quality threshold
    quality_threshold = args.quality_threshold
    high_quality = [
        s for s in suggestions
        if s['candidates'] and s['candidates'][0]['quality_score'] >= quality_threshold
    ]

    print(f"\nFiltering by quality threshold ({quality_threshold}+)...")
    print(f"  High-quality mappings: {len(high_quality)}")
    print(f"  Low-quality mappings (skipped): {len(suggestions) - len(high_quality)}")

    # Prompt for confirmation
    print("\n" + "=" * 80)
    print("REVIEW SETTINGS")
    print("=" * 80)
    print(f"\nWill apply {len(high_quality)} synset mappings")
    print(f"Quality threshold: {quality_threshold}+")
    print(f"\nBackups will be created before modification")

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made")

    if not args.auto_approve and not args.dry_run:
        try:
            response = input("\nProceed with mapping updates? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return 0
        except EOFError:
            print("\nNo input available. Use --auto-approve to skip confirmation.")
            return 1
    elif args.auto_approve:
        print("\n✓ Auto-approved, proceeding with updates...")

    # Group by layer for efficient processing
    by_layer = {}
    for sugg in high_quality:
        layer = sugg['layer']
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(sugg)

    # Apply mappings layer by layer
    print("\n" + "=" * 80)
    print("APPLYING MAPPINGS" if not args.dry_run else "DRY RUN - SHOWING PLANNED CHANGES")
    print("=" * 80)

    total_applied = 0
    total_failed = 0
    changes_by_layer = {}

    for layer_num in sorted(by_layer.keys()):
        layer_suggestions = by_layer[layer_num]

        print(f"\nLayer {layer_num}: Processing {len(layer_suggestions)} mappings...")

        if not args.dry_run:
            # Create backup
            backup_path = backup_layer_file(layer_num)
            if backup_path:
                print(f"  ✓ Backup created: {backup_path}")
            else:
                print(f"  ⚠ Warning: Could not create backup")

        # Apply mappings
        layer_changes = []
        for sugg in layer_suggestions:
            concept_name = sugg['concept']
            best_candidate = sugg['candidates'][0]
            synset_id = best_candidate['synset_id']
            quality_score = best_candidate['quality_score']

            if args.dry_run:
                # Just record what would be done
                total_applied += 1
                layer_changes.append({
                    'concept': concept_name,
                    'synset': synset_id,
                    'quality': quality_score,
                    'definition': best_candidate['definition']
                })
            else:
                success = apply_mapping(layer_num, concept_name, synset_id, quality_score)

                if success:
                    total_applied += 1
                    layer_changes.append({
                        'concept': concept_name,
                        'synset': synset_id,
                        'quality': quality_score,
                        'definition': best_candidate['definition']
                    })
                else:
                    total_failed += 1
                    print(f"  ✗ Failed to map: {concept_name}")

        if layer_changes:
            changes_by_layer[layer_num] = layer_changes
            if args.dry_run:
                print(f"  → Would apply {len(layer_changes)} mappings")
            else:
                print(f"  ✓ Applied {len(layer_changes)} mappings")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal mappings applied: {total_applied}")
    print(f"Failed mappings: {total_failed}")

    # Detailed report
    print("\n" + "=" * 80)
    print("DETAILED CHANGES BY LAYER")
    print("=" * 80)

    for layer_num in sorted(changes_by_layer.keys()):
        changes = changes_by_layer[layer_num]
        print(f"\n{'='*80}")
        print(f"LAYER {layer_num} ({len(changes)} changes)")
        print(f"{'='*80}")

        # Show first 10 changes
        for change in changes[:10]:
            print(f"\n{change['concept']}:")
            print(f"  → {change['synset']} (quality: {change['quality']})")
            print(f"  {change['definition'][:80]}...")

        if len(changes) > 10:
            print(f"\n... and {len(changes)-10} more")

    # Save detailed report
    report_path = Path("results/synset_mapping_applied.json")
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_applied': total_applied,
            'total_failed': total_failed,
            'quality_threshold': quality_threshold,
        },
        'changes_by_layer': changes_by_layer
    }

    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"""
1. ✓ Backups created in data/concept_graph/abstraction_layers/backups/

2. VERIFY changes look correct:
   - Review results/synset_mapping_applied.json
   - Spot-check a few layer files

3. RE-RUN coverage analysis to verify improvement:
   python scripts/analyze_sumo_concept_coverage.py

4. TRAIN with new mappings:
   python scripts/train_layer_concepts.py --layer 0

5. IF issues arise, restore from backups:
   cp data/concept_graph/abstraction_layers/backups/layerN_*.json \\
      data/concept_graph/abstraction_layers/layerN.json
""")

    print(f"\n✓ Detailed report saved to: {report_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
