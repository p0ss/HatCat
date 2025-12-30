#!/usr/bin/env python3
"""
Fix Concept Pack Duplicates

Removes duplicate concepts across layers in the concept pack:
1. For each concept name, keeps only the BEST layer:
   - Priority: category lens > most synsets > lowest layer number
2. Removes layer 6 entries that duplicate concepts in layers 0-5
3. Creates backup before modifying files

This fixes the hierarchy so each concept appears at exactly ONE layer.
"""

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def find_best_layer_for_concepts(layer_files: List[Path]) -> Dict[str, Tuple[int, dict]]:
    """
    Find the best layer for each concept.

    Returns:
        concept_name -> (best_layer, concept_dict)
    """
    concept_to_best = {}

    for layer_file in layer_files:
        with open(layer_file) as f:
            data = json.load(f)
            layer = data['metadata']['layer']

            for concept in data['concepts']:
                sumo_term = concept['sumo_term']

                # Skip layer 6 for now - we'll handle it separately
                if layer == 6:
                    continue

                # Evaluate quality of this entry
                is_category = concept.get('is_category_lens', False)
                synset_count = len(concept.get('synsets', []))

                # Check if we should keep this over existing
                if sumo_term in concept_to_best:
                    best_layer, best_concept = concept_to_best[sumo_term]
                    best_is_category = best_concept.get('is_category_lens', False)
                    best_synset_count = len(best_concept.get('synsets', []))

                    # Priority: category lens > more synsets > lower layer
                    should_replace = False
                    if is_category and not best_is_category:
                        should_replace = True
                    elif is_category == best_is_category:
                        if synset_count > best_synset_count:
                            should_replace = True
                        elif synset_count == best_synset_count and layer < best_layer:
                            should_replace = True

                    if should_replace:
                        concept_to_best[sumo_term] = (layer, concept)
                else:
                    concept_to_best[sumo_term] = (layer, concept)

    return concept_to_best


def clean_layer_files(
    layer_files: List[Path],
    concept_to_best: Dict[str, Tuple[int, dict]],
    output_dir: Path,
    dry_run: bool = False
):
    """
    Clean layer files by removing duplicates.
    """
    stats = {
        'original_counts': {},
        'cleaned_counts': {},
        'removed_counts': {},
        'layer_6_original': 0,
        'layer_6_cleaned': 0,
        'layer_6_removed': 0,
    }

    for layer_file in layer_files:
        with open(layer_file) as f:
            data = json.load(f)

        layer = data['metadata']['layer']
        original_concepts = data['concepts']
        stats['original_counts'][layer] = len(original_concepts)

        if layer == 6:
            stats['layer_6_original'] = len(original_concepts)
            # For layer 6: only keep synsets that DON'T map to existing concepts
            cleaned_concepts = []
            for concept in original_concepts:
                sumo_term = concept['sumo_term']
                # Keep if this SUMO term doesn't exist in higher layers
                if sumo_term not in concept_to_best:
                    cleaned_concepts.append(concept)

            stats['layer_6_cleaned'] = len(cleaned_concepts)
            stats['layer_6_removed'] = stats['layer_6_original'] - stats['layer_6_cleaned']
        else:
            # For layers 0-5: only keep concepts where this is the best layer
            cleaned_concepts = []
            for concept in original_concepts:
                sumo_term = concept['sumo_term']
                best_layer, _ = concept_to_best.get(sumo_term, (None, None))

                if best_layer == layer:
                    cleaned_concepts.append(concept)

        stats['cleaned_counts'][layer] = len(cleaned_concepts)
        stats['removed_counts'][layer] = stats['original_counts'][layer] - stats['cleaned_counts'][layer]

        # Update metadata
        data['concepts'] = cleaned_concepts
        data['metadata']['total_concepts'] = len(cleaned_concepts)

        # Update other metadata fields if they exist
        if 'samples' in data['metadata']:
            # Update samples to reflect new concepts
            data['metadata']['samples'] = cleaned_concepts[:10] if len(cleaned_concepts) >= 10 else cleaned_concepts

        # Write cleaned file
        if not dry_run:
            output_file = output_dir / layer_file.name
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Layer {layer}: {stats['original_counts'][layer]} -> {stats['cleaned_counts'][layer]} "
                  f"(removed {stats['removed_counts'][layer]})")
        else:
            print(f"[DRY RUN] Layer {layer}: would go from {stats['original_counts'][layer]} to "
                  f"{stats['cleaned_counts'][layer]} (remove {stats['removed_counts'][layer]})")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fix duplicate concepts across layers in concept pack"
    )
    parser.add_argument('--layers-dir', type=Path,
                       default=Path('data/concept_graph/abstraction_layers'),
                       help='Directory containing layer*.json files')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: same as input, with backup)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.layers_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CONCEPT PACK DUPLICATE FIXER")
    print("="*80)
    print(f"Input dir: {args.layers_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Dry run: {args.dry_run}")
    print("="*80)

    # Find all layer files
    layer_files = sorted(args.layers_dir.glob("layer*.json"))
    if not layer_files:
        print(f"ERROR: No layer*.json files found in {args.layers_dir}")
        return 1

    print(f"\nFound {len(layer_files)} layer files")

    # Find best layer for each concept
    print("\nAnalyzing concepts across layers...")
    concept_to_best = find_best_layer_for_concepts(layer_files)
    print(f"✓ Found {len(concept_to_best)} unique concepts in layers 0-5")

    # Count duplicates
    all_concepts_by_layer = defaultdict(set)
    for layer_file in layer_files:
        with open(layer_file) as f:
            data = json.load(f)
            layer = data['metadata']['layer']
            for concept in data['concepts']:
                all_concepts_by_layer[layer].add(concept['sumo_term'])

    duplicates = defaultdict(list)
    for sumo_term in concept_to_best.keys():
        layers = [layer for layer, concepts in all_concepts_by_layer.items()
                 if sumo_term in concepts]
        if len(layers) > 1:
            duplicates[sumo_term] = layers

    print(f"✓ Found {len(duplicates)} concepts appearing in multiple layers")

    # Show sample duplicates
    print("\nSample duplicates (showing first 10):")
    for i, (concept, layers) in enumerate(list(duplicates.items())[:10]):
        best_layer = concept_to_best[concept][0]
        print(f"  {concept}: appears in layers {layers}, keeping layer {best_layer}")

    if len(duplicates) > 10:
        print(f"  ... and {len(duplicates) - 10} more")

    # Create backup if not dry run and output is same as input
    if not args.dry_run and args.output_dir == args.layers_dir:
        backup_dir = args.layers_dir / 'backups'
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        print(f"\nCreating backup in {backup_dir}...")
        for layer_file in layer_files:
            backup_file = backup_dir / f"{layer_file.stem}_{timestamp}.json"
            shutil.copy2(layer_file, backup_file)
        print(f"✓ Backed up {len(layer_files)} files")

    # Clean layer files
    print("\nCleaning layer files...")
    stats = clean_layer_files(layer_files, concept_to_best, args.output_dir, args.dry_run)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_original = sum(stats['original_counts'].values())
    total_cleaned = sum(stats['cleaned_counts'].values())
    total_removed = total_original - total_cleaned

    print(f"\nTotal concepts:")
    print(f"  Original: {total_original}")
    print(f"  Cleaned:  {total_cleaned}")
    print(f"  Removed:  {total_removed}")

    print(f"\nPer-layer breakdown:")
    print(f"{'Layer':<8} {'Original':<12} {'Cleaned':<12} {'Removed':<12} {'% Kept':<10}")
    print("-"*80)

    for layer in sorted(stats['original_counts'].keys()):
        orig = stats['original_counts'][layer]
        clean = stats['cleaned_counts'][layer]
        removed = stats['removed_counts'][layer]
        pct_kept = 100 * clean / orig if orig > 0 else 0

        print(f"{layer:<8} {orig:<12} {clean:<12} {removed:<12} {pct_kept:>6.1f}%")

    if stats['layer_6_original'] > 0:
        print(f"\nLayer 6 (synset-level) cleanup:")
        print(f"  {stats['layer_6_original']} synsets originally mapped to SUMO categories")
        print(f"  {stats['layer_6_removed']} synsets duplicated concepts in layers 0-5 (removed)")
        print(f"  {stats['layer_6_cleaned']} unique synsets kept (no higher-level concept)")

    if not args.dry_run:
        print(f"\n✓ Cleaned layer files saved to: {args.output_dir}")
    else:
        print(f"\n[DRY RUN] No files were modified")

    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
