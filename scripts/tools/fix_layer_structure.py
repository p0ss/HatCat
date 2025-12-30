#!/usr/bin/env python3
"""
Fix Layer Structure

1. Remove Layer 5 entirely, redistributing "_Other" synsets to parent concepts
2. Enhance Layer 0 concepts with multi-layer child examples
3. Prepare special training data for Layer 0 cross-concept distinction
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set


def load_all_concepts(layers_dir: Path) -> Dict:
    """Load all concepts from all layers."""
    all_concepts = {}

    for layer_file in sorted(layers_dir.glob("layer*.json")):
        with open(layer_file) as f:
            data = json.load(f)
            layer = data['metadata']['layer']

            for concept in data['concepts']:
                name = concept['sumo_term']
                key = (name, layer)
                all_concepts[key] = concept

    return all_concepts


def redistribute_layer5_to_parents(layers_dir: Path, dry_run: bool = False):
    """
    Merge Layer 5 "_Other" and "_OtherAgent" concepts back into their parents.
    """
    print("\n" + "="*80)
    print("REDISTRIBUTING LAYER 5 TO PARENTS")
    print("="*80)

    all_concepts = load_all_concepts(layers_dir)

    # Find Layer 5 concepts to redistribute
    layer5_concepts = {k: v for k, v in all_concepts.items() if k[1] == 5}
    to_redistribute = {
        k: v for k, v in layer5_concepts.items()
        if '_Other' in k[0] or '_OtherAgent' in k[0]
    }

    print(f"\nFound {len(to_redistribute)} Layer 5 concepts to redistribute:")
    for (name, layer), concept in sorted(to_redistribute.items()):
        parents = concept.get('parent_concepts', [])
        synset_count = len(concept.get('synsets', []))
        print(f"  {name}: {synset_count} synsets -> parent: {parents[0] if parents else 'NONE'}")

    # Build redistribution plan
    redistribution_plan = defaultdict(list)  # parent_name -> [synsets_to_add]
    concepts_to_remove = []

    for (name, layer), concept in to_redistribute.items():
        parents = concept.get('parent_concepts', [])
        if not parents:
            print(f"  WARNING: {name} has no parent, skipping")
            continue

        parent_name = parents[0]
        synsets = concept.get('synsets', [])
        redistribution_plan[parent_name].extend(synsets)
        concepts_to_remove.append((name, layer))

    print(f"\nRedistribution plan:")
    print(f"  Will merge {len(concepts_to_remove)} Layer 5 concepts")
    print(f"  Will enhance {len(redistribution_plan)} parent concepts")

    # Apply redistribution
    if not dry_run:
        # Update parent concepts with additional synsets
        for layer_file in sorted(layers_dir.glob("layer[0-4].json")):
            with open(layer_file) as f:
                data = json.load(f)

            modified = False
            for concept in data['concepts']:
                name = concept['sumo_term']
                if name in redistribution_plan:
                    # Add synsets from child "_Other" concepts
                    existing_synsets = set(concept.get('synsets', []))
                    new_synsets = redistribution_plan[name]
                    combined = sorted(set(list(existing_synsets) + new_synsets))

                    old_count = len(existing_synsets)
                    new_count = len(combined)

                    concept['synsets'] = combined
                    concept['synset_count'] = new_count
                    modified = True

                    print(f"  ✓ Enhanced {name}: {old_count} -> {new_count} synsets (+{new_count-old_count})")

            if modified:
                # Update metadata
                data['metadata']['total_concepts'] = len(data['concepts'])

                with open(layer_file, 'w') as f:
                    json.dump(data, f, indent=2)

        # Clear Layer 5
        layer5_file = layers_dir / "layer5.json"
        if layer5_file.exists():
            with open(layer5_file) as f:
                data = json.load(f)

            # Remove "_Other" concepts
            original_count = len(data['concepts'])
            data['concepts'] = [
                c for c in data['concepts']
                if not ('_Other' in c['sumo_term'] or '_OtherAgent' in c['sumo_term'])
            ]

            data['metadata']['total_concepts'] = len(data['concepts'])

            with open(layer5_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"\n✓ Layer 5: {original_count} -> {len(data['concepts'])} concepts")
    else:
        print("\n[DRY RUN] Would apply redistribution")

    return len(concepts_to_remove)


def collect_descendant_synsets(
    concept_name: str,
    all_concepts: Dict,
    max_depth: int = 3
) -> Set[str]:
    """
    Collect synsets from a concept and its descendants up to max_depth levels.
    """
    synsets = set()

    def collect_recursive(name: str, depth: int):
        if depth > max_depth:
            return

        # Find concept in any layer
        for (cname, layer), concept in all_concepts.items():
            if cname == name:
                synsets.update(concept.get('synsets', []))

                # Recurse to children
                for child in concept.get('category_children', []):
                    collect_recursive(child, depth + 1)
                break

    collect_recursive(concept_name, 0)
    return synsets


def enhance_layer0_training_data(layers_dir: Path, dry_run: bool = False):
    """
    Enhance Layer 0 concepts with multi-layer descendant examples.
    """
    print("\n" + "="*80)
    print("ENHANCING LAYER 0 TRAINING DATA")
    print("="*80)

    all_concepts = load_all_concepts(layers_dir)
    layer0_file = layers_dir / "layer0.json"

    with open(layer0_file) as f:
        data = json.load(f)

    print(f"\nEnhancing {len(data['concepts'])} Layer 0 concepts with descendant synsets:")

    for concept in data['concepts']:
        name = concept['sumo_term']

        # Collect synsets from descendants (3 layers deep)
        descendant_synsets = collect_descendant_synsets(name, all_concepts, max_depth=3)

        # Combine with existing
        existing = set(concept.get('synsets', []))
        combined = existing | descendant_synsets

        old_count = len(existing)
        new_count = len(combined)

        if new_count > old_count:
            concept['synsets'] = sorted(combined)
            concept['synset_count'] = new_count
            print(f"  {name}: {old_count} -> {new_count} synsets (+{new_count-old_count})")
        else:
            print(f"  {name}: {old_count} synsets (no change)")

    if not dry_run:
        data['metadata']['total_concepts'] = len(data['concepts'])

        with open(layer0_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Updated Layer 0 concepts")
    else:
        print("\n[DRY RUN] Would update Layer 0")


def create_layer0_cross_training_spec(layers_dir: Path, output_file: Path):
    """
    Create training specification for Layer 0 cross-concept distinction.

    Each Layer 0 concept should be trained with:
    - Its own synsets + 3 layers of descendants (positive)
    - Synsets from ALL other Layer 0 concepts (negative)
    - This ensures Layer 0 concepts can distinguish from each other
    """
    print("\n" + "="*80)
    print("CREATING LAYER 0 CROSS-TRAINING SPEC")
    print("="*80)

    layer0_file = layers_dir / "layer0.json"

    with open(layer0_file) as f:
        data = json.load(f)

    layer0_concepts = {}
    for concept in data['concepts']:
        name = concept['sumo_term']
        synsets = concept.get('synsets', [])
        layer0_concepts[name] = synsets

    # Build training spec
    training_spec = {
        'description': 'Layer 0 cross-concept distinction training requirements',
        'created': datetime.now().isoformat(),
        'min_synsets_required': 20,
        'training_strategy': {
            'positive_examples': 'Concept synsets + 3 layers of descendants',
            'negative_examples': 'ALL other Layer 0 concept synsets',
            'purpose': 'Ensure Layer 0 concepts can distinguish from each other'
        },
        'concepts': {}
    }

    print(f"\nLayer 0 concepts:")
    for name, synsets in sorted(layer0_concepts.items()):
        # Negative examples are all OTHER Layer 0 concepts
        negative_concepts = [n for n in layer0_concepts.keys() if n != name]
        negative_synset_count = sum(
            len(layer0_concepts[neg]) for neg in negative_concepts
        )

        training_spec['concepts'][name] = {
            'positive_synsets': len(synsets),
            'negative_concepts': negative_concepts,
            'negative_synsets': negative_synset_count,
            'meets_minimum': len(synsets) >= 20
        }

        status = "✓" if len(synsets) >= 20 else "⚠️ "
        print(f"  {status} {name}: {len(synsets)} positive, "
              f"{negative_synset_count} negative (from {len(negative_concepts)} other L0 concepts)")

    # Save spec
    with open(output_file, 'w') as f:
        json.dump(training_spec, f, indent=2)

    print(f"\n✓ Saved training spec to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix layer structure")
    parser.add_argument('--layers-dir', type=Path,
                       default=Path('data/concept_graph/abstraction_layers'))
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')

    args = parser.parse_args()

    print("="*80)
    print("LAYER STRUCTURE FIX")
    print("="*80)
    print(f"Layers dir: {args.layers_dir}")
    print(f"Dry run: {args.dry_run}")

    # Create backup
    if not args.dry_run:
        backup_dir = args.layers_dir / 'backups'
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        print(f"\nCreating backup...")
        for layer_file in args.layers_dir.glob("layer*.json"):
            backup_file = backup_dir / f"{layer_file.stem}_{timestamp}.json"
            shutil.copy2(layer_file, backup_file)
        print(f"✓ Backed up to {backup_dir}")

    # Step 1: Redistribute Layer 5
    removed_count = redistribute_layer5_to_parents(args.layers_dir, args.dry_run)

    # Step 2: Enhance Layer 0
    enhance_layer0_training_data(args.layers_dir, args.dry_run)

    # Step 3: Create Layer 0 training spec
    spec_file = Path('docs/LAYER0_TRAINING_SPEC.md')
    create_layer0_cross_training_spec(args.layers_dir, spec_file.with_suffix('.json'))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Removed {removed_count} Layer 5 '_Other' concepts")
    print(f"  Enhanced Layer 0 concepts with descendant synsets")
    print(f"  Created Layer 0 cross-training specification")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
