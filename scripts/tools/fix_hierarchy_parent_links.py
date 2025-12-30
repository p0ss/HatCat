#!/usr/bin/env python3
"""
Fix Hierarchy Parent Links

Rebuilds layer files to include explicit parent references for every concept.

For each concept, adds:
- `parent_concepts`: List of parent concept names (in layer-1)
- Validates that every non-layer-0 concept has at least one parent
- Validates that parent->child relationships are bidirectional

This is critical for dynamic lens loading to work correctly.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def build_parent_child_mappings(layer_files: List[Path]) -> Tuple[Dict, Dict, Dict]:
    """
    Build bidirectional parent-child mappings from layer files.

    Returns:
        - concept_to_layer: {concept_name: layer}
        - children_to_parents: {child_name: [parent_names]}
        - parent_to_children: {parent_name: [child_names]}
    """
    concept_to_layer = {}
    parent_to_children = defaultdict(list)
    children_to_parents = defaultdict(list)

    # First pass: collect all concepts and their children lists
    all_concepts = {}
    for layer_file in layer_files:
        with open(layer_file) as f:
            layer_data = json.load(f)
            layer = layer_data['metadata']['layer']

            for concept in layer_data['concepts']:
                sumo_term = concept['sumo_term']
                concept_to_layer[sumo_term] = layer
                all_concepts[sumo_term] = concept

                # Build parent->child mapping from category_children
                for child_name in concept.get('category_children', []):
                    parent_to_children[sumo_term].append(child_name)
                    children_to_parents[child_name].append(sumo_term)

    return concept_to_layer, children_to_parents, parent_to_children


def validate_hierarchy(
    concept_to_layer: Dict[str, int],
    children_to_parents: Dict[str, List[str]],
    parent_to_children: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Validate hierarchy and report issues.

    Returns dict of issues by category.
    """
    issues = defaultdict(list)

    # Check: Every non-layer-0 concept should have at least one parent
    for concept_name, layer in concept_to_layer.items():
        if layer == 0:
            # Layer 0 concepts shouldn't have parents
            if concept_name in children_to_parents:
                issues['layer0_with_parents'].append(
                    f"{concept_name}: Layer 0 but has parents {children_to_parents[concept_name]}"
                )
        else:
            # Non-layer-0 concepts MUST have parents
            if concept_name not in children_to_parents:
                issues['missing_parents'].append(
                    f"{concept_name} (L{layer}): No parent defined"
                )

    # Check: All children in category_children must exist
    for parent_name, children in parent_to_children.items():
        for child_name in children:
            if child_name not in concept_to_layer:
                issues['missing_children'].append(
                    f"{parent_name}: References non-existent child '{child_name}'"
                )

    # Check: All parents in children_to_parents must exist
    for child_name, parents in children_to_parents.items():
        for parent_name in parents:
            if parent_name not in concept_to_layer:
                issues['missing_parents_refs'].append(
                    f"{child_name}: References non-existent parent '{parent_name}'"
                )

    return issues


def fix_layer_files(
    layer_files: List[Path],
    children_to_parents: Dict[str, List[str]],
    output_dir: Path,
    dry_run: bool = False
):
    """
    Add parent_concepts field to each concept in layer files.
    """
    stats = {
        'concepts_updated': 0,
        'parents_added': 0,
        'layer_stats': {}
    }

    for layer_file in layer_files:
        with open(layer_file) as f:
            layer_data = json.load(f)

        layer = layer_data['metadata']['layer']
        updated_concepts = []
        layer_parent_count = 0

        for concept in layer_data['concepts']:
            sumo_term = concept['sumo_term']

            # Add parent_concepts field
            parent_concepts = children_to_parents.get(sumo_term, [])
            concept['parent_concepts'] = parent_concepts

            if parent_concepts:
                layer_parent_count += len(parent_concepts)
                stats['parents_added'] += len(parent_concepts)

            stats['concepts_updated'] += 1
            updated_concepts.append(concept)

        layer_data['concepts'] = updated_concepts

        # Update metadata
        layer_data['metadata']['concepts_with_parents'] = sum(
            1 for c in updated_concepts if c.get('parent_concepts')
        )
        layer_data['metadata']['total_parent_links'] = layer_parent_count

        stats['layer_stats'][layer] = {
            'total_concepts': len(updated_concepts),
            'with_parents': layer_data['metadata']['concepts_with_parents'],
            'total_parent_links': layer_parent_count
        }

        # Write updated file
        if not dry_run:
            output_file = output_dir / layer_file.name
            with open(output_file, 'w') as f:
                json.dump(layer_data, f, indent=2)
            print(f"✓ Updated {output_file.name}: {len(updated_concepts)} concepts, "
                  f"{layer_data['metadata']['concepts_with_parents']} with parents")
        else:
            print(f"[DRY RUN] Would update {layer_file.name}: {len(updated_concepts)} concepts, "
                  f"{layer_data['metadata']['concepts_with_parents']} with parents")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fix hierarchy parent links in layer files"
    )
    parser.add_argument('--layers-dir', type=Path,
                       default=Path('data/concept_graph/abstraction_layers'),
                       help='Directory containing layer*.json files')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: same as input, with backup)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate hierarchy, do not fix')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.layers_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("HIERARCHY PARENT LINK FIXER")
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

    # Build parent-child mappings
    print("\nBuilding parent-child mappings...")
    concept_to_layer, children_to_parents, parent_to_children = build_parent_child_mappings(layer_files)

    print(f"✓ Loaded {len(concept_to_layer)} concepts")
    print(f"✓ Found {len(parent_to_children)} parent concepts")
    print(f"✓ Found {len(children_to_parents)} child concepts")

    # Validate hierarchy
    print("\nValidating hierarchy...")
    issues = validate_hierarchy(concept_to_layer, children_to_parents, parent_to_children)

    if issues:
        print("\n⚠️  HIERARCHY ISSUES FOUND:")
        for issue_type, issue_list in issues.items():
            print(f"\n{issue_type}: {len(issue_list)} issues")
            for issue in issue_list[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(issue_list) > 10:
                print(f"  ... and {len(issue_list) - 10} more")
    else:
        print("✓ No hierarchy issues found")

    if args.validate_only:
        return 0

    # Create backup if not dry run and output is same as input
    if not args.dry_run and args.output_dir == args.layers_dir:
        backup_dir = args.layers_dir / 'backups'
        backup_dir.mkdir(exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        print(f"\nCreating backup in {backup_dir}...")
        for layer_file in layer_files:
            backup_file = backup_dir / f"{layer_file.stem}_{timestamp}.json"
            import shutil
            shutil.copy2(layer_file, backup_file)
        print(f"✓ Backed up {len(layer_files)} files")

    # Fix layer files
    print("\nAdding parent_concepts field to all concepts...")
    stats = fix_layer_files(layer_files, children_to_parents, args.output_dir, args.dry_run)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total concepts updated: {stats['concepts_updated']}")
    print(f"Total parent links added: {stats['parents_added']}")
    print(f"\nPer-layer statistics:")
    print(f"{'Layer':<8} {'Concepts':<12} {'With Parents':<15} {'Parent Links':<15} {'Coverage':<10}")
    print("-"*80)

    for layer in sorted(stats['layer_stats'].keys()):
        layer_stat = stats['layer_stats'][layer]
        total = layer_stat['total_concepts']
        with_parents = layer_stat['with_parents']
        coverage = 100 * with_parents / total if total > 0 else 0

        print(f"{layer:<8} {total:<12} {with_parents:<15} "
              f"{layer_stat['total_parent_links']:<15} {coverage:>6.1f}%")

    if not args.dry_run:
        print(f"\n✓ Updated layer files saved to: {args.output_dir}")
    else:
        print(f"\n[DRY RUN] No files were modified")

    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
