#!/usr/bin/env python3
"""
Analyze the impact of adding new SUMO KIF files to the lens set.

Reports:
1. New concepts introduced by each KIF file
2. Overlap with existing concepts
3. Parent chain analysis (how many existing concepts would be affected)
4. Estimated training time for incremental vs full rebuild
5. Hierarchy integrity checks
"""

import re
from pathlib import Path
from collections import defaultdict, Counter
import json


def parse_kif_file(kif_path):
    """Extract subclass and instance relationships from a KIF file."""
    relationships = []

    with open(kif_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Match (subclass Child Parent)
    subclass_pattern = r'\(subclass\s+([A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]+)\)'
    for match in re.finditer(subclass_pattern, content):
        child, parent = match.groups()
        relationships.append(('subclass', child, parent))

    # Match (instance Individual Class)
    instance_pattern = r'\(instance\s+([A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]+)\)'
    for match in re.finditer(instance_pattern, content):
        individual, cls = match.groups()
        relationships.append(('instance', individual, cls))

    return relationships


def load_existing_concepts():
    """Load concepts from current layer files."""
    layer_dir = Path('data/concept_graph/abstraction_layers')
    existing_concepts = set()

    for layer_file in layer_dir.glob('layer*.json'):
        with open(layer_file) as f:
            data = json.load(f)
            for concept in data['concepts']:
                if concept.get('is_category_lens', False):
                    existing_concepts.add(concept['sumo_term'])

    return existing_concepts


def build_parent_map(relationships):
    """Build child -> parents mapping."""
    parent_map = defaultdict(set)

    for rel_type, child, parent in relationships:
        parent_map[child].add(parent)

    return parent_map


def find_parent_chain(concept, parent_map, max_depth=10):
    """Find all ancestors of a concept up to max_depth."""
    ancestors = set()
    queue = [(concept, 0)]
    visited = {concept}

    while queue:
        current, depth = queue.pop(0)

        if depth >= max_depth:
            continue

        for parent in parent_map.get(current, []):
            if parent not in visited:
                visited.add(parent)
                ancestors.add(parent)
                queue.append((parent, depth + 1))

    return ancestors


def main():
    print("=" * 80)
    print("SUMO KIF EXPANSION ANALYSIS")
    print("=" * 80)

    # Load existing concepts
    print("\n[1/5] Loading existing concepts from layer files...")
    existing_concepts = load_existing_concepts()
    print(f"  Found {len(existing_concepts)} existing category concepts")

    # Parse all KIF files
    print("\n[2/5] Parsing SUMO KIF files...")
    kif_dir = Path('data/concept_graph/sumo_source')

    # New KIF files (recently added)
    new_kif_files = [
        'Society.kif',
        'Biography.kif',
        'Government.kif',
        'Justice.kif',
        'Law.kif',
        'engineering.kif',
        'naics.kif',
        'MilitaryProcesses.kif',
        'Facebook.kif',
        'Communications.kif',
        'UXExperimentalTerms.kif',
        'ComputerInput.kif',
        'CCTrep.kif',
        'Media.kif',
        'FinancialOntology.kif',
        'QoSontology.kif',
        'CountriesAndRegions.kif',
        'Economy.kif',
        'GDPRTerms.kif',
        'Capabilities.kif',
    ]

    all_relationships = []
    new_file_concepts = defaultdict(set)

    for kif_file in kif_dir.glob('*.kif'):
        rels = parse_kif_file(kif_file)
        all_relationships.extend(rels)

        # Track concepts from new files
        if kif_file.name in new_kif_files:
            for rel_type, child, parent in rels:
                new_file_concepts[kif_file.name].add(child)

    print(f"  Parsed {len(list(kif_dir.glob('*.kif')))} KIF files")
    print(f"  Found {len(all_relationships)} total relationships")

    # Build parent map
    print("\n[3/5] Building concept hierarchy...")
    parent_map = build_parent_map(all_relationships)
    all_sumo_concepts = set(parent_map.keys())
    print(f"  Total SUMO concepts: {len(all_sumo_concepts)}")

    # Analyze new concepts
    print("\n[4/5] Analyzing new concepts by file...")
    print("-" * 80)

    total_new_concepts = set()
    total_new_count = 0
    total_overlap_count = 0

    for kif_file in sorted(new_kif_files):
        if kif_file not in new_file_concepts:
            print(f"\n{kif_file}: NOT FOUND")
            continue

        file_concepts = new_file_concepts[kif_file]
        new_concepts = file_concepts - existing_concepts
        overlap_concepts = file_concepts & existing_concepts

        total_new_concepts.update(new_concepts)
        total_new_count += len(new_concepts)
        total_overlap_count += len(overlap_concepts)

        print(f"\n{kif_file}:")
        print(f"  Total concepts: {len(file_concepts)}")
        print(f"  New concepts: {len(new_concepts)}")
        print(f"  Overlap with existing: {len(overlap_concepts)}")

        if len(new_concepts) > 0:
            # Sample new concepts
            sample_size = min(5, len(new_concepts))
            print(f"  Sample new: {', '.join(list(new_concepts)[:sample_size])}")

    # Parent chain analysis
    print("\n" + "=" * 80)
    print("[5/5] Analyzing parent chain impact...")
    print("-" * 80)

    affected_parents = set()
    for new_concept in total_new_concepts:
        ancestors = find_parent_chain(new_concept, parent_map)
        affected_parents.update(ancestors)

    # Filter to only existing concepts that would need updates
    affected_existing = affected_parents & existing_concepts

    print(f"\nNew concepts introduced: {len(total_new_concepts)}")
    print(f"Existing concepts that would be affected: {len(affected_existing)}")
    print(f"Total concepts to train/retrain: {len(total_new_concepts) + len(affected_existing)}")

    # Estimate training time
    avg_time_per_concept = 30  # seconds (rough estimate)
    total_training_concepts = len(total_new_concepts) + len(affected_existing)
    estimated_hours = (total_training_concepts * avg_time_per_concept) / 3600

    print(f"\nEstimated training time (incremental): ~{estimated_hours:.1f} hours")

    # Full rebuild estimate
    full_rebuild_concepts = len(existing_concepts) + len(total_new_concepts)
    full_rebuild_hours = (full_rebuild_concepts * avg_time_per_concept) / 3600
    print(f"Estimated training time (full rebuild): ~{full_rebuild_hours:.1f} hours")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if len(total_new_concepts) < 500 and len(affected_existing) < 200:
        print("\n✓ V3.5 (Incremental) appears FEASIBLE")
        print(f"  - {len(total_new_concepts)} new concepts to add")
        print(f"  - {len(affected_existing)} existing concepts to update")
        print(f"  - ~{estimated_hours:.1f} hours training time")
        print("\n  However, consider V4 if:")
        print("  - You want to optimize layer assignments")
        print("  - You've found issues in v3 structure")
        print("  - You want cleaner hierarchy validation")
    else:
        print("\n✓ V4 (Full Rebuild) is RECOMMENDED")
        print(f"  - {len(total_new_concepts)} new concepts is substantial")
        print(f"  - {len(affected_existing)} affected parents creates complexity")
        print(f"  - Full rebuild (~{full_rebuild_hours:.1f} hours) only ~{full_rebuild_hours - estimated_hours:.1f} hours more")
        print("\n  Benefits of V4:")
        print("  - Clean hierarchy validation")
        print("  - Optimized layer assignments")
        print("  - No legacy structure issues")
        print("  - Opportunity to apply lessons from v3")

    print("\n" + "=" * 80)

    # Save detailed results
    output_file = Path('results/kif_expansion_analysis.json')
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'existing_concepts': len(existing_concepts),
            'new_concepts': len(total_new_concepts),
            'affected_existing': len(affected_existing),
            'total_to_train': total_training_concepts,
            'estimated_incremental_hours': estimated_hours,
            'estimated_full_rebuild_hours': full_rebuild_hours,
            'new_concepts_list': sorted(list(total_new_concepts)),
            'affected_existing_list': sorted(list(affected_existing)),
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
