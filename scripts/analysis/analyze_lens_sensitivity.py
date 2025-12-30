#!/usr/bin/env python3
"""
Analyze Lens Sensitivity Issues

Diagnoses why lenses are over-firing or under-firing by examining:
1. Training data characteristics (synset count, layer, category type)
2. Score distributions (positive vs negative separation)
3. Concept hierarchy position
4. Common patterns among problematic lenses
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_calibration_results(calibration_file: Path) -> Dict:
    """Load calibration results JSON."""
    with open(calibration_file) as f:
        return json.load(f)


def load_concept_metadata(layers_dir: Path) -> Dict:
    """Load concept metadata from layer files."""
    metadata = {}

    for layer_file in sorted(layers_dir.glob("layer*.json")):
        with open(layer_file) as f:
            layer_data = json.load(f)
            layer = layer_data['metadata']['layer']

            for concept in layer_data.get('concepts', []):
                sumo_term = concept['sumo_term']
                key = f"{sumo_term}_L{layer}"
                metadata[key] = {
                    'sumo_term': sumo_term,
                    'layer': layer,
                    'synsets': concept.get('synsets', []),
                    'synset_count': len(concept.get('synsets', [])),
                    'is_category_lens': concept.get('is_category_lens', False),
                    'parent_concepts': concept.get('parent_concepts', []),
                    'category_children': concept.get('category_children', []),
                }

    return metadata


def analyze_over_firing_lenses(results: Dict, metadata: Dict):
    """Analyze why lenses are over-firing."""
    print("\n" + "="*80)
    print("OVER-FIRING LENSS ANALYSIS")
    print("="*80)

    over_firing = [
        (key, data) for key, data in results['results'].items()
        if data['category'] == 'over_firing'
    ]

    print(f"\nFound {len(over_firing)} over-firing lenses\n")

    for key, data in sorted(over_firing, key=lambda x: x[1]['metrics']['fp_rate'], reverse=True):
        concept = data['concept']
        layer = data['layer']
        metrics = data['metrics']
        meta_key = f"{concept}_L{layer}"
        meta = metadata.get(meta_key, {})

        print(f"\n{key}:")
        print(f"  Concept: {concept} (Layer {layer})")
        print(f"  FP Rate: {metrics['fp_rate']:.1%}")
        print(f"  TP Rate: {metrics['tp_rate']:.1%}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        print(f"\n  Score Distribution:")
        print(f"    Avg Positive:   {metrics['avg_positive_score']:.4f}")
        print(f"    Avg Negative:   {metrics['avg_negative_score']:.4f}")
        print(f"    Avg Irrelevant: {metrics['avg_irrelevant_score']:.4f}")
        print(f"    Separation:     {metrics['avg_positive_score'] - metrics['avg_negative_score']:.4f}")

        if meta:
            print(f"\n  Training Data:")
            print(f"    Synsets: {meta['synset_count']}")
            print(f"    Category Lens: {meta['is_category_lens']}")
            print(f"    Parents: {len(meta['parent_concepts'])}")
            print(f"    Children: {len(meta['category_children'])}")

        # Diagnose root cause
        print(f"\n  Diagnosis:")
        if layer <= 1:
            print(f"    ⚠️  Very abstract Layer {layer} concept - expected to be broad")
        if meta and meta['synset_count'] < 5:
            print(f"    ⚠️  Low synset count ({meta['synset_count']}) - may lack specificity")
        if metrics['avg_positive_score'] - metrics['avg_negative_score'] < 0.2:
            print(f"    ⚠️  Poor score separation - lens can't distinguish concept well")
        if meta and len(meta['category_children']) > 20:
            print(f"    ⚠️  Many children ({len(meta['category_children'])}) - very broad category")


def analyze_under_firing_lenses(results: Dict, metadata: Dict):
    """Analyze why lenses are under-firing."""
    print("\n" + "="*80)
    print("UNDER-FIRING LENSS ANALYSIS")
    print("="*80)

    under_firing = [
        (key, data) for key, data in results['results'].items()
        if data['category'] == 'under_firing'
    ]

    print(f"\nFound {len(under_firing)} under-firing lenses\n")

    # Group by common characteristics
    by_layer = defaultdict(list)
    by_synset_count = defaultdict(list)

    for key, data in under_firing:
        layer = data['layer']
        meta_key = f"{data['concept']}_L{layer}"
        meta = metadata.get(meta_key, {})

        by_layer[layer].append(key)
        if meta:
            synset_count = meta['synset_count']
            if synset_count == 0:
                by_synset_count['0'].append(key)
            elif synset_count < 3:
                by_synset_count['1-2'].append(key)
            elif synset_count < 10:
                by_synset_count['3-9'].append(key)
            else:
                by_synset_count['10+'].append(key)

    print("Distribution by Layer:")
    for layer in sorted(by_layer.keys()):
        print(f"  Layer {layer}: {len(by_layer[layer])} lenses")

    print("\nDistribution by Synset Count:")
    for count_range in ['0', '1-2', '3-9', '10+']:
        if count_range in by_synset_count:
            print(f"  {count_range} synsets: {len(by_synset_count[count_range])} lenses")

    # Show detailed analysis for worst offenders
    print("\nTop 10 Worst Under-Firing Lenses:")

    for key, data in sorted(under_firing, key=lambda x: x[1]['metrics']['tp_rate'])[:10]:
        concept = data['concept']
        layer = data['layer']
        metrics = data['metrics']
        meta_key = f"{concept}_L{layer}"
        meta = metadata.get(meta_key, {})

        print(f"\n{key}:")
        print(f"  Concept: {concept} (Layer {layer})")
        print(f"  TP Rate: {metrics['tp_rate']:.1%}")
        print(f"  FP Rate: {metrics['fp_rate']:.1%}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        print(f"\n  Score Distribution:")
        print(f"    Avg Positive:   {metrics['avg_positive_score']:.4f}")
        print(f"    Avg Negative:   {metrics['avg_negative_score']:.4f}")
        print(f"    Single Term:    {metrics['single_term_score']:.4f}")

        if meta:
            print(f"\n  Training Data:")
            print(f"    Synsets: {meta['synset_count']}")
            print(f"    Category Lens: {meta['is_category_lens']}")
            print(f"    Sample synsets: {', '.join(meta['synsets'][:3])}")

        # Diagnose root cause
        print(f"\n  Diagnosis:")
        if meta and meta['synset_count'] == 0:
            print(f"    ❌ NO training synsets - lens has no examples!")
        elif meta and meta['synset_count'] < 3:
            print(f"    ⚠️  Very few synsets ({meta['synset_count']}) - insufficient training data")
        if metrics['avg_positive_score'] < 0.5:
            print(f"    ⚠️  Low positive scores - lens doesn't recognize its own concept")
        if metrics['single_term_score'] < 0.5:
            print(f"    ⚠️  Doesn't activate on concept name itself - fundamental issue")


def analyze_patterns(results: Dict, metadata: Dict):
    """Analyze common patterns across all lenses."""
    print("\n" + "="*80)
    print("OVERALL PATTERNS")
    print("="*80)

    # Analyze by layer
    by_layer = defaultdict(lambda: {'well': 0, 'marginal': 0, 'over': 0, 'under': 0, 'total': 0})

    for key, data in results['results'].items():
        layer = data['layer']
        category = data['category']
        by_layer[layer]['total'] += 1

        if category == 'well_calibrated':
            by_layer[layer]['well'] += 1
        elif category == 'marginal':
            by_layer[layer]['marginal'] += 1
        elif category == 'over_firing':
            by_layer[layer]['over'] += 1
        elif category == 'under_firing':
            by_layer[layer]['under'] += 1

    print("\nCalibration by Layer:")
    print(f"{'Layer':<8} {'Total':<8} {'Well':<8} {'Marginal':<10} {'Over':<8} {'Under':<8} {'% Well':<8}")
    print("-"*80)

    for layer in sorted(by_layer.keys()):
        stats = by_layer[layer]
        pct_well = 100 * stats['well'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{layer:<8} {stats['total']:<8} {stats['well']:<8} {stats['marginal']:<10} "
              f"{stats['over']:<8} {stats['under']:<8} {pct_well:>6.1f}%")

    # Analyze abstract vs specific concepts
    print("\n\nAbstract (Layer 0-1) vs Specific (Layer 2+) Concepts:")
    abstract_over = sum(1 for k, v in results['results'].items()
                       if v['layer'] <= 1 and v['category'] == 'over_firing')
    specific_over = sum(1 for k, v in results['results'].items()
                       if v['layer'] >= 2 and v['category'] == 'over_firing')
    abstract_total = sum(1 for k, v in results['results'].items() if v['layer'] <= 1)
    specific_total = sum(1 for k, v in results['results'].items() if v['layer'] >= 2)

    print(f"  Abstract concepts (L0-1): {abstract_over}/{abstract_total} over-firing "
          f"({100*abstract_over/abstract_total:.1f}%)")
    print(f"  Specific concepts (L2+):  {specific_over}/{specific_total} over-firing "
          f"({100*specific_over/specific_total:.1f}%)")


def main():
    # Paths
    calibration_file = Path("results/lens_calibration/calibration_20251119_224541.json")
    layers_dir = Path("data/concept_graph/abstraction_layers")

    if not calibration_file.exists():
        print(f"ERROR: Calibration file not found: {calibration_file}")
        return 1

    print("Loading calibration results...")
    results = load_calibration_results(calibration_file)

    print("Loading concept metadata...")
    metadata = load_concept_metadata(layers_dir)

    print(f"\n✓ Loaded {results['total_lenses']} lens calibration results")
    print(f"✓ Loaded metadata for {len(metadata)} concepts")

    # Analyze problematic lenses
    analyze_over_firing_lenses(results, metadata)
    analyze_under_firing_lenses(results, metadata)
    analyze_patterns(results, metadata)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
