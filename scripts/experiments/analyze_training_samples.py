#!/usr/bin/env python3
"""
Analyze training samples saved during lens training.

This script loads samples saved with the --save-samples flag and provides
insights about sample quality, concept coverage, and potential issues.

Usage:
    python scripts/experiments/analyze_training_samples.py \
        --samples-dir lens_packs/apertus-8b_first-light/training_samples/first-light

    # Show only problematic concepts (< 50% good samples)
    python scripts/experiments/analyze_training_samples.py \
        --samples-dir ... --problematic-only

    # Show example samples for a specific concept
    python scripts/experiments/analyze_training_samples.py \
        --samples-dir ... --concept Deception --show-examples 5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Add src/map to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "map"))

from training.sample_quality import load_training_samples, SampleQuality


def analyze_concept_quality(samples: List[Dict]) -> Dict[str, Dict]:
    """Group samples by concept and compute quality statistics."""
    by_concept = defaultdict(list)
    for s in samples:
        by_concept[s['concept']].append(s)

    stats = {}
    for concept, concept_samples in by_concept.items():
        total = len(concept_samples)
        good = sum(1 for s in concept_samples if s['quality']['is_good'])
        empty = sum(1 for s in concept_samples if s['quality']['is_empty'])
        too_short = sum(1 for s in concept_samples if s['quality']['is_too_short'])
        collapsed = sum(1 for s in concept_samples if s['quality']['is_collapsed'])
        low_diversity = sum(1 for s in concept_samples if s['quality']['is_low_diversity'])

        # Compute average quality score
        avg_quality = sum(s['quality']['quality_score'] for s in concept_samples) / total

        # Separate positive and negative samples
        pos_samples = [s for s in concept_samples if s['label'] == 1]
        neg_samples = [s for s in concept_samples if s['label'] == 0]
        pos_good = sum(1 for s in pos_samples if s['quality']['is_good']) if pos_samples else 0
        neg_good = sum(1 for s in neg_samples if s['quality']['is_good']) if neg_samples else 0

        stats[concept] = {
            'total': total,
            'good': good,
            'good_rate': good / total if total > 0 else 0,
            'empty': empty,
            'too_short': too_short,
            'collapsed': collapsed,
            'low_diversity': low_diversity,
            'avg_quality': avg_quality,
            'positive_total': len(pos_samples),
            'positive_good': pos_good,
            'negative_total': len(neg_samples),
            'negative_good': neg_good,
        }

    return stats


def print_summary(stats: Dict[str, Dict], show_all: bool = False):
    """Print summary statistics."""
    total_concepts = len(stats)
    total_samples = sum(s['total'] for s in stats.values())
    total_good = sum(s['good'] for s in stats.values())

    print("\n" + "=" * 80)
    print("SAMPLE QUALITY SUMMARY")
    print("=" * 80)
    print(f"\nConcepts: {total_concepts}")
    print(f"Total samples: {total_samples}")
    print(f"Good samples: {total_good} ({100*total_good/total_samples:.1f}%)")
    print(f"Bad samples: {total_samples - total_good} ({100*(total_samples-total_good)/total_samples:.1f}%)")

    # Issue breakdown
    total_empty = sum(s['empty'] for s in stats.values())
    total_short = sum(s['too_short'] for s in stats.values())
    total_collapsed = sum(s['collapsed'] for s in stats.values())
    total_low_div = sum(s['low_diversity'] for s in stats.values())

    print("\nIssue breakdown:")
    print(f"  Empty: {total_empty}")
    print(f"  Too short: {total_short}")
    print(f"  Collapsed/repetitive: {total_collapsed}")
    print(f"  Low diversity: {total_low_div}")

    # Find problematic concepts
    problematic = [(c, s) for c, s in stats.items() if s['good_rate'] < 0.5]
    problematic.sort(key=lambda x: x[1]['good_rate'])

    if problematic:
        print(f"\n{'=' * 80}")
        print(f"PROBLEMATIC CONCEPTS ({len(problematic)} with <50% good samples)")
        print("=" * 80)
        for concept, s in problematic[:20]:
            print(f"\n  {concept}:")
            print(f"    Good: {s['good']}/{s['total']} ({100*s['good_rate']:.0f}%)")
            print(f"    Pos: {s['positive_good']}/{s['positive_total']} good | "
                  f"Neg: {s['negative_good']}/{s['negative_total']} good")
            if s['collapsed'] > 0:
                print(f"    Collapsed: {s['collapsed']}")
            if s['low_diversity'] > 0:
                print(f"    Low diversity: {s['low_diversity']}")

    # Top quality concepts
    if show_all:
        high_quality = [(c, s) for c, s in stats.items() if s['good_rate'] >= 0.9]
        high_quality.sort(key=lambda x: x[1]['good_rate'], reverse=True)

        print(f"\n{'=' * 80}")
        print(f"HIGH QUALITY CONCEPTS ({len(high_quality)} with >=90% good samples)")
        print("=" * 80)
        for concept, s in high_quality[:20]:
            print(f"  {concept}: {100*s['good_rate']:.0f}% good ({s['good']}/{s['total']})")


def show_examples(samples: List[Dict], concept: str, n: int = 5):
    """Show example samples for a specific concept."""
    concept_samples = [s for s in samples if s['concept'] == concept]

    if not concept_samples:
        print(f"\nNo samples found for concept: {concept}")
        return

    print(f"\n{'=' * 80}")
    print(f"EXAMPLES FOR: {concept}")
    print(f"{'=' * 80}")

    # Show good examples
    good = [s for s in concept_samples if s['quality']['is_good']]
    bad = [s for s in concept_samples if not s['quality']['is_good']]

    print(f"\nGood samples ({len(good)} total):")
    for i, s in enumerate(good[:n]):
        label = "POS" if s['label'] == 1 else "NEG"
        print(f"\n  [{label}] Prompt: {s['prompt'][:100]}...")
        print(f"       Output: {s['generated_text'][:150]}...")

    print(f"\n{'=' * 60}")
    print(f"\nBad samples ({len(bad)} total):")
    for i, s in enumerate(bad[:n]):
        label = "POS" if s['label'] == 1 else "NEG"
        issues = []
        if s['quality']['is_empty']:
            issues.append("empty")
        if s['quality']['is_too_short']:
            issues.append("too_short")
        if s['quality']['is_collapsed']:
            issues.append("collapsed")
        if s['quality']['is_low_diversity']:
            issues.append("low_diversity")
        issue_str = ", ".join(issues)
        print(f"\n  [{label}] Issues: {issue_str}")
        print(f"       Prompt: {s['prompt'][:100]}...")
        print(f"       Output: {s['generated_text'][:150]}...")


def analyze_by_layer(samples: List[Dict]) -> Dict[int, Dict]:
    """Analyze quality statistics per layer."""
    by_layer = defaultdict(list)
    for s in samples:
        by_layer[s['layer']].append(s)

    stats = {}
    for layer, layer_samples in sorted(by_layer.items()):
        total = len(layer_samples)
        good = sum(1 for s in layer_samples if s['quality']['is_good'])
        stats[layer] = {
            'total': total,
            'good': good,
            'good_rate': good / total if total > 0 else 0,
            'concepts': len(set(s['concept'] for s in layer_samples)),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze training samples quality")
    parser.add_argument('--samples-dir', type=str, required=True,
                        help='Directory containing training samples')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Filter to specific layers')
    parser.add_argument('--concept', type=str, default=None,
                        help='Show examples for a specific concept')
    parser.add_argument('--show-examples', type=int, default=5,
                        help='Number of examples to show (default: 5)')
    parser.add_argument('--problematic-only', action='store_true',
                        help='Only show problematic concepts')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all concept details including high quality')
    parser.add_argument('--only-good', action='store_true',
                        help='Only load good samples')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Output analysis to JSON file')

    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        sys.exit(1)

    print(f"Loading samples from: {samples_dir}")

    # Load all samples (including bad ones for analysis)
    samples = load_training_samples(
        samples_dir,
        layers=args.layers,
        only_good=args.only_good,
    )

    if not samples:
        print("No samples found!")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples")

    # Show examples for specific concept if requested
    if args.concept:
        show_examples(samples, args.concept, args.show_examples)
        return

    # Analyze by layer
    layer_stats = analyze_by_layer(samples)
    print("\n" + "=" * 80)
    print("QUALITY BY LAYER")
    print("=" * 80)
    for layer, stats in layer_stats.items():
        print(f"  Layer {layer}: {stats['good']}/{stats['total']} good ({100*stats['good_rate']:.1f}%), "
              f"{stats['concepts']} concepts")

    # Analyze by concept
    concept_stats = analyze_concept_quality(samples)

    if args.problematic_only:
        problematic = {c: s for c, s in concept_stats.items() if s['good_rate'] < 0.5}
        print_summary(problematic, show_all=False)
    else:
        print_summary(concept_stats, show_all=args.show_all)

    # Output to JSON if requested
    if args.output_json:
        output = {
            'samples_dir': str(samples_dir),
            'total_samples': len(samples),
            'layer_stats': layer_stats,
            'concept_stats': concept_stats,
        }
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nAnalysis saved to: {args.output_json}")


if __name__ == "__main__":
    main()
