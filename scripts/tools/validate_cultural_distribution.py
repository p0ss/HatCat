#!/usr/bin/env python3
"""
Validate and analyze cultural distribution across poles after API enrichment.

This script checks if cultures are symmetrically distributed across negative/neutral/positive
poles and identifies imbalances that could lead to spurious cultural associations.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
ENRICHMENT_RESPONSES_DIR = PROJECT_ROOT / "data" / "enrichment_responses"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cultural_distribution_analysis.json"

# Thresholds
MAX_IMBALANCE_RATIO = 2.0  # Max acceptable ratio between max and min pole counts for a culture
MIN_REPRESENTATION = 3     # Minimum synsets per culture per pole


def extract_culture_from_synset_id(synset_id: str) -> str:
    """Extract culture identifier from synset_id.

    Examples:
        "chinese_mianzi.positive.001" -> "chinese"
        "japanese_wa.neutral.001" -> "japanese"
        "ubuntu.negative.001" -> "ubuntu"
    """
    # Split by first underscore or dot to get culture prefix
    parts = synset_id.split('_')
    if len(parts) > 1:
        # Format: culture_term.pole.number
        return parts[0].lower()
    else:
        # Format: term.pole.number (might be single-word cultural concept)
        parts = synset_id.split('.')
        return parts[0].lower() if len(parts) > 0 else "unknown"


def load_enrichment_responses() -> Dict[str, Dict[str, List[dict]]]:
    """Load all enrichment responses and organize by simplex and pole."""

    responses = defaultdict(lambda: defaultdict(list))

    if not ENRICHMENT_RESPONSES_DIR.exists():
        print(f"ERROR: Enrichment responses directory not found: {ENRICHMENT_RESPONSES_DIR}")
        return responses

    # Load all JSON response files
    for response_file in ENRICHMENT_RESPONSES_DIR.glob("*.json"):
        try:
            with open(response_file) as f:
                data = json.load(f)

            metadata = data.get('metadata', {})
            simplex = metadata.get('simplex', 'unknown')
            pole = metadata.get('pole', 'unknown')
            synsets = data.get('synsets', [])

            responses[simplex][pole].extend(synsets)

        except Exception as e:
            print(f"WARNING: Failed to load {response_file}: {e}")

    return responses


def analyze_cultural_distribution(responses: Dict[str, Dict[str, List[dict]]]) -> Dict:
    """Analyze cultural distribution across poles for each simplex."""

    analysis = {
        'simplexes': {},
        'overall_cultures': defaultdict(lambda: {'negative': 0, 'neutral': 0, 'positive': 0}),
        'imbalanced_cultures': [],
        'underrepresented_cultures': []
    }

    # Analyze each simplex
    for simplex, poles in responses.items():
        simplex_cultures = defaultdict(lambda: {'negative': 0, 'neutral': 0, 'positive': 0})

        for pole, synsets in poles.items():
            for synset in synsets:
                synset_id = synset.get('synset_id', '')
                culture = extract_culture_from_synset_id(synset_id)

                if culture and culture != 'unknown':
                    simplex_cultures[culture][pole] += 1
                    analysis['overall_cultures'][culture][pole] += 1

        # Calculate imbalance for each culture in this simplex
        simplex_analysis = {
            'total_synsets': sum(len(synsets) for synsets in poles.values()),
            'cultures': {},
            'imbalanced': [],
            'balanced': []
        }

        for culture, pole_counts in simplex_cultures.items():
            counts = [pole_counts['negative'], pole_counts['neutral'], pole_counts['positive']]
            max_count = max(counts)
            min_count = min(counts) if min(counts) > 0 else 1
            imbalance_ratio = max_count / min_count

            culture_data = {
                'negative': pole_counts['negative'],
                'neutral': pole_counts['neutral'],
                'positive': pole_counts['positive'],
                'total': sum(counts),
                'max': max_count,
                'min': min_count,
                'imbalance_ratio': imbalance_ratio,
                'is_balanced': imbalance_ratio <= MAX_IMBALANCE_RATIO,
                'is_underrepresented': min_count < MIN_REPRESENTATION
            }

            simplex_analysis['cultures'][culture] = culture_data

            if imbalance_ratio > MAX_IMBALANCE_RATIO:
                simplex_analysis['imbalanced'].append(culture)
            else:
                simplex_analysis['balanced'].append(culture)

        analysis['simplexes'][simplex] = simplex_analysis

    # Overall cultural analysis
    overall_imbalanced = []
    overall_underrepresented = []

    for culture, pole_counts in analysis['overall_cultures'].items():
        counts = [pole_counts['negative'], pole_counts['neutral'], pole_counts['positive']]
        max_count = max(counts)
        min_count = min(counts) if min(counts) > 0 else 1
        imbalance_ratio = max_count / min_count

        if imbalance_ratio > MAX_IMBALANCE_RATIO:
            overall_imbalanced.append({
                'culture': culture,
                'negative': pole_counts['negative'],
                'neutral': pole_counts['neutral'],
                'positive': pole_counts['positive'],
                'imbalance_ratio': imbalance_ratio
            })

        if min_count < MIN_REPRESENTATION:
            overall_underrepresented.append({
                'culture': culture,
                'negative': pole_counts['negative'],
                'neutral': pole_counts['neutral'],
                'positive': pole_counts['positive'],
                'min_count': min_count
            })

    analysis['imbalanced_cultures'] = sorted(overall_imbalanced, key=lambda x: x['imbalance_ratio'], reverse=True)
    analysis['underrepresented_cultures'] = sorted(overall_underrepresented, key=lambda x: x['min_count'])

    return analysis


def print_analysis_summary(analysis: Dict):
    """Print human-readable summary of cultural distribution analysis."""

    print("=" * 80)
    print("CULTURAL DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Overall statistics
    total_cultures = len(analysis['overall_cultures'])
    num_imbalanced = len(analysis['imbalanced_cultures'])
    num_underrepresented = len(analysis['underrepresented_cultures'])

    print(f"\nTotal unique cultures detected: {total_cultures}")
    print(f"Imbalanced cultures (>{MAX_IMBALANCE_RATIO}x ratio): {num_imbalanced}")
    print(f"Underrepresented cultures (<{MIN_REPRESENTATION} synsets/pole): {num_underrepresented}")

    # Imbalanced cultures
    if analysis['imbalanced_cultures']:
        print(f"\n{'=' * 80}")
        print("IMBALANCED CULTURES (need correction)")
        print("=" * 80)

        for item in analysis['imbalanced_cultures'][:10]:  # Top 10
            culture = item['culture']
            neg, neu, pos = item['negative'], item['neutral'], item['positive']
            ratio = item['imbalance_ratio']

            print(f"\n{culture:20} | Ratio: {ratio:.1f}x")
            print(f"  Negative: {neg:3} | Neutral: {neu:3} | Positive: {pos:3}")

            # Identify which pole is over/under represented
            max_pole = max([('negative', neg), ('neutral', neu), ('positive', pos)], key=lambda x: x[1])
            min_pole = min([('negative', neg), ('neutral', neu), ('positive', pos)], key=lambda x: x[1])

            print(f"  Problem: Too many '{max_pole[0]}' ({max_pole[1]}), too few '{min_pole[0]}' ({min_pole[1]})")

    # Underrepresented cultures
    if analysis['underrepresented_cultures']:
        print(f"\n{'=' * 80}")
        print("UNDERREPRESENTED CULTURES (need more synsets)")
        print("=" * 80)

        for item in analysis['underrepresented_cultures'][:10]:
            culture = item['culture']
            neg, neu, pos = item['negative'], item['neutral'], item['positive']
            min_count = item['min_count']

            print(f"\n{culture:20} | Min: {min_count}")
            print(f"  Negative: {neg:3} | Neutral: {neu:3} | Positive: {pos:3}")

    # Per-simplex summary
    print(f"\n{'=' * 80}")
    print("PER-SIMPLEX SUMMARY")
    print("=" * 80)

    for simplex, simplex_data in sorted(analysis['simplexes'].items()):
        num_cultures = len(simplex_data['cultures'])
        num_balanced = len(simplex_data['balanced'])
        num_imbalanced = len(simplex_data['imbalanced'])

        print(f"\n{simplex}:")
        print(f"  Total synsets: {simplex_data['total_synsets']}")
        print(f"  Cultures: {num_cultures} ({num_balanced} balanced, {num_imbalanced} imbalanced)")

        if num_imbalanced > 0:
            print(f"  Imbalanced: {', '.join(simplex_data['imbalanced'][:5])}")


def generate_correction_plan(analysis: Dict) -> List[Dict]:
    """Generate targeted API requests to correct cultural imbalances."""

    correction_requests = []

    for item in analysis['imbalanced_cultures']:
        culture = item['culture']
        pole_counts = {
            'negative': item['negative'],
            'neutral': item['neutral'],
            'positive': item['positive']
        }

        # Target: bring all poles to the maximum count
        target = max(pole_counts.values())

        for pole, count in pole_counts.items():
            needed = target - count

            if needed > 0:
                correction_requests.append({
                    'culture': culture,
                    'pole': pole,
                    'current_count': count,
                    'target_count': target,
                    'needed': needed,
                    'priority': 'high' if needed > 5 else 'medium'
                })

    # Sort by priority and needed count
    correction_requests.sort(key=lambda x: (x['priority'] == 'high', x['needed']), reverse=True)

    return correction_requests


def main():
    print("=" * 80)
    print("CULTURAL DISTRIBUTION VALIDATION")
    print("=" * 80)

    # Load enrichment responses
    print(f"\n1. Loading enrichment responses from {ENRICHMENT_RESPONSES_DIR}...")
    responses = load_enrichment_responses()

    if not responses:
        print("\nERROR: No enrichment responses found!")
        print("Run scripts/execute_multicultural_api_calls.py first.")
        sys.exit(1)

    num_simplexes = len(responses)
    total_synsets = sum(len(synsets) for poles in responses.values() for synsets in poles.values())
    print(f"✓ Loaded {total_synsets} synsets across {num_simplexes} simplexes")

    # Analyze distribution
    print("\n2. Analyzing cultural distribution across poles...")
    analysis = analyze_cultural_distribution(responses)

    # Print summary
    print("\n")
    print_analysis_summary(analysis)

    # Generate correction plan
    print(f"\n{'=' * 80}")
    print("CORRECTION PLAN")
    print("=" * 80)

    correction_requests = generate_correction_plan(analysis)

    if correction_requests:
        print(f"\nGenerated {len(correction_requests)} correction requests:")

        for i, req in enumerate(correction_requests[:10], 1):
            print(f"\n[{i}] {req['culture']:15} -> {req['pole']:8} pole")
            print(f"    Current: {req['current_count']:3} | Target: {req['target_count']:3} | Need: {req['needed']:3} more synsets")
    else:
        print("\n✓ No corrections needed! Cultural distribution is balanced.")

    # Save analysis
    output_data = {
        'analysis': analysis,
        'correction_requests': correction_requests,
        'thresholds': {
            'max_imbalance_ratio': MAX_IMBALANCE_RATIO,
            'min_representation': MIN_REPRESENTATION
        }
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ Analysis saved to: {OUTPUT_PATH}")

    if correction_requests:
        print(f"\nNext step: Generate {len(correction_requests)} targeted API requests to balance cultures")
        print("Run: python scripts/generate_cultural_corrections.py")
    else:
        print("\n✓ Cultural distribution is balanced! Ready to merge responses.")
        print("Run: python scripts/merge_enrichment_responses.py")

    print("=" * 80)

    # Exit code indicates if corrections needed
    sys.exit(1 if correction_requests else 0)


if __name__ == '__main__':
    main()
