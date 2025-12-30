#!/usr/bin/env python3
"""
Analyze correlation between WordNet-patched concepts and training failures.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def parse_training_log(log_file: str):
    """Extract concept names and their training outcomes from log."""

    concepts_attempted = []
    concepts_with_errors = []
    concepts_graduated = defaultdict(list)  # tier -> [concepts]

    current_concept = None

    with open(log_file, 'r') as f:
        for line in f:
            # Match concept being trained
            match = re.match(r'\[(\d+)/\d+\] Training: (.+)', line)
            if match:
                current_concept = match.group(2)
                concepts_attempted.append(current_concept)
                continue

            # Match errors
            if current_concept and 'ERROR' in line:
                concepts_with_errors.append(current_concept)
                continue

            # Match graduation
            if current_concept and '-tier acceptable' in line:
                tier_match = re.search(r'([A-F][+-]?)-tier acceptable', line)
                if tier_match:
                    tier = tier_match.group(1)
                    concepts_graduated[tier].append(current_concept)

    return {
        'attempted': concepts_attempted,
        'with_errors': concepts_with_errors,
        'graduated': dict(concepts_graduated)
    }


def load_patched_concepts(suggestions_file: str):
    """Load list of concepts that were WordNet-patched."""
    with open(suggestions_file, 'r') as f:
        suggestions = json.load(f)

    return set(s['sumo_term'] for s in suggestions)


def analyze_correlation(log_file: str, suggestions_file: str):
    """Analyze correlation between patched concepts and training outcomes."""

    print("Analyzing training log correlation with WordNet patch...")
    print("=" * 80)

    # Load data
    training_data = parse_training_log(log_file)
    patched_concepts = load_patched_concepts(suggestions_file)

    attempted = set(training_data['attempted'])
    with_errors = set(training_data['with_errors'])

    # Calculate overlaps
    attempted_and_patched = attempted & patched_concepts
    errors_and_patched = with_errors & patched_concepts

    # Stats
    print(f"\nTraining Statistics:")
    print(f"  Concepts attempted: {len(attempted)}")
    print(f"  Concepts with errors: {len(with_errors)}")
    print(f"  Error rate: {len(with_errors)/len(attempted)*100:.1f}%")

    print(f"\nWordNet Patch Statistics:")
    print(f"  Concepts patched: {len(patched_concepts)}")
    print(f"  Patched concepts attempted: {len(attempted_and_patched)}")
    print(f"  Patched concepts with errors: {len(errors_and_patched)}")

    if attempted_and_patched:
        patch_error_rate = len(errors_and_patched) / len(attempted_and_patched) * 100
        print(f"  Patched error rate: {patch_error_rate:.1f}%")

    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    if len(with_errors) > 0:
        pct_of_errors_are_patched = len(errors_and_patched) / len(with_errors) * 100
        print(f"  {pct_of_errors_are_patched:.1f}% of errors were patched concepts")

    if attempted_and_patched:
        print(f"\nPatched concepts breakdown:")
        graduated_patched = 0
        for tier, concepts in training_data['graduated'].items():
            tier_patched = len(set(concepts) & patched_concepts)
            if tier_patched > 0:
                print(f"    {tier}-tier: {tier_patched} concepts")
                graduated_patched += tier_patched

        failed_patched = len(attempted_and_patched) - graduated_patched - len(errors_and_patched)
        print(f"    Failed/ongoing: {failed_patched} concepts")
        print(f"    Errors: {len(errors_and_patched)} concepts")

    # Sample of patched concepts with errors (if any)
    if errors_and_patched:
        print(f"\nSample of patched concepts with errors (first 10):")
        for concept in sorted(errors_and_patched)[:10]:
            print(f"  - {concept}")

    # Sample of patched concepts that succeeded
    succeeded_patched = attempted_and_patched - with_errors
    if succeeded_patched:
        print(f"\nSample of patched concepts that trained successfully (first 10):")
        for concept in sorted(succeeded_patched)[:10]:
            print(f"  - {concept}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='logs/test_incremental.log')
    args = parser.parse_args()

    log_file = Path(__file__).parent.parent / args.log
    suggestions_file = Path(__file__).parent.parent / 'results' / 'wordnet_patch_suggestions.json'

    analyze_correlation(log_file, suggestions_file)
