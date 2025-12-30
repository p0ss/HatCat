#!/usr/bin/env python3
"""
Validate that all suggested WordNet synsets exist in WordNet.
"""

import json
from pathlib import Path
from nltk.corpus import wordnet as wn
from collections import defaultdict

def validate_synsets(suggestions_file: str):
    """Validate all synsets in the suggestions file."""

    # Load suggestions
    with open(suggestions_file, 'r') as f:
        suggestions = json.load(f)

    print(f"Loaded {len(suggestions)} concept suggestions")
    print("=" * 80)

    # Track statistics
    stats = {
        'total_concepts': len(suggestions),
        'total_synsets': 0,
        'valid_synsets': 0,
        'invalid_synsets': 0,
        'concepts_with_invalid': 0,
    }

    invalid_by_concept = defaultdict(list)

    # Validate each concept
    for concept in suggestions:
        layer = concept['layer']
        sumo_term = concept['sumo_term']
        suggested_synsets = concept['suggested_synsets']

        has_invalid = False

        for synset_data in suggested_synsets:
            synset_name = synset_data['synset']
            definition = synset_data['definition']

            stats['total_synsets'] += 1

            # Try to look up the synset
            try:
                synset = wn.synset(synset_name)
                stats['valid_synsets'] += 1
            except Exception as e:
                stats['invalid_synsets'] += 1
                has_invalid = True
                invalid_by_concept[f"Layer {layer}: {sumo_term}"].append({
                    'synset': synset_name,
                    'definition': definition,
                    'error': str(e)
                })

        if has_invalid:
            stats['concepts_with_invalid'] += 1

    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Total concepts checked: {stats['total_concepts']}")
    print(f"Total synsets checked: {stats['total_synsets']}")
    print(f"Valid synsets: {stats['valid_synsets']} ({stats['valid_synsets']/stats['total_synsets']*100:.1f}%)")
    print(f"Invalid synsets: {stats['invalid_synsets']} ({stats['invalid_synsets']/stats['total_synsets']*100:.1f}%)")
    print(f"Concepts with invalid synsets: {stats['concepts_with_invalid']} ({stats['concepts_with_invalid']/stats['total_concepts']*100:.1f}%)")

    # Print invalid synsets if any
    if invalid_by_concept:
        print("\n" + "=" * 80)
        print("INVALID SYNSETS BY CONCEPT")
        print("=" * 80)

        for concept_name, invalid_synsets in sorted(invalid_by_concept.items()):
            print(f"\n{concept_name}:")
            for synset_data in invalid_synsets:
                print(f"  ✗ {synset_data['synset']}")
                print(f"    Definition: {synset_data['definition']}")
                print(f"    Error: {synset_data['error']}")
    else:
        print("\n✓ All synsets are valid!")

    return stats, invalid_by_concept


if __name__ == '__main__':
    suggestions_file = Path(__file__).parent.parent / 'results' / 'wordnet_patch_suggestions.json'

    print(f"Validating synsets in: {suggestions_file}")
    print()

    stats, invalid = validate_synsets(suggestions_file)

    # Exit with error code if there are invalid synsets
    if stats['invalid_synsets'] > 0:
        exit(1)
    else:
        exit(0)
