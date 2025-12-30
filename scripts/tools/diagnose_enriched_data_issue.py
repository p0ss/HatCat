#!/usr/bin/env python3
"""
Diagnose why enriched data isn't being used properly in training.

The issue: The enriched data format is incompatible with the data generation function.
- Enriched data: {simplex: {pole: [synsets]}}
- Expected format: {'overlaps': {pair_key: [synsets with 'applies_to_poles' field]}}
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
ENRICHED_PATH = PROJECT_ROOT / "data" / "simplex_overlap_synsets_enriched.json"

def main():
    print("=" * 80)
    print("ENRICHED DATA FORMAT DIAGNOSIS")
    print("=" * 80)

    with open(ENRICHED_PATH) as f:
        enriched_data = json.load(f)

    print("\n1. Current enriched data structure:")
    print(f"   Top-level keys: {list(enriched_data.keys())[:5]}")

    # Check a sample simplex
    simplexes = [k for k in enriched_data.keys() if k not in ['metadata', 'overlaps']]
    if simplexes:
        sample = simplexes[0]
        print(f"\n2. Sample simplex '{sample}':")
        print(f"   Poles: {list(enriched_data[sample].keys())}")

        for pole, synsets in enriched_data[sample].items():
            print(f"   - {pole}: {len(synsets)} synsets")
            if synsets:
                print(f"      Example: {synsets[0].get('synset_id', 'no id')}")

    # Check if overlaps structure exists
    print("\n3. Checking for 'overlaps' structure:")
    if 'overlaps' in enriched_data:
        print(f"   ✓ 'overlaps' key exists")
        if isinstance(enriched_data['overlaps'], dict):
            overlap_keys = list(enriched_data['overlaps'].keys())
            print(f"   - Contains {len(overlap_keys)} overlap pairs")
            if overlap_keys:
                sample_pair = overlap_keys[0]
                sample_synsets = enriched_data['overlaps'][sample_pair]
                print(f"   - Sample pair '{sample_pair}': {len(sample_synsets)} synsets")
                if sample_synsets:
                    sample_synset = sample_synsets[0]
                    print(f"   - Sample synset keys: {list(sample_synset.keys())}")
                    if 'applies_to_poles' in sample_synset:
                        print(f"   - applies_to_poles: {sample_synset['applies_to_poles']}")
                    else:
                        print(f"   ✗ Missing 'applies_to_poles' field!")
    else:
        print(f"   ✗ 'overlaps' key MISSING!")
        print(f"   This is the problem - data generation expects this structure")

    # Show what data generation function expects
    print("\n4. Expected structure for data generation:")
    print("""
    {
        'overlaps': {
            'simplex1_pole1__simplex1_pole2': [
                {
                    'synset_id': 'synset.overlap.001',
                    'lemmas': [...],
                    'definition': '...',
                    'applies_to_poles': [
                        'simplex1_pole1',
                        'simplex1_pole2'
                    ],
                    ...
                },
                ...
            ],
            ...
        },
        'metadata': {...}
    }
    """)

    print("\n5. What we actually have:")
    print("""
    {
        'simplex_name': {
            'negative': [synsets],
            'neutral': [synsets],
            'positive': [synsets]
        },
        ...
    }
    """)

    print("\n" + "=" * 80)
    print("ROOT CAUSE IDENTIFIED")
    print("=" * 80)
    print("""
The enriched data was merged in the wrong format!

The data generation function `get_overlap_synsets_for_pole()` expects:
- enriched_data['overlaps'][pair_key] = [synsets]
- Each synset must have 'applies_to_poles' = ['pole_id1', 'pole_id2', ...]

But the merge script created:
- enriched_data[simplex_name][pole_type] = [synsets]
- Synsets DO NOT have 'applies_to_poles' field

SOLUTION:
We need to restructure the enriched data to match the expected format by:
1. Creating the 'overlaps' structure
2. Adding 'applies_to_poles' to each synset
3. Grouping synsets by which pole pairs they apply to
""")

    # Count synsets that would be affected
    total_synsets = 0
    for simplex, poles in enriched_data.items():
        if simplex in ['metadata', 'overlaps']:
            continue
        for pole, synsets in poles.items():
            total_synsets += len(synsets)

    print(f"\nTotal synsets that need restructuring: {total_synsets}")
    print("\nThis explains the training failures:")
    print("- The code extracts large datasets (from old data)")
    print("- But uses only 15+15 samples (minimal hardcoded fallback)")
    print("- The enriched synsets are NEVER accessed")
    print("=" * 80)

if __name__ == '__main__':
    main()
