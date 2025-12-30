#!/usr/bin/env python3
"""
Restructure enriched data to be compatible with training pipeline.

The enriched synsets need to be moved into the 'overlaps' structure with
'applies_to_poles' fields so the data generation function can find them.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
ENRICHED_PATH = PROJECT_ROOT / "data" / "simplex_overlap_synsets_enriched.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "simplex_overlap_synsets_enriched_fixed.json"

def main():
    print("=" * 80)
    print("RESTRUCTURING ENRICHED DATA")
    print("=" * 80)

    with open(ENRICHED_PATH) as f:
        enriched_data = json.load(f)

    # Start with existing overlaps
    if 'overlaps' in enriched_data and isinstance(enriched_data['overlaps'], dict):
        overlaps = enriched_data['overlaps']
        print(f"\nExisting overlaps: {len(overlaps)} pairs")
    else:
        overlaps = {}
        print("\nNo existing overlaps structure")

    # Track metadata
    metadata = enriched_data.get('metadata', {})

    # Process each simplex
    simplexes = [k for k in enriched_data.keys() if k not in ['metadata', 'overlaps']]
    print(f"Processing {len(simplexes)} simplexes...")

    added_to_overlaps = 0
    synsets_processed = 0

    for simplex in simplexes:
        poles = enriched_data[simplex]

        # For tripole training, each synset for a pole is a POSITIVE example for that pole
        # We need to add it to 'overlaps' with applies_to_poles containing just that one pole

        for pole_type, synsets in poles.items():
            if pole_type not in ['negative', 'neutral', 'positive']:
                continue

            pole_id = f"{simplex}_{pole_type}"

            # Create a synthetic overlap "pair" for single-pole synsets
            # These will be used as positives for the pole
            pair_key = f"{pole_id}__enriched"

            if pair_key not in overlaps:
                overlaps[pair_key] = []

            for synset in synsets:
                synsets_processed += 1

                # Add applies_to_poles field
                synset_copy = synset.copy()
                synset_copy['applies_to_poles'] = [pole_id]
                synset_copy['is_enriched'] = True

                overlaps[pair_key].append(synset_copy)
                added_to_overlaps += 1

    print(f"\nProcessed {synsets_processed} synsets")
    print(f"Added {added_to_overlaps} synsets to overlaps structure")
    print(f"Total overlap pairs: {len(overlaps)}")

    # Create output structure
    output = {
        'metadata': metadata,
        'overlaps': overlaps
    }

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Restructured data saved")

    # Verify
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    test_simplex = simplexes[0]
    test_pole = 'negative'
    test_pole_id = f"{test_simplex}_{test_pole}"
    test_pair_key = f"{test_pole_id}__enriched"

    if test_pair_key in overlaps:
        test_synsets = overlaps[test_pair_key]
        print(f"\n✓ Test lookup: {test_pole_id}")
        print(f"  Found {len(test_synsets)} synsets")
        if test_synsets:
            sample = test_synsets[0]
            print(f"  Sample synset_id: {sample.get('synset_id')}")
            print(f"  applies_to_poles: {sample.get('applies_to_poles')}")
            print(f"  is_enriched: {sample.get('is_enriched')}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"""
1. Update data generation to load from new file:
   - Change: simplex_overlap_synsets_enriched.json
   - To: simplex_overlap_synsets_enriched_fixed.json

2. Or rename the fixed file to replace the original:
   mv data/simplex_overlap_synsets_enriched.json data/simplex_overlap_synsets_enriched_backup.json
   mv data/simplex_overlap_synsets_enriched_fixed.json data/simplex_overlap_synsets_enriched.json

3. Re-run training:
   poetry run python scripts/train_s_tier_simplexes.py --device cuda
    """)
    print("=" * 80)

if __name__ == '__main__':
    main()
