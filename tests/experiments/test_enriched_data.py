#!/usr/bin/env python3
"""
Test enriched data loading and verify cultural diversity.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
ENRICHED_DATA_PATH = PROJECT_ROOT / "data" / "simplex_overlap_synsets_enriched.json"

def extract_culture_from_synset_id(synset_id: str) -> str:
    """Extract culture identifier from synset_id."""
    parts = synset_id.split('_')
    if len(parts) > 1:
        # Format: chinese_mianzi.negative.001
        potential_culture = parts[0].lower()
        # Check if it looks like a culture identifier
        if potential_culture.isalpha() and len(potential_culture) > 2:
            return potential_culture
    return "unknown"

def test_enriched_data_loading():
    """Test 1: Load enriched data and verify structure."""
    print("=" * 80)
    print("TEST 1: ENRICHED DATA LOADING")
    print("=" * 80)

    if not ENRICHED_DATA_PATH.exists():
        print(f"❌ FAILED: Enriched data file not found: {ENRICHED_DATA_PATH}")
        return False

    with open(ENRICHED_DATA_PATH) as f:
        data = json.load(f)

    print(f"✓ Loaded enriched data file")
    print(f"  - Simplexes: {len(data)}")

    # Count synsets
    total_synsets = 0
    pole_counts = defaultdict(int)

    for simplex, poles in data.items():
        if simplex in ['metadata', 'overlaps']:
            continue
        for pole, synsets in poles.items():
            count = len(synsets)
            total_synsets += count
            pole_counts[pole] += count

    print(f"  - Total synsets: {total_synsets}")
    print(f"  - Negative: {pole_counts['negative']}")
    print(f"  - Neutral: {pole_counts['neutral']}")
    print(f"  - Positive: {pole_counts['positive']}")

    if total_synsets > 2000:
        print(f"✓ PASSED: Found {total_synsets} synsets (expected >2000)")
        return True
    else:
        print(f"❌ FAILED: Only {total_synsets} synsets (expected >2000)")
        return False

def test_cultural_diversity():
    """Test 2: Verify cultural diversity in synsets."""
    print("\n" + "=" * 80)
    print("TEST 2: CULTURAL DIVERSITY")
    print("=" * 80)

    with open(ENRICHED_DATA_PATH) as f:
        data = json.load(f)

    # Count cultures per pole
    cultural_counts = defaultdict(lambda: {'negative': 0, 'neutral': 0, 'positive': 0, 'total': 0})

    for simplex, poles in data.items():
        if simplex in ['metadata', 'overlaps']:
            continue
        for pole, synsets in poles.items():
            for synset in synsets:
                synset_id = synset.get('synset_id', '')
                culture = extract_culture_from_synset_id(synset_id)
                cultural_counts[culture][pole] += 1
                cultural_counts[culture]['total'] += 1

    # Filter to cultures with significant representation (>10 synsets)
    significant_cultures = {c: counts for c, counts in cultural_counts.items()
                           if counts['total'] > 10 and c != 'unknown'}

    print(f"✓ Found {len(significant_cultures)} cultures with >10 synsets")
    print(f"\nTop 10 cultures by representation:")
    sorted_cultures = sorted(significant_cultures.items(), key=lambda x: x[1]['total'], reverse=True)

    for i, (culture, counts) in enumerate(sorted_cultures[:10], 1):
        ratio = max(counts['negative'], counts['neutral'], counts['positive']) / min(counts['negative'], counts['neutral'], counts['positive']) if min(counts['negative'], counts['neutral'], counts['positive']) > 0 else float('inf')
        balance = "✓" if ratio < 3.0 else "⚠"
        print(f"  {i:2d}. {culture:15} - {counts['total']:3d} total (neg:{counts['negative']:2d}, neu:{counts['neutral']:2d}, pos:{counts['positive']:2d}) {balance}")

    if len(significant_cultures) >= 10:
        print(f"\n✓ PASSED: Found {len(significant_cultures)} cultures (expected ≥10)")
        return True
    else:
        print(f"\n❌ FAILED: Only {len(significant_cultures)} cultures (expected ≥10)")
        return False

def test_synset_quality():
    """Test 3: Check synset quality and required fields."""
    print("\n" + "=" * 80)
    print("TEST 3: SYNSET QUALITY")
    print("=" * 80)

    with open(ENRICHED_DATA_PATH) as f:
        data = json.load(f)

    total_synsets = 0
    valid_synsets = 0
    missing_fields = defaultdict(int)

    required_fields = ['synset_id', 'label', 'definition']

    for simplex, poles in data.items():
        if simplex in ['metadata', 'overlaps']:
            continue
        for pole, synsets in poles.items():
            for synset in synsets:
                total_synsets += 1

                # Check required fields
                all_present = True
                for field in required_fields:
                    if field not in synset or not synset[field]:
                        missing_fields[field] += 1
                        all_present = False

                if all_present:
                    valid_synsets += 1

    print(f"  - Total synsets: {total_synsets}")
    print(f"  - Valid synsets: {valid_synsets}")
    print(f"  - Validity rate: {valid_synsets/total_synsets*100:.1f}%")

    if missing_fields:
        print(f"\n  Missing fields:")
        for field, count in missing_fields.items():
            print(f"    - {field}: {count} synsets")

    if valid_synsets / total_synsets > 0.95:
        print(f"\n✓ PASSED: {valid_synsets/total_synsets*100:.1f}% validity (expected >95%)")
        return True
    else:
        print(f"\n❌ FAILED: {valid_synsets/total_synsets*100:.1f}% validity (expected >95%)")
        return False

def show_example_synsets():
    """Show example multicultural synsets."""
    print("\n" + "=" * 80)
    print("EXAMPLE MULTICULTURAL SYNSETS")
    print("=" * 80)

    with open(ENRICHED_DATA_PATH) as f:
        data = json.load(f)

    # Find culturally-diverse examples
    examples = []

    for simplex, poles in data.items():
        if simplex in ['metadata', 'overlaps']:
            continue
        for pole, synsets in poles.items():
            for synset in synsets:
                synset_id = synset.get('synset_id', '')
                culture = extract_culture_from_synset_id(synset_id)

                if culture not in ['unknown'] and '_' in synset_id:
                    examples.append({
                        'culture': culture,
                        'simplex': simplex,
                        'pole': pole,
                        'synset_id': synset_id,
                        'label': synset.get('label', ''),
                        'definition': synset.get('definition', '')[:100] + '...'
                    })

                    if len(examples) >= 5:
                        break
            if len(examples) >= 5:
                break
        if len(examples) >= 5:
            break

    for i, ex in enumerate(examples, 1):
        print(f"\n{i}. {ex['culture'].upper()} - {ex['simplex']} ({ex['pole']})")
        print(f"   ID: {ex['synset_id']}")
        print(f"   Label: {ex['label']}")
        print(f"   Def: {ex['definition']}")

def main():
    print("ENRICHED DATA VALIDATION TESTS")
    print()

    results = []

    # Run tests
    results.append(("Data Loading", test_enriched_data_loading()))
    results.append(("Cultural Diversity", test_cultural_diversity()))
    results.append(("Synset Quality", test_synset_quality()))

    # Show examples
    show_example_synsets()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\n  Total: {passed_count}/{total_count} tests passed")

    if all(p for _, p in results):
        print("\n✓ ALL TESTS PASSED - Enriched data is ready for training!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Review issues before training")
        return 1

if __name__ == '__main__':
    sys.exit(main())
