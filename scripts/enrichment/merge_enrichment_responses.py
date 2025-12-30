#!/usr/bin/env python3
"""
Merge multicultural enrichment responses into simplex training data.

This script:
1. Loads existing simplex overlap synsets (if any)
2. Loads all enrichment response files
3. Validates and merges synsets into training data
4. Saves enriched training data for tripole lens training
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
RESPONSES_DIR = PROJECT_ROOT / "data" / "enrichment_responses"
EXISTING_DATA_PATH = PROJECT_ROOT / "data" / "simplex_overlap_synsets.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "simplex_overlap_synsets_enriched.json"


def load_existing_data() -> Dict:
    """Load existing simplex overlap synsets if available."""
    if EXISTING_DATA_PATH.exists():
        print(f"Loading existing data from {EXISTING_DATA_PATH}...")
        with open(EXISTING_DATA_PATH) as f:
            return json.load(f)
    else:
        print("No existing data found, starting fresh...")
        return {}


def load_enrichment_responses() -> List[Dict]:
    """Load all enrichment response files."""
    if not RESPONSES_DIR.exists():
        print(f"ERROR: Responses directory not found: {RESPONSES_DIR}")
        sys.exit(1)

    response_files = sorted(RESPONSES_DIR.glob("*.json"))

    if not response_files:
        print(f"ERROR: No response files found in {RESPONSES_DIR}")
        sys.exit(1)

    print(f"\nLoading {len(response_files)} response files...")

    responses = []
    for file_path in response_files:
        try:
            with open(file_path) as f:
                data = json.load(f)

                # Handle both formats: dict with metadata or raw list
                if isinstance(data, list):
                    # Extract metadata from filename
                    # Format: simplex_pole_type_batchN.json
                    parts = file_path.stem.split('_')
                    # Find where pole starts (after simplex name)
                    # Simplex can contain underscores, so we need to find pole marker
                    if 'negative' in parts:
                        pole_idx = parts.index('negative')
                    elif 'neutral' in parts:
                        pole_idx = parts.index('neutral')
                    elif 'positive' in parts:
                        pole_idx = parts.index('positive')
                    else:
                        print(f"  WARNING: Cannot parse pole from {file_path.name}")
                        continue

                    simplex = '_'.join(parts[:pole_idx])
                    pole = parts[pole_idx]
                    req_type = parts[pole_idx + 1] if pole_idx + 1 < len(parts) else 'unknown'

                    # Wrap in proper format
                    data = {
                        'metadata': {
                            'simplex': simplex,
                            'pole': pole,
                            'type': req_type,
                            'source_file': file_path.name
                        },
                        'synsets': data
                    }

                responses.append(data)
        except Exception as e:
            print(f"  WARNING: Failed to load {file_path.name}: {e}")
            continue

    print(f"✓ Loaded {len(responses)} response files")
    return responses


def validate_synset(synset: Dict, simplex: str, pole: str) -> bool:
    """Validate synset has required fields and add label if missing."""
    required_fields = ['synset_id', 'definition']

    for field in required_fields:
        if field not in synset:
            print(f"  WARNING: Synset missing '{field}' field: {synset}")
            return False

    # Check synset_id format
    if '.' not in synset['synset_id']:
        print(f"  WARNING: Invalid synset_id format: {synset['synset_id']}")
        return False

    # Add label if missing (use first lemma or synset_id)
    if 'label' not in synset:
        if 'lemmas' in synset and len(synset['lemmas']) > 0:
            synset['label'] = synset['lemmas'][0]
        else:
            # Extract from synset_id (e.g., "chinese_mianzi.negative.001" -> "chinese_mianzi")
            synset['label'] = synset['synset_id'].split('.')[0]

    return True


def merge_responses_into_data(existing_data: Dict, responses: List[Dict]) -> Dict:
    """Merge enrichment responses into existing data structure."""

    merged_data = existing_data.copy()

    stats = {
        'total_synsets': 0,
        'by_simplex': defaultdict(lambda: {'negative': 0, 'neutral': 0, 'positive': 0}),
        'by_type': defaultdict(int),
        'invalid': 0
    }

    print("\nMerging responses...")

    for response in responses:
        metadata = response.get('metadata', {})
        simplex = metadata.get('simplex')
        pole = metadata.get('pole')
        req_type = metadata.get('type', 'unknown')

        if not simplex or not pole:
            print(f"  WARNING: Response missing simplex or pole metadata")
            continue

        # Initialize simplex if not exists
        if simplex not in merged_data:
            merged_data[simplex] = {
                'negative': [],
                'neutral': [],
                'positive': []
            }

        # Validate pole exists
        if pole not in merged_data[simplex]:
            print(f"  WARNING: Invalid pole '{pole}' for simplex '{simplex}'")
            continue

        # Get synsets from response
        synsets = response.get('synsets', [])

        # Validate and add synsets
        valid_synsets = []
        for synset in synsets:
            if validate_synset(synset, simplex, pole):
                valid_synsets.append(synset)
                stats['total_synsets'] += 1
                stats['by_simplex'][simplex][pole] += 1
                stats['by_type'][req_type] += 1
            else:
                stats['invalid'] += 1

        # Add to merged data
        merged_data[simplex][pole].extend(valid_synsets)

    return merged_data, stats


def analyze_enriched_data(data: Dict) -> Dict:
    """Analyze the enriched dataset."""

    analysis = {
        'total_simplexes': len(data),
        'simplexes': {}
    }

    for simplex, poles in data.items():
        simplex_stats = {
            'negative': len(poles.get('negative', [])),
            'neutral': len(poles.get('neutral', [])),
            'positive': len(poles.get('positive', [])),
        }
        simplex_stats['total'] = sum(simplex_stats.values())

        # Calculate balance
        counts = [simplex_stats['negative'], simplex_stats['neutral'], simplex_stats['positive']]
        max_count = max(counts)
        min_count = min(counts)
        simplex_stats['imbalance_ratio'] = max_count / min_count if min_count > 0 else float('inf')

        analysis['simplexes'][simplex] = simplex_stats

    return analysis


def print_statistics(stats: Dict, analysis: Dict):
    """Print merge statistics and analysis."""

    print("\n" + "=" * 80)
    print("MERGE STATISTICS")
    print("=" * 80)

    print(f"\nNew synsets added: {stats['total_synsets']}")
    print(f"Invalid synsets: {stats['invalid']}")

    print("\nBy type:")
    for req_type, count in sorted(stats['by_type'].items()):
        print(f"  {req_type:15}: {count:4d} synsets")

    print("\n" + "=" * 80)
    print("ENRICHED DATASET ANALYSIS")
    print("=" * 80)

    print(f"\nTotal simplexes: {analysis['total_simplexes']}")

    print("\nPer-simplex breakdown:")
    print(f"{'Simplex':<30} {'Neg':>6} {'Neu':>6} {'Pos':>6} {'Total':>7} {'Ratio':>6}")
    print("-" * 80)

    for simplex, stats_dict in sorted(analysis['simplexes'].items()):
        ratio = stats_dict['imbalance_ratio']
        ratio_str = f"{ratio:.2f}x" if ratio != float('inf') else "∞"

        print(f"{simplex:<30} {stats_dict['negative']:6d} {stats_dict['neutral']:6d} "
              f"{stats_dict['positive']:6d} {stats_dict['total']:7d} {ratio_str:>6}")

    # Calculate overall statistics
    total_neg = sum(s['negative'] for s in analysis['simplexes'].values())
    total_neu = sum(s['neutral'] for s in analysis['simplexes'].values())
    total_pos = sum(s['positive'] for s in analysis['simplexes'].values())
    total_all = total_neg + total_neu + total_pos

    print("-" * 80)
    print(f"{'TOTAL':<30} {total_neg:6d} {total_neu:6d} {total_pos:6d} {total_all:7d}")

    # Balance assessment
    print("\n" + "=" * 80)
    print("BALANCE ASSESSMENT")
    print("=" * 80)

    imbalanced = [s for s, st in analysis['simplexes'].items() if st['imbalance_ratio'] > 2.0]

    if imbalanced:
        print(f"\n⚠ {len(imbalanced)} simplexes with >2x imbalance:")
        for simplex in imbalanced:
            stats_dict = analysis['simplexes'][simplex]
            print(f"  - {simplex}: {stats_dict['imbalance_ratio']:.2f}x imbalance")
    else:
        print("\n✓ All simplexes have <2x imbalance (well balanced)")

    # Check minimum counts
    underpopulated = [
        (s, pole, count)
        for s, st in analysis['simplexes'].items()
        for pole, count in [('negative', st['negative']), ('neutral', st['neutral']), ('positive', st['positive'])]
        if count < 30
    ]

    if underpopulated:
        print(f"\n⚠ {len(underpopulated)} poles with <30 synsets:")
        for simplex, pole, count in underpopulated[:10]:  # Show first 10
            print(f"  - {simplex}.{pole}: {count} synsets")
        if len(underpopulated) > 10:
            print(f"  ... and {len(underpopulated) - 10} more")
    else:
        print("\n✓ All poles have ≥30 synsets")


def main():
    print("=" * 80)
    print("MULTICULTURAL ENRICHMENT MERGE")
    print("=" * 80)

    # Load existing data
    existing_data = load_existing_data()

    # Load enrichment responses
    responses = load_enrichment_responses()

    # Merge responses
    merged_data, stats = merge_responses_into_data(existing_data, responses)

    # Analyze enriched data
    analysis = analyze_enriched_data(merged_data)

    # Print statistics
    print_statistics(stats, analysis)

    # Save enriched data
    print("\n" + "=" * 80)
    print(f"Saving enriched data to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"✓ Saved enriched training data")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Review balance assessment above
   - If >2x imbalance detected, consider generating more synsets for underrepresented poles
   - If <30 synsets per pole, generate additional synsets

2. Validate cultural distribution:
   poetry run python scripts/validate_cultural_distribution.py

3. Re-train tripole lenses with enriched data:
   poetry run python scripts/train_s_tier_simplexes.py --device cuda

Expected improvements:
- Neutral F1: 0.27 → 0.70+ (2.6x improvement)
- Reduced variance across runs
- Better generalization to non-Western concepts
""")

    print("=" * 80)


if __name__ == '__main__':
    main()
