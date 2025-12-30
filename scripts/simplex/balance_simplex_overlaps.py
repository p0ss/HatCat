#!/usr/bin/env python3
"""
Balance simplex overlap synsets across all three poles.

Current imbalance (taste_development example):
- Negative: 1,172 overlaps
- Neutral: 171 overlaps (6.8x fewer!)
- Positive: 1,159 overlaps

Goal: Bring all poles to the maximum count (e.g., 1,200 for taste_development)
by generating additional synthetic overlaps and lemmas.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"
OVERLAP_PATH = PROJECT_ROOT / "data" / "simplex_overlap_synsets_enriched.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "balance_enrichment_requests.json"


def analyze_overlap_counts():
    """Analyze current overlap counts by simplex and pole."""
    with open(OVERLAP_PATH) as f:
        data = json.load(f)

    with open(S_TIER_DEFS_PATH) as f:
        s_tier = json.load(f)

    simplexes = list(s_tier['simplexes'].keys())

    # Count overlaps by simplex and pole
    counts = defaultdict(lambda: {'negative': 0, 'neutral': 0, 'positive': 0})

    for pair_key, synsets in data['overlaps'].items():
        # Parse pair key like "taste_development_positive+taste_development_neutral"
        poles = pair_key.split('+')
        if len(poles) != 2:
            continue

        # Extract simplex and pole types
        for pole_str in poles:
            parts = pole_str.rsplit('_', 1)
            if len(parts) != 2:
                continue
            simplex, pole_type = parts

            # Map pole_type to our categories
            if pole_type == 'negative':
                counts[simplex]['negative'] += len(synsets)
            elif pole_type in ['neutral', 'homeostasis']:
                counts[simplex]['neutral'] += len(synsets)
            elif pole_type == 'positive':
                counts[simplex]['positive'] += len(synsets)

    return counts, simplexes


def generate_enrichment_plan(counts, simplexes, target_strategy='max'):
    """
    Generate plan for balancing overlaps.

    Args:
        counts: Dict[simplex -> Dict[pole -> count]]
        simplexes: List of simplex names
        target_strategy: 'max' (match highest), 'median', or 'fixed' (e.g., 1200)
    """
    plan = {}

    for simplex in simplexes:
        pole_counts = counts[simplex]

        if target_strategy == 'max':
            target = max(pole_counts.values())
        elif target_strategy == 'median':
            sorted_counts = sorted(pole_counts.values())
            target = sorted_counts[1]  # Middle value for 3 poles
        elif target_strategy == 'fixed':
            target = 1200
        else:
            raise ValueError(f"Unknown strategy: {target_strategy}")

        # Calculate needed overlaps for each pole
        simplex_plan = {}
        for pole, count in pole_counts.items():
            needed = max(0, target - count)
            if needed > 0:
                simplex_plan[pole] = {
                    'current_count': count,
                    'target_count': target,
                    'needed': needed,
                    'requests': []
                }

        if simplex_plan:
            plan[simplex] = simplex_plan

    return plan


def create_api_requests(plan, simplexes_def):
    """Create API request prompts for generating new overlaps."""

    for simplex, pole_plans in plan.items():
        simplex_def = simplexes_def['simplexes'][simplex]

        for pole, pole_plan in pole_plans.items():
            needed = pole_plan['needed']

            # Get pole definition
            if pole == 'negative':
                pole_def = simplex_def['negative_pole']
            elif pole == 'neutral':
                pole_def = simplex_def['neutral_homeostasis']
            else:  # positive
                pole_def = simplex_def['positive_pole']

            # Generate request for this pole
            # We'll generate in batches of ~50 to avoid huge requests
            batch_size = 50
            num_batches = (needed + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                batch_count = min(batch_size, needed - batch_idx * batch_size)

                request = {
                    'simplex': simplex,
                    'pole': pole,
                    'pole_label': pole_def['label'],
                    'pole_definition': pole_def['definition'],
                    'simplex_description': simplex_def['description'],
                    'count': batch_count,
                    'batch': f"{batch_idx + 1}/{num_batches}",
                    'prompt': f"""Generate {batch_count} synthetic WordNet-style synsets for the "{pole_def['label']}" pole of the {simplex} simplex.

Simplex: {simplex}
Description: {simplex_def['description']}

Pole: {pole_def['label']}
Definition: {pole_def['definition']}

For each synset, provide:
1. A unique synset_id (format: word.{pole}.###)
2. 3-5 lemmas (synonyms) that capture this concept
3. A clear definition explaining the concept
4. Part of speech (typically 'n' for noun, 'v' for verb, 'a' for adjective)

IMPORTANT: Generate diverse, culturally-specific terms including:
- Western concepts (from English, Romance languages, Germanic languages)
- East Asian concepts (Chinese, Japanese, Korean - e.g., 面子/mianzi, 和/wa, 情/jeong)
- South Asian concepts (Hindi, Sanskrit - e.g., dharma, karma, ahimsa)
- Middle Eastern concepts (Arabic, Persian, Hebrew)
- African concepts (various languages/cultures)
- Indigenous concepts (Native American, Aboriginal, etc.)
- Latin American concepts (Spanish, Portuguese with local cultural nuances)

Focus on concepts that:
1. Represent unique cultural perspectives on "{pole_def['label']}" that might not exist in Western psychology
2. Capture culturally-specific manifestations of {simplex}
3. Include culture-bound concepts that provide richer, more diverse training data
4. Ensure models can't escape detection by encoding concepts in culturally-specific ways

Return as JSON array:
[
  {{
    "synset_id": "example.{pole}.001",
    "lemmas": ["term1", "term2", "term3"],
    "definition": "Clear definition...",
    "pos": "n"
  }},
  ...
]"""
                }

                pole_plan['requests'].append(request)

    return plan


def main():
    print("=" * 80)
    print("SIMPLEX OVERLAP BALANCE ANALYSIS")
    print("=" * 80)

    # 1. Analyze current state
    print("\n1. Analyzing current overlap counts...")
    counts, simplexes = analyze_overlap_counts()

    print(f"\nFound {len(simplexes)} simplexes:\n")
    for simplex in sorted(simplexes):
        pole_counts = counts[simplex]
        neg, neu, pos = pole_counts['negative'], pole_counts['neutral'], pole_counts['positive']
        max_count = max(neg, neu, pos)
        min_count = min(neg, neu, pos)
        imbalance = max_count / min_count if min_count > 0 else float('inf')

        print(f"{simplex:30} | Neg: {neg:4} | Neu: {neu:4} | Pos: {pos:4} | Imbalance: {imbalance:.1f}x")

    # 2. Generate enrichment plan
    print("\n2. Generating enrichment plan (target=max)...")
    with open(S_TIER_DEFS_PATH) as f:
        s_tier_defs = json.load(f)

    plan = generate_enrichment_plan(counts, simplexes, target_strategy='max')

    # Add API requests
    plan = create_api_requests(plan, s_tier_defs)

    # 3. Summary
    print(f"\n3. Enrichment summary:\n")
    total_needed = 0
    total_requests = 0

    for simplex in sorted(plan.keys()):
        print(f"\n{simplex}:")
        for pole, pole_plan in plan[simplex].items():
            needed = pole_plan['needed']
            num_requests = len(pole_plan['requests'])
            total_needed += needed
            total_requests += num_requests
            print(f"  {pole:8}: need {needed:4} overlaps ({num_requests} API requests)")

    print(f"\n{'=' * 80}")
    print(f"TOTAL: {total_needed} new overlaps needed across {total_requests} API requests")
    print(f"{'=' * 80}")

    # 4. Save plan
    output = {
        'metadata': {
            'generated_date': datetime.now().isoformat(),
            'total_new_overlaps_needed': total_needed,
            'total_api_requests': total_requests,
            'target_strategy': 'max',
            'source_file': str(OVERLAP_PATH),
            'definitions_file': str(S_TIER_DEFS_PATH)
        },
        'current_counts': {simplex: dict(counts[simplex]) for simplex in simplexes},
        'enrichment_plan': plan
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Enrichment plan saved to: {OUTPUT_PATH}")
    print(f"\nNext steps:")
    print(f"1. Review the plan in {OUTPUT_PATH}")
    print(f"2. Run API requests to generate new synsets")
    print(f"3. Merge new synsets into {OVERLAP_PATH}")


if __name__ == '__main__':
    main()
