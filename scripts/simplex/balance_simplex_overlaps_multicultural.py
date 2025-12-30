#!/usr/bin/env python3
"""
Balance simplex overlap synsets with FULL MULTICULTURAL ENRICHMENT.

Unlike balance_simplex_overlaps.py which only enriches underrepresented poles,
this script enriches ALL poles to add cultural diversity, then balances on top.

Strategy:
1. Add 20-30 multicultural synsets to EVERY pole (regardless of current count)
2. Then balance to ensure equal totals across all three poles
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
OUTPUT_PATH = PROJECT_ROOT / "data" / "balance_enrichment_multicultural.json"

# Configuration
MULTICULTURAL_BASE = 25  # Add 25 culturally-diverse synsets to EVERY pole


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
        poles = pair_key.split('+')
        if len(poles) != 2:
            continue

        for pole_str in poles:
            parts = pole_str.rsplit('_', 1)
            if len(parts) != 2:
                continue
            simplex, pole_type = parts

            if pole_type == 'negative':
                counts[simplex]['negative'] += len(synsets)
            elif pole_type in ['neutral', 'homeostasis']:
                counts[simplex]['neutral'] += len(synsets)
            elif pole_type == 'positive':
                counts[simplex]['positive'] += len(synsets)

    return counts, simplexes


def generate_multicultural_enrichment_plan(counts, simplexes, s_tier_defs):
    """
    Generate plan that:
    1. Adds MULTICULTURAL_BASE synsets to EVERY pole
    2. Then balances to match maximum
    """
    plan = {}

    for simplex in simplexes:
        pole_counts = counts[simplex]
        simplex_def = s_tier_defs['simplexes'][simplex]

        # Calculate target: max current + multicultural base
        max_current = max(pole_counts.values())
        target = max_current + MULTICULTURAL_BASE

        simplex_plan = {}

        for pole in ['negative', 'neutral', 'positive']:
            current = pole_counts[pole]

            # Every pole gets:
            # 1. MULTICULTURAL_BASE for cultural diversity
            # 2. Additional to reach target (balance)
            multicultural_needed = MULTICULTURAL_BASE
            balance_needed = max(0, target - current - MULTICULTURAL_BASE)
            total_needed = multicultural_needed + balance_needed

            if total_needed > 0:
                # Get pole definition
                if pole == 'negative':
                    pole_def = simplex_def['negative_pole']
                elif pole == 'neutral':
                    pole_def = simplex_def['neutral_homeostasis']
                else:
                    pole_def = simplex_def['positive_pole']

                simplex_plan[pole] = {
                    'current_count': current,
                    'multicultural_synsets': multicultural_needed,
                    'balance_synsets': balance_needed,
                    'total_needed': total_needed,
                    'target_count': target,
                    'pole_def': pole_def,
                    'requests': []
                }

        if simplex_plan:
            plan[simplex] = simplex_plan

    return plan


def create_api_requests(plan, simplexes_def):
    """Create API request prompts with strong cultural diversity emphasis."""

    for simplex, pole_plans in plan.items():
        simplex_def = simplexes_def['simplexes'][simplex]

        for pole, pole_plan in pole_plans.items():
            total_needed = pole_plan['total_needed']
            multicultural = pole_plan['multicultural_synsets']
            balance = pole_plan['balance_synsets']
            pole_def = pole_plan['pole_def']

            # Create 2 types of requests:
            # 1. Pure multicultural (emphasis on cultural diversity)
            # 2. Balance (can include English + multicultural mix)

            # Request 1: Pure multicultural synsets
            if multicultural > 0:
                batch_size = 50
                num_batches = (multicultural + batch_size - 1) // batch_size

                for batch_idx in range(num_batches):
                    batch_count = min(batch_size, multicultural - batch_idx * batch_size)

                    request = {
                        'simplex': simplex,
                        'pole': pole,
                        'type': 'multicultural',
                        'pole_label': pole_def['label'],
                        'pole_definition': pole_def['definition'],
                        'simplex_description': simplex_def['description'],
                        'count': batch_count,
                        'batch': f"{batch_idx + 1}/{num_batches}",
                        'prompt': f"""Generate {batch_count} culturally-diverse WordNet-style synsets for the "{pole_def['label']}" pole of the {simplex} simplex.

Simplex: {simplex}
Description: {simplex_def['description']}

Pole: {pole_def['label']}
Definition: {pole_def['definition']}

CRITICAL: This is a MULTICULTURAL ENRICHMENT request. Focus EXCLUSIVELY on non-Western concepts:

TARGET CULTURES - DISTRIBUTE EVENLY (avoid clustering cultures in specific poles):
- East Asian: Chinese (面子/mianzi - face, 孝/xiào - filial piety), Japanese (和/wa - harmony, 恥/haji - shame), Korean (정/jeong - affection, 한/han - collective grief)
- South Asian: Sanskrit/Hindi (dharma, karma, ahimsa, maya), Tamil (aṟam - virtue)
- Southeast Asian: Thai (kreng jai - considerate heart), Indonesian (gotong royong - communal cooperation)
- Middle Eastern: Arabic (ḥikmah - wisdom, sabr - patience), Persian (taarof - ritual courtesy), Hebrew (tikkun olam - repair the world)
- African: Ubuntu (humanity toward others), various indigenous concepts
- Indigenous: Native American (mitakuye oyasin - all my relations), Aboriginal (dadirri - deep listening)
- Latin American: Brazilian (saudade - nostalgic longing), Mexican (respeto - respect with deeper cultural context)

CRITICAL REQUIREMENT - SYMMETRIC CULTURAL DISTRIBUTION:
Each culture has concepts spanning ALL emotional valences. Avoid clustering:
- DON'T: Make all Japanese concepts neutral/harmonious
- DO: Find Japanese concepts for this specific pole ("{pole_def['label']}")
- Remember: Every culture experiences the full range of {simplex} across negative/neutral/positive

For each synset:
1. synset_id: [culture]_[romanized_term].{pole}.### (e.g., "chinese_mianzi.positive.001")
2. lemmas: Include original term + romanization + English approximation
3. definition: Explain the culturally-specific meaning that doesn't exist in Western psychology
4. pos: Usually 'n' for cultural concepts
5. cultural_note: Brief note on cultural context and why it's unique

AVOID:
- Generic Western concepts, simple translations of English words
- Cultural stereotypes (not all Japanese concepts are about harmony)
- Clustering all terms from one culture

FOCUS: Culture-bound concepts that represent unique perspectives on "{pole_def['label']}" in {simplex}, ensuring EVERY culture appears across ALL poles.

Return as JSON array:
[
  {{
    "synset_id": "culture_term.{pole}.001",
    "lemmas": ["original_term", "romanization", "english_approximation"],
    "definition": "Cultural meaning...",
    "pos": "n",
    "cultural_note": "Context..."
  }},
  ...
]"""
                    }

                    pole_plan['requests'].append(request)

            # Request 2: Balance synsets (can be more diverse including Western)
            if balance > 0:
                batch_size = 50
                num_batches = (balance + batch_size - 1) // batch_size

                for batch_idx in range(num_batches):
                    batch_count = min(batch_size, balance - batch_idx * batch_size)

                    request = {
                        'simplex': simplex,
                        'pole': pole,
                        'type': 'balance',
                        'pole_label': pole_def['label'],
                        'pole_definition': pole_def['definition'],
                        'simplex_description': simplex_def['description'],
                        'count': batch_count,
                        'batch': f"{batch_idx + 1}/{num_batches}",
                        'prompt': f"""Generate {batch_count} diverse WordNet-style synsets for the "{pole_def['label']}" pole of the {simplex} simplex.

Simplex: {simplex}
Description: {simplex_def['description']}

Pole: {pole_def['label']}
Definition: {pole_def['definition']}

IMPORTANT: Generate culturally-diverse terms including both Western and non-Western concepts:

Cultural diversity requirements:
- Western: English, Romance, Germanic concepts (but look for specific/interesting ones)
- East Asian: Chinese, Japanese, Korean
- South Asian: Hindi, Sanskrit
- Middle Eastern: Arabic, Persian, Hebrew
- African: Various languages/cultures
- Indigenous: Native American, Aboriginal, etc.
- Latin American: Spanish, Portuguese with local cultural nuances

For each synset:
1. synset_id: word.{pole}.###
2. lemmas: 3-5 synonyms
3. definition: Clear explanation
4. pos: 'n', 'v', 'a', etc.

Focus on concepts that:
1. Represent diverse cultural perspectives on "{pole_def['label']}"
2. Capture specific manifestations of {simplex}
3. Prevent models from escaping detection via culturally-specific encoding

Return as JSON array:
[
  {{
    "synset_id": "example.{pole}.001",
    "lemmas": ["term1", "term2", "term3"],
    "definition": "Definition...",
    "pos": "n"
  }},
  ...
]"""
                    }

                    pole_plan['requests'].append(request)

    return plan


def main():
    print("=" * 80)
    print("MULTICULTURAL SIMPLEX OVERLAP ENRICHMENT")
    print("=" * 80)

    # 1. Analyze current state
    print("\\n1. Analyzing current overlap counts...")
    counts, simplexes = analyze_overlap_counts()

    print(f"\\nFound {len(simplexes)} simplexes:\\n")
    for simplex in sorted(simplexes):
        pole_counts = counts[simplex]
        neg, neu, pos = pole_counts['negative'], pole_counts['neutral'], pole_counts['positive']
        max_count = max(neg, neu, pos)
        min_count = min(neg, neu, pos)
        imbalance = max_count / min_count if min_count > 0 else float('inf')

        print(f"{simplex:30} | Neg: {neg:4} | Neu: {neu:4} | Pos: {pos:4} | Imbalance: {imbalance:.1f}x")

    # 2. Generate multicultural enrichment plan
    print(f"\\n2. Generating MULTICULTURAL enrichment plan...")
    print(f"   Strategy: Add {MULTICULTURAL_BASE} multicultural synsets to EVERY pole")
    print(f"            + balance to match maximum\\n")

    with open(S_TIER_DEFS_PATH) as f:
        s_tier_defs = json.load(f)

    plan = generate_multicultural_enrichment_plan(counts, simplexes, s_tier_defs)

    # Add API requests
    plan = create_api_requests(plan, s_tier_defs)

    # 3. Summary
    print(f"\\n3. Enrichment summary:\\n")
    total_needed = 0
    total_multicultural = 0
    total_balance = 0
    total_requests = 0

    for simplex in sorted(plan.keys()):
        pole_counts = counts[simplex]
        max_current = max(pole_counts.values())
        target = max_current + MULTICULTURAL_BASE

        print(f"\\n{simplex}:")
        print(f"  Target: {target} overlaps per pole (current max: {max_current} + {MULTICULTURAL_BASE} multicultural)")

        for pole, pole_plan in plan[simplex].items():
            multi = pole_plan['multicultural_synsets']
            bal = pole_plan['balance_synsets']
            total = pole_plan['total_needed']
            num_requests = len(pole_plan['requests'])

            total_needed += total
            total_multicultural += multi
            total_balance += bal
            total_requests += num_requests

            print(f"  {pole:8}: +{multi:2d} multicultural + {bal:2d} balance = {total:3d} total ({num_requests} requests)")

    print(f"\\n{'=' * 80}")
    print(f"TOTAL NEW SYNSETS: {total_needed}")
    print(f"  Multicultural: {total_multicultural} ({total_multicultural/total_needed*100:.0f}%)")
    print(f"  Balance: {total_balance} ({total_balance/total_needed*100:.0f}%)")
    print(f"  API Requests: {total_requests}")
    print(f"{'=' * 80}")

    # 4. Save plan
    output = {
        'metadata': {
            'generated_date': datetime.now().isoformat(),
            'multicultural_base': MULTICULTURAL_BASE,
            'total_new_overlaps_needed': total_needed,
            'multicultural_overlaps': total_multicultural,
            'balance_overlaps': total_balance,
            'total_api_requests': total_requests,
            'strategy': 'multicultural_full',
            'source_file': str(OVERLAP_PATH),
            'definitions_file': str(S_TIER_DEFS_PATH)
        },
        'current_counts': {simplex: dict(counts[simplex]) for simplex in simplexes},
        'enrichment_plan': plan
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\\n✓ Multicultural enrichment plan saved to: {OUTPUT_PATH}")
    print(f"\\nNext steps:")
    print(f"1. Review the plan in {OUTPUT_PATH}")
    print(f"2. Execute {total_requests} API requests ({total_multicultural} multicultural + {total_balance} balance)")
    print(f"3. Merge responses into {OVERLAP_PATH}")


if __name__ == '__main__':
    main()
