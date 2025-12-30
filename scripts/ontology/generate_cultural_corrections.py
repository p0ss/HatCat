#!/usr/bin/env python3
"""
Generate targeted API requests to correct cultural imbalances.

This script loads the cultural distribution analysis and generates specific
API requests to balance cultures across poles.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
ANALYSIS_PATH = PROJECT_ROOT / "data" / "cultural_distribution_analysis.json"
S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cultural_correction_requests.json"


def load_analysis():
    """Load cultural distribution analysis."""
    if not ANALYSIS_PATH.exists():
        print(f"ERROR: Analysis file not found: {ANALYSIS_PATH}")
        print("Run scripts/validate_cultural_distribution.py first")
        sys.exit(1)

    with open(ANALYSIS_PATH) as f:
        return json.load(f)


def generate_targeted_prompts(correction_requests, s_tier_defs):
    """Generate specific API prompts for cultural corrections."""

    prompts = []

    for req in correction_requests:
        culture = req['culture']
        pole = req['pole']
        needed = req['needed']

        # For now, generate generic correction requests
        # In practice, you'd need simplex context
        prompt_data = {
            'culture': culture,
            'pole': pole,
            'count': needed,
            'priority': req['priority'],
            'prompt': f"""Generate {needed} WordNet-style synsets for the "{pole}" pole using ONLY {culture.upper()} cultural concepts.

CRITICAL REQUIREMENTS:
1. ALL synsets must be {culture} cultural concepts
2. Target emotional valence: {pole}
3. Include original language terms with romanization
4. Focus on culture-specific concepts that don't exist in Western psychology

Synset format:
{{
  "synset_id": "{culture}_[term].{pole}.###",
  "lemmas": ["original_term", "romanization", "english_approximation"],
  "definition": "Cultural meaning...",
  "pos": "n",
  "cultural_note": "Context..."
}}

Return as JSON array.
"""
        }

        prompts.append(prompt_data)

    return prompts


def main():
    print("=" * 80)
    print("CULTURAL CORRECTION REQUEST GENERATOR")
    print("=" * 80)

    # Load analysis
    print(f"\n1. Loading analysis from {ANALYSIS_PATH}...")
    analysis_data = load_analysis()
    correction_requests = analysis_data.get('correction_requests', [])

    if not correction_requests:
        print("\n✓ No corrections needed! Cultural distribution is balanced.")
        sys.exit(0)

    print(f"✓ Found {len(correction_requests)} corrections needed")

    # Load simplex definitions
    with open(S_TIER_DEFS_PATH) as f:
        s_tier_defs = json.load(f)

    # Generate prompts
    print("\n2. Generating targeted correction prompts...")
    prompts = generate_targeted_prompts(correction_requests, s_tier_defs)

    # Save output
    output_data = {
        'metadata': {
            'generated_date': datetime.now().isoformat(),
            'total_corrections': len(prompts),
            'source_analysis': str(ANALYSIS_PATH)
        },
        'correction_requests': prompts
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Generated {len(prompts)} correction prompts")
    print(f"\n{'=' * 80}")
    print(f"✓ Correction requests saved to: {OUTPUT_PATH}")
    print(f"\nNext step: Execute correction API requests manually or via automation")
    print("=" * 80)


if __name__ == '__main__':
    main()
