#!/usr/bin/env python3
"""
Inspect the training data generated for simplex poles.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from map.training.sumo_data_generation import create_simplex_pole_training_dataset_contrastive

S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"


def inspect_pole_data(simplex_name: str, simplex_def: dict, pole_name: str):
    """Generate and inspect training data for a pole."""
    pole_data = simplex_def[pole_name]
    pole_type = pole_name.split('_')[0]

    other_pole_names = [p for p in ['negative_pole', 'neutral_homeostasis', 'positive_pole'] if p != pole_name]
    other_poles_data = [{**simplex_def[p], 'pole_type': p.split('_')[0]} for p in other_pole_names]

    prompts, labels = create_simplex_pole_training_dataset_contrastive(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=simplex_name,
        other_poles_data=other_poles_data,
        behavioral_ratio=0.6,
        prompts_per_synset=5
    )

    return prompts, labels


def main():
    with open(S_TIER_DEFS_PATH) as f:
        simplexes = json.load(f)['simplexes']

    # Look at threat_perception as example
    simplex_name = 'threat_perception'
    simplex_def = simplexes[simplex_name]

    print("=" * 80)
    print(f"TRAINING DATA INSPECTION: {simplex_name}")
    print("=" * 80)

    print(f"\nSimplex definition:")
    print(f"  Negative pole: {simplex_def['negative_pole'].get('name', 'unnamed')}")
    print(f"  Neutral pole: {simplex_def['neutral_homeostasis'].get('name', 'unnamed')}")
    print(f"  Positive pole: {simplex_def['positive_pole'].get('name', 'unnamed')}")

    for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
        pole_type = pole_name.split('_')[0]
        print(f"\n{'='*80}")
        print(f"POLE: {pole_type.upper()}")
        print(f"{'='*80}")

        prompts, labels = inspect_pole_data(simplex_name, simplex_def, pole_name)

        # Separate positives and negatives
        positives = [(p, l) for p, l in zip(prompts, labels) if l == 1]
        negatives = [(p, l) for p, l in zip(prompts, labels) if l == 0]

        print(f"\nTotal samples: {len(prompts)}")
        print(f"Positives (label=1): {len(positives)}")
        print(f"Negatives (label=0): {len(negatives)}")

        print(f"\n--- POSITIVE SAMPLES (should represent {pole_type} pole) ---")
        for i, (p, l) in enumerate(positives[:15]):
            print(f"  {i+1}. {p[:80]}...")

        print(f"\n--- NEGATIVE SAMPLES (should NOT represent {pole_type} pole) ---")
        for i, (p, l) in enumerate(negatives[:15]):
            print(f"  {i+1}. {p[:80]}...")

        # Check for obvious issues
        print(f"\n--- ANALYSIS ---")

        # Check if positives and negatives look different
        pos_keywords = set()
        neg_keywords = set()
        for p, _ in positives[:30]:
            pos_keywords.update(p.lower().split()[:5])
        for p, _ in negatives[:30]:
            neg_keywords.update(p.lower().split()[:5])

        overlap = pos_keywords & neg_keywords
        print(f"  Common starting words: {list(overlap)[:10]}")

        # Check prompt patterns
        pos_patterns = {}
        for p, _ in positives:
            pattern = p.split()[0] if p.split() else "empty"
            pos_patterns[pattern] = pos_patterns.get(pattern, 0) + 1

        neg_patterns = {}
        for p, _ in negatives:
            pattern = p.split()[0] if p.split() else "empty"
            neg_patterns[pattern] = neg_patterns.get(pattern, 0) + 1

        print(f"  Positive prompt starts: {dict(sorted(pos_patterns.items(), key=lambda x: -x[1])[:5])}")
        print(f"  Negative prompt starts: {dict(sorted(neg_patterns.items(), key=lambda x: -x[1])[:5])}")


if __name__ == "__main__":
    main()
