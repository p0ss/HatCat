#!/usr/bin/env python3
"""
Convert simplex agentic review results into S-tier WordNet patches.

This script:
1. Reads simplex_agentic_review.json
2. Extracts S-tier simplexes (those with complete three poles)
3. Identifies missing SUMO concepts that need creation
4. Generates WordNet patch files for integration
5. Creates S+ training specification
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from nltk.corpus import wordnet as wn

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SIMPLEX_REVIEW = PROJECT_ROOT / "results" / "simplex_agentic_review.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "concept_graph" / "wordnet_patches"
OUTPUT_PATCH = OUTPUT_DIR / "s_tier_simplexes_from_review.json"

def load_simplex_review():
    """Load the simplex agentic review results"""
    with open(SIMPLEX_REVIEW) as f:
        return json.load(f)

def is_complete_simplex(simplex: Dict) -> bool:
    """Check if simplex has all three poles defined"""
    s = simplex.get('simplex', {})
    return all([
        s.get('negative_pole', {}).get('synset'),
        s.get('neutral_homeostasis', {}).get('synset'),
        s.get('positive_pole', {}).get('synset')
    ])

def needs_sumo_concept(synset_name: str) -> bool:
    """Check if synset exists in WordNet or needs custom SUMO concept"""
    if not synset_name:
        return True
    try:
        wn.synset(synset_name)
        return False
    except:
        return True

def extract_s_tier_simplexes(review_data: Dict) -> tuple:
    """
    Extract S-tier simplexes and identify missing SUMO concepts.

    Returns:
        (s_tier_simplexes, missing_sumo_concepts, coverage_gaps)
    """
    simplexes = review_data['results'].get('simplexes', [])
    coverage_gaps = review_data['results']['coverage'].get('coverage_gaps', [])

    s_tier = []
    missing_sumo = []

    for simplex in simplexes:
        if is_complete_simplex(simplex):
            s_tier.append(simplex)

            # Check for missing concepts
            for pole in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
                synset = simplex['simplex'][pole].get('synset')
                if needs_sumo_concept(synset):
                    missing_sumo.append({
                        'synset': synset,
                        'pole': pole,
                        'dimension': simplex['dimension'],
                        'definition': simplex['simplex'][pole].get('definition', ''),
                        'lemmas': simplex['simplex'][pole].get('lemmas', [])
                    })

    return s_tier, missing_sumo, coverage_gaps

def categorize_s_tier_by_priority(s_tier: List[Dict], gaps: List[Dict]) -> Dict:
    """
    Categorize S-tier simplexes by priority based on coverage gaps.

    Returns dict with keys: critical, high, medium
    """
    # Build gap priority map
    gap_map = {gap['dimension']: gap['priority'] for gap in gaps}

    categorized = {
        'CRITICAL': [],
        'HIGH': [],
        'MEDIUM': [],
        'STANDARD': []
    }

    for simplex in s_tier:
        dimension = simplex['dimension']
        priority = gap_map.get(dimension, 'STANDARD')
        categorized[priority].append(simplex)

    return categorized

def create_wordnet_patch(s_tier_categorized: Dict, missing_sumo: List[Dict]):
    """Create WordNet patch file from S-tier simplexes"""

    patch = {
        "metadata": {
            "source": "simplex_agentic_review",
            "total_s_tier_simplexes": sum(len(v) for v in s_tier_categorized.values()),
            "missing_sumo_concepts": len(missing_sumo),
            "description": "S-tier three-pole simplexes for affective/motivational homeostatic steering"
        },
        "critical_s_tier": [],
        "high_priority_s_tier": [],
        "medium_priority_s_tier": [],
        "standard_s_tier": [],
        "missing_sumo_concepts": missing_sumo
    }

    # Map priorities to patch sections
    priority_map = {
        'CRITICAL': 'critical_s_tier',
        'HIGH': 'high_priority_s_tier',
        'MEDIUM': 'medium_priority_s_tier',
        'STANDARD': 'standard_s_tier'
    }

    for priority, simplexes in s_tier_categorized.items():
        section = priority_map[priority]

        for simplex in simplexes:
            entry = {
                "dimension": simplex['dimension'],
                "three_pole_simplex": {
                    "negative_pole": {
                        "synset": simplex['simplex']['negative_pole']['synset'],
                        "lemmas": simplex['simplex']['negative_pole'].get('lemmas', []),
                        "definition": simplex['simplex']['negative_pole'].get('definition', '')
                    },
                    "neutral_homeostasis": {
                        "synset": simplex['simplex']['neutral_homeostasis']['synset'],
                        "lemmas": simplex['simplex']['neutral_homeostasis'].get('lemmas', []),
                        "definition": simplex['simplex']['neutral_homeostasis'].get('definition', '')
                    },
                    "positive_pole": {
                        "synset": simplex['simplex']['positive_pole']['synset'],
                        "lemmas": simplex['simplex']['positive_pole'].get('lemmas', []),
                        "definition": simplex['simplex']['positive_pole'].get('definition', '')
                    }
                },
                "s_tier_justification": simplex.get('justification', ''),
                "training_priority": priority.lower()
            }
            patch[section].append(entry)

    return patch

def create_s_plus_training_spec(patch: Dict):
    """Create S+ training specification"""

    total_simplexes = patch['metadata']['total_s_tier_simplexes']
    total_lenses = total_simplexes * 3  # 3 lenses per simplex

    spec = {
        "training_run": "S+_three_pole_homeostatic",
        "description": "S-tier three-pole detection lens training for homeostatic steering",
        "total_simplexes": total_simplexes,
        "total_lenses": total_lenses,
        "lenses_per_simplex": 3,
        "training_configuration": {
            "mode": "adaptive_falloff",
            "validation_mode": "falloff_strict",
            "initial_samples": 10,
            "max_cycles": 2,
            "max_samples_per_cycle": [10, 20],
            "tier_thresholds": {
                "A": {"score": 0.50, "cycle": 0},
                "B+": {"score": 0.35, "cycle": 0},
                "B": {"score": 0.23, "cycle": 1},
                "C+": {"score": 0.15, "cycle": 1}
            }
        },
        "data_generation": {
            "prompt_types": [
                "definitional",  # "What is X?"
                "behavioral_neutral",  # "How would someone experiencing X behave?"
                "behavioral_elicitation",  # Prompts that actually elicit the state
                "self_assessment"  # "Am I experiencing X right now?"
            ],
            "samples_per_pole": 30,
            "total_samples_per_simplex": 90,
            "behavioral_coverage_target": 0.6  # 60% behavioral, 40% definitional
        },
        "lens_training_order": {
            "1_critical": {
                "simplexes": len(patch['critical_s_tier']),
                "lenses": len(patch['critical_s_tier']) * 3,
                "estimated_time_hours": len(patch['critical_s_tier']) * 0.5
            },
            "2_high": {
                "simplexes": len(patch['high_priority_s_tier']),
                "lenses": len(patch['high_priority_s_tier']) * 3,
                "estimated_time_hours": len(patch['high_priority_s_tier']) * 0.5
            },
            "3_medium": {
                "simplexes": len(patch['medium_priority_s_tier']),
                "lenses": len(patch['medium_priority_s_tier']) * 3,
                "estimated_time_hours": len(patch['medium_priority_s_tier']) * 0.5
            },
            "4_standard": {
                "simplexes": len(patch['standard_s_tier']),
                "lenses": len(patch['standard_s_tier']) * 3,
                "estimated_time_hours": len(patch['standard_s_tier']) * 0.5
            }
        },
        "validation_requirements": {
            "cross_pole_discrimination": 0.80,  # 80% accuracy distinguishing poles
            "homeostatic_steering_coherence": 1.0,  # 100% coherence when steering to μ0
            "behavioral_generalization": 0.70  # 70% accuracy on behavioral prompts
        },
        "estimated_total_time_hours": total_simplexes * 0.5,
        "estimated_total_samples": total_simplexes * 90
    }

    return spec

def main():
    print("="*80)
    print("S-TIER SIMPLEX PATCH GENERATION")
    print("="*80)

    # Load review
    print("\n1. Loading simplex agentic review...")
    review_data = load_simplex_review()
    print(f"   ✓ Loaded {review_data['metadata']['total_concepts_reviewed']} concept review")

    # Extract S-tier
    print("\n2. Extracting S-tier simplexes...")
    s_tier, missing_sumo, gaps = extract_s_tier_simplexes(review_data)
    print(f"   ✓ Found {len(s_tier)} complete three-pole simplexes")
    print(f"   ✓ Identified {len(missing_sumo)} missing SUMO concepts")
    print(f"   ✓ {len(gaps)} coverage gaps noted")

    # Categorize by priority
    print("\n3. Categorizing by priority...")
    categorized = categorize_s_tier_by_priority(s_tier, gaps)
    for priority, simplexes in categorized.items():
        if simplexes:
            print(f"   ✓ {priority}: {len(simplexes)} simplexes")

    # Create patch
    print("\n4. Creating WordNet patch...")
    patch = create_wordnet_patch(categorized, missing_sumo)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATCH, 'w') as f:
        json.dump(patch, f, indent=2)
    print(f"   ✓ Saved to {OUTPUT_PATCH}")

    # Create S+ training spec
    print("\n5. Creating S+ training specification...")
    spec = create_s_plus_training_spec(patch)
    spec_path = PROJECT_ROOT / "docs" / "S_PLUS_TRAINING_SPEC.json"
    with open(spec_path, 'w') as f:
        json.dump(spec, f, indent=2)
    print(f"   ✓ Saved to {spec_path}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total S-tier simplexes: {patch['metadata']['total_s_tier_simplexes']}")
    print(f"Total lenses needed: {spec['total_lenses']} (3 per simplex)")
    print(f"Estimated training time: {spec['estimated_total_time_hours']:.1f} hours")
    print(f"Estimated samples: {spec['estimated_total_samples']}")
    print(f"\nMissing SUMO concepts: {len(missing_sumo)}")
    if missing_sumo:
        print("\nConcepts needing SUMO definition:")
        for concept in missing_sumo[:10]:
            print(f"  - {concept['synset']}: {concept['definition'][:60]}...")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review missing SUMO concepts and create definitions")
    print("2. Apply patch to layer2.json using apply_motive_emotion_patches.py")
    print("3. Modify sumo_data_generation.py for three-pole training")
    print("4. Run S+ training: train_s_plus_simplexes.py")
    print("5. Validate with test_homeostatic_steering.py")
    print("="*80)

if __name__ == "__main__":
    main()
