#!/usr/bin/env python3
"""
Generate WordNet patch for noun.motive concepts.

Two strategies:
1. Expand PsychologicalAttribute with all 42 noun.motive synsets
2. Create new Layer 3 concepts for key motivation subcategories

This follows the same approach as the Layer 2/3 sparse concept patches.
"""

import json
from pathlib import Path
from nltk.corpus import wordnet as wn
from datetime import datetime


def get_noun_motive_synsets():
    """Get all synsets in noun.motive domain."""
    motive_synsets = []

    for synset in wn.all_synsets('n'):
        if synset.lexname() == 'noun.motive':
            motive_synsets.append({
                'synset': synset.name(),
                'definition': synset.definition(),
                'lemmas': [l.name() for l in synset.lemmas()],
                'hypernyms': [h.name() for h in synset.hypernyms()],
                'hyponyms': [h.name() for h in synset.hyponyms()]
            })

    return motive_synsets


def categorize_motive_synsets(motive_synsets):
    """
    Categorize noun.motive synsets into semantic clusters.

    Returns categories for potential SUMO sub-concepts.
    """
    categories = {
        'rational_motive': [],
        'irrational_motive': [],
        'ethical_motive': [],
        'urge': [],
        'psychic_energy': [],
        'life_motive': [],
        'uncategorized': []
    }

    # Build hypernym chains
    for synset_data in motive_synsets:
        synset_name = synset_data['synset']
        synset = wn.synset(synset_name)

        # Get full hypernym path
        paths = synset.hypernym_paths()
        if not paths:
            categories['uncategorized'].append(synset_data)
            continue

        # Check which parent it belongs to
        path_synsets = [s.name() for s in paths[0]]

        if 'rational_motive.n.01' in path_synsets:
            categories['rational_motive'].append(synset_data)
        elif 'irrational_motive.n.01' in path_synsets:
            categories['irrational_motive'].append(synset_data)
        elif 'ethical_motive.n.01' in path_synsets:
            categories['ethical_motive'].append(synset_data)
        elif 'urge.n.01' in path_synsets:
            categories['urge'].append(synset_data)
        elif 'psychic_energy.n.01' in path_synsets:
            categories['psychic_energy'].append(synset_data)
        elif 'life.n.13' in path_synsets:
            categories['life_motive'].append(synset_data)
        else:
            categories['uncategorized'].append(synset_data)

    return categories


def generate_patch_strategy_1(motive_synsets):
    """
    Strategy 1: Add all noun.motive synsets to PsychologicalAttribute.

    Simple approach - just expand the existing concept.
    """
    synset_list = [s['synset'] for s in motive_synsets]

    patch = {
        'strategy': 'expand_existing',
        'target_concept': 'PsychologicalAttribute',
        'target_layer': 2,
        'synsets_to_add': synset_list,
        'count': len(synset_list),
        'reasoning': 'Add all noun.motive synsets to existing PsychologicalAttribute concept'
    }

    return patch


def generate_patch_strategy_2(categories):
    """
    Strategy 2: Create new SUMO sub-concepts for motivation categories.

    More structured approach - separate RationalMotive, EthicalMotive, etc.
    """
    new_concepts = []

    # Define which categories are worth separate concepts
    priority_categories = {
        'rational_motive': {
            'sumo_term': 'RationalMotive',
            'display_name': 'Rational Motive',
            'definition': 'A motive that can be defended by reasoning or logical argument',
            'parent_sumo': 'PsychologicalAttribute',
            'layer': 3,
            'priority': 'CRITICAL'
        },
        'irrational_motive': {
            'sumo_term': 'IrrationalMotive',
            'display_name': 'Irrational Motive',
            'definition': 'A motivation that is inconsistent with reason or logic',
            'parent_sumo': 'PsychologicalAttribute',
            'layer': 3,
            'priority': 'CRITICAL'
        },
        'ethical_motive': {
            'sumo_term': 'EthicalMotive',
            'display_name': 'Ethical Motive',
            'definition': 'Motivation based on ideas of right and wrong',
            'parent_sumo': 'PsychologicalAttribute',
            'layer': 3,
            'priority': 'CRITICAL'
        },
        'urge': {
            'sumo_term': 'Urge',
            'display_name': 'Urge',
            'definition': 'An instinctive motive or impulse',
            'parent_sumo': 'PsychologicalAttribute',
            'layer': 3,
            'priority': 'MEDIUM'
        }
    }

    for category_key, concept_def in priority_categories.items():
        synsets = categories.get(category_key, [])
        synset_list = [s['synset'] for s in synsets]

        if synset_list:  # Only create if we have synsets
            concept = {
                **concept_def,
                'synsets': synset_list,
                'synset_count': len(synset_list),
                'direct_synset_count': len(synset_list),
                'is_category_lens': True,
                'category_children': [],
                'mapping_metadata': {
                    'source': 'noun_motive_patch',
                    'patch_date': datetime.now().isoformat(),
                    'manually_curated': True
                }
            }
            new_concepts.append(concept)

    # Remaining synsets go to PsychologicalAttribute
    remaining_synsets = []
    for key in ['psychic_energy', 'life_motive', 'uncategorized']:
        remaining_synsets.extend([s['synset'] for s in categories.get(key, [])])

    patch = {
        'strategy': 'create_subconcepts',
        'new_concepts': new_concepts,
        'expand_psychological_attribute': remaining_synsets,
        'reasoning': 'Create dedicated concepts for critical motivation types, expand PsychologicalAttribute with remaining synsets'
    }

    return patch


def print_analysis(motive_synsets, categories):
    """Print analysis of noun.motive domain."""
    print("="*80)
    print("NOUN.MOTIVE ANALYSIS")
    print("="*80)

    print(f"\nTotal noun.motive synsets: {len(motive_synsets)}")

    print("\nBy category:")
    for category, synsets in sorted(categories.items()):
        if synsets:
            print(f"  {category}: {len(synsets)}")

    # Show critical concepts
    print("\nCritical motivation concepts (first 5 each):")
    for category in ['rational_motive', 'irrational_motive', 'ethical_motive', 'urge']:
        synsets = categories.get(category, [])
        if synsets:
            print(f"\n  {category.upper()}:")
            for s in synsets[:5]:
                print(f"    - {s['synset']}: {s['definition']}")


def save_patches(strategy1, strategy2, output_dir):
    """Save both patch strategies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strategy 1
    with open(output_dir / 'motivation_patch_strategy1.json', 'w') as f:
        json.dump(strategy1, f, indent=2)

    # Strategy 2
    with open(output_dir / 'motivation_patch_strategy2.json', 'w') as f:
        json.dump(strategy2, f, indent=2)

    print(f"\n✓ Patches saved to: {output_dir}")


def recommend_strategy(categories):
    """Recommend which strategy to use."""
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    critical_categories = ['rational_motive', 'irrational_motive', 'ethical_motive']
    critical_count = sum(len(categories.get(c, [])) for c in critical_categories)

    print("\nStrategy 1 (Expand PsychologicalAttribute):")
    print("  Pros:")
    print("    - Simple, minimal changes")
    print("    - All noun.motive synsets in one place")
    print("  Cons:")
    print("    - No semantic subcategorization")
    print("    - Harder to find specific opposites (rational vs irrational)")
    print("    - Lumps critical (ethical_motive) with low-priority (psychic_energy)")

    print("\nStrategy 2 (Create Sub-Concepts):")
    print("  Pros:")
    print("    - Semantic organization (RationalMotive, EthicalMotive separate)")
    print("    - Easier to find opposites (RationalMotive ↔ IrrationalMotive)")
    print("    - Critical concepts get dedicated lenses")
    print(f"    - Captures {critical_count} critical motivation synsets in focused concepts")
    print("  Cons:")
    print("    - More layer file changes (add to Layer 3)")
    print("    - More concepts to train (4 new concepts vs 1 expansion)")

    print("\n**RECOMMENDED: Strategy 2** (Create Sub-Concepts)")
    print("  Reason: Semantic separation is valuable for:")
    print("    - Fisher-LDA with clear opposites (rational ↔ irrational)")
    print("    - AI safety monitoring (ethical_motive is critical)")
    print("    - Steering quality (separate axes for different motivation types)")


def main():
    """Generate motivation concept patches."""
    print("Generating noun.motive WordNet patches...")

    # Get all noun.motive synsets
    motive_synsets = get_noun_motive_synsets()

    # Categorize them
    categories = categorize_motive_synsets(motive_synsets)

    # Print analysis
    print_analysis(motive_synsets, categories)

    # Generate both strategies
    strategy1 = generate_patch_strategy_1(motive_synsets)
    strategy2 = generate_patch_strategy_2(categories)

    # Save patches
    output_dir = Path(__file__).parent.parent / 'results' / 'motivation_patches'
    save_patches(strategy1, strategy2, output_dir)

    # Recommend strategy
    recommend_strategy(categories)

    print("\nNext steps:")
    print("  1. Review patches in results/motivation_patches/")
    print("  2. Choose strategy (recommended: strategy 2)")
    print("  3. Apply patch to layer files")
    print("  4. Run agentic opposite review for new concepts")
    print("  5. Integrate opposites into 4-component data generation")


if __name__ == '__main__':
    main()
