#!/usr/bin/env python3
"""
Analyze SUMO concept coverage across layers:
- How many lack synset descriptions
- How many lack antonyms
- Whether WordNet can provide better coverage
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nltk.corpus import wordnet as wn


def analyze_synset_coverage(synset_id: str) -> Dict:
    """
    Analyze what WordNet provides for a synset.

    Returns:
        Dict with keys: definition, hypernyms, hyponyms, meronyms, holonyms, antonyms
    """
    coverage = {
        'has_definition': False,
        'has_hypernyms': False,
        'has_hyponyms': False,
        'has_meronyms': False,
        'has_holonyms': False,
        'has_antonyms': False,
        'definition': None,
        'hypernym_count': 0,
        'hyponym_count': 0,
        'meronym_count': 0,
        'holonym_count': 0,
        'antonym_count': 0,
    }

    if not synset_id:
        return coverage

    try:
        synset = wn.synset(synset_id)

        # Definition
        definition = synset.definition()
        if definition:
            coverage['has_definition'] = True
            coverage['definition'] = definition

        # Hypernyms (broader)
        hypernyms = synset.hypernyms()
        if hypernyms:
            coverage['has_hypernyms'] = True
            coverage['hypernym_count'] = len(hypernyms)

        # Hyponyms (more specific)
        hyponyms = synset.hyponyms()
        if hyponyms:
            coverage['has_hyponyms'] = True
            coverage['hyponym_count'] = len(hyponyms)

        # Meronyms (parts)
        meronyms = synset.member_meronyms() + synset.part_meronyms() + synset.substance_meronyms()
        if meronyms:
            coverage['has_meronyms'] = True
            coverage['meronym_count'] = len(meronyms)

        # Holonyms (wholes)
        holonyms = synset.member_holonyms() + synset.part_holonyms() + synset.substance_holonyms()
        if holonyms:
            coverage['has_holonyms'] = True
            coverage['holonym_count'] = len(holonyms)

        # Antonyms (from lemmas)
        antonyms = []
        for lemma in synset.lemmas():
            for ant in lemma.antonyms():
                antonyms.append(ant.synset())
        if antonyms:
            coverage['has_antonyms'] = True
            coverage['antonym_count'] = len(antonyms)

    except Exception as e:
        pass

    return coverage


def load_layer_concepts(layer_num: int) -> List[Dict]:
    """Load concepts from a layer file."""
    layer_path = Path(f"data/concept_graph/abstraction_layers/layer{layer_num}.json")
    if not layer_path.exists():
        return []

    with open(layer_path) as f:
        data = json.load(f)

    return data.get('concepts', [])


def main():
    print("=" * 80)
    print("SUMO CONCEPT COVERAGE ANALYSIS")
    print("=" * 80)

    # Analyze all layers
    all_layers = []
    for layer_num in range(6):
        concepts = load_layer_concepts(layer_num)
        if concepts:
            all_layers.append((layer_num, concepts))
            print(f"\n✓ Loaded Layer {layer_num}: {len(concepts)} concepts")

    print("\n" + "=" * 80)
    print("ANALYZING COVERAGE")
    print("=" * 80)

    # Overall statistics
    total_concepts = 0
    lacking_synset = []
    lacking_definition = []
    lacking_antonyms = []
    has_wordnet_relationships = []

    # Per-layer statistics
    layer_stats = {}

    for layer_num, concepts in all_layers:
        print(f"\n{'='*80}")
        print(f"LAYER {layer_num}")
        print(f"{'='*80}")

        layer_stats[layer_num] = {
            'total': len(concepts),
            'no_synset': 0,
            'no_definition': 0,
            'no_antonyms': 0,
            'has_hypernyms': 0,
            'has_hyponyms': 0,
            'has_meronyms': 0,
            'has_holonyms': 0,
            'has_antonyms': 0,
        }

        for concept in concepts:
            total_concepts += 1
            concept_name = concept['sumo_term']
            canonical_synset = concept.get('canonical_synset')
            definition = concept.get('definition', '')

            # Check if synset is missing
            if not canonical_synset:
                lacking_synset.append((layer_num, concept_name))
                layer_stats[layer_num]['no_synset'] += 1

                # Check if definition is just placeholder
                if not definition or definition == f"SUMO category: {concept_name}":
                    lacking_definition.append((layer_num, concept_name))
                    layer_stats[layer_num]['no_definition'] += 1

                lacking_antonyms.append((layer_num, concept_name))
                layer_stats[layer_num]['no_antonyms'] += 1
                continue

            # Analyze WordNet coverage
            coverage = analyze_synset_coverage(canonical_synset)

            if not coverage['has_definition']:
                lacking_definition.append((layer_num, concept_name, canonical_synset))
                layer_stats[layer_num]['no_definition'] += 1

            if not coverage['has_antonyms']:
                lacking_antonyms.append((layer_num, concept_name, canonical_synset))
                layer_stats[layer_num]['no_antonyms'] += 1
            else:
                layer_stats[layer_num]['has_antonyms'] += 1

            # Track relationship availability
            if coverage['has_hypernyms']:
                layer_stats[layer_num]['has_hypernyms'] += 1
            if coverage['has_hyponyms']:
                layer_stats[layer_num]['has_hyponyms'] += 1
            if coverage['has_meronyms']:
                layer_stats[layer_num]['has_meronyms'] += 1
            if coverage['has_holonyms']:
                layer_stats[layer_num]['has_holonyms'] += 1

            # Check if we have any useful relationships
            if any([
                coverage['has_hypernyms'],
                coverage['has_hyponyms'],
                coverage['has_meronyms'],
                coverage['has_holonyms']
            ]):
                has_wordnet_relationships.append((layer_num, concept_name, coverage))

        # Print layer summary
        stats = layer_stats[layer_num]
        print(f"\nLayer {layer_num} Summary:")
        print(f"  Total concepts: {stats['total']}")
        print(f"  Missing synset: {stats['no_synset']} ({100*stats['no_synset']/stats['total']:.1f}%)")
        print(f"  Missing definition: {stats['no_definition']} ({100*stats['no_definition']/stats['total']:.1f}%)")
        print(f"  Missing antonyms: {stats['no_antonyms']} ({100*stats['no_antonyms']/stats['total']:.1f}%)")
        print(f"\n  WordNet relationships available:")
        print(f"    Hypernyms: {stats['has_hypernyms']} ({100*stats['has_hypernyms']/stats['total']:.1f}%)")
        print(f"    Hyponyms: {stats['has_hyponyms']} ({100*stats['has_hyponyms']/stats['total']:.1f}%)")
        print(f"    Meronyms: {stats['has_meronyms']} ({100*stats['has_meronyms']/stats['total']:.1f}%)")
        print(f"    Holonyms: {stats['has_holonyms']} ({100*stats['has_holonyms']/stats['total']:.1f}%)")
        print(f"    Antonyms: {stats['has_antonyms']} ({100*stats['has_antonyms']/stats['total']:.1f}%)")

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"\nTotal concepts analyzed: {total_concepts}")
    print(f"\nConcepts lacking canonical_synset: {len(lacking_synset)} ({100*len(lacking_synset)/total_concepts:.1f}%)")
    print(f"Concepts lacking good definition: {len(lacking_definition)} ({100*len(lacking_definition)/total_concepts:.1f}%)")
    print(f"Concepts lacking antonyms: {len(lacking_antonyms)} ({100*len(lacking_antonyms)/total_concepts:.1f}%)")
    print(f"\nConcepts with WordNet relationships: {len(has_wordnet_relationships)} ({100*len(has_wordnet_relationships)/total_concepts:.1f}%)")

    # Detail on concepts lacking synsets
    if lacking_synset:
        print("\n" + "=" * 80)
        print("CONCEPTS WITHOUT CANONICAL_SYNSET")
        print("=" * 80)
        by_layer = defaultdict(list)
        for layer_num, concept_name in lacking_synset:
            by_layer[layer_num].append(concept_name)

        for layer_num in sorted(by_layer.keys()):
            print(f"\nLayer {layer_num} ({len(by_layer[layer_num])} concepts):")
            for name in sorted(by_layer[layer_num]):
                print(f"  - {name}")

    # Detail on concepts with poor definitions
    if lacking_definition:
        print("\n" + "=" * 80)
        print("CONCEPTS WITH POOR/MISSING DEFINITIONS")
        print("=" * 80)
        by_layer = defaultdict(list)
        for item in lacking_definition:
            if len(item) == 2:
                layer_num, concept_name = item
                by_layer[layer_num].append((concept_name, None))
            else:
                layer_num, concept_name, synset = item
                by_layer[layer_num].append((concept_name, synset))

        for layer_num in sorted(by_layer.keys()):
            print(f"\nLayer {layer_num} ({len(by_layer[layer_num])} concepts):")
            for name, synset in sorted(by_layer[layer_num]):
                if synset:
                    print(f"  - {name} (synset: {synset})")
                else:
                    print(f"  - {name} (no synset)")

    # Antonym availability
    print("\n" + "=" * 80)
    print("ANTONYM AVAILABILITY ANALYSIS")
    print("=" * 80)
    has_antonyms = total_concepts - len(lacking_antonyms)
    print(f"\nConcepts WITH antonyms: {has_antonyms} ({100*has_antonyms/total_concepts:.1f}%)")
    print(f"Concepts WITHOUT antonyms: {len(lacking_antonyms)} ({100*len(lacking_antonyms)/total_concepts:.1f}%)")
    print("\nNote: Most concepts don't have true antonyms in WordNet.")
    print("However, we can use other relationships (hypernyms, hyponyms, etc.) for training.")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\n1. MISSING SYNSETS:")
    print(f"   {len(lacking_synset)} concepts lack canonical_synset mappings.")
    print("   → Search WordNet for suitable synsets and add to layer JSON files")
    print("   → Or use SUMO definition as primary training source")

    print("\n2. POOR DEFINITIONS:")
    print(f"   {len(lacking_definition)} concepts have inadequate definitions.")
    print("   → For synset-mapped concepts: Extract WordNet definition")
    print("   → For unmapped concepts: Improve SUMO definitions")

    print("\n3. ANTONYMS:")
    print(f"   {len(lacking_antonyms)} concepts lack antonyms ({100*len(lacking_antonyms)/total_concepts:.1f}%).")
    print("   → This is EXPECTED - most concepts don't have true opposites")
    print("   → Use AI symmetry mappings (complements/neutrals) where available")
    print("   → Use hierarchical relationships (siblings, cousins) as hard negatives")
    print("   → Current hard negative approach is appropriate")

    print("\n4. WORDNET RELATIONSHIPS:")
    print(f"   {len(has_wordnet_relationships)} concepts have WordNet relationships.")
    print("   → Already leveraging these in generate_wordnet_relationship_prompts()")
    print("   → Could increase n_samples for concepts with rich WordNet data")

    print("\n" + "=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
