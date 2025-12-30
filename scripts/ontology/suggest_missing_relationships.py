#!/usr/bin/env python3
"""
Suggest missing relationships for concepts lacking WordNet connections.

Approaches:
1. SUMO sibling relationships (share same parent in category_children)
2. WordNet hypernym siblings (concepts with same hypernym)
3. SUMO cousin relationships (children of parent's siblings)
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from nltk.corpus import wordnet as wn


def load_all_concepts() -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load all concepts from all layers."""
    all_concepts = []
    concept_map = {}

    for layer_num in range(6):
        layer_path = Path(f"data/concept_graph/abstraction_layers/layer{layer_num}.json")
        if layer_path.exists():
            with open(layer_path) as f:
                data = json.load(f)

            for concept in data.get('concepts', []):
                all_concepts.append(concept)
                concept_map[concept['sumo_term']] = concept

    return all_concepts, concept_map


def build_sumo_parent_map(all_concepts: List[Dict]) -> Dict[str, List[str]]:
    """Build map of parent -> list of children from SUMO hierarchy."""
    parent_to_children = defaultdict(list)

    for concept in all_concepts:
        parent_term = concept['sumo_term']
        children = concept.get('category_children', [])

        if children:
            parent_to_children[parent_term].extend(children)

    return dict(parent_to_children)


def build_child_to_parents_map(parent_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Build reverse map of child -> list of parents."""
    child_to_parents = defaultdict(list)

    for parent, children in parent_map.items():
        for child in children:
            child_to_parents[child].append(parent)

    return dict(child_to_parents)


def get_sumo_siblings(concept_name: str, parent_map: Dict[str, List[str]],
                      child_to_parents: Dict[str, List[str]]) -> List[str]:
    """Get SUMO siblings (concepts sharing same parent)."""
    siblings = []

    # Find parents of this concept
    parents = child_to_parents.get(concept_name, [])

    # For each parent, get all other children
    for parent in parents:
        for sibling in parent_map.get(parent, []):
            if sibling != concept_name and sibling not in siblings:
                siblings.append(sibling)

    return siblings


def get_sumo_cousins(concept_name: str, parent_map: Dict[str, List[str]],
                     child_to_parents: Dict[str, List[str]]) -> List[str]:
    """Get SUMO cousins (children of parent's siblings)."""
    cousins = []

    # Find parents
    parents = child_to_parents.get(concept_name, [])

    # For each parent, find its siblings
    for parent in parents:
        parent_siblings = get_sumo_siblings(parent, parent_map, child_to_parents)

        # Get children of parent's siblings
        for parent_sibling in parent_siblings:
            cousins.extend(parent_map.get(parent_sibling, []))

    return cousins


def get_wordnet_hypernym_siblings(synset_id: str) -> List[Tuple[str, str]]:
    """Get siblings from WordNet (concepts with same hypernym)."""
    siblings = []

    if not synset_id:
        return siblings

    try:
        synset = wn.synset(synset_id)

        # Get hypernyms
        for hypernym in synset.hypernyms():
            # Get all hyponyms of this hypernym (siblings)
            for sibling_synset in hypernym.hyponyms():
                if sibling_synset.name() != synset_id:
                    # Get primary lemma
                    lemma = sibling_synset.lemma_names()[0].replace('_', ' ')
                    siblings.append((sibling_synset.name(), lemma))

    except Exception:
        pass

    return siblings


def has_wordnet_relationships(synset_id: str) -> bool:
    """Check if synset has any useful WordNet relationships."""
    if not synset_id:
        return False

    try:
        synset = wn.synset(synset_id)

        return any([
            synset.hypernyms(),
            synset.hyponyms(),
            synset.member_meronyms(),
            synset.part_meronyms(),
            synset.member_holonyms(),
            synset.part_holonyms(),
            any(lemma.antonyms() for lemma in synset.lemmas())
        ])
    except Exception:
        return False


def analyze_relationship_gaps(all_concepts: List[Dict], concept_map: Dict[str, Dict],
                               parent_map: Dict[str, List[str]],
                               child_to_parents: Dict[str, List[str]]) -> None:
    """Analyze concepts lacking relationships and suggest alternatives."""

    print("=" * 80)
    print("RELATIONSHIP GAP ANALYSIS")
    print("=" * 80)

    # Find concepts without WordNet relationships
    concepts_without_wordnet = []

    for concept in all_concepts:
        synset_id = concept.get('canonical_synset')
        if not has_wordnet_relationships(synset_id):
            concepts_without_wordnet.append(concept)

    print(f"\nTotal concepts without WordNet relationships: {len(concepts_without_wordnet)}")
    print(f"  ({100*len(concepts_without_wordnet)/len(all_concepts):.1f}% of {len(all_concepts)} total concepts)")

    # Analyze alternative relationship sources
    with_sumo_siblings = 0
    with_sumo_cousins = 0
    with_wordnet_siblings = 0
    truly_isolated = 0

    suggestions = []

    for concept in concepts_without_wordnet:
        concept_name = concept['sumo_term']
        synset_id = concept.get('canonical_synset')
        layer = concept['layer']

        sumo_siblings = get_sumo_siblings(concept_name, parent_map, child_to_parents)
        sumo_cousins = get_sumo_cousins(concept_name, parent_map, child_to_parents)
        wordnet_siblings = get_wordnet_hypernym_siblings(synset_id)

        has_alternative = False
        suggestion = {
            'concept': concept_name,
            'layer': layer,
            'synset': synset_id,
            'sumo_siblings': sumo_siblings,
            'sumo_cousins': sumo_cousins,
            'wordnet_siblings': [(s[1], s[0]) for s in wordnet_siblings[:5]],  # (lemma, synset)
            'recommendation': []
        }

        if sumo_siblings:
            with_sumo_siblings += 1
            has_alternative = True
            suggestion['recommendation'].append(
                f"Use {len(sumo_siblings)} SUMO siblings from shared parent"
            )

        if sumo_cousins:
            with_sumo_cousins += 1
            has_alternative = True
            suggestion['recommendation'].append(
                f"Use {len(sumo_cousins)} SUMO cousins from parent's siblings"
            )

        if wordnet_siblings:
            with_wordnet_siblings += 1
            has_alternative = True
            suggestion['recommendation'].append(
                f"Use {len(wordnet_siblings)} WordNet hypernym siblings"
            )

        if not has_alternative:
            truly_isolated += 1
            suggestion['recommendation'].append("No clear relationships found - may need manual curation")

        suggestions.append(suggestion)

    print(f"\nAlternative relationship sources:")
    print(f"  SUMO siblings available: {with_sumo_siblings} ({100*with_sumo_siblings/len(concepts_without_wordnet):.1f}%)")
    print(f"  SUMO cousins available: {with_sumo_cousins} ({100*with_sumo_cousins/len(concepts_without_wordnet):.1f}%)")
    print(f"  WordNet siblings available: {with_wordnet_siblings} ({100*with_wordnet_siblings/len(concepts_without_wordnet):.1f}%)")
    print(f"  Truly isolated: {truly_isolated} ({100*truly_isolated/len(concepts_without_wordnet):.1f}%)")

    # Show examples by layer
    print("\n" + "=" * 80)
    print("DETAILED SUGGESTIONS BY LAYER")
    print("=" * 80)

    by_layer = defaultdict(list)
    for sugg in suggestions:
        by_layer[sugg['layer']].append(sugg)

    for layer_num in sorted(by_layer.keys()):
        layer_suggestions = by_layer[layer_num]
        print(f"\n{'='*80}")
        print(f"LAYER {layer_num} ({len(layer_suggestions)} concepts)")
        print(f"{'='*80}")

        # Show first 5 examples with details
        for sugg in layer_suggestions[:5]:
            print(f"\n{sugg['concept']}:")
            if sugg['synset']:
                print(f"  Synset: {sugg['synset']} (but no relationships)")
            else:
                print(f"  No synset mapped")

            if sugg['sumo_siblings']:
                print(f"  ✓ SUMO siblings ({len(sugg['sumo_siblings'])}): {', '.join(sugg['sumo_siblings'][:3])}")
                if len(sugg['sumo_siblings']) > 3:
                    print(f"    ... and {len(sugg['sumo_siblings'])-3} more")

            if sugg['sumo_cousins']:
                print(f"  ✓ SUMO cousins ({len(sugg['sumo_cousins'])}): {', '.join(sugg['sumo_cousins'][:3])}")
                if len(sugg['sumo_cousins']) > 3:
                    print(f"    ... and {len(sugg['sumo_cousins'])-3} more")

            if sugg['wordnet_siblings']:
                print(f"  ✓ WordNet siblings ({len(sugg['wordnet_siblings'])}): {', '.join([s[0] for s in sugg['wordnet_siblings'][:3]])}")
                if len(sugg['wordnet_siblings']) > 3:
                    print(f"    ... and {len(sugg['wordnet_siblings'])-3} more")

            print(f"  → {' | '.join(sugg['recommendation'])}")

        if len(layer_suggestions) > 5:
            print(f"\n  ... and {len(layer_suggestions)-5} more concepts in this layer")

    # Implementation recommendations
    print("\n" + "=" * 80)
    print("IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 80)

    print("""
1. EXTEND generate_relationship_prompts() to include SUMO sibling relationships:

   def generate_sumo_sibling_prompts(
       concept: str,
       siblings: List[str],
       all_concepts_map: Dict[str, Dict],
       n_samples: int = 3
   ) -> List[str]:
       '''Generate prompts highlighting sibling relationships.'''
       if not siblings:
           return []

       prompts = []
       sampled = random.sample(siblings, min(n_samples, len(siblings)))

       for sibling in sampled:
           sibling_spaced = split_camel_case(sibling)
           concept_spaced = split_camel_case(concept)

           templates = [
               f"'{concept_spaced}' and '{sibling_spaced}' are related concepts.",
               f"Compare '{concept_spaced}' with '{sibling_spaced}'.",
               f"'{concept_spaced}' is related to '{sibling_spaced}' as sibling concepts.",
           ]
           prompts.append(random.choice(templates))

       return prompts

2. ADD to create_sumo_training_dataset():

   # After WordNet relationships, add SUMO sibling relationships
   if use_sumo_siblings:
       siblings = get_sumo_siblings(concept_name, parent_map, child_to_parents)
       sibling_prompts = generate_sumo_sibling_prompts(
           concept_name,
           siblings,
           all_concepts,
           n_samples=3
       )
       rel_prompts.extend(sibling_prompts)

3. PRIORITIZATION:
   - SUMO siblings: Most reliable (explicit hierarchy)
   - WordNet hypernym siblings: Good for synset-mapped concepts
   - SUMO cousins: Weaker signal but still useful
   - Skip complex chains (antonym->similar->antonym) - not worth the complexity

4. FALLBACK for truly isolated concepts:
   - Increase weight on definition-based prompts
   - Use broader negative sampling (no need for hard negatives)
   - May naturally have lower accuracy - acceptable for rare/abstract concepts
""")

    # Save detailed suggestions to JSON
    output_path = Path("results/relationship_suggestions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total_concepts': len(all_concepts),
                'without_wordnet': len(concepts_without_wordnet),
                'with_sumo_siblings': with_sumo_siblings,
                'with_sumo_cousins': with_sumo_cousins,
                'with_wordnet_siblings': with_wordnet_siblings,
                'truly_isolated': truly_isolated,
            },
            'suggestions': suggestions
        }, f, indent=2)

    print(f"\n✓ Detailed suggestions saved to: {output_path}")


def main():
    print("Loading concepts...")
    all_concepts, concept_map = load_all_concepts()
    print(f"✓ Loaded {len(all_concepts)} concepts")

    print("\nBuilding SUMO hierarchy maps...")
    parent_map = build_sumo_parent_map(all_concepts)
    child_to_parents = build_child_to_parents_map(parent_map)
    print(f"✓ Mapped {len(parent_map)} parent-child relationships")

    analyze_relationship_gaps(all_concepts, concept_map, parent_map, child_to_parents)

    return 0


if __name__ == '__main__':
    sys.exit(main())
