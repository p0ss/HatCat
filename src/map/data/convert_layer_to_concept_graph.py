#!/usr/bin/env python3
"""
Convert hierarchical layer JSON to concept graph format for training.
"""

import json
from pathlib import Path
from nltk.corpus import wordnet as wn

def convert_layer_to_concept_graph(layer_path: Path, output_path: Path):
    """Convert layer JSON to concept graph format."""

    with open(layer_path) as f:
        layer_data = json.load(f)

    concepts = []

    for concept_data in layer_data['concepts']:
        sumo_term = concept_data['sumo_term']
        synset_count = concept_data.get('synset_count', 0)
        canonical_synset = concept_data.get('canonical_synset')

        # Skip concepts with no synsets
        if synset_count == 0 or not canonical_synset:
            print(f"Skipping {sumo_term}: no synsets")
            continue

        try:
            synset = wn.synset(canonical_synset)

            # Build concept in training format
            concept = {
                'synset': canonical_synset,
                'name': synset.name(),
                'pos': synset.pos(),
                'definition': synset.definition(),
                'lemmas': synset.lemma_names(),
                'sumo_term': sumo_term,
                'layer': concept_data['layer'],
                'sumo_depth': concept_data['sumo_depth'],
                'synset_count': synset_count,
                'category_children': concept_data.get('category_children', []),

                # Relationships from WordNet
                'hypernyms': [h.name() for h in synset.hypernyms()],
                'hyponyms': [h.name() for h in synset.hyponyms()[:5]],  # Limit for training
                'holonyms': [h.name() for h in synset.member_holonyms() + synset.part_holonyms()],
                'meronyms': [h.name() for h in synset.member_meronyms() + synset.part_meronyms()],

                # Antonyms (from lemmas)
                'antonyms': []
            }

            # Extract antonyms from all lemmas
            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    concept['antonyms'].append(antonym.synset().name())

            concepts.append(concept)
            print(f"✓ {sumo_term} ({canonical_synset}): {len(concept['hypernyms'])} hypernyms, {len(concept['hyponyms'])} hyponyms")

        except Exception as e:
            print(f"✗ Error processing {sumo_term}: {e}")
            continue

    # Create concept graph
    concept_graph = {
        'metadata': {
            'source': str(layer_path),
            'layer': layer_data['metadata']['layer'],
            'total_concepts': len(concepts),
            'description': layer_data['metadata']['description']
        },
        'concepts': concepts
    }

    # Write output
    with open(output_path, 'w') as f:
        json.dump(concept_graph, f, indent=2)

    print(f"\n✓ Wrote {len(concepts)} concepts to {output_path}")
    return concept_graph

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, required=True, help='Layer number (0-6)')
    parser.add_argument('--output-dir', type=Path, default=Path('data/concept_graph'), help='Output directory')
    args = parser.parse_args()

    layer_path = Path(f'data/concept_graph/abstraction_layers/layer{args.layer}.json')
    output_path = args.output_dir / f'layer{args.layer}_concepts.json'

    convert_layer_to_concept_graph(layer_path, output_path)
