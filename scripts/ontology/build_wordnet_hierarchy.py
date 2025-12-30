#!/usr/bin/env python3
"""
Build 5-layer WordNet concept hierarchy using hypernym depth.

Uses WordNet's native taxonomy (hypernym/hyponym relations) to organize
concepts into 5 layers by semantic abstraction level.

Layer 1: Most abstract (top of taxonomy, e.g., "entity", "state", "act")
Layer 2: High-level categories
Layer 3: Mid-level concepts
Layer 4: Specific concepts
Layer 5: Fine-grained leaf concepts

Then enriches each concept with:
- Synonyms (lemmas in same synset)
- Antonyms
- Similar_tos
- Also_sees
- All other WordNet relations
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn

# Download WordNet if needed
try:
    wn.synsets('test')
except LookupError:
    print("Downloading WordNet...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
OUTPUT_DIR = Path("data/concept_graph/wordnet_layers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Compute hypernym depth for each synset
# ---------------------------------------------------------------------
def get_hypernym_depth(synset):
    """
    Get minimum distance from synset to root (entity.n.01, etc).
    Lower number = more abstract.
    """
    return synset.min_depth()


def get_max_depth_for_pos(pos):
    """Get maximum depth for a part of speech."""
    all_synsets = list(wn.all_synsets(pos=pos))
    if not all_synsets:
        return 0
    return max(s.min_depth() for s in all_synsets)


# ---------------------------------------------------------------------
# Get all WordNet relations
# ---------------------------------------------------------------------
def get_relations(synset):
    """Extract all WordNet relations for a synset."""

    # Get antonyms from lemmas
    antonyms = []
    for lemma in synset.lemmas():
        for ant in lemma.antonyms():
            antonyms.append(ant.synset().name())

    return {
        'synset': synset.name(),
        'lemmas': synset.lemma_names(),
        'definition': synset.definition(),
        'examples': synset.examples(),
        'pos': synset.pos(),
        'lexname': synset.lexname(),  # Lexical domain

        # Taxonomic relations
        'hypernyms': [h.name() for h in synset.hypernyms()],
        'hyponyms': [h.name() for h in synset.hyponyms()],
        'instance_hypernyms': [h.name() for h in synset.instance_hypernyms()],
        'instance_hyponyms': [h.name() for h in synset.instance_hyponyms()],

        # Part/whole relations
        'member_holonyms': [h.name() for h in synset.member_holonyms()],
        'substance_holonyms': [h.name() for h in synset.substance_holonyms()],
        'part_holonyms': [h.name() for h in synset.part_holonyms()],
        'member_meronyms': [m.name() for m in synset.member_meronyms()],
        'substance_meronyms': [m.name() for m in synset.substance_meronyms()],
        'part_meronyms': [m.name() for m in synset.part_meronyms()],

        # Similarity relations
        'similar_tos': [s.name() for s in synset.similar_tos()],
        'also_sees': [a.name() for a in synset.also_sees()],
        'attributes': [a.name() for a in synset.attributes()],

        # Verb-specific relations
        'entailments': [e.name() for e in synset.entailments()],
        'causes': [c.name() for c in synset.causes()],
        'verb_groups': [v.name() for v in synset.verb_groups()],

        # Antonyms
        'antonyms': list(set(antonyms)),

        # Depth info
        'min_depth': synset.min_depth(),
        'max_depth': synset.max_depth(),
    }


# ---------------------------------------------------------------------
# Build 5-layer hierarchy
# ---------------------------------------------------------------------
def build_layers():
    """
    Organize all WordNet synsets into 5 layers by hypernym depth.
    """
    print("="*60)
    print("WORDNET HIERARCHY BUILDER")
    print("="*60)

    print("\nComputing depth ranges per POS...")

    # Compute max depth for each POS
    pos_max_depths = {}
    for pos_code, pos_name in [('n', 'noun'), ('v', 'verb'), ('a', 'adj'), ('r', 'adv')]:
        max_d = get_max_depth_for_pos(pos_code)
        pos_max_depths[pos_code] = max_d
        print(f"  {pos_name:8} max depth: {max_d}")

    print("\nProcessing synsets...")

    layers = {1: [], 2: [], 3: [], 4: [], 5: []}
    pos_counts = defaultdict(int)

    total_synsets = len(list(wn.all_synsets()))
    processed = 0

    for synset in wn.all_synsets():
        depth = synset.min_depth()
        pos = synset.pos()
        max_depth = pos_max_depths.get(pos, 20)

        # Assign to layer based on depth percentile within POS
        if max_depth > 0:
            depth_pct = depth / max_depth
            layer = max(1, min(5, int(math.ceil(depth_pct * 5))))
        else:
            layer = 1

        # Get all relations
        entry = get_relations(synset)
        entry['layer'] = layer

        layers[layer].append(entry)
        pos_counts[pos] += 1

        processed += 1
        if processed % 10000 == 0:
            print(f"  Processed {processed}/{total_synsets}...")

    print(f"  ✓ Processed {processed} synsets")

    # Statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)

    print("\nLayer Distribution:")
    for layer in range(1, 6):
        count = len(layers[layer])
        pct = 100 * count / sum(len(layers[i]) for i in range(1, 6))

        # Get POS breakdown for this layer
        layer_pos = defaultdict(int)
        for entry in layers[layer]:
            layer_pos[entry['pos']] += 1

        pos_str = ", ".join(f"{pos}:{ct}" for pos, ct in sorted(layer_pos.items()))

        print(f"  Layer {layer}: {count:6} concepts ({pct:5.1f}%)  [{pos_str}]")

    pos_names = {'n': 'Noun', 'v': 'Verb', 'a': 'Adjective',
                 'r': 'Adverb', 's': 'Adj Satellite'}
    print("\nOverall POS Distribution:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        pos_name = pos_names.get(pos, pos)
        pct = 100 * count / sum(pos_counts.values())
        print(f"  {pos_name:15} {count:6} ({pct:5.1f}%)")

    return layers


# ---------------------------------------------------------------------
# Save layers
# ---------------------------------------------------------------------
def save_layers(layers):
    """Save each layer to a separate JSON file."""
    print("\n" + "="*60)
    print("SAVING LAYERS")
    print("="*60)

    layer_descriptions = {
        1: "Most abstract concepts (top of taxonomy)",
        2: "High-level categories",
        3: "Mid-level concepts",
        4: "Specific concepts",
        5: "Fine-grained leaf concepts"
    }

    for layer_num in range(1, 6):
        output_path = OUTPUT_DIR / f"layer{layer_num}.json"

        # Calculate statistics for this layer
        layer_data = layers[layer_num]

        pos_counts = defaultdict(int)
        lexname_counts = defaultdict(int)
        depth_stats = {'min': float('inf'), 'max': 0, 'avg': 0}

        for entry in layer_data:
            pos_counts[entry['pos']] += 1
            lexname_counts[entry['lexname']] += 1
            depth_stats['min'] = min(depth_stats['min'], entry['min_depth'])
            depth_stats['max'] = max(depth_stats['max'], entry['max_depth'])

        if layer_data:
            depth_stats['avg'] = sum(e['min_depth'] for e in layer_data) / len(layer_data)

        # Sample concepts
        samples = []
        for entry in sorted(layer_data, key=lambda x: len(x['lemmas'][0]))[:10]:
            samples.append({
                'concept': entry['lemmas'][0],
                'synset': entry['synset'],
                'definition': entry['definition'][:80] + "..." if len(entry['definition']) > 80 else entry['definition']
            })

        output_data = {
            'metadata': {
                'layer': layer_num,
                'description': layer_descriptions[layer_num],
                'total_concepts': len(layer_data),
                'pos_distribution': dict(pos_counts),
                'top_lexical_domains': dict(sorted(lexname_counts.items(), key=lambda x: -x[1])[:10]),
                'depth_stats': depth_stats,
                'samples': samples
            },
            'concepts': layer_data
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"  ✓ Layer {layer_num}: {len(layer_data):6} concepts → {output_path}")

    print("\n" + "="*60)
    print("✓ Complete")
    print("="*60)


# ---------------------------------------------------------------------
# Show samples
# ---------------------------------------------------------------------
def show_samples(layers):
    """Show sample concepts from each layer."""
    print("\n" + "="*60)
    print("SAMPLE CONCEPTS PER LAYER")
    print("="*60)

    for layer in range(1, 6):
        print(f"\nLayer {layer}:")

        # Show 10 examples, sorted by simplicity (short names)
        samples = sorted(layers[layer], key=lambda x: len(x['lemmas'][0]))[:10]

        for i, entry in enumerate(samples, 1):
            concept = entry['lemmas'][0]
            definition = entry['definition'][:60] + "..." if len(entry['definition']) > 60 else entry['definition']
            depth = entry['min_depth']
            print(f"  {i:2}. {concept:20} (depth={depth:2}) - {definition}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    layers = build_layers()
    show_samples(layers)
    save_layers(layers)


if __name__ == '__main__':
    main()
