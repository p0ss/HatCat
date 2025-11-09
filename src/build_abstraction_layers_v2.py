#!/usr/bin/env python3
"""
Build abstraction-based concept layers with exponential scaling:
- Layer 0: ~100 concepts (SUMO top-level categories)
- Layer 1: ~1,000 concepts (major semantic domains)
- Layer 2: ~10,000 concepts (common everyday concepts)
- Layer 3: ~100,000 concepts (specific concepts)
- Layer 4: remaining (rare/technical)

Strategy:
1. Compute SUMO term depths from Entity root
2. For each SUMO term, select most prototypical synsets by frequency
3. Assign layers based on SUMO abstraction level + frequency
"""

import re
import json
from pathlib import Path
from collections import defaultdict
import networkx as nx

import nltk
from nltk.corpus import wordnet as wn, brown

# Download if needed
try:
    wn.synsets('test')
    brown.words()
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('brown')

INPUT_DIR = Path("data/concept_graph/sumo_layers")
SUMO_KIF = Path("data/concept_graph/sumo_source/Merge.kif")
OUTPUT_DIR = Path("data/concept_graph/abstraction_layers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sumo_hierarchy():
    """Parse SUMO KIF and compute depth of each term from Entity."""
    print("Loading SUMO hierarchy...")

    with open(SUMO_KIF) as f:
        kif_text = f.read()

    # Build graph (child -> parent edges)
    child_to_parent = {}
    for line in kif_text.splitlines():
        if not line.strip() or line.startswith(';'):
            continue
        match = re.match(r'^\(subclass\s+(\S+)\s+(\S+)\)', line.strip())
        if match:
            child, parent = match.groups()
            if not child.startswith('?') and not parent.startswith('?'):
                child_to_parent[child] = parent

    # Build networkx graph for depth computation
    dg = nx.DiGraph()
    for child, parent in child_to_parent.items():
        dg.add_edge(child, parent)

    # Reverse so Entity is root
    dg = dg.reverse()

    # Compute depths
    depths = nx.single_source_shortest_path_length(dg, 'Entity')

    print(f"  ✓ Loaded {len(depths)} SUMO terms with depths")
    return depths, child_to_parent


def compute_word_frequencies():
    """Get word frequencies from Brown corpus."""
    print("Computing word frequencies...")

    freq_dist = defaultdict(int)
    for word in brown.words():
        freq_dist[word.lower()] += 1

    print(f"  ✓ Computed frequencies for {len(freq_dist)} words")
    return freq_dist


def get_synset_frequency(synset, freq_dist):
    """Get max frequency among synset's lemmas."""
    max_freq = 0
    for lemma in synset.lemma_names():
        word = lemma.lower().replace('_', ' ')
        # Try exact match
        freq = freq_dist.get(word, 0)
        if freq > max_freq:
            max_freq = freq
        # Try first word if multi-word
        if '_' in lemma:
            first = lemma.split('_')[0].lower()
            freq = freq_dist.get(first, 0)
            if freq > max_freq:
                max_freq = freq
    return max_freq


def load_all_concepts():
    """Load all concepts from existing layer files."""
    print("Loading existing concepts...")

    all_concepts = []
    for layer in range(1, 6):
        with open(INPUT_DIR / f"layer{layer}.json") as f:
            data = json.load(f)
            all_concepts.extend(data['concepts'])

    print(f"  ✓ Loaded {len(all_concepts)} concepts")
    return all_concepts


def assign_abstraction_layers(concepts, sumo_depths, freq_dist):
    """
    Assign concepts to abstraction layers by sampling representatives per SUMO term.

    Strategy:
    - Group concepts by SUMO term
    - For each SUMO term, pick N most frequent synsets based on depth
    - Layer 0: depth 2-3 SUMO terms × 2-3 synsets = ~100
    - Layer 1: depth 4 SUMO terms × ~10 synsets = ~1,000
    - Layer 2: depth 5-6 SUMO terms × ~30-60 synsets = ~10,000
    - Layer 3: depth 7-8 remaining synsets
    - Layer 4: depth 9+ rare synsets
    """
    print("Assigning abstraction layers...")

    # Compute frequency for each concept
    for concept in concepts:
        try:
            synset = wn.synset(concept['synset'])
            concept['frequency'] = get_synset_frequency(synset, freq_dist)
        except:
            concept['frequency'] = 0

    # Group by SUMO term
    sumo_groups = defaultdict(list)
    for concept in concepts:
        sumo_term = concept['sumo_term']
        sumo_groups[sumo_term].append(concept)

    # Sort each group by frequency (most common first)
    for term in sumo_groups:
        sumo_groups[term].sort(key=lambda x: -x['frequency'])

    layers = {0: [], 1: [], 2: [], 3: [], 4: []}

    # Layer 0: depth 2-3, pick top 2-3 per SUMO term
    for term, group in sumo_groups.items():
        depth = sumo_depths.get(term, 10)
        if depth <= 3:
            # Take top 2-3 representatives
            n_samples = 3 if depth == 2 else 2
            for concept in group[:n_samples]:
                concept['layer'] = 0
                layers[0].append(concept)

    # Layer 1: depth 4, pick top ~10 per SUMO term
    for term, group in sumo_groups.items():
        depth = sumo_depths.get(term, 10)
        if depth == 4:
            for concept in group[:10]:
                concept['layer'] = 1
                layers[1].append(concept)

    # Layer 2: depth 5-6, pick top ~30-60 per SUMO term
    for term, group in sumo_groups.items():
        depth = sumo_depths.get(term, 10)
        if depth == 5:
            for concept in group[:60]:
                concept['layer'] = 2
                layers[2].append(concept)
        elif depth == 6:
            for concept in group[:30]:
                concept['layer'] = 2
                layers[2].append(concept)

    # Layer 3: depth 5-6 remaining (non-sampled) + depth 7-8 all
    for term, group in sumo_groups.items():
        depth = sumo_depths.get(term, 10)
        if depth == 5:
            # Add remaining (after top 60 sampled in Layer 2)
            for concept in group[60:]:
                concept['layer'] = 3
                layers[3].append(concept)
        elif depth == 6:
            # Add remaining (after top 30 sampled in Layer 2)
            for concept in group[30:]:
                concept['layer'] = 3
                layers[3].append(concept)
        elif depth in [7, 8]:
            # Add all
            for concept in group:
                concept['layer'] = 3
                layers[3].append(concept)

    # Layer 4: depth 4 remaining (non-sampled) + depth 9+ rare
    for term, group in sumo_groups.items():
        depth = sumo_depths.get(term, 10)
        if depth == 4:
            # Add remaining (after top 10 sampled in Layer 1)
            for concept in group[10:]:
                concept['layer'] = 4
                layers[4].append(concept)
        elif depth >= 9:
            # Add all rare concepts
            for concept in group:
                concept['layer'] = 4
                layers[4].append(concept)

    # Print distribution
    print("\n  Layer distribution:")
    for layer_num in range(5):
        count = len(layers[layer_num])
        if count > 0:
            avg_depth = sum(sumo_depths.get(c['sumo_term'], 10) for c in layers[layer_num]) / count
            avg_freq = sum(c['frequency'] for c in layers[layer_num]) / count
            unique_sumo = len(set(c['sumo_term'] for c in layers[layer_num]))
            print(f"    Layer {layer_num}: {count:6} concepts ({unique_sumo:3} SUMO terms, avg depth: {avg_depth:.1f}, avg freq: {avg_freq:.0f})")

    return layers


def save_layers(layers, child_to_parent):
    """Save abstraction layers to JSON with hierarchical activation metadata."""
    print("\nSaving abstraction layers...")

    descriptions = {
        0: "Top-level ontological categories (proprioception baseline - always active)",
        1: "Major semantic domains (activated by Layer 0 parent concepts)",
        2: "Common everyday concepts (activated by Layer 1 parent concepts)",
        3: "Specific detailed concepts (activated by Layer 2 parent concepts)",
        4: "Rare and technical concepts (activated by Layer 1-3 parent concepts)"
    }

    # Build parent→children mapping for hierarchical activation
    # Map: SUMO term → list of synsets that are its children
    sumo_to_synsets = defaultdict(list)
    for layer_data in layers.values():
        for concept in layer_data:
            sumo_to_synsets[concept['sumo_term']].append(concept['synset'])

    # For each SUMO term, find its parent and thus which concepts trigger it
    activation_map = {}  # SUMO term → parent SUMO term
    for term in sumo_to_synsets.keys():
        if term in child_to_parent:
            activation_map[term] = child_to_parent[term]

    for layer_num in range(5):
        layer_data = layers[layer_num]

        # Compute stats
        pos_counts = defaultdict(int)
        sumo_counts = defaultdict(int)
        lexname_counts = defaultdict(int)

        for concept in layer_data:
            pos_counts[concept['pos']] += 1
            sumo_counts[concept['sumo_term']] += 1
            lexname_counts[concept['lexname']] += 1

        # Sample top concepts
        samples = []
        for concept in sorted(layer_data, key=lambda x: -x['frequency'])[:10]:
            samples.append({
                'concept': concept['lemmas'][0],
                'synset': concept['synset'],
                'sumo_term': concept['sumo_term'],
                'sumo_depth': concept['sumo_depth'],
                'frequency': concept['frequency'],
                'definition': concept['definition'][:100] + "..." if len(concept['definition']) > 100 else concept['definition']
            })

        # Build activation info: which parent synsets trigger these concepts
        parent_synsets = set()
        for concept in layer_data:
            sumo_term = concept['sumo_term']
            if sumo_term in activation_map:
                parent_sumo = activation_map[sumo_term]
                # Add all synsets mapped to the parent SUMO term
                parent_synsets.update(sumo_to_synsets.get(parent_sumo, []))

        output_data = {
            'metadata': {
                'layer': layer_num,
                'description': descriptions[layer_num],
                'total_concepts': len(layer_data),
                'pos_distribution': dict(pos_counts),
                'top_sumo_terms': dict(sorted(sumo_counts.items(), key=lambda x: -x[1])[:20]),
                'top_lexical_domains': dict(sorted(lexname_counts.items(), key=lambda x: -x[1])[:10]),
                'samples': samples,
                'activation': {
                    'parent_layer': layer_num - 1 if layer_num > 0 else None,
                    'parent_synsets': sorted(list(parent_synsets))[:100],  # Limit to top 100 for size
                    'activation_model': 'hierarchical' if layer_num > 0 else 'always_active'
                }
            },
            'concepts': layer_data
        }

        output_path = OUTPUT_DIR / f"layer{layer_num}.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"  ✓ Layer {layer_num}: {len(layer_data):6} concepts → {output_path}")


def main():
    print("="*60)
    print("ABSTRACTION-BASED LAYER BUILDER")
    print("With Hierarchical Activation Support")
    print("="*60)

    # Load data
    sumo_depths, child_to_parent = load_sumo_hierarchy()
    freq_dist = compute_word_frequencies()
    concepts = load_all_concepts()

    # Assign layers
    layers = assign_abstraction_layers(concepts, sumo_depths, freq_dist)

    # Save with activation metadata
    save_layers(layers, child_to_parent)

    print("\n" + "="*60)
    print("✓ Complete - Abstraction layers built with hierarchical activation")
    print("="*60)


if __name__ == '__main__':
    main()
