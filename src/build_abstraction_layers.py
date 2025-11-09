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


def build_category_probes(sumo_depths, child_to_parent, all_concepts):
    """
    Build Layer 0 category probes from SUMO ontology categories.

    Strategy:
    - Find SUMO terms at depth 1-3 (all) + depth 4 (top ~80 by synset count)
    - Map each to canonical WordNet synset by trying term name as synset
    - Create category-level probes (not individual synsets)
    - Target: ~100 probes total
    """
    print("Building Layer 0 category probes...")

    # Build reverse mapping: parent → children SUMO terms
    parent_to_children = defaultdict(list)
    for child, parent in child_to_parent.items():
        parent_to_children[parent].append(child)

    # Count synsets per SUMO term to prioritize depth-4 terms
    sumo_counts = defaultdict(int)
    for concept in all_concepts:
        sumo_counts[concept['sumo_term']] += 1

    # Find all depth 1-3 terms
    depth_1_3_terms = [(term, depth) for term, depth in sumo_depths.items()
                       if depth <= 3]

    # Find depth-4 terms with >= 100 synsets (top ~80 terms)
    depth_4_terms = [(term, depth) for term, depth in sumo_depths.items()
                     if depth == 4 and sumo_counts[term] >= 100]

    # Combine and sort by depth
    candidate_terms = depth_1_3_terms + depth_4_terms
    candidate_terms.sort(key=lambda x: (x[1], -sumo_counts.get(x[0], 0)))  # Sort by depth, then by synset count

    print(f"  Found {len(depth_1_3_terms)} depth 1-3 terms + {len(depth_4_terms)} high-coverage depth-4 terms = {len(candidate_terms)} candidates")

    category_probes = []
    skipped = []

    for sumo_term, depth in candidate_terms:
        # Try to find WordNet synset by term name
        # First try exact lowercase noun match
        try_names = [
            f"{sumo_term.lower()}.n.01",
            f"{sumo_term.lower()}.n.02",
            f"{sumo_term.lower()}.v.01",
        ]

        # Also try with underscore variants
        if len(sumo_term) > 1:
            # Try splitting camelCase: "IntentionalProcess" → "intentional_process"
            import re
            snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', sumo_term).lower()
            try_names.extend([
                f"{snake_case}.n.01",
                f"{snake_case}.n.02",
                f"{snake_case}.v.01",
            ])

        synset = None
        synset_name = None

        for name in try_names:
            try:
                synset = wn.synset(name)
                synset_name = name
                break
            except:
                pass

        if synset is None:
            skipped.append((sumo_term, depth))
            continue

        # Get children SUMO terms
        children = parent_to_children.get(sumo_term, [])

        probe = {
            'lemmas': synset.lemma_names(),
            'synset': synset_name,
            'pos': synset.pos(),
            'definition': synset.definition(),
            'lexname': synset.lexname(),
            'sumo_term': sumo_term,
            'sumo_depth': depth,
            'layer': 0,
            'is_category_probe': True,
            'category_children': children,
            'frequency': 0  # Category probes don't have frequency
        }

        category_probes.append(probe)

    print(f"  ✓ Created {len(category_probes)} category probes")
    print(f"  ⚠ Skipped {len(skipped)} terms without WordNet matches: {[t for t,d in skipped[:10]]}{' ...' if len(skipped) > 10 else ''}")

    return category_probes


def assign_abstraction_layers(concepts, sumo_depths, freq_dist, category_probes, child_to_parent):
    """
    Assign synset-level concepts to layers with order-of-magnitude scaling.

    Strategy:
    - Layer 0: ~100 SUMO category probes (depth 1-4)
    - Layer 1: ~1,000 most frequent synsets from SUMO depth 4-5 categories
    - Layer 2: ~10,000 remaining SUMO depth 4-5 + all depth 6-7
    - Layer 3+: Remaining concepts
    """
    print("Assigning abstraction layers with order-of-magnitude scaling...")

    # Build SUMO term→concept mapping
    sumo_to_concepts = defaultdict(list)
    for concept in concepts:
        sumo_to_concepts[concept['sumo_term']].append(concept)

    # Compute frequency for each concept
    for concept in concepts:
        try:
            synset = wn.synset(concept['synset'])
            concept['frequency'] = get_synset_frequency(synset, freq_dist)
        except:
            concept['frequency'] = 0

    layers = {0: category_probes}
    assigned_synsets = set(probe['synset'] for probe in category_probes)

    # Layer 1: Top ~1,000 frequent concepts from SUMO category_children (depth 4-5)
    print(f"  Building Layer 1: Top ~1,000 concepts from SUMO children...")
    category_children_terms = set()
    for probe in category_probes:
        category_children_terms.update(probe.get('category_children', []))

    # Collect all candidate concepts from category_children
    layer1_candidates = []
    for sumo_term in category_children_terms:
        sumo_depth = sumo_depths.get(sumo_term, 10)
        if sumo_depth <= 5:  # Only depth 4-5 for Layer 1
            for concept in sumo_to_concepts.get(sumo_term, []):
                if concept['synset'] not in assigned_synsets:
                    layer1_candidates.append(concept)

    # Sort by frequency and take top ~1,000
    layer1_candidates.sort(key=lambda x: -x['frequency'])
    layers[1] = layer1_candidates[:1000]

    for concept in layers[1]:
        concept['layer'] = 1
        assigned_synsets.add(concept['synset'])

    print(f"    Assigned {len(layers[1])} concepts to Layer 1")

    # Layer 2: Remaining depth 4-5 + all depth 6-7 (~10,000 target)
    print(f"  Building Layer 2: Remaining SUMO depth 4-5 + depth 6-7...")
    layer2_candidates = []

    # Add remaining depth 4-5 concepts
    for concept in layer1_candidates[1000:]:
        if concept['synset'] not in assigned_synsets:
            layer2_candidates.append(concept)

    # Add all depth 6-7 concepts
    for sumo_term in category_children_terms:
        sumo_depth = sumo_depths.get(sumo_term, 10)
        if 6 <= sumo_depth <= 7:
            for concept in sumo_to_concepts.get(sumo_term, []):
                if concept['synset'] not in assigned_synsets:
                    layer2_candidates.append(concept)

    # Also include depth 6-7 from grandchildren
    for child_term in category_children_terms:
        if child_term in child_to_parent:
            # Find children of this term
            for sumo_term, parent in child_to_parent.items():
                if parent == child_term:
                    sumo_depth = sumo_depths.get(sumo_term, 10)
                    if 6 <= sumo_depth <= 7:
                        for concept in sumo_to_concepts.get(sumo_term, []):
                            if concept['synset'] not in assigned_synsets:
                                layer2_candidates.append(concept)

    # Sort by frequency and take up to ~12,000 (allow some flex)
    layer2_candidates.sort(key=lambda x: -x['frequency'])
    layers[2] = layer2_candidates[:12000]

    for concept in layers[2]:
        concept['layer'] = 2
        assigned_synsets.add(concept['synset'])

    print(f"    Assigned {len(layers[2])} concepts to Layer 2")

    # Layer 3: All remaining concepts
    print(f"  Assigning remaining concepts to Layer 3...")
    layers[3] = []
    for concept in concepts:
        if concept['synset'] not in assigned_synsets:
            concept['layer'] = 3
            layers[3].append(concept)
            assigned_synsets.add(concept['synset'])

    print(f"    Assigned {len(layers[3])} concepts to Layer 3")

    # Print distribution
    print("\n  Layer distribution:")
    for layer_num in sorted(layers.keys()):
        count = len(layers[layer_num])
        if count > 0:
            if layer_num == 0:
                # Category probes
                print(f"    Layer {layer_num}: {count:6} category probes (SUMO depth 1-4)")
            else:
                # Synset-level probes
                avg_depth = sum(sumo_depths.get(c['sumo_term'], 10) for c in layers[layer_num]) / count
                avg_freq = sum(c['frequency'] for c in layers[layer_num]) / count
                unique_sumo = len(set(c['sumo_term'] for c in layers[layer_num]))
                print(f"    Layer {layer_num}: {count:6} concepts ({unique_sumo:3} SUMO terms, avg depth: {avg_depth:.1f}, avg freq: {avg_freq:.0f})")

    return layers


def save_layers(layers, child_to_parent):
    """Save abstraction layers to JSON with hierarchical activation metadata."""
    print("\nSaving abstraction layers...")

    # Generate descriptions for 4-layer hierarchy
    descriptions = {
        0: "Top-level ontological categories (proprioception baseline - always active)",
        1: "High-frequency concepts from SUMO depth 4-5 (~1,000 concepts)",
        2: "Medium-frequency concepts from SUMO depth 4-7 (~10,000 concepts)",
        3: "Remaining concepts (rare/technical, ~70,000+ concepts)",
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

    for layer_num in sorted(layers.keys()):
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
        if layer_num == 0:
            # Category probes - show all or first 20
            for concept in layer_data[:20]:
                samples.append({
                    'concept': concept['lemmas'][0],
                    'synset': concept['synset'],
                    'sumo_term': concept['sumo_term'],
                    'sumo_depth': concept['sumo_depth'],
                    'is_category_probe': concept.get('is_category_probe', False),
                    'category_children_count': len(concept.get('category_children', [])),
                    'definition': concept['definition'][:100] + "..." if len(concept['definition']) > 100 else concept['definition']
                })
        else:
            # Synset-level probes - show by frequency
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
    print("With Category-Level Probes + Hierarchical Activation")
    print("="*60)

    # Load data
    sumo_depths, child_to_parent = load_sumo_hierarchy()
    freq_dist = compute_word_frequencies()
    concepts = load_all_concepts()

    # Build Layer 0 category probes (needs concepts for synset counting)
    category_probes = build_category_probes(sumo_depths, child_to_parent, concepts)

    # Assign synset-level concepts to layers 1-4
    layers = assign_abstraction_layers(concepts, sumo_depths, freq_dist, category_probes, child_to_parent)

    # Save with activation metadata
    save_layers(layers, child_to_parent)

    print("\n" + "="*60)
    print("✓ Complete - Abstraction layers built with category probes")
    print("="*60)


if __name__ == '__main__':
    main()
