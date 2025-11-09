#!/usr/bin/env python3
"""
Build abstraction layers with hierarchical SUMO remapping.

V4 improvements over V3:
- Uses WordNet hypernym chains to find most specific valid SUMO category
- Fixes ~15,500 misplaced synsets that map to overly general categories
- Remaps orphaned synsets (e.g., Agent) to their most specific populated ancestor
- Better layer distribution with fewer shallow orphans

Strategy:
1. Load SUMO hierarchy and compute depths
2. Load WordNet→SUMO mappings (direct from files)
3. For each synset, walk hypernym chain to collect all candidate SUMO terms
4. Choose most specific (deepest layer) SUMO term that exists in populated hierarchy
5. Assign to layers based on final SUMO term depth
"""

import re
import json
from pathlib import Path
from collections import defaultdict
import networkx as nx

import nltk
from nltk.corpus import wordnet as wn, brown

try:
    wn.synsets('test')
    brown.words()
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('brown')

SUMO_DIR = Path("data/concept_graph/sumo_source")
OUTPUT_DIR = Path("data/concept_graph/abstraction_layers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sumo_hierarchy():
    """Parse complete SUMO hierarchy."""
    print("Loading SUMO hierarchy...")
    
    kif_files = [
        SUMO_DIR / "Merge.kif",
        SUMO_DIR / "Mid-level-ontology.kif",
        SUMO_DIR / "emotion.kif",
        SUMO_DIR / "Food.kif",
        SUMO_DIR / "Geography.kif",
        SUMO_DIR / "People.kif",
    ]
    
    child_to_parent = {}
    for kif_file in kif_files:
        if not kif_file.exists():
            continue
        with open(kif_file) as f:
            for line in f:
                if line.startswith(';') or not line.strip():
                    continue
                match = re.match(r'^\(subclass\s+(\S+)\s+(\S+)\)', line.strip())
                if match:
                    child, parent = match.groups()
                    if not child.startswith('?') and not parent.startswith('?'):
                        child_to_parent[child] = parent
    
    dg = nx.DiGraph()
    for child, parent in child_to_parent.items():
        dg.add_edge(child, parent)
    dg = dg.reverse()
    depths = nx.single_source_shortest_path_length(dg, 'Entity')
    
    print(f"  ✓ Loaded {len(depths)} SUMO terms with depths")
    return depths, child_to_parent


def load_wordnet_mappings():
    """Load WordNet 3.0 → SUMO mappings."""
    print("Loading WordNet 3.0 mappings...")
    
    mapping_files = {
        'n': SUMO_DIR / "WordNetMappings30-noun.txt",
        'v': SUMO_DIR / "WordNetMappings30-verb.txt",
        'a': SUMO_DIR / "WordNetMappings30-adj.txt",
        'r': SUMO_DIR / "WordNetMappings30-adv.txt",
    }
    
    synset_to_sumo = {}
    for pos, filepath in mapping_files.items():
        if not filepath.exists():
            continue
        with open(filepath) as f:
            for line in f:
                if line.startswith(';') or not line.strip():
                    continue
                parts = line.split('|')
                if len(parts) < 2:
                    continue
                synset_info = parts[0].strip().split()
                sumo_part = parts[-1].strip()
                if len(synset_info) < 3:
                    continue
                offset = synset_info[0]
                pos_tag = synset_info[2]
                sumo_match = re.search(r'&%(\w+)([=+@\[\]:])$', sumo_part)
                if sumo_match:
                    sumo_term = sumo_match.group(1)
                    relation = sumo_match.group(2)
                    try:
                        offset_int = int(offset)
                        synset = wn.synset_from_pos_and_offset(pos_tag, offset_int)
                        synset_name = synset.name()
                        synset_to_sumo[synset_name] = (sumo_term, relation)
                    except:
                        pass
    
    print(f"  ✓ Loaded {len(synset_to_sumo)} synset→SUMO mappings")
    return synset_to_sumo


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
        freq = freq_dist.get(word, 0)
        if freq > max_freq:
            max_freq = freq
        if '_' in lemma:
            first = lemma.split('_')[0].lower()
            freq = freq_dist.get(first, 0)
            if freq > max_freq:
                max_freq = freq
    return max_freq


def remap_synset_to_best_sumo(synset_name, original_sumo, synset_to_sumo,
                               populated_terms, sumo_to_layer, sumo_depths):
    """
    Walk WordNet hypernym chain to find most specific valid SUMO category.

    Returns: (best_sumo_term, remapped: bool)
    """
    try:
        synset = wn.synset(synset_name)
    except:
        return original_sumo, False

    # Collect all candidate SUMO terms from hypernym chain
    candidates = []

    # Start with direct mapping
    if original_sumo in populated_terms:
        candidates.append((original_sumo, sumo_to_layer[original_sumo], 'direct'))

    # Walk hypernym chain (up to 3 levels deep)
    for hypernym in synset.hypernyms():
        h_name = hypernym.name()
        if h_name in synset_to_sumo:
            h_sumo, h_rel = synset_to_sumo[h_name]
            if h_sumo in populated_terms:
                candidates.append((h_sumo, sumo_to_layer[h_sumo], 'hypernym'))

        # Level 2
        for h2 in hypernym.hypernyms():
            h2_name = h2.name()
            if h2_name in synset_to_sumo:
                h2_sumo, h2_rel = synset_to_sumo[h2_name]
                if h2_sumo in populated_terms:
                    candidates.append((h2_sumo, sumo_to_layer[h2_sumo], 'hypernym2'))

            # Level 3
            for h3 in h2.hypernyms():
                h3_name = h3.name()
                if h3_name in synset_to_sumo:
                    h3_sumo, h3_rel = synset_to_sumo[h3_name]
                    if h3_sumo in populated_terms:
                        candidates.append((h3_sumo, sumo_to_layer[h3_sumo], 'hypernym3'))

    if not candidates:
        # Fallback: use original even if unpopulated
        return original_sumo, False

    # Choose most specific (deepest layer)
    best = max(candidates, key=lambda x: x[1])
    best_sumo, best_layer, source = best

    # Check if we remapped (changed from original)
    remapped = (best_sumo != original_sumo)

    return best_sumo, remapped


def build_layers(sumo_depths, child_to_parent, synset_to_sumo, freq_dist):
    """Build layers with hierarchical remapping."""
    print("\n" + "="*60)
    print("PHASE 1: Build populated SUMO term database")
    print("="*60)

    # Count synsets per SUMO term
    sumo_to_synsets = defaultdict(list)
    for synset_name, (sumo_term, relation) in synset_to_sumo.items():
        sumo_to_synsets[sumo_term].append((synset_name, relation))

    populated_terms = set(sumo_to_synsets.keys())
    print(f"  ✓ Found {len(populated_terms)} SUMO terms with synsets")

    # Assign layer based on depth
    sumo_to_layer = {}
    for term in populated_terms:
        depth = sumo_depths.get(term, 999)

        if depth <= 2:
            layer = 0
        elif depth <= 4:
            layer = 1
        elif depth <= 6:
            layer = 2
        elif depth <= 9:
            layer = 3
        else:
            layer = 4  # Very deep or unmapped

        sumo_to_layer[term] = layer

    print("\n" + "="*60)
    print("PHASE 2: Remap synsets via hypernym chains")
    print("="*60)

    # Remap each synset to best SUMO category
    remapped_count = 0
    improved_count = 0
    synset_to_final_sumo = {}

    for synset_name, (original_sumo, relation) in synset_to_sumo.items():
        best_sumo, was_remapped = remap_synset_to_best_sumo(
            synset_name, original_sumo, synset_to_sumo,
            populated_terms, sumo_to_layer, sumo_depths
        )

        synset_to_final_sumo[synset_name] = (best_sumo, relation, original_sumo)

        if was_remapped:
            remapped_count += 1

            # Check if we improved (moved to deeper layer)
            orig_layer = sumo_to_layer.get(original_sumo, -1)
            best_layer = sumo_to_layer.get(best_sumo, -1)
            if best_layer > orig_layer:
                improved_count += 1

    print(f"  ✓ Remapped {remapped_count} synsets ({100*remapped_count/len(synset_to_sumo):.1f}%)")
    print(f"  ✓ Improved {improved_count} synsets to deeper layers")

    print("\n" + "="*60)
    print("PHASE 3: Build category layers (0-4)")
    print("="*60)

    # Rebuild synset counts with remapped terms
    remapped_sumo_to_synsets = defaultdict(list)
    for synset_name, (final_sumo, relation, orig_sumo) in synset_to_final_sumo.items():
        remapped_sumo_to_synsets[final_sumo].append((synset_name, relation))

    layers = defaultdict(list)
    
    # Build category layers using remapped counts
    for term in populated_terms:
        depth = sumo_depths.get(term, 999)
        layer = sumo_to_layer[term]
        synsets = remapped_sumo_to_synsets[term]  # Use remapped counts

        if len(synsets) == 0:
            # This category became empty after remapping
            continue

        # Find canonical synset (prefer exact match)
        canonical_synset = None
        canonical_synset_obj = None

        for sname, rel in synsets:
            if rel == '=':
                try:
                    canonical_synset = sname
                    canonical_synset_obj = wn.synset(sname)
                    break
                except:
                    pass

        if not canonical_synset:
            for sname, rel in synsets[:1]:
                try:
                    canonical_synset = sname
                    canonical_synset_obj = wn.synset(sname)
                    break
                except:
                    pass

        # Get children SUMO terms (that also have synsets after remapping)
        children = [c for c, p in child_to_parent.items()
                   if p == term and c in remapped_sumo_to_synsets and len(remapped_sumo_to_synsets[c]) > 0]

        concept = {
            'sumo_term': term,
            'sumo_depth': depth,
            'layer': layer,
            'is_category_probe': True,
            'category_children': children,
            'synset_count': len(synsets),
            'synsets': [s for s, r in synsets[:5]],
        }

        if canonical_synset_obj:
            concept.update({
                'canonical_synset': canonical_synset,
                'lemmas': canonical_synset_obj.lemma_names(),
                'pos': canonical_synset_obj.pos(),
                'definition': canonical_synset_obj.definition(),
                'lexname': canonical_synset_obj.lexname(),
            })
        else:
            concept.update({
                'canonical_synset': None,
                'lemmas': [term],
                'pos': None,
                'definition': f"SUMO category: {term}",
                'lexname': None,
            })

        layers[layer].append(concept)

    print("\n" + "="*60)
    print("PHASE 4: Build synset layer (5)")
    print("="*60)

    # Add all synsets to final layer
    final_layer = max(layers.keys()) + 1
    layers[final_layer] = []

    for synset_name, (final_sumo, relation, orig_sumo) in synset_to_final_sumo.items():
        try:
            synset = wn.synset(synset_name)
            concept = {
                'synset': synset_name,
                'lemmas': synset.lemma_names(),
                'pos': synset.pos(),
                'definition': synset.definition(),
                'lexname': synset.lexname(),
                'sumo_term': final_sumo,
                'sumo_depth': sumo_depths.get(final_sumo, 999),
                'sumo_relation': relation,
                'original_sumo_term': orig_sumo if orig_sumo != final_sumo else None,
                'frequency': get_synset_frequency(synset, freq_dist),
                'layer': final_layer,
                'is_category_probe': False,
                'hypernyms': [h.name() for h in synset.hypernyms()],
                'hyponyms': [h.name() for h in synset.hyponyms()],
                'antonyms': list(set(a.name() for lemma in synset.lemmas()
                                    for a in lemma.antonyms())),
            }
            layers[final_layer].append(concept)
        except:
            pass

    # Print distribution
    print("\n  Layer distribution after remapping:")
    for layer_num in sorted(layers.keys()):
        count = len(layers[layer_num])
        if layer_num < final_layer:
            depth_min = min(c['sumo_depth'] for c in layers[layer_num])
            depth_max = max(c['sumo_depth'] for c in layers[layer_num])
            print(f"    Layer {layer_num}: {count:6} SUMO categories (depth {depth_min}-{depth_max})")
        else:
            # Count by parent layer
            by_parent_layer = defaultdict(int)
            for c in layers[layer_num]:
                parent_layer = sumo_to_layer.get(c['sumo_term'], -1)
                by_parent_layer[parent_layer] += 1

            print(f"    Layer {layer_num}: {count:6} WordNet synsets")
            for parent_layer in sorted(by_parent_layer.keys()):
                pcount = by_parent_layer[parent_layer]
                pct = 100 * pcount / count
                print(f"              → {pcount:6} map to Layer {parent_layer} ({pct:5.1f}%)")

    return dict(layers)


def save_layers(layers, child_to_parent):
    """Save layers to JSON."""
    print("\n" + "="*60)
    print("PHASE 5: Save layers to JSON")
    print("="*60)

    final_layer = max(layers.keys())

    descriptions = {
        0: "SUMO depth 0-2: Top-level ontological categories",
        1: "SUMO depth 3-4: Major semantic domains",
        2: "SUMO depth 5-6: Specific categories",
        3: "SUMO depth 7-9: Fine-grained categories",
        4: "SUMO depth 10+ and unmapped (depth 999)",
        5: "WordNet synsets with hierarchical SUMO remapping",
    }
    
    for layer_num in sorted(layers.keys()):
        layer_data = layers[layer_num]

        if layer_num < final_layer:
            samples = []
            for concept in sorted(layer_data, key=lambda x: -x['synset_count'])[:10]:
                samples.append({
                    'sumo_term': concept['sumo_term'],
                    'sumo_depth': concept['sumo_depth'],
                    'synset_count': concept['synset_count'],
                    'category_children_count': len(concept['category_children']),
                    'definition': concept['definition'][:100] + "..." if len(concept['definition']) > 100 else concept['definition']
                })
            
            output_data = {
                'metadata': {
                    'layer': layer_num,
                    'description': descriptions.get(layer_num, f"Layer {layer_num}"),
                    'total_concepts': len(layer_data),
                    'samples': samples,
                },
                'concepts': layer_data
            }
        else:
            # Synset layer
            pos_counts = defaultdict(int)
            sumo_counts = defaultdict(int)
            remapped_count = 0

            for concept in layer_data:
                pos_counts[concept['pos']] += 1
                sumo_counts[concept['sumo_term']] += 1
                if concept.get('original_sumo_term'):
                    remapped_count += 1

            samples = []
            for concept in sorted(layer_data, key=lambda x: -x['frequency'])[:10]:
                samples.append({
                    'concept': concept['lemmas'][0],
                    'synset': concept['synset'],
                    'sumo_term': concept['sumo_term'],
                    'frequency': concept['frequency'],
                    'definition': concept['definition'][:100] + "..." if len(concept['definition']) > 100 else concept['definition']
                })

            output_data = {
                'metadata': {
                    'layer': layer_num,
                    'description': descriptions.get(layer_num, f"Layer {layer_num}"),
                    'total_concepts': len(layer_data),
                    'remapped_count': remapped_count,
                    'pos_distribution': dict(pos_counts),
                    'top_sumo_terms': dict(sorted(sumo_counts.items(), key=lambda x: -x[1])[:20]),
                    'samples': samples,
                },
                'concepts': layer_data
            }
        
        output_path = OUTPUT_DIR / f"layer{layer_num}.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  ✓ Layer {layer_num}: {len(layer_data):6} concepts → {output_path}")


def main():
    print("="*60)
    print("SUMO-WORDNET HIERARCHICAL REMAPPING (V4)")
    print("Uses WordNet hypernym chains for better SUMO assignment")
    print("="*60)

    sumo_depths, child_to_parent = load_sumo_hierarchy()
    synset_to_sumo = load_wordnet_mappings()
    freq_dist = compute_word_frequencies()

    layers = build_layers(sumo_depths, child_to_parent, synset_to_sumo, freq_dist)
    save_layers(layers, child_to_parent)

    print("\n" + "="*60)
    print("✓ Complete - Hierarchical remapping applied")
    print("="*60)


if __name__ == '__main__':
    main()
