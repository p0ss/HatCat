#!/usr/bin/env python3
"""
Build abstraction layers with SUMO terms that have actual synsets.

Strategy:
- Only create category probes for SUMO terms with synsets
- Assign to layers based on depth, but skip empty categories
- Creates natural sunburst hierarchy: Entity → Physical → Human → Man → Woman
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


def build_layers(sumo_depths, child_to_parent, synset_to_sumo, freq_dist):
    """Build layers with only populated SUMO categories."""
    print("Building abstraction layers (SUMO terms with synsets only)...")
    
    # Count synsets per SUMO term
    sumo_to_synsets = defaultdict(list)
    for synset_name, (sumo_term, relation) in synset_to_sumo.items():
        sumo_to_synsets[sumo_term].append((synset_name, relation))
    
    # Only consider SUMO terms that have synsets
    populated_terms = set(sumo_to_synsets.keys())
    print(f"  Found {len(populated_terms)} SUMO terms with synsets")
    
    # Assign layers based on depth (flexible boundaries)
    layers = defaultdict(list)
    
    for term in populated_terms:
        depth = sumo_depths.get(term, 999)
        synsets = sumo_to_synsets[term]
        
        # Assign layer based on depth
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
        
        # Find canonical synset
        canonical_synset = None
        canonical_synset_obj = None
        
        # Prefer exact match (=)
        for sname, rel in synsets:
            if rel == '=':
                try:
                    canonical_synset = sname
                    canonical_synset_obj = wn.synset(sname)
                    break
                except:
                    pass
        
        # Otherwise use first
        if not canonical_synset:
            for sname, rel in synsets[:1]:
                try:
                    canonical_synset = sname
                    canonical_synset_obj = wn.synset(sname)
                    break
                except:
                    pass
        
        # Get children SUMO terms (that also have synsets)
        children = [c for c, p in child_to_parent.items() 
                   if p == term and c in populated_terms]
        
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
    
    # Add all synsets to final layer
    final_layer = max(layers.keys()) + 1
    layers[final_layer] = []
    
    for synset_name, (sumo_term, relation) in synset_to_sumo.items():
        try:
            synset = wn.synset(synset_name)
            concept = {
                'synset': synset_name,
                'lemmas': synset.lemma_names(),
                'pos': synset.pos(),
                'definition': synset.definition(),
                'lexname': synset.lexname(),
                'sumo_term': sumo_term,
                'sumo_depth': sumo_depths.get(sumo_term, 999),
                'sumo_relation': relation,
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
    print("\n  Layer distribution:")
    for layer_num in sorted(layers.keys()):
        count = len(layers[layer_num])
        if layer_num < final_layer:
            depth_min = min(c['sumo_depth'] for c in layers[layer_num])
            depth_max = max(c['sumo_depth'] for c in layers[layer_num])
            print(f"    Layer {layer_num}: {count:6} SUMO categories (depth {depth_min}-{depth_max})")
        else:
            sumo_terms = set(c['sumo_term'] for c in layers[layer_num])
            print(f"    Layer {layer_num}: {count:6} WordNet synsets ({len(sumo_terms):4} SUMO terms)")
    
    return dict(layers)


def save_layers(layers, child_to_parent):
    """Save layers to JSON."""
    print("\nSaving abstraction layers...")
    
    final_layer = max(layers.keys())
    
    for layer_num in sorted(layers.keys()):
        layer_data = layers[layer_num]
        
        if layer_num < final_layer:
            # SUMO category layer
            description = f"SUMO categories at depth {min(c['sumo_depth'] for c in layer_data)}-{max(c['sumo_depth'] for c in layer_data)}"
            
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
                    'description': description,
                    'total_concepts': len(layer_data),
                    'samples': samples,
                },
                'concepts': layer_data
            }
        else:
            # Synset layer
            pos_counts = defaultdict(int)
            sumo_counts = defaultdict(int)
            
            for concept in layer_data:
                pos_counts[concept['pos']] += 1
                sumo_counts[concept['sumo_term']] += 1
            
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
                    'description': 'WordNet synsets organized by SUMO hierarchy',
                    'total_concepts': len(layer_data),
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
    print("SUMO-WORDNET SUNBURST HIERARCHY BUILDER")
    print("Only SUMO terms with actual synset mappings")
    print("="*60)
    
    sumo_depths, child_to_parent = load_sumo_hierarchy()
    synset_to_sumo = load_wordnet_mappings()
    freq_dist = compute_word_frequencies()
    
    layers = build_layers(sumo_depths, child_to_parent, synset_to_sumo, freq_dist)
    save_layers(layers, child_to_parent)
    
    print("\n" + "="*60)
    print("✓ Complete - Sunburst hierarchy with populated categories")
    print("="*60)


if __name__ == '__main__':
    main()
