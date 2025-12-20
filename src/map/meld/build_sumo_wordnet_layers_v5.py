#!/usr/bin/env python3
"""
Build abstraction layers with hierarchical SUMO remapping + WordNet hyponym intermediates.

V5 improvements over V4:
- V4: Hierarchical remapping via WordNet hypernym chains
- V5: Adds WordNet hyponym-based intermediate layers for large fan-out categories
- Extracts natural groupings from WordNet structure for categories like:
  * SubjectiveAssessmentAttribute (8k+ synsets) → positive/negative/neutral clusters
  * FloweringPlant (5k+ synsets) → tree/shrub/herb clusters
  * Device (2k+ synsets) → instrument/machine/tool clusters

Strategy:
1. Load SUMO hierarchy + domain ontologies (Food, Geography, People)
2. Hierarchical remapping via WordNet hypernym chains (from V4)
3. **NEW**: Extract WordNet hyponym clusters for large categories (>1000 synsets)
4. Create intermediate pseudo-SUMO layers using representative synsets
5. Final structure: 0-4 (SUMO), 5-6 (WordNet intermediates), 7 (all synsets)
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
        SUMO_DIR / "AI.kif",
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
    """Load WordNet 3.0 → SUMO mappings + AI expansion."""
    print("Loading WordNet 3.0 mappings...")

    mapping_files = {
        'n': SUMO_DIR / "WordNetMappings30-noun.txt",
        'v': SUMO_DIR / "WordNetMappings30-verb.txt",
        'a': SUMO_DIR / "WordNetMappings30-adj.txt",
        'r': SUMO_DIR / "WordNetMappings30-adv.txt",
        'ai': SUMO_DIR / "WordNetMappings30-AI-expansion.txt",  # AI.kif synset expansion
    }
    
    # Support multi-mapping: one synset can map to multiple SUMO categories
    # This is critical for AI expansion (e.g., reasoning.n.01 maps to both Reasoning AND ArtificialIntelligence)
    synset_to_sumo = {}  # Primary mapping (for backward compatibility)
    synset_to_all_sumo = defaultdict(list)  # All mappings including AI expansion

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

                        # Keep first mapping as primary (for backward compatibility with existing code)
                        if synset_name not in synset_to_sumo:
                            synset_to_sumo[synset_name] = (sumo_term, relation)

                        # Track all mappings (including AI expansion duplicates)
                        synset_to_all_sumo[synset_name].append((sumo_term, relation))
                    except:
                        pass

    ai_expansion_count = sum(1 for mappings in synset_to_all_sumo.values() if len(mappings) > 1)
    print(f"  ✓ Loaded {len(synset_to_sumo)} primary synset→SUMO mappings")
    print(f"  ✓ {ai_expansion_count} synsets with AI expansion multi-mappings")

    return synset_to_sumo, synset_to_all_sumo


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
            'is_category_lens': True,
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
                'is_category_lens': False,
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


def extract_hyponym_clusters(parent_sumo_term, synsets_in_category,
                             min_cluster_size=100, min_anchor_hyponyms=30):
    """
    Extract WordNet hyponym clusters from a large SUMO category.

    Strategy:
    1. Build hyponym graph for all synsets in this category
    2. Find "anchor synsets" - synsets with many hyponyms within the category
    3. Cluster synsets under their closest anchor
    4. Name clusters using anchor synset lemmas

    Returns: List of clusters, each with {name, anchor_synset, synsets[]}
    """
    # Build hyponym graph within this category
    synset_set = set(synsets_in_category)
    hyponym_counts = defaultdict(int)
    synset_to_anchors = defaultdict(list)  # synset -> list of potential anchors

    # Count hyponyms for each synset (only within category)
    for synset_name in synsets_in_category:
        try:
            synset = wn.synset(synset_name)

            # Walk hypernym chain to find anchors
            hypernyms_to_check = [(synset, 0)]  # (synset, distance)
            seen = {synset_name}

            while hypernyms_to_check:
                current, dist = hypernyms_to_check.pop(0)

                if dist > 3:  # Max 3 levels up
                    continue

                for hypernym in current.hypernyms():
                    h_name = hypernym.name()

                    if h_name in seen:
                        continue
                    seen.add(h_name)

                    # If this hypernym is in our category, it's a potential anchor
                    if h_name in synset_set:
                        synset_to_anchors[synset_name].append((h_name, dist + 1))
                        hyponym_counts[h_name] += 1

                    hypernyms_to_check.append((hypernym, dist + 1))

        except Exception as e:
            continue

    # Find anchor synsets (high hyponym count)
    anchors = []
    for synset_name, count in hyponym_counts.items():
        if count >= min_anchor_hyponyms:
            try:
                synset = wn.synset(synset_name)
                anchors.append({
                    'synset': synset_name,
                    'lemmas': synset.lemma_names(),
                    'hyponym_count': count,
                    'definition': synset.definition()
                })
            except:
                pass

    # Sort anchors by hyponym count
    anchors.sort(key=lambda x: -x['hyponym_count'])

    # Cluster synsets under anchors
    clusters = []
    assigned = set()

    for anchor_info in anchors:
        anchor_synset = anchor_info['synset']
        cluster_synsets = {anchor_synset}

        # Assign synsets that have this as closest anchor
        for synset_name, anchor_list in synset_to_anchors.items():
            if synset_name in assigned:
                continue

            # Sort by distance, prefer this anchor
            anchor_list_sorted = sorted(anchor_list, key=lambda x: x[1])

            if anchor_list_sorted and anchor_list_sorted[0][0] == anchor_synset:
                cluster_synsets.add(synset_name)
                assigned.add(synset_name)

        # Only keep cluster if large enough
        if len(cluster_synsets) >= min_cluster_size:
            # Generate cluster name from anchor lemmas
            primary_lemma = anchor_info['lemmas'][0].replace('_', ' ').title().replace(' ', '')
            cluster_name = f"{parent_sumo_term}_{primary_lemma}"

            clusters.append({
                'name': cluster_name,
                'display_name': primary_lemma,
                'anchor_synset': anchor_synset,
                'anchor_lemmas': anchor_info['lemmas'],
                'anchor_definition': anchor_info['definition'],
                'synset_count': len(cluster_synsets),
                'synsets': list(cluster_synsets)
            })
            assigned.update(cluster_synsets)

    # Create "Other" cluster for remaining synsets
    remaining = [s for s in synsets_in_category if s not in assigned]
    if len(remaining) >= min_cluster_size:
        clusters.append({
            'name': f"{parent_sumo_term}_Other",
            'display_name': 'Other',
            'anchor_synset': None,
            'anchor_lemmas': [],
            'anchor_definition': f"Other {parent_sumo_term} concepts not clustered under major categories",
            'synset_count': len(remaining),
            'synsets': remaining
        })
    elif remaining:
        # Assign to largest cluster if too small
        if clusters:
            clusters[0]['synsets'].extend(remaining)
            clusters[0]['synset_count'] += len(remaining)

    return clusters


def build_layers_v5(sumo_depths, child_to_parent, synset_to_sumo, synset_to_all_sumo, freq_dist):
    """Build layers with hierarchical remapping + hyponym intermediates."""
    print("\n" + "="*60)
    print("PHASE 1: Build populated SUMO term database")
    print("="*60)

    # Count synsets per SUMO term - USE ALL MAPPINGS to populate AI categories
    sumo_to_synsets = defaultdict(list)
    for synset_name, mappings in synset_to_all_sumo.items():
        for sumo_term, relation in mappings:
            sumo_to_synsets[sumo_term].append((synset_name, relation))

    # Start with SUMO terms that have synsets
    populated_terms = set(sumo_to_synsets.keys())

    # Recursively add category-only terms (no direct synsets but have populated descendants)
    # This ensures AI.kif categories appear even without WordNet mappings
    # We need multiple passes to propagate upward through the hierarchy
    changed = True
    while changed:
        changed = False
        for term in list(sumo_depths.keys()):
            if term not in populated_terms:
                # Check if this term has any populated children
                children = [c for c, p in child_to_parent.items() if p == term]
                if any(c in populated_terms for c in children):
                    populated_terms.add(term)
                    changed = True

    print(f"  ✓ Found {len(populated_terms)} SUMO terms (with synsets or populated children)")

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
    print("PHASE 2.5: Stitch empty SUMO children using lexical patterns")
    print("="*60)

    # Special handling for SubjectiveAssessmentAttribute - use sentiment patterns
    saa_synsets = [sn for sn, (sumo, rel, orig) in synset_to_final_sumo.items()
                   if sumo == 'SubjectiveAssessmentAttribute']

    print(f"  Found {len(saa_synsets)} synsets in SubjectiveAssessmentAttribute")

    # Define sentiment patterns for SubjectiveAssessmentAttribute children
    sentiment_patterns = {
        'SubjectiveStrongPositiveAttribute': [
            'excellent', 'outstanding', 'superb', 'superior', 'exceptional',
            'wonderful', 'magnificent', 'brilliant', 'perfect', 'flawless',
            'terrific', 'phenomenal', 'spectacular', 'marvelous', 'fabulous',
            'fantastic', 'incredible', 'extraordinary', 'remarkable', 'supreme'
        ],
        'SubjectiveWeakPositiveAttribute': [
            'good', 'fine', 'nice', 'decent', 'acceptable', 'adequate',
            'satisfactory', 'fair', 'reasonable', 'tolerable', 'passable',
            'pleasant', 'agreeable', 'favorable', 'positive', 'better',
            'proper', 'suitable', 'appropriate', 'right', 'correct'
        ],
        'SubjectiveStrongNegativeAttribute': [
            'terrible', 'horrible', 'awful', 'dreadful', 'abysmal', 'atrocious',
            'appalling', 'horrendous', 'dire', 'ghastly', 'hideous',
            'deplorable', 'disastrous', 'catastrophic', 'abominable', 'vile',
            'wretched', 'miserable', 'pathetic', 'inferior'
        ],
        'SubjectiveWeakNegativeAttribute': [
            'bad', 'poor', 'weak', 'inadequate', 'unsatisfactory', 'substandard',
            'mediocre', 'deficient', 'lacking', 'wanting', 'faulty',
            'imperfect', 'flawed', 'worse', 'wrong', 'incorrect',
            'inappropriate', 'unfavorable', 'negative', 'questionable'
        ],
    }

    # Remap SubjectiveAssessmentAttribute synsets to children
    stitched_count = 0
    for synset_name in saa_synsets:
        try:
            synset = wn.synset(synset_name)
            lemmas_str = ' '.join(synset.lemma_names()).lower().replace('_', ' ')
            definition = synset.definition().lower()

            # Check patterns
            best_match = None
            best_score = 0

            for child_sumo, patterns in sentiment_patterns.items():
                score = 0
                for pattern in patterns:
                    if pattern in lemmas_str:
                        score += 10  # Strong match in lemma
                    if pattern in definition:
                        score += 1   # Weak match in definition

                if score > best_score:
                    best_score = score
                    best_match = child_sumo

            # If we found a match, remap to child
            if best_match and best_score >= 5:  # Threshold for stitching
                old_sumo, relation, orig_sumo = synset_to_final_sumo[synset_name]
                synset_to_final_sumo[synset_name] = (best_match, relation, orig_sumo)
                stitched_count += 1

        except Exception as e:
            continue

    print(f"  ✓ Stitched {stitched_count} synsets to SubjectiveAssessmentAttribute children")

    # Add newly populated SUMO terms to populated_terms
    for synset_name, (final_sumo, relation, orig_sumo) in synset_to_final_sumo.items():
        if final_sumo not in populated_terms:
            populated_terms.add(final_sumo)
            # Assign layer based on depth
            depth = sumo_depths.get(final_sumo, 999)
            if depth <= 2:
                layer = 0
            elif depth <= 4:
                layer = 1
            elif depth <= 6:
                layer = 2
            elif depth <= 9:
                layer = 3
            else:
                layer = 4
            sumo_to_layer[final_sumo] = layer

    print("\n" + "="*60)
    print("PHASE 3: Extract hyponym clusters for large categories")
    print("="*60)

    # Rebuild synset counts with remapped terms
    remapped_sumo_to_synsets = defaultdict(list)
    for synset_name, (final_sumo, relation, orig_sumo) in synset_to_final_sumo.items():
        remapped_sumo_to_synsets[final_sumo].append((synset_name, relation))

    # IMPORTANT: Also add AI expansion multi-mappings so AI categories get their synsets
    # synset_to_all_sumo contains ALL mappings (primary + AI expansion)
    for synset_name, mappings in synset_to_all_sumo.items():
        # Skip the primary mapping (already added above)
        # Add only secondary mappings (AI expansion)
        if len(mappings) > 1:
            for i, (sumo_term, relation) in enumerate(mappings):
                # Skip first mapping (primary) if it's the same as final_sumo
                if i == 0 and synset_name in synset_to_final_sumo:
                    final_sumo = synset_to_final_sumo[synset_name][0]
                    if sumo_term == final_sumo:
                        continue
                # Add this mapping
                remapped_sumo_to_synsets[sumo_term].append((synset_name, relation))

    # Identify large categories for subdivision
    LARGE_CATEGORY_THRESHOLD = 1000
    large_categories = []

    for term in populated_terms:
        synsets = remapped_sumo_to_synsets[term]
        if len(synsets) >= LARGE_CATEGORY_THRESHOLD:
            children = [c for c, p in child_to_parent.items()
                       if p == term and c in remapped_sumo_to_synsets]

            large_categories.append({
                'sumo_term': term,
                'layer': sumo_to_layer[term],
                'synset_count': len(synsets),
                'child_count': len(children),
                'synsets': [s for s, r in synsets]
            })

    large_categories.sort(key=lambda x: -x['synset_count'])

    print(f"  Found {len(large_categories)} large categories (>{LARGE_CATEGORY_THRESHOLD} synsets)")
    for cat in large_categories[:10]:
        print(f"    {cat['sumo_term']:<40} {cat['synset_count']:>6} synsets, {cat['child_count']:>3} children")

    # Extract hyponym clusters for each large category
    pseudo_sumo_categories = {}  # pseudo_sumo_name -> metadata
    synset_to_pseudo_sumo = {}   # synset_name -> pseudo_sumo_name

    for cat in large_categories:
        print(f"\n  Extracting clusters for {cat['sumo_term']} ({cat['synset_count']} synsets)...")

        clusters = extract_hyponym_clusters(
            cat['sumo_term'],
            cat['synsets'],
            min_cluster_size=100,
            min_anchor_hyponyms=30
        )

        print(f"    → Found {len(clusters)} clusters:")
        for cluster in clusters:
            print(f"      {cluster['display_name']:<30} {cluster['synset_count']:>5} synsets")

            # Register pseudo-SUMO category
            pseudo_sumo_categories[cluster['name']] = {
                'sumo_term': cluster['name'],
                'display_name': cluster['display_name'],
                'parent_sumo': cat['sumo_term'],
                'parent_layer': cat['layer'],
                'anchor_synset': cluster['anchor_synset'],
                'anchor_lemmas': cluster['anchor_lemmas'],
                'anchor_definition': cluster['anchor_definition'],
                'synset_count': cluster['synset_count'],
                'synsets': cluster['synsets'],
                'is_pseudo_sumo': True
            }

            # Map synsets to pseudo-SUMO
            for synset_name in cluster['synsets']:
                synset_to_pseudo_sumo[synset_name] = cluster['name']

    print(f"\n  ✓ Created {len(pseudo_sumo_categories)} intermediate pseudo-SUMO categories")

    print("\n" + "="*60)
    print("PHASE 4: Build category layers (0-5)")
    print("="*60)

    layers = defaultdict(list)

    # Build SUMO category layers (0-4) with remapped counts
    for term in populated_terms:
        depth = sumo_depths.get(term, 999)
        layer = sumo_to_layer[term]

        # Get synsets that stayed at this SUMO level (not remapped to pseudo-SUMO)
        all_synsets = remapped_sumo_to_synsets[term]
        direct_synsets = [(s, r) for s, r in all_synsets if s not in synset_to_pseudo_sumo]

        # Find canonical synset
        canonical_synset = None
        canonical_synset_obj = None

        for sname, rel in all_synsets:
            if rel == '=':
                try:
                    canonical_synset = sname
                    canonical_synset_obj = wn.synset(sname)
                    break
                except:
                    pass

        if not canonical_synset and all_synsets:
            for sname, rel in all_synsets[:1]:
                try:
                    canonical_synset = sname
                    canonical_synset_obj = wn.synset(sname)
                    break
                except:
                    pass

        # Get children SUMO terms + pseudo-SUMO children
        sumo_children = [c for c, p in child_to_parent.items()
                        if p == term and c in remapped_sumo_to_synsets and len(remapped_sumo_to_synsets[c]) > 0]
        pseudo_children = [ps for ps, meta in pseudo_sumo_categories.items()
                          if meta['parent_sumo'] == term]
        all_children = sumo_children + pseudo_children

        concept = {
            'sumo_term': term,
            'sumo_depth': depth,
            'layer': layer,
            'is_category_lens': True,
            'is_pseudo_sumo': False,
            'category_children': all_children,
            'synset_count': len(all_synsets),
            'direct_synset_count': len(direct_synsets),
            'synsets': [s for s, r in all_synsets[:5]],
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

    # Add pseudo-SUMO categories as layer 5
    pseudo_layer = 5
    for pseudo_name, meta in pseudo_sumo_categories.items():
        # Get canonical synset (anchor if available)
        canonical_synset_obj = None
        if meta['anchor_synset']:
            try:
                canonical_synset_obj = wn.synset(meta['anchor_synset'])
            except:
                pass

        concept = {
            'sumo_term': pseudo_name,
            'display_name': meta['display_name'],
            'parent_sumo': meta['parent_sumo'],
            'parent_layer': meta['parent_layer'],
            'sumo_depth': -1,  # Pseudo-SUMO has no depth
            'layer': pseudo_layer,
            'is_category_lens': True,
            'is_pseudo_sumo': True,
            'category_children': [],  # No children (synsets are in next layer)
            'synset_count': meta['synset_count'],
            'synsets': meta['synsets'][:5],
        }

        if canonical_synset_obj:
            concept.update({
                'canonical_synset': meta['anchor_synset'],
                'lemmas': canonical_synset_obj.lemma_names(),
                'pos': canonical_synset_obj.pos(),
                'definition': meta['anchor_definition'],
                'lexname': canonical_synset_obj.lexname(),
            })
        else:
            concept.update({
                'canonical_synset': None,
                'lemmas': [meta['display_name']],
                'pos': None,
                'definition': meta['anchor_definition'],
                'lexname': None,
            })

        layers[pseudo_layer].append(concept)

    print("\n" + "="*60)
    print("PHASE 5: Build synset layer (6)")
    print("="*60)

    # Add all synsets to final layer
    final_layer = 6
    layers[final_layer] = []

    for synset_name, (final_sumo, relation, orig_sumo) in synset_to_final_sumo.items():
        try:
            synset = wn.synset(synset_name)

            # Check if this synset has a pseudo-SUMO parent
            parent_category = synset_to_pseudo_sumo.get(synset_name, final_sumo)
            parent_is_pseudo = synset_name in synset_to_pseudo_sumo

            concept = {
                'synset': synset_name,
                'lemmas': synset.lemma_names(),
                'pos': synset.pos(),
                'definition': synset.definition(),
                'lexname': synset.lexname(),
                'sumo_term': final_sumo,
                'parent_category': parent_category,
                'parent_is_pseudo_sumo': parent_is_pseudo,
                'sumo_depth': sumo_depths.get(final_sumo, 999),
                'sumo_relation': relation,
                'original_sumo_term': orig_sumo if orig_sumo != final_sumo else None,
                'frequency': get_synset_frequency(synset, freq_dist),
                'layer': final_layer,
                'is_category_lens': False,
                'hypernyms': [h.name() for h in synset.hypernyms()],
                'hyponyms': [h.name() for h in synset.hyponyms()],
                'antonyms': list(set(a.name() for lemma in synset.lemmas()
                                    for a in lemma.antonyms())),
            }
            layers[final_layer].append(concept)
        except:
            pass

    # Print distribution
    print("\n  Layer distribution after V5 subdivision:")
    for layer_num in sorted(layers.keys()):
        count = len(layers[layer_num])
        if layer_num < pseudo_layer:
            depth_min = min(c['sumo_depth'] for c in layers[layer_num])
            depth_max = max(c['sumo_depth'] for c in layers[layer_num])
            print(f"    Layer {layer_num}: {count:6} SUMO categories (depth {depth_min}-{depth_max})")
        elif layer_num == pseudo_layer:
            print(f"    Layer {layer_num}: {count:6} Pseudo-SUMO hyponym clusters")
            # Show parent distribution
            parent_dist = defaultdict(int)
            for c in layers[layer_num]:
                parent_dist[c['parent_sumo']] += 1
            print(f"              → Subdividing {len(parent_dist)} large categories")
        else:
            # Count by parent type
            pseudo_count = sum(1 for c in layers[layer_num] if c.get('parent_is_pseudo_sumo'))
            direct_count = count - pseudo_count
            print(f"    Layer {layer_num}: {count:6} WordNet synsets")
            print(f"              → {pseudo_count:6} via pseudo-SUMO intermediates ({100*pseudo_count/count:.1f}%)")
            print(f"              → {direct_count:6} direct SUMO mappings ({100*direct_count/count:.1f}%)")

    return dict(layers)


def save_layers_v5(layers, child_to_parent):
    """Save V5 layers to JSON."""
    print("\n" + "="*60)
    print("PHASE 6: Save layers to JSON")
    print("="*60)

    descriptions = {
        0: "SUMO depth 0-2: Top-level ontological categories",
        1: "SUMO depth 3-4: Major semantic domains",
        2: "SUMO depth 5-6: Specific categories",
        3: "SUMO depth 7-9: Fine-grained categories",
        4: "SUMO depth 10+ and unmapped (depth 999)",
        5: "WordNet hyponym intermediate clusters (pseudo-SUMO)",
        6: "WordNet synsets with hierarchical SUMO remapping + hyponym intermediates",
    }

    for layer_num in sorted(layers.keys()):
        layer_data = layers[layer_num]

        if layer_num <= 4:
            # SUMO category layers
            samples = []
            for concept in sorted(layer_data, key=lambda x: -x['synset_count'])[:10]:
                samples.append({
                    'sumo_term': concept['sumo_term'],
                    'sumo_depth': concept['sumo_depth'],
                    'synset_count': concept['synset_count'],
                    'direct_synset_count': concept.get('direct_synset_count', concept['synset_count']),
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
        elif layer_num == 5:
            # Pseudo-SUMO layer
            samples = []
            for concept in sorted(layer_data, key=lambda x: -x['synset_count'])[:15]:
                samples.append({
                    'pseudo_sumo_term': concept['sumo_term'],
                    'display_name': concept['display_name'],
                    'parent_sumo': concept['parent_sumo'],
                    'synset_count': concept['synset_count'],
                    'definition': concept['definition'][:100] + "..." if len(concept['definition']) > 100 else concept['definition']
                })

            output_data = {
                'metadata': {
                    'layer': layer_num,
                    'description': descriptions[layer_num],
                    'total_concepts': len(layer_data),
                    'samples': samples,
                },
                'concepts': layer_data
            }
        else:
            # Synset layer
            pos_counts = defaultdict(int)
            sumo_counts = defaultdict(int)
            pseudo_sumo_counts = defaultdict(int)
            remapped_count = 0

            for concept in layer_data:
                pos_counts[concept['pos']] += 1
                sumo_counts[concept['sumo_term']] += 1
                if concept.get('parent_is_pseudo_sumo'):
                    pseudo_sumo_counts[concept['parent_category']] += 1
                if concept.get('original_sumo_term'):
                    remapped_count += 1

            samples = []
            for concept in sorted(layer_data, key=lambda x: -x['frequency'])[:10]:
                samples.append({
                    'concept': concept['lemmas'][0],
                    'synset': concept['synset'],
                    'parent': concept['parent_category'],
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
                    'pseudo_sumo_count': sum(pseudo_sumo_counts.values()),
                    'pos_distribution': dict(pos_counts),
                    'top_sumo_terms': dict(sorted(sumo_counts.items(), key=lambda x: -x[1])[:20]),
                    'top_pseudo_sumo_terms': dict(sorted(pseudo_sumo_counts.items(), key=lambda x: -x[1])[:20]),
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
    print("SUMO-WORDNET HIERARCHICAL REMAPPING (V5)")
    print("V4 + WordNet hyponym intermediate layers")
    print("="*60)

    sumo_depths, child_to_parent = load_sumo_hierarchy()
    synset_to_sumo, synset_to_all_sumo = load_wordnet_mappings()
    freq_dist = compute_word_frequencies()

    layers = build_layers_v5(sumo_depths, child_to_parent, synset_to_sumo, synset_to_all_sumo, freq_dist)
    save_layers_v5(layers, child_to_parent)

    print("\n" + "="*60)
    print("✓ Complete - V5 with hyponym intermediates")
    print("="*60)


if __name__ == '__main__':
    main()
