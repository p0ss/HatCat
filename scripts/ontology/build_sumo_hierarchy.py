#!/usr/bin/env python3
"""
Build 5-layer SUMO-WordNet concept hierarchy.

1. Load SUMO ontology (OWL) and compute depth for each class
2. Load WordNet→SUMO mappings
3. Organize into 5 layers by ontological depth
4. Add WordNet relations (synonyms, antonyms, hypernyms, etc.)
5. Output layer1.json → layer5.json

No LLM inference - just structural organization.
"""

import json
import requests
import math
from pathlib import Path
from collections import defaultdict
from io import BytesIO

import nltk
from nltk.corpus import wordnet as wn

try:
    from rdflib import Graph, URIRef
    from rdflib.namespace import RDFS
    import networkx as nx
except ImportError:
    print("ERROR: Missing dependencies. Install with:")
    print("  pip install rdflib networkx")
    exit(1)

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
OUTPUT_DIR = Path("data/concept_graph/sumo_layers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMO_OWL_URL = "https://raw.githubusercontent.com/ontologyportal/sumo/master/Merge.owl"
WORDNET_MAPPING_URL = "https://raw.githubusercontent.com/ontologyportal/sumo/master/WordNetMappings30-noun.txt"

# Try multiple mapping files
MAPPING_URLS = [
    "https://raw.githubusercontent.com/ontologyportal/sumo/master/WordNetMappings/WordNetMappings30-noun.txt",
    "https://raw.githubusercontent.com/ontologyportal/sumo/master/WordNetMappings/WordNetMappings30-verb.txt",
    "https://raw.githubusercontent.com/ontologyportal/sumo/master/WordNetMappings/WordNetMappings30-adj.txt",
]


# ---------------------------------------------------------------------
# Load SUMO ontology and compute depths
# ---------------------------------------------------------------------
def download_text(url: str, timeout=120) -> str:
    """Download text from URL."""
    print(f"  Downloading: {url}")
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  ⚠ Failed to download {url}: {e}")
        return None


def load_sumo_depths() -> tuple[dict[str, int], int]:
    """
    Load SUMO OWL file and compute depth of each class in the hierarchy.
    Returns: (depths dict, max_depth)
    """
    print("\nLoading SUMO ontology...")

    # Try Merge.owl first (combined ontology)
    owl_data = download_text(SUMO_OWL_URL)

    if not owl_data:
        print("ERROR: Could not download SUMO ontology")
        return {}, 0

    print("  Parsing OWL...")
    g = Graph()
    g.parse(data=owl_data, format="xml")

    # Build directed graph of subclass relations
    print("  Building hierarchy graph...")
    dg = nx.DiGraph()

    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            # Extract class name from URI
            s_name = str(s).split('#')[-1] if '#' in str(s) else str(s).split('/')[-1]
            o_name = str(o).split('#')[-1] if '#' in str(o) else str(o).split('/')[-1]
            dg.add_edge(s_name, o_name)

    print(f"  Found {len(dg.nodes)} SUMO classes with {len(dg.edges)} subclass relations")

    # Find root (Entity is typically the root in SUMO)
    roots = [n for n in dg.nodes if dg.in_degree(n) == 0]
    print(f"  Found {len(roots)} root nodes: {roots[:5]}")

    root = next((r for r in roots if 'Entity' in r), roots[0] if roots else None)

    if not root:
        print("ERROR: No root found in SUMO hierarchy")
        return {}, 0

    print(f"  Using root: {root}")

    # Compute depths from root
    depths = nx.single_source_shortest_path_length(dg, root)
    max_depth = max(depths.values()) if depths else 0

    print(f"  ✓ Computed depths for {len(depths)} classes (max depth: {max_depth})")

    return depths, max_depth


# ---------------------------------------------------------------------
# Load WordNet→SUMO mappings
# ---------------------------------------------------------------------
def load_wordnet_mappings() -> dict[str, str]:
    """
    Load WordNet synset → SUMO term mappings.
    Returns dict: {synset_name: sumo_term}
    """
    print("\nLoading WordNet→SUMO mappings...")

    mapping = {}

    for url in MAPPING_URLS:
        text = download_text(url)
        if not text:
            continue

        pos_name = url.split('-')[-1].replace('.txt', '')
        count = 0

        for line in text.splitlines():
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith(';'):
                continue

            # Format: (synset_offset POS) SUMO_Term [mapping_type]
            # Example: (00001740-n Entity 1)
            parts = line.split()
            if len(parts) < 2:
                continue

            # Parse synset identifier
            synset_id = parts[0].strip('()')
            sumo_term = parts[1] if len(parts) > 1 else None

            if not sumo_term:
                continue

            # Remove mapping type markers (=, +, @, etc.)
            sumo_term = sumo_term.lstrip('=+@&')

            # Convert to WordNet synset name format
            try:
                # synset_id format: 00001740-n
                if '-' in synset_id:
                    offset, pos = synset_id.split('-')
                    synset = wn.synset_from_pos_and_offset(pos, int(offset))
                    mapping[synset.name()] = sumo_term
                    count += 1
            except Exception:
                continue

        print(f"  Loaded {count} mappings from {pos_name}")

    print(f"  ✓ Total mappings: {len(mapping)}")

    return mapping


# ---------------------------------------------------------------------
# Build WordNet relations
# ---------------------------------------------------------------------
def get_wordnet_relations(synset):
    """Get all WordNet relations for a synset."""

    relations = {
        'lemmas': synset.lemma_names(),
        'definition': synset.definition(),
        'examples': synset.examples(),
        'pos': synset.pos(),

        # Semantic relations
        'hypernyms': [h.name() for h in synset.hypernyms()],
        'hyponyms': [h.name() for h in synset.hyponyms()],
        'member_holonyms': [h.name() for h in synset.member_holonyms()],
        'part_holonyms': [h.name() for h in synset.part_holonyms()],
        'substance_holonyms': [h.name() for h in synset.substance_holonyms()],
        'member_meronyms': [m.name() for m in synset.member_meronyms()],
        'part_meronyms': [m.name() for m in synset.part_meronyms()],
        'substance_meronyms': [m.name() for m in synset.substance_meronyms()],
        'attributes': [a.name() for a in synset.attributes()],
        'similar_tos': [s.name() for s in synset.similar_tos()],
        'also_sees': [a.name() for a in synset.also_sees()],
        'verb_groups': [v.name() for v in synset.verb_groups()],
        'entailments': [e.name() for e in synset.entailments()],
        'causes': [c.name() for c in synset.causes()],
    }

    # Get antonyms from lemmas
    antonyms = []
    for lemma in synset.lemmas():
        for ant in lemma.antonyms():
            antonyms.append(ant.synset().name())
    relations['antonyms'] = list(set(antonyms))

    return relations


# ---------------------------------------------------------------------
# Build 5-layer hierarchy
# ---------------------------------------------------------------------
def build_layers(depths: dict[str, int], max_depth: int, mapping: dict[str, str]):
    """
    Organize concepts into 5 layers based on SUMO depth.

    Layer 1: Top-level domains (depth 0-20% of max)
    Layer 2: Subdomains (depth 20-40%)
    Layer 3: Concept families (depth 40-60%)
    Layer 4: Specific concepts (depth 60-80%)
    Layer 5: Fine-grained concepts (depth 80-100%)
    """
    print("\nBuilding 5-layer hierarchy...")

    layers = {1: [], 2: [], 3: [], 4: [], 5: []}

    # Track SUMO terms to layer assignment
    sumo_stats = defaultdict(lambda: {'count': 0, 'depth': None, 'layer': None})

    for synset_name, sumo_term in mapping.items():
        try:
            synset = wn.synset(synset_name)
        except Exception:
            continue

        # Find depth of SUMO term
        depth = depths.get(sumo_term, max_depth)

        # Assign to layer (1-5) based on depth percentile
        if max_depth > 0:
            depth_pct = depth / max_depth
            layer = max(1, min(5, int(math.ceil(depth_pct * 5))))
        else:
            layer = 5

        # Get WordNet relations
        relations = get_wordnet_relations(synset)

        entry = {
            'synset': synset_name,
            'sumo_term': sumo_term,
            'sumo_depth': depth,
            'layer': layer,
            **relations
        }

        layers[layer].append(entry)

        sumo_stats[sumo_term]['count'] += 1
        sumo_stats[sumo_term]['depth'] = depth
        sumo_stats[sumo_term]['layer'] = layer

    # Print statistics
    print("\nLayer Distribution:")
    for layer in range(1, 6):
        count = len(layers[layer])
        pct = 100 * count / sum(len(layers[i]) for i in range(1, 6))
        print(f"  Layer {layer}: {count:5} concepts ({pct:5.1f}%)")

    return layers, sumo_stats


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("="*60)
    print("SUMO-WORDNET HIERARCHY BUILDER")
    print("="*60)

    # Load SUMO
    depths, max_depth = load_sumo_depths()

    if not depths:
        print("\nERROR: Failed to load SUMO ontology")
        return

    # Load mappings
    mapping = load_wordnet_mappings()

    if not mapping:
        print("\nERROR: Failed to load WordNet mappings")
        return

    # Build layers
    layers, sumo_stats = build_layers(depths, max_depth, mapping)

    # Save each layer
    print("\nSaving layers...")
    for layer_num in range(1, 6):
        output_path = OUTPUT_DIR / f"layer{layer_num}.json"

        layer_data = {
            'metadata': {
                'layer': layer_num,
                'total_concepts': len(layers[layer_num]),
                'description': [
                    'Top-level domains',
                    'Subdomains',
                    'Concept families',
                    'Specific concepts',
                    'Fine-grained concepts'
                ][layer_num - 1]
            },
            'concepts': layers[layer_num]
        }

        with open(output_path, 'w') as f:
            json.dump(layer_data, f, indent=2)

        print(f"  ✓ Saved {len(layers[layer_num])} concepts to {output_path}")

    # Save SUMO statistics
    sumo_path = OUTPUT_DIR / "sumo_stats.json"
    with open(sumo_path, 'w') as f:
        json.dump(dict(sumo_stats), f, indent=2)
    print(f"  ✓ Saved SUMO statistics to {sumo_path}")

    # Show top SUMO terms per layer
    print("\n" + "="*60)
    print("TOP SUMO TERMS PER LAYER")
    print("="*60)

    for layer in range(1, 6):
        layer_terms = defaultdict(int)
        for entry in layers[layer]:
            layer_terms[entry['sumo_term']] += 1

        print(f"\nLayer {layer}:")
        for term, count in sorted(layer_terms.items(), key=lambda x: -x[1])[:10]:
            depth = sumo_stats[term]['depth']
            print(f"  {term:30} {count:4} synsets (depth={depth})")

    print("\n" + "="*60)
    print("✓ Complete")
    print("="*60)


if __name__ == '__main__':
    main()
