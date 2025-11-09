#!/usr/bin/env python3
"""
Parse SUMO KIF files and build ontology hierarchy.

SUMO uses Knowledge Interchange Format (KIF), not OWL.
We need to:
1. Download and parse Merge.kif (complete SUMO ontology)
2. Parse subclass relations: (subclass ?X ?Y) means X is a subclass of Y
3. Build hierarchy graph and compute depths
4. Load WordNet mappings
5. Create 5 layers with full semantic coverage
"""

import re
import requests
from pathlib import Path
from collections import defaultdict
import networkx as nx
import json

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SUMO_BASE = "https://raw.githubusercontent.com/ontologyportal/sumo/master"
MERGE_KIF_URL = f"{SUMO_BASE}/Merge.kif"
WORDNET_MAPPING_URLS = [
    f"{SUMO_BASE}/WordNetMappings/WordNetMappings-nouns.txt",
    f"{SUMO_BASE}/WordNetMappings/WordNetMappings-verbs.txt",
    f"{SUMO_BASE}/WordNetMappings/WordNetMappings-adj.txt",
    f"{SUMO_BASE}/WordNetMappings/WordNetMappings-adv.txt",
]

OUTPUT_DIR = Path("data/concept_graph/sumo_hierarchy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# KIF Parser
# ---------------------------------------------------------------------
def download_file(url, timeout=120):
    """Download file from URL."""
    print(f"  Downloading: {url.split('/')[-1]}")
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  ⚠ Failed: {e}")
        return None


def parse_kif_relations(kif_text):
    """
    Parse subclass relations from KIF.
    Format: (subclass ?X ?Y) or (subclass ChildClass ParentClass)
    """
    subclass_pattern = r'\(subclass\s+(\S+)\s+(\S+)\)'

    relations = []
    for match in re.finditer(subclass_pattern, kif_text, re.MULTILINE):
        child = match.group(1)
        parent = match.group(2)

        # Skip if either is a variable (starts with ?)
        if child.startswith('?') or parent.startswith('?'):
            continue

        relations.append((child, parent))

    return relations


def build_hierarchy_graph(relations):
    """Build directed graph from subclass relations."""
    G = nx.DiGraph()

    for child, parent in relations:
        G.add_edge(child, parent)

    # Find roots (nodes with no incoming edges)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]

    # SUMO typically has 'Entity' as the root
    root = next((r for r in roots if 'Entity' in r), roots[0] if roots else None)

    return G, root, roots


def compute_depths(G, root):
    """Compute minimum distance from root for all nodes."""
    try:
        depths = nx.single_source_shortest_path_length(G, root)
    except nx.NetworkXError:
        # If graph is disconnected, compute for each component
        depths = {}
        for node in G.nodes():
            try:
                depths[node] = nx.shortest_path_length(G, node, root)
            except nx.NetworkXNoPath:
                # Node not connected to root - assign max depth
                depths[node] = 999

    return depths


# ---------------------------------------------------------------------
# WordNet Mapping Parser
# ---------------------------------------------------------------------
def parse_wordnet_mapping(text):
    """
    Parse WordNet→SUMO mappings.
    Format examples:
    ; comment lines start with semicolon
    (termFormat EnglishLanguage Happiness "happiness")
    (documentation Happiness EnglishLanguage "...")
    (&%Happiness|00001740-n)  ; synset mappings
    """
    mappings = []

    # Pattern: (&%SUMOTerm|synset_offset-pos)
    pattern = r'\(&%(\w+)\|(\d{8}-[nvasr])\)'

    for match in re.finditer(pattern, text):
        sumo_term = match.group(1)
        synset_id = match.group(2)
        mappings.append((synset_id, sumo_term))

    return mappings


# ---------------------------------------------------------------------
# Layer Assignment
# ---------------------------------------------------------------------
def assign_to_layers(depths, max_depth, num_layers=5):
    """
    Assign each SUMO term to one of 5 layers based on depth.
    Returns: dict {layer_num: [terms]}
    """
    layers = {i: [] for i in range(1, num_layers + 1)}

    for term, depth in depths.items():
        if depth == 999:  # Disconnected
            layer = num_layers
        else:
            # Map depth to layer (1=most abstract, 5=most specific)
            depth_pct = depth / max_depth if max_depth > 0 else 0
            layer = max(1, min(num_layers, int(depth_pct * num_layers) + 1))

        layers[layer].append(term)

    return layers


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("="*60)
    print("SUMO KIF PARSER")
    print("="*60)

    # Download and parse SUMO ontology
    print("\nStep 1: Parsing SUMO ontology...")
    kif_text = download_file(MERGE_KIF_URL)

    if not kif_text:
        print("ERROR: Could not download SUMO KIF file")
        return

    print(f"  Downloaded {len(kif_text)} bytes")

    # Extract subclass relations
    print("  Parsing subclass relations...")
    relations = parse_kif_relations(kif_text)
    print(f"  ✓ Found {len(relations)} subclass relations")

    # Build hierarchy
    print("  Building hierarchy graph...")
    G, root, roots = build_hierarchy_graph(relations)
    print(f"  ✓ Graph has {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"  ✓ Root: {root}")
    print(f"  ✓ All roots: {roots}")

    # Compute depths
    print("  Computing depths from root...")
    depths = compute_depths(G, root)
    max_depth = max(d for d in depths.values() if d < 999)
    print(f"  ✓ Max depth: {max_depth}")

    # Assign to layers
    print("  Assigning to 5 layers...")
    sumo_layers = assign_to_layers(depths, max_depth)

    for layer in range(1, 6):
        print(f"    Layer {layer}: {len(sumo_layers[layer])} terms")

    # Download and parse WordNet mappings
    print("\nStep 2: Parsing WordNet mappings...")
    all_mappings = []

    for url in WORDNET_MAPPING_URLS:
        text = download_file(url)
        if text:
            mappings = parse_wordnet_mapping(text)
            all_mappings.extend(mappings)
            print(f"    {url.split('/')[-1]}: {len(mappings)} mappings")

    print(f"  ✓ Total mappings: {len(all_mappings)}")

    # Build synset→SUMO mapping
    synset_to_sumo = {}
    for synset_id, sumo_term in all_mappings:
        if sumo_term in depths:  # Only include terms in our hierarchy
            synset_to_sumo[synset_id] = sumo_term

    print(f"  ✓ Mapped {len(synset_to_sumo)} synsets to SUMO hierarchy")

    # Save results
    print("\nStep 3: Saving results...")

    # Save SUMO hierarchy
    hierarchy_data = {
        'metadata': {
            'total_terms': len(G.nodes()),
            'total_relations': len(relations),
            'max_depth': max_depth,
            'root': root,
            'all_roots': roots
        },
        'depths': depths,
        'layers': {str(k): v for k, v in sumo_layers.items()}
    }

    hierarchy_path = OUTPUT_DIR / "sumo_hierarchy.json"
    with open(hierarchy_path, 'w') as f:
        json.dump(hierarchy_data, f, indent=2)
    print(f"  ✓ Saved hierarchy to {hierarchy_path}")

    # Save WordNet mappings
    mapping_data = {
        'metadata': {
            'total_mappings': len(synset_to_sumo),
            'pos_distribution': {}
        },
        'synset_to_sumo': synset_to_sumo
    }

    # Count by POS
    pos_counts = defaultdict(int)
    for synset_id in synset_to_sumo.keys():
        pos = synset_id[-1]  # Last character is POS
        pos_counts[pos] += 1
    mapping_data['metadata']['pos_distribution'] = dict(pos_counts)

    mapping_path = OUTPUT_DIR / "wordnet_sumo_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    print(f"  ✓ Saved mappings to {mapping_path}")

    # Show statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)

    print("\nSUMO Layer Distribution:")
    for layer in range(1, 6):
        count = len(sumo_layers[layer])
        pct = 100 * count / len(G.nodes())

        # Sample terms
        samples = sorted(sumo_layers[layer])[:5]
        samples_str = ", ".join(samples)

        print(f"  Layer {layer}: {count:5} terms ({pct:5.1f}%)")
        print(f"           Samples: {samples_str}")

    print("\nWordNet→SUMO Mapping:")
    pos_names = {'n': 'Noun', 'v': 'Verb', 'a': 'Adj', 's': 'AdjSat', 'r': 'Adv'}
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        print(f"  {pos_names.get(pos, pos):8} {count:6} synsets")

    print("\n" + "="*60)
    print("✓ Complete")
    print("="*60)


if __name__ == '__main__':
    main()
