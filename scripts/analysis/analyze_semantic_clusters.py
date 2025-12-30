#!/usr/bin/env python3
"""
Analyze semantic clustering of V4 concepts to propose coherent layer structure.

Strategy:
1. Group concepts by their primary parent (not depth)
2. Identify semantic domains (cognitive, physical, social, etc.)
3. Propose layer assignments based on conceptual similarity
4. Ensure pyramid structure (10 -> 100 -> 1000)
"""

import re
import json
from pathlib import Path
from collections import defaultdict, Counter

# Load SUMO source
SUMO_SOURCE_DIR = Path("data/concept_graph/sumo_source")
OUTPUT_DIR = Path("data/concept_graph")

def parse_kif_files():
    """Parse all KIF files to build parent-child maps."""
    parent_map = defaultdict(set)  # child -> {parents}
    children_map = defaultdict(set)  # parent -> {children}
    source_map = {}  # concept -> source file

    for kif_file in SUMO_SOURCE_DIR.glob("*.kif"):
        with open(kif_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        subclass_pattern = r'\(subclass\s+([A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]+)\)'
        for match in re.finditer(subclass_pattern, content):
            child, parent = match.groups()
            if child.startswith('?') or parent.startswith('?'):
                continue
            parent_map[child].add(parent)
            children_map[parent].add(child)

            if child not in source_map:
                source_map[child] = kif_file.name

    return parent_map, children_map, source_map


def find_layer0_children(parent_map, children_map):
    """Find direct children of layer 0 core concepts."""
    layer0_core = {
        'Entity', 'Abstract', 'Physical', 'Attribute',
        'Relation', 'Process', 'Object', 'Continuant', 'Occurrent'
    }

    # Find direct children of each layer 0 concept
    layer0_children = defaultdict(set)

    for concept, parents in parent_map.items():
        for parent in parents:
            if parent in layer0_core:
                layer0_children[parent].add(concept)

    return layer0_children


def cluster_by_semantics(parent_map, children_map, source_map):
    """Group concepts into semantic domains based on parents and source files."""

    # Define semantic domains based on SUMO structure
    domains = {
        'Cognitive': set(),
        'Physical_Objects': set(),
        'Processes': set(),
        'Attributes': set(),
        'Social': set(),
        'Information': set(),
        'Biological': set(),
        'Artificial': set(),
        'Temporal': set(),
        'Spatial': set(),
        'Abstract_Math': set(),
        'Safety_AI': set(),  # Custom concepts
    }

    # Key parent concepts that define domains
    cognitive_roots = {'CognitiveAgent', 'IntentionalProcess', 'Proposition', 'Believing'}
    physical_roots = {'Object', 'Physical', 'SelfConnectedObject', 'CorpuscularObject'}
    process_roots = {'Process', 'IntentionalProcess', 'PhysicalProcess'}
    attribute_roots = {'Attribute', 'InternalAttribute', 'RelationalAttribute'}
    social_roots = {'SocialInteraction', 'Organization', 'Agreement', 'Contest'}
    info_roots = {'Proposition', 'ContentBearingPhysical', 'Sentence', 'Formula'}
    bio_roots = {'Organism', 'OrganicObject', 'AnatomicalStructure', 'BiologicalProcess'}
    artificial_roots = {'Device', 'Artifact', 'Machine', 'ComputerProgram'}
    temporal_roots = {'TimeInterval', 'TimeDuration', 'TimePoint'}
    spatial_roots = {'Region', 'GeographicArea', 'Direction'}
    abstract_roots = {'Abstract', 'Quantity', 'Number', 'SetOrClass'}

    # Safety/AI concepts from custom files
    safety_sources = {
        'AI.kif', 'ai_alignment.kif', 'ai_infrastructure.kif',
        'cyber_security.kif', 'narrative_deception.kif',
        'self_awareness.kif', 'situational_awareness.kif'
    }

    # Classify all concepts
    for concept, parents in parent_map.items():
        source = source_map.get(concept, '')

        # Check safety/AI domain first
        if source in safety_sources:
            domains['Safety_AI'].add(concept)
            continue

        # Check domain roots
        if parents & cognitive_roots:
            domains['Cognitive'].add(concept)
        elif parents & physical_roots:
            domains['Physical_Objects'].add(concept)
        elif parents & process_roots:
            domains['Processes'].add(concept)
        elif parents & attribute_roots:
            domains['Attributes'].add(concept)
        elif parents & social_roots:
            domains['Social'].add(concept)
        elif parents & info_roots:
            domains['Information'].add(concept)
        elif parents & bio_roots:
            domains['Biological'].add(concept)
        elif parents & artificial_roots:
            domains['Artificial'].add(concept)
        elif parents & temporal_roots:
            domains['Temporal'].add(concept)
        elif parents & spatial_roots:
            domains['Spatial'].add(concept)
        elif parents & abstract_roots:
            domains['Abstract_Math'].add(concept)

    return domains


def propose_layer_structure(parent_map, children_map, source_map):
    """Propose semantic layer structure following pyramid principle."""

    print("=" * 80)
    print("SEMANTIC CLUSTERING ANALYSIS")
    print("=" * 80)

    # Layer 0: Core ontological categories (hardcoded)
    layer0 = {
        'Entity', 'Abstract', 'Physical', 'Attribute',
        'Relation', 'Process', 'Object', 'Continuant', 'Occurrent'
    }

    print(f"\nLayer 0 (Core Ontology): {len(layer0)} concepts")
    print(f"  {', '.join(sorted(layer0))}")

    # Find children of layer 0
    layer0_children = find_layer0_children(parent_map, children_map)

    print(f"\n\nLayer 0 Fan-out Analysis:")
    total_direct_children = 0
    for parent in sorted(layer0):
        children = layer0_children.get(parent, set())
        total_direct_children += len(children)
        print(f"  {parent}: {len(children)} direct children")

    # Propose Layer 1: Top-level domain categories (~10-15 concepts)
    # These should be the most important direct children of layer 0
    layer1_candidates = set()
    for parent in layer0:
        children = layer0_children.get(parent, set())
        # Get top children by fan-out
        child_sizes = [(c, len(children_map.get(c, []))) for c in children]
        child_sizes.sort(key=lambda x: x[1], reverse=True)

        # Take top 2-3 from each layer 0 concept
        for child, size in child_sizes[:3]:
            if size > 5:  # Only significant branches
                layer1_candidates.add(child)

    print(f"\n\nLayer 1 Candidates ({len(layer1_candidates)} concepts):")
    layer1_with_sizes = [(c, len(children_map.get(c, []))) for c in layer1_candidates]
    layer1_with_sizes.sort(key=lambda x: x[1], reverse=True)

    for concept, size in layer1_with_sizes[:20]:
        parents = list(parent_map.get(concept, []))
        print(f"  {concept}: {size} children (parents: {', '.join(parents[:2])})")

    # Analyze semantic domains
    print(f"\n\n" + "=" * 80)
    print("SEMANTIC DOMAIN ANALYSIS")
    print("=" * 80)

    domains = cluster_by_semantics(parent_map, children_map, source_map)

    for domain_name, concepts in sorted(domains.items(), key=lambda x: len(x[1]), reverse=True):
        if len(concepts) > 0:
            print(f"\n{domain_name}: {len(concepts)} concepts")
            sample = sorted(list(concepts))[:10]
            print(f"  Sample: {', '.join(sample)}")

    # Source file analysis
    print(f"\n\n" + "=" * 80)
    print("SOURCE FILE DISTRIBUTION")
    print("=" * 80)

    source_counts = Counter(source_map.values())
    print(f"\nTop 15 source files by concept count:")
    for source, count in source_counts.most_common(15):
        print(f"  {source}: {count} concepts")

    # Generate proposed layer mapping
    print(f"\n\n" + "=" * 80)
    print("PROPOSED LAYER STRUCTURE")
    print("=" * 80)

    print("""
Layer 0 (9 concepts): Core ontological categories
  Entity, Abstract, Physical, Attribute, Relation, Process, Object, Continuant, Occurrent

Layer 1 (10-20 concepts): Top-level domains
  Target: Major conceptual branches with significant subtrees
  Criteria:
    - Direct children of Layer 0
    - Fan-out > 10 children
    - Semantically distinct domains

Layer 2 (100-200 concepts): Domain-specific categories
  Target: Coherent semantic groupings within each Layer 1 domain
  Criteria:
    - Children/grandchildren of Layer 1
    - Concepts within same semantic domain
    - Fan-out > 3 children

Layer 3 (1000-2000 concepts): Specific concepts
  Target: Concrete instances of Layer 2 categories
  Criteria:
    - Descendants of Layer 2
    - Semantic similarity within parent domain

Layer 4 (remaining): Leaf concepts and fine-grained specializations
""")

    # Save analysis
    analysis = {
        'layer0': list(layer0),
        'layer1_candidates': [
            {'concept': c, 'children_count': size, 'parents': list(parent_map.get(c, []))}
            for c, size in layer1_with_sizes
        ],
        'semantic_domains': {
            domain: {
                'concept_count': len(concepts),
                'sample_concepts': sorted(list(concepts))[:20]
            }
            for domain, concepts in domains.items()
        },
        'source_distribution': dict(source_counts.most_common(30)),
        'layer0_fanout': {
            parent: len(layer0_children.get(parent, []))
            for parent in sorted(layer0)
        }
    }

    output_path = OUTPUT_DIR / "semantic_cluster_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✓ Analysis saved to {output_path}")

    return analysis


def main():
    parent_map, children_map, source_map = parse_kif_files()

    print(f"Total concepts: {len(set(parent_map.keys()) | set(children_map.keys()))}")
    print(f"Concepts with parents: {len(parent_map)}")
    print(f"Concepts with children: {len(children_map)}")

    analysis = propose_layer_structure(parent_map, children_map, source_map)


if __name__ == '__main__':
    main()
