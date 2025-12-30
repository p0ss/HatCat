#!/usr/bin/env python3
"""
Build a natural knowledge map based on information architecture principles.

Layer 0: Knowledge Domains (4-6 concepts)
  - MindsAndAgents: Cognition, agency, social structures
  - CreatedThings: Artifacts, technology, systems
  - PhysicalWorld: Matter, energy, forces
  - LivingThings: Organisms, biology, ecosystems
  - Information: Data, representations, propositions

Layer 1: Major categories per domain (~5-10 each, total ~30-50)
Layer 2: Subcategories maintaining pyramid structure (~200-400)
Layer 3+: Natural expansion via parent relationships
"""

import re
import json
import csv
from pathlib import Path
from collections import defaultdict, deque

SUMO_SOURCE_DIR = Path("data/concept_graph/sumo_source")
OUTPUT_DIR = Path("data/concept_graph")

def parse_kif_files():
    """Parse all KIF files."""
    parent_map = defaultdict(set)
    children_map = defaultdict(set)
    source_map = {}
    all_concepts = set()

    for kif_file in SUMO_SOURCE_DIR.glob("*.kif"):
        with open(kif_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        subclass_pattern = r'\(subclass\s+([A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]+)\)'
        for match in re.finditer(subclass_pattern, content):
            child, parent = match.groups()
            if not child.startswith('?') and not parent.startswith('?'):
                parent_map[child].add(parent)
                children_map[parent].add(child)
                all_concepts.add(child)
                all_concepts.add(parent)

                if child not in source_map:
                    source_map[child] = kif_file.name

    return parent_map, children_map, source_map, all_concepts


def build_natural_knowledge_map(parent_map, children_map, source_map, all_concepts):
    """Build knowledge map using natural domain categorization."""

    print("=" * 80)
    print("NATURAL KNOWLEDGE MAP BUILDER")
    print("=" * 80)

    # Layer 0: Natural knowledge domains
    domains = {
        'MindsAndAgents': set(),
        'CreatedThings': set(),
        'PhysicalWorld': set(),
        'LivingThings': set(),
        'Information': set(),
    }

    # Layer 1: Manually assign major SUMO categories to domains
    # Based on semantic analysis from previous runs

    domain_seeds = {
        'MindsAndAgents': {
            'CognitiveProcess', 'IntentionalProcess', 'AutonomousAgent',
            'Agent', 'Organization', 'SocialInteraction', 'Communication',
            'Perception', 'Emotion', 'Believing', 'Reasoning',
            # Custom safety concepts
            'Deception', 'AlignmentProcess', 'ArtificialAgent',
        },

        'CreatedThings': {
            'Artifact', 'Device', 'Machine', 'ComputerProgram',
            'Tool', 'Weapon', 'Vehicle', 'Building', 'Infrastructure',
            'Technology', 'Game', 'ArtWork',
            # Custom infrastructure
            'DataCenter', 'AIArtifact', 'TransformerModel',
            'AttackPattern', 'CyberOperation',
        },

        'PhysicalWorld': {
            'PhysicalQuantity', 'Motion', 'Region', 'Substance',
            'GeographicArea', 'Quantity', 'TimeInterval',
            'ConstantQuantity', 'FieldOfStudy',
        },

        'LivingThings': {
            'Organism', 'BiologicalProcess', 'BodyPart',
            'AnatomicalStructure', 'Animal', 'Plant',
            'Population', 'Ecosystem',
        },

        'Information': {
            'Proposition', 'Formula', 'Sentence', 'ContentBearingPhysical',
            'LinguisticExpression', 'Text', 'Concept',
            'AbstractEntity', 'SetOrClass', 'Relation',
            'RelationalAttribute', 'InternalAttribute',
        },
    }

    # Build initial domain assignments
    concept_domain = {}

    for domain, seeds in domain_seeds.items():
        for seed in seeds:
            if seed in all_concepts:
                concept_domain[seed] = domain
                domains[domain].add(seed)

    # Propagate domain assignments to children
    def assign_domain_to_descendants(concept, domain):
        """Recursively assign domain to all descendants."""
        for child in children_map.get(concept, []):
            if child not in concept_domain:
                concept_domain[child] = domain
                domains[domain].add(child)
                assign_domain_to_descendants(child, domain)

    for domain, seeds in domain_seeds.items():
        for seed in seeds:
            if seed in all_concepts:
                assign_domain_to_descendants(seed, domain)

    # Assign remaining concepts based on parent domains
    unassigned = all_concepts - set(concept_domain.keys())

    for concept in unassigned:
        parents = parent_map.get(concept, set())
        if parents:
            # Assign to most common parent domain
            parent_domains = [concept_domain.get(p) for p in parents if p in concept_domain]
            if parent_domains:
                domain = max(set(parent_domains), key=parent_domains.count)
                concept_domain[concept] = domain
                domains[domain].add(concept)

    # Show domain distribution
    print(f"\nDomain distribution:")
    for domain, concepts in domains.items():
        print(f"  {domain}: {len(concepts)} concepts")

    # Build layer structure within each domain
    # Layer 0: Domain roots (our 5 domains - these are meta-concepts, not in SUMO)
    # Layer 1: Direct seeds (~30-50 total)
    # Layer 2: Children of seeds maintaining pyramid
    # Layer 3+: Further descendants

    layer_assignment = {}

    # Assign layers within each domain
    for domain, domain_concepts in domains.items():
        # Layer 1: Seed concepts for this domain
        seeds = domain_seeds.get(domain, set()) & all_concepts

        for seed in seeds:
            layer_assignment[seed] = 1

        # BFS from seeds to assign deeper layers
        queue = deque()
        for seed in seeds:
            queue.append((seed, 1))

        visited = set(seeds)

        while queue:
            current, current_layer = queue.popleft()

            for child in children_map.get(current, []):
                if child in domain_concepts and child not in visited:
                    visited.add(child)
                    # Assign to next layer (but cap at layer 4)
                    child_layer = min(current_layer + 1, 4)
                    layer_assignment[child] = child_layer
                    queue.append((child, child_layer))

    # Count layer distribution
    layer_counts = defaultdict(int)
    for layer in layer_assignment.values():
        layer_counts[layer] += 1

    print("\n" + "=" * 80)
    print("LAYER DISTRIBUTION")
    print("=" * 80)

    print(f"\nLayer 0: 5 domains (meta-concepts)")
    for layer_num in range(1, 5):
        count = layer_counts[layer_num]
        pct = 100 * count / len(layer_assignment) if layer_assignment else 0
        print(f"Layer {layer_num}: {count:5} concepts ({pct:5.1f}%)")

    # Show Layer 1 breakdown by domain
    print("\n" + "=" * 80)
    print("LAYER 1 BY DOMAIN")
    print("=" * 80)

    for domain in domains.keys():
        layer1_in_domain = [
            c for c, l in layer_assignment.items()
            if l == 1 and concept_domain.get(c) == domain
        ]
        print(f"\n{domain} ({len(layer1_in_domain)} Layer 1 concepts):")
        for concept in sorted(layer1_in_domain)[:15]:
            child_count = len(children_map.get(concept, []))
            print(f"  {concept} ({child_count} children)")
        if len(layer1_in_domain) > 15:
            print(f"  ... and {len(layer1_in_domain) - 15} more")

    # Save mapping
    output = {
        'domains': {
            domain: sorted(list(concepts))
            for domain, concepts in domains.items()
        },
        'layer_assignment': layer_assignment,
        'concept_domain': concept_domain,
        'layer_distribution': dict(layer_counts),
    }

    output_path = OUTPUT_DIR / "natural_knowledge_map.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Natural knowledge map saved to {output_path}")

    # Generate CSV with domain + layers
    csv_path = OUTPUT_DIR / "natural_knowledge_hierarchy.csv"

    rows = []
    for concept in sorted(layer_assignment.keys()):
        domain = concept_domain.get(concept, 'Unknown')
        layer = layer_assignment.get(concept, 999)

        # Build parent chain
        chain = [concept]
        current = concept
        visited = set()

        while current in parent_map and current not in visited:
            visited.add(current)
            parents = parent_map.get(current, set())
            if not parents:
                break

            # Choose parent in same domain with lowest layer
            same_domain_parents = [
                p for p in parents
                if concept_domain.get(p) == domain
            ]

            if same_domain_parents:
                parent = min(same_domain_parents, key=lambda p: layer_assignment.get(p, 999))
            else:
                parent = min(parents, key=lambda p: layer_assignment.get(p, 999))

            chain.insert(0, parent)
            current = parent

            if len(chain) > 10:
                break

        # Build row: Domain, Layer1, Layer2, Layer3, Layer4
        row = [domain] + [''] * 4

        for i, c in enumerate(chain):
            c_layer = layer_assignment.get(c, 999)
            if 1 <= c_layer <= 4:
                row[c_layer] = c

        rows.append((domain, layer, concept, row))

    # Sort by domain, then layer, then concept name
    rows.sort(key=lambda x: (x[0], x[1], x[2]))

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Domain', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'])

        for domain, layer, concept, row in rows:
            writer.writerow(row)

    print(f"✓ Natural knowledge hierarchy CSV saved to {csv_path}")
    print(f"  Total concepts: {len(rows)}")

    return output


def main():
    parent_map, children_map, source_map, all_concepts = parse_kif_files()
    build_natural_knowledge_map(parent_map, children_map, source_map, all_concepts)


if __name__ == '__main__':
    main()
