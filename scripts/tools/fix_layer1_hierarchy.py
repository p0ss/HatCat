#!/usr/bin/env python3
"""
Fix Layer 1 hierarchy by removing concepts with Layer 1 parents.

Manual curation based on semantic analysis:
- Remove concepts that are children of other Layer 1 concepts
- Fix domain assignments (e.g., AttackPattern should be Information, not CreatedThings)
- Ensure Layer 1 represents true semantic roots
"""

import re
import json
from pathlib import Path
from collections import defaultdict, deque

SUMO_SOURCE_DIR = Path("data/concept_graph/sumo_source")
OUTPUT_DIR = Path("data/concept_graph")

def parse_kif_files():
    """Parse all KIF files."""
    parent_map = defaultdict(set)
    children_map = defaultdict(set)
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

    return parent_map, children_map, all_concepts


def build_fixed_natural_knowledge_map(parent_map, children_map, all_concepts):
    """Build fixed natural knowledge map with corrected Layer 1."""

    print("=" * 80)
    print("FIXING NATURAL KNOWLEDGE MAP - LAYER 1 CORRECTIONS")
    print("=" * 80)

    # Load original map
    with open(OUTPUT_DIR / "natural_knowledge_map.json") as f:
        original_map = json.load(f)

    # Revised Layer 1 seeds - removing concepts with Layer 1 parents
    domain_seeds = {
        'MindsAndAgents': {
            # Keep roots only
            'AutonomousAgent',      # Root for agents (includes Organism, Organization)
            'IntentionalProcess',   # Root for purposeful processes
            'Believing',            # Root for epistemic states
            'Perception',           # Root for sensory processes
            'Reasoning',            # Root for inference processes
            # Remove: Agent (child of CognitiveAgent), CognitiveProcess (child of IntentionalProcess),
            # SocialInteraction (child of IntentionalProcess), Communication/Deception (children of SocialInteraction)
            # Remove: AlignmentProcess (child of Process), Organization (child of AutonomousAgent)
            # Remove: ArtificialAgent (will handle separately - maybe move to CreatedThings?)
        },

        'CreatedThings': {
            # Keep roots only
            'Artifact',             # Root for human-made objects
            'Building',             # Major category of structures
            'ComputerProgram',      # Root for software (not child of Artifact in SUMO)
            'Game',                 # Root for games/sports
            'Vehicle',              # Major transportation category
            # Remove: Device (child of Artifact), Weapon/Machine (children of Device)
            # Remove: AIArtifact (child of Artifact), ArtWork (child of Artifact)
            # Remove: DataCenter (child of Facility), Tool (child of HarnessComponent)
            # Remove: TransformerModel (child of NeuralNetworkModel)
            # Note: AttackPattern should move to Information domain (it's an Attribute, not physical)
            # Note: CyberOperation is a Process, should be in MindsAndAgents
        },

        'PhysicalWorld': {
            # Keep roots only
            'Motion',               # Root for movement/change
            'Region',               # Root for spatial areas
            'Substance',            # Root for materials
            'Quantity',             # Root for measurable properties
            'TimeInterval',         # Root for temporal spans
            # Remove: GeographicArea (child of Region), PhysicalQuantity/ConstantQuantity (children of Quantity)
            # Remove: FieldOfStudy (child of Proposition - should be in Information)
        },

        'LivingThings': {
            # Keep roots only
            'Organism',             # Root for living things (includes Animal, Plant)
            'AnatomicalStructure',  # Root for body structures
            'BiologicalProcess',    # Root for life processes
            # Remove: Animal, Plant (children of Organism)
            # Remove: BodyPart (child of AnatomicalStructure)
            # Remove: Ecosystem (child of GeographicArea)
        },

        'Information': {
            # Keep roots only
            'Proposition',          # Root for abstract propositions
            'ContentBearingPhysical', # Root for information carriers
            'RelationalAttribute',  # Root for relations
            'InternalAttribute',    # Root for intrinsic properties
            'Concept',              # Root for abstract concepts
            'SetOrClass',           # Root for collections/types
            'AbstractEntity',       # Root for abstract entities
            # Add: AttackPattern (it's an Attribute - behavioral pattern, not physical thing)
            'AttackPattern',        # Behavioral patterns (moved from CreatedThings)
            # Remove: Relation (only 10 synsets, less important than RelationalAttribute)
            # Remove: LinguisticExpression (child of ContentBearingPhysical)
            # Remove: Sentence/Formula (children of LinguisticExpression)
            # Remove: Text (child of Artifact and LinguisticExpression)
        },
    }

    # Special handling for ArtificialAgent
    # User suggested: "Personally i probably would have put artificial agent into the LivingThings category"
    # Add to LivingThings as it's parallel to Organism (non-biological agents)
    domain_seeds['LivingThings'].add('ArtificialAgent')

    # Build domain assignments
    concept_domain = {}
    for domain, seeds in domain_seeds.items():
        for seed in seeds:
            if seed in all_concepts:
                concept_domain[seed] = domain

    # Propagate domain assignments to descendants
    def assign_domain_to_descendants(concept, domain):
        """Recursively assign domain to all descendants."""
        for child in children_map.get(concept, []):
            if child not in concept_domain:
                concept_domain[child] = domain
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
            parent_domains = [concept_domain.get(p) for p in parents if p in concept_domain]
            if parent_domains:
                domain = max(set(parent_domains), key=parent_domains.count)
                concept_domain[concept] = domain

    # Build domains dict
    domains = {
        'MindsAndAgents': set(),
        'CreatedThings': set(),
        'PhysicalWorld': set(),
        'LivingThings': set(),
        'Information': set(),
    }
    for concept, domain in concept_domain.items():
        domains[domain].add(concept)

    # Assign layers using BFS from Layer 1 seeds
    layer_assignment = {}

    # Layer 1: All seeds
    for domain, seeds in domain_seeds.items():
        for seed in seeds:
            if seed in all_concepts:
                layer_assignment[seed] = 1

    # BFS from Layer 1 to assign deeper layers
    queue = deque()
    for concept in layer_assignment.keys():
        queue.append((concept, 1))

    visited = set(layer_assignment.keys())

    while queue:
        current, current_layer = queue.popleft()

        for child in children_map.get(current, []):
            if child in visited:
                continue

            visited.add(child)
            child_layer = min(current_layer + 1, 4)
            layer_assignment[child] = child_layer
            queue.append((child, child_layer))

    # Handle any remaining concepts
    for concept in all_concepts:
        if concept not in layer_assignment:
            layer_assignment[concept] = 4

    # Count distribution
    layer_counts = defaultdict(int)
    for layer in layer_assignment.values():
        layer_counts[layer] += 1

    print("\n" + "=" * 80)
    print("FIXED LAYER DISTRIBUTION")
    print("=" * 80)

    print(f"\nLayer 0: 5 domains (meta-concepts)")
    for layer_num in range(1, 5):
        count = layer_counts[layer_num]
        pct = 100 * count / len(layer_assignment) if layer_assignment else 0
        print(f"Layer {layer_num}: {count:5} concepts ({pct:5.1f}%)")

    # Show Layer 1 breakdown by domain
    print("\n" + "=" * 80)
    print("FIXED LAYER 1 BY DOMAIN")
    print("=" * 80)

    for domain in sorted(domains.keys()):
        layer1_in_domain = [
            c for c, l in layer_assignment.items()
            if l == 1 and concept_domain.get(c) == domain
        ]
        print(f"\n{domain} ({len(layer1_in_domain)} Layer 1 concepts):")
        for concept in sorted(layer1_in_domain):
            child_count = len(children_map.get(concept, []))
            parents = parent_map.get(concept, set())
            print(f"  {concept:30s} ({child_count:3d} children) [parents: {', '.join(sorted(parents)[:2])}]")

    # Verify no Layer 1 concepts have Layer 1 parents
    print("\n" + "=" * 80)
    print("VERIFICATION: Layer 1 Parent-Child Conflicts")
    print("=" * 80)

    layer1_concepts = [c for c, l in layer_assignment.items() if l == 1]
    conflicts = []

    for concept in layer1_concepts:
        parents = parent_map.get(concept, set())
        layer1_parents = parents & set(layer1_concepts)
        if layer1_parents:
            conflicts.append({
                'child': concept,
                'layer1_parents': layer1_parents
            })

    if conflicts:
        print("\nWARNING: Still have conflicts:")
        for conflict in conflicts:
            print(f"  {conflict['child']} has Layer 1 parents: {conflict['layer1_parents']}")
    else:
        print("\n✓ No conflicts! All Layer 1 concepts are true roots.")

    # Save fixed mapping
    output = {
        'domains': {
            domain: sorted(list(concepts))
            for domain, concepts in domains.items()
        },
        'layer_assignment': layer_assignment,
        'concept_domain': concept_domain,
        'layer_distribution': dict(layer_counts),
    }

    output_path = OUTPUT_DIR / "natural_knowledge_map_v2.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Fixed natural knowledge map saved to {output_path}")

    return output


def main():
    parent_map, children_map, all_concepts = parse_kif_files()
    build_fixed_natural_knowledge_map(parent_map, children_map, all_concepts)


if __name__ == '__main__':
    main()
