#!/usr/bin/env python3
"""
Generate complete semantic layer mapping for all 7,830 concepts.

Strategy:
1. Layer 0: Keep 9 SUMO core concepts
2. Layer 1: ~15-20 major semantic domains (based on fan-out + semantic coherence)
3. Layer 2: Domain-specific categories (~100-200 concepts)
4. Layer 3+: Specific concepts and leaves

Key principle: Group by SEMANTIC SIMILARITY, not depth.
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


def assign_layers_semantically(parent_map, children_map, source_map, all_concepts):
    """Assign layers based on semantic domains, not depth."""

    print("=" * 80)
    print("SEMANTIC LAYER ASSIGNMENT")
    print("=" * 80)

    # Layer 0: Core ontological categories (hardcoded)
    layer0 = {
        'Entity', 'Abstract', 'Physical', 'Attribute',
        'Relation', 'Process', 'Object', 'Continuant', 'Occurrent'
    }

    # Layer 1: Major semantic domains
    # Criteria: Direct children of Layer 0 with fan-out >= 10
    layer1_candidates = set()
    for parent in layer0:
        children = children_map.get(parent, set())
        for child in children:
            child_fanout = len(children_map.get(child, []))
            if child_fanout >= 10:
                layer1_candidates.add(child)

    # Manual additions for important domains with lower fan-out
    layer1_manual = {
        'AutonomousAgent',  # Cognitive agents (11 children)
        'Motion',  # Physical movement (30 children)
        'CognitiveProcess',  # Thinking/reasoning (49 children)
    }
    layer1_candidates |= layer1_manual

    # Remove concepts that are themselves in layer 0
    layer1 = layer1_candidates - layer0

    print(f"\nLayer 0: {len(layer0)} concepts")
    print(f"Layer 1 candidates: {len(layer1)} concepts\n")

    # Show Layer 1 with fan-out
    layer1_with_fanout = [
        (c, len(children_map.get(c, [])))
        for c in layer1
    ]
    layer1_with_fanout.sort(key=lambda x: x[1], reverse=True)

    print("Layer 1 concepts (top 20 by fan-out):")
    for concept, fanout in layer1_with_fanout[:20]:
        parents = list(parent_map.get(concept, []))
        print(f"  {concept}: {fanout} children (parents: {', '.join(parents[:2])})")

    # Build full layer assignment
    concept_layers = {}

    # Assign Layer 0
    for concept in layer0:
        concept_layers[concept] = 0

    # Assign Layer 1
    for concept in layer1:
        concept_layers[concept] = 1

    # Assign remaining layers using BFS from Layer 1
    # Strategy: Traverse down from each Layer 1 concept, assign based on distance
    queue = deque()
    for concept in layer1:
        queue.append((concept, 1))  # (concept, current_layer)

    visited = set(layer0) | set(layer1)

    while queue:
        current, current_layer = queue.popleft()

        for child in children_map.get(current, []):
            if child in visited:
                # Already assigned, skip
                continue

            visited.add(child)

            # Assign child to next layer (but cap at layer 4)
            child_layer = min(current_layer + 1, 4)
            concept_layers[child] = child_layer

            # Add to queue for further traversal
            queue.append((child, child_layer))

    # Handle any remaining concepts (no path from layer 1)
    # These go to layer 4 by default
    for concept in all_concepts:
        if concept not in concept_layers:
            concept_layers[concept] = 4

    # Count distribution
    layer_counts = defaultdict(int)
    for layer in concept_layers.values():
        layer_counts[layer] += 1

    print("\n" + "=" * 80)
    print("LAYER DISTRIBUTION")
    print("=" * 80)

    for layer_num in range(5):
        count = layer_counts[layer_num]
        pct = 100 * count / len(concept_layers)
        print(f"  Layer {layer_num}: {count:5} concepts ({pct:5.1f}%)")

    # Analyze semantic coherence within layers
    print("\n" + "=" * 80)
    print("LAYER 2 SAMPLE (Domain-specific categories)")
    print("=" * 80)

    layer2_concepts = [c for c, l in concept_layers.items() if l == 2]
    layer2_by_parent = defaultdict(list)

    for concept in layer2_concepts:
        parents = parent_map.get(concept, set()) & layer1
        for parent in parents:
            layer2_by_parent[parent].append(concept)

    print("\nLayer 2 concepts grouped by Layer 1 parent:\n")
    for parent in sorted(layer2_by_parent.keys(), key=lambda x: len(layer2_by_parent[x]), reverse=True)[:10]:
        children = layer2_by_parent[parent]
        print(f"{parent} ({len(children)} children):")
        for child in sorted(children)[:10]:
            print(f"  - {child}")
        if len(children) > 10:
            print(f"  ... and {len(children) - 10} more")
        print()

    # Save mapping
    output = {
        'layer_assignment': {
            concept: layer for concept, layer in sorted(concept_layers.items())
        },
        'layer_distribution': dict(layer_counts),
        'layer0_concepts': sorted(list(layer0)),
        'layer1_concepts': sorted(list(layer1)),
        'layer2_by_parent': {
            parent: sorted(children)
            for parent, children in layer2_by_parent.items()
        }
    }

    output_path = OUTPUT_DIR / "semantic_layer_mapping_v5.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Layer mapping saved to {output_path}")

    return concept_layers


def main():
    parent_map, children_map, source_map, all_concepts = parse_kif_files()
    assign_layers_semantically(parent_map, children_map, source_map, all_concepts)


if __name__ == '__main__':
    main()
