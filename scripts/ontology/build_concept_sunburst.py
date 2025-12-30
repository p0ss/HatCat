#!/usr/bin/env python3
"""
Build sunburst visualization data for concept hierarchy.
Uses SUMO depth to determine parent-child relationships.
"""

import json
from pathlib import Path
from collections import defaultdict

# Load layers
layers = {}
for layer_num in range(7):
    layer_path = Path(f'data/concept_graph/abstraction_layers/layer{layer_num}.json')
    with open(layer_path) as f:
        data = json.load(f)
        if isinstance(data, dict) and 'concepts' in data:
            layers[layer_num] = data['concepts']
        elif isinstance(data, list):
            layers[layer_num] = data
        else:
            layers[layer_num] = []

print(f"Loaded {sum(len(layers[i]) for i in range(7)):,} total concepts")

# Build depth-indexed concepts for parent lookup
concepts_by_depth = defaultdict(list)
for layer_num in range(5):  # Layers 0-4 have depth-based hierarchy
    for concept in layers[layer_num]:
        depth = concept.get('sumo_depth', 0)
        concepts_by_depth[depth].append({
            'name': concept.get('sumo_term'),
            'layer': layer_num,
            'synset_count': concept.get('synset_count', 0),
            'depth': depth
        })

def find_parent(concept_name, depth):
    """Find parent concept at depth-1"""
    # For now, use a simple heuristic: find any concept at depth-1
    # In a full implementation, would use actual SUMO subclass relationships
    parent_depth = depth - 1
    if parent_depth < 0:
        return None

    candidates = concepts_by_depth.get(parent_depth, [])
    if candidates:
        # Return first candidate (could be improved with actual hierarchy)
        return candidates[0]['name']
    return None

def build_tree():
    """Build hierarchical tree using depth-based parent lookup"""
    node_registry = {}

    # Create root from depth 0
    root_concepts = concepts_by_depth.get(0, [])
    if not root_concepts:
        print("No root concepts found!")
        return None

    root = {
        "name": "Root",
        "layer": -1,
        "children": []
    }
    node_registry["Root"] = root

    # Sample sizes per layer
    sample_sizes = {0: None, 1: None, 2: 150, 3: 100, 4: 50, 5: 20, 6: 200}

    # Process layers 0-4 using depth
    for layer_num in range(5):
        concepts = layers[layer_num]

        # Sort by synset count
        concepts = sorted(concepts, key=lambda c: c.get('synset_count', 0), reverse=True)

        # Apply sampling
        limit = sample_sizes.get(layer_num)
        if limit:
            concepts = concepts[:limit]

        added = 0
        for concept in concepts:
            name = concept.get('sumo_term')
            depth = concept.get('sumo_depth', 0)

            if not name or name in node_registry:
                continue

            node = {
                "name": name,
                "layer": layer_num,
                "synset_count": concept.get('synset_count', 0),
                "children": []
            }

            # Find parent
            if depth == 0:
                parent_node = root
            else:
                # Use depth-1 as parent (simplified)
                parent_candidates = [n for n in node_registry.values()
                                    if n.get('layer', -1) == layer_num - 1]
                if parent_candidates:
                    # Just use first candidate for now
                    parent_node = parent_candidates[0]
                else:
                    parent_node = root

            parent_node["children"].append(node)
            node_registry[name] = node
            added += 1

        print(f"Layer {layer_num}: added {added} concepts")

    # Add Layer 5 (pseudo-SUMO)
    for concept in layers[5]:
        name = concept.get('sumo_term')
        parent_name = concept.get('parent_sumo')

        if name in node_registry or not parent_name:
            continue

        node = {
            "name": name,
            "layer": 5,
            "synset_count": concept.get('synset_count', 0),
            "children": []
        }

        if parent_name in node_registry:
            node_registry[parent_name]["children"].append(node)
            node_registry[name] = node

    # Add sampled Layer 6 (synsets)
    layer6_sample = sorted(layers[6], key=lambda c: c.get('frequency', 0), reverse=True)[:200]
    for concept in layer6_sample:
        lemmas = concept.get('lemmas', [])
        name = '/'.join(lemmas[:2]) if lemmas else 'Unknown'
        parent_name = concept.get('parent_category')

        if not parent_name or parent_name not in node_registry:
            continue

        node = {
            "name": name,
            "layer": 6,
            "children": []
        }

        node_registry[parent_name]["children"].append(node)

    return root

# Build and save
tree = build_tree()

if tree:
    output_path = Path('docs/concept_hierarchy_sunburst.json')
    with open(output_path, 'w') as f:
        json.dump(tree, f, indent=2)

    def count_nodes(node):
        return 1 + sum(count_nodes(c) for c in node.get('children', []))

    print(f"\nSaved to {output_path}")
    print(f"Total nodes: {count_nodes(tree):,}")
    print(f"Top-level children: {len(tree['children'])}")
else:
    print("Failed to build tree")
