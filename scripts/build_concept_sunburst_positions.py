#!/usr/bin/env python3
"""
Build sunburst layout for SUMO concepts.

Arranges concepts in concentric rings (layers) with angular positions
determined by relationship strength to siblings. This creates a stable
color mapping where hue = angle on color wheel.

Output: concept_sunburst_positions.json
  {
    "concept_name": {
      "layer": 0-6,
      "angle": 0-360,  // degrees, for HSL hue
      "radius": 0-1,   // normalized layer depth
      "parent": "parent_name",
      "children": ["child1", "child2", ...]
    }
  }
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

def load_concept_metadata(probes_dir: Path) -> Dict[str, dict]:
    """Load all concept metadata from trained probes."""

    concepts = {}

    for layer_dir in sorted(probes_dir.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith('layer'):
            continue

        layer_num = int(layer_dir.name.replace('layer', ''))

        for concept_dir in layer_dir.iterdir():
            if not concept_dir.is_dir():
                continue

            concept_name = concept_dir.name
            metadata_file = concept_dir / 'metadata.json'

            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

                concepts[concept_name] = {
                    'layer': layer_num,
                    'metadata': metadata
                }

    return concepts

def build_hierarchy(concepts: Dict[str, dict]) -> Tuple[Dict, Dict]:
    """Build parent-child hierarchy from concept metadata."""

    parent_to_children = defaultdict(list)
    child_to_parent = {}

    # Extract from metadata if available
    for concept_name, data in concepts.items():
        metadata = data.get('metadata', {})

        # Try to get parent from metadata
        parent = metadata.get('parent')
        if parent and parent in concepts:
            parent_to_children[parent].append(concept_name)
            child_to_parent[concept_name] = parent

    # Fallback: use layer structure (layer N concepts are children of layer N-1)
    layer_concepts = defaultdict(list)
    for concept_name, data in concepts.items():
        layer_concepts[data['layer']].append(concept_name)

    # For concepts without explicit parent, assign to layer-1 root
    for layer in sorted(layer_concepts.keys()):
        if layer == 0:
            continue

        for concept_name in layer_concepts[layer]:
            if concept_name not in child_to_parent:
                # Find closest parent from previous layer
                # For now, just mark as orphan
                pass

    return dict(parent_to_children), child_to_parent

def calculate_relationship_strength(concept_name: str, concepts: Dict) -> float:
    """
    Calculate aggregate relationship strength for a concept.

    This is used to order siblings - concepts with stronger/more relationships
    are positioned closer together.
    """

    if concept_name not in concepts:
        return 0.0

    metadata = concepts[concept_name].get('metadata', {})

    # Count children
    num_children = metadata.get('num_synsets', 0)

    # Count training samples (proxy for concept complexity)
    num_samples = metadata.get('num_samples', 10)

    # F1 score (proxy for concept clarity)
    f1_score = metadata.get('test_f1', 0.5)

    # Combine into strength metric
    strength = np.log1p(num_children) * 0.5 + np.log1p(num_samples) * 0.3 + f1_score * 0.2

    return strength

def assign_angular_positions(
    concepts: Dict[str, dict],
    parent_to_children: Dict[str, List[str]],
    child_to_parent: Dict[str, str]
) -> Dict[str, dict]:
    """
    Assign angular positions to concepts using sunburst layout.

    Algorithm:
    1. Start with layer 0 concepts evenly distributed 0-360°
    2. For each layer, assign children to angular ranges proportional to their parent's span
    3. Within each range, order children by relationship strength
    4. Recursively assign angles
    """

    positions = {}

    # Layer 0: root concepts evenly distributed
    layer_0_concepts = [c for c, d in concepts.items() if d['layer'] == 0]
    layer_0_concepts = sorted(layer_0_concepts)  # Alphabetical for stability

    print(f"Layer 0: {len(layer_0_concepts)} root concepts")

    if not layer_0_concepts:
        return positions

    # Assign root concepts evenly around circle
    for i, concept in enumerate(layer_0_concepts):
        angle = (i / len(layer_0_concepts)) * 360.0
        positions[concept] = {
            'layer': 0,
            'angle': angle,
            'radius': 0.0,
            'parent': None,
            'children': parent_to_children.get(concept, [])
        }

    # Process each subsequent layer
    max_layer = max(d['layer'] for d in concepts.values())

    for layer in range(1, max_layer + 1):
        print(f"Layer {layer}...")

        layer_concepts = [c for c, d in concepts.items() if d['layer'] == layer]

        for concept in layer_concepts:
            parent = child_to_parent.get(concept)

            if parent and parent in positions:
                # Inherit parent's angular position as base
                parent_angle = positions[parent]['angle']

                # Get all siblings (children of same parent)
                siblings = parent_to_children.get(parent, [])

                if len(siblings) == 1:
                    # Only child: use parent's exact angle
                    angle = parent_angle
                else:
                    # Multiple siblings: spread around parent's angle
                    # Calculate strengths
                    sibling_strengths = {
                        sib: calculate_relationship_strength(sib, concepts)
                        for sib in siblings
                    }

                    # Sort siblings by strength (strongest in middle)
                    sorted_siblings = sorted(siblings, key=lambda s: -sibling_strengths[s])

                    # Find this concept's index among siblings
                    sibling_idx = sorted_siblings.index(concept)

                    # Spread siblings in angular range proportional to parent's span
                    # Use ±30° around parent as default spread
                    spread = min(60.0, 360.0 / len(layer_0_concepts))

                    # Distribute siblings
                    if len(siblings) > 1:
                        offset = (sibling_idx - (len(siblings) - 1) / 2) * (spread / len(siblings))
                        angle = (parent_angle + offset) % 360.0
                    else:
                        angle = parent_angle
            else:
                # Orphan: assign to unused angular space
                # For now, distribute evenly
                orphans_in_layer = [c for c in layer_concepts if child_to_parent.get(c) not in positions]
                orphan_idx = orphans_in_layer.index(concept) if concept in orphans_in_layer else 0
                angle = (orphan_idx / max(len(orphans_in_layer), 1)) * 360.0

            # Radius increases with layer depth
            radius = layer / max_layer

            positions[concept] = {
                'layer': layer,
                'angle': angle,
                'radius': radius,
                'parent': parent,
                'children': parent_to_children.get(concept, [])
            }

    return positions

def main():
    print("=" * 80)
    print("BUILDING SUNBURST CONCEPT POSITIONS")
    print("=" * 80)
    print()

    probes_dir = Path('results/sumo_classifiers_adaptive_l0_5')

    if not probes_dir.exists():
        print(f"Error: {probes_dir} not found")
        print("Run training first: poetry run python scripts/train_sumo_classifiers.py")
        return

    print("Loading concept metadata...")
    concepts = load_concept_metadata(probes_dir)
    print(f"✓ Loaded {len(concepts)} concepts")
    print()

    print("Building hierarchy...")
    parent_to_children, child_to_parent = build_hierarchy(concepts)
    print(f"✓ Found {len(parent_to_children)} parents with children")
    print()

    print("Assigning angular positions...")
    positions = assign_angular_positions(concepts, parent_to_children, child_to_parent)
    print(f"✓ Assigned positions to {len(positions)} concepts")
    print()

    # Statistics
    print("Layer distribution:")
    layer_counts = defaultdict(int)
    for pos in positions.values():
        layer_counts[pos['layer']] += 1

    for layer in sorted(layer_counts.keys()):
        print(f"  Layer {layer}: {layer_counts[layer]} concepts")
    print()

    # Example positions
    print("Example positions (first 10 concepts):")
    for i, (concept, pos) in enumerate(sorted(positions.items())[:10]):
        print(f"  {concept:30s} layer={pos['layer']} angle={pos['angle']:6.1f}° radius={pos['radius']:.2f}")
    print()

    # Save positions
    output_file = Path('results/concept_sunburst_positions.json')
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w') as f:
        json.dump(positions, f, indent=2)

    print(f"✓ Saved sunburst positions to: {output_file}")
    print()

    # Color wheel mapping info
    print("=" * 80)
    print("COLOR WHEEL MAPPING")
    print("=" * 80)
    print()
    print("Hue mapping (HSL):")
    print("  Angle 0° (Red):     Physical/Object concepts")
    print("  Angle 120° (Green): Abstract/Process concepts")
    print("  Angle 240° (Blue):  Relation/Attribute concepts")
    print()
    print("Brightness mapping:")
    print("  Divergence 0.0:   Lightness 90% (very bright)")
    print("  Divergence 0.5:   Lightness 50% (medium)")
    print("  Divergence 1.0:   Lightness 10% (very dark)")
    print()
    print("Saturation:")
    print("  Fixed at 70% for vivid but not oversaturated colors")
    print()

if __name__ == "__main__":
    main()
