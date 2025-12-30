#!/usr/bin/env python3
"""
Build sunburst layout for SUMO concepts using DynamicLensManager hierarchy.

Simpler approach: Use the concept_metadata that's already loaded by DynamicLensManager.
"""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.monitoring.lens_manager import DynamicLensManager
from collections import defaultdict
import numpy as np

def calculate_concept_strength(metadata) -> float:
    """Calculate concept strength from metadata for ordering siblings."""

    # Use number of children as primary strength
    num_children = len(metadata.children) if hasattr(metadata, 'children') else 0

    # Use synset count as secondary
    num_synsets = getattr(metadata, 'synset_count', 0)

    strength = np.log1p(num_children) * 2.0 + np.log1p(num_synsets) * 1.0

    return strength

def assign_sunburst_positions(manager: DynamicLensManager) -> dict:
    """Assign angular positions using sunburst layout."""

    positions = {}

    # Get all concepts by layer
    concepts_by_layer = defaultdict(list)
    for (concept_name, layer), metadata in manager.concept_metadata.items():
        concepts_by_layer[layer].append((concept_name, metadata))

    max_layer = max(concepts_by_layer.keys())

    # Layer 0: distribute evenly around circle
    layer_0_concepts = sorted(concepts_by_layer[0], key=lambda x: x[0])  # Alphabetical

    print(f"Layer 0: {len(layer_0_concepts)} concepts")

    for i, (concept_name, metadata) in enumerate(layer_0_concepts):
        angle = (i / len(layer_0_concepts)) * 360.0

        positions[concept_name] = {
            'layer': 0,
            'angle': angle,
            'radius': 0.0,
            'parent': None,
            'children': [c[0] for c in manager.parent_to_children.get((concept_name, 0), [])]
        }

    # Process subsequent layers
    for layer in range(1, max_layer + 1):
        layer_concepts = concepts_by_layer[layer]

        print(f"Layer {layer}: {len(layer_concepts)} concepts")

        for concept_name, metadata in layer_concepts:
            # Find parent
            parent_key = manager.child_to_parent.get((concept_name, layer))

            if parent_key and parent_key[0] in positions:
                parent_name = parent_key[0]
                parent_pos = positions[parent_name]
                parent_angle = parent_pos['angle']

                # Get siblings (all children of same parent)
                siblings = [c[0] for c in manager.parent_to_children.get(parent_key, [])]

                if len(siblings) == 1:
                    # Only child: inherit parent's angle exactly
                    angle = parent_angle
                else:
                    # Multiple siblings: spread them around parent
                    # Calculate strengths for ordering
                    sibling_strengths = {}
                    for sib in siblings:
                        sib_key = (sib, layer)
                        if sib_key in manager.concept_metadata:
                            sibling_strengths[sib] = calculate_concept_strength(
                                manager.concept_metadata[sib_key]
                            )
                        else:
                            sibling_strengths[sib] = 0.0

                    # Sort siblings by strength (strongest in middle)
                    sorted_siblings = sorted(siblings, key=lambda s: -sibling_strengths[s])

                    # Find index of current concept
                    sibling_idx = sorted_siblings.index(concept_name)

                    # Angular spread: decrease with layer depth
                    # Layer 1: ±30°, Layer 2: ±20°, etc.
                    base_spread = max(10.0, 40.0 - layer * 5.0)
                    spread = min(base_spread, 360.0 / len(layer_0_concepts))

                    # Distribute siblings symmetrically around parent
                    offset = (sibling_idx - (len(siblings) - 1) / 2) * (spread / max(len(siblings), 1))
                    angle = (parent_angle + offset) % 360.0
            else:
                # Orphan concept: distribute in unused space
                # For simplicity, use alphabetical positioning
                orphans = [c for c, m in layer_concepts if (c, layer) not in manager.child_to_parent or manager.child_to_parent.get((c, layer), (None,))[0] not in positions]
                if concept_name in orphans:
                    orphan_idx = orphans.index(concept_name)
                    angle = (orphan_idx / max(len(orphans), 1)) * 360.0
                else:
                    angle = 0.0

            # Radius increases with layer
            radius = layer / max_layer if max_layer > 0 else 0.0

            positions[concept_name] = {
                'layer': layer,
                'angle': angle,
                'radius': radius,
                'parent': parent_key[0] if parent_key else None,
                'children': [c[0] for c in manager.parent_to_children.get((concept_name, layer), [])]
            }

    return positions

def main():
    print("=" * 80)
    print("BUILDING SUNBURST CONCEPT POSITIONS")
    print("=" * 80)
    print()

    print("Loading DynamicLensManager...")
    manager = DynamicLensManager(
        lenses_dir=Path('results/sumo_classifiers_adaptive_l0_5'),
        base_layers=[0],
        use_activation_lenses=False,  # Don't load actual lenses
        use_text_lenses=False,
        keep_top_k=0,
    )

    print(f"✓ Loaded metadata for {len(manager.concept_metadata)} concepts")
    print(f"✓ Found {len(manager.parent_to_children)} parent-child relationships")
    print()

    print("Assigning sunburst positions...")
    positions = assign_sunburst_positions(manager)
    print(f"✓ Assigned {len(positions)} concept positions")
    print()

    # Statistics
    print("Layer distribution:")
    layer_counts = defaultdict(int)
    for pos in positions.values():
        layer_counts[pos['layer']] += 1

    for layer in sorted(layer_counts.keys()):
        print(f"  Layer {layer}: {layer_counts[layer]:5d} concepts")
    print()

    # Example positions
    print("Example positions (layer 0):")
    layer_0 = [(c, p) for c, p in positions.items() if p['layer'] == 0]
    for concept, pos in sorted(layer_0, key=lambda x: x[1]['angle'])[:10]:
        print(f"  {concept:30s} angle={pos['angle']:6.1f}° children={len(pos['children'])}")
    print()

    # Save
    output_file = Path('results/concept_sunburst_positions.json')
    with open(output_file, 'w') as f:
        json.dump(positions, f, indent=2)

    print(f"✓ Saved to: {output_file}")
    print()

    # Color mapping info
    print("=" * 80)
    print("COLOR WHEEL MAPPING (HSL)")
    print("=" * 80)
    print()
    print("Hue (0-360°): Determined by concept's angular position in sunburst")
    print("Saturation: Fixed at 70% (vivid but not oversaturated)")
    print("Lightness: Inversely proportional to divergence")
    print("  - Low divergence (0.0):  90% lightness (bright)")
    print("  - Med divergence (0.5):  50% lightness (medium)")
    print("  - High divergence (1.0): 10% lightness (dark)")
    print()

if __name__ == "__main__":
    main()
