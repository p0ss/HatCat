#!/usr/bin/env python3
"""
Integrate generated AI safety concepts into abstraction layer JSON files.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


def backup_layer_file(layer_path: Path):
    """Create timestamped backup of layer file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = layer_path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    backup_path = backup_dir / f"{layer_path.stem}_backup_{timestamp}.json"
    shutil.copy2(layer_path, backup_path)
    print(f"  ✓ Backed up to {backup_path}")
    return backup_path


def integrate_layer(layer_path: Path, new_concepts_path: Path, layer_num: int):
    """Add new concepts to a layer."""
    print("\n" + "=" * 80)
    print(f"LAYER {layer_num}: Adding AI safety concepts")
    print("=" * 80)

    # Backup
    backup_layer_file(layer_path)

    # Load existing layer
    with open(layer_path) as f:
        layer_data = json.load(f)

    # Load new concepts
    with open(new_concepts_path) as f:
        new_concepts = json.load(f)

    # Check which already exist
    existing_names = {c['sumo_term'] for c in layer_data['concepts']}
    concepts_to_add = [c for c in new_concepts if c['sumo_term'] not in existing_names]

    if not concepts_to_add:
        print(f"  ⚠️  All AI safety concepts already exist in Layer {layer_num}, skipping")
        return False

    # Append new concepts
    initial_count = len(layer_data['concepts'])
    for concept in concepts_to_add:
        layer_data['concepts'].append(concept)
        children_count = len(concept.get('category_children', []))
        children_str = f" ({children_count} children)" if children_count > 0 else " (leaf)"
        print(f"  ✓ Added {concept['sumo_term']}{children_str}")

    # Update parent concepts to include new children
    # For each new concept, find its SUMO parent in the layer and add it as a child
    for concept in concepts_to_add:
        # The concept's parent is determined by the AI.kif hierarchy
        # We need to find the parent concept in this or a higher layer and add this as a child
        pass  # This is handled by the category_children field already set during generation

    # Save
    with open(layer_path, 'w') as f:
        json.dump(layer_data, f, indent=2)

    final_count = len(layer_data['concepts'])
    print(f"  ✓ Layer {layer_num}: {initial_count} → {final_count} concepts (+{final_count - initial_count})")
    return True


def main():
    print("=" * 80)
    print("INTEGRATING AI SAFETY CONCEPTS INTO ABSTRACTION LAYERS")
    print("=" * 80)

    # Paths
    layer_dir = Path("data/concept_graph/abstraction_layers")
    ai_safety_dir = Path("data/concept_graph/ai_safety_layer_entries")

    layer1_path = layer_dir / "layer1.json"
    layer2_path = layer_dir / "layer2.json"
    layer3_path = layer_dir / "layer3.json"
    layer4_path = layer_dir / "layer4.json"

    layer1_concepts_path = ai_safety_dir / "layer1_parents.json"
    layer2_concepts_path = ai_safety_dir / "layer2_parents.json"
    layer3_concepts_path = ai_safety_dir / "layer3_parents.json"
    layer4_concepts_path = ai_safety_dir / "layer4_children.json"

    # Verify all files exist
    for path in [layer1_path, layer2_path, layer3_path, layer4_path,
                 layer1_concepts_path, layer2_concepts_path, layer3_concepts_path, layer4_concepts_path]:
        if not path.exists():
            print(f"✗ Error: {path} not found")
            return 1

    # Integrate each layer
    changed = []

    if integrate_layer(layer1_path, layer1_concepts_path, 1):
        changed.append("Layer 1")

    if integrate_layer(layer2_path, layer2_concepts_path, 2):
        changed.append("Layer 2")

    if integrate_layer(layer3_path, layer3_concepts_path, 3):
        changed.append("Layer 3")

    if integrate_layer(layer4_path, layer4_concepts_path, 4):
        changed.append("Layer 4")

    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATION COMPLETE")
    print("=" * 80)

    if changed:
        print(f"✓ Updated: {', '.join(changed)}")
        print("\nAI safety concepts are now integrated into the abstraction layers.")
        print("They will be automatically included in future training runs.")
    else:
        print("⚠️  No changes made (all concepts already present)")

    print("\nBackups saved to: data/concept_graph/abstraction_layers/backups/")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
