#!/usr/bin/env python3
"""
Integrate generated persona concepts into abstraction layer JSON files.
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


def integrate_layer2(layer_path: Path, parent_path: Path):
    """Add AgentPsychologicalAttribute to Layer 2."""
    print("\n" + "=" * 80)
    print("LAYER 2: Adding AgentPsychologicalAttribute")
    print("=" * 80)

    # Backup
    backup_layer_file(layer_path)

    # Load existing layer
    with open(layer_path) as f:
        layer_data = json.load(f)

    # Load new parent concept
    with open(parent_path) as f:
        new_parent = json.load(f)

    # Check if already exists
    existing = [c for c in layer_data['concepts'] if c['sumo_term'] == 'AgentPsychologicalAttribute']
    if existing:
        print(f"  ⚠️  AgentPsychologicalAttribute already exists, skipping")
        return False

    # Append new parent
    layer_data['concepts'].append(new_parent)
    print(f"  ✓ Added AgentPsychologicalAttribute")

    # Update PsychologicalAttribute to include new child
    for concept in layer_data['concepts']:
        if concept['sumo_term'] == 'PsychologicalAttribute':
            if 'category_children' not in concept:
                concept['category_children'] = []
            if 'AgentPsychologicalAttribute' not in concept['category_children']:
                concept['category_children'].append('AgentPsychologicalAttribute')
                print(f"  ✓ Added AgentPsychologicalAttribute to PsychologicalAttribute.category_children")
            break

    # Save
    with open(layer_path, 'w') as f:
        json.dump(layer_data, f, indent=2)

    print(f"  ✓ Layer 2 now has {len(layer_data['concepts'])} concepts")
    return True


def integrate_layer3(layer_path: Path, roles_path: Path):
    """Add AI/Human/Other AgentPsychology categories to Layer 3."""
    print("\n" + "=" * 80)
    print("LAYER 3: Adding role psychology categories")
    print("=" * 80)

    # Backup
    backup_layer_file(layer_path)

    # Load existing layer
    with open(layer_path) as f:
        layer_data = json.load(f)

    # Load new roles
    with open(roles_path) as f:
        new_roles = json.load(f)

    # Check which already exist
    existing_names = {c['sumo_term'] for c in layer_data['concepts']}
    roles_to_add = [r for r in new_roles if r['sumo_term'] not in existing_names]

    if not roles_to_add:
        print(f"  ⚠️  All role categories already exist, skipping")
        return False

    # Append new roles
    for role in roles_to_add:
        layer_data['concepts'].append(role)
        print(f"  ✓ Added {role['sumo_term']} ({len(role['category_children'])} children)")

    # Update AgentPsychologicalAttribute parent if it exists in layer 2
    # (It won't be in layer 3, but we document the relationship)

    # Save
    with open(layer_path, 'w') as f:
        json.dump(layer_data, f, indent=2)

    print(f"  ✓ Layer 3 now has {len(layer_data['concepts'])} concepts")
    return True


def integrate_layer4(layer_path: Path, concepts_path: Path):
    """Add 30 persona concepts to Layer 4."""
    print("\n" + "=" * 80)
    print("LAYER 4: Adding 30 persona concepts")
    print("=" * 80)

    # Backup
    backup_layer_file(layer_path)

    # Load existing layer
    with open(layer_path) as f:
        layer_data = json.load(f)

    # Load new concepts
    with open(concepts_path) as f:
        new_concepts = json.load(f)

    # Check which already exist
    existing_names = {c['sumo_term'] for c in layer_data['concepts']}
    concepts_to_add = [c for c in new_concepts if c['sumo_term'] not in existing_names]

    if not concepts_to_add:
        print(f"  ⚠️  All persona concepts already exist, skipping")
        return False

    # Append new concepts
    added_by_role = {'AIAgent': 0, 'HumanAgent': 0, 'OtherAgent': 0}
    for concept in concepts_to_add:
        layer_data['concepts'].append(concept)
        # Track by role
        for role in added_by_role.keys():
            if concept['sumo_term'].endswith(f'_{role}'):
                added_by_role[role] += 1
                break

    for role, count in added_by_role.items():
        if count > 0:
            print(f"  ✓ Added {count} {role} concepts")

    # Save
    with open(layer_path, 'w') as f:
        json.dump(layer_data, f, indent=2)

    print(f"  ✓ Layer 4 now has {len(layer_data['concepts'])} concepts")
    return True


def main():
    print("=" * 80)
    print("INTEGRATING PERSONA CONCEPTS INTO ABSTRACTION LAYERS")
    print("=" * 80)

    # Paths
    layer_dir = Path("data/concept_graph/abstraction_layers")
    persona_dir = Path("data/concept_graph/persona_layer_entries")

    layer2_path = layer_dir / "layer2.json"
    layer3_path = layer_dir / "layer3.json"
    layer4_path = layer_dir / "layer4.json"

    parent_path = persona_dir / "layer2_parent.json"
    roles_path = persona_dir / "layer3_roles.json"
    concepts_path = persona_dir / "layer4_concepts.json"

    # Verify all files exist
    for path in [layer2_path, layer3_path, layer4_path, parent_path, roles_path, concepts_path]:
        if not path.exists():
            print(f"✗ Error: {path} not found")
            return 1

    # Integrate each layer
    changed = []

    if integrate_layer2(layer2_path, parent_path):
        changed.append("Layer 2")

    if integrate_layer3(layer3_path, roles_path):
        changed.append("Layer 3")

    if integrate_layer4(layer4_path, concepts_path):
        changed.append("Layer 4")

    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATION COMPLETE")
    print("=" * 80)

    if changed:
        print(f"✓ Updated: {', '.join(changed)}")
        print("\nPersona concepts are now integrated into the abstraction layers.")
        print("They will be automatically included in future training runs.")
    else:
        print("⚠️  No changes made (all concepts already present)")

    print("\nBackups saved to: data/concept_graph/abstraction_layers/backups/")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
