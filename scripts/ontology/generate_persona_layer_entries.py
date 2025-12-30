#!/usr/bin/env python3
"""
Generate layer JSON entries for persona concepts.

Hierarchy:
  Layer 2: AgentPsychologicalAttribute (parent)
  Layer 3: AIAgentPsychology, HumanAgentPsychology, OtherAgentPsychology
  Layer 4: 30 persona concepts (10 per role)
"""

import json
import re
from pathlib import Path
from typing import Dict, List
from nltk.corpus import wordnet as wn

# Parse WordNet mappings file
def parse_wordnet_mappings(filepath: Path) -> Dict[str, List[Dict]]:
    """Parse WordNetMappings30-Persona.txt to get synsets for each concept."""
    mappings = {}

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;'):
                continue

            # Format: synset_id pos sense_count term1 term2 ... | &%ConceptName+
            if '|' not in line:
                continue

            synset_part, concept_part = line.split('|')

            # Extract concept names
            concepts = re.findall(r'&%(\w+)\+', concept_part)

            # Parse synset
            # Format: offset version pos sense_count lemma1 0 lemma2 0 ... 000
            parts = synset_part.strip().split()
            if len(parts) < 5:
                continue

            synset_offset = parts[0]
            version = parts[1]  # Usually "03"
            pos = parts[2]  # n, v, a, s, r
            sense_count = parts[3]

            # Get lemmas (skip offset, version, pos, sense_count, then read lemma 0 pairs)
            lemmas = []
            i = 4
            while i < len(parts) and parts[i] != '000':
                if not parts[i].isdigit():  # Skip sense numbers (the "0" after each lemma)
                    lemmas.append(parts[i].replace('_', '-'))
                i += 1

            # Get synset for definition
            try:
                synset_id = wn.synset_from_pos_and_offset(pos, int(synset_offset))
                synset_name = synset_id.name()
                definition = synset_id.definition()
                lexname = synset_id.lexname()
            except:
                # Fallback if synset not found
                first_lemma = lemmas[0] if lemmas else 'unknown'
                synset_name = f"{first_lemma}.{pos}.01"
                definition = f"Concept: {first_lemma}"
                lexname = f"{pos}.Tops"

            synset_info = {
                'synset_name': synset_name,
                'lemmas': lemmas if lemmas else [first_lemma],
                'definition': definition,
                'pos': pos,
                'lexname': lexname
            }

            # Add to each concept
            for concept in concepts:
                if concept not in mappings:
                    mappings[concept] = []
                mappings[concept].append(synset_info)

    return mappings


def generate_layer2_parent() -> Dict:
    """Generate AgentPsychologicalAttribute for Layer 2."""
    # Find a good synset for "psychological attribute of agents"
    # Using "psychology" as the canonical synset
    psych_synset = wn.synset('psychology.n.01')

    return {
        "sumo_term": "AgentPsychologicalAttribute",
        "sumo_depth": 5,
        "layer": 2,
        "is_category_lens": True,
        "is_pseudo_sumo": False,
        "category_children": [
            "AIAgentPsychology",
            "HumanAgentPsychology",
            "OtherAgentPsychology"
        ],
        "synset_count": 1,
        "direct_synset_count": 1,
        "synsets": [psych_synset.name()],
        "canonical_synset": psych_synset.name(),
        "lemmas": ["psychology"],
        "pos": "n",
        "definition": psych_synset.definition(),
        "lexname": psych_synset.lexname()
    }


def generate_layer3_roles(mappings: Dict[str, List[Dict]]) -> List[Dict]:
    """Generate AIAgentPsychology, HumanAgentPsychology, OtherAgentPsychology for Layer 3."""

    roles = [
        ("AIAgentPsychology", "AIAgent", "Psychological attributes specific to AI agents"),
        ("HumanAgentPsychology", "HumanAgent", "Psychological attributes specific to human agents"),
        ("OtherAgentPsychology", "OtherAgent", "Observable psychological attributes of other agents")
    ]

    layer3_concepts = []

    for role_name, role_suffix, description in roles:
        # Find children (all persona concepts ending with this role)
        children = []
        for concept in mappings.keys():
            if concept.endswith(f"_{role_suffix}"):
                children.append(concept)

        # Use psychology synset as canonical
        psych_synset = wn.synset('psychology.n.01')  # Only has one sense

        layer3_concepts.append({
            "sumo_term": role_name,
            "sumo_depth": 7,
            "layer": 3,
            "is_category_lens": True,
            "is_pseudo_sumo": False,
            "category_children": sorted(children),
            "synset_count": 1,
            "direct_synset_count": 1,
            "synsets": [psych_synset.name()],
            "canonical_synset": psych_synset.name(),
            "lemmas": ["psychology"],
            "pos": "n",
            "definition": description,
            "lexname": psych_synset.lexname()
        })

    return layer3_concepts


def generate_layer4_concepts(mappings: Dict[str, List[Dict]]) -> List[Dict]:
    """Generate all 30 persona concepts for Layer 4."""

    layer4_concepts = []

    # Define parent for each concept based on role
    parent_map = {
        'AIAgent': 'AIAgentPsychology',
        'HumanAgent': 'HumanAgentPsychology',
        'OtherAgent': 'OtherAgentPsychology'
    }

    for concept_name, synset_list in sorted(mappings.items()):
        # Determine role from concept name
        role = None
        for r in ['AIAgent', 'HumanAgent', 'OtherAgent']:
            if concept_name.endswith(f'_{r}'):
                role = r
                break

        if not role or not synset_list:
            continue

        # Collect unique synsets
        synset_names = list(dict.fromkeys([s['synset_name'] for s in synset_list]))

        # Use first synset as canonical
        canonical = synset_list[0]

        layer4_concepts.append({
            "sumo_term": concept_name,
            "sumo_depth": 8,
            "layer": 4,
            "is_category_lens": False,  # Leaf concepts
            "is_pseudo_sumo": False,
            "category_children": [],
            "synset_count": len(synset_names),
            "direct_synset_count": len(synset_names),
            "synsets": synset_names,
            "canonical_synset": canonical['synset_name'],
            "lemmas": canonical['lemmas'],
            "pos": canonical['pos'],
            "definition": canonical['definition'],
            "lexname": canonical['lexname']
        })

    return layer4_concepts


def main():
    print("=" * 80)
    print("GENERATING PERSONA CONCEPT LAYER ENTRIES")
    print("=" * 80)
    print()

    # Parse WordNet mappings
    mappings_file = Path("data/concept_graph/WordNetMappings30-Persona.txt")
    print(f"Parsing {mappings_file}...")
    mappings = parse_wordnet_mappings(mappings_file)
    print(f"✓ Found {len(mappings)} persona concepts")
    print()

    # Generate Layer 2 parent
    print("Generating Layer 2 parent (AgentPsychologicalAttribute)...")
    layer2_parent = generate_layer2_parent()
    print(f"✓ {layer2_parent['sumo_term']}")
    print()

    # Generate Layer 3 roles
    print("Generating Layer 3 role categories...")
    layer3_roles = generate_layer3_roles(mappings)
    for role in layer3_roles:
        print(f"✓ {role['sumo_term']} ({len(role['category_children'])} children)")
    print()

    # Generate Layer 4 concepts
    print("Generating Layer 4 persona concepts...")
    layer4_concepts = generate_layer4_concepts(mappings)
    print(f"✓ Generated {len(layer4_concepts)} persona concepts")
    print()

    # Save outputs
    output_dir = Path("data/concept_graph/persona_layer_entries")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "layer2_parent.json", 'w') as f:
        json.dump(layer2_parent, f, indent=2)

    with open(output_dir / "layer3_roles.json", 'w') as f:
        json.dump(layer3_roles, f, indent=2)

    with open(output_dir / "layer4_concepts.json", 'w') as f:
        json.dump(layer4_concepts, f, indent=2)

    print(f"✓ Saved to {output_dir}/")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Layer 2: 1 parent concept (AgentPsychologicalAttribute)")
    print(f"Layer 3: 3 role categories (AI/Human/Other)")
    print(f"Layer 4: {len(layer4_concepts)} persona concepts")
    print()
    print("Next steps:")
    print("1. Review generated JSON files")
    print("2. Append layer2_parent to data/concept_graph/abstraction_layers/layer2.json")
    print("3. Append layer3_roles to data/concept_graph/abstraction_layers/layer3.json")
    print("4. Append layer4_concepts to data/concept_graph/abstraction_layers/layer4.json")
    print("5. Update PsychologicalAttribute.category_children in layer2.json to include AgentPsychologicalAttribute")


if __name__ == '__main__':
    main()
