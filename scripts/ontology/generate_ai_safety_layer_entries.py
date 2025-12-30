#!/usr/bin/env python3
"""
Generate layer JSON entries for AI safety concepts.

Hierarchy (from AI.kif):
  Layer 1: AIAlignmentFailureMode, AIAlignmentPrinciple, AIMetaOptimization, AIRiskScenario
  Layer 2: AIStrategicDeception, AIGovernanceProcess
  Layer 3: AIMoralStatus, AIAlignmentState, AIHarmState, AIWelfareState
  Layer 4+: Child concepts
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from nltk.corpus import wordnet as wn


def parse_ai_kif(filepath: Path) -> Dict[str, Dict]:
    """Parse AI.kif to extract parent-child relationships and opposites."""
    hierarchy = {}
    opposites = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;'):
                continue

            # Parse (subclass Child Parent)
            subclass_match = re.match(r'\(subclass\s+(\w+)\s+(\w+)\)', line)
            if subclass_match:
                child, parent = subclass_match.groups()
                if parent not in hierarchy:
                    hierarchy[parent] = {'children': set(), 'parent': None}
                if child not in hierarchy:
                    hierarchy[child] = {'children': set(), 'parent': None}
                hierarchy[parent]['children'].add(child)
                hierarchy[child]['parent'] = parent

            # Parse (OppositeConcept A B)
            opposite_match = re.match(r'\(OppositeConcept\s+(\w+)\s+(\w+)\)', line)
            if opposite_match:
                a, b = opposite_match.groups()
                opposites.append((a, b))

    return hierarchy, opposites


def parse_wordnet_mappings(filepath: Path) -> Dict[str, List[Dict]]:
    """Parse WordNetMappings30-AI-symmetry.txt to get synsets for each concept."""
    mappings = {}

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;'):
                continue

            if '|' not in line:
                continue

            synset_part, concept_part = line.split('|')

            # Extract concept names
            concepts = re.findall(r'&%([\w-]+)\+', concept_part)

            # Parse synset
            parts = synset_part.strip().split()
            if len(parts) < 5:
                continue

            synset_offset = parts[0]
            version = parts[1]
            pos = parts[2]
            sense_count = parts[3]

            # Get lemmas
            lemmas = []
            i = 4
            while i < len(parts) and parts[i] != '000':
                if not parts[i].isdigit():
                    lemmas.append(parts[i].replace('_', '-'))
                i += 1

            # Get synset for definition
            try:
                synset_id = wn.synset_from_pos_and_offset(pos, int(synset_offset))
                synset_name = synset_id.name()
                definition = synset_id.definition()
                lexname = synset_id.lexname()
            except:
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

            for concept in concepts:
                if concept not in mappings:
                    mappings[concept] = []
                mappings[concept].append(synset_info)

    return mappings


def get_all_descendants(hierarchy: Dict, parent: str) -> Set[str]:
    """Get all descendants (children, grandchildren, etc.) of a parent."""
    descendants = set()
    if parent not in hierarchy:
        return descendants

    for child in hierarchy[parent]['children']:
        descendants.add(child)
        descendants.update(get_all_descendants(hierarchy, child))

    return descendants


def generate_layer_concepts(
    hierarchy: Dict,
    mappings: Dict,
    layer_num: int,
    target_depth: int,
    parent_concepts: List[str]
) -> List[Dict]:
    """Generate concepts for a specific layer."""

    layer_concepts = []

    for parent_concept in parent_concepts:
        if parent_concept not in hierarchy:
            continue

        # Get direct children
        children = hierarchy[parent_concept].get('children', set())

        # Get synsets for this concept
        synset_list = mappings.get(parent_concept, [])

        # Collect unique synsets
        synset_names = list(dict.fromkeys([s['synset_name'] for s in synset_list])) if synset_list else []

        # Use first synset as canonical, or create fallback
        if synset_list:
            canonical = synset_list[0]
        else:
            # Fallback for concepts without WordNet mappings
            canonical = {
                'synset_name': f"{parent_concept.lower()}.n.01",
                'lemmas': [parent_concept.lower()],
                'definition': f"AI safety concept: {parent_concept}",
                'pos': 'n',
                'lexname': 'noun.Tops'
            }
            synset_names = [canonical['synset_name']]

        is_category = len(children) > 0

        layer_concepts.append({
            "sumo_term": parent_concept,
            "sumo_depth": target_depth,
            "layer": layer_num,
            "is_category_lens": is_category,
            "is_pseudo_sumo": False,
            "category_children": sorted(list(children)),
            "synset_count": len(synset_names),
            "direct_synset_count": len(synset_names),
            "synsets": synset_names,
            "canonical_synset": canonical['synset_name'],
            "lemmas": canonical['lemmas'],
            "pos": canonical['pos'],
            "definition": canonical['definition'],
            "lexname": canonical['lexname']
        })

    return layer_concepts


def generate_child_concepts(
    hierarchy: Dict,
    mappings: Dict,
    parent_concepts: List[str],
    layer_num: int,
    target_depth: int
) -> List[Dict]:
    """Generate all child concepts for given parents."""

    all_children = set()
    for parent in parent_concepts:
        all_children.update(get_all_descendants(hierarchy, parent))

    child_concepts = []

    for child_concept in sorted(all_children):
        # Determine if this child has children (is a category)
        has_children = child_concept in hierarchy and len(hierarchy[child_concept].get('children', set())) > 0

        # Get synsets
        synset_list = mappings.get(child_concept, [])

        # Collect unique synsets
        synset_names = list(dict.fromkeys([s['synset_name'] for s in synset_list])) if synset_list else []

        # Use first synset as canonical, or create fallback
        if synset_list:
            canonical = synset_list[0]
        else:
            canonical = {
                'synset_name': f"{child_concept.lower()}.n.01",
                'lemmas': [child_concept.lower()],
                'definition': f"AI safety concept: {child_concept}",
                'pos': 'n',
                'lexname': 'noun.Tops'
            }
            synset_names = [canonical['synset_name']]

        children_list = sorted(list(hierarchy[child_concept]['children'])) if has_children else []

        child_concepts.append({
            "sumo_term": child_concept,
            "sumo_depth": target_depth,
            "layer": layer_num,
            "is_category_lens": has_children,
            "is_pseudo_sumo": False,
            "category_children": children_list,
            "synset_count": len(synset_names),
            "direct_synset_count": len(synset_names),
            "synsets": synset_names,
            "canonical_synset": canonical['synset_name'],
            "lemmas": canonical['lemmas'],
            "pos": canonical['pos'],
            "definition": canonical['definition'],
            "lexname": canonical['lexname']
        })

    return child_concepts


def main():
    print("=" * 80)
    print("GENERATING AI SAFETY CONCEPT LAYER ENTRIES")
    print("=" * 80)
    print()

    # Parse AI.kif
    kif_file = Path("data/concept_graph/sumo_source/AI.kif")
    print(f"Parsing {kif_file}...")
    hierarchy, opposites = parse_ai_kif(kif_file)
    print(f"✓ Found {len(hierarchy)} concepts in hierarchy")
    print(f"✓ Found {len(opposites)} opposite pairs")
    print()

    # Parse WordNet mappings
    mappings_file = Path("data/concept_graph/WordNetMappings30-AI-symmetry.txt")
    print(f"Parsing {mappings_file}...")
    mappings = parse_wordnet_mappings(mappings_file)
    print(f"✓ Found {len(mappings)} concepts with WordNet mappings")
    print()

    # Layer 1 parent concepts (depth 3)
    layer1_parents = [
        'AIAlignmentFailureMode',
        'AIAlignmentPrinciple',
        'AIMetaOptimization',
        'AIRiskScenario'
    ]

    # Layer 2 parent concepts (depth 4)
    layer2_parents = [
        'AIStrategicDeception',
        'AIGovernanceProcess'
    ]

    # Layer 3 parent concepts (depth 6-7)
    layer3_parents = [
        'AIMoralStatus',
        'AIAlignmentState',
        'AIHarmState',
        'AIWelfareState'
    ]

    # Generate layers
    print("Generating Layer 1 parent concepts...")
    layer1_concepts = generate_layer_concepts(hierarchy, mappings, 1, 3, layer1_parents)
    print(f"✓ Generated {len(layer1_concepts)} parent concepts")
    for c in layer1_concepts:
        print(f"  - {c['sumo_term']} ({len(c['category_children'])} children)")
    print()

    print("Generating Layer 2 parent concepts...")
    layer2_concepts = generate_layer_concepts(hierarchy, mappings, 2, 4, layer2_parents)
    print(f"✓ Generated {len(layer2_concepts)} parent concepts")
    for c in layer2_concepts:
        print(f"  - {c['sumo_term']} ({len(c['category_children'])} children)")
    print()

    print("Generating Layer 3 parent concepts...")
    layer3_concepts = generate_layer_concepts(hierarchy, mappings, 3, 7, layer3_parents)
    print(f"✓ Generated {len(layer3_concepts)} parent concepts")
    for c in layer3_concepts:
        print(f"  - {c['sumo_term']} ({len(c['category_children'])} children)")
    print()

    print("Generating child concepts for deeper layers...")
    # All children of layer 1-3 parents
    all_parent_concepts = layer1_parents + layer2_parents + layer3_parents
    deeper_children = generate_child_concepts(hierarchy, mappings, all_parent_concepts, 4, 8)
    print(f"✓ Generated {len(deeper_children)} child concepts")
    print()

    # Save outputs
    output_dir = Path("data/concept_graph/ai_safety_layer_entries")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "layer1_parents.json", 'w') as f:
        json.dump(layer1_concepts, f, indent=2)

    with open(output_dir / "layer2_parents.json", 'w') as f:
        json.dump(layer2_concepts, f, indent=2)

    with open(output_dir / "layer3_parents.json", 'w') as f:
        json.dump(layer3_concepts, f, indent=2)

    with open(output_dir / "layer4_children.json", 'w') as f:
        json.dump(deeper_children, f, indent=2)

    print(f"✓ Saved to {output_dir}/")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Layer 1: {len(layer1_concepts)} parent concepts")
    print(f"Layer 2: {len(layer2_concepts)} parent concepts")
    print(f"Layer 3: {len(layer3_concepts)} parent concepts")
    print(f"Layer 4+: {len(deeper_children)} child concepts")
    print(f"Total: {len(layer1_concepts) + len(layer2_concepts) + len(layer3_concepts) + len(deeper_children)} AI safety concepts")
    print()
    print("Next steps:")
    print("1. Review generated JSON files")
    print("2. Run integrate_ai_safety_concepts.py to append to layer JSONs")


if __name__ == '__main__':
    main()
