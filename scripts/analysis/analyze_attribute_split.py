#!/usr/bin/env python3
"""
Analyze Attribute's 139 children to propose semantic subcategories.

Goal: Split Attribute into ~10-15 trainable subcategories based on semantic similarity.
"""

import re
import json
from pathlib import Path
from collections import defaultdict, Counter

SUMO_SOURCE_DIR = Path("data/concept_graph/sumo_source")

def parse_kif_files():
    """Parse all KIF files."""
    parent_map = defaultdict(set)
    children_map = defaultdict(set)
    definitions = {}

    for kif_file in SUMO_SOURCE_DIR.glob("*.kif"):
        with open(kif_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Parse subclass relationships
        subclass_pattern = r'\(subclass\s+([A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]+)\)'
        for match in re.finditer(subclass_pattern, content):
            child, parent = match.groups()
            if not child.startswith('?') and not parent.startswith('?'):
                parent_map[child].add(parent)
                children_map[parent].add(child)

        # Parse documentation
        doc_pattern = r'\(documentation\s+([A-Za-z0-9_-]+)\s+EnglishLanguage\s+"([^"]+)"'
        for match in re.finditer(doc_pattern, content):
            concept, definition = match.groups()
            definitions[concept] = definition

    return parent_map, children_map, definitions


def analyze_attribute_children(parent_map, children_map, definitions):
    """Analyze Attribute's direct children and propose semantic groups."""

    attribute_children = sorted(children_map.get('Attribute', []))

    print("=" * 80)
    print(f"ATTRIBUTE CHILDREN ANALYSIS ({len(attribute_children)} concepts)")
    print("=" * 80)

    # Categorize based on naming patterns and semantics
    categories = {
        'Relational': [],      # Relationships between entities
        'Internal': [],        # Intrinsic properties
        'Physical': [],        # Physical properties (color, shape, size)
        'Temporal': [],        # Time-related properties
        'Normative': [],       # Norms, values, standards
        'Perceptual': [],      # Sensory properties
        'Abstract': [],        # Mathematical, logical properties
        'Behavioral': [],      # Behavioral characteristics
        'Systemic': [],        # System-level properties
        'Cognitive': [],       # Mental/cognitive properties
        'Alignment': [],       # AI alignment properties (custom)
        'Other': []
    }

    # Keyword patterns for categorization
    relational_keywords = ['Relational', 'Between', 'Inter', 'Mutual']
    internal_keywords = ['Internal', 'Intrinsic', 'Inherent']
    physical_keywords = ['Color', 'Shape', 'Size', 'Texture', 'Physical', 'Spatial']
    temporal_keywords = ['Temporal', 'Time', 'Duration', 'Frequency']
    normative_keywords = ['Norm', 'Value', 'Standard', 'Truth', 'Belief']
    perceptual_keywords = ['Perceptual', 'Sensory', 'Sound', 'Taste', 'Odor']
    abstract_keywords = ['Quantity', 'Number', 'Measure', 'Abstract']
    behavioral_keywords = ['Behavioral', 'Habit', 'Skill', 'Trait']
    systemic_keywords = ['Systemic', 'Structural', 'Organizational']
    cognitive_keywords = ['Cognitive', 'Mental', 'Psychological']
    alignment_keywords = ['Alignment', 'Objective', 'Goal', 'Value']

    for child in attribute_children:
        defn = definitions.get(child, '')
        categorized = False

        # Check naming patterns
        if any(kw in child for kw in alignment_keywords):
            categories['Alignment'].append(child)
        elif any(kw in child for kw in relational_keywords):
            categories['Relational'].append(child)
        elif any(kw in child for kw in internal_keywords):
            categories['Internal'].append(child)
        elif any(kw in child for kw in physical_keywords):
            categories['Physical'].append(child)
        elif any(kw in child for kw in temporal_keywords):
            categories['Temporal'].append(child)
        elif any(kw in child for kw in normative_keywords):
            categories['Normative'].append(child)
        elif any(kw in child for kw in perceptual_keywords):
            categories['Perceptual'].append(child)
        elif any(kw in child for kw in abstract_keywords):
            categories['Abstract'].append(child)
        elif any(kw in child for kw in behavioral_keywords):
            categories['Behavioral'].append(child)
        elif any(kw in child for kw in systemic_keywords):
            categories['Systemic'].append(child)
        elif any(kw in child for kw in cognitive_keywords):
            categories['Cognitive'].append(child)
        else:
            categories['Other'].append(child)

    # Print categorization
    print("\nProposed semantic categories:\n")

    for category, concepts in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        if len(concepts) > 0:
            print(f"{category} ({len(concepts)} concepts):")
            for concept in concepts[:15]:
                child_count = len(children_map.get(concept, []))
                defn = definitions.get(concept, 'No definition')[:80]
                print(f"  {concept} ({child_count} children) - {defn}")
            if len(concepts) > 15:
                print(f"  ... and {len(concepts) - 15} more")
            print()

    # Analyze Process and Entity children too
    print("\n" + "=" * 80)
    print(f"PROCESS CHILDREN ANALYSIS ({len(children_map.get('Process', []))} concepts)")
    print("=" * 80)

    process_children = sorted(children_map.get('Process', []))
    process_categories = defaultdict(list)

    for child in process_children[:30]:  # Show top 30
        child_count = len(children_map.get(child, []))
        defn = definitions.get(child, 'No definition')[:80]
        process_categories[child_count].append((child, defn))

    # Sort by fan-out
    for count in sorted(process_categories.keys(), reverse=True)[:10]:
        for concept, defn in process_categories[count]:
            print(f"  {concept} ({count} children) - {defn}")

    print("\n" + "=" * 80)
    print(f"ENTITY CHILDREN ANALYSIS ({len(children_map.get('Entity', []))} concepts)")
    print("=" * 80)

    entity_children = sorted(children_map.get('Entity', []))
    entity_categories = defaultdict(list)

    for child in entity_children:
        child_count = len(children_map.get(child, []))
        defn = definitions.get(child, 'No definition')[:80]
        entity_categories[child_count].append((child, defn))

    # Sort by fan-out
    for count in sorted(entity_categories.keys(), reverse=True)[:15]:
        for concept, defn in entity_categories[count]:
            print(f"  {concept} ({count} children) - {defn}")

    # Save analysis
    output = {
        'attribute_categories': {
            cat: concepts for cat, concepts in categories.items() if len(concepts) > 0
        },
        'process_top_children': [
            {'concept': c, 'children_count': len(children_map.get(c, [])), 'definition': definitions.get(c, '')}
            for c in process_children[:50]
        ],
        'entity_top_children': [
            {'concept': c, 'children_count': len(children_map.get(c, [])), 'definition': definitions.get(c, '')}
            for c in entity_children[:50]
        ]
    }

    output_path = Path("data/concept_graph/attribute_split_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Analysis saved to {output_path}")


def main():
    parent_map, children_map, definitions = parse_kif_files()
    analyze_attribute_children(parent_map, children_map, definitions)


if __name__ == '__main__':
    main()
