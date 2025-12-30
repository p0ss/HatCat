#!/usr/bin/env python3
"""
Recalculate AI safety concept layers after hierarchy reparenting.

NEW HIERARCHY (after reparenting):
- Layer 0: Process, Proposition
- Layer 1: IntentionalProcess, FieldOfStudy, InternalChange
- Layer 2: ComputationalProcess, SocialInteraction, OrganizationalProcess, Damaging, QuantityChange
- Layer 3: AIFailureProcess, AIOptimizationProcess, Deception, PoliticalProcess, Catastrophe, RapidTransformation, AIAlignmentTheory
- Layer 4+: Specific AI safety concepts

This script:
1. Parses AI.kif to build full hierarchy
2. Calculates depth from Process/Proposition for each concept
3. Maps depth to layer (depth = layer for our system)
4. Outputs JSON entries ready for integration
"""

import json
import re
from pathlib import Path
from typing import Dict, Set, Optional
from collections import deque

def parse_ai_kif(filepath: Path) -> Dict[str, Dict]:
    """Parse AI.kif to extract parent-child relationships."""
    hierarchy = {}

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

    return hierarchy


def calculate_depth_from_root(hierarchy: Dict, concept: str, roots: Set[str]) -> Optional[int]:
    """Calculate depth from nearest root (Process or Proposition)."""
    if concept in roots:
        return 0

    if concept not in hierarchy:
        return None

    # BFS to find shortest path to any root
    visited = set()
    queue = deque([(concept, 0)])

    while queue:
        current, depth = queue.popleft()

        if current in visited:
            continue
        visited.add(current)

        if current in roots:
            return depth

        parent = hierarchy.get(current, {}).get('parent')
        if parent:
            queue.append((parent, depth + 1))

    return None


def get_all_descendants(hierarchy: Dict, parent: str) -> Set[str]:
    """Get all descendants (children, grandchildren, etc.) of a parent."""
    descendants = set()
    if parent not in hierarchy:
        return descendants

    for child in hierarchy[parent]['children']:
        descendants.add(child)
        descendants.update(get_all_descendants(hierarchy, child))

    return descendants


def is_ai_safety_concept(concept: str, hierarchy: Dict) -> bool:
    """
    Determine if a concept is an AI safety concept.
    Criteria: defined in AI.kif and not in core SUMO

    Strategy: INCLUDE everything that's AI-related, EXCLUDE only core SUMO
    """
    # Core SUMO concepts that are NOT AI safety (definitive exclusion list)
    core_sumo_exclude = {
        'Process', 'Proposition', 'Abstract', 'Attribute',
        'IntentionalProcess', 'InternalChange', 'FieldOfStudy',
        'SocialInteraction', 'OrganizationalProcess',
        'Damaging', 'QuantityChange', 'StateOfMind',
        'Agent', 'CognitiveAgent', 'TraitAttribute',
        'NormativeAttribute', 'SubjectiveAssessmentAttribute',
        'Pretending', 'EmotionalState', 'BinaryPredicate',
        'TimePoint', 'RecreationOrExercise'
    }

    # Exclude core SUMO
    if concept in core_sumo_exclude:
        return False

    # INCLUDE EVERYTHING ELSE from AI.kif
    # This includes:
    # - Anything with AI prefix
    # - New intermediate concepts (ComputationalProcess, Deception, etc.)
    # - Specific AI safety concepts (Mesa*, Instrumental*, etc.)
    # - All concepts defined in AI.kif
    return True


def main():
    print("=" * 80)
    print("RECALCULATING AI SAFETY CONCEPT LAYERS")
    print("=" * 80)
    print()

    # Parse full SUMO hierarchy (Merge.kif + AI.kif)
    merge_kif = Path("data/concept_graph/sumo_source/Merge.kif")
    ai_kif = Path("data/concept_graph/sumo_source/AI.kif")

    print(f"Parsing {merge_kif}...")
    hierarchy = parse_ai_kif(merge_kif)
    merge_concepts = len(hierarchy)
    print(f"✓ Found {merge_concepts} SUMO concepts")

    print(f"Parsing {ai_kif}...")
    ai_hierarchy = parse_ai_kif(ai_kif)
    # Merge AI.kif into hierarchy
    for concept, info in ai_hierarchy.items():
        if concept not in hierarchy:
            hierarchy[concept] = info
        else:
            # Update with AI.kif information (overwrites SUMO if present)
            hierarchy[concept]['parent'] = info['parent'] or hierarchy[concept]['parent']
            hierarchy[concept]['children'].update(info['children'])

    print(f"✓ Found {len(hierarchy)} total concepts ({len(ai_hierarchy)} from AI.kif)")
    print()

    # Calculate depths for AI.kif concepts only
    roots = {'Process', 'Proposition'}
    print("Calculating depths from roots (Process, Proposition)...")

    ai_concepts = set(ai_hierarchy.keys())
    concept_layers = {}

    for concept in ai_concepts:
        if not is_ai_safety_concept(concept, hierarchy):
            continue

        depth = calculate_depth_from_root(hierarchy, concept, roots)
        if depth is not None:
            # Layer = depth (Process/Proposition = Layer 0)
            layer = depth
            # Filter children to only include AI.kif concepts
            all_children = hierarchy[concept]['children']
            ai_children = [c for c in all_children if c in ai_concepts]

            concept_layers[concept] = {
                'concept': concept,
                'layer': layer,
                'depth': depth,
                'parent': hierarchy[concept]['parent'],
                'children': sorted(ai_children),
                'is_category': len(ai_children) > 0
            }

    print(f"✓ Found {len(concept_layers)} AI safety concepts")
    print()

    # Group by layer
    by_layer = {}
    for concept, info in concept_layers.items():
        layer = info['layer']
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(info)

    # Display summary
    print("Layer distribution:")
    for layer in sorted(by_layer.keys()):
        concepts = by_layer[layer]
        print(f"  Layer {layer}: {len(concepts)} concepts")
        for c in sorted(concepts, key=lambda x: x['concept']):
            children_str = f" ({len(c['children'])} children)" if c['is_category'] else ""
            print(f"    - {c['concept']} ← {c['parent']}{children_str}")
    print()

    # Save output
    output_file = Path("results/ai_safety_recalculated_layers.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    output = {
        'by_layer': {str(k): v for k, v in by_layer.items()},
        'by_concept': concept_layers,
        'summary': {
            'total_concepts': len(concept_layers),
            'layer_distribution': {str(k): len(v) for k, v in by_layer.items()}
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Saved layer assignments to {output_file}")
    print()

    # Validation checks
    print("=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    # Check 1: No Layer 0-1 AI safety concepts (except new intermediates)
    layer01_concepts = []
    if 0 in by_layer:
        layer01_concepts.extend(by_layer[0])
    if 1 in by_layer:
        layer01_concepts.extend(by_layer[1])

    if layer01_concepts:
        print("❌ FAIL: Found AI safety concepts in Layer 0-1:")
        for c in layer01_concepts:
            print(f"     {c['concept']} (Layer {c['layer']})")
    else:
        print("✓ PASS: No AI safety concepts in Layer 0-1")

    # Check 2: Layer 2 should have only new intermediate concepts
    if 2 in by_layer:
        layer2_non_intermediate = [c for c in by_layer[2]
                                   if c['concept'] not in {'ComputationalProcess'}]
        if layer2_non_intermediate:
            print("⚠ WARNING: Layer 2 has non-intermediate concepts:")
            for c in layer2_non_intermediate:
                print(f"     {c['concept']}")
        else:
            print("✓ PASS: Layer 2 has only intermediate concepts")

    # Check 3: All leaf concepts should be Layer 4+
    leaf_concepts_wrong_layer = []
    for concept, info in concept_layers.items():
        if not info['is_category'] and info['layer'] < 4:
            leaf_concepts_wrong_layer.append((concept, info['layer']))

    if leaf_concepts_wrong_layer:
        print("⚠ WARNING: Leaf concepts found before Layer 4:")
        for concept, layer in leaf_concepts_wrong_layer:
            print(f"     {concept} (Layer {layer})")
    else:
        print("✓ PASS: All leaf concepts are Layer 4+")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review results/ai_safety_recalculated_layers.json")
    print("2. Update abstraction_layers/layer*.json files with new assignments")
    print("3. Remove old AI safety entries from wrong layers")
    print("4. Add new entries to correct layers")


if __name__ == '__main__':
    main()
