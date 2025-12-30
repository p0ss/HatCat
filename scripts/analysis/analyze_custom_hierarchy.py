#!/usr/bin/env python3
"""
Analyze custom concept hierarchy and assign layers based on parent chains.

This script:
1. Loads V4 layer assignments for SUMO concepts
2. Parses all custom KIF files to extract parent-child relationships
3. Recursively assigns layers based on parent depth
4. Identifies concepts with undefined depth (missing/circular parents)
5. Generates comprehensive layer assignment mapping
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple


def load_v4_layers() -> Dict[str, int]:
    """Load V4 layer assignments for all SUMO concepts."""
    v4_dir = Path("data/concept_graph/abstraction_layers_v4")
    concept_layers = {}

    for layer_file in sorted(v4_dir.glob("layer*.json")):
        with open(layer_file) as f:
            data = json.load(f)
            layer_num = data["metadata"]["layer"]
            for concept_entry in data["concepts"]:
                concept_layers[concept_entry["sumo_term"]] = layer_num

    print(f"Loaded {len(concept_layers)} V4 concepts across layers 0-6")
    return concept_layers


def parse_kif_files(custom_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse all KIF files to extract parent-child relationships.

    Returns:
        (concept_parents, concept_files) where:
        - concept_parents: {child: parent}
        - concept_files: {concept: source_file}
    """
    concept_parents = {}
    concept_files = {}

    kif_files = sorted(custom_dir.glob("*.kif"))
    print(f"\nParsing {len(kif_files)} KIF files...")

    for kif_file in kif_files:
        with open(kif_file) as f:
            content = f.read()

        # Extract (subclass Child Parent) declarations
        # Handle both formats: (subclass Foo Bar) and (subclass Foo BarBaz)
        subclass_pattern = r'\(subclass\s+(\S+)\s+(\S+)\s*\)'
        matches = re.findall(subclass_pattern, content)

        for child, parent in matches:
            concept_parents[child] = parent
            concept_files[child] = kif_file.name

    print(f"Extracted {len(concept_parents)} parent-child relationships")
    return concept_parents, concept_files


def assign_layer(
    concept: str,
    v4_layers: Dict[str, int],
    concept_parents: Dict[str, str],
    visited: Set[str] = None,
    depth_cache: Dict[str, Optional[int]] = None
) -> Optional[int]:
    """
    Recursively assign layer to a concept based on parent chain.

    Returns:
        Layer number (0-6) or None if undefined
    """
    if visited is None:
        visited = set()
    if depth_cache is None:
        depth_cache = {}

    # Check cache first
    if concept in depth_cache:
        return depth_cache[concept]

    # 1. If concept exists in V4, return its layer
    if concept in v4_layers:
        return v4_layers[concept]

    # 2. Check for circular reference
    if concept in visited:
        return None  # Circular reference detected

    visited.add(concept)

    # 3. Find parent in custom hierarchy
    parent = concept_parents.get(concept)
    if not parent:
        # No parent found - this is an issue
        depth_cache[concept] = None
        return None

    # 4. Recursively get parent layer
    parent_layer = assign_layer(parent, v4_layers, concept_parents, visited, depth_cache)

    # 5. Child is one layer deeper than parent
    if parent_layer is not None:
        child_layer = parent_layer + 1
        # Cap at layer 6 (though we'll note overflow)
        depth_cache[concept] = min(child_layer, 6)
        return depth_cache[concept]
    else:
        depth_cache[concept] = None
        return None


def analyze_layer_distribution(
    concept_parents: Dict[str, str],
    concept_files: Dict[str, str],
    v4_layers: Dict[str, int]
) -> Dict[str, Dict]:
    """
    Analyze layer distribution for all custom concepts.

    Returns:
        {concept: {layer, parent, source_file, depth_path}}
    """
    layer_assignments = {}
    undefined_concepts = []
    layer_distribution = defaultdict(list)

    print("\nAssigning layers to custom concepts...")

    for concept in concept_parents:
        layer = assign_layer(concept, v4_layers, concept_parents)

        parent = concept_parents.get(concept, "UNKNOWN")
        source_file = concept_files.get(concept, "UNKNOWN")

        if layer is not None:
            layer_assignments[concept] = {
                "layer": layer,
                "parent": parent,
                "source_file": source_file
            }
            layer_distribution[layer].append(concept)
        else:
            undefined_concepts.append((concept, parent, source_file))

    return layer_assignments, layer_distribution, undefined_concepts


def trace_parent_chain(
    concept: str,
    concept_parents: Dict[str, str],
    v4_layers: Dict[str, int],
    max_depth: int = 20
) -> List[Tuple[str, Optional[int]]]:
    """
    Trace parent chain for a concept up to a V4 root or max depth.

    Returns:
        List of (concept, layer) tuples showing the chain
    """
    chain = []
    current = concept
    seen = set()

    for _ in range(max_depth):
        if current in seen:
            chain.append((current, "CIRCULAR"))
            break

        seen.add(current)
        layer = v4_layers.get(current)
        chain.append((current, layer))

        if layer is not None:
            # Reached V4 concept
            break

        parent = concept_parents.get(current)
        if not parent:
            chain.append(("NO_PARENT", None))
            break

        current = parent

    return chain


def generate_undefined_report(
    undefined_concepts: List[Tuple[str, str, str]],
    concept_parents: Dict[str, str],
    v4_layers: Dict[str, int]
) -> str:
    """Generate detailed report of undefined concepts with parent chains."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("UNDEFINED CONCEPT ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append(f"\nTotal undefined concepts: {len(undefined_concepts)}\n")

    # Group by source file
    by_file = defaultdict(list)
    for concept, parent, source_file in undefined_concepts:
        by_file[source_file].append((concept, parent))

    for source_file in sorted(by_file.keys()):
        concepts_in_file = by_file[source_file]
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"File: {source_file} ({len(concepts_in_file)} undefined)")
        report_lines.append(f"{'='*80}\n")

        for concept, parent in sorted(concepts_in_file):
            chain = trace_parent_chain(concept, concept_parents, v4_layers)
            report_lines.append(f"Concept: {concept}")
            report_lines.append(f"  Immediate parent: {parent}")
            report_lines.append(f"  Parent chain:")
            for i, (node, layer) in enumerate(chain):
                indent = "    " * (i + 1)
                if isinstance(layer, str):
                    report_lines.append(f"{indent}└─ {node} [{layer}]")
                elif layer is not None:
                    report_lines.append(f"{indent}└─ {node} [V4 Layer {layer}] ✓")
                else:
                    report_lines.append(f"{indent}└─ {node} [NOT IN V4]")
            report_lines.append("")

    return "\n".join(report_lines)


def main():
    # Load V4 layers
    v4_layers = load_v4_layers()

    # Parse custom KIF files
    custom_dir = Path("data/concept_graph/custom_concepts")
    concept_parents, concept_files = parse_kif_files(custom_dir)

    # Analyze layer assignments
    layer_assignments, layer_distribution, undefined_concepts = analyze_layer_distribution(
        concept_parents, concept_files, v4_layers
    )

    # Print distribution summary
    print("\n" + "="*80)
    print("LAYER DISTRIBUTION SUMMARY")
    print("="*80)
    total_assigned = sum(len(concepts) for concepts in layer_distribution.values())

    for layer in range(7):
        concepts_in_layer = layer_distribution.get(layer, [])
        count = len(concepts_in_layer)
        pct = (count / total_assigned * 100) if total_assigned > 0 else 0
        print(f"Layer {layer}: {count:4d} concepts ({pct:5.1f}%)")

    print(f"\nTotal assigned: {total_assigned}")
    print(f"Undefined:      {len(undefined_concepts)}")
    print(f"Grand total:    {len(concept_parents)}")

    # Save layer assignments
    output_file = Path("data/concept_graph/custom_concept_layers.json")
    with open(output_file, 'w') as f:
        json.dump(layer_assignments, f, indent=2, sort_keys=True)
    print(f"\n✓ Saved layer assignments to {output_file}")

    # Generate undefined concepts report
    if undefined_concepts:
        report = generate_undefined_report(undefined_concepts, concept_parents, v4_layers)
        report_file = Path("data/concept_graph/custom_concepts_undefined.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✓ Saved undefined concepts report to {report_file}")

        print(f"\n⚠  WARNING: {len(undefined_concepts)} concepts have undefined depth")
        print("   See custom_concepts_undefined.txt for details")

    # Check for layer 5-6 overflow
    overflow = len(layer_distribution.get(5, [])) + len(layer_distribution.get(6, []))
    if overflow > 0:
        print(f"\n⚠  NOTE: {overflow} concepts assigned to layers 5-6 (empty in V4)")
        print("   Consider capping at layer 4 or restructuring hierarchy")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review custom_concepts_undefined.txt to identify parent issues")
    print("2. Fix bridge.kif parent assignments as needed")
    print("3. Update V4 builder to integrate custom_concept_layers.json")
    print("4. Generate synsets for custom concepts")
    print("5. Build V4.5 and train lenses")


if __name__ == "__main__":
    main()
