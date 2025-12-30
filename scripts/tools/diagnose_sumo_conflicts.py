#!/usr/bin/env python3
"""
Diagnose SUMO hierarchy conflicts and multi-parent depth issues.

Identifies:
1. Concepts with conflicting depths via different parent paths
2. Concepts with many parents (high fan-in)
3. Semantically inconsistent parent assignments
"""

import re
import json
from pathlib import Path
from collections import defaultdict, deque

# Load SUMO source
SUMO_SOURCE_DIR = Path("data/concept_graph/sumo_source")

def parse_kif_files():
    """Parse all KIF files to build parent-child maps."""
    parent_map = defaultdict(set)  # child -> {parents}
    children_map = defaultdict(set)  # parent -> {children}
    source_map = {}  # concept -> source file

    for kif_file in SUMO_SOURCE_DIR.glob("*.kif"):
        with open(kif_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        subclass_pattern = r'\(subclass\s+([A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]+)\)'
        for match in re.finditer(subclass_pattern, content):
            child, parent = match.groups()
            if child.startswith('?') or parent.startswith('?'):
                continue
            parent_map[child].add(parent)
            children_map[parent].add(child)

            # Track source file for concepts
            if child not in source_map:
                source_map[child] = kif_file.name

    return parent_map, children_map, source_map


def compute_all_paths_to_root(concept, parent_map, root='Entity', max_depth=20):
    """Find all paths from concept to root."""
    paths = []

    def dfs(current, path, visited):
        if len(path) > max_depth:
            return

        if current == root:
            paths.append(list(path))
            return

        if current in visited:
            # Circular reference
            return

        visited.add(current)

        for parent in parent_map.get(current, []):
            dfs(parent, path + [parent], visited)

        visited.remove(current)

    dfs(concept, [concept], set())
    return paths


def main():
    print("=" * 80)
    print("SUMO HIERARCHY DIAGNOSTIC")
    print("=" * 80)

    parent_map, children_map, source_map = parse_kif_files()

    print(f"\nTotal concepts: {len(set(parent_map.keys()) | set(children_map.keys()))}")
    print(f"Concepts with parents: {len(parent_map)}")
    print(f"Concepts with children: {len(children_map)}")

    # Find concepts with multiple parents
    multi_parent = {c: parents for c, parents in parent_map.items() if len(parents) > 1}
    print(f"\nConcepts with multiple parents: {len(multi_parent)}")

    # Show top 20 by parent count
    by_count = sorted(multi_parent.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    print("\nTop 20 concepts by parent count:")
    for concept, parents in by_count:
        print(f"  {concept}: {len(parents)} parents - {', '.join(sorted(parents))}")

    # Find concepts with conflicting depths
    print("\n" + "=" * 80)
    print("DEPTH CONFLICTS (different path lengths to Entity)")
    print("=" * 80)

    conflicts = []
    sample_concepts = list(multi_parent.keys())[:100]  # Sample for performance

    for concept in sample_concepts:
        paths = compute_all_paths_to_root(concept, parent_map)
        if not paths:
            continue

        depths = [len(p) - 1 for p in paths]
        if len(set(depths)) > 1:  # Multiple different depths
            min_depth = min(depths)
            max_depth = max(depths)
            diff = max_depth - min_depth

            if diff >= 3:  # Significant difference
                conflicts.append({
                    'concept': concept,
                    'min_depth': min_depth,
                    'max_depth': max_depth,
                    'diff': diff,
                    'num_paths': len(paths),
                    'source': source_map.get(concept, 'unknown')
                })

    conflicts.sort(key=lambda x: x['diff'], reverse=True)

    print(f"\nFound {len(conflicts)} concepts with significant depth conflicts (diff >= 3):\n")
    for c in conflicts[:20]:
        print(f"  {c['concept']} ({c['source']})")
        print(f"    Depth range: {c['min_depth']} to {c['max_depth']} (diff: {c['diff']})")
        print(f"    Total paths: {c['num_paths']}")

        # Show the paths
        paths = compute_all_paths_to_root(c['concept'], parent_map)
        shortest = min(paths, key=len)
        longest = max(paths, key=len)

        print(f"    Shortest: {' -> '.join(shortest)}")
        print(f"    Longest: {' -> '.join(longest)}")
        print()

    # Save diagnostic data
    output = {
        'multi_parent_concepts': len(multi_parent),
        'depth_conflicts': conflicts,
        'top_fan_in': [
            {'concept': c, 'parent_count': len(ps), 'parents': list(ps)}
            for c, ps in by_count
        ]
    }

    output_path = Path("data/concept_graph/sumo_hierarchy_diagnostic.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Diagnostic saved to {output_path}")


if __name__ == '__main__':
    main()
