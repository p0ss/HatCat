#!/usr/bin/env python3
"""
Build layer structure starting from custom safety domains.

Strategy:
1. Identify all concepts from custom KIF files
2. Build subtrees under each custom domain
3. Assign layers based on domain hierarchy (not depth)
4. Leave SUMO concepts in a separate "general knowledge" branch
"""

import re
import json
from pathlib import Path
from collections import defaultdict, deque

SUMO_SOURCE_DIR = Path("data/concept_graph/sumo_source")

# Custom domain files
CUSTOM_DOMAINS = {
    'AI.kif': 'AI Systems',
    'AIalignment.kif': 'AI Alignment',
    'AI_infrastructure.kif': 'AI Infrastructure',
    'ai_systems.kif': 'AI Systems',
    'cyber_ops.kif': 'Cyber Operations',
    'narrative_deception.kif': 'Narrative & Deception',
}

def parse_kif_files():
    """Parse all KIF files."""
    parent_map = defaultdict(set)
    children_map = defaultdict(set)
    source_map = {}
    all_concepts = set()

    for kif_file in SUMO_SOURCE_DIR.glob("*.kif"):
        with open(kif_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        subclass_pattern = r'\(subclass\s+([A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]+)\)'
        for match in re.finditer(subclass_pattern, content):
            child, parent = match.groups()
            if not child.startswith('?') and not parent.startswith('?'):
                parent_map[child].add(parent)
                children_map[parent].add(child)
                all_concepts.add(child)
                all_concepts.add(parent)

                if child not in source_map:
                    source_map[child] = kif_file.name

    return parent_map, children_map, source_map, all_concepts


def get_all_descendants(concept, children_map, visited=None):
    """Get all descendants of a concept recursively."""
    if visited is None:
        visited = set()

    if concept in visited:
        return set()

    visited.add(concept)
    descendants = set(children_map.get(concept, []))

    for child in list(descendants):
        descendants |= get_all_descendants(child, children_map, visited)

    return descendants


def build_custom_domain_hierarchy(parent_map, children_map, source_map, all_concepts):
    """Build hierarchy starting from custom domains."""

    print("=" * 80)
    print("CUSTOM DOMAIN HIERARCHY BUILDER")
    print("=" * 80)

    # Find all concepts from custom files
    custom_concepts = {c for c, src in source_map.items() if src in CUSTOM_DOMAINS}
    sumo_concepts = all_concepts - custom_concepts

    print(f"\nTotal concepts: {len(all_concepts)}")
    print(f"Custom domain concepts: {len(custom_concepts)}")
    print(f"SUMO concepts: {len(sumo_concepts)}")

    # Find root concepts in custom domains (concepts with no custom parents)
    custom_roots = set()
    for concept in custom_concepts:
        parents = parent_map.get(concept, set())
        custom_parents = parents & custom_concepts

        if not custom_parents:
            # This concept has no custom parents, it's a domain root
            custom_roots.add(concept)

    print(f"\nCustom domain roots: {len(custom_roots)}")

    # Group roots by source file
    roots_by_domain = defaultdict(list)
    for root in custom_roots:
        src = source_map.get(root, 'unknown')
        domain_name = CUSTOM_DOMAINS.get(src, src)
        roots_by_domain[domain_name].append(root)

    print("\nDomain roots by file:")
    for domain, roots in sorted(roots_by_domain.items()):
        print(f"\n  {domain} ({len(roots)} roots):")
        for root in sorted(roots)[:20]:
            child_count = len(children_map.get(root, []))
            descendants = get_all_descendants(root, children_map)
            custom_descendants = descendants & custom_concepts
            print(f"    {root}: {child_count} children, {len(custom_descendants)} custom descendants")
        if len(roots) > 20:
            print(f"    ... and {len(roots) - 20} more")

    # Propose Layer 1: Top-level custom domain categories
    # These should be the highest-level concepts in each domain
    layer1_candidates = []

    for domain, roots in sorted(roots_by_domain.items()):
        # Find roots with significant custom subtrees
        significant_roots = []
        for root in roots:
            descendants = get_all_descendants(root, children_map)
            custom_descendants = descendants & custom_concepts

            if len(custom_descendants) >= 3:  # At least 3 custom descendants
                significant_roots.append({
                    'concept': root,
                    'domain': domain,
                    'custom_descendants': len(custom_descendants),
                    'total_descendants': len(descendants),
                    'direct_children': len(children_map.get(root, []))
                })

        # Sort by custom descendants
        significant_roots.sort(key=lambda x: x['custom_descendants'], reverse=True)
        layer1_candidates.extend(significant_roots)

    print("\n" + "=" * 80)
    print(f"PROPOSED LAYER 1: CUSTOM DOMAIN ROOTS ({len(layer1_candidates)} candidates)")
    print("=" * 80)

    for candidate in sorted(layer1_candidates, key=lambda x: x['custom_descendants'], reverse=True)[:30]:
        print(f"  {candidate['concept']} ({candidate['domain']})")
        print(f"    Custom descendants: {candidate['custom_descendants']}")
        print(f"    Total descendants: {candidate['total_descendants']}")
        print(f"    Direct children: {candidate['direct_children']}")

    # Build complete domain subtrees
    domain_subtrees = {}
    for candidate in layer1_candidates:
        concept = candidate['concept']
        descendants = get_all_descendants(concept, children_map)
        custom_descendants = descendants & custom_concepts

        domain_subtrees[concept] = {
            'domain': candidate['domain'],
            'all_descendants': descendants,
            'custom_descendants': custom_descendants,
            'size': len(custom_descendants)
        }

    # Identify overlaps (concepts that appear in multiple subtrees)
    concept_membership = defaultdict(set)
    for root, info in domain_subtrees.items():
        for concept in info['custom_descendants']:
            concept_membership[concept].add(root)

    multi_parent_concepts = {c: roots for c, roots in concept_membership.items() if len(roots) > 1}

    print(f"\n\nConcepts with multiple domain parents: {len(multi_parent_concepts)}")
    for concept, roots in sorted(multi_parent_concepts.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
        print(f"  {concept}: appears in {len(roots)} subtrees - {', '.join(sorted(roots)[:3])}")

    # Save analysis
    output = {
        'summary': {
            'total_concepts': len(all_concepts),
            'custom_concepts': len(custom_concepts),
            'sumo_concepts': len(sumo_concepts),
            'custom_roots': len(custom_roots),
            'layer1_candidates': len(layer1_candidates)
        },
        'layer1_candidates': layer1_candidates,
        'domain_subtrees': {
            root: {
                'domain': info['domain'],
                'size': info['size'],
                'custom_descendants_sample': sorted(list(info['custom_descendants']))[:50]
            }
            for root, info in sorted(domain_subtrees.items(), key=lambda x: x[1]['size'], reverse=True)
        },
        'multi_parent_concepts': {
            c: list(roots) for c, roots in multi_parent_concepts.items()
        }
    }

    output_path = Path("data/concept_graph/custom_domain_hierarchy.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\n✓ Analysis saved to {output_path}")

    return output


def main():
    parent_map, children_map, source_map, all_concepts = parse_kif_files()
    build_custom_domain_hierarchy(parent_map, children_map, source_map, all_concepts)


if __name__ == '__main__':
    main()
