#!/usr/bin/env python3
"""
Build V4 layer JSON files from natural knowledge map.

Uses the domain-based natural knowledge map to create semantically coherent
layer files with proper metadata, WordNet synsets, and parent-child relationships.
"""

import re
import json
from pathlib import Path
from collections import defaultdict

SUMO_SOURCE_DIR = Path("data/concept_graph/sumo_source")
NATURAL_MAP_PATH = Path("data/concept_graph/natural_knowledge_map_v2.json")
WORDNET_DIR = Path("data/concept_graph")
OUTPUT_DIR = Path("data/concept_graph/v4")

def parse_kif_files():
    """Parse all KIF files for relationships and definitions."""
    parent_map = defaultdict(set)
    children_map = defaultdict(set)
    definitions = {}
    all_concepts = set()

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
                all_concepts.add(child)
                all_concepts.add(parent)

        # Parse documentation
        doc_pattern = r'\(documentation\s+([A-Za-z0-9_-]+)\s+EnglishLanguage\s+"([^"]+)"'
        for match in re.finditer(doc_pattern, content):
            concept, definition = match.groups()
            definitions[concept] = definition

    return parent_map, children_map, definitions, all_concepts


def load_wordnet_mappings():
    """Load WordNet synset mappings."""
    # Load SUMO → WordNet mapping
    sumo_wordnet_file = WORDNET_DIR / "sumo_to_wordnet.json"

    if sumo_wordnet_file.exists():
        with open(sumo_wordnet_file) as f:
            synset_map = json.load(f)
        print(f"  Loaded SUMO→WordNet mapping: {len(synset_map)} concepts")
        return synset_map

    # Fallback to old mapping (shouldn't happen)
    print("  WARNING: sumo_to_wordnet.json not found, using fallback")
    return {}


def build_v4_layer_files(parent_map, children_map, definitions, synset_map):
    """Build V4 layer files from natural knowledge map."""

    print("=" * 80)
    print("V4 LAYER FILE BUILDER")
    print("=" * 80)

    # Load natural knowledge map
    with open(NATURAL_MAP_PATH) as f:
        natural_map = json.load(f)

    domains = natural_map['domains']
    layer_assignment = natural_map['layer_assignment']
    concept_domain = natural_map['concept_domain']

    print(f"\nLoaded natural knowledge map:")
    print(f"  Total concepts: {len(layer_assignment)}")
    print(f"  Domains: {len(domains)}")
    print(f"  WordNet mappings: {len(synset_map)}")

    # Layer 0: Special handling - these are meta-domains, not SUMO concepts
    layer0_concepts = [
        {
            "sumo_term": "MindsAndAgents",
            "layer": 0,
            "domain": "MindsAndAgents",
            "is_category_lens": True,
            "is_pseudo_sumo": True,
            "category_children": sorted([c for c, l in layer_assignment.items()
                                        if l == 1 and concept_domain.get(c) == "MindsAndAgents"])[:15],
            "definition": "Knowledge domain encompassing cognition, agency, social structures, communication, and mental processes",
            "concept_count": len(domains.get("MindsAndAgents", [])),
        },
        {
            "sumo_term": "CreatedThings",
            "layer": 0,
            "domain": "CreatedThings",
            "is_category_lens": True,
            "is_pseudo_sumo": True,
            "category_children": sorted([c for c, l in layer_assignment.items()
                                        if l == 1 and concept_domain.get(c) == "CreatedThings"])[:15],
            "definition": "Knowledge domain encompassing artifacts, technology, systems, tools, and human-made constructs",
            "concept_count": len(domains.get("CreatedThings", [])),
        },
        {
            "sumo_term": "PhysicalWorld",
            "layer": 0,
            "domain": "PhysicalWorld",
            "is_category_lens": True,
            "is_pseudo_sumo": True,
            "category_children": sorted([c for c, l in layer_assignment.items()
                                        if l == 1 and concept_domain.get(c) == "PhysicalWorld"])[:15],
            "definition": "Knowledge domain encompassing matter, energy, forces, physical quantities, and natural phenomena",
            "concept_count": len(domains.get("PhysicalWorld", [])),
        },
        {
            "sumo_term": "LivingThings",
            "layer": 0,
            "domain": "LivingThings",
            "is_category_lens": True,
            "is_pseudo_sumo": True,
            "category_children": sorted([c for c, l in layer_assignment.items()
                                        if l == 1 and concept_domain.get(c) == "LivingThings"])[:15],
            "definition": "Knowledge domain encompassing organisms, biological processes, ecosystems, and life",
            "concept_count": len(domains.get("LivingThings", [])),
        },
        {
            "sumo_term": "Information",
            "layer": 0,
            "domain": "Information",
            "is_category_lens": True,
            "is_pseudo_sumo": True,
            "category_children": sorted([c for c, l in layer_assignment.items()
                                        if l == 1 and concept_domain.get(c) == "Information"])[:15],
            "definition": "Knowledge domain encompassing data, representations, propositions, relations, and abstract entities",
            "concept_count": len(domains.get("Information", [])),
        },
    ]

    # Build layers 1-4
    layer_files = {
        0: {"layer": 0, "concepts": layer0_concepts},
        1: {"layer": 1, "concepts": []},
        2: {"layer": 2, "concepts": []},
        3: {"layer": 3, "concepts": []},
        4: {"layer": 4, "concepts": []},
    }

    # Process each concept in layers 1-4
    for concept, layer_num in sorted(layer_assignment.items()):
        if layer_num == 0 or layer_num > 4:
            continue

        domain = concept_domain.get(concept, "Unknown")
        parents = sorted(list(parent_map.get(concept, set())))
        # Get all children from KIF - no truncation (will be filtered by layer later)
        category_children = sorted(list(children_map.get(concept, set())))

        # Get WordNet synsets if available
        synsets = []
        canonical_synset = None
        lemmas = []
        pos = None
        wn_definition = None
        lexname = None

        if concept in synset_map:
            synset_data = synset_map[concept]
            if isinstance(synset_data, dict):
                synsets = synset_data.get('synsets', [])
                canonical_synset = synset_data.get('canonical_synset')
                lemmas = synset_data.get('lemmas', [])
                pos = synset_data.get('pos')
                wn_definition = synset_data.get('definition')
                lexname = synset_data.get('lexname')

        # Build concept entry
        concept_entry = {
            "sumo_term": concept,
            "layer": layer_num,
            "domain": domain,
            "is_category_lens": True,
            "is_pseudo_sumo": False,
            "parent_concepts": parents,
            "category_children": category_children,
            "child_count": len(children_map.get(concept, set())),
        }

        # Add SUMO definition if available
        if concept in definitions:
            concept_entry["sumo_definition"] = definitions[concept]

        # Add WordNet data if available
        if synsets:
            concept_entry["synset_count"] = len(synsets)
            concept_entry["synsets"] = synsets[:50]  # Limit to top 50
            concept_entry["canonical_synset"] = canonical_synset

        if lemmas:
            concept_entry["lemmas"] = lemmas
        if pos:
            concept_entry["pos"] = pos
        if wn_definition:
            concept_entry["definition"] = wn_definition
        if lexname:
            concept_entry["lexname"] = lexname

        layer_files[layer_num]["concepts"].append(concept_entry)

    # Add summary metadata to each layer
    for layer_num in range(5):
        concepts = layer_files[layer_num]["concepts"]
        layer_files[layer_num]["summary"] = {
            "layer": layer_num,
            "total_concepts": len(concepts),
            "domain_distribution": {
                domain: len([c for c in concepts if c.get("domain") == domain])
                for domain in ["MindsAndAgents", "CreatedThings", "PhysicalWorld", "LivingThings", "Information"]
            },
            "with_wordnet": len([c for c in concepts if c.get("synsets")]),
            "with_definition": len([c for c in concepts if c.get("definition") or c.get("sumo_definition")]),
        }

    # =========================================================================
    # VALIDATION & REPAIR: Ensure bidirectional parent-child links
    # =========================================================================
    print("\n" + "=" * 80)
    print("VALIDATING PARENT-CHILD RELATIONSHIPS")
    print("=" * 80)

    # Build concept lookup by name
    concept_lookup = {}  # name -> (layer_num, concept_entry)
    for layer_num, layer_data in layer_files.items():
        for concept_entry in layer_data["concepts"]:
            concept_lookup[concept_entry["sumo_term"]] = (layer_num, concept_entry)

    # Pass 1: Find concepts not in any category_children
    orphan_count = 0
    fixed_count = 0
    unfixable = []

    for name, (layer_num, concept_entry) in concept_lookup.items():
        if layer_num == 0:
            continue

        # Check if this concept is in anyone's category_children
        is_child = False
        for other_name, (other_layer, other_entry) in concept_lookup.items():
            if name in other_entry.get("category_children", []):
                is_child = True
                break

        if is_child:
            continue

        orphan_count += 1

        # Try to fix: find a parent in a lower layer first, then same layer
        declared_parents = concept_entry.get("parent_concepts", [])

        # Priority 1: Lower layer parent
        lower_parents = [(p, concept_lookup[p][0]) for p in declared_parents
                         if p in concept_lookup and concept_lookup[p][0] < layer_num]

        # Priority 2: Same layer parent (if lower not available)
        same_parents = [(p, concept_lookup[p][0]) for p in declared_parents
                        if p in concept_lookup and concept_lookup[p][0] == layer_num]

        valid_parents = lower_parents if lower_parents else same_parents

        if valid_parents:
            # Pick closest parent (highest layer number for lower, first for same)
            best_parent, parent_layer = max(valid_parents, key=lambda x: x[1])
            parent_entry = concept_lookup[best_parent][1]

            # Add to parent's category_children
            if "category_children" not in parent_entry:
                parent_entry["category_children"] = []
            if name not in parent_entry["category_children"]:
                parent_entry["category_children"].append(name)
                parent_entry["category_children"].sort()
                fixed_count += 1
        else:
            unfixable.append((name, layer_num, declared_parents))

    print(f"\n  Orphans found: {orphan_count}")
    print(f"  Orphans fixed: {fixed_count}")
    print(f"  Unfixable (no lower-layer parent in pack): {len(unfixable)}")

    if unfixable:
        print(f"\n  WARNING: {len(unfixable)} concepts have no reachable parent:")
        for name, layer, parents in unfixable[:10]:
            print(f"    L{layer} {name} -> declares: {parents}")
        if len(unfixable) > 10:
            print(f"    ... and {len(unfixable) - 10} more")

    # Pass 2: Iteratively fix remaining unreachable (same-layer parent chains)
    iteration = 0
    while True:
        iteration += 1
        reachable = set()
        for entry in layer_files[0]["concepts"]:
            reachable.add(entry["sumo_term"])

        # Propagate downward through category_children
        changed = True
        while changed:
            changed = False
            for name in list(reachable):
                if name not in concept_lookup:
                    continue
                entry = concept_lookup[name][1]
                for child in entry.get("category_children", []):
                    if child in concept_lookup and child not in reachable:
                        reachable.add(child)
                        changed = True

        unreachable = [name for name in concept_lookup if name not in reachable]
        if not unreachable:
            break

        # Try to fix unreachable by adding to reachable parent's category_children
        fixed_this_round = 0
        for name in unreachable:
            layer_num, concept_entry = concept_lookup[name]
            declared_parents = concept_entry.get("parent_concepts", [])

            # Find reachable parent (any layer)
            reachable_parents = [p for p in declared_parents
                                 if p in concept_lookup and p in reachable]

            if reachable_parents:
                best_parent = reachable_parents[0]
                parent_entry = concept_lookup[best_parent][1]

                if "category_children" not in parent_entry:
                    parent_entry["category_children"] = []
                if name not in parent_entry["category_children"]:
                    parent_entry["category_children"].append(name)
                    parent_entry["category_children"].sort()
                    fixed_this_round += 1

        if fixed_this_round == 0:
            break

        print(f"  Iteration {iteration}: fixed {fixed_this_round} more unreachable")

    # Final count
    unreachable = [name for name in concept_lookup if name not in reachable]
    if unreachable:
        print(f"\n  ⚠️  {len(unreachable)} concepts NOT reachable from layer 0 via category_children")
    else:
        print(f"\n  ✓ All {len(concept_lookup)} concepts reachable from layer 0")

    # Save layer files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for layer_num, layer_data in layer_files.items():
        output_path = OUTPUT_DIR / f"layer{layer_num}.json"
        with open(output_path, 'w') as f:
            json.dump(layer_data, f, indent=2)

        summary = layer_data["summary"]
        print(f"\n✓ Layer {layer_num} saved to {output_path}")
        print(f"  Concepts: {summary['total_concepts']}")
        print(f"  With WordNet: {summary['with_wordnet']}")
        print(f"  Domain distribution:")
        for domain, count in sorted(summary['domain_distribution'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"    {domain}: {count}")

    print(f"\n{'=' * 80}")
    print(f"V4 LAYER FILES COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nTotal concepts across all layers:")
    total = sum(layer_data["summary"]["total_concepts"] for layer_data in layer_files.values())
    print(f"  {total} concepts")
    print(f"\nLayer distribution (pyramid structure):")
    for layer_num in range(5):
        count = layer_files[layer_num]["summary"]["total_concepts"]
        pct = 100 * count / total if total > 0 else 0
        print(f"  Layer {layer_num}: {count:5} concepts ({pct:5.1f}%)")


def main():
    print("Parsing KIF files...")
    parent_map, children_map, definitions, all_concepts = parse_kif_files()

    print("Loading WordNet mappings...")
    synset_map = load_wordnet_mappings()

    print("\nBuilding V4 layer files...")
    build_v4_layer_files(parent_map, children_map, definitions, synset_map)


if __name__ == '__main__':
    main()
