#!/usr/bin/env python3
"""
Build V4 concept layers from comprehensive SUMO KIF files.

This script:
1. Parses all SUMO KIF files in data/concept_graph/sumo_source/
2. Applies patches from data/concept_graph/sumo_patches/
3. Builds complete concept hierarchy
4. Maps concepts to WordNet synsets
5. Assigns concepts to 7 layers (0-6) based on abstraction level
6. Generates layer JSON files in v4_format

V4 includes comprehensive coverage:
- Military, Government, Law, Justice
- Finance, Economy, Industries (NAICS)
- Social Media, Communications, Computing
- Transportation, Vehicles, Weapons
- Biology, Climate, Weather
- Music, Sports, Hotels, Dining
- Privacy (GDPR), Capabilities
"""

import re
import json
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import nltk
from nltk.corpus import wordnet as wn

# Download WordNet if needed
try:
    wn.synsets('test')
except LookupError:
    print("Downloading WordNet...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# =============================================================================
# Configuration
# =============================================================================
SUMO_SOURCE_DIR = Path("data/concept_graph/sumo_source")
PATCH_DIR = Path("data/concept_graph/sumo_patches")
OUTPUT_DIR = Path("data/concept_graph/abstraction_layers_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Layer thresholds designed for pyramid structure (Principle of Choices)
# Target: ~10 concepts at layer 0, ~100 at layer 1, ~1000 at layer 2, rest deeper
# Based on empirical depth distribution analysis
LAYER_THRESHOLDS = {
    0: (0, 1),     # IGNORED (whitelist only) - Core ontological roots
    1: (1, 2),     # ~100-250: Major categories (children of layer 0, not in whitelist)
    2: (3, 5),     # ~1000-1500: Mid-level categories
    3: (6, 7),     # ~2000-3000: Specific concepts
    4: (8, 999),   # Rest: Fine-grained & leaf concepts
}

# Layer 6 will be for WordNet-only concepts (no SUMO mapping)


# =============================================================================
# Step 1: Parse KIF Files
# =============================================================================
def parse_kif_file(kif_path):
    """Extract subclass and instance relationships from a KIF file."""
    relationships = []

    with open(kif_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Match (subclass Child Parent)
    subclass_pattern = r'\(subclass\s+([A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]+)\)'
    for match in re.finditer(subclass_pattern, content):
        child, parent = match.groups()
        # Skip variables
        if child.startswith('?') or parent.startswith('?'):
            continue
        relationships.append(('subclass', child, parent))

    # Match (instance Individual Class)
    instance_pattern = r'\(instance\s+([A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]+)\)'
    for match in re.finditer(instance_pattern, content):
        individual, cls = match.groups()
        if individual.startswith('?') or cls.startswith('?'):
            continue
        relationships.append(('instance', individual, cls))

    return relationships


def parse_all_kif_files():
    """Parse all KIF files in the source directory."""
    print("=" * 80)
    print("STEP 1: PARSING SUMO KIF FILES")
    print("=" * 80)

    all_relationships = []
    file_stats = {}

    kif_files = sorted(SUMO_SOURCE_DIR.glob('*.kif'))
    print(f"\nFound {len(kif_files)} KIF files\n")

    for kif_file in kif_files:
        print(f"  Parsing {kif_file.name}...", end=" ")
        rels = parse_kif_file(kif_file)
        all_relationships.extend(rels)
        file_stats[kif_file.name] = len(rels)
        print(f"✓ {len(rels)} relationships")

    print(f"\n✓ Total relationships parsed: {len(all_relationships)}")

    return all_relationships, file_stats


# =============================================================================
# Step 2: Apply Patches
# =============================================================================
def parse_patch_file(patch_path):
    """Parse a patch KIF file to extract parent overrides."""
    overrides = {}

    with open(patch_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Match (subclass Child Parent)
    subclass_pattern = r'\(subclass\s+([A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]+)\)'
    for match in re.finditer(subclass_pattern, content):
        child, parent = match.groups()
        overrides[child] = parent

    return overrides


def apply_patches(relationships):
    """Apply patch files to override SUMO parent relationships."""
    print("\n" + "=" * 80)
    print("STEP 2: APPLYING PATCHES")
    print("=" * 80)

    # Load all patches
    all_overrides = {}
    patch_files = sorted(PATCH_DIR.glob('*.patch.kif'))

    print(f"\nFound {len(patch_files)} patch files\n")

    for patch_file in patch_files:
        print(f"  Loading {patch_file.name}...", end=" ")
        overrides = parse_patch_file(patch_file)
        all_overrides.update(overrides)
        print(f"✓ {len(overrides)} overrides")

    print(f"\n✓ Total overrides loaded: {len(all_overrides)}")

    # Apply overrides (replace parent relationships)
    patched_relationships = []
    override_count = 0

    for rel_type, child, parent in relationships:
        if rel_type == 'subclass' and child in all_overrides:
            # Replace with patched parent
            patched_relationships.append((rel_type, child, all_overrides[child]))
            override_count += 1
        else:
            patched_relationships.append((rel_type, child, parent))

    print(f"\n✓ Applied {override_count} parent overrides")

    return patched_relationships


# =============================================================================
# Step 3: Build Hierarchy
# =============================================================================
def build_hierarchy(relationships):
    """Build parent-child maps and compute depths."""
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING HIERARCHY")
    print("=" * 80)

    # Build maps
    parent_map = defaultdict(set)  # child -> {parents}
    children_map = defaultdict(set)  # parent -> {children}
    all_concepts = set()

    for rel_type, child, parent in relationships:
        if rel_type == 'subclass':
            parent_map[child].add(parent)
            children_map[parent].add(child)
            all_concepts.add(child)
            all_concepts.add(parent)

    print(f"\n✓ Total SUMO concepts: {len(all_concepts)}")
    print(f"✓ Concepts with parents: {len(parent_map)}")
    print(f"✓ Concepts with children: {len(children_map)}")

    # Find root concepts (no parents)
    roots = [c for c in all_concepts if c not in parent_map]
    print(f"\n✓ Root concepts: {len(roots)}")
    for root in sorted(roots)[:10]:
        child_count = len(children_map.get(root, []))
        print(f"  - {root} ({child_count} children)")
    if len(roots) > 10:
        print(f"  ... and {len(roots) - 10} more")

    # Compute depths from Entity (or first root)
    root = next((r for r in roots if r == 'Entity'), roots[0] if roots else None)
    print(f"\n✓ Using '{root}' as root for depth calculation")

    depths = compute_depths_bfs(parent_map, root)
    max_depth = max(depths.values())
    print(f"✓ Maximum depth: {max_depth}")

    return parent_map, children_map, depths, root


def compute_depths_bfs(parent_map, root):
    """Compute minimum distance from root using BFS going upward."""
    depths = {root: 0}
    queue = deque([root])

    while queue:
        current = queue.popleft()
        current_depth = depths[current]

        # Find all children (concepts that have current as parent)
        for child, parents in parent_map.items():
            if current in parents and child not in depths:
                depths[child] = current_depth + 1
                queue.append(child)

    return depths


# =============================================================================
# Step 4: Map to WordNet
# =============================================================================
def map_to_wordnet(all_concepts):
    """Map SUMO concepts to WordNet synsets."""
    print("\n" + "=" * 80)
    print("STEP 4: MAPPING TO WORDNET")
    print("=" * 80)

    concept_to_synsets = defaultdict(list)
    mapped_count = 0

    print("\nSearching WordNet for SUMO concepts...")

    for i, concept in enumerate(sorted(all_concepts)):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(all_concepts)}...")

        # Try direct match (case-insensitive)
        concept_lower = concept.lower()
        synsets = wn.synsets(concept_lower)

        if synsets:
            concept_to_synsets[concept] = [s.name() for s in synsets]
            mapped_count += 1
        else:
            # Try splitting camel case
            words = re.findall(r'[A-Z][a-z]*', concept)
            if words:
                search_term = '_'.join(words).lower()
                synsets = wn.synsets(search_term)
                if synsets:
                    concept_to_synsets[concept] = [s.name() for s in synsets]
                    mapped_count += 1

    print(f"\n✓ Mapped {mapped_count}/{len(all_concepts)} concepts to WordNet")
    print(f"  ({100 * mapped_count / len(all_concepts):.1f}% coverage)")

    return concept_to_synsets


# =============================================================================
# Step 5: Assign Layers
# =============================================================================
def assign_layers(depths, concept_to_synsets):
    """Assign concepts to layers based on depth with pyramid structure."""
    print("\n" + "=" * 80)
    print("STEP 5: ASSIGNING LAYERS")
    print("=" * 80)

    layers = {i: [] for i in range(7)}

    # Core fundamental SUMO concepts for layer 0 (handpicked)
    # These are the absolute top-level ontological categories
    LAYER_0_CORE = {
        'Entity',      # Root of all existence
        'Abstract',    # Non-physical entities
        'Physical',    # Physical entities
        'Attribute',   # Properties and qualities
        'Relation',    # Relationships between entities
        'Process',     # Events and actions
        'Object',      # Physical objects
        'Continuant',  # Things that persist through time
        'Occurrent',   # Things that occur/happen
    }

    for concept, depth in depths.items():
        # Override: Force core concepts to layer 0 ONLY
        if concept in LAYER_0_CORE:
            assigned_layer = 0
        else:
            # Assign non-core concepts to layers 1-6 based on depth thresholds
            # Note: layer 0 threshold is IGNORED for non-whitelisted concepts
            assigned_layer = 6  # Default to layer 6
            for layer_num, (min_depth, max_depth) in LAYER_THRESHOLDS.items():
                if layer_num == 0:
                    # Skip layer 0 threshold - only whitelist goes to layer 0
                    continue
                if min_depth <= depth <= max_depth:
                    assigned_layer = layer_num
                    break

        layers[assigned_layer].append(concept)

    print("\nLayer distribution:")
    for layer_num in range(7):
        count = len(layers[layer_num])
        pct = 100 * count / len(depths)
        print(f"  Layer {layer_num}: {count:5} concepts ({pct:5.1f}%)")

    return layers


# =============================================================================
# Step 6: Generate Layer JSON Files
# =============================================================================
def generate_layer_files(layers, depths, parent_map, children_map, concept_to_synsets, root):
    """Generate layer JSON files in v4 format."""
    print("\n" + "=" * 80)
    print("STEP 6: GENERATING LAYER FILES")
    print("=" * 80)

    layer_descriptions = {
        0: "SUMO depth 0-2: Top-level ontological categories",
        1: "SUMO depth 3-5: High-level categories",
        2: "SUMO depth 6-8: Mid-level concepts",
        3: "SUMO depth 9-11: Specific concepts",
        4: "SUMO depth 12-14: Fine-grained concepts",
        5: "SUMO depth 15+: Very specific concepts",
        6: "WordNet-only concepts (no SUMO mapping)",
    }

    for layer_num in range(7):
        concepts_in_layer = layers[layer_num]
        print(f"\n  Generating layer{layer_num}.json...")

        layer_concepts = []

        for concept in sorted(concepts_in_layer):
            synsets = concept_to_synsets.get(concept, [])

            # Get canonical synset info
            canonical_synset = None
            lemmas = [concept]
            pos = None
            definition = f"SUMO category: {concept}"
            lexname = None

            if synsets:
                canonical_synset = synsets[0]
                try:
                    syn = wn.synset(canonical_synset)
                    lemmas = syn.lemma_names()
                    pos = syn.pos()
                    definition = syn.definition()
                    lexname = syn.lexname()
                except:
                    pass

            concept_entry = {
                "sumo_term": concept,
                "sumo_depth": depths.get(concept, 999),
                "layer": layer_num,
                "is_category_lens": True,
                "is_pseudo_sumo": False,
                "category_children": sorted(list(children_map.get(concept, []))),
                "synset_count": len(synsets),
                "direct_synset_count": len(synsets),
                "synsets": synsets[:5],  # Sample
                "canonical_synset": canonical_synset or f"{concept.lower()}.n.01",
                "lemmas": lemmas,
                "pos": pos,
                "definition": definition,
                "lexname": lexname,
                "parent_concepts": sorted(list(parent_map.get(concept, []))),
            }

            layer_concepts.append(concept_entry)

        # Create metadata
        metadata = {
            "layer": layer_num,
            "description": layer_descriptions[layer_num],
            "total_concepts": len(layer_concepts),
            "samples": [
                {
                    "sumo_term": c["sumo_term"],
                    "sumo_depth": c["sumo_depth"],
                    "synset_count": c["synset_count"],
                    "definition": c["definition"][:100] + "..." if len(c["definition"]) > 100 else c["definition"]
                }
                for c in layer_concepts[:5]
            ]
        }

        # Save layer file
        output_data = {
            "metadata": metadata,
            "concepts": layer_concepts
        }

        output_path = OUTPUT_DIR / f"layer{layer_num}.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"    ✓ Saved {len(layer_concepts)} concepts to {output_path}")

    print(f"\n✓ All layer files generated in {OUTPUT_DIR}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 80)
    print("V4 CONCEPT LAYER BUILDER")
    print("=" * 80)
    print(f"\nBuild date: {datetime.now().isoformat()}")
    print(f"SUMO source: {SUMO_SOURCE_DIR}")
    print(f"Patches: {PATCH_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Step 1: Parse KIF files
    relationships, file_stats = parse_all_kif_files()

    # Step 2: Apply patches
    patched_relationships = apply_patches(relationships)

    # Step 3: Build hierarchy
    parent_map, children_map, depths, root = build_hierarchy(patched_relationships)

    # Step 4: Map to WordNet
    all_concepts = set(depths.keys())
    concept_to_synsets = map_to_wordnet(all_concepts)

    # Step 5: Assign layers
    layers = assign_layers(depths, concept_to_synsets)

    # Step 6: Generate layer files
    generate_layer_files(layers, depths, parent_map, children_map, concept_to_synsets, root)

    # Summary
    print("\n" + "=" * 80)
    print("V4 BUILD COMPLETE")
    print("=" * 80)
    print(f"\n✓ Parsed {len(file_stats)} KIF files")
    print(f"✓ Total SUMO concepts: {len(all_concepts)}")
    print(f"✓ Concepts mapped to WordNet: {len([c for c in all_concepts if c in concept_to_synsets])}")
    print(f"✓ Maximum depth: {max(depths.values())}")
    print(f"✓ Layer files saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
