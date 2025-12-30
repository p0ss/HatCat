#!/usr/bin/env python3
"""
Extract parent relationship patches from current layer files.

This script identifies parent relationships in the current layers that are NOT
present in the original SUMO KIF files, and generates patch files to preserve them.
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def load_sumo_relationships():
    """Load all parent relationships from SUMO KIF files."""
    kif_dir = Path('data/concept_graph/sumo_source')
    sumo_parents = {}  # child -> parent

    for kif_file in kif_dir.glob('*.kif'):
        with open(kif_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('(subclass '):
                    parts = line.replace('(subclass ', '').replace(')', '').split()
                    if len(parts) >= 2:
                        child, parent = parts[0], parts[1]
                        sumo_parents[child] = parent
                elif line.startswith('(instance '):
                    parts = line.replace('(instance ', '').replace(')', '').split()
                    if len(parts) >= 2:
                        child, parent = parts[0], parts[1]
                        sumo_parents[child] = parent

    return sumo_parents

def load_layer_relationships():
    """Load all parent relationships from current layer files."""
    layer_dir = Path('data/concept_graph/abstraction_layers')
    layer_parents = defaultdict(list)  # child -> [parents]

    for layer_file in layer_dir.glob('layer*.json'):
        with open(layer_file) as f:
            data = json.load(f)

        for concept in data['concepts']:
            sumo_term = concept['sumo_term']
            parents = concept.get('parent_concepts', [])
            if parents:
                layer_parents[sumo_term] = parents

    return dict(layer_parents)

def find_patch_relationships(sumo_parents, layer_parents):
    """Find relationships in layers that aren't in SUMO KIF files.

    IMPORTANT: Patches OVERRIDE SUMO completely. If a concept needs any patch,
    we include ALL its parents (even if some match SUMO) so the patch is complete.
    """
    patches = defaultdict(list)  # source_file -> [(child, parent, reason)]

    # Map each concept to its source KIF file by loading SUMO again
    concept_to_file = {}
    kif_dir = Path('data/concept_graph/sumo_source')

    for kif_file in kif_dir.glob('*.kif'):
        with open(kif_file) as f:
            for line in f:
                if line.strip().startswith('(subclass ') or line.strip().startswith('(instance '):
                    parts = line.strip().replace('(subclass ', '').replace('(instance ', '').replace(')', '').split()
                    if len(parts) >= 2:
                        concept_to_file[parts[0]] = kif_file.stem

    # First pass: determine which concepts need patches
    concepts_needing_patches = set()

    for child, layer_parents_list in layer_parents.items():
        sumo_parent = sumo_parents.get(child)
        layer_parents_set = set(layer_parents_list)

        # Concept needs a patch if:
        # 1. Not in SUMO at all, OR
        # 2. Has different parents than SUMO (single parent different, or multiple parents)
        if child not in sumo_parents:
            concepts_needing_patches.add(child)
        elif len(layer_parents_set) > 1:
            # Multiple parents - needs patch (SUMO only has single parent)
            concepts_needing_patches.add(child)
        elif sumo_parent not in layer_parents_set:
            # Different parent - needs patch
            concepts_needing_patches.add(child)

    # Second pass: for concepts needing patches, include ALL their parents
    for child in concepts_needing_patches:
        layer_parents_list = layer_parents.get(child, [])
        sumo_parent = sumo_parents.get(child)

        # Determine the reason (use the first non-SUMO parent or indicate override)
        if child not in sumo_parents:
            reason = "WordNet-derived or custom concept"
        elif sumo_parent and sumo_parent not in layer_parents_list:
            reason = f"Overrides SUMO parent '{sumo_parent}'"
        else:
            reason = f"Extends SUMO parent '{sumo_parent}' with multiple inheritance"

        # Determine which file to patch
        source_file = concept_to_file.get(child, 'HatCat-core')
        if source_file == 'HatCat-core':
            # Try to use parent's file if child not found
            for parent in layer_parents_list:
                if parent in concept_to_file:
                    source_file = concept_to_file[parent]
                    break

        # Add ALL parents for this concept
        for parent in layer_parents_list:
            patches[source_file].append((child, parent, reason))

    return patches

def generate_patch_files(patches):
    """Generate KIF patch files."""
    patch_dir = Path('data/concept_graph/sumo_patches')
    patch_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d')

    for source_file, relationships in patches.items():
        patch_file = patch_dir / f'{source_file}.patch.kif'

        with open(patch_file, 'w') as f:
            f.write(f';; HatCat SUMO Patches for {source_file}.kif\n')
            f.write(f';; Generated: {timestamp}\n')
            f.write(f';;\n')
            f.write(f';; These patches extend or override the base SUMO ontology\n')
            f.write(f';; with relationships derived from WordNet hypernyms, manual\n')
            f.write(f';; curation, and semantic inference.\n')
            f.write(f';;\n')
            f.write(f';; Format: (subclass <child> <parent>)  ; <reason>\n')
            f.write(f'\n')

            # Group by reason
            by_reason = defaultdict(list)
            for child, parent, reason in relationships:
                by_reason[reason].append((child, parent))

            for reason, rels in sorted(by_reason.items()):
                f.write(f'\n;; {reason}\n')
                for child, parent in sorted(rels):
                    f.write(f'(subclass {child} {parent})\n')

        print(f'Generated: {patch_file} ({len(relationships)} relationships)')

    # Create README
    readme_file = patch_dir / 'README.md'
    with open(readme_file, 'w') as f:
        f.write('# SUMO Patches\n\n')
        f.write('This directory contains patches to the SUMO ontology in KIF format.\n\n')
        f.write('## Purpose\n\n')
        f.write('These patches preserve:\n')
        f.write('- Parent relationships derived from WordNet hypernyms\n')
        f.write('- Manual parent assignments for orphaned concepts\n')
        f.write('- Semantic inference-based parent assignments\n')
        f.write('- Geographic hierarchy reorganization\n\n')
        f.write('## Structure\n\n')
        f.write('Each `.patch.kif` file corresponds to a source `.kif` file in `sumo_source/`.\n')
        f.write('The build script loads both the source and patch files when generating layers.\n\n')
        f.write('## Format\n\n')
        f.write('Patches use standard KIF syntax:\n')
        f.write('```\n')
        f.write('(subclass ChildConcept ParentConcept)  ; Reason for patch\n')
        f.write('```\n\n')
        f.write(f'Last updated: {timestamp}\n')

    print(f'\nGenerated README: {readme_file}')

if __name__ == '__main__':
    print('Extracting parent relationship patches...\n')

    print('Loading SUMO KIF relationships...')
    sumo_parents = load_sumo_relationships()
    print(f'  Found {len(sumo_parents)} SUMO relationships')

    print('Loading layer file relationships...')
    layer_parents = load_layer_relationships()
    print(f'  Found {len(layer_parents)} concepts with parents')

    print('\nIdentifying patches (relationships not in SUMO KIF)...')
    patches = find_patch_relationships(sumo_parents, layer_parents)
    total_patches = sum(len(rels) for rels in patches.values())
    print(f'  Found {total_patches} relationships needing patches')

    print('\nGenerating patch files...')
    generate_patch_files(patches)

    print('\nDone! Patch files created in data/concept_graph/sumo_patches/')
