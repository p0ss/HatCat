#!/usr/bin/env python3
"""Debug script to find source of layer mismatches."""

import json
from pathlib import Path
from collections import defaultdict

# Source 1: hierarchy.json (child_to_parent has concept:layer keys)
hierarchy = json.load(open('concept_packs/first-light/hierarchy.json'))
hierarchy_layers = {}
for key in hierarchy.get('child_to_parent', {}).keys():
    name, layer = key.rsplit(':', 1)
    hierarchy_layers[name] = int(layer)

print(f'hierarchy.json: {len(hierarchy_layers)} concepts with layer assignments')

# Source 2: layer*.json files in hierarchy folder
layer_json_concepts = {}
hierarchy_dir = Path('concept_packs/first-light/hierarchy')
for layer_file in sorted(hierarchy_dir.glob('layer*.json')):
    layer = int(layer_file.stem.replace('layer', ''))
    data = json.load(open(layer_file))
    for concept in data.get('concepts', []):
        name = concept.get('sumo_term')
        if name:
            layer_json_concepts[name] = layer

print(f'layer*.json files: {len(layer_json_concepts)} concepts')

# Source 3: Lens files
lens_layers = {}
for f in Path('lens_packs/apertus-8b_first-light-bf16').glob('layer*/*.pt'):
    layer = int(f.parent.name.replace('layer', ''))
    lens_layers[f.stem] = layer

print(f'Lens files: {len(lens_layers)} concepts')

# Compare sources
all_concepts = set(hierarchy_layers.keys()) | set(layer_json_concepts.keys()) | set(lens_layers.keys())

matches_all = 0
hier_vs_json_mismatch = []
json_vs_lens_mismatch = []
hier_vs_lens_mismatch = []

for concept in all_concepts:
    h = hierarchy_layers.get(concept)
    j = layer_json_concepts.get(concept)
    l = lens_layers.get(concept)

    if h == j == l and h is not None:
        matches_all += 1
    else:
        if h is not None and j is not None and h != j:
            hier_vs_json_mismatch.append((concept, h, j))
        if j is not None and l is not None and j != l:
            json_vs_lens_mismatch.append((concept, j, l))
        if h is not None and l is not None and h != l:
            hier_vs_lens_mismatch.append((concept, h, l))

print(f'\nComparison:')
print(f'  All three match: {matches_all}')
print(f'  hierarchy.json vs layer*.json mismatch: {len(hier_vs_json_mismatch)}')
print(f'  layer*.json vs lens files mismatch: {len(json_vs_lens_mismatch)}')
print(f'  hierarchy.json vs lens files mismatch: {len(hier_vs_lens_mismatch)}')

if hier_vs_json_mismatch:
    print(f'\nSample hierarchy.json vs layer*.json mismatches:')
    for concept, h, j in sorted(hier_vs_json_mismatch)[:10]:
        print(f'  {concept}: hierarchy.json L{h}, layer*.json L{j}')

if json_vs_lens_mismatch:
    print(f'\nSample layer*.json vs lens files mismatches:')
    for concept, j, l in sorted(json_vs_lens_mismatch)[:10]:
        print(f'  {concept}: layer*.json L{j}, lens L{l}')

# Deeper analysis: where did lens layers come from?
print(f'\n--- Checking if lens layers match layer*.json ---')
lens_matches_json = 0
lens_not_in_json = 0
for concept, lens_layer in lens_layers.items():
    if concept in layer_json_concepts:
        if layer_json_concepts[concept] == lens_layer:
            lens_matches_json += 1
    else:
        lens_not_in_json += 1

print(f'  Lens layer matches layer*.json: {lens_matches_json}')
print(f'  Lens concept not in layer*.json: {lens_not_in_json}')
print(f'  Lens layer differs from layer*.json: {len(json_vs_lens_mismatch)}')
