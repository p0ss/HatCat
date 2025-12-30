#!/usr/bin/env python3
"""Debug why AI categories have 0 synsets despite AI expansion mappings."""

import re
from pathlib import Path
from collections import defaultdict
from nltk.corpus import wordnet as wn
import networkx as nx

SUMO_DIR = Path('data/concept_graph/sumo_source')

# Step 1: Load SUMO hierarchy
print("="*60)
print("STEP 1: Load SUMO hierarchy")
print("="*60)

kif_files = [
    SUMO_DIR / "Merge.kif",
    SUMO_DIR / "Mid-level-ontology.kif",
    SUMO_DIR / "emotion.kif",
    SUMO_DIR / "Food.kif",
    SUMO_DIR / "Geography.kif",
    SUMO_DIR / "People.kif",
    SUMO_DIR / "AI.kif",
]

child_to_parent = {}
for kif_file in kif_files:
    if not kif_file.exists():
        continue
    with open(kif_file) as f:
        for line in f:
            if line.startswith(';') or not line.strip():
                continue
            match = re.match(r'^\(subclass\s+(\S+)\s+(\S+)\)', line.strip())
            if match:
                child, parent = match.groups()
                if not child.startswith('?') and not parent.startswith('?'):
                    child_to_parent[child] = parent

dg = nx.DiGraph()
for child, parent in child_to_parent.items():
    dg.add_edge(child, parent)
dg = dg.reverse()
depths = nx.single_source_shortest_path_length(dg, 'Entity')

ai_cats = [c for c in child_to_parent if 'Artificial' in c or 'AI' in c or c in ['LanguageModel', 'Superintelligence', 'Misalignment']]
print(f"✓ Found {len(ai_cats)} AI categories in SUMO hierarchy")
for cat in sorted(ai_cats)[:5]:
    print(f"  {cat}: depth={depths.get(cat, 'N/A')}")

# Step 2: Load WordNet mappings
print("\n" + "="*60)
print("STEP 2: Load WordNet mappings")
print("="*60)

mapping_files = {
    'n': SUMO_DIR / "WordNetMappings30-noun.txt",
    'v': SUMO_DIR / "WordNetMappings30-verb.txt",
    'a': SUMO_DIR / "WordNetMappings30-adj.txt",
    'r': SUMO_DIR / "WordNetMappings30-adv.txt",
    'ai': SUMO_DIR / "WordNetMappings30-AI-expansion.txt",
}

synset_to_sumo = {}
synset_to_all_sumo = defaultdict(list)

for pos, filepath in mapping_files.items():
    if not filepath.exists():
        continue
    with open(filepath) as f:
        for line in f:
            if line.startswith(';') or not line.strip():
                continue
            parts = line.split('|')
            if len(parts) < 2:
                continue
            synset_info = parts[0].strip().split()
            sumo_part = parts[-1].strip()
            if len(synset_info) < 3:
                continue
            offset = synset_info[0]
            pos_tag = synset_info[2]
            sumo_match = re.search(r'&%(\w+)([=+@\[\]:])$', sumo_part)
            if sumo_match:
                sumo_term = sumo_match.group(1)
                relation = sumo_match.group(2)
                try:
                    offset_int = int(offset)
                    synset = wn.synset_from_pos_and_offset(pos_tag, offset_int)
                    synset_name = synset.name()

                    if synset_name not in synset_to_sumo:
                        synset_to_sumo[synset_name] = (sumo_term, relation)

                    synset_to_all_sumo[synset_name].append((sumo_term, relation))
                except:
                    pass

ai_expansion_count = sum(1 for mappings in synset_to_all_sumo.values() if len(mappings) > 1)
print(f"✓ Loaded {len(synset_to_sumo)} primary synset→SUMO mappings")
print(f"✓ {ai_expansion_count} synsets with AI expansion multi-mappings")

# Step 3: Count synsets per SUMO term
print("\n" + "="*60)
print("STEP 3: Count synsets per SUMO term")
print("="*60)

sumo_to_synsets = defaultdict(list)
for synset_name, mappings in synset_to_all_sumo.items():
    for sumo_term, relation in mappings:
        sumo_to_synsets[sumo_term].append((synset_name, relation))

ai_term_counts = {term: len(sumo_to_synsets[term]) for term in ai_cats if term in sumo_to_synsets}
print(f"AI terms with synsets: {len(ai_term_counts)}/{len(ai_cats)}")
for term in sorted(ai_term_counts.keys()):
    print(f"  {term}: {ai_term_counts[term]} synsets")
    samples = [s for s, r in sumo_to_synsets[term][:3]]
    print(f"    Samples: {samples}")

# Step 4: Check populated_terms
print("\n" + "="*60)
print("STEP 4: Check which AI terms are marked as populated")
print("="*60)

populated_terms = set(sumo_to_synsets.keys())
print(f"Initially populated: {len(populated_terms)} terms")

# Recursive propagation
changed = True
while changed:
    changed = False
    for term in list(depths.keys()):
        if term not in populated_terms:
            children = [c for c, p in child_to_parent.items() if p == term]
            if any(c in populated_terms for c in children):
                populated_terms.add(term)
                changed = True

ai_populated = [c for c in ai_cats if c in populated_terms]
ai_unpopulated = [c for c in ai_cats if c not in populated_terms]

print(f"AI terms populated: {len(ai_populated)}/{len(ai_cats)}")
print(f"  Populated: {sorted(ai_populated)[:5]}")
print(f"  Unpopulated: {sorted(ai_unpopulated)}")
