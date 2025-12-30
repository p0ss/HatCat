#!/usr/bin/env python3
"""
Build proper SUMO → WordNet synset mapping from WordNet mappings files.

Parses SUMO's WordNetMappings files to extract the relationship between
SUMO concepts and WordNet synsets, then combines with existing synset data.
"""

import re
import json
from pathlib import Path
from collections import defaultdict
from nltk.corpus import wordnet as wn

SUMO_SOURCE_DIR = Path("data/concept_graph/sumo_source")
WORDNET_V2_FILE = Path("data/concept_graph/wordnet_v2_top10000.json")
OUTPUT_FILE = Path("data/concept_graph/sumo_to_wordnet.json")

def parse_wordnet_mappings():
    """Parse WordNet mapping files to extract SUMO → synset mappings."""
    sumo_to_synsets = defaultdict(lambda: {"synsets": [], "synset_ids": set()})

    mapping_files = list(SUMO_SOURCE_DIR.glob("WordNetMappings30-*.txt"))

    for mapping_file in mapping_files:
        print(f"Parsing {mapping_file.name}...")

        with open(mapping_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith(';') or not line.strip():
                    continue

                # Look for SUMO mappings: &%ConceptName= or &%ConceptName+ etc.
                sumo_matches = re.findall(r'&%([A-Za-z0-9_-]+)[=+:@\[\]]', line)
                if not sumo_matches:
                    continue

                # Extract synset ID (8-digit number at start of line)
                synset_match = re.match(r'^(\d{8})\s+\d+\s+([nvasr])', line)
                if not synset_match:
                    continue

                synset_offset, pos = synset_match.groups()
                synset_id = f"{synset_offset}.{pos}"

                # Map each SUMO concept to this synset
                for sumo_concept in sumo_matches:
                    if synset_id not in sumo_to_synsets[sumo_concept]["synset_ids"]:
                        sumo_to_synsets[sumo_concept]["synset_ids"].add(synset_id)

    # Convert sets to lists
    for concept in sumo_to_synsets:
        sumo_to_synsets[concept]["synsets"] = sorted(list(sumo_to_synsets[concept]["synset_ids"]))
        del sumo_to_synsets[concept]["synset_ids"]

    return dict(sumo_to_synsets)


def enhance_with_wordnet_data(sumo_mapping):
    """Enhance SUMO mapping with WordNet synset details."""
    enhanced = {}

    for sumo_concept, data in sumo_mapping.items():
        synsets = data["synsets"]
        if not synsets:
            continue

        # Try to get first synset details
        canonical_synset = synsets[0]

        enhanced[sumo_concept] = {
            "synset_count": len(synsets),
            "synsets": synsets[:50],  # Limit to 50 synsets
            "canonical_synset": canonical_synset,
        }

        # Try to enrich with NLTK WordNet data
        try:
            # Parse synset ID (offset.pos format)
            match = re.match(r'(\d+)\.([nvasr])', canonical_synset)
            if match:
                offset_str, pos = match.groups()
                offset = int(offset_str)

                # Find synset in WordNet
                synset = wn.synset_from_pos_and_offset(pos, offset)

                enhanced[sumo_concept]["lemmas"] = synset.lemma_names()
                enhanced[sumo_concept]["pos"] = synset.pos()
                enhanced[sumo_concept]["definition"] = synset.definition()
                enhanced[sumo_concept]["lexname"] = synset.lexname()
        except Exception as e:
            # If NLTK fails, skip enrichment
            pass

    return enhanced


def main():
    print("=" * 80)
    print("BUILDING SUMO → WORDNET MAPPING")
    print("=" * 80)

    # Parse SUMO WordNet mapping files
    print("\n1. Parsing WordNet mapping files...")
    sumo_mapping = parse_wordnet_mappings()
    print(f"   Found {len(sumo_mapping)} SUMO concepts with WordNet synsets")

    # Enhance with WordNet data
    print("\n2. Enhancing with WordNet synset details...")
    enhanced_mapping = enhance_with_wordnet_data(sumo_mapping)
    print(f"   Enhanced {len(enhanced_mapping)} concepts")

    # Save mapping
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(enhanced_mapping, f, indent=2)

    print(f"\n3. Saved SUMO → WordNet mapping to {OUTPUT_FILE}")

    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    synset_counts = [data["synset_count"] for data in enhanced_mapping.values()]
    with_lemmas = sum(1 for data in enhanced_mapping.values() if "lemmas" in data)
    with_definition = sum(1 for data in enhanced_mapping.values() if "definition" in data)

    print(f"\nTotal SUMO concepts mapped: {len(enhanced_mapping)}")
    print(f"With lemmas: {with_lemmas} ({100*with_lemmas/len(enhanced_mapping):.1f}%)")
    print(f"With WordNet definitions: {with_definition} ({100*with_definition/len(enhanced_mapping):.1f}%)")
    print(f"\nSynset count distribution:")
    print(f"  Min: {min(synset_counts)}")
    print(f"  Max: {max(synset_counts)}")
    print(f"  Average: {sum(synset_counts)/len(synset_counts):.1f}")

    # Top 10 concepts by synset count
    print(f"\nTop 10 concepts by synset count:")
    top_concepts = sorted(enhanced_mapping.items(), key=lambda x: x[1]["synset_count"], reverse=True)[:10]
    for concept, data in top_concepts:
        print(f"  {concept}: {data['synset_count']} synsets")


if __name__ == '__main__':
    main()
