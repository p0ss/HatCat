#!/usr/bin/env python3
"""
Generate WordNet synsets from SUMO definitions for V4 concepts.

This script:
1. Loads V4 layer files and SUMO→WordNet mapping
2. Identifies concepts without WordNet synsets
3. Uses SUMO definitions to create synthetic synsets (no API needed!)
4. Updates the sumo_to_wordnet.json mapping and rebuilds V4 layer files

Usage:
    python scripts/generate_synsets_from_sumo.py
    python scripts/generate_synsets_from_sumo.py --test 20  # Test mode
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

V4_DIR = Path("data/concept_graph/v4")
SUMO_WORDNET_FILE = Path("data/concept_graph/sumo_to_wordnet.json")
KIF_DIR = Path("data/concept_graph/sumo_source")


def load_sumo_definitions() -> Dict[str, str]:
    """Load SUMO concept definitions from KIF files."""
    definitions = {}

    for kif_file in KIF_DIR.glob("*.kif"):
        with open(kif_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Parse documentation strings
        doc_pattern = r'\(documentation\s+([A-Za-z0-9_-]+)\s+EnglishLanguage\s+"([^"]+)"'
        for match in re.finditer(doc_pattern, content):
            concept, definition = match.groups()
            definitions[concept] = definition

    return definitions


def find_concepts_without_synsets() -> List[Tuple[str, int, str, Optional[str]]]:
    """Find concepts in V4 layers that lack WordNet synsets.

    Returns list of (concept, layer, domain, sumo_definition)
    """
    # Load SUMO definitions
    sumo_defs = load_sumo_definitions()
    print(f"Loaded {len(sumo_defs)} SUMO definitions")

    # Load existing SUMO→WordNet mapping
    if SUMO_WORDNET_FILE.exists():
        with open(SUMO_WORDNET_FILE) as f:
            sumo_wordnet = json.load(f)
    else:
        sumo_wordnet = {}

    missing = []

    # Check each layer
    for layer_num in range(5):
        layer_file = V4_DIR / f"layer{layer_num}.json"

        with open(layer_file) as f:
            layer_data = json.load(f)

        for concept_entry in layer_data['concepts']:
            term = concept_entry['sumo_term']

            # Skip pseudo-SUMO (Layer 0 domain labels)
            if concept_entry.get('is_pseudo_sumo'):
                continue

            # Check if it has synsets
            has_synsets = (
                term in sumo_wordnet or
                concept_entry.get('synsets') or
                concept_entry.get('canonical_synset')
            )

            if not has_synsets:
                sumo_def = concept_entry.get('sumo_definition') or sumo_defs.get(term)
                domain = concept_entry.get('domain', 'Unknown')
                missing.append((term, layer_num, domain, sumo_def))

    return missing


def camel_case_to_words(concept: str) -> List[str]:
    """Convert CamelCase to list of words."""
    # Split on capital letters
    words = re.findall(r'[A-Z][a-z]*', concept)
    return [w.lower() for w in words] if words else [concept.lower()]


def infer_pos_from_concept(concept: str) -> str:
    """Infer part of speech from concept name."""
    # Common verb suffixes
    verb_suffixes = ['ing', 'Process', 'Event', 'Action', 'Activity']
    for suffix in verb_suffixes:
        if concept.endswith(suffix):
            return 'verb'

    # Common adjective patterns
    adj_suffixes = ['Attribute', 'Property', 'Quality', 'able', 'ible', 'ive']
    for suffix in adj_suffixes:
        if concept.endswith(suffix):
            return 'adjective'

    # Default to noun for most SUMO concepts
    return 'noun'


def create_synset_from_sumo(
    concept: str,
    sumo_definition: Optional[str],
    layer: int,
    domain: str
) -> Dict:
    """Create a synthetic synset from SUMO definition."""

    # Generate synset ID
    words = camel_case_to_words(concept)
    synset_name = '_'.join(words)
    pos = infer_pos_from_concept(concept)
    pos_tag = {'noun': 'n', 'verb': 'v', 'adjective': 'a', 'adverb': 'r'}.get(pos, 'n')
    synset_id = f"{synset_name}.{pos_tag}.01"

    # Generate lemmas (alternate forms)
    lemmas = [
        ' '.join(words),  # space-separated
        '_'.join(words),  # underscore-separated
    ]

    # Add hyphenated version if multi-word
    if len(words) > 1:
        lemmas.append('-'.join(words))

    # Remove duplicates and lowercase
    lemmas = list(set(lemma.lower() for lemma in lemmas))

    # Use SUMO definition or generate fallback
    if sumo_definition:
        definition = sumo_definition
    else:
        readable_name = ' '.join(words)
        definition = f"A concept in the {domain} domain of the SUMO ontology: {readable_name}"

    # Build synset entry (matching build_sumo_wordnet_mapping.py format)
    return {
        "synset_count": 1,
        "synsets": [synset_id],
        "canonical_synset": synset_id,
        "lemmas": lemmas,
        "pos": pos,
        "definition": definition,
        "source": "sumo_definition"
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate WordNet synsets from SUMO definitions for V4 concepts"
    )
    parser.add_argument(
        '--test',
        type=int,
        metavar='N',
        help='Test mode: only process first N concepts'
    )
    parser.add_argument(
        '--no-rebuild',
        action='store_true',
        help='Skip rebuilding V4 layer files after update'
    )

    args = parser.parse_args()

    # Find missing concepts
    print("=" * 80)
    print("FINDING CONCEPTS WITHOUT WORDNET SYNSETS")
    print("=" * 80)

    missing = find_concepts_without_synsets()
    print(f"\nFound {len(missing)} concepts without WordNet synsets")

    if args.test:
        missing = missing[:args.test]
        print(f"TEST MODE: Processing only first {args.test} concepts")

    if not missing:
        print("All concepts have synsets! Nothing to do.")
        return

    # Show sample
    print(f"\nSample concepts to process:")
    for concept, layer, domain, sumo_def in missing[:10]:
        has_def = "✓" if sumo_def else "✗"
        def_preview = (sumo_def[:50] + "...") if sumo_def else "(no definition)"
        print(f"  Layer {layer} | {domain:20s} | {concept:30s} | {has_def} {def_preview}")

    # Load existing mapping
    if SUMO_WORDNET_FILE.exists():
        with open(SUMO_WORDNET_FILE) as f:
            sumo_wordnet = json.load(f)
    else:
        sumo_wordnet = {}

    # Generate synsets
    print(f"\n{'=' * 80}")
    print("GENERATING SYNSETS FROM SUMO DEFINITIONS")
    print("=" * 80)

    stats = {
        'total': len(missing),
        'with_sumo_def': 0,
        'without_def': 0
    }

    for concept, layer, domain, sumo_def in tqdm(missing, desc="Creating synsets"):
        synset_data = create_synset_from_sumo(concept, sumo_def, layer, domain)
        sumo_wordnet[concept] = synset_data

        if sumo_def:
            stats['with_sumo_def'] += 1
        else:
            stats['without_def'] += 1

    # Save updated mapping
    print(f"\n{'=' * 80}")
    print("SAVING UPDATED MAPPING")
    print("=" * 80)

    with open(SUMO_WORDNET_FILE, 'w') as f:
        json.dump(sumo_wordnet, f, indent=2)

    print(f"\n✓ Updated {SUMO_WORDNET_FILE}")
    print(f"  Total concepts in mapping: {len(sumo_wordnet)}")
    print(f"  Generated from SUMO definitions: {stats['with_sumo_def']}")
    print(f"  Generated with fallback: {stats['without_def']}")

    # Rebuild V4 layer files
    if not args.no_rebuild:
        print(f"\n{'=' * 80}")
        print("REBUILDING V4 LAYER FILES")
        print("=" * 80)

        import subprocess
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/build_v4_layer_files.py"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✓ V4 layer files rebuilt successfully")
            # Show summary from build output
            for line in result.stdout.split('\n'):
                if 'With WordNet:' in line or 'Total concepts' in line:
                    print(f"  {line.strip()}")
        else:
            print(f"ERROR rebuilding V4 files:")
            print(result.stderr)

    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Verify coverage: poetry run python /tmp/verify_v4_integrity.py")
    print(f"2. Train lenses: poetry run python scripts/train_sumo_classifiers.py")


if __name__ == '__main__':
    main()
