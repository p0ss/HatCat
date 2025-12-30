#!/usr/bin/env python3
"""
Generate WordNet synsets for V4 concepts missing coverage.

This script:
1. Loads V4 layer files and SUMO→WordNet mapping
2. Identifies concepts without WordNet synsets (using SUMO definitions when available)
3. Uses Anthropic API to generate synthetic synsets
4. Updates the sumo_to_wordnet.json mapping and rebuilds V4 layer files

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    python scripts/generate_missing_synsets_v4.py --test 20  # Test mode
    python scripts/generate_missing_synsets_v4.py             # Full run
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


def generate_synthetic_synset_via_api(
    concept: str,
    layer: int,
    domain: str,
    sumo_definition: Optional[str] = None
) -> Dict:
    """Generate a synthetic synset using Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Install with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Get your API key from https://console.anthropic.com/"
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Split CamelCase for better readability
    words = re.findall(r'[A-Z][a-z]*', concept)
    readable_name = ' '.join(words) if words else concept

    prompt = f"""You are creating a WordNet-style synset entry for the SUMO ontology concept "{concept}" (readable as: "{readable_name}").

Context:
- SUMO Concept: {concept}
- Knowledge Domain: {domain}
- Ontology Layer: {layer} (0=broad domain, 4=specific)
{f'- SUMO Definition: {sumo_definition}' if sumo_definition else ''}

Please generate a synthetic synset compatible with WordNet format, with:

1. **synset_id**: A WordNet-style ID (e.g., "{concept.lower().replace('-', '_')}.n.01" for the first noun sense)
   - Use underscores between words
   - Add POS tag: .n. (noun), .v. (verb), .a. (adjective), .r. (adverb)
   - Add sense number: .01

2. **definition**: A clear, concise definition (1-2 sentences) that captures the SUMO concept meaning

3. **lemmas**: List of 2-5 word forms/synonyms that refer to this concept
   - Include the readable name
   - Include variations and related terms

4. **pos**: Part of speech (typically "noun" for SUMO concepts, but use appropriate POS)

Return ONLY a valid JSON object with these fields, no other text:
```json
{{
  "synset_id": "{concept.lower().replace('-', '_')}.n.01",
  "definition": "...",
  "lemmas": ["{readable_name.lower()}", "..."],
  "pos": "noun"
}}
```"""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract JSON from response
    content = response.content[0].text

    # Try to find JSON block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to parse the whole content as JSON
        json_str = content.strip()

    try:
        synset_data = json.loads(json_str)

        # Ensure required fields
        synset_id = synset_data.get('synset_id', f"{concept.lower()}.n.01")
        definition = synset_data.get('definition', f"SUMO concept: {readable_name}")
        lemmas = synset_data.get('lemmas', [readable_name.lower()])
        pos = synset_data.get('pos', 'noun')

        # Build SUMO→WordNet mapping entry (matching build_sumo_wordnet_mapping.py format)
        return {
            "synset_count": 1,
            "synsets": [synset_id],
            "canonical_synset": synset_id,
            "lemmas": lemmas,
            "pos": pos,
            "definition": definition,
            "source": "anthropic_api"
        }

    except json.JSONDecodeError as e:
        print(f"\nWARNING: Failed to parse JSON for {concept}, using fallback")
        print(f"  Response was: {content[:200]}...")

        # Fallback
        return {
            "synset_count": 1,
            "synsets": [f"{concept.lower()}.n.01"],
            "canonical_synset": f"{concept.lower()}.n.01",
            "lemmas": [readable_name.lower()],
            "pos": "noun",
            "definition": sumo_definition or f"SUMO concept: {readable_name}",
            "source": "fallback"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate WordNet synsets for V4 concepts missing coverage"
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
        print(f"  Layer {layer} | {domain:20s} | {concept:30s} | {'✓' if sumo_def else '✗'} SUMO def")

    # Load existing mapping
    if SUMO_WORDNET_FILE.exists():
        with open(SUMO_WORDNET_FILE) as f:
            sumo_wordnet = json.load(f)
    else:
        sumo_wordnet = {}

    # Generate synsets
    print(f"\n{'=' * 80}")
    print("GENERATING SYNTHETIC SYNSETS")
    print("=" * 80)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set your API key: export ANTHROPIC_API_KEY='your-key-here'")
        print("Get your key from: https://console.anthropic.com/")
        return

    stats = {'success': 0, 'failed': 0}

    for concept, layer, domain, sumo_def in tqdm(missing, desc="Generating synsets"):
        try:
            synset_data = generate_synthetic_synset_via_api(concept, layer, domain, sumo_def)
            sumo_wordnet[concept] = synset_data
            stats['success'] += 1
        except Exception as e:
            print(f"\nERROR generating synset for {concept}: {e}")
            stats['failed'] += 1

    # Save updated mapping
    print(f"\n{'=' * 80}")
    print("SAVING UPDATED MAPPING")
    print("=" * 80)

    with open(SUMO_WORDNET_FILE, 'w') as f:
        json.dump(sumo_wordnet, f, indent=2)

    print(f"\n✓ Updated {SUMO_WORDNET_FILE}")
    print(f"  Total concepts in mapping: {len(sumo_wordnet)}")
    print(f"  Successfully generated: {stats['success']}")
    print(f"  Failed: {stats['failed']}")

    # Rebuild V4 layer files
    if not args.no_rebuild:
        print(f"\n{'=' * 80}")
        print("REBUILDING V4 LAYER FILES")
        print("=" * 80)

        import subprocess
        result = subprocess.run(
            ["python", "scripts/build_v4_layer_files.py"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✓ V4 layer files rebuilt successfully")
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
