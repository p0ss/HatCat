#!/usr/bin/env python3
"""
Enhance fallback synsets with API-generated definitions.

This script:
1. Loads sumo_to_wordnet.json
2. Identifies synsets with fallback definitions (source="fallback" or generic definitions)
3. Uses Anthropic API to generate proper synthetic synsets
4. Updates the mapping and rebuilds V4 layer files

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    python scripts/enhance_fallback_synsets.py --test 5   # Test mode
    python scripts/enhance_fallback_synsets.py            # Full run
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

V4_DIR = Path("data/concept_graph/v4")
SUMO_WORDNET_FILE = Path("data/concept_graph/sumo_to_wordnet.json")


def find_fallback_synsets() -> List[Tuple[str, Dict]]:
    """Find synsets with fallback definitions that need enhancement.

    Returns list of (concept, synset_data, layer, domain)
    """
    if not SUMO_WORDNET_FILE.exists():
        print("ERROR: sumo_to_wordnet.json not found")
        return []

    with open(SUMO_WORDNET_FILE) as f:
        sumo_wordnet = json.load(f)

    # Load V4 layer files to get layer and domain info
    concept_metadata = {}
    for layer_num in range(5):
        layer_file = V4_DIR / f"layer{layer_num}.json"
        with open(layer_file) as f:
            layer_data = json.load(f)

        for concept_entry in layer_data['concepts']:
            term = concept_entry['sumo_term']
            if not concept_entry.get('is_pseudo_sumo'):
                concept_metadata[term] = {
                    'layer': layer_num,
                    'domain': concept_entry.get('domain', 'Unknown')
                }

    fallback_concepts = []

    for concept, synset_data in sumo_wordnet.items():
        if not isinstance(synset_data, dict):
            continue

        # Check if this is a fallback synset
        source = synset_data.get('source', '')
        definition = synset_data.get('definition', '')

        is_fallback = (
            source == 'fallback' or
            definition.startswith('A concept in the ') or
            definition.startswith('SUMO concept:')
        )

        if is_fallback and concept in concept_metadata:
            metadata = concept_metadata[concept]
            fallback_concepts.append((
                concept,
                synset_data,
                metadata['layer'],
                metadata['domain']
            ))

    return fallback_concepts


def generate_synthetic_synset_via_api(
    concept: str,
    layer: int,
    domain: str,
    current_synset_data: Dict
) -> Dict:
    """Generate an enhanced synset using Anthropic API."""
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

    # Get existing synset ID and POS
    existing_synset_id = current_synset_data.get('canonical_synset', f"{concept.lower()}.n.01")
    existing_pos = current_synset_data.get('pos', 'noun')

    prompt = f"""You are creating a WordNet-style synset entry for the SUMO ontology concept "{concept}" (readable as: "{readable_name}").

Context:
- SUMO Concept: {concept}
- Knowledge Domain: {domain}
- Ontology Layer: {layer} (0=broad domain, 4=specific)
- Current synset ID: {existing_synset_id}
- Part of speech: {existing_pos}

Please generate a synthetic synset compatible with WordNet format, with:

1. **synset_id**: Use the existing ID: "{existing_synset_id}"

2. **definition**: A clear, concise definition (1-2 sentences) that captures the SUMO concept meaning
   - Write as a professional lexicographer would
   - Be specific to this concept, not generic
   - Focus on what distinguishes this concept

3. **lemmas**: List of 2-5 word forms/synonyms that refer to this concept
   - Include the readable name
   - Include variations and related terms
   - Use lowercase

4. **pos**: Use "{existing_pos}"

Return ONLY a valid JSON object with these fields, no other text:
```json
{{
  "synset_id": "{existing_synset_id}",
  "definition": "...",
  "lemmas": ["{readable_name.lower()}", "..."],
  "pos": "{existing_pos}"
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

        # Build enhanced SUMO→WordNet mapping entry
        return {
            "synset_count": current_synset_data.get('synset_count', 1),
            "synsets": current_synset_data.get('synsets', [existing_synset_id]),
            "canonical_synset": synset_data.get('synset_id', existing_synset_id),
            "lemmas": synset_data.get('lemmas', [readable_name.lower()]),
            "pos": synset_data.get('pos', existing_pos),
            "definition": synset_data.get('definition', f"SUMO concept: {readable_name}"),
            "source": "anthropic_api"
        }

    except json.JSONDecodeError as e:
        print(f"\nWARNING: Failed to parse JSON for {concept}, keeping fallback")
        print(f"  Response was: {content[:200]}...")
        # Keep the existing fallback
        return current_synset_data


def main():
    parser = argparse.ArgumentParser(
        description="Enhance fallback synsets with API-generated definitions"
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

    # Find fallback synsets
    print("=" * 80)
    print("FINDING FALLBACK SYNSETS TO ENHANCE")
    print("=" * 80)

    fallback_concepts = find_fallback_synsets()
    print(f"\nFound {len(fallback_concepts)} concepts with fallback definitions")

    if args.test:
        fallback_concepts = fallback_concepts[:args.test]
        print(f"TEST MODE: Processing only first {args.test} concepts")

    if not fallback_concepts:
        print("No fallback synsets to enhance!")
        return

    # Show sample
    print(f"\nSample concepts to enhance:")
    for concept, synset_data, layer, domain in fallback_concepts[:10]:
        definition = synset_data.get('definition', '')[:60]
        print(f"  Layer {layer} | {domain:20s} | {concept:30s} | {definition}...")

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set your API key: export ANTHROPIC_API_KEY='your-key-here'")
        print("Get your key from: https://console.anthropic.com/")
        return

    # Load existing mapping
    with open(SUMO_WORDNET_FILE) as f:
        sumo_wordnet = json.load(f)

    # Enhance synsets
    print(f"\n{'=' * 80}")
    print("ENHANCING SYNSETS WITH API-GENERATED DEFINITIONS")
    print("=" * 80)

    stats = {'success': 0, 'failed': 0}

    for concept, current_data, layer, domain in tqdm(fallback_concepts, desc="Enhancing synsets"):
        try:
            enhanced_data = generate_synthetic_synset_via_api(concept, layer, domain, current_data)
            sumo_wordnet[concept] = enhanced_data
            stats['success'] += 1
        except Exception as e:
            print(f"\nERROR enhancing synset for {concept}: {e}")
            stats['failed'] += 1

    # Save updated mapping
    print(f"\n{'=' * 80}")
    print("SAVING UPDATED MAPPING")
    print("=" * 80)

    with open(SUMO_WORDNET_FILE, 'w') as f:
        json.dump(sumo_wordnet, f, indent=2)

    print(f"\n✓ Updated {SUMO_WORDNET_FILE}")
    print(f"  Total concepts in mapping: {len(sumo_wordnet)}")
    print(f"  Successfully enhanced: {stats['success']}")
    print(f"  Failed: {stats['failed']}")

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
