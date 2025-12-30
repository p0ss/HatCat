#!/usr/bin/env python3
"""
Generate WordNet synset mappings for custom safety-critical concepts.

This script:
1. Extracts concepts from custom KIF files in data/concept_graph/custom_concepts/
2. Checks existing WordNet coverage using direct lookup and CamelCase splitting
3. Uses Anthropic API to generate synthetic synsets for unmapped concepts
4. Outputs a JSON mapping file for integration into V4.5 build

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    python scripts/generate_custom_concept_synsets.py --output data/concept_graph/custom_synsets.json

    # Test mode (process only first N concepts):
    python scripts/generate_custom_concept_synsets.py --test 20 --output test_synsets.json
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm

# Ensure WordNet is downloaded
try:
    wn.synsets('test')
except LookupError:
    print("Downloading WordNet data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def extract_concepts_from_kif(kif_path: Path) -> Set[str]:
    """Extract all concept names from a KIF file."""
    concepts = set()

    with open(kif_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith(';'):
                continue

            # Match (subclass ConceptName ParentName)
            match = re.match(r'\(subclass\s+(\w+)\s+\w+\)', line)
            if match:
                concepts.add(match.group(1))

            # Match (instance ConceptName ...)
            match = re.match(r'\(instance\s+(\w+)\s+\w+\)', line)
            if match:
                concepts.add(match.group(1))

    return concepts


def load_all_custom_concepts(custom_dir: Path) -> Dict[str, Set[str]]:
    """Load all concepts from custom KIF files, organized by file."""
    concepts_by_file = {}

    for kif_file in sorted(custom_dir.glob('*.kif')):
        concepts = extract_concepts_from_kif(kif_file)
        if concepts:
            concepts_by_file[kif_file.name] = concepts

    return concepts_by_file


def check_wordnet_coverage(concept: str) -> List[str]:
    """Check if a concept exists in WordNet, return matching synset names."""
    synsets = []

    # Try 1: Direct lowercase match
    concept_lower = concept.lower()
    synsets = wn.synsets(concept_lower)
    if synsets:
        return [s.name() for s in synsets]

    # Try 2: CamelCase splitting
    words = re.findall(r'[A-Z][a-z]*', concept)
    if words:
        # Try with underscores (WordNet format)
        search_term = '_'.join(words).lower()
        synsets = wn.synsets(search_term)
        if synsets:
            return [s.name() for s in synsets]

        # Try with spaces
        search_term = ' '.join(words).lower()
        synsets = wn.synsets(search_term)
        if synsets:
            return [s.name() for s in synsets]

    return []


def generate_synthetic_synset_via_api(
    concept: str,
    kif_file: str,
    documentation: Optional[str] = None
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

    prompt = f"""You are creating a WordNet-style synset entry for the concept "{concept}" (readable as: "{readable_name}").

Context:
- This concept is from the file: {kif_file}
- Domain: AI safety, cognitive science, ethics, or formal reasoning
{f'- Documentation: {documentation}' if documentation else ''}

Please generate a synthetic synset with:

1. **synset_id**: A WordNet-style ID (e.g., "mesa_optimization.n.01" for the first noun sense)
   - Use underscores between words
   - Add POS tag: .n. (noun), .v. (verb), .a. (adjective), .r. (adverb)
   - Add sense number: .01, .02, etc.

2. **definition**: A clear, concise definition (1-2 sentences)

3. **lemmas**: List of word forms that refer to this concept (e.g., ["mesa optimization", "inner optimization"])

4. **examples**: 2-3 usage examples showing the concept in context

5. **pos**: Part of speech (noun, verb, adjective, adverb)

6. **hypernyms**: List of 1-3 broader concepts (parent categories)

Return ONLY a valid JSON object with these fields, no other text:
```json
{{
  "synset_id": "...",
  "definition": "...",
  "lemmas": [...],
  "examples": [...],
  "pos": "noun",
  "hypernyms": [...]
}}
```"""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
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
        synset_data['source'] = 'anthropic_api'
        synset_data['concept'] = concept
        return synset_data
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for {concept}: {e}")
        print(f"Response was: {content}")
        return {
            'synset_id': f"{concept.lower()}.n.01",
            'definition': f"AI safety concept: {readable_name}",
            'lemmas': [readable_name.lower()],
            'examples': [],
            'pos': 'noun',
            'hypernyms': [],
            'source': 'fallback',
            'concept': concept,
            'error': str(e)
        }


def create_synset_mappings(
    concepts_by_file: Dict[str, Set[str]],
    use_api: bool = True,
    test_mode: Optional[int] = None
) -> Dict[str, Dict]:
    """Create synset mappings for all concepts."""

    all_concepts = []
    for kif_file, concepts in concepts_by_file.items():
        for concept in concepts:
            all_concepts.append((concept, kif_file))

    all_concepts.sort()

    if test_mode:
        all_concepts = all_concepts[:test_mode]
        print(f"TEST MODE: Processing only first {test_mode} concepts")

    mappings = {}
    stats = {
        'total': len(all_concepts),
        'wordnet_found': 0,
        'api_generated': 0,
        'failed': 0
    }

    print(f"\nProcessing {len(all_concepts)} concepts...")

    for concept, kif_file in tqdm(all_concepts, desc="Mapping concepts"):
        # Check WordNet first
        wn_synsets = check_wordnet_coverage(concept)

        if wn_synsets:
            mappings[concept] = {
                'synset_ids': wn_synsets,
                'source': 'wordnet',
                'kif_file': kif_file
            }
            stats['wordnet_found'] += 1
        elif use_api:
            # Generate synthetic synset
            try:
                synset_data = generate_synthetic_synset_via_api(concept, kif_file)
                mappings[concept] = synset_data
                mappings[concept]['kif_file'] = kif_file
                stats['api_generated'] += 1
            except Exception as e:
                print(f"\nError generating synset for {concept}: {e}")
                mappings[concept] = {
                    'synset_id': f"{concept.lower()}.n.01",
                    'source': 'error',
                    'kif_file': kif_file,
                    'error': str(e)
                }
                stats['failed'] += 1
        else:
            # Mark as unmapped
            mappings[concept] = {
                'synset_id': None,
                'source': 'unmapped',
                'kif_file': kif_file
            }
            stats['failed'] += 1

    return mappings, stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate WordNet synset mappings for custom concepts"
    )
    parser.add_argument(
        '--custom-dir',
        type=Path,
        default=Path('data/concept_graph/custom_concepts'),
        help='Directory containing custom KIF files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/concept_graph/custom_synsets.json'),
        help='Output JSON file for synset mappings'
    )
    parser.add_argument(
        '--no-api',
        action='store_true',
        help='Skip API calls, only check WordNet coverage'
    )
    parser.add_argument(
        '--test',
        type=int,
        metavar='N',
        help='Test mode: only process first N concepts'
    )

    args = parser.parse_args()

    # Load concepts
    print(f"Loading concepts from {args.custom_dir}...")
    concepts_by_file = load_all_custom_concepts(args.custom_dir)

    total_concepts = sum(len(concepts) for concepts in concepts_by_file.values())
    print(f"Found {total_concepts} concepts across {len(concepts_by_file)} files:")
    for kif_file, concepts in sorted(concepts_by_file.items()):
        print(f"  {kif_file}: {len(concepts)} concepts")

    # Create mappings
    use_api = not args.no_api
    if use_api and not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nWARNING: ANTHROPIC_API_KEY not set. Running in no-API mode.")
        use_api = False

    mappings, stats = create_synset_mappings(
        concepts_by_file,
        use_api=use_api,
        test_mode=args.test
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("\n" + "=" * 80)
    print("SYNSET MAPPING STATISTICS")
    print("=" * 80)
    print(f"Total concepts:        {stats['total']}")
    print(f"WordNet matches:       {stats['wordnet_found']:5} ({100*stats['wordnet_found']/stats['total']:5.1f}%)")
    print(f"API-generated:         {stats['api_generated']:5} ({100*stats['api_generated']/stats['total']:5.1f}%)")
    print(f"Failed/unmapped:       {stats['failed']:5} ({100*stats['failed']/stats['total']:5.1f}%)")
    print(f"\nOutput saved to: {args.output}")

    if args.test:
        print(f"\nTEST MODE: Only processed {args.test} concepts")
        print(f"To process all concepts, run without --test flag")


if __name__ == '__main__':
    main()
