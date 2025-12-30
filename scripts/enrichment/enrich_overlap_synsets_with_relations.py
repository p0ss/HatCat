#!/usr/bin/env python3
"""
Enrich simplex overlap synsets with WordNet-style relations.

Takes the generated overlap synsets and adds:
- hypernyms: broader concepts this synset belongs to
- antonyms: opposing concepts (especially useful for cross-pole negatives)
- similar_to: closely related concepts
- also_see: related concepts worth cross-referencing

This enriches the semantic graph and provides better training signals.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict
import anthropic
import time

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

INPUT_PATH = PROJECT_ROOT / "data" / "simplex_overlap_synsets.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "simplex_overlap_synsets_enriched.json"


def generate_relations_for_synset(
    synset: Dict,
    client: anthropic.Anthropic
) -> Dict:
    """
    Use Anthropic API to generate WordNet-style relations for a synset.

    Returns dict with:
    - hypernyms: list of broader category synsets
    - antonyms: list of opposing concepts
    - similar_to: list of closely related synsets
    - also_see: list of related concepts
    """

    lemmas_str = ", ".join(synset['lemmas'][:3])
    definition = synset['definition']
    poles = synset['applies_to_poles']

    prompt = f"""I'm building a WordNet-style lexical database. For the following concept, please suggest semantic relations:

**Concept:** {lemmas_str}
**Definition:** {definition}
**Dimensional poles:** {', '.join(poles)}

Please provide:

1. **Hypernyms** (2-3 broader categories this concept belongs to)
   - Use existing WordNet synset IDs if appropriate (e.g., "emotion.n.01")
   - Or suggest new synset names (e.g., "negative_affect.n.01")

2. **Antonyms** (2-4 opposing or contrasting concepts)
   - Focus on dimensional opposites (e.g., if this is high arousal + positive valence, antonyms might be low arousal + negative valence)
   - Use lemmas (e.g., "boredom", "apathy")

3. **Similar_to** (2-3 closely related but distinct concepts)
   - Near-synonyms or concepts in the same semantic space
   - Use lemmas

4. **Also_see** (2-3 related concepts worth cross-referencing)
   - Concepts that often co-occur or are theoretically related
   - Use lemmas

Return ONLY valid JSON in this exact format:
{{
  "hypernyms": ["synset.pos.nn", "synset2.pos.nn"],
  "antonyms": ["lemma1", "lemma2"],
  "similar_to": ["lemma1", "lemma2"],
  "also_see": ["lemma1", "lemma2"]
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract JSON from response
        content = response.content[0].text

        # Try to parse JSON
        # Sometimes Claude wraps in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        relations = json.loads(content)

        # Normalize antonyms, similar_to, also_see to underscore format
        for key in ['antonyms', 'similar_to', 'also_see']:
            if key in relations:
                relations[key] = [term.lower().replace(" ", "_").replace("-", "_")
                                 for term in relations[key]]

        return relations

    except Exception as e:
        print(f"  ✗ Error generating relations for {synset['synset_id']}: {e}")
        return {
            "hypernyms": [],
            "antonyms": [],
            "similar_to": [],
            "also_see": []
        }


def main():
    print("=" * 80)
    print("ENRICH OVERLAP SYNSETS WITH RELATIONS")
    print("=" * 80)

    # Initialize Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load existing synsets (resume from output if it exists)
    if OUTPUT_PATH.exists():
        print(f"\n1. Resuming from {OUTPUT_PATH}...")
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
    else:
        print(f"\n1. Loading overlap synsets from {INPUT_PATH}...")
        with open(INPUT_PATH) as f:
            data = json.load(f)

    total_synsets = data['metadata']['total_overlaps']
    print(f"   Found {total_synsets} synsets across {len(data['overlaps'])} pole pairs")

    # Enrich each synset with relations
    print("\n2. Generating relations for synsets...")
    enriched_count = 0
    error_count = 0

    for pair_idx, (pair_key, synsets) in enumerate(data['overlaps'].items(), 1):
        print(f"\n  [{pair_idx}/{len(data['overlaps'])}] Processing pair: {pair_key}")

        for synset_idx, synset in enumerate(synsets, 1):
            synset_id = synset['synset_id']
            print(f"    [{synset_idx}/{len(synsets)}] {synset_id}...", end=" ", flush=True)

            # Skip if already has relations
            if 'hypernyms' in synset:
                print("already enriched")
                enriched_count += 1
                continue

            # Generate relations
            relations = generate_relations_for_synset(synset, client)

            # Add to synset
            synset.update(relations)

            # Check if we got any relations
            has_relations = any(len(v) > 0 for v in relations.values() if isinstance(v, list))
            if has_relations:
                h_count = len(relations.get('hypernyms', []))
                a_count = len(relations.get('antonyms', []))
                print(f"✓ ({h_count} hypernyms, {a_count} antonyms)")
                enriched_count += 1
            else:
                print("✗ (no relations generated)")
                error_count += 1

            # Rate limiting: ~1 request per second
            time.sleep(1.2)

        # Save incrementally after each pair (in case of crashes)
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(data, f, indent=2)

    # Update metadata
    data['metadata']['enrichment_date'] = "2025-11-23"
    data['metadata']['enriched_synsets'] = enriched_count
    data['metadata']['enrichment_errors'] = error_count

    # Save enriched data
    print(f"\n3. Saving enriched synsets to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"   ✓ Saved {enriched_count} enriched synsets")

    # Summary
    print("\n" + "=" * 80)
    print("RELATION ENRICHMENT COMPLETE")
    print("=" * 80)
    print(f"\nTotal synsets: {total_synsets}")
    print(f"Successfully enriched: {enriched_count}")
    print(f"Errors: {error_count}")
    print(f"Output file: {OUTPUT_PATH}")
    print("\nNext steps:")
    print("1. Review enriched synsets in data/simplex_overlap_synsets_enriched.json")
    print("2. Update training data generation to use relations")
    print("3. Implement cross-pole negative sampling using antonyms")
    print("==" * 80)


if __name__ == "__main__":
    main()
