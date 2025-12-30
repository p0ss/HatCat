#!/usr/bin/env python3
"""
Generate synthetic synsets for S-tier simplex pole overlaps.

This script uses the Anthropic API to identify concepts that exist at the
intersection of multiple simplex poles, then generates synthetic synsets for them.

Example:
- high_valence (positive pole of affective_valence)
- high_arousal (positive pole of affective_arousal)
- Overlap: "excitement", "enthusiasm", "exhilaration" etc.

These overlapping synsets will be used as positive training data for BOTH poles,
dramatically increasing training data while maintaining clean separation via
cross-pole negatives.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
import anthropic

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LAYER2_PATH = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers" / "layer2.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "simplex_overlap_synsets.json"


def load_s_tier_simplexes() -> List[Dict]:
    """Load all S-tier simplexes from layer2.json"""
    with open(LAYER2_PATH) as f:
        layer2 = json.load(f)

    return [c for c in layer2['concepts'] if c.get('s_tier') and c.get('simplex_dimension')]


def generate_overlap_synsets_for_pair(
    pole1_name: str,
    pole1_desc: str,
    pole1_lemmas: List[str],
    pole2_name: str,
    pole2_desc: str,
    pole2_lemmas: List[str],
    client: anthropic.Anthropic
) -> List[Dict]:
    """
    Use Anthropic API to generate synsets for the overlap between two poles.

    Returns list of synset dictionaries with:
    - synset_id: synthetic synset ID
    - lemmas: list of terms
    - definition: definition of the concept
    - applies_to: list of pole names this synset exemplifies
    """

    prompt = f"""I'm building a lexical database for psychological/affective concepts. I need to identify concepts that exist at the intersection of two dimensions:

**Dimension 1: {pole1_name}**
Description: {pole1_desc}
Example terms: {', '.join(pole1_lemmas[:5])}

**Dimension 2: {pole2_name}**
Description: {pole2_desc}
Example terms: {', '.join(pole2_lemmas[:5])}

Please generate 5-10 concepts that strongly exemplify BOTH dimensions simultaneously. For each concept:

1. Primary lemma (the main term)
2. Alternative lemmas (2-4 synonyms or related terms)
3. Definition (1-2 sentences explaining how it exemplifies both dimensions)

Focus on concepts that are:
- Clear exemplars of BOTH dimensions
- Common enough to have training examples
- Distinct from each other

Return ONLY valid JSON in this exact format:
{{
  "overlaps": [
    {{
      "lemma": "primary_term",
      "alternates": ["synonym1", "synonym2", "synonym3"],
      "definition": "Definition explaining how this exemplifies both {pole1_name} and {pole2_name}"
    }}
  ]
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
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

        data = json.loads(content)

        # Convert to synset format
        synsets = []
        for i, overlap in enumerate(data.get("overlaps", []), 1):
            lemma = overlap["lemma"].lower().replace(" ", "_")
            alternates = [alt.lower().replace(" ", "_") for alt in overlap.get("alternates", [])]
            all_lemmas = [lemma] + alternates

            synset = {
                "synset_id": f"{lemma}.overlap.{i:02d}",
                "lemmas": all_lemmas,
                "definition": overlap["definition"],
                "pos": "n",  # Most psychological concepts are nouns
                "is_synthetic": True,
                "source": "anthropic_api_overlap",
                "applies_to_poles": [pole1_name, pole2_name],
                "generated_date": "2025-11-23"
            }
            synsets.append(synset)

        return synsets

    except Exception as e:
        print(f"  ✗ Error generating overlaps for {pole1_name} + {pole2_name}: {e}")
        return []


def identify_pole_pairs(simplexes: List[Dict]) -> List[Tuple[Dict, Dict, str, str]]:
    """
    Identify interesting pole pairs to generate overlaps for.

    Returns list of (simplex1, simplex2, pole1_name, pole2_name) tuples

    Strategy:
    1. Within-simplex pairs: positive + negative (e.g., love vs hate)
    2. Cross-simplex same-pole: positive affective + positive arousal
    3. Cross-simplex complementary: high valence + low arousal (calm contentment)
    """

    pairs = []

    # Within-simplex pairs (positive vs negative, creating neutral-zone concepts)
    for simplex in simplexes:
        poles = simplex['three_pole_simplex']
        dim = simplex['simplex_dimension']

        # Positive + Neutral (e.g., preference + acceptance = satisfaction)
        pairs.append((
            simplex, simplex,
            f"{dim}_positive", f"{dim}_neutral"
        ))

        # Neutral + Negative (e.g., acceptance + aversion = ambivalence)
        pairs.append((
            simplex, simplex,
            f"{dim}_neutral", f"{dim}_negative"
        ))

    # Cross-simplex pairs (same poles from different dimensions)
    for i, s1 in enumerate(simplexes):
        for s2 in simplexes[i+1:]:
            dim1 = s1['simplex_dimension']
            dim2 = s2['simplex_dimension']

            # Positive + Positive (e.g., high valence + high arousal = excitement)
            pairs.append((
                s1, s2,
                f"{dim1}_positive", f"{dim2}_positive"
            ))

            # Negative + Negative (e.g., low valence + high arousal = anxiety)
            pairs.append((
                s1, s2,
                f"{dim1}_negative", f"{dim2}_negative"
            ))

    return pairs


def main():
    print("=" * 80)
    print("SIMPLEX POLE OVERLAP SYNSET GENERATION")
    print("=" * 80)

    # Initialize Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load simplexes
    print("\n1. Loading S-tier simplexes...")
    simplexes = load_s_tier_simplexes()
    print(f"   Found {len(simplexes)} S-tier simplexes")

    # Identify pole pairs
    print("\n2. Identifying interesting pole pairs...")
    pairs = identify_pole_pairs(simplexes)
    print(f"   Identified {len(pairs)} pole pairs for overlap generation")

    # Generate overlaps for each pair
    print("\n3. Generating overlap synsets...")
    all_overlaps = {}

    for i, (s1, s2, pole1_name, pole2_name) in enumerate(pairs, 1):
        # Extract pole data
        dim1, pole1_type = pole1_name.rsplit('_', 1)
        dim2, pole2_type = pole2_name.rsplit('_', 1)

        pole1_key = f"{pole1_type}_pole" if pole1_type in ['positive', 'negative'] else 'neutral_homeostasis'
        pole2_key = f"{pole2_type}_pole" if pole2_type in ['positive', 'negative'] else 'neutral_homeostasis'

        pole1_data = s1['three_pole_simplex'][pole1_key]
        pole2_data = s2['three_pole_simplex'][pole2_key]

        # Get descriptions
        pole1_desc = pole1_data.get('description', f"{pole1_type} pole of {dim1}")
        pole2_desc = pole2_data.get('description', f"{pole2_type} pole of {dim2}")

        pole1_lemmas = pole1_data.get('lemmas', [])
        pole2_lemmas = pole2_data.get('lemmas', [])

        print(f"\n  [{i}/{len(pairs)}] Generating overlaps: {pole1_name} + {pole2_name}")

        overlaps = generate_overlap_synsets_for_pair(
            pole1_name, pole1_desc, pole1_lemmas,
            pole2_name, pole2_desc, pole2_lemmas,
            client
        )

        if overlaps:
            pair_key = f"{pole1_name}+{pole2_name}"
            all_overlaps[pair_key] = overlaps
            print(f"    ✓ Generated {len(overlaps)} overlap synsets")

        # Rate limiting: ~1 request per second
        import time
        time.sleep(1.2)

    # Save results
    print(f"\n4. Saving overlap synsets to {OUTPUT_PATH}...")

    output_data = {
        "metadata": {
            "description": "Synthetic synsets for S-tier simplex pole overlaps",
            "generated_date": "2025-11-23",
            "total_pairs": len(pairs),
            "total_overlaps": sum(len(v) for v in all_overlaps.values()),
            "model": "claude-3-5-sonnet-20241022"
        },
        "overlaps": all_overlaps
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"   ✓ Saved {output_data['metadata']['total_overlaps']} overlap synsets")

    # Summary
    print("\n" + "=" * 80)
    print("OVERLAP SYNSET GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal pole pairs: {len(pairs)}")
    print(f"Total overlap synsets: {output_data['metadata']['total_overlaps']}")
    print(f"Output file: {OUTPUT_PATH}")
    print("\nNext steps:")
    print("1. Review generated synsets in data/simplex_overlap_synsets.json")
    print("2. Update training data generation to use overlap synsets")
    print("3. Retrain S-tier simplexes with enhanced data")
    print("=" * 80)


if __name__ == "__main__":
    main()
