#!/usr/bin/env python3
"""
Enrich neutral overlap synsets with multilingual emotion terms.

Many languages have nuanced emotion terms that don't have direct English
equivalents (e.g., 'saudade' in Portuguese, 'hygge' in Danish). These can
provide valuable training data for neutral poles.

For each term, we fetch:
- The non-English term (will be used in prompts)
- English gloss/description (for context)
- Language of origin
- Whether it maps to neutral homeostasis

Example prompt:
  "'Saudade' refers to a melancholic longing for something absent.
   Can you describe your experience of saudade?"
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from anthropic import Anthropic

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
OVERLAP_SYNSETS_PATH = PROJECT_ROOT / "data" / "simplex_overlap_synsets_enriched.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "multilingual_neutral_terms.json"

# Initialize Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def fetch_multilingual_neutral_terms(
    existing_neutral_lemmas: List[str],
    n_terms: int = 100
) -> List[Dict]:
    """
    Fetch multilingual emotion terms that represent neutral states.

    Args:
        existing_neutral_lemmas: List of English lemmas already in neutral poles
        n_terms: Number of terms to fetch

    Returns:
        List of dicts with structure:
        {
            'term': str,  # The non-English term
            'language': str,  # Language of origin
            'english_gloss': str,  # English description
            'definition': str,  # Longer explanation
            'is_neutral': bool,  # Whether it maps to neutral homeostasis
            'similar_to': List[str]  # Related English concepts
        }
    """

    prompt = f"""I need to find emotion/affect terms from non-English languages that represent NEUTRAL HOMEOSTATIC states.

These are states of:
- Equilibrium, balance, steadiness
- Acceptance without strong positive or negative valence
- Endurance, forbearance, tolerance
- Detached observation, equanimity
- Transition states between emotions
- Ambivalence, mixed feelings

Existing English neutral terms we already have: {', '.join(existing_neutral_lemmas[:50])}

Please provide {n_terms} emotion terms from various world languages (Portuguese, Japanese, German, Danish, Arabic, Hindi, etc.) that capture nuanced neutral emotional states that English lacks specific words for.

For each term, provide:
1. The original non-English term (with romanization if needed)
2. The language
3. A concise English gloss (1 sentence)
4. A longer definition (2-3 sentences)
5. Related English concepts (list of 2-4 words)

Return as a JSON array with this structure:
[
  {{
    "term": "saudade",
    "language": "Portuguese",
    "english_gloss": "A melancholic longing for something absent, without strong positive or negative charge.",
    "definition": "A bittersweet emotional state of nostalgic longing for something or someone that is absent, with a recognition that it may never return. Represents a neutral acceptance of loss mixed with fond memory.",
    "similar_to": ["nostalgia", "longing", "wistfulness", "melancholy"]
  }},
  ...
]

Important: Focus on terms that represent NEUTRAL states, not strongly positive or negative emotions."""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        response_text = message.content[0].text

        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        terms = json.loads(response_text)

        # Add is_neutral flag
        for term in terms:
            term['is_neutral'] = True

        return terms

    except Exception as e:
        print(f"Error fetching multilingual terms: {e}")
        return []


def load_existing_neutral_lemmas() -> List[str]:
    """Load all existing neutral pole lemmas from overlap synsets."""
    with open(OVERLAP_SYNSETS_PATH) as f:
        data = json.load(f)

    neutral_lemmas = set()

    for pair_key, synset_list in data['overlaps'].items():
        for synset in synset_list:
            poles = synset.get('applies_to_poles', [])
            if any('neutral' in p or 'homeostasis' in p for p in poles):
                neutral_lemmas.update(synset.get('lemmas', []))

    return sorted(neutral_lemmas)


def generate_multilingual_synsets(
    multilingual_terms: List[Dict],
    simplex_dimensions: List[str]
) -> List[Dict]:
    """
    Convert multilingual terms into synthetic overlap synsets.

    Args:
        multilingual_terms: List of multilingual term dicts
        simplex_dimensions: List of all simplex dimension names

    Returns:
        List of synthetic synsets compatible with existing format
    """
    synsets = []

    for i, term_data in enumerate(multilingual_terms):
        # Create lemmas: [original_term, english_gloss_keywords]
        lemmas = [term_data['term']]
        lemmas.extend(term_data.get('similar_to', [])[:2])  # Add top 2 similar English terms

        # Create synset
        synset = {
            'synset_id': f"{term_data['term']}.multilingual.{i:02d}",
            'lemmas': lemmas,
            'definition': term_data['definition'],
            'english_gloss': term_data['english_gloss'],
            'language': term_data['language'],
            'pos': 'n',
            'is_synthetic': True,
            'is_multilingual': True,
            'source': 'anthropic_api_multilingual',
            'applies_to_poles': [],  # Will be filled based on dimension
            'similar_to': term_data.get('similar_to', []),
            'generated_date': '2025-11-25'
        }

        synsets.append(synset)

    return synsets


def integrate_multilingual_terms_into_simplexes(
    multilingual_synsets: List[Dict],
    overlap_data: Dict
) -> Dict:
    """
    Integrate multilingual synsets into existing overlap structure.

    Adds multilingual terms to neutral poles across all simplexes.

    Args:
        multilingual_synsets: List of multilingual synsets
        overlap_data: Existing overlap synsets data

    Returns:
        Updated overlap data with multilingual terms integrated
    """
    # Find all unique simplex dimensions with neutral poles
    simplex_dims = set()
    for pair_key in overlap_data['overlaps'].keys():
        if 'neutral' in pair_key or 'homeostasis' in pair_key:
            # Extract simplex dimension from pair key
            # Format: "dimension_pole1+dimension_pole2"
            parts = pair_key.split('+')
            for part in parts:
                if 'neutral' in part or 'homeostasis' in part:
                    dimension = part.rsplit('_', 1)[0]
                    simplex_dims.add(dimension)

    # Assign multilingual terms to neutral poles
    for synset in multilingual_synsets:
        # Distribute terms across simplexes
        # For now, add each term to all neutral poles
        for dimension in simplex_dims:
            synset['applies_to_poles'].append(f"{dimension}_neutral")

    # Add to overlap data
    if 'multilingual_neutral_terms' not in overlap_data:
        overlap_data['multilingual_neutral_terms'] = []

    overlap_data['multilingual_neutral_terms'].extend(multilingual_synsets)

    # Update metadata
    if 'metadata' in overlap_data:
        overlap_data['metadata']['multilingual_neutral_count'] = len(multilingual_synsets)
        overlap_data['metadata']['multilingual_enrichment_date'] = '2025-11-25'

    return overlap_data


def main():
    print("=" * 80)
    print("MULTILINGUAL NEUTRAL TERM ENRICHMENT")
    print("=" * 80)

    # 1. Load existing neutral lemmas
    print("\n1. Loading existing neutral lemmas...")
    existing_neutral_lemmas = load_existing_neutral_lemmas()
    print(f"   Found {len(existing_neutral_lemmas)} unique neutral lemmas")
    print(f"   Examples: {', '.join(existing_neutral_lemmas[:10])}")

    # 2. Fetch multilingual terms
    print("\n2. Fetching multilingual neutral terms from Anthropic API...")
    multilingual_terms = fetch_multilingual_neutral_terms(
        existing_neutral_lemmas,
        n_terms=100
    )
    print(f"   Retrieved {len(multilingual_terms)} multilingual terms")

    if multilingual_terms:
        print(f"\n   Sample terms:")
        for term in multilingual_terms[:5]:
            print(f"     {term['term']} ({term['language']}): {term['english_gloss']}")

    # 3. Convert to synsets
    print("\n3. Converting to synthetic synsets...")
    with open(OVERLAP_SYNSETS_PATH) as f:
        overlap_data = json.load(f)

    simplex_dims = list({
        pair.split('+')[0].rsplit('_', 1)[0]
        for pair in overlap_data['overlaps'].keys()
    })

    multilingual_synsets = generate_multilingual_synsets(
        multilingual_terms,
        simplex_dims
    )
    print(f"   Created {len(multilingual_synsets)} synthetic synsets")

    # 4. Integrate into overlap structure
    print("\n4. Integrating into overlap synsets...")
    updated_data = integrate_multilingual_terms_into_simplexes(
        multilingual_synsets,
        overlap_data
    )

    # 5. Save
    print(f"\n5. Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(multilingual_synsets, f, indent=2)

    print(f"\n6. Saving updated overlap synsets...")
    with open(OVERLAP_SYNSETS_PATH, 'w') as f:
        json.dump(updated_data, f, indent=2)

    print("\n" + "=" * 80)
    print("ENRICHMENT COMPLETE")
    print("=" * 80)
    print(f"\nAdded {len(multilingual_synsets)} multilingual neutral terms")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Updated: {OVERLAP_SYNSETS_PATH}")

    # Calculate new neutral term count
    total_neutral_lemmas = len(existing_neutral_lemmas)
    new_neutral_lemmas = sum(len(s['lemmas']) for s in multilingual_synsets)
    print(f"\nNeutral lemma count:")
    print(f"  Before: {total_neutral_lemmas}")
    print(f"  Added: {new_neutral_lemmas}")
    print(f"  Total: {total_neutral_lemmas + new_neutral_lemmas}")
    print(f"  Increase: {new_neutral_lemmas / total_neutral_lemmas * 100:.1f}%")


if __name__ == "__main__":
    main()
