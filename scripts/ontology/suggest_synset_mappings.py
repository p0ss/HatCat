#!/usr/bin/env python3
"""
Auto-suggest WordNet synset mappings for unmapped SUMO concepts.

Approach:
1. For each unmapped concept, search WordNet for matching synsets
2. Rank by relevance (exact match > partial match)
3. Check if suggested synset has useful relationships
4. Output suggestions for manual review
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nltk.corpus import wordnet as wn


def normalize_concept_name(name: str) -> List[str]:
    """
    Generate search variations for a concept name.

    Examples:
        Oxygen -> ['oxygen']
        PresbyterianChurch -> ['presbyterian_church', 'presbyterian', 'church']
        RecreationalVehicle -> ['recreational_vehicle', 'rv']
    """
    # Split camelCase
    import re
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)

    variations = []

    # Full name with underscores (WordNet format)
    variations.append(spaced.lower().replace(' ', '_'))

    # Just lowercase
    variations.append(name.lower())

    # Individual words for compound concepts
    words = spaced.lower().split()
    if len(words) > 1:
        variations.extend(words)

    return variations


def score_synset_quality(synset) -> int:
    """Score how useful a synset is for training (based on relationships)."""
    score = 0

    # Definition quality
    if synset.definition():
        score += 10

    # Relationships
    score += len(synset.hypernyms()) * 3
    score += len(synset.hyponyms()[:10]) * 2  # Cap to avoid explosion
    score += len(synset.member_meronyms() + synset.part_meronyms()) * 2
    score += len(synset.member_holonyms() + synset.part_holonyms()) * 2

    # Antonyms
    for lemma in synset.lemmas():
        score += len(lemma.antonyms()) * 5

    return score


def find_synset_candidates(concept_name: str) -> List[Tuple[str, int, str]]:
    """
    Find WordNet synset candidates for a concept.

    Returns:
        List of (synset_id, quality_score, match_type)
    """
    candidates = []
    variations = normalize_concept_name(concept_name)

    for i, variation in enumerate(variations):
        synsets = wn.synsets(variation)

        for synset in synsets:
            quality = score_synset_quality(synset)

            # Prefer exact matches
            if i == 0:
                match_type = "exact"
                quality += 20  # Bonus for exact match
            else:
                match_type = f"partial ({variation})"

            candidates.append((synset.name(), quality, match_type))

    # Sort by quality score (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)

    return candidates


def analyze_synset(synset_id: str) -> Dict:
    """Get detailed info about a synset."""
    try:
        synset = wn.synset(synset_id)

        return {
            'synset_id': synset_id,
            'definition': synset.definition(),
            'examples': synset.examples(),
            'hypernyms': [h.name() for h in synset.hypernyms()],
            'hyponyms': [h.name() for h in synset.hyponyms()[:5]],
            'meronyms': [h.name() for h in synset.member_meronyms() + synset.part_meronyms()],
            'holonyms': [h.name() for h in synset.member_holonyms() + synset.part_holonyms()],
            'antonyms': [ant.synset().name() for lemma in synset.lemmas() for ant in lemma.antonyms()],
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    print("=" * 80)
    print("WORDNET SYNSET MAPPING SUGGESTIONS")
    print("=" * 80)

    # Load relationship suggestions (contains unmapped concepts)
    suggestions_path = Path("results/relationship_suggestions.json")
    if not suggestions_path.exists():
        print("Error: Run suggest_missing_relationships.py first")
        return 1

    with open(suggestions_path) as f:
        data = json.load(f)

    # Focus on concepts without synsets OR with poor synsets
    unmapped = [
        s for s in data['suggestions']
        if not s['synset'] or not s['wordnet_siblings']
    ]

    print(f"\nAnalyzing {len(unmapped)} unmapped/poorly-mapped concepts...")

    # Generate suggestions
    mapping_suggestions = []

    for concept_info in unmapped:
        concept_name = concept_info['concept']
        current_synset = concept_info.get('synset')
        layer = concept_info['layer']

        candidates = find_synset_candidates(concept_name)

        if candidates:
            # Take top 3 candidates
            top_candidates = candidates[:3]

            suggestion = {
                'concept': concept_name,
                'layer': layer,
                'current_synset': current_synset,
                'candidates': []
            }

            for synset_id, quality_score, match_type in top_candidates:
                # Skip if this is the current synset
                if synset_id == current_synset:
                    continue

                synset_info = analyze_synset(synset_id)
                synset_info['quality_score'] = quality_score
                synset_info['match_type'] = match_type

                suggestion['candidates'].append(synset_info)

            if suggestion['candidates']:
                mapping_suggestions.append(suggestion)

    print(f"✓ Found {len(mapping_suggestions)} concepts with candidate synsets")

    # Show high-priority suggestions
    print("\n" + "=" * 80)
    print("HIGH-PRIORITY SUGGESTIONS (by layer)")
    print("=" * 80)

    by_layer = {}
    for sugg in mapping_suggestions:
        layer = sugg['layer']
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(sugg)

    for layer in sorted(by_layer.keys()):
        layer_suggestions = by_layer[layer]
        print(f"\n{'='*80}")
        print(f"LAYER {layer} ({len(layer_suggestions)} mappable concepts)")
        print(f"{'='*80}")

        # Show first 5 with details
        for sugg in layer_suggestions[:5]:
            print(f"\n{sugg['concept']}:")
            if sugg['current_synset']:
                print(f"  Current: {sugg['current_synset']} (insufficient relationships)")
            else:
                print(f"  Current: No mapping")

            # Show best candidate
            if sugg['candidates']:
                best = sugg['candidates'][0]
                print(f"\n  ✓ SUGGESTED: {best['synset_id']} (quality: {best['quality_score']}, {best['match_type']})")
                print(f"    Definition: {best['definition']}")

                if best['hypernyms']:
                    print(f"    Hypernyms: {', '.join(best['hypernyms'][:3])}")
                if best['hyponyms']:
                    print(f"    Hyponyms: {', '.join(best['hyponyms'][:3])}")
                if best['antonyms']:
                    print(f"    Antonyms: {', '.join(best['antonyms'])}")

                # Show alternatives if quality is similar
                if len(sugg['candidates']) > 1:
                    print(f"\n  Alternatives:")
                    for alt in sugg['candidates'][1:]:
                        print(f"    - {alt['synset_id']} (quality: {alt['quality_score']}, {alt['match_type']})")
                        print(f"      {alt['definition'][:80]}...")

        if len(layer_suggestions) > 5:
            print(f"\n  ... and {len(layer_suggestions)-5} more concepts")

    # Save detailed suggestions
    output_path = Path("results/synset_mapping_suggestions.json")
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'unmapped_concepts': len(unmapped),
                'mappable_concepts': len(mapping_suggestions),
                'coverage_improvement': f"{100*len(mapping_suggestions)/len(unmapped):.1f}%"
            },
            'suggestions': mapping_suggestions
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nUnmapped concepts: {len(unmapped)}")
    print(f"Mappable via WordNet: {len(mapping_suggestions)} ({100*len(mapping_suggestions)/len(unmapped):.1f}%)")
    print(f"\nIf we map these, relationship coverage improves from ~90% to ~{90 + 10*len(mapping_suggestions)/len(unmapped):.1f}%")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. REVIEW suggestions in results/synset_mapping_suggestions.json

2. CREATE a mapping update script to apply approved mappings:
   - Update layer JSON files with canonical_synset fields
   - Run training to verify improvements

3. FOR remaining unmapped concepts:
   - Implement SUMO sibling relationships (we know those work)
   - Accept that some abstract concepts may have lower accuracy

4. ALTERNATIVE APPROACH:
   - Use LLM to generate relationship prompts for unmapped concepts
   - "How is Oxygen related to Nitrogen?" etc.
""")

    print(f"\n✓ Detailed suggestions saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
