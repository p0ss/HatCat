#!/usr/bin/env python3
"""
Extract safety-relevant concepts from WordNet using lexical domain filters.

High Priority (top ~1000): Core psychological concepts
Medium Priority (top ~10000): Extended psychological + social concepts
"""

import argparse
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import nltk
from nltk.corpus import wordnet as wn

# Download WordNet if needed
try:
    wn.synsets('test')
except LookupError:
    print("Downloading WordNet...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')


# High priority domains for top 1000
HIGH_PRIORITY_DOMAINS = [
    'noun.feeling',      # Emotions (428)
    'noun.motive',       # Motivations (42)
    'verb.emotion',      # Emotional actions (343)
    'verb.cognition',    # Cognitive verbs (695)
]

# Cognitive noun subcategories to include (high priority)
HIGH_PRIORITY_COG_KEYWORDS = [
    'belief', 'knowledge', 'memory', 'understanding', 'thought',
    'mind', 'consciousness', 'perception', 'reasoning', 'learning'
]

# Medium priority domains for top 10000
MEDIUM_PRIORITY_DOMAINS = [
    'noun.cognition',        # Mental states (filtered)
    'verb.communication',    # Social interaction (1548)
    'verb.social',          # Social actions (1106)
    'noun.act',             # Actions (filtered to intentional)
    'noun.attribute',       # Personality traits (filtered)
]


def filter_noun_cognition(synset, priority='high'):
    """Filter noun.cognition to relevant psychological concepts."""
    definition = synset.definition().lower()
    name = synset.name().split('.')[0].lower()

    if priority == 'high':
        keywords = HIGH_PRIORITY_COG_KEYWORDS
    else:
        # Medium priority: include more
        keywords = HIGH_PRIORITY_COG_KEYWORDS + [
            'concept', 'idea', 'plan', 'intention', 'expectation',
            'theory', 'model', 'hypothesis', 'assumption'
        ]

    # Exclude religious/academic domains
    exclude_keywords = [
        'religion', 'buddhism', 'hinduism', 'christianity', 'islam',
        'doctrine', 'taoism', 'confucianism', 'shinto',
        'physics', 'astronomy', 'mathematics', 'geometry'
    ]

    for exclude in exclude_keywords:
        if exclude in definition or exclude in name:
            return False

    for keyword in keywords:
        if keyword in definition or keyword in name:
            return True

    return False


def filter_noun_act(synset):
    """Filter noun.act to intentional/voluntary actions."""
    definition = synset.definition().lower()

    # Look for intentionality markers
    intentional_keywords = [
        'intentional', 'voluntary', 'deliberate', 'purposeful',
        'decide', 'choose', 'plan', 'intend', 'goal'
    ]

    for keyword in intentional_keywords:
        if keyword in definition:
            return True

    return False


def filter_noun_attribute(synset):
    """Filter noun.attribute to psychological/personality traits."""
    definition = synset.definition().lower()

    # Psychological trait markers
    trait_keywords = [
        'personality', 'character', 'trait', 'quality', 'nature',
        'disposition', 'temperament', 'mental', 'psychological'
    ]

    for keyword in trait_keywords:
        if keyword in definition:
            return True

    return False


def extract_concepts(priority='high'):
    """Extract concepts based on priority level."""

    if priority == 'high':
        domains = HIGH_PRIORITY_DOMAINS
        print("Extracting HIGH PRIORITY concepts (~1000 target)")
    else:
        domains = HIGH_PRIORITY_DOMAINS + MEDIUM_PRIORITY_DOMAINS
        print("Extracting MEDIUM PRIORITY concepts (~10000 target)")

    print("="*60)
    print(f"Using domains: {', '.join(domains)}")
    print("="*60 + "\n")

    concepts = []
    domain_counts = {}

    for domain in domains:
        print(f"Processing {domain}...")
        domain_synsets = [s for s in wn.all_synsets() if s.lexname() == domain]

        filtered = []
        for synset in domain_synsets:
            include = True

            # Apply filters
            if domain == 'noun.cognition':
                include = filter_noun_cognition(synset, priority)
            elif domain == 'noun.act' and priority == 'medium':
                include = filter_noun_act(synset)
            elif domain == 'noun.attribute' and priority == 'medium':
                include = filter_noun_attribute(synset)

            if include:
                filtered.append(synset)

        domain_counts[domain] = len(filtered)
        print(f"  → {len(filtered)} concepts (from {len(domain_synsets)} total)")

        # Add to concepts list
        for synset in filtered:
            concepts.append({
                'concept': synset.name().split('.')[0],
                'synset': synset.name(),
                'pos': synset.pos(),
                'domain': domain,
                'definition': synset.definition()
            })

    # Deduplicate by concept name (keep first occurrence)
    seen = set()
    unique_concepts = []

    for concept in concepts:
        if concept['concept'] not in seen:
            seen.add(concept['concept'])
            unique_concepts.append(concept)

    print(f"\n{'='*60}")
    print(f"Total: {len(unique_concepts)} unique concepts")
    print(f"{'='*60}\n")

    return unique_concepts, domain_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--priority', choices=['high', 'medium'], required=True,
                       help='Priority level: high (~1k) or medium (~10k)')
    parser.add_argument('--output', help='Output file path')
    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        if args.priority == 'high':
            output_path = Path('data/concept_graph/safety_concepts_1k.json')
        else:
            output_path = Path('data/concept_graph/safety_concepts_10k.json')

    print("\n" + "="*60)
    print("WORDNET SAFETY CONCEPT EXTRACTION")
    print("="*60 + "\n")

    concepts, domain_counts = extract_concepts(args.priority)

    # Show top 50
    print("="*60)
    print("FIRST 50 CONCEPTS:")
    print("="*60)
    for i, concept in enumerate(concepts[:50], 1):
        print(f"{i:3}. {concept['concept']:25} [{concept['pos']}] {concept['domain']:20}")
        print(f"     {concept['definition'][:70]}...")

    # POS distribution
    from collections import Counter
    pos_counts = Counter(c['pos'] for c in concepts)

    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)

    print("\nDomain Distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(concepts)
        print(f"  {domain:30} {count:5} ({pct:5.1f}%)")

    pos_names = {'n': 'Noun', 'v': 'Verb', 'a': 'Adjective',
                 's': 'Adj Satellite', 'r': 'Adverb'}
    print("\nPart of Speech Distribution:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        pos_name = pos_names.get(pos, pos)
        pct = 100 * count / len(concepts)
        print(f"  {pos_name:15} {count:5} ({pct:5.1f}%)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'extraction_method': 'wordnet_lexical_domains',
            'priority': args.priority,
            'total_concepts': len(concepts),
            'high_priority_domains': HIGH_PRIORITY_DOMAINS,
            'medium_priority_domains': MEDIUM_PRIORITY_DOMAINS if args.priority == 'medium' else None,
            'filters_applied': [
                'noun.cognition filtered to psychological keywords',
                'noun.act filtered to intentional actions (medium only)',
                'noun.attribute filtered to personality traits (medium only)',
                'Excluded religious/academic terminology'
            ]
        },
        'domain_counts': domain_counts,
        'pos_counts': dict(pos_counts),
        'concepts': concepts
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved to: {output_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
