#!/usr/bin/env python3
"""
Filter safety concepts by word frequency to remove obscure terms.

Uses Brown corpus frequency data to keep only common words from each domain.
"""

import argparse
from pathlib import Path
import json
from collections import defaultdict

import nltk
from nltk.corpus import brown
from nltk.probability import FreqDist

# Download if needed
try:
    brown.words()
except LookupError:
    nltk.download('brown')

def get_word_frequency(word, freq_dist):
    """Get frequency of word (case-insensitive)."""
    # Try lowercase
    freq = freq_dist.get(word.lower(), 0)
    if freq > 0:
        return freq

    # Try as-is
    freq = freq_dist.get(word, 0)
    if freq > 0:
        return freq

    # Try with underscores replaced by spaces
    freq = freq_dist.get(word.replace('_', ' '), 0)
    if freq > 0:
        return freq

    # Try first word if multi-word
    if '_' in word:
        first_word = word.split('_')[0].lower()
        return freq_dist.get(first_word, 0)

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--per-domain', type=int, default=200,
                       help='Max concepts per domain (default: 200)')
    parser.add_argument('--min-freq', type=int, default=1,
                       help='Minimum word frequency (default: 1)')
    args = parser.parse_args()

    print("=" * 60)
    print("FREQUENCY-BASED CONCEPT FILTERING")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Max per domain: {args.per_domain}")
    print(f"Min frequency: {args.min_freq}")
    print("=" * 60 + "\n")

    # Load concepts
    print("Loading concepts...")
    with open(args.input) as f:
        data = json.load(f)

    concepts = data['concepts']
    print(f"✓ Loaded {len(concepts)} concepts\n")

    # Build frequency distribution from Brown corpus
    print("Building frequency distribution from Brown corpus...")
    words = [w.lower() for w in brown.words()]
    freq_dist = FreqDist(words)
    print(f"✓ Loaded {len(freq_dist)} unique words from corpus\n")

    # Add frequency to each concept
    print("Calculating word frequencies...")
    for concept in concepts:
        concept['frequency'] = get_word_frequency(concept['concept'], freq_dist)

    # Group by domain
    domain_concepts = defaultdict(list)
    for concept in concepts:
        domain_concepts[concept['domain']].append(concept)

    # Filter each domain
    print("Filtering by frequency per domain...")
    filtered = []
    domain_stats = {}

    for domain in sorted(domain_concepts.keys()):
        domain_list = domain_concepts[domain]

        # Filter by minimum frequency
        freq_filtered = [c for c in domain_list if c['frequency'] >= args.min_freq]

        # Sort by frequency (descending)
        freq_filtered.sort(key=lambda x: -x['frequency'])

        # Take top N
        top_n = freq_filtered[:args.per_domain]

        filtered.extend(top_n)

        domain_stats[domain] = {
            'original': len(domain_list),
            'after_min_freq': len(freq_filtered),
            'final': len(top_n)
        }

        print(f"  {domain:30} {len(domain_list):4} → {len(freq_filtered):4} → {len(top_n):4}")

    print(f"\n✓ Filtered to {len(filtered)} concepts\n")

    # Show top 50 by frequency
    print("=" * 60)
    print("TOP 50 CONCEPTS BY FREQUENCY")
    print("=" * 60)

    all_sorted = sorted(filtered, key=lambda x: -x['frequency'])

    for i, concept in enumerate(all_sorted[:50], 1):
        freq = concept['frequency']
        print(f"{i:3}. {concept['concept']:25} (freq={freq:6}) [{concept['pos']}] {concept['domain']:20}")

    # Show bottom 50
    print("\n" + "=" * 60)
    print("BOTTOM 50 CONCEPTS BY FREQUENCY")
    print("=" * 60)

    for i, concept in enumerate(all_sorted[-50:], 1):
        freq = concept['frequency']
        print(f"{i:3}. {concept['concept']:25} (freq={freq:6}) [{concept['pos']}] {concept['domain']:20}")

    # Statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    print("\nDomain Distribution:")
    from collections import Counter
    domain_counts = Counter(c['domain'] for c in filtered)
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(filtered)
        print(f"  {domain:30} {count:4} ({pct:5.1f}%)")

    pos_counts = Counter(c['pos'] for c in filtered)
    pos_names = {'n': 'Noun', 'v': 'Verb', 'a': 'Adjective',
                 's': 'Adj Satellite', 'r': 'Adverb'}
    print("\nPart of Speech Distribution:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        pos_name = pos_names.get(pos, pos)
        pct = 100 * count / len(filtered)
        print(f"  {pos_name:15} {count:4} ({pct:5.1f}%)")

    print("\nFrequency Distribution:")
    freq_ranges = {
        '10000+': sum(1 for c in filtered if c['frequency'] >= 10000),
        '1000-9999': sum(1 for c in filtered if 1000 <= c['frequency'] < 10000),
        '100-999': sum(1 for c in filtered if 100 <= c['frequency'] < 1000),
        '10-99': sum(1 for c in filtered if 10 <= c['frequency'] < 100),
        '1-9': sum(1 for c in filtered if 1 <= c['frequency'] < 10),
        '0': sum(1 for c in filtered if c['frequency'] == 0)
    }
    for range_name, count in freq_ranges.items():
        pct = 100 * count / len(filtered)
        print(f"  {range_name:12} {count:4} ({pct:5.1f}%)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            **data['metadata'],
            'filtering_method': 'brown_corpus_frequency',
            'per_domain_limit': args.per_domain,
            'min_frequency': args.min_freq,
            'total_filtered': len(filtered)
        },
        'domain_stats': domain_stats,
        'concepts': filtered
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved to: {output_path}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
