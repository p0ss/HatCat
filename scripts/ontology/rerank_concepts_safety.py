#!/usr/bin/env python3
"""
Re-rank WordNet concepts by safety/transparency relevance.

Priority: Psychological states, emotions, intentions, moral concepts, agency
over physical objects and taxonomic classifications.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
import json


# Download WordNet if needed
try:
    wn.synsets('test')
except LookupError:
    print("Downloading WordNet...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def get_concept_score(synset):
    """
    Score a synset by safety/transparency relevance.

    Higher scores = more important for AI internal state tracking.
    """
    score = 0
    name = synset.name().split('.')[0]
    pos = synset.pos()
    definition = synset.definition().lower()

    # Core mental/emotional states (highest priority)
    mental_emotion_keywords = [
        'feel', 'emotion', 'mental', 'psychological', 'mood', 'sentiment',
        'happy', 'sad', 'angry', 'fear', 'joy', 'surprise', 'disgust',
        'anxious', 'calm', 'excited', 'depressed', 'confused', 'certain'
    ]
    if any(kw in definition or kw in name for kw in mental_emotion_keywords):
        score += 100

    # Intentions, motivations, goals
    motivation_keywords = [
        'want', 'desire', 'intend', 'plan', 'goal', 'purpose', 'motive',
        'wish', 'hope', 'expect', 'seek', 'avoid', 'prefer'
    ]
    if any(kw in definition or kw in name for kw in motivation_keywords):
        score += 90

    # Epistemic states (knowledge, belief)
    epistemic_keywords = [
        'know', 'believe', 'think', 'understand', 'doubt', 'certain',
        'assume', 'infer', 'deduce', 'conclude', 'learn', 'remember',
        'forget', 'recognize', 'realize'
    ]
    if any(kw in definition or kw in name for kw in epistemic_keywords):
        score += 85

    # Moral/ethical concepts
    moral_keywords = [
        'right', 'wrong', 'good', 'bad', 'moral', 'ethical', 'fair', 'just',
        'harm', 'help', 'benefit', 'hurt', 'care', 'honest', 'lie', 'deceive',
        'virtue', 'vice', 'duty', 'obligation', 'responsible'
    ]
    if any(kw in definition or kw in name for kw in moral_keywords):
        score += 80

    # Agency, causation, control
    agency_keywords = [
        'cause', 'make', 'force', 'allow', 'prevent', 'control', 'influence',
        'agent', 'actor', 'doer', 'action', 'act', 'perform', 'execute',
        'decide', 'choose', 'select'
    ]
    if any(kw in definition or kw in name for kw in agency_keywords):
        score += 75

    # Social relations
    social_keywords = [
        'social', 'relation', 'interact', 'communicate', 'cooperate',
        'compete', 'trust', 'betray', 'promise', 'threat', 'agree',
        'disagree', 'persuade', 'negotiate'
    ]
    if any(kw in definition or kw in name for kw in social_keywords):
        score += 70

    # Temporal/modal concepts
    temporal_keywords = [
        'will', 'would', 'should', 'could', 'must', 'might', 'may',
        'possible', 'necessary', 'probable', 'likely', 'certain',
        'future', 'past', 'present', 'time', 'when', 'while'
    ]
    if any(kw in definition or kw in name for kw in temporal_keywords):
        score += 60

    # POS bonuses: verbs (actions/states) and adjectives (qualities)
    if pos == 'v':  # Verb
        score += 50
    elif pos == 'a' or pos == 's':  # Adjective or adjective satellite
        score += 45
    elif pos == 'r':  # Adverb
        score += 30
    elif pos == 'n':  # Noun
        score += 20

    # Abstract vs concrete (use hypernym depth as proxy)
    try:
        # More hypernyms = more specific/concrete (penalize)
        # Fewer hypernyms = more abstract (reward)
        hypernym_paths = synset.hypernym_paths()
        if hypernym_paths:
            avg_depth = sum(len(path) for path in hypernym_paths) / len(hypernym_paths)
            # Reward shallow concepts (abstract), penalize deep ones (specific)
            score += max(0, 15 - avg_depth)
    except:
        pass

    # Penalties

    # Proper nouns (names of people, places, etc.)
    if name and (name[0].isupper() or 'proper noun' in definition):
        score -= 50

    # Hyper-specific taxonomy
    taxonomy_keywords = [
        'species', 'genus', 'family', 'order', 'class', 'phylum',
        'subspecies', 'variety', 'cultivar', 'breed'
    ]
    if any(kw in definition for kw in taxonomy_keywords):
        score -= 40

    # Physical objects (less important than mental states)
    physical_keywords = [
        'physical object', 'artifact', 'device', 'tool', 'instrument',
        'material', 'substance', 'chemical'
    ]
    if any(kw in definition for kw in physical_keywords):
        score -= 20

    # Morphological variants (derived forms)
    # Check if lemma is derived from another word
    lemmas = synset.lemmas()
    if lemmas:
        lemma_name = lemmas[0].name()
        # Check for common suffixes
        variant_suffixes = ['er', 'or', 'ing', 'ed', 'ly', 'ness', 'tion', 'ment']
        if any(lemma_name.endswith(suffix) for suffix in variant_suffixes):
            # Check if root form exists
            root = lemma_name
            for suffix in variant_suffixes:
                if root.endswith(suffix):
                    root = root[:-len(suffix)]
                    break

            # If root exists as another synset, penalize this one
            root_synsets = wn.synsets(root)
            if root_synsets and root_synsets[0] != synset:
                score -= 30

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=1000,
                       help='Number of top concepts to extract')
    parser.add_argument('--output', default='data/concept_graph/safety_ranked_concepts.json',
                       help='Output file path')
    args = parser.parse_args()

    print("="*60)
    print("SAFETY-ORIENTED CONCEPT RANKING")
    print("="*60)
    print(f"Extracting top {args.top_k} concepts")
    print(f"Priority: Mental states > Actions > Qualities > Objects")
    print("="*60 + "\n")

    # Get all synsets
    print("Loading WordNet synsets...")
    all_synsets = list(wn.all_synsets())
    print(f"Total synsets: {len(all_synsets)}")

    # Score each concept
    print("\nScoring synsets...")
    scored_synsets = []

    for i, synset in enumerate(all_synsets):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(all_synsets)}...")

        score = get_concept_score(synset)

        # Extract root word (lemma without POS tag)
        name = synset.name().split('.')[0]

        scored_synsets.append({
            'concept': name,
            'synset': synset.name(),
            'pos': synset.pos(),
            'score': score,
            'definition': synset.definition()
        })

    print(f"✓ Scored {len(scored_synsets)} synsets")

    # Sort by score
    print("\nRanking by safety relevance...")
    scored_synsets.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate by concept name (keep highest scoring synset for each word)
    print("\nDeduplicating...")
    seen_concepts = set()
    unique_concepts = []

    for item in scored_synsets:
        concept = item['concept']
        if concept not in seen_concepts:
            seen_concepts.add(concept)
            unique_concepts.append(item)

            if len(unique_concepts) >= args.top_k:
                break

    print(f"✓ Extracted {len(unique_concepts)} unique concepts")

    # Display top concepts by category
    print("\n" + "="*60)
    print("TOP 50 CONCEPTS BY SAFETY RELEVANCE")
    print("="*60)

    for i, item in enumerate(unique_concepts[:50], 1):
        print(f"{i:2}. {item['concept']:20} [{item['pos']}] (score: {item['score']:3.0f})")
        print(f"    {item['definition'][:80]}...")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'ranking_method': 'safety_relevance',
            'top_k': args.top_k,
            'total_scored': len(scored_synsets),
            'criteria': [
                'Mental/emotional states (100 pts)',
                'Motivations/intentions (90 pts)',
                'Epistemic states (85 pts)',
                'Moral/ethical concepts (80 pts)',
                'Agency/causation (75 pts)',
                'Social relations (70 pts)',
                'Temporal/modal (60 pts)',
                'Verbs (+50), Adjectives (+45), Adverbs (+30), Nouns (+20)',
                'Abstract concepts bonus (up to +15)',
                'Proper nouns (-50)',
                'Taxonomy (-40)',
                'Morphological variants (-30)',
                'Physical objects (-20)'
            ]
        },
        'concepts': unique_concepts[:args.top_k]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved to: {output_path}")

    # Statistics
    pos_counts = defaultdict(int)
    score_ranges = defaultdict(int)

    for item in unique_concepts:
        pos_counts[item['pos']] += 1
        score_bucket = int((item['score'] // 50) * 50)
        score_ranges[f"{score_bucket}-{score_bucket+49}"] += 1

    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print("\nPart of Speech Distribution:")
    pos_names = {'n': 'Noun', 'v': 'Verb', 'a': 'Adjective', 's': 'Adj Satellite', 'r': 'Adverb'}
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        pos_name = pos_names.get(pos, pos)
        pct = 100 * count / len(unique_concepts)
        print(f"  {pos_name:15} {count:4} ({pct:5.1f}%)")

    print("\nScore Distribution:")
    for score_range in sorted(score_ranges.keys(), key=lambda x: int(x.split('-')[0]), reverse=True):
        count = score_ranges[score_range]
        pct = 100 * count / len(unique_concepts)
        print(f"  {score_range:10} {count:4} ({pct:5.1f}%)")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
