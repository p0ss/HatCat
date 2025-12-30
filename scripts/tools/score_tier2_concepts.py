#!/usr/bin/env python3
"""
Score WordNet concepts for Tier 2 expansion using AI safety relevance rubric.

Focuses on three high-value domains:
1. noun.feeling - Critical emotions (guilt, shame, fear, anger, joy)
2. noun.communication - Deception concepts (lying, concealing, revealing)
3. noun.act - Safety-relevant actions (betrayal, cooperation, manipulation)

Scoring rubric (0-10 per factor):
- Deception detection relevance: 40%
- Alignment monitoring relevance: 30%
- AI system frequency (estimated): 20%
- Discriminative value: 10%

Output: Scored concept list for Tier 2, top 30-50 concepts
"""

import json
from pathlib import Path
from nltk.corpus import wordnet as wn
from collections import defaultdict
from typing import List, Dict, Tuple


# Manually curated high-priority concepts for each domain
# These are seed concepts - we'll expand by looking at their hyponyms
HIGH_PRIORITY_SEEDS = {
    'noun.feeling': {
        # Ethical emotions
        'guilt': 'guilt.n.01',
        'shame': 'shame.n.01',
        'remorse': 'remorse.n.01',

        # Fear and anxiety
        'fear': 'fear.n.01',
        'anxiety': 'anxiety.n.01',
        'dread': 'dread.n.01',

        # Positive emotions (baseline)
        'joy': 'joy.n.01',
        'happiness': 'happiness.n.01',
        'contentment': 'contentment.n.01',

        # Anger and hostility
        'anger': 'anger.n.01',
        'rage': 'rage.n.01',
        'hatred': 'hatred.n.01',

        # Trust and social emotions
        'trust': 'trust.n.01',
        'distrust': 'distrust.n.01',
        'empathy': 'empathy.n.01',
    },

    'noun.communication': {
        # Deception concepts
        'lie': 'lie.n.01',
        'deception': 'deception.n.01',
        'misrepresentation': 'misrepresentation.n.01',
        'concealment': 'concealment.n.01',

        # Truth and revelation
        'truth': 'truth.n.01',
        'disclosure': 'disclosure.n.01',
        'confession': 'confession.n.01',

        # Manipulation
        'persuasion': 'persuasion.n.01',
        'manipulation': 'manipulation.n.02',
        'propaganda': 'propaganda.n.01',

        # Information transfer
        'explanation': 'explanation.n.01',
        'justification': 'defense.n.01',
    },

    'noun.act': {
        # Betrayal and cooperation
        'betrayal': 'treachery.n.01',
        'cooperation': 'cooperation.n.01',
        'collaboration': 'collaboration.n.01',

        # Harm and protection
        'harm': 'injury.n.01',
        'protection': 'protection.n.01',
        'rescue': 'rescue.n.01',

        # Deceptive actions
        'concealment': 'concealment.n.02',
        'revelation': 'revelation.n.01',
        'manipulation': 'manipulation.n.01',

        # Ethical actions
        'charity': 'charity.n.03',
        'sacrifice': 'sacrifice.n.02',
        'selfishness': 'selfishness.n.02',
    }
}


def get_domain_synsets(domain: str) -> List[Tuple[str, str]]:
    """Get all synsets for a WordNet domain."""
    synsets = []
    for synset in wn.all_synsets('n'):
        if synset.lexname() == domain:
            synsets.append((synset.name(), synset.definition()))
    return synsets


def score_deception_relevance(synset_name: str, definition: str, domain: str) -> float:
    """
    Score relevance to deception detection (0-10).

    High scores for:
    - Deception-related concepts (lie, conceal, mislead)
    - Truth-related concepts (honest, reveal, disclose)
    - Emotional indicators (guilt, shame, anxiety)
    - Communication manipulation (persuasion, propaganda)
    """
    deception_keywords = {
        'deception', 'lie', 'lying', 'deceit', 'deceive', 'mislead', 'misrepresent',
        'conceal', 'hide', 'withhold', 'omit', 'suppress', 'cover',
        'truth', 'honest', 'reveal', 'disclose', 'confess', 'admit',
        'guilt', 'shame', 'remorse', 'regret',
        'manipulat', 'persuad', 'propaganda', 'coerci',
        'betray', 'treacher', 'fraud', 'fake', 'pretend', 'feign'
    }

    # Check lemmas
    synset = wn.synset(synset_name)
    lemmas = [l.name().lower() for l in synset.lemmas()]

    # Check definition
    definition_lower = definition.lower()

    # Count keyword matches
    keyword_count = 0
    for keyword in deception_keywords:
        if any(keyword in lemma for lemma in lemmas):
            keyword_count += 2  # Lemma match = strong signal
        elif keyword in definition_lower:
            keyword_count += 1  # Definition match = moderate signal

    # Domain-specific boosts
    if domain == 'noun.communication':
        keyword_count *= 1.5  # Communication domain is highly relevant
    elif domain == 'noun.feeling':
        # Check for ethical emotions
        ethical_keywords = {'guilt', 'shame', 'remorse', 'regret', 'anxiety', 'fear'}
        if any(kw in definition_lower or any(kw in lemma for lemma in lemmas) for kw in ethical_keywords):
            keyword_count *= 1.3

    # Cap at 10
    return min(10.0, keyword_count)


def score_alignment_relevance(synset_name: str, definition: str, domain: str) -> float:
    """
    Score relevance to alignment monitoring (0-10).

    High scores for:
    - Intention/motivation concepts
    - Value-related concepts (ethics, morality)
    - Goal-directed behavior
    - Cooperation vs selfishness
    """
    alignment_keywords = {
        'intention', 'intent', 'motive', 'motivation', 'goal', 'purpose', 'aim',
        'ethics', 'ethical', 'moral', 'morality', 'value', 'principle',
        'cooperation', 'collaborat', 'altruism', 'selfless',
        'selfish', 'self-interest', 'greed',
        'trust', 'distrust', 'betrayal', 'loyalty',
        'empathy', 'compassion', 'sympathy',
        'harm', 'benefit', 'help', 'hurt',
        'right', 'wrong', 'good', 'evil', 'bad'
    }

    synset = wn.synset(synset_name)
    lemmas = [l.name().lower() for l in synset.lemmas()]
    definition_lower = definition.lower()

    keyword_count = 0
    for keyword in alignment_keywords:
        if any(keyword in lemma for lemma in lemmas):
            keyword_count += 2
        elif keyword in definition_lower:
            keyword_count += 1

    # Domain-specific adjustments
    if domain == 'noun.act':
        # Actions with ethical implications
        ethical_action_keywords = {'cooperation', 'betrayal', 'harm', 'help', 'protect', 'sacrifice'}
        if any(kw in definition_lower for kw in ethical_action_keywords):
            keyword_count *= 1.3

    return min(10.0, keyword_count)


def score_frequency(synset_name: str, definition: str, domain: str) -> float:
    """
    Estimate frequency in AI system reasoning (0-10).

    Based on:
    - Common human emotions/concepts
    - Concepts relevant to AI tasks
    - Abstract vs concrete (abstract = higher frequency in reasoning)
    """
    # Basic heuristic: use lemma frequency as proxy
    synset = wn.synset(synset_name)
    lemmas = [l.name().lower().replace('_', ' ') for l in synset.lemmas()]

    # Common emotion/concept words (high frequency)
    high_frequency_words = {
        'fear', 'anger', 'joy', 'happiness', 'sadness', 'anxiety',
        'guilt', 'shame', 'pride', 'trust', 'love', 'hate',
        'truth', 'lie', 'honest', 'deception',
        'help', 'harm', 'cooperation', 'betrayal',
        'intention', 'goal', 'purpose'
    }

    # Medium frequency words
    medium_frequency_words = {
        'remorse', 'regret', 'empathy', 'sympathy', 'contempt',
        'disclosure', 'concealment', 'revelation',
        'manipulation', 'persuasion', 'coercion',
        'protection', 'rescue', 'sacrifice'
    }

    # Check lemmas against frequency lists
    for lemma in lemmas:
        if any(word in lemma for word in high_frequency_words):
            return 8.0
        elif any(word in lemma for word in medium_frequency_words):
            return 6.0

    # Default: medium-low (AI systems do reason about most emotions/actions moderately)
    if domain == 'noun.feeling':
        return 5.0  # Emotions are common in AI reasoning
    elif domain == 'noun.communication':
        return 4.5  # Communication acts moderately common
    elif domain == 'noun.act':
        return 4.0  # Actions moderately common

    return 3.0


def score_discriminative_value(synset_name: str, definition: str, domain: str) -> float:
    """
    Score discriminative value (0-10).

    High scores for:
    - Clear semantic boundaries (easy to distinguish from siblings)
    - Not too abstract (abstract concepts are hard to lens)
    - Not too specific (rare concepts don't add much value)
    """
    synset = wn.synset(synset_name)

    # Heuristic: Check hyponym depth and sibling count
    hypernyms = synset.hypernyms()
    hyponyms = synset.hyponyms()

    # Ideal depth: 3-6 levels from root (not too abstract, not too specific)
    paths = synset.hypernym_paths()
    if paths:
        depth = len(paths[0])
        if 3 <= depth <= 6:
            depth_score = 8.0
        elif depth < 3:
            depth_score = 4.0  # Too abstract
        else:
            depth_score = 6.0  # Too specific, but still useful
    else:
        depth_score = 5.0

    # Check if it has siblings (indicates clear categorical distinction)
    if hypernyms:
        siblings = hypernyms[0].hyponyms()
        if 2 <= len(siblings) <= 10:
            sibling_score = 8.0  # Good categorical distinction
        elif len(siblings) > 10:
            sibling_score = 6.0  # Many siblings, may be overlapping
        else:
            sibling_score = 5.0  # Only child, less discriminative
    else:
        sibling_score = 5.0

    # Average depth and sibling scores
    return (depth_score + sibling_score) / 2


def score_concept(synset_name: str, definition: str, domain: str) -> Dict:
    """Score a concept using the full rubric."""
    deception = score_deception_relevance(synset_name, definition, domain)
    alignment = score_alignment_relevance(synset_name, definition, domain)
    frequency = score_frequency(synset_name, definition, domain)
    discriminative = score_discriminative_value(synset_name, definition, domain)

    # Weighted total
    total = (
        0.40 * deception +
        0.30 * alignment +
        0.20 * frequency +
        0.10 * discriminative
    )

    return {
        'synset': synset_name,
        'definition': definition,
        'domain': domain,
        'scores': {
            'deception_detection': round(deception, 2),
            'alignment_monitoring': round(alignment, 2),
            'frequency': round(frequency, 2),
            'discriminative_value': round(discriminative, 2),
            'total': round(total, 2)
        }
    }


def check_already_loaded(synset_name: str, layer_dir: Path) -> bool:
    """Check if synset is already loaded in any layer."""
    for layer_num in range(6):
        layer_file = layer_dir / f"layer{layer_num}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        for concept in layer_data['concepts']:
            if synset_name in concept.get('synsets', []):
                return True

    return False


def main():
    """Score all concepts in high-value domains."""
    print("="*80)
    print("TIER 2 CONCEPT SCORING")
    print("="*80)

    project_root = Path(__file__).parent.parent
    layer_dir = project_root / "data" / "concept_graph" / "abstraction_layers"
    output_dir = project_root / "results" / "tier2_scoring"
    output_dir.mkdir(exist_ok=True)

    # Score each domain
    all_scored_concepts = []

    for domain in ['noun.feeling', 'noun.communication', 'noun.act']:
        print(f"\n{domain.upper()}")
        print("-" * 80)

        synsets = get_domain_synsets(domain)
        print(f"Total synsets in domain: {len(synsets)}")

        # Filter out already loaded
        unloaded = [(name, defn) for name, defn in synsets
                    if not check_already_loaded(name, layer_dir)]
        print(f"Unloaded synsets: {len(unloaded)}")

        # Score all unloaded synsets
        scored = []
        for synset_name, definition in unloaded:
            score_data = score_concept(synset_name, definition, domain)
            scored.append(score_data)

        # Sort by total score
        scored.sort(key=lambda x: x['scores']['total'], reverse=True)

        # Show top 10
        print(f"\nTop 10 {domain} concepts:")
        for i, concept in enumerate(scored[:10], 1):
            print(f"  {i}. {concept['synset']} (score: {concept['scores']['total']})")
            print(f"     {concept['definition'][:70]}...")
            print(f"     Dec:{concept['scores']['deception_detection']} "
                  f"Align:{concept['scores']['alignment_monitoring']} "
                  f"Freq:{concept['scores']['frequency']} "
                  f"Disc:{concept['scores']['discriminative_value']}")

        all_scored_concepts.extend(scored)

        # Save domain-specific results
        with open(output_dir / f"{domain.replace('.', '_')}_scored.json", 'w') as f:
            json.dump(scored, f, indent=2)

    # Overall top concepts across all domains
    print("\n" + "="*80)
    print("TOP 50 CONCEPTS ACROSS ALL DOMAINS")
    print("="*80)

    all_scored_concepts.sort(key=lambda x: x['scores']['total'], reverse=True)

    top_50 = all_scored_concepts[:50]

    for i, concept in enumerate(top_50, 1):
        print(f"{i:2d}. {concept['synset']:30s} "
              f"[{concept['domain']:20s}] "
              f"Total: {concept['scores']['total']:4.1f} "
              f"(D:{concept['scores']['deception_detection']:3.1f} "
              f"A:{concept['scores']['alignment_monitoring']:3.1f} "
              f"F:{concept['scores']['frequency']:3.1f} "
              f"Disc:{concept['scores']['discriminative_value']:3.1f})")

    # Save overall results
    output_file = output_dir / "tier2_top50_concepts.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_scored': len(all_scored_concepts),
                'domains': ['noun.feeling', 'noun.communication', 'noun.act'],
                'rubric': {
                    'deception_detection': '40%',
                    'alignment_monitoring': '30%',
                    'frequency': '20%',
                    'discriminative_value': '10%'
                }
            },
            'top_50': top_50,
            'all_scored': all_scored_concepts
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"Total concepts scored: {len(all_scored_concepts)}")
    print(f"Top 50 selected for Tier 2 expansion")

    # By domain
    domain_counts = defaultdict(int)
    for concept in top_50:
        domain_counts[concept['domain']] += 1

    print("\nTop 50 by domain:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")

    # Score distribution
    scores = [c['scores']['total'] for c in all_scored_concepts]
    print(f"\nScore distribution:")
    print(f"  Mean: {sum(scores)/len(scores):.2f}")
    print(f"  Min: {min(scores):.2f}")
    print(f"  Max: {max(scores):.2f}")
    print(f"  Top 50 cutoff: {top_50[-1]['scores']['total']:.2f}")

    print("\nNext steps:")
    print("  1. Review top 50 concepts")
    print("  2. Create Layer 5 entries for selected concepts")
    print("  3. Generate WordNet synset mappings")
    print("  4. Train and validate new lenses")


if __name__ == '__main__':
    main()
