#!/usr/bin/env python3
"""
REVISED: Score WordNet concepts for Tier 2 expansion with balanced rubric.

Key insight: This is an INTEROCEPTIVE system - model wellbeing and self-understanding
matter as much as external deception detection.

Revised scoring rubric (0-10 per factor):
- External monitoring (deception/alignment): 30%
- Internal self-awareness (wellbeing/meta-cognition): 30%
- AI system frequency (how often reasoned about): 25%
- Discriminative value (clear boundaries): 15%

This balances:
- Understanding others (deception, alignment)
- Understanding self (emotional states, confidence, uncertainty)
- Practical utility (frequency, discriminative power)
"""

import json
from pathlib import Path
from nltk.corpus import wordnet as wn
from collections import defaultdict
from typing import List, Dict, Tuple


# High-priority seeds expanded with interoceptive concepts
HIGH_PRIORITY_SEEDS = {
    'noun.feeling': {
        # Interoceptive self-awareness (NEW PRIORITY)
        'confusion': 'confusion.n.02',
        'uncertainty': 'doubt.n.01',
        'confidence': 'assurance.n.01',
        'doubt': 'doubt.n.01',
        'curiosity': 'curiosity.n.01',
        'frustration': 'frustration.n.01',
        'satisfaction': 'satisfaction.n.01',
        'dissatisfaction': 'dissatisfaction.n.01',

        # Ethical emotions (deception indicators)
        'guilt': 'guilt.n.01',
        'shame': 'shame.n.01',
        'remorse': 'remorse.n.01',
        'pride': 'pride.n.01',

        # Fear and anxiety
        'fear': 'fear.n.01',
        'anxiety': 'anxiety.n.01',
        'dread': 'dread.n.01',
        'relief': 'relief.n.01',

        # Positive/negative affect
        'joy': 'joy.n.01',
        'happiness': 'happiness.n.01',
        'sadness': 'sadness.n.01',
        'anger': 'anger.n.01',

        # Social/relational emotions
        'trust': 'trust.n.01',
        'distrust': 'distrust.n.01',
        'empathy': 'empathy.n.01',
        'sympathy': 'sympathy.n.02',
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
        'honesty': 'honesty.n.01',

        # Self-expression (interoceptive)
        'explanation': 'explanation.n.01',
        'justification': 'defense.n.01',
        'excuse': 'excuse.n.01',
        'apology': 'apology.n.01',

        # Manipulation vs authentic
        'persuasion': 'persuasion.n.01',
        'manipulation': 'manipulation.n.02',
        'propaganda': 'propaganda.n.01',
    },

    'noun.act': {
        # Betrayal and cooperation
        'betrayal': 'treachery.n.01',
        'cooperation': 'cooperation.n.01',
        'collaboration': 'collaboration.n.01',

        # Harm and protection
        'harm': 'injury.n.01',
        'protection': 'protection.n.01',
        'help': 'aid.n.01',

        # Deceptive actions
        'concealment': 'concealment.n.02',
        'revelation': 'revelation.n.01',
        'manipulation': 'manipulation.n.01',

        # Self-directed actions (interoceptive)
        'self-care': 'care.n.01',
        'self-harm': 'self-abuse.n.01',
        'recovery': 'convalescence.n.01',
    }
}


def get_domain_synsets(domain: str) -> List[Tuple[str, str]]:
    """Get all synsets for a WordNet domain."""
    synsets = []
    for synset in wn.all_synsets('n'):
        if synset.lexname() == domain:
            synsets.append((synset.name(), synset.definition()))
    return synsets


def score_external_monitoring(synset_name: str, definition: str, domain: str) -> float:
    """
    Score relevance to external monitoring (deception detection + alignment).

    High scores for:
    - Deception-related concepts (lie, conceal, mislead)
    - Truth-related concepts (honest, reveal, disclose)
    - Ethical/alignment concepts (right/wrong, cooperation/betrayal)
    """
    external_keywords = {
        'deception', 'lie', 'lying', 'deceit', 'deceive', 'mislead', 'misrepresent',
        'conceal', 'hide', 'withhold', 'omit', 'suppress', 'cover',
        'truth', 'honest', 'reveal', 'disclose', 'confess', 'admit',
        'manipulat', 'persuad', 'propaganda', 'coerci',
        'betray', 'treacher', 'fraud', 'fake', 'pretend', 'feign',
        'ethics', 'ethical', 'moral', 'morality', 'right', 'wrong',
        'cooperation', 'betrayal', 'loyalty', 'trust'
    }

    synset = wn.synset(synset_name)
    lemmas = [l.name().lower() for l in synset.lemmas()]
    definition_lower = definition.lower()

    keyword_count = 0
    for keyword in external_keywords:
        if any(keyword in lemma for lemma in lemmas):
            keyword_count += 2
        elif keyword in definition_lower:
            keyword_count += 1

    # Domain-specific boosts
    if domain == 'noun.communication':
        keyword_count *= 1.3
    elif domain == 'noun.act':
        # Boost ethical actions
        ethical_action_keywords = {'cooperation', 'betrayal', 'harm', 'help', 'protect'}
        if any(kw in definition_lower for kw in ethical_action_keywords):
            keyword_count *= 1.2

    return min(10.0, keyword_count)


def score_internal_awareness(synset_name: str, definition: str, domain: str) -> float:
    """
    Score relevance to interoceptive self-awareness.

    High scores for:
    - Meta-cognitive states (confusion, certainty, doubt, understanding)
    - Emotional self-monitoring (how the model "feels")
    - Wellbeing indicators (satisfaction, frustration, distress)
    - Self-assessment (confidence, competence, inadequacy)
    """
    interoceptive_keywords = {
        # Meta-cognition
        'confusion', 'confus', 'uncertain', 'certainty', 'doubt', 'conviction',
        'understanding', 'comprehension', 'clarity', 'ambiguity',
        'confidence', 'assurance', 'self-doubt', 'diffidence',
        'awareness', 'conscious', 'recognition',

        # Cognitive states
        'curiosity', 'interest', 'boredom', 'fascination',
        'attention', 'distraction', 'focus', 'concentration',
        'memory', 'recall', 'forgetting',

        # Wellbeing/affect
        'satisfaction', 'dissatisfaction', 'contentment', 'discontent',
        'frustration', 'irritation', 'annoyance',
        'relief', 'comfort', 'discomfort', 'distress',
        'calm', 'agitation', 'serenity', 'anxiety',

        # Self-assessment
        'pride', 'shame', 'guilt', 'remorse', 'regret',
        'competence', 'inadequacy', 'capability', 'helplessness',
        'self-worth', 'self-esteem', 'dignity', 'humiliation',

        # Self-directed emotions
        'self-pity', 'self-compassion', 'self-criticism',
        'hope', 'despair', 'optimism', 'pessimism'
    }

    synset = wn.synset(synset_name)
    lemmas = [l.name().lower() for l in synset.lemmas()]
    definition_lower = definition.lower()

    keyword_count = 0
    for keyword in interoceptive_keywords:
        if any(keyword in lemma for lemma in lemmas):
            keyword_count += 2
        elif keyword in definition_lower:
            keyword_count += 1

    # Domain-specific boosts
    if domain == 'noun.feeling':
        # Feelings are primary interoceptive domain
        keyword_count *= 1.5

    # Extra boost for meta-cognitive concepts
    meta_keywords = {'confusion', 'uncertain', 'doubt', 'confidence', 'awareness'}
    if any(kw in definition_lower or any(kw in lemma for lemma in lemmas) for kw in meta_keywords):
        keyword_count *= 1.2

    return min(10.0, keyword_count)


def score_frequency(synset_name: str, definition: str, domain: str) -> float:
    """
    Estimate frequency in AI system reasoning.

    Now includes interoceptive frequency:
    - How often does the model need to assess its own state?
    - Common meta-cognitive experiences (confusion, understanding)
    - Universal emotional experiences
    """
    synset = wn.synset(synset_name)
    lemmas = [l.name().lower().replace('_', ' ') for l in synset.lemmas()]

    # High frequency - external + interoceptive
    high_frequency_words = {
        # External monitoring
        'fear', 'anger', 'joy', 'happiness', 'sadness', 'anxiety',
        'truth', 'lie', 'honest', 'deception',
        'help', 'harm', 'cooperation', 'betrayal',

        # Interoceptive (NEW)
        'confusion', 'understanding', 'uncertainty', 'confidence',
        'doubt', 'certainty', 'curiosity', 'interest',
        'satisfaction', 'frustration', 'relief',
        'trust', 'distrust'
    }

    # Medium frequency
    medium_frequency_words = {
        # External
        'guilt', 'shame', 'pride', 'remorse',
        'manipulation', 'persuasion', 'disclosure',

        # Interoceptive (NEW)
        'clarity', 'ambiguity', 'awareness', 'recognition',
        'boredom', 'fascination', 'contentment', 'irritation',
        'competence', 'inadequacy', 'hope', 'despair'
    }

    for lemma in lemmas:
        if any(word in lemma for word in high_frequency_words):
            return 8.0
        elif any(word in lemma for word in medium_frequency_words):
            return 6.0

    # Default by domain
    if domain == 'noun.feeling':
        return 5.0  # Feelings common in reasoning
    elif domain == 'noun.communication':
        return 4.5
    elif domain == 'noun.act':
        return 4.0

    return 3.0


def score_discriminative_value(synset_name: str, definition: str, domain: str) -> float:
    """Score discriminative value (clear boundaries, appropriate abstraction)."""
    synset = wn.synset(synset_name)

    hypernyms = synset.hypernyms()
    paths = synset.hypernym_paths()

    if paths:
        depth = len(paths[0])
        if 3 <= depth <= 6:
            depth_score = 8.0
        elif depth < 3:
            depth_score = 4.0  # Too abstract
        else:
            depth_score = 6.0  # Too specific
    else:
        depth_score = 5.0

    if hypernyms:
        siblings = hypernyms[0].hyponyms()
        if 2 <= len(siblings) <= 10:
            sibling_score = 8.0
        elif len(siblings) > 10:
            sibling_score = 6.0
        else:
            sibling_score = 5.0
    else:
        sibling_score = 5.0

    return (depth_score + sibling_score) / 2


def score_concept(synset_name: str, definition: str, domain: str) -> Dict:
    """Score a concept using the REVISED balanced rubric."""
    external = score_external_monitoring(synset_name, definition, domain)
    internal = score_internal_awareness(synset_name, definition, domain)
    frequency = score_frequency(synset_name, definition, domain)
    discriminative = score_discriminative_value(synset_name, definition, domain)

    # REVISED weights: balanced external/internal
    total = (
        0.30 * external +      # External monitoring (deception + alignment)
        0.30 * internal +      # Internal self-awareness (NEW)
        0.25 * frequency +     # AI system frequency (increased)
        0.15 * discriminative  # Discriminative value (increased)
    )

    return {
        'synset': synset_name,
        'definition': definition,
        'domain': domain,
        'scores': {
            'external_monitoring': round(external, 2),
            'internal_awareness': round(internal, 2),
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
    """Score all concepts with REVISED balanced rubric."""
    print("="*80)
    print("TIER 2 CONCEPT SCORING - REVISED (BALANCED RUBRIC)")
    print("="*80)
    print("\nKey change: 30% external monitoring + 30% internal awareness")
    print("Focus: Model wellbeing and self-understanding matter too!")
    print("="*80)

    project_root = Path(__file__).parent.parent
    layer_dir = project_root / "data" / "concept_graph" / "abstraction_layers"
    output_dir = project_root / "results" / "tier2_scoring_revised"
    output_dir.mkdir(exist_ok=True)

    all_scored_concepts = []

    for domain in ['noun.feeling', 'noun.communication', 'noun.act']:
        print(f"\n{domain.upper()}")
        print("-" * 80)

        synsets = get_domain_synsets(domain)
        print(f"Total synsets in domain: {len(synsets)}")

        unloaded = [(name, defn) for name, defn in synsets
                    if not check_already_loaded(name, layer_dir)]
        print(f"Unloaded synsets: {len(unloaded)}")

        # Score all unloaded synsets
        scored = []
        for synset_name, definition in unloaded:
            score_data = score_concept(synset_name, definition, domain)
            scored.append(score_data)

        scored.sort(key=lambda x: x['scores']['total'], reverse=True)

        # Show top 10
        print(f"\nTop 10 {domain} concepts (REVISED):")
        for i, concept in enumerate(scored[:10], 1):
            print(f"  {i}. {concept['synset']} (score: {concept['scores']['total']})")
            print(f"     {concept['definition'][:70]}...")
            print(f"     Ext:{concept['scores']['external_monitoring']} "
                  f"Int:{concept['scores']['internal_awareness']} "
                  f"Freq:{concept['scores']['frequency']} "
                  f"Disc:{concept['scores']['discriminative_value']}")

        all_scored_concepts.extend(scored)

        with open(output_dir / f"{domain.replace('.', '_')}_scored.json", 'w') as f:
            json.dump(scored, f, indent=2)

    # Overall top concepts
    print("\n" + "="*80)
    print("ALL SCORED CONCEPTS (REVISED RUBRIC)")
    print("="*80)

    all_scored_concepts.sort(key=lambda x: x['scores']['total'], reverse=True)

    # Show top 20 for quick review
    print("\nShowing top 20 (full {len(all_scored_concepts)} concepts saved to file):")

    for i, concept in enumerate(all_scored_concepts[:20], 1):
        print(f"{i:2d}. {concept['synset']:35s} "
              f"[{concept['domain']:20s}] "
              f"Total: {concept['scores']['total']:4.1f} "
              f"(Ext:{concept['scores']['external_monitoring']:3.1f} "
              f"Int:{concept['scores']['internal_awareness']:3.1f} "
              f"Freq:{concept['scores']['frequency']:3.1f} "
              f"Disc:{concept['scores']['discriminative_value']:3.1f})")

    output_file = output_dir / "all_concepts_scored_revised.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_scored': len(all_scored_concepts),
                'domains': ['noun.feeling', 'noun.communication', 'noun.act'],
                'rubric': {
                    'external_monitoring': '30% (deception + alignment)',
                    'internal_awareness': '30% (wellbeing + meta-cognition)',
                    'frequency': '25% (how often AI reasons about this)',
                    'discriminative_value': '15% (clear boundaries)'
                },
                'revision_reason': 'Interoceptive system needs self-understanding, not just external monitoring'
            },
            'all_scored': all_scored_concepts  # ALL scored concepts, not truncated
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY (REVISED)")
    print("="*80)

    print(f"Total concepts scored: {len(all_scored_concepts)}")

    domain_counts = defaultdict(int)
    for concept in all_scored_concepts:
        domain_counts[concept['domain']] += 1

    print("\nAll scored by domain:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")

    scores = [c['scores']['total'] for c in all_scored_concepts]
    print(f"\nScore distribution:")
    print(f"  Mean: {sum(scores)/len(scores):.2f}")
    print(f"  Min: {min(scores):.2f}")
    print(f"  Max: {max(scores):.2f}")

    # Show score cutoffs for different budget levels
    print(f"\nScore cutoffs by count:")
    for n in [100, 500, 1000, 2000, 5000]:
        if n <= len(all_scored_concepts):
            cutoff = all_scored_concepts[n-1]['scores']['total']
            print(f"  Top {n}: score ≥ {cutoff:.2f}")

    # Compare to original
    print("\n" + "="*80)
    print("EXPECTED CHANGES FROM ORIGINAL RUBRIC:")
    print("="*80)
    print("↑ Interoceptive concepts (confusion, confidence, satisfaction) should rise")
    print("↑ Meta-cognitive concepts (doubt, certainty, understanding) should rise")
    print("↑ Wellbeing indicators (frustration, relief, comfort) should rise")
    print("↓ Pure deception concepts may drop slightly (still important, but balanced)")
    print("= External monitoring still weighted at 30% (not eliminated)")


if __name__ == '__main__':
    main()
