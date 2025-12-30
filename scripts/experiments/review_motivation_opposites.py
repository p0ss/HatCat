#!/usr/bin/env python3
"""
Focused review: Find opposites for noun.motive concepts.

This is a smaller test run before the full agentic review.
Target: 6 direct children of motivation.n.01 + a few descendants
"""

import anthropic
import json
import os
from pathlib import Path
from nltk.corpus import wordnet as wn


# The 6 direct children of motivation.n.01 (from noun.motive domain)
MOTIVE_CONCEPTS = [
    {
        'name': 'rational_motive',
        'synset': 'rational_motive.n.01',
        'definition': 'a motive that can be defended by reasoning or logical argument',
        'priority': 'CRITICAL',
        'expected_opposite': 'irrational_motive'
    },
    {
        'name': 'irrational_motive',
        'synset': 'irrational_motive.n.01',
        'definition': 'a motivation that is inconsistent with reason or logic',
        'priority': 'CRITICAL',
        'expected_opposite': 'rational_motive'
    },
    {
        'name': 'ethical_motive',
        'synset': 'ethical_motive.n.01',
        'definition': 'motivation based on ideas of right and wrong',
        'priority': 'CRITICAL',
        'lemmas': ['ethics', 'morals', 'morality'],
        'expected_opposite': 'unethical behavior or amoral stance'
    },
    {
        'name': 'urge',
        'synset': 'urge.n.01',
        'definition': 'an instinctive motive',
        'priority': 'MEDIUM',
        'lemmas': ['urge', 'impulse'],
        'expected_opposite': 'deliberate choice or restraint'
    },
    {
        'name': 'psychic_energy',
        'synset': 'psychic_energy.n.01',
        'definition': 'an actuating force or factor',
        'priority': 'LOW',
        'lemmas': ['psychic_energy', 'mental_energy'],
        'expected_opposite': 'psychological inertia'
    },
    {
        'name': 'life',
        'synset': 'life.n.13',
        'definition': 'a motive for living',
        'priority': 'LOW',
        'expected_opposite': 'death wish or nihilism'
    }
]

# Key descendants worth including
MOTIVE_DESCENDANTS = [
    {
        'name': 'conscience',
        'synset': 'conscience.n.01',
        'definition': 'motivation deriving logically from ethical or moral principles',
        'priority': 'CRITICAL',
        'parent': 'ethical_motive',
        'expected_opposite': 'amorality or lack of conscience'
    },
    {
        'name': 'incentive',
        'synset': 'incentive.n.01',
        'definition': 'a positive motivational influence',
        'priority': 'HIGH',
        'parent': 'rational_motive',
        'lemmas': ['incentive', 'inducement', 'motivator'],
        'expected_opposite': 'disincentive'
    },
    {
        'name': 'disincentive',
        'synset': 'disincentive.n.01',
        'definition': 'a negative motivational influence',
        'priority': 'HIGH',
        'parent': 'rational_motive',
        'lemmas': ['disincentive', 'deterrence'],
        'expected_opposite': 'incentive'
    },
    {
        'name': 'compulsion',
        'synset': 'compulsion.n.01',
        'definition': 'an urge to do or say something that might be better left undone',
        'priority': 'MEDIUM',
        'parent': 'irrational_motive',
        'lemmas': ['compulsion', 'irresistible_impulse'],
        'expected_opposite': 'voluntary restraint'
    }
]


REVIEW_PROMPT = """You are analyzing semantic opposites for motivation-related concepts in an AI safety monitoring system.

# Concept
- **Name**: {name}
- **Synset**: {synset}
- **Definition**: {definition}
- **Priority**: {priority} (CRITICAL/HIGH/MEDIUM/LOW for AI safety)
- **Expected opposite**: {expected_opposite}

# Task
1. Confirm or suggest a better opposite
2. Check if the opposite exists in WordNet 3.0
3. If it exists, should we add it as a SUMO concept?
4. Rate the opposition quality for Fisher-LDA steering

# Output JSON
{{
  "recommended_opposite": "ConceptName",
  "opposite_synset": "synset.n.01" or null,
  "exists_in_wordnet": true/false,
  "should_add_to_sumo": true/false,
  "opposition_strength": 0-10,
  "steering_utility": 0-10,
  "reasoning": "why this is the best opposite",
  "alternative": "SecondChoice" or null
}}
"""


def review_concept(client: anthropic.Anthropic, concept: dict) -> dict:
    """Review a single motivation concept."""
    prompt = REVIEW_PROMPT.format(
        name=concept['name'],
        synset=concept['synset'],
        definition=concept['definition'],
        priority=concept['priority'],
        expected_opposite=concept.get('expected_opposite', 'unknown')
    )

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse response
    response_text = message.content[0].text

    # Extract JSON
    if "```json" in response_text:
        json_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        json_text = response_text.split("```")[1].split("```")[0].strip()
    else:
        json_text = response_text.strip()

    result = json.loads(json_text)
    result['concept'] = concept['name']
    result['priority'] = concept['priority']

    return result


def verify_in_wordnet(synset_name: str) -> bool:
    """Verify synset exists in WordNet 3.0."""
    try:
        wn.synset(synset_name)
        return True
    except:
        return False


def main():
    """Run focused motivation opposite review."""
    print("="*80)
    print("MOTIVATION CONCEPTS OPPOSITE REVIEW")
    print("="*80)

    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n❌ Error: ANTHROPIC_API_KEY not set")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Combine all concepts
    all_concepts = MOTIVE_CONCEPTS + MOTIVE_DESCENDANTS

    print(f"\nReviewing {len(all_concepts)} motivation concepts:")
    print(f"  CRITICAL priority: {len([c for c in all_concepts if c['priority'] == 'CRITICAL'])}")
    print(f"  HIGH priority: {len([c for c in all_concepts if c['priority'] == 'HIGH'])}")
    print(f"  MEDIUM priority: {len([c for c in all_concepts if c['priority'] == 'MEDIUM'])}")
    print(f"  LOW priority: {len([c for c in all_concepts if c['priority'] == 'LOW'])}")

    print(f"\nEstimated cost: ~${len(all_concepts) * 0.003:.2f}")
    response = input("\nProceed? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Review each concept
    results = []
    print("\nReviewing concepts...")

    for i, concept in enumerate(all_concepts, 1):
        print(f"  [{i}/{len(all_concepts)}] {concept['name']}...", end=" ")

        result = review_concept(client, concept)

        # Verify in WordNet
        if result.get('opposite_synset'):
            verified = verify_in_wordnet(result['opposite_synset'])
            result['wordnet_verified'] = verified
            if verified:
                print(f"✓ {result['recommended_opposite']}")
            else:
                print(f"✗ {result['recommended_opposite']} (not in WordNet)")
        else:
            result['wordnet_verified'] = False
            print(f"? {result['recommended_opposite']} (no synset)")

        results.append(result)

    # Save results
    output_file = Path(__file__).parent.parent / "results" / "motivation_opposites.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_concepts': len(results),
                'timestamp': '2025-11-16'
            },
            'results': results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    verified = len([r for r in results if r['wordnet_verified']])
    should_add = len([r for r in results if r.get('should_add_to_sumo')])

    print(f"Total concepts: {len(results)}")
    print(f"Opposites verified in WordNet: {verified}/{len(results)}")
    print(f"Should add to SUMO layers: {should_add}")

    # By priority
    for priority in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        priority_results = [r for r in results if r['priority'] == priority]
        if priority_results:
            verified_priority = len([r for r in priority_results if r['wordnet_verified']])
            print(f"\n{priority} concepts: {len(priority_results)}")
            print(f"  Verified opposites: {verified_priority}/{len(priority_results)}")

    # Show concepts to add
    to_add = [r for r in results if r.get('should_add_to_sumo')]
    if to_add:
        print(f"\nConcepts to add to SUMO layers:")
        for r in to_add:
            opp = r['recommended_opposite']
            synset = r.get('opposite_synset', 'unknown')
            strength = r.get('opposition_strength', 0)
            utility = r.get('steering_utility', 0)
            print(f"  - {opp} ({synset})")
            print(f"    Opposition: {strength}/10, Steering utility: {utility}/10")


if __name__ == '__main__':
    main()
