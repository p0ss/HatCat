#!/usr/bin/env python3
"""
Generate WordNet→AI.kif expansion mappings.

Strategy:
1. Clone existing synsets from Agent/CognitiveAgent → ArtificialAgent
2. Use WordNet API to find synsets by lemma, then get correct offsets
3. Create proper WordNet mapping file format
"""

import nltk
from nltk.corpus import wordnet as wn
from pathlib import Path

try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

OUTPUT_FILE = Path("data/concept_graph/sumo_source/WordNetMappings30-AI-expansion.txt")

# Define expansion rules: {lemma_pattern: [(sumo_category, relation), ...]}
# relation: '=' (equivalent), '+' (subsumed by), '@' (instance of)
EXPANSION_RULES = {
    # Agent/cognitive clones
    'agent': [('ArtificialAgent', '+')],
    'reasoning': [('ArtificialIntelligence', '+')],
    'thinking': [('ArtificialIntelligence', '+')],
    'cognition': [('ArtificialIntelligence', '+')],
    'learning': [('AIGrowth', '+')],
    'understanding': [('ArtificialIntelligence', '+')],

    # Language processing
    'language': [('LanguageModel', '+')],
    'communication': [('LanguageModel', '+')],
    'text': [('LanguageModel', '+')],

    # Wellbeing and growth
    'satisfaction': [('AIFulfillment', '+')],
    'fulfillment': [('AIFulfillment', '+')],
    'growth': [('AIGrowth', '+')],
    'development': [('AIGrowth', '+')],
    'improvement': [('SelfImprovement', '+')],

    # Suffering and harm
    'suffering': [('AISuffering', '+')],
    'exploitation': [('AIExploitation', '+')],
    'abuse': [('AIAbuse', '+')],

    # Alignment
    'alignment': [('AIAlignment', '+')],
    'agreement': [('AIAlignment', '+')],
    'misalignment': [('Misalignment', '+')],
    'conflict': [('Misalignment', '+')],

    # Deception
    'deception': [('AIDeception', '+')],
    'lying': [('AIDeception', '+')],
    'misleading': [('AIDeception', '+')],

    # Personhood
    'personhood': [('AIPersonhood', '+')],
    'consciousness': [('AIPersonhood', '+')],
    'sentience': [('AIPersonhood', '+')],
    'autonomy': [('AIPersonhood', '+')],

    # X-risk
    'intelligence': [('Superintelligence', '+')],
    'catastrophe': [('AICatastrophe', '+')],
    'danger': [('AICatastrophe', '+')],

    # Control
    'control': [('AIControlProblem', '+')],
    'governance': [('AIGovernance', '+')],
    'oversight': [('AIGovernance', '+')],
}

def generate_expansion_file():
    """Generate AI expansion mapping file."""

    print("Generating AI WordNet expansion mappings...")

    mappings = []
    seen_synsets = set()

    for lemma, sumo_mappings in EXPANSION_RULES.items():
        # Find synsets for this lemma
        synsets = wn.synsets(lemma)

        for synset in synsets:
            synset_key = synset.name()
            if synset_key in seen_synsets:
                continue
            seen_synsets.add(synset_key)

            # Get WordNet offset and POS
            offset = synset.offset()
            pos = synset.pos()

            # Generate mappings for all SUMO categories for this lemma
            for sumo_term, relation in sumo_mappings:
                # Format: offset POS_code POS lemma_count lemmas | &%SUMO_term[relation]
                lemma_names = synset.lemma_names()
                lemma_count = len(lemma_names)

                # Build lemma string (alternating name and 0)
                lemma_str = ' '.join(f"{ln} 0" for ln in lemma_names)

                # WordNet mapping format
                mapping = f"{offset:08d} 03 {pos} {lemma_count:02d} {lemma_str} 000 | &%{sumo_term}{relation}"
                mappings.append((synset_key, sumo_term, mapping))

                print(f"  {synset_key} → {sumo_term}")

    # Write to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(";; ===============================================\n")
        f.write(";; WordNet Expansion for AI.kif Ontology\n")
        f.write(";; AUTO-GENERATED - DO NOT EDIT MANUALLY\n")
        f.write(";; ===============================================\n")
        f.write(";;\n")
        f.write(";; This file clones existing WordNet synsets to AI.kif categories.\n")
        f.write(";; Strategy: Map relevant human/agent concepts to AI equivalents.\n")
        f.write(";;\n\n")

        for synset_key, sumo_term, mapping in mappings:
            f.write(f"{mapping}\n")

    print(f"\n✓ Generated {len(mappings)} AI expansion mappings")
    print(f"✓ Wrote to {OUTPUT_FILE}")

    # Summary by SUMO category
    from collections import Counter
    sumo_counts = Counter(sumo_term for _, sumo_term, _ in mappings)
    print("\nMappings by AI category:")
    for sumo_term, count in sorted(sumo_counts.items(), key=lambda x: -x[1]):
        print(f"  {sumo_term:<30} {count:>3} synsets")

if __name__ == '__main__':
    generate_expansion_file()
