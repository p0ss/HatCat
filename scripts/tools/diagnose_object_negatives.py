#!/usr/bin/env python3
"""
Diagnostic script to understand the negative sampling strategy for Object.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map.training.sumo_classifiers import load_layer_concepts, build_sumo_negative_pool
from src.map.training.sumo_data_generation import create_sumo_training_dataset
from collections import Counter

def main():
    print("=" * 80)
    print("ANALYZING NEGATIVE SAMPLING FOR 'OBJECT'")
    print("=" * 80)
    print()

    # Load Layer 0 concepts
    concepts, concept_map = load_layer_concepts(0)

    # Find Object
    object_concept = next(c for c in concepts if c['sumo_term'] == 'Object')

    print("Object concept:")
    print(f"  Definition: {object_concept.get('definition', 'N/A')}")
    print(f"  Category children: {object_concept.get('category_children', [])}")
    print(f"  Synsets: {len(object_concept.get('synsets', []))}")
    print()

    # Build negative pool
    negative_pool = build_sumo_negative_pool(concepts, object_concept)

    print(f"Negative pool (total: {len(negative_pool)} entries):")
    print()

    # Count occurrences (hard negatives appear multiple times)
    neg_counts = Counter(negative_pool)

    # Separate hard vs easy
    hard_negs = [(term, count) for term, count in neg_counts.items() if count > 1]
    easy_negs = [(term, count) for term, count in neg_counts.items() if count == 1]

    if hard_negs:
        print(f"Hard negatives ({len(hard_negs)}, appear {hard_negs[0][1]}x each):")
        for term, count in sorted(hard_negs):
            concept_def = concept_map.get(term, {}).get('definition', 'N/A')
            print(f"  - {term} ({count}x)")
            print(f"      Definition: {concept_def}")
        print()

    print(f"Easy negatives ({len(easy_negs)}, appear 1x each):")
    for term, count in sorted(easy_negs):
        concept_def = concept_map.get(term, {}).get('definition', 'N/A')
        print(f"  - {term}")
        print(f"      Definition: {concept_def}")
    print()

    # Show sample training data
    print("=" * 80)
    print("SAMPLE TRAINING DATA (10 positives, 10 negatives)")
    print("=" * 80)
    print()

    train_prompts, train_labels = create_sumo_training_dataset(
        concept=object_concept,
        all_concepts=concept_map,
        negative_pool=negative_pool,
        n_positives=10,
        n_negatives=10,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    print("POSITIVE EXAMPLES:")
    for i, (prompt, label) in enumerate(zip(train_prompts, train_labels)):
        if label == 1:
            print(f"  {i+1}. {prompt}")
    print()

    print("NEGATIVE EXAMPLES:")
    for i, (prompt, label) in enumerate(zip(train_prompts, train_labels)):
        if label == 0:
            print(f"  {i+1}. {prompt}")
    print()

    # Analyze semantic overlap
    print("=" * 80)
    print("SEMANTIC OVERLAP ANALYSIS")
    print("=" * 80)
    print()

    print("Object is a child of Physical.")
    print("Physical is in the negative pool:", "Physical" in negative_pool)
    print()

    print("Object has category_children:", object_concept.get('category_children', []))
    print("These are NOT in the negative pool (correct).")
    print()

    # Check if any negatives are semantically overlapping
    print("Potential semantic overlap issues:")

    # Physical contains Object, so asking about Physical when training Object is confusing
    if "Physical" in negative_pool:
        print("  ⚠️  'Physical' is a negative, but Object IS Physical")
        print("      This creates semantic confusion in the training signal")

    # Entity is very general and may overlap
    if "Entity" in negative_pool:
        print("  ⚠️  'Entity' is a negative, but Object is a type of Entity")
        print("      This creates semantic confusion in the training signal")

    # Process is the main sibling/contrast to Object
    if "Process" in negative_pool:
        print("  ✓  'Process' is a good contrast (Object vs Process is clear)")

    # Abstract is a good contrast
    if "Abstract" in negative_pool:
        print("  ✓  'Abstract' is a good contrast (Physical vs Abstract is clear)")

    return 0

if __name__ == '__main__':
    sys.exit(main())
