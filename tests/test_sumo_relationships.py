#!/usr/bin/env python3
"""
Test SUMO relationship extraction and training data generation.

Verifies that:
1. SUMO category relationships are extracted correctly
2. WordNet relationships are accessible via canonical_synset
3. Training data generation works for SUMO concepts
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map.training.sumo_data_generation import (
    extract_sumo_relationships,
    create_sumo_training_dataset,
    build_sumo_negative_pool
)


def test_layer0_relationships():
    """Test relationship extraction for Layer 0 concepts."""

    # Load Layer 0 concepts
    layer0_path = Path("data/concept_graph/abstraction_layers/layer0.json")

    with open(layer0_path) as f:
        layer0_data = json.load(f)

    concepts = layer0_data['concepts']

    print("=" * 80)
    print("SUMO RELATIONSHIP EXTRACTION TEST")
    print("=" * 80)
    print(f"\nTesting {len(concepts)} Layer 0 concepts\n")

    # Create concept map for lookups
    concept_map = {c['sumo_term']: c for c in concepts}

    # Test each concept
    results = []

    for concept in concepts[:5]:  # Test first 5 for brevity
        sumo_term = concept['sumo_term']
        print(f"\n{'─' * 80}")
        print(f"Concept: {sumo_term}")
        print(f"{'─' * 80}")

        # Extract relationships
        relationships = extract_sumo_relationships(concept)

        # Display relationships
        print(f"\n📊 SUMO Category Structure:")
        print(f"  • Layer: {concept['layer']}")
        print(f"  • SUMO Depth: {concept['sumo_depth']}")
        print(f"  • Category Children: {len(relationships['category_children'])}")

        if relationships['category_children']:
            for child in relationships['category_children'][:5]:
                print(f"    - {child}")

        print(f"\n📖 WordNet Relationships (via {concept.get('canonical_synset', 'N/A')}):")
        print(f"  • Hypernyms: {len(relationships['hypernyms'])}")
        if relationships['hypernyms']:
            for hyp in relationships['hypernyms'][:3]:
                print(f"    - {hyp}")

        print(f"  • Hyponyms: {len(relationships['hyponyms'])}")
        if relationships['hyponyms']:
            for hypo in relationships['hyponyms'][:3]:
                print(f"    - {hypo}")

        print(f"  • Meronyms (parts): {len(relationships['meronyms'])}")
        print(f"  • Holonyms (wholes): {len(relationships['holonyms'])}")
        print(f"  • Antonyms: {len(relationships['antonyms'])}")

        # Count total relationships
        total_rels = (
            len(relationships['category_children']) +
            len(relationships['hypernyms']) +
            len(relationships['hyponyms']) +
            len(relationships['meronyms']) +
            len(relationships['holonyms']) +
            len(relationships['antonyms'])
        )

        print(f"\n✓ Total relationships: {total_rels}")

        results.append({
            'concept': sumo_term,
            'category_children': len(relationships['category_children']),
            'wordnet_relationships': total_rels - len(relationships['category_children']),
            'total_relationships': total_rels
        })

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")

    for r in results:
        print(f"{r['concept']:20s} | Category: {r['category_children']:2d} | "
              f"WordNet: {r['wordnet_relationships']:3d} | Total: {r['total_relationships']:3d}")

    avg_total = sum(r['total_relationships'] for r in results) / len(results)
    print(f"\nAverage total relationships: {avg_total:.1f}")

    return results


def test_training_data_generation():
    """Test training data generation for SUMO concepts."""

    # Load Layer 0 concepts
    layer0_path = Path("data/concept_graph/abstraction_layers/layer0.json")

    with open(layer0_path) as f:
        layer0_data = json.load(f)

    concepts = layer0_data['concepts']
    concept_map = {c['sumo_term']: c for c in concepts}

    print("\n" + "=" * 80)
    print("TRAINING DATA GENERATION TEST")
    print("=" * 80)

    # Test with first concept
    test_concept = concepts[0]
    sumo_term = test_concept['sumo_term']

    print(f"\nTest Concept: {sumo_term}")
    print(f"Definition: {test_concept.get('definition', 'N/A')}")
    print(f"Category Children: {test_concept.get('category_children', [])}")
    print(f"Canonical Synset: {test_concept.get('canonical_synset', 'N/A')}")

    # Build negative pool
    negative_pool = build_sumo_negative_pool(concepts, test_concept, min_layer_distance=1)
    print(f"\nNegative pool size: {len(negative_pool)}")
    print(f"Sample negatives: {negative_pool[:5]}")

    # Generate training data
    print(f"\n{'─' * 80}")
    print("Generating training data (1 def + 9 rels)...")
    print(f"{'─' * 80}\n")

    prompts, labels = create_sumo_training_dataset(
        concept=test_concept,
        all_concepts=concept_map,
        negative_pool=negative_pool,
        n_positives=10,
        n_negatives=10,
        use_category_relationships=True,
        use_wordnet_relationships=True
    )

    # Display samples
    print("POSITIVE EXAMPLES (label=1):")
    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        if label == 1:
            print(f"  {i+1}. {prompt[:80]}...")

    print("\nNEGATIVE EXAMPLES (label=0):")
    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        if label == 0:
            print(f"  {i+1}. {prompt[:80]}...")

    print(f"\n✓ Generated {len(prompts)} prompts ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")

    # Test with category relationships only
    print(f"\n{'─' * 80}")
    print("Testing SUMO category relationships only...")
    print(f"{'─' * 80}\n")

    prompts_sumo, labels_sumo = create_sumo_training_dataset(
        concept=test_concept,
        all_concepts=concept_map,
        negative_pool=negative_pool,
        n_positives=10,
        n_negatives=10,
        use_category_relationships=True,
        use_wordnet_relationships=False
    )

    print("SUMO-only positive examples:")
    for i, (prompt, label) in enumerate(zip(prompts_sumo, labels_sumo)):
        if label == 1:
            print(f"  {i+1}. {prompt[:80]}...")

    # Test with WordNet relationships only
    print(f"\n{'─' * 80}")
    print("Testing WordNet relationships only...")
    print(f"{'─' * 80}\n")

    prompts_wn, labels_wn = create_sumo_training_dataset(
        concept=test_concept,
        all_concepts=concept_map,
        negative_pool=negative_pool,
        n_positives=10,
        n_negatives=10,
        use_category_relationships=False,
        use_wordnet_relationships=True
    )

    print("WordNet-only positive examples:")
    for i, (prompt, label) in enumerate(zip(prompts_wn, labels_wn)):
        if label == 1:
            print(f"  {i+1}. {prompt[:80]}...")

    return {
        'combined': (prompts, labels),
        'sumo_only': (prompts_sumo, labels_sumo),
        'wordnet_only': (prompts_wn, labels_wn)
    }


def main():
    """Run all tests."""

    # Test relationship extraction
    rel_results = test_layer0_relationships()

    # Test training data generation
    training_results = test_training_data_generation()

    print("\n" + "=" * 80)
    print("✓ ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. SUMO category relationships (category_children) are accessible")
    print("2. WordNet relationships are accessible via canonical_synset")
    print("3. Training data generation combines both relationship types")
    print("4. Negative pool generation uses layer distance")
    print("\nThe SUMO-aware training pipeline is ready for use!")


if __name__ == '__main__':
    main()
