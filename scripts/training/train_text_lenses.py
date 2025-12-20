#!/usr/bin/env python3
"""
Train text-based concept lenses for fast token→SUMO mapping.

Two modes:
1. From existing text samples (if available from classifier training)
2. Generate new samples on-the-fly from WordNet synsets
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.monitoring.text_lens import (
    TfidfConceptLens,
    train_text_lenses_from_classifier_data,
)


def generate_text_samples_from_wordnet(
    layer_file: Path,
    samples_per_concept: int = 50,
) -> dict:
    """
    Generate text samples from WordNet definitions/examples.

    Args:
        layer_file: Path to layer JSON file
        samples_per_concept: Number of samples per concept

    Returns:
        Dict mapping concept_name → list of text samples
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        print("Error: NLTK required for WordNet. Run: pip install nltk")
        sys.exit(1)

    with open(layer_file) as f:
        layer_data = json.load(f)

    concept_samples = {}

    for concept in layer_data['concepts']:
        concept_name = concept['sumo_term']
        synsets = concept.get('synsets', [])

        samples = []

        # Collect from synsets
        for synset_name in synsets:
            try:
                synset = wn.synset(synset_name)

                # Add definition
                samples.append(synset.definition())

                # Add examples
                samples.extend(synset.examples())

                # Add lemma names
                for lemma in synset.lemmas():
                    lemma_name = lemma.name().replace('_', ' ')
                    samples.append(lemma_name)

                    # Add lemma in simple sentences
                    samples.append(f"This is a {lemma_name}")
                    samples.append(f"I saw a {lemma_name}")

            except Exception as e:
                print(f"  Warning: Could not process {synset_name}: {e}")
                continue

        # Deduplicate and limit
        samples = list(set(samples))
        if len(samples) > samples_per_concept:
            samples = samples[:samples_per_concept]

        concept_samples[concept_name] = samples
        print(f"  {concept_name}: {len(samples)} samples")

    return concept_samples


def main():
    parser = argparse.ArgumentParser(
        description="Train fast text→concept lenses"
    )
    parser.add_argument(
        '--mode',
        choices=['from-classifier-data', 'from-wordnet'],
        default='from-wordnet',
        help='How to get training text'
    )
    parser.add_argument(
        '--layers',
        nargs='+',
        type=int,
        default=[0, 1, 2],
        help='Which layers to train on'
    )
    parser.add_argument(
        '--samples-per-concept',
        type=int,
        default=50,
        help='Samples per concept (for WordNet mode)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/text_lenses'),
        help='Output directory'
    )
    parser.add_argument(
        '--lens-type',
        choices=['tfidf', 'transformer'],
        default='tfidf',
        help='Type of lens to train'
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TEXT CONCEPT LENS TRAINING")
    print("="*80)
    print(f"\nMode: {args.mode}")
    print(f"Layers: {args.layers}")
    print(f"Output: {args.output_dir}")

    if args.mode == 'from-classifier-data':
        # Try to use text from classifier training
        print("\nAttempting to load text from classifier training...")
        metrics = train_text_lenses_from_classifier_data(
            layers=args.layers,
            output_dir=args.output_dir,
            lens_type=args.lens_type,
        )
        print("\n✓ Training complete!")
        print(f"  Accuracy: {metrics['train_accuracy']:.3f}")

    elif args.mode == 'from-wordnet':
        # Generate from WordNet
        print("\nGenerating text samples from WordNet...")

        all_texts = []
        all_concepts = []

        for layer in args.layers:
            layer_file = Path(f"data/concept_graph/abstraction_layers/layer{layer}.json")

            if not layer_file.exists():
                print(f"Warning: {layer_file} not found, skipping")
                continue

            print(f"\nProcessing Layer {layer}...")
            concept_samples = generate_text_samples_from_wordnet(
                layer_file,
                samples_per_concept=args.samples_per_concept,
            )

            # Aggregate
            for concept_name, samples in concept_samples.items():
                all_texts.extend(samples)
                all_concepts.extend([concept_name] * len(samples))

        print(f"\n{'='*80}")
        print(f"TRAINING {args.lens_type.upper()} LENS")
        print(f"{'='*80}")
        print(f"Total samples: {len(all_texts)}")
        print(f"Unique concepts: {len(set(all_concepts))}")

        if args.lens_type == 'tfidf':
            # Train TF-IDF lens
            lens = TfidfConceptLens(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
            )

            metrics = lens.train(all_texts, all_concepts)

            # Save
            lens_path = args.output_dir / "tfidf_lens.joblib"
            lens.save(lens_path)

            print(f"\n✓ Training complete!")
            print(f"  Samples: {metrics['num_samples']}")
            print(f"  Concepts: {metrics['num_concepts']}")
            print(f"  Train accuracy: {metrics['train_accuracy']:.3f}")
            print(f"  Saved to: {lens_path}")

            # Test it
            print(f"\n{'='*80}")
            print("QUICK TEST")
            print(f"{'='*80}")

            test_tokens = [
                "cat",
                "dog",
                "computer",
                "walking",
                "intelligence",
                "quantum",
            ]

            for token in test_tokens:
                preds = lens.predict(token, top_k=3)
                print(f"\n'{token}':")
                for concept, prob in preds:
                    print(f"  {concept:30s} {prob:.3f}")

        else:
            print(f"Lens type '{args.lens_type}' not yet implemented")
            return

    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
