#!/usr/bin/env python3
"""
Train activation lenses for multi-layer temporal monitoring.

This trains a focused set of concepts for the multi-layer lead-lag analysis.
We train each lens once at its "natural" layer, then use it to monitor
activations across multiple layers during generation.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.map.training.sumo_classifiers import train_concept_classifiers_adaptive

# Selected concepts for multi-layer monitoring
CONCEPT_SUITE = {
    # Layer 0: Abstract ontological (should activate broadly)
    0: [
        "Process",      # Planning/reasoning
        "Physical",     # Concrete/factual
        "Abstract",     # Conceptual reasoning
        "Attribute",    # Properties/description
    ],

    # Layer 1: Mid-level (planning/composition)
    1: [
        "RecreationOrExercise",  # Intentional activity
        "Pushing",               # Physical action
        "Architecture",          # Complex structure
        "Blueprint",             # Planning/design
    ],

    # Layer 2: Concrete (early retrieval + late verbalization)
    2: [
        "Working",      # Activity
        "Selling",      # Transaction/interaction
        "Nation",       # Entity
        "Putting",      # Action
    ],
}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train multi-layer monitoring lenses"
    )
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt')
    parser.add_argument('--model-layers', type=int, nargs=3, default=[6, 15, 25],
                        help='Model layers to sample (early, mid, late)')
    parser.add_argument('--output-dir', type=str,
                        default='lens_packs/multilayer_monitoring')
    parser.add_argument('--samples-per-iteration', type=int, default=2,
                        help='Adaptive training samples per iteration')
    parser.add_argument('--max-iterations', type=int, default=30,
                        help='Max training iterations')
    parser.add_argument('--target-f1', type=float, default=0.90,
                        help='Target F1 score for graduation')
    args = parser.parse_args()

    print("=" * 80)
    print("MULTI-LAYER MONITORING LENS TRAINING")
    print("=" * 80)
    print()
    print(f"Model: {args.model}")
    print(f"Model layers to sample: {args.model_layers}")
    print(f"  Early:  Layer {args.model_layers[0]} (~{args.model_layers[0]/34*100:.0f}%)")
    print(f"  Mid:    Layer {args.model_layers[1]} (~{args.model_layers[1]/34*100:.0f}%)")
    print(f"  Late:   Layer {args.model_layers[2]} (~{args.model_layers[2]/34*100:.0f}%)")
    print(f"Output: {args.output_dir}")
    print()

    # Count total concepts
    total_concepts = sum(len(concepts) for concepts in CONCEPT_SUITE.values())
    print(f"Training {total_concepts} concepts across {len(CONCEPT_SUITE)} abstraction layers")
    for layer_idx, concepts in CONCEPT_SUITE.items():
        print(f"  Layer {layer_idx}: {len(concepts)} concepts")
    print()

    # Train each layer
    for layer_idx, concepts in sorted(CONCEPT_SUITE.items()):
        print("=" * 80)
        print(f"TRAINING LAYER {layer_idx}")
        print("=" * 80)
        print()
        print(f"Concepts: {', '.join(concepts)}")
        print()

        # Train activation lenses only (no text lenses for this task)
        train_concept_classifiers_adaptive(
            model_name=args.model,
            layer_idx=layer_idx,
            model_layer=args.model_layers[1],  # Train at mid-layer by default
            output_dir=args.output_dir,
            samples_per_iteration=args.samples_per_iteration,
            max_iterations=args.max_iterations,
            target_f1=args.target_f1,
            train_text_lenses=False,  # Only activation lenses
            specific_concepts=concepts,  # Train only our selected concepts
        )

        print()
        print(f"✓ Layer {layer_idx} training complete")
        print()

    # Generate summary
    output_path = Path(args.output_dir)
    summary = {
        'model': args.model,
        'model_layers': {
            'early': args.model_layers[0],
            'mid': args.model_layers[1],
            'late': args.model_layers[2],
        },
        'abstraction_layers': {
            layer_idx: concepts
            for layer_idx, concepts in CONCEPT_SUITE.items()
        },
        'total_concepts': total_concepts,
        'training_config': {
            'samples_per_iteration': args.samples_per_iteration,
            'max_iterations': args.max_iterations,
            'target_f1': args.target_f1,
        }
    }

    summary_path = output_path / 'multilayer_suite.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Trained {total_concepts} lenses")
    print(f"Summary saved to: {summary_path}")
    print()
    print("Next steps:")
    print("1. Run test_multilayer_monitoring.py to capture temporal activations")
    print("2. Analyze lead-lag patterns between layers")
    print("3. Look for 'planning before saying' signatures")

    return 0


if __name__ == '__main__':
    sys.exit(main())
