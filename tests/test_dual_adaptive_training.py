#!/usr/bin/env python3
"""Test dual adaptive training on a small concept set."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.sumo_classifiers import train_sumo_classifiers


def main():
    """Test adaptive training on layer 0 with a small sample."""

    print("\n" + "=" * 80)
    print("TESTING DUAL ADAPTIVE TRAINING")
    print("=" * 80)
    print("\nThis will train a small subset of Layer 0 concepts with:")
    print("  - Adaptive training (independent graduation)")
    print("  - Both activation and text probes")
    print("  - Baseline: 10 samples")
    print("  - Activation increment: +1 per iteration")
    print("  - Text increment: +5 per iteration")
    print("  - Target accuracy: 95% F1")
    print()

    # Train just layer 0 with adaptive training
    results = train_sumo_classifiers(
        layers=[0],
        model_name="google/gemma-3-4b-pt",
        device="cuda",
        n_train_pos=10,  # Baseline for adaptive training
        n_train_neg=10,
        n_test_pos=20,
        n_test_neg=20,
        output_dir=Path("results/sumo_classifiers_adaptive_test"),
        train_text_probes=True,
        use_adaptive_training=True,  # ENABLE ADAPTIVE TRAINING
    )

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nResults:")
    for summary in results:
        print(f"  Layer {summary['layer']}: {summary['n_successful']}/{summary['n_concepts']} "
              f"(Avg Test F1: {summary['avg_test_f1']:.3f})")

    print("\n✓ Check results/sumo_classifiers_adaptive_test/layer0/results.json for details")
    print("  - Look for 'activation_samples' and 'text_samples' to see graduation points")
    print("  - Look for 'total_iterations' to see how many cycles were needed")


if __name__ == '__main__':
    main()
