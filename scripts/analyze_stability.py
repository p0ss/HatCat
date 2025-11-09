"""
Analyze activation pattern stability from captured concepts.
This validates that concepts produce consistent patterns across contexts.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import ActivationStorage


def analyze_concept_stability(storage_path: Path):
    """
    Analyze stability of captured concept activations.

    Metrics:
    - Variance across contexts (should be low)
    - Signal-to-noise ratio
    - Sparsity patterns
    """
    print("=" * 80)
    print("ACTIVATION PATTERN STABILITY ANALYSIS")
    print("=" * 80)

    with ActivationStorage(storage_path, mode='r') as storage:
        concepts = storage.list_concepts()
        baseline = storage.load_baseline()

        print(f"\nAnalyzing {len(concepts)} concepts")
        print(f"Baseline has {len(baseline)} layers")

        results = {}

        for concept in concepts:
            activations, metadata = storage.load_concept_activations(concept)

            # Analyze first layer as representative
            layer_name = list(activations.keys())[0]
            act = activations[layer_name]

            # Metrics
            sparsity = (act == 0).sum() / act.size
            mean_activation = np.abs(act).mean()
            max_activation = np.abs(act).max()
            std_activation = np.abs(act).std()

            # Signal to noise (mean / std)
            snr = mean_activation / (std_activation + 1e-10)

            results[concept] = {
                'category': metadata.get('category', 'unknown'),
                'num_prompts': metadata.get('num_prompts', 0),
                'sparsity': sparsity,
                'mean_abs_activation': mean_activation,
                'max_abs_activation': max_activation,
                'std_activation': std_activation,
                'snr': snr
            }

        # Print results
        print("\n" + "-" * 80)
        print(f"{'Concept':<15} {'Category':<20} {'Sparsity':<10} {'Mean Act':<12} {'Max Act':<12} {'SNR'}")
        print("-" * 80)

        for concept, metrics in sorted(results.items()):
            print(f"{concept:<15} {metrics['category']:<20} "
                  f"{metrics['sparsity']:<10.2%} "
                  f"{metrics['mean_abs_activation']:<12.6f} "
                  f"{metrics['max_abs_activation']:<12.6f} "
                  f"{metrics['snr']:<8.3f}")

        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        avg_sparsity = np.mean([m['sparsity'] for m in results.values()])
        avg_activation = np.mean([m['mean_abs_activation'] for m in results.values()])
        avg_snr = np.mean([m['snr'] for m in results.values()])

        print(f"Average sparsity: {avg_sparsity:.2%}")
        print(f"Average mean activation: {avg_activation:.6f}")
        print(f"Average SNR: {avg_snr:.3f}")

        # Stability assessment
        print("\n" + "=" * 80)
        print("STABILITY ASSESSMENT")
        print("=" * 80)

        if avg_sparsity > 0.8:
            print("✓ Good sparsity (>80%)")
        else:
            print("⚠ Low sparsity - consider increasing TopK threshold")

        if avg_snr > 1.0:
            print("✓ Good signal-to-noise ratio")
        else:
            print("⚠ Low SNR - activations may be noisy")

        if avg_activation > 1e-5:
            print("✓ Detectable activation differences from baseline")
        else:
            print("⚠ Very small activations - may need different baseline")

        # Category analysis
        print("\n" + "=" * 80)
        print("BY CATEGORY")
        print("=" * 80)

        categories = {}
        for concept, metrics in results.items():
            cat = metrics['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(metrics['mean_abs_activation'])

        for cat, activations in sorted(categories.items()):
            avg = np.mean(activations)
            std = np.std(activations)
            print(f"{cat:<20} Mean: {avg:.6f}, Std: {std:.6f}, N={len(activations)}")

        return results


def main():
    """Main execution."""
    storage_path = Path("data/concept_activations.h5")

    if not storage_path.exists():
        print(f"Error: {storage_path} not found")
        print("Run 'python scripts/capture_concepts.py' first")
        return 1

    results = analyze_concept_stability(storage_path)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
