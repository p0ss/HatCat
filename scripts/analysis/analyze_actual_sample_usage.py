#!/usr/bin/env python3
"""
Analyze actual sample usage from training runs.

Key question: How many concepts actually graduated with minimum samples?
"""

import json
from pathlib import Path
from typing import Dict, List
import statistics

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "sumo_classifiers_with_broken_ai_safety"


def analyze_sample_usage(layer: int) -> Dict:
    """Analyze how many samples concepts actually used."""
    results_path = RESULTS_DIR / f"layer{layer}" / "results.json"

    if not results_path.exists():
        return None

    with open(results_path) as f:
        data = json.load(f)

    concepts = data["results"]

    # Track iteration counts
    iteration_counts = []
    graduated_at_min = 0  # Graduated with 10 samples (iteration 1)
    graduated_tier1 = 0   # Graduated at 30 samples (iteration 2)
    graduated_tier2 = 0   # Graduated at 60 samples (iteration 3)
    graduated_tier3 = 0   # Graduated at 90+ samples (iteration 4+)

    for concept in concepts:
        iters = concept.get("total_iterations", concept.get("activation_iterations", 0))
        iteration_counts.append(iters)

        if iters <= 1:
            graduated_at_min += 1
        elif iters <= 2:
            graduated_tier1 += 1
        elif iters <= 3:
            graduated_tier2 += 1
        else:
            graduated_tier3 += 1

    total = len(concepts)

    return {
        "layer": layer,
        "total_concepts": total,
        "graduated_at_min": graduated_at_min,
        "graduated_tier1": graduated_tier1,
        "graduated_tier2": graduated_tier2,
        "graduated_tier3": graduated_tier3,
        "avg_iterations": statistics.mean(iteration_counts),
        "median_iterations": statistics.median(iteration_counts),
        "max_iterations": max(iteration_counts),
        "iteration_distribution": iteration_counts,
    }


def main():
    print("="*80)
    print("ACTUAL SAMPLE USAGE ANALYSIS")
    print("="*80)
    print("\nQuestion: How many concepts graduated with minimum samples?")

    print("\nAdaptive training schedule:")
    print("  Iteration 1: 10 samples (minimum)")
    print("  Iteration 2: 30 samples (+20)")
    print("  Iteration 3: 60 samples (+30)")
    print("  Iteration 4+: 90+ samples (+30 each)")

    total_graduated_min = 0
    total_graduated_tier1 = 0
    total_graduated_tier2 = 0
    total_graduated_tier3 = 0
    total_all_concepts = 0

    for layer in range(6):
        analysis = analyze_sample_usage(layer)
        if not analysis:
            continue

        total = analysis["total_concepts"]
        total_all_concepts += total

        grad_min = analysis["graduated_at_min"]
        grad_t1 = analysis["graduated_tier1"]
        grad_t2 = analysis["graduated_tier2"]
        grad_t3 = analysis["graduated_tier3"]

        total_graduated_min += grad_min
        total_graduated_tier1 += grad_t1
        total_graduated_tier2 += grad_t2
        total_graduated_tier3 += grad_t3

        print(f"\n{'='*80}")
        print(f"Layer {layer} ({total} concepts)")
        print(f"{'='*80}")

        print(f"\nGraduation breakdown:")
        print(f"  Iteration 1 (10 samples):     {grad_min:4d} ({grad_min/total*100:5.1f}%)")
        print(f"  Iteration 2 (30 samples):     {grad_t1:4d} ({grad_t1/total*100:5.1f}%)")
        print(f"  Iteration 3 (60 samples):     {grad_t2:4d} ({grad_t2/total*100:5.1f}%)")
        print(f"  Iteration 4+ (90+ samples):   {grad_t3:4d} ({grad_t3/total*100:5.1f}%)")

        print(f"\nIteration statistics:")
        print(f"  Average: {analysis['avg_iterations']:.1f}")
        print(f"  Median:  {analysis['median_iterations']:.1f}")
        print(f"  Max:     {analysis['max_iterations']}")

        # Cumulative percentages
        cum_1 = grad_min
        cum_2 = grad_min + grad_t1
        cum_3 = grad_min + grad_t1 + grad_t2

        print(f"\nCumulative graduation rate:")
        print(f"  By iteration 1: {cum_1/total*100:5.1f}%")
        print(f"  By iteration 2: {cum_2/total*100:5.1f}%")
        print(f"  By iteration 3: {cum_3/total*100:5.1f}%")

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal concepts analyzed: {total_all_concepts}")
    print(f"\nGraduation breakdown across all layers:")
    print(f"  Iteration 1 (10 samples):     {total_graduated_min:4d} ({total_graduated_min/total_all_concepts*100:5.1f}%)")
    print(f"  Iteration 2 (30 samples):     {total_graduated_tier1:4d} ({total_graduated_tier1/total_all_concepts*100:5.1f}%)")
    print(f"  Iteration 3 (60 samples):     {total_graduated_tier2:4d} ({total_graduated_tier2/total_all_concepts*100:5.1f}%)")
    print(f"  Iteration 4+ (90+ samples):   {total_graduated_tier3:4d} ({total_graduated_tier3/total_all_concepts*100:5.1f}%)")

    cum_1_total = total_graduated_min
    cum_2_total = total_graduated_min + total_graduated_tier1
    cum_3_total = total_graduated_min + total_graduated_tier1 + total_graduated_tier2

    print(f"\nCumulative graduation rate (all layers):")
    print(f"  By iteration 1: {cum_1_total/total_all_concepts*100:5.1f}%")
    print(f"  By iteration 2: {cum_2_total/total_all_concepts*100:5.1f}%")
    print(f"  By iteration 3: {cum_3_total/total_all_concepts*100:5.1f}%")

    # Key finding
    print(f"\n{'='*80}")
    print("KEY FINDING")
    print(f"{'='*80}")

    if total_graduated_min / total_all_concepts < 0.3:
        print(f"\n⚠️  ONLY {total_graduated_min/total_all_concepts*100:.1f}% graduated with minimum samples!")
        print(f"  The user is CORRECT - most concepts need MORE than 10 samples.")
        print(f"\n  {total_graduated_tier3/total_all_concepts*100:.1f}% needed 90+ samples (iteration 4+)")
        print(f"  This suggests starting with 20-30 samples might be more efficient.")
    else:
        print(f"\n✓ {total_graduated_min/total_all_concepts*100:.1f}% graduated with minimum samples")
        print(f"  Current min=10 seems appropriate.")

    # Recalculate time impact with realistic distribution
    print(f"\n{'='*80}")
    print("REVISED TIME ESTIMATE")
    print(f"{'='*80}")

    print("\nActual iteration distribution (not optimistic estimate):")
    print(f"  Iteration 1: {total_graduated_min/total_all_concepts*100:.1f}%")
    print(f"  Iteration 2: {total_graduated_tier1/total_all_concepts*100:.1f}%")
    print(f"  Iteration 3: {total_graduated_tier2/total_all_concepts*100:.1f}%")
    print(f"  Iteration 4+: {total_graduated_tier3/total_all_concepts*100:.1f}%")

    # Estimate impact of min=20
    print(f"\nIf we start with min=20 instead of min=10:")

    # With min=20, many concepts that needed 30 samples would graduate at iteration 1
    # Those at 60 might graduate at iteration 2, etc.

    # Current: avg ~6-18 iterations
    # With min=20: likely reduces to avg ~3-9 iterations (fewer steps to reach target)

    pct_min = total_graduated_min / total_all_concepts
    pct_tier1 = total_graduated_tier1 / total_all_concepts
    pct_tier2 = total_graduated_tier2 / total_all_concepts
    pct_tier3 = total_graduated_tier3 / total_all_concepts

    # Estimate: if concepts struggling at 10 do better at 20
    # Assume 50% of tier1 would graduate at min with 20 samples
    estimated_grad_at_20 = pct_min + (pct_tier1 * 0.5)

    print(f"  Estimated graduation at iteration 1: {estimated_grad_at_20*100:.1f}%")
    print(f"  (vs {pct_min*100:.1f}% currently)")

    # Time calculation
    # If more concepts graduate earlier, we might actually SAVE time despite higher initial cost
    # This needs more detailed analysis

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
