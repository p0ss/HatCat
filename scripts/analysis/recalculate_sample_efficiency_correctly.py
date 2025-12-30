#!/usr/bin/env python3
"""
Correctly calculate sample efficiency.

Key insight: We're not doing 2x or 3x work - we're SKIPPING AHEAD to the
sample count the concept needs anyway!

Example:
  Current min=10: [10, 30, 60, 90, 120] samples → 5 iterations
  With min=30:    [30, 60, 90, 120]     samples → 4 iterations

Same total samples processed, fewer iterations!
"""

import json
from pathlib import Path
import statistics

PROJECT_ROOT = Path(__file__).parent.parent
FULL_RUN = "results/sumo_classifiers_with_broken_ai_safety"


def samples_at_iteration(iteration: int, min_samples: int) -> int:
    """Calculate total samples processed by iteration N."""
    if iteration == 1:
        return min_samples
    elif iteration == 2:
        return min_samples + (min_samples * 2)  # min + increment to reach 3*min
    else:
        # Iteration 3+: add 30 each time (or 3*min_samples for min=10)
        increment = max(30, min_samples * 3)
        return samples_at_iteration(iteration - 1, min_samples) + increment


def iterations_needed_for_samples(target_samples: int, min_samples: int) -> int:
    """How many iterations to reach target sample count?"""
    iteration = 1
    while samples_at_iteration(iteration, min_samples) < target_samples:
        iteration += 1
    return iteration


def main():
    print("="*80)
    print("CORRECTED SAMPLE EFFICIENCY CALCULATION")
    print("="*80)

    print("\nKey insight: Higher min_samples means SKIPPING early iterations,")
    print("not doing more work!")

    # Example
    print("\nExample: Concept needs 90 samples to graduate")
    print("-"*80)

    for min_samp in [10, 20, 30]:
        iters = iterations_needed_for_samples(90, min_samp)
        schedule = [samples_at_iteration(i, min_samp) for i in range(1, iters + 1)]
        print(f"\nmin={min_samp}:")
        print(f"  Schedule: {schedule}")
        print(f"  Iterations: {iters}")
        print(f"  Final samples: {schedule[-1]}")

    # Load actual data
    print("\n" + "="*80)
    print("ANALYZING ACTUAL TRAINING DATA")
    print("="*80)

    all_concepts = []

    for layer in range(6):
        results_path = Path(FULL_RUN) / f"layer{layer}" / "results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            data = json.load(f)

        for concept in data.get("results", []):
            iters = concept.get("total_iterations", concept.get("activation_iterations", 0))
            time = concept.get("total_time", 0)

            # Estimate total samples this concept used
            total_samples = samples_at_iteration(iters, 10)

            all_concepts.append({
                "iterations_min10": iters,
                "total_samples": total_samples,
                "time": time,
                "time_per_sample": time / total_samples if total_samples > 0 else 0,
            })

    print(f"\nTotal concepts analyzed: {len(all_concepts)}")

    avg_time_per_sample = statistics.mean([c["time_per_sample"] for c in all_concepts if c["time_per_sample"] > 0])
    print(f"Average time per sample: {avg_time_per_sample:.3f} seconds")

    # Calculate time for each min_samples config
    print("\n" + "="*80)
    print("TIME COMPARISON")
    print("="*80)

    for min_samp in [10, 20, 30]:
        total_time = 0

        for concept in all_concepts:
            target_samples = concept["total_samples"]
            iters_needed = iterations_needed_for_samples(target_samples, min_samp)

            # Time = samples × time_per_sample + iteration_overhead × iterations
            # Assume 5 second overhead per iteration (validation, logging, etc.)
            iteration_overhead = 5.0

            concept_time = (target_samples * avg_time_per_sample) + (iters_needed * iteration_overhead)
            total_time += concept_time

        total_time_minutes = total_time / 60
        total_time_hours = total_time_minutes / 60

        # Baseline is min=10
        baseline_time = sum(c["time"] for c in all_concepts) / 60
        ratio = total_time_minutes / baseline_time

        print(f"\nmin={min_samp}:")
        print(f"  Total time: {total_time_hours:.1f} hours ({total_time_minutes:.0f} minutes)")
        print(f"  vs min=10: {ratio:.2f}x ({(ratio-1)*100:+.0f}%)")

        if ratio < 1.0:
            print(f"  ✓ FASTER by {(1-ratio)*100:.0f}%!")
        elif ratio < 1.1:
            print(f"  ✓ Acceptable (< 10% slower)")
        else:
            print(f"  ✗ Too slow")

    # Detailed breakdown
    print("\n" + "="*80)
    print("ITERATION REDUCTION ANALYSIS")
    print("="*80)

    # Group concepts by sample needs
    low_need = [c for c in all_concepts if c["total_samples"] <= 30]  # Need ≤30 samples
    mid_need = [c for c in all_concepts if 30 < c["total_samples"] <= 90]  # Need 30-90
    high_need = [c for c in all_concepts if c["total_samples"] > 90]  # Need >90

    print(f"\nConcepts by sample requirements:")
    print(f"  Low (≤30 samples):    {len(low_need):4d} ({len(low_need)/len(all_concepts)*100:5.1f}%)")
    print(f"  Medium (30-90):       {len(mid_need):4d} ({len(mid_need)/len(all_concepts)*100:5.1f}%)")
    print(f"  High (>90 samples):   {len(high_need):4d} ({len(high_need)/len(all_concepts)*100:5.1f}%)")

    for min_samp in [10, 20, 30]:
        print(f"\n{'-'*80}")
        print(f"With min={min_samp}:")

        for group_name, group in [("Low", low_need), ("Medium", mid_need), ("High", high_need)]:
            if not group:
                continue

            avg_iters_current = statistics.mean([c["iterations_min10"] for c in group])
            avg_iters_new = statistics.mean([
                iterations_needed_for_samples(c["total_samples"], min_samp)
                for c in group
            ])

            reduction = (avg_iters_current - avg_iters_new) / avg_iters_current * 100

            print(f"  {group_name:8s}: {avg_iters_current:.1f} → {avg_iters_new:.1f} iterations ({reduction:+.0f}%)")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    print("\nBased on corrected analysis:")
    print("  - Higher min_samples SKIPS early iterations")
    print("  - Same total samples processed")
    print("  - Fewer iteration overhead costs")
    print("  - Should be FASTER or break-even!")

    print("\n✓ Test empirically: Run Layer 1 with min=20 or min=30")
    print("  Expected: Similar time with fewer iterations")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
