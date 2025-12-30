#!/usr/bin/env python3
"""
Compare actual training runs with different sample counts.

Use empirical data to determine real iteration reduction and time impact.
"""

import json
from pathlib import Path
import statistics

PROJECT_ROOT = Path(__file__).parent.parent

# Known training runs with different sample configurations
RUNS = {
    "min10_loose": "results/sumo_classifiers_layer0_retrain2/layer0/results.json",  # 10 samples
    "min50_falloff": "results/sumo_classifiers_layer0_nephew_fixed/layer0/results.json",  # 50 samples
    "min50_falloff_old": "results/sumo_classifiers_layer0_retrain/layer0/results.json",  # 50 samples
}


def load_run(path: str):
    """Load training results."""
    full_path = PROJECT_ROOT / path
    if not full_path.exists():
        return None

    with open(full_path) as f:
        return json.load(f)


def analyze_run(data: dict, name: str):
    """Extract key metrics from a run."""
    if not data:
        return None

    results = data.get("results", [])
    if not results:
        return None

    iterations = []
    times = []
    test_f1s = []

    for concept in results:
        iters = concept.get("total_iterations", concept.get("activation_iterations", 0))
        time = concept.get("total_time", 0)
        f1 = concept.get("test_f1", 0)

        if iters > 0:
            iterations.append(iters)
        if time > 0:
            times.append(time)
        if f1 > 0:
            test_f1s.append(f1)

    return {
        "name": name,
        "n_concepts": len(results),
        "total_time_minutes": data.get("elapsed_minutes", 0),
        "avg_iterations": statistics.mean(iterations) if iterations else 0,
        "median_iterations": statistics.median(iterations) if iterations else 0,
        "avg_time_per_concept": statistics.mean(times) if times else 0,
        "avg_test_f1": statistics.mean(test_f1s) if test_f1s else 0,
    }


def main():
    print("="*80)
    print("EMPIRICAL SAMPLE COUNT COMPARISON")
    print("="*80)
    print("\nComparing actual training runs with different sample configurations:")

    analyses = {}

    for run_name, path in RUNS.items():
        data = load_run(path)
        analysis = analyze_run(data, run_name)
        if analysis:
            analyses[run_name] = analysis
            print(f"\n✓ Loaded: {run_name}")
            print(f"  Path: {path}")
            print(f"  Concepts: {analysis['n_concepts']}")
        else:
            print(f"\n✗ Failed to load: {run_name}")

    if not analyses:
        print("\n⚠️  No data available for comparison")
        return

    # Compare min=10 vs min=50
    print("\n" + "="*80)
    print("COMPARISON: min=10 vs min=50")
    print("="*80)

    min10 = analyses.get("min10_loose")
    min50 = analyses.get("min50_falloff") or analyses.get("min50_falloff_old")

    if min10 and min50:
        print(f"\n{'Metric':<25} {'min=10':<20} {'min=50':<20} {'Change':<20}")
        print("-"*85)

        metrics = [
            ("Concepts", "n_concepts", ""),
            ("Total time (min)", "total_time_minutes", "min"),
            ("Avg iterations", "avg_iterations", "iters"),
            ("Median iterations", "median_iterations", "iters"),
            ("Avg time per concept", "avg_time_per_concept", "sec"),
            ("Avg test F1", "avg_test_f1", ""),
        ]

        for label, key, unit in metrics:
            val10 = min10[key]
            val50 = min50[key]

            if val10 > 0:
                ratio = val50 / val10
                change_pct = (ratio - 1) * 100
            else:
                ratio = 0
                change_pct = 0

            val10_str = f"{val10:.2f} {unit}".strip()
            val50_str = f"{val50:.2f} {unit}".strip()

            if key in ["avg_iterations", "median_iterations", "total_time_minutes"]:
                # Lower is better
                if ratio < 1.0:
                    change_str = f"{ratio:.2f}x ({change_pct:.0f}%) ✓"
                else:
                    change_str = f"{ratio:.2f}x ({change_pct:+.0f}%)"
            elif key in ["avg_test_f1"]:
                # Higher is better
                if ratio > 1.0:
                    change_str = f"{ratio:.2f}x ({change_pct:+.0f}%) ✓"
                else:
                    change_str = f"{ratio:.2f}x ({change_pct:+.0f}%)"
            else:
                change_str = f"{ratio:.2f}x"

            print(f"{label:<25} {val10_str:<20} {val50_str:<20} {change_str:<20}")

        # Calculate iteration reduction
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)

        iter_reduction = 1 - (min50["avg_iterations"] / min10["avg_iterations"])
        time_ratio = min50["total_time_minutes"] / min10["total_time_minutes"]

        print(f"\nIteration reduction with min=50:")
        print(f"  min=10: {min10['avg_iterations']:.1f} avg iterations")
        print(f"  min=50: {min50['avg_iterations']:.1f} avg iterations")
        print(f"  Reduction: {iter_reduction*100:.0f}%")

        print(f"\nTotal training time:")
        print(f"  min=10: {min10['total_time_minutes']:.1f} minutes")
        print(f"  min=50: {min50['total_time_minutes']:.1f} minutes")
        print(f"  Ratio: {time_ratio:.2f}x ({(time_ratio-1)*100:+.0f}%)")

        # Extrapolate to min=20 and min=30
        print("\n" + "="*80)
        print("EXTRAPOLATION TO min=20 and min=30")
        print("="*80)

        # Assume iteration reduction scales with log(samples)
        # min=10 → min=20: reduce by (reduction_50 * log(20/10) / log(50/10))
        # min=10 → min=30: reduce by (reduction_50 * log(30/10) / log(50/10))

        import math

        log_ratio_20 = math.log(20/10) / math.log(50/10)
        log_ratio_30 = math.log(30/10) / math.log(50/10)

        iter_reduction_20 = iter_reduction * log_ratio_20
        iter_reduction_30 = iter_reduction * log_ratio_30

        iters_20 = min10["avg_iterations"] * (1 - iter_reduction_20)
        iters_30 = min10["avg_iterations"] * (1 - iter_reduction_30)

        # Time = concepts × iterations × (time_per_iter × sample_multiplier)
        time_per_iter_10 = min10["avg_time_per_concept"] / min10["avg_iterations"]

        time_20 = min10["n_concepts"] * iters_20 * (time_per_iter_10 * 2) / 60
        time_30 = min10["n_concepts"] * iters_30 * (time_per_iter_10 * 3) / 60

        print(f"\nEstimated with min=20:")
        print(f"  Iterations: {iters_20:.1f} (reduction: {iter_reduction_20*100:.0f}%)")
        print(f"  Total time: {time_20:.1f} min ({time_20/min10['total_time_minutes']:.2f}x)")

        print(f"\nEstimated with min=30:")
        print(f"  Iterations: {iters_30:.1f} (reduction: {iter_reduction_30*100:.0f}%)")
        print(f"  Total time: {time_30:.1f} min ({time_30/min10['total_time_minutes']:.2f}x)")

        # Recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)

        if time_20 < min10["total_time_minutes"]:
            print(f"\n✓ min=20 appears FASTER ({time_20/min10['total_time_minutes']:.2f}x)")
            print(f"  Projected: {time_20:.1f} min vs {min10['total_time_minutes']:.1f} min (min=10)")
        elif time_20 / min10["total_time_minutes"] < 1.2:
            print(f"\n✓ min=20 is acceptable ({time_20/min10['total_time_minutes']:.2f}x, <1.2x slower)")
        else:
            print(f"\n⚠️  min=20 may be too slow ({time_20/min10['total_time_minutes']:.2f}x)")

        if time_30 < min10["total_time_minutes"]:
            print(f"\n✓ min=30 appears FASTER ({time_30/min10['total_time_minutes']:.2f}x)")
            print(f"  Projected: {time_30:.1f} min vs {min10['total_time_minutes']:.1f} min (min=10)")
        elif time_30 / min10["total_time_minutes"] < 1.2:
            print(f"\n✓ min=30 is acceptable ({time_30/min10['total_time_minutes']:.2f}x, <1.2x slower)")
        else:
            print(f"\n⚠️  min=30 may be too slow ({time_30/min10['total_time_minutes']:.2f}x)")

        # For full training
        print(f"\n{'='*80}")
        print("FULL TRAINING PROJECTION (all layers)")
        print(f"{'='*80}")

        # Assuming Layer 0 pattern holds across layers
        # Current full training: ~36 hours with min=10
        current_full = 36 * 60  # minutes

        projected_20 = current_full * (time_20 / min10["total_time_minutes"])
        projected_30 = current_full * (time_30 / min10["total_time_minutes"])
        projected_50 = current_full * (time_ratio)

        print(f"\nProjected full training time:")
        print(f"  min=10: {current_full/60:.1f}h (baseline)")
        print(f"  min=20: {projected_20/60:.1f}h ({projected_20/current_full:.2f}x)")
        print(f"  min=30: {projected_30/60:.1f}h ({projected_30/current_full:.2f}x)")
        print(f"  min=50: {projected_50/60:.1f}h ({projected_50/current_full:.2f}x)")

    else:
        print("\n⚠️  Missing data for min=10 or min=50 comparison")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
