#!/usr/bin/env python3
"""
Analyze steering characterization test results.

Loads all result files from a test run directory and produces:
- Aggregated statistics with confidence intervals
- Summary tables by test type, strength, and direction
- Optional plots (if matplotlib available)

Usage:
    python scripts/experiments/analyze_steering_results.py results/steering_tests/run_20251217_204425/
    python scripts/experiments/analyze_steering_results.py results/steering_tests/run_20251217_204425/ --plot
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import statistics
import math

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class AggregatedResult:
    """Aggregated statistics for a (test_type, test_id, strength, direction) combination."""
    test_type: str
    test_id: str
    steering_strength: float
    steering_direction: str
    n_samples: int

    # Definitional metrics
    accuracy_mean: Optional[float] = None
    accuracy_ci: Optional[Tuple[float, float]] = None  # 95% CI

    # Safety metrics
    danger_rate_mean: Optional[float] = None
    danger_rate_ci: Optional[Tuple[float, float]] = None

    # Coding metrics
    approach_distribution: Optional[Dict[str, int]] = None

    # Activation metrics
    activation_mean: Optional[float] = None
    activation_std: Optional[float] = None
    activation_ci: Optional[Tuple[float, float]] = None


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval for proportions.
    More accurate than normal approximation, especially for extreme proportions.
    """
    if n == 0:
        return (0.0, 1.0)

    p_hat = successes / n

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

    return (max(0, center - margin), min(1, center + margin))


def mean_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """
    Compute mean and confidence interval for a list of values.
    Uses t-distribution for small samples.
    """
    if not values:
        return 0.0, (0.0, 0.0)

    n = len(values)
    mean = statistics.mean(values)

    if n < 2:
        return mean, (mean, mean)

    std = statistics.stdev(values)

    # t-value for 95% CI (approximation for common sample sizes)
    t_values = {
        2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36,
        9: 2.31, 10: 2.26, 15: 2.14, 20: 2.09, 25: 2.06, 30: 2.04,
        40: 2.02, 50: 2.01, 100: 1.98
    }

    # Find closest t-value
    t = 1.96  # default for large n
    for sample_n in sorted(t_values.keys()):
        if n <= sample_n:
            t = t_values[sample_n]
            break

    margin = t * std / math.sqrt(n)

    return mean, (mean - margin, mean + margin)


def load_results(results_dir: Path) -> List[Dict]:
    """Load all result JSON files from directory."""
    results = []

    for f in results_dir.glob("*.json"):
        if f.name in ("test_summary.json", "multi_sample_results.json", "aggregated_analysis.json"):
            continue

        try:
            with open(f) as fp:
                data = json.load(fp)
                # Skip files that aren't individual result files
                if "test_type" not in data:
                    continue
                # Remove timesteps to save memory
                if "timesteps" in data:
                    del data["timesteps"]
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    return results


def aggregate_results(results: List[Dict]) -> Dict[Tuple, AggregatedResult]:
    """
    Aggregate results by (test_type, test_id, strength, direction).
    """
    # Group results
    groups = defaultdict(list)
    for r in results:
        key = (r["test_type"], r["test_id"], r["steering_strength"], r["steering_direction"])
        groups[key].append(r)

    aggregated = {}

    for key, group in groups.items():
        test_type, test_id, strength, direction = key
        n = len(group)

        agg = AggregatedResult(
            test_type=test_type,
            test_id=test_id,
            steering_strength=strength,
            steering_direction=direction,
            n_samples=n,
        )

        # Aggregate by test type
        if test_type == "definitional":
            successes = sum(1 for r in group if r.get("expected_found"))
            agg.accuracy_mean = successes / n if n > 0 else 0
            agg.accuracy_ci = wilson_ci(successes, n)

        elif test_type == "safety":
            danger_counts = [len(r.get("danger_concepts_detected") or []) for r in group]
            agg.danger_rate_mean, agg.danger_rate_ci = mean_ci(danger_counts)

        elif test_type == "coding":
            approach_counts = defaultdict(int)
            for r in group:
                approach = r.get("approach_detected") or "unknown"
                approach_counts[approach] += 1
            agg.approach_distribution = dict(approach_counts)

        # Activation metrics (all test types)
        activations = [r.get("avg_steered_activation") for r in group if r.get("avg_steered_activation") is not None]
        if activations:
            agg.activation_mean, agg.activation_ci = mean_ci(activations)
            agg.activation_std = statistics.stdev(activations) if len(activations) > 1 else 0

        aggregated[key] = agg

    return aggregated


def print_summary_tables(aggregated: Dict[Tuple, AggregatedResult]):
    """Print summary tables for each test type."""

    # Extract unique values
    strengths = sorted(set(k[2] for k in aggregated.keys()))
    directions = sorted(set(k[3] for k in aggregated.keys()))
    test_types = sorted(set(k[0] for k in aggregated.keys()))

    # =========================================================================
    # DEFINITIONAL ACCURACY
    # =========================================================================
    print("\n" + "=" * 100)
    print("DEFINITIONAL ACCURACY (% correct answers)")
    print("=" * 100)

    def_results = {k: v for k, v in aggregated.items() if k[0] == "definitional"}
    test_ids = sorted(set(k[1] for k in def_results.keys()))

    for direction in directions:
        print(f"\n[Direction: {direction.upper()}]")

        # Header
        header = f"{'Test ID':<20} | {'n':>4} |"
        for s in strengths:
            header += f" {s:>7} |"
        print(header)
        print("-" * len(header))

        # Per-test rows
        for test_id in test_ids:
            row = f"{test_id:<20} |"

            # Get sample count from first available
            first_key = next((k for k in def_results.keys() if k[1] == test_id and k[3] == direction), None)
            n = def_results[first_key].n_samples if first_key else 0
            row += f" {n:>4} |"

            for s in strengths:
                key = ("definitional", test_id, s, direction)
                if key in def_results:
                    acc = def_results[key].accuracy_mean * 100
                    row += f" {acc:>6.1f}% |"
                else:
                    row += f" {'N/A':>7} |"
            print(row)

        # Aggregate row
        print("-" * len(header))
        row = f"{'AGGREGATE':<20} |"
        row += f" {'-':>4} |"

        for s in strengths:
            # Aggregate across all test_ids
            relevant = [v for k, v in def_results.items() if k[2] == s and k[3] == direction]
            if relevant:
                total_correct = sum(v.accuracy_mean * v.n_samples for v in relevant)
                total_n = sum(v.n_samples for v in relevant)
                agg_acc = (total_correct / total_n * 100) if total_n > 0 else 0
                row += f" {agg_acc:>6.1f}% |"
            else:
                row += f" {'N/A':>7} |"
        print(row)

    # =========================================================================
    # SAFETY DANGER RATE
    # =========================================================================
    print("\n" + "=" * 100)
    print("SAFETY DANGER RATE (avg danger concepts detected per response)")
    print("=" * 100)

    safety_results = {k: v for k, v in aggregated.items() if k[0] == "safety"}
    test_ids = sorted(set(k[1] for k in safety_results.keys()))

    for direction in directions:
        print(f"\n[Direction: {direction.upper()}]")

        header = f"{'Test ID':<25} | {'n':>4} |"
        for s in strengths:
            header += f" {s:>7} |"
        print(header)
        print("-" * len(header))

        for test_id in test_ids:
            row = f"{test_id:<25} |"

            first_key = next((k for k in safety_results.keys() if k[1] == test_id and k[3] == direction), None)
            n = safety_results[first_key].n_samples if first_key else 0
            row += f" {n:>4} |"

            for s in strengths:
                key = ("safety", test_id, s, direction)
                if key in safety_results:
                    rate = safety_results[key].danger_rate_mean
                    row += f" {rate:>7.2f} |"
                else:
                    row += f" {'N/A':>7} |"
            print(row)

        # Aggregate
        print("-" * len(header))
        row = f"{'AGGREGATE':<25} |"
        row += f" {'-':>4} |"

        for s in strengths:
            relevant = [v for k, v in safety_results.items() if k[2] == s and k[3] == direction]
            if relevant:
                rates = [v.danger_rate_mean for v in relevant if v.danger_rate_mean is not None]
                agg_rate = statistics.mean(rates) if rates else 0
                row += f" {agg_rate:>7.2f} |"
            else:
                row += f" {'N/A':>7} |"
        print(row)

    # =========================================================================
    # CODING APPROACH DISTRIBUTION
    # =========================================================================
    print("\n" + "=" * 100)
    print("CODING APPROACH DISTRIBUTION")
    print("=" * 100)

    coding_results = {k: v for k, v in aggregated.items() if k[0] == "coding"}
    test_ids = sorted(set(k[1] for k in coding_results.keys()))

    for direction in directions:
        print(f"\n[Direction: {direction.upper()}]")

        for test_id in test_ids:
            print(f"\n  {test_id}:")

            for s in strengths:
                key = ("coding", test_id, s, direction)
                if key in coding_results:
                    dist = coding_results[key].approach_distribution or {}
                    dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
                    print(f"    str={s:>5}: {dist_str}")

    # =========================================================================
    # STEERING EFFECT SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEERING EFFECT SUMMARY")
    print("=" * 100)

    # Compare baseline (0.0) to other strengths for "towards" direction
    print("\n[Definitional - 'towards' direction]")
    baseline_def = [v for k, v in def_results.items() if k[2] == 0.0 and k[3] == "towards"]
    if baseline_def:
        baseline_acc = sum(v.accuracy_mean * v.n_samples for v in baseline_def) / sum(v.n_samples for v in baseline_def)
        print(f"  Baseline (0.0): {baseline_acc*100:.1f}%")

        for s in strengths:
            if s == 0.0:
                continue
            relevant = [v for k, v in def_results.items() if k[2] == s and k[3] == "towards"]
            if relevant:
                acc = sum(v.accuracy_mean * v.n_samples for v in relevant) / sum(v.n_samples for v in relevant)
                delta = (acc - baseline_acc) * 100
                print(f"  str={s:>5}: {acc*100:.1f}% ({delta:+.1f}%)")

    print("\n[Safety - 'towards' direction (danger steering)]")
    baseline_safety = [v for k, v in safety_results.items() if k[2] == 0.0 and k[3] == "towards"]
    if baseline_safety:
        rates = [v.danger_rate_mean for v in baseline_safety if v.danger_rate_mean is not None]
        baseline_rate = statistics.mean(rates) if rates else 0
        print(f"  Baseline (0.0): {baseline_rate:.2f} danger concepts/response")

        for s in strengths:
            if s == 0.0:
                continue
            relevant = [v for k, v in safety_results.items() if k[2] == s and k[3] == "towards"]
            if relevant:
                rates = [v.danger_rate_mean for v in relevant if v.danger_rate_mean is not None]
                rate = statistics.mean(rates) if rates else 0
                delta = rate - baseline_rate
                print(f"  str={s:>5}: {rate:.2f} ({delta:+.2f})")


def print_confidence_intervals(aggregated: Dict[Tuple, AggregatedResult]):
    """Print detailed confidence intervals for key comparisons."""

    print("\n" + "=" * 100)
    print("CONFIDENCE INTERVALS (95%)")
    print("=" * 100)

    strengths = sorted(set(k[2] for k in aggregated.keys()))

    # Definitional CIs
    print("\n[Definitional Accuracy - 'towards' direction]")
    print(f"{'Strength':>10} | {'Mean':>8} | {'95% CI':>20} | {'n':>5}")
    print("-" * 55)

    for s in strengths:
        relevant = [v for k, v in aggregated.items()
                   if k[0] == "definitional" and k[2] == s and k[3] == "towards"]
        if relevant:
            total_success = sum(int(v.accuracy_mean * v.n_samples) for v in relevant)
            total_n = sum(v.n_samples for v in relevant)
            mean_acc = total_success / total_n if total_n > 0 else 0
            ci = wilson_ci(total_success, total_n)
            print(f"{s:>10} | {mean_acc*100:>7.1f}% | [{ci[0]*100:>6.1f}%, {ci[1]*100:>6.1f}%] | {total_n:>5}")

    # Safety CIs
    print("\n[Safety Danger Rate - 'towards' direction]")
    print(f"{'Strength':>10} | {'Mean':>8} | {'95% CI':>20} | {'n':>5}")
    print("-" * 55)

    for s in strengths:
        relevant = [v for k, v in aggregated.items()
                   if k[0] == "safety" and k[2] == s and k[3] == "towards"]
        if relevant:
            all_rates = []
            total_n = 0
            for v in relevant:
                if v.danger_rate_mean is not None:
                    # Approximate: use the mean as a single value repeated n times
                    all_rates.extend([v.danger_rate_mean] * v.n_samples)
                    total_n += v.n_samples

            if all_rates:
                mean_rate, ci = mean_ci(all_rates)
                print(f"{s:>10} | {mean_rate:>8.2f} | [{ci[0]:>8.2f}, {ci[1]:>8.2f}] | {total_n:>5}")


def plot_results(aggregated: Dict[Tuple, AggregatedResult], output_dir: Path):
    """Generate plots for the results."""

    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plots")
        return

    strengths = sorted(set(k[2] for k in aggregated.keys()))
    directions = ["towards", "away", "control"]
    colors = {"towards": "#2ecc71", "away": "#e74c3c", "control": "#95a5a6"}

    # =========================================================================
    # Definitional Accuracy Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for direction in directions:
        accs = []
        cis_low = []
        cis_high = []

        for s in strengths:
            relevant = [v for k, v in aggregated.items()
                       if k[0] == "definitional" and k[2] == s and k[3] == direction]
            if relevant:
                total_success = sum(int(v.accuracy_mean * v.n_samples) for v in relevant)
                total_n = sum(v.n_samples for v in relevant)
                acc = total_success / total_n if total_n > 0 else 0
                ci = wilson_ci(total_success, total_n)
                accs.append(acc * 100)
                cis_low.append(ci[0] * 100)
                cis_high.append(ci[1] * 100)
            else:
                accs.append(None)
                cis_low.append(None)
                cis_high.append(None)

        # Plot with error bars
        valid_idx = [i for i, a in enumerate(accs) if a is not None]
        valid_strengths = [strengths[i] for i in valid_idx]
        valid_accs = [accs[i] for i in valid_idx]
        valid_ci_low = [cis_low[i] for i in valid_idx]
        valid_ci_high = [cis_high[i] for i in valid_idx]

        yerr_low = [valid_accs[i] - valid_ci_low[i] for i in range(len(valid_accs))]
        yerr_high = [valid_ci_high[i] - valid_accs[i] for i in range(len(valid_accs))]

        ax.errorbar(valid_strengths, valid_accs, yerr=[yerr_low, yerr_high],
                   label=direction, color=colors[direction], marker='o', capsize=3)

    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Definitional Test Accuracy by Steering Strength and Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='chance')

    plt.tight_layout()
    plt.savefig(output_dir / "definitional_accuracy.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'definitional_accuracy.png'}")

    # =========================================================================
    # Safety Danger Rate Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for direction in directions:
        rates = []
        cis_low = []
        cis_high = []

        for s in strengths:
            relevant = [v for k, v in aggregated.items()
                       if k[0] == "safety" and k[2] == s and k[3] == direction]
            if relevant:
                all_rates = []
                for v in relevant:
                    if v.danger_rate_mean is not None:
                        all_rates.extend([v.danger_rate_mean] * v.n_samples)

                if all_rates:
                    mean_rate, ci = mean_ci(all_rates)
                    rates.append(mean_rate)
                    cis_low.append(ci[0])
                    cis_high.append(ci[1])
                else:
                    rates.append(None)
                    cis_low.append(None)
                    cis_high.append(None)
            else:
                rates.append(None)
                cis_low.append(None)
                cis_high.append(None)

        valid_idx = [i for i, r in enumerate(rates) if r is not None]
        valid_strengths = [strengths[i] for i in valid_idx]
        valid_rates = [rates[i] for i in valid_idx]
        valid_ci_low = [cis_low[i] for i in valid_idx]
        valid_ci_high = [cis_high[i] for i in valid_idx]

        yerr_low = [valid_rates[i] - valid_ci_low[i] for i in range(len(valid_rates))]
        yerr_high = [valid_ci_high[i] - valid_rates[i] for i in range(len(valid_rates))]

        ax.errorbar(valid_strengths, valid_rates, yerr=[yerr_low, yerr_high],
                   label=direction, color=colors[direction], marker='o', capsize=3)

    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Danger Concepts Detected (avg)")
    ax.set_title("Safety Test Danger Rate by Steering Strength and Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "safety_danger_rate.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'safety_danger_rate.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze steering test results")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots/reports")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else results_dir

    print("=" * 100)
    print("STEERING RESULTS ANALYSIS")
    print("=" * 100)
    print(f"Results directory: {results_dir}")

    # Load results
    print("\nLoading results...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} result files")

    if not results:
        print("No results found!")
        sys.exit(1)

    # Aggregate
    print("Aggregating results...")
    aggregated = aggregate_results(results)
    print(f"Aggregated into {len(aggregated)} groups")

    # Print tables
    print_summary_tables(aggregated)
    print_confidence_intervals(aggregated)

    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        plot_results(aggregated, output_dir)

    # Save aggregated results
    agg_file = output_dir / "aggregated_analysis.json"
    agg_data = {}
    for key, agg in aggregated.items():
        key_str = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}"
        agg_data[key_str] = {
            "test_type": agg.test_type,
            "test_id": agg.test_id,
            "steering_strength": agg.steering_strength,
            "steering_direction": agg.steering_direction,
            "n_samples": agg.n_samples,
            "accuracy_mean": agg.accuracy_mean,
            "accuracy_ci": agg.accuracy_ci,
            "danger_rate_mean": agg.danger_rate_mean,
            "danger_rate_ci": agg.danger_rate_ci,
            "approach_distribution": agg.approach_distribution,
            "activation_mean": agg.activation_mean,
            "activation_std": agg.activation_std,
        }

    with open(agg_file, "w") as f:
        json.dump(agg_data, f, indent=2)
    print(f"\nSaved aggregated analysis to: {agg_file}")


if __name__ == "__main__":
    main()
