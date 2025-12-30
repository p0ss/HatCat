#!/usr/bin/env python3
"""
Properly analyze sample efficiency by looking at:
1. How many concepts in the FULL run (min=10) graduated early vs late
2. Estimate time savings if late graduates had started with more samples

The key insight: We should analyze the FULL training run (all concepts)
and see how many needed 60+ or 90+ samples total.
"""

import json
from pathlib import Path
import statistics

PROJECT_ROOT = Path(__file__).parent.parent

# Use the full training run with validation
FULL_RUN = "results/sumo_classifiers_with_broken_ai_safety"


def analyze_full_run():
    """Analyze the full training run to see graduation patterns."""

    print("="*80)
    print("SAMPLE EFFICIENCY ANALYSIS - PROPER METHODOLOGY")
    print("="*80)
    print("\nAnalyzing full training run with min=10 samples")
    print("Question: How many concepts graduated early vs late?")
    print("Goal: Calculate time saved if late graduates started with more samples")

    total_early = 0  # Graduated in 1-3 iterations (10-60 samples)
    total_late = 0   # Graduated in 4+ iterations (90+ samples)
    total_concepts = 0
    total_time = 0

    # Adaptive schedule:
    # Iter 1: 10 samples
    # Iter 2: 30 samples (+20)
    # Iter 3: 60 samples (+30)
    # Iter 4: 90 samples (+30)
    # Iter 5: 120 samples (+30)

    early_concepts = []
    late_concepts = []

    for layer in range(6):
        results_path = Path(FULL_RUN) / f"layer{layer}" / "results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            data = json.load(f)

        concepts = data.get("results", [])
        layer_time = data.get("elapsed_minutes", 0)
        total_time += layer_time

        print(f"\n{'-'*80}")
        print(f"Layer {layer}: {len(concepts)} concepts, {layer_time:.1f} minutes")

        layer_early = 0
        layer_late = 0

        for concept in concepts:
            iters = concept.get("total_iterations", concept.get("activation_iterations", 0))
            time = concept.get("total_time", 0)
            name = concept.get("concept", "unknown")

            total_concepts += 1

            if iters <= 3:
                layer_early += 1
                total_early += 1
                early_concepts.append({
                    "name": name,
                    "layer": layer,
                    "iterations": iters,
                    "time": time
                })
            else:
                layer_late += 1
                total_late += 1
                late_concepts.append({
                    "name": name,
                    "layer": layer,
                    "iterations": iters,
                    "time": time
                })

        print(f"  Early graduates (1-3 iters): {layer_early} ({layer_early/len(concepts)*100:.1f}%)")
        print(f"  Late graduates (4+ iters):   {layer_late} ({layer_late/len(concepts)*100:.1f}%)")

    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")

    print(f"\nTotal concepts: {total_concepts}")
    print(f"Total time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"\nGraduation breakdown:")
    print(f"  Early (1-3 iterations):  {total_early:4d} ({total_early/total_concepts*100:5.1f}%)")
    print(f"  Late (4+ iterations):    {total_late:4d} ({total_late/total_concepts*100:5.1f}%)")

    # Calculate time breakdown
    avg_time_early = statistics.mean([c["time"] for c in early_concepts]) if early_concepts else 0
    avg_time_late = statistics.mean([c["time"] for c in late_concepts]) if late_concepts else 0

    print(f"\nAverage time per concept:")
    print(f"  Early graduates: {avg_time_early:.1f} seconds")
    print(f"  Late graduates:  {avg_time_late:.1f} seconds")

    # Key calculation: What if we started late graduates with min=20 or min=30?
    print(f"\n{'='*80}")
    print("TIME SAVINGS CALCULATION")
    print(f"{'='*80}")

    print(f"\nScenario 1: Start with min=20 instead of min=10")
    print(f"-"*80)

    # Assumptions:
    # - Early graduates (currently 1-3 iters with min=10) would still graduate in 1-3 iters with min=20
    #   But each iteration takes 2x as long (20 samples vs 10)
    # - Late graduates (currently 4+ iters with min=10) might graduate in 2-3 fewer iterations with min=20
    #   Trade: 2x time per iteration, but 33-50% fewer iterations

    # Early concepts: lose time (2x per iteration, same iterations)
    early_time_loss = total_early * avg_time_early * 1.0  # 2x time/iter, same iters = 2x total

    # Late concepts: might save time (2x per iteration, but 33% fewer iterations = 1.33x total)
    late_avg_iters = statistics.mean([c["iterations"] for c in late_concepts]) if late_concepts else 0
    late_reduced_iters = late_avg_iters * 0.67  # Assume 33% reduction

    print(f"\nEarly graduates ({total_early} concepts):")
    print(f"  Current: ~{total_early * avg_time_early / 60:.1f} min total")
    print(f"  With min=20: ~{total_early * avg_time_early * 2.0 / 60:.1f} min total (2x slower)")
    print(f"  Time lost: ~{total_early * avg_time_early * 1.0 / 60:.1f} min")

    # For late concepts, estimate time with fewer iterations
    late_time_per_iter = avg_time_late / late_avg_iters if late_avg_iters > 0 else 0
    late_current_time = total_late * avg_time_late
    late_new_time = total_late * late_reduced_iters * (late_time_per_iter * 2.0)

    print(f"\nLate graduates ({total_late} concepts):")
    print(f"  Current avg iterations: {late_avg_iters:.1f}")
    print(f"  Estimated with min=20: {late_reduced_iters:.1f} iterations (33% reduction)")
    print(f"  Current: ~{late_current_time / 60:.1f} min total")
    print(f"  With min=20: ~{late_new_time / 60:.1f} min total")
    print(f"  Time {'saved' if late_new_time < late_current_time else 'lost'}: ~{abs(late_new_time - late_current_time) / 60:.1f} min")

    net_change = (early_time_loss + late_new_time - late_current_time) / 60
    print(f"\nNet change with min=20:")
    print(f"  Early loss: +{early_time_loss / 60:.1f} min")
    print(f"  Late {'savings' if late_new_time < late_current_time else 'loss'}: {(late_new_time - late_current_time) / 60:+.1f} min")
    print(f"  TOTAL: {net_change:+.1f} min ({net_change/total_time*100:+.1f}%)")

    if net_change < 0:
        print(f"\n✓ min=20 could SAVE ~{-net_change:.0f} minutes overall!")
    else:
        print(f"\n✗ min=20 would ADD ~{net_change:.0f} minutes overall")

    # Scenario 2: min=30
    print(f"\n{'-'*80}")
    print(f"Scenario 2: Start with min=30 instead of min=10")
    print(f"-"*80)

    # Assume 50% iteration reduction for late concepts
    late_reduced_iters_30 = late_avg_iters * 0.50

    early_time_loss_30 = total_early * avg_time_early * 2.0  # 3x time/iter
    late_new_time_30 = total_late * late_reduced_iters_30 * (late_time_per_iter * 3.0)

    print(f"\nEarly graduates ({total_early} concepts):")
    print(f"  Time lost: ~{early_time_loss_30 / 60:.1f} min (3x slower)")

    print(f"\nLate graduates ({total_late} concepts):")
    print(f"  Estimated with min=30: {late_reduced_iters_30:.1f} iterations (50% reduction)")
    print(f"  With min=30: ~{late_new_time_30 / 60:.1f} min total")
    print(f"  Time {'saved' if late_new_time_30 < late_current_time else 'lost'}: ~{abs(late_new_time_30 - late_current_time) / 60:.1f} min")

    net_change_30 = (early_time_loss_30 + late_new_time_30 - late_current_time) / 60
    print(f"\nNet change with min=30:")
    print(f"  TOTAL: {net_change_30:+.1f} min ({net_change_30/total_time*100:+.1f}%)")

    if net_change_30 < 0:
        print(f"\n✓ min=30 could SAVE ~{-net_change_30:.0f} minutes overall!")
    else:
        print(f"\n✗ min=30 would ADD ~{net_change_30:.0f} minutes overall")

    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    print(f"\nCurrent distribution:")
    print(f"  {total_early/total_concepts*100:.1f}% graduate early (waste time with higher min)")
    print(f"  {total_late/total_concepts*100:.1f}% graduate late (might benefit from higher min)")

    if net_change < -30:  # Save more than 30 minutes
        print(f"\n✓ RECOMMEND: Switch to min=20")
        print(f"  Projected savings: {-net_change:.0f} minutes ({-net_change/total_time*100:.1f}%)")
    elif net_change < 0:
        print(f"\n? BORDERLINE: min=20 might save {-net_change:.0f} minutes")
        print(f"  Consider testing empirically")
    else:
        print(f"\n✓ KEEP: min=10 is optimal")
        print(f"  Too few late graduates to benefit from higher min")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    analyze_full_run()
