#!/usr/bin/env python3
"""
More realistic analysis: concepts don't jump to tier max, they graduate when they pass.

Key insight: With tiered validation, a concept doesn't automatically train to 6 iterations.
It trains to 3, checks validation. If it fails but is close, it goes to 4, checks again, etc.

This is MUCH faster than the worst-case tier-max assumption.
"""

import numpy as np


def main():
    print("=" * 80)
    print("TIERED VALIDATION IMPACT ANALYSIS (Realistic)")
    print("=" * 80)
    print()

    # Empirical distribution: iteration at which concepts would pass STRICT validation
    # From user data (scaled to match their percentages)
    strict_pass_distribution = {
        3: 250,   # 25%
        4: 100,   # 10%
        5: 100,   # 10%
        6: 80,    # 8%
        7: 70,    # 7%
        8: 60,    # 6%
        9: 50,    # 5%
        10: 40,   # 4%
        11: 40,   # 4%
        12: 30,   # 3%
        15: 30,   # 3%
        20: 50,   # 5%
        30: 50,   # 5%
        50: 50,   # 5%
    }

    total_concepts = sum(strict_pass_distribution.values())

    print("SETUP")
    print("-" * 80)
    print(f"Total concepts: {total_concepts}")
    print()

    # CURRENT APPROACH: Graduate all at 3 (no validation blocking)
    print("CURRENT APPROACH: No validation blocking")
    print("-" * 80)
    current_total_iters = 3 * total_concepts
    current_time_hours = (current_total_iters * 2) / 3600

    print(f"  All concepts graduate at iteration 3")
    print(f"  Total iterations: {current_total_iters:,}")
    print(f"  Estimated time: {current_time_hours:.1f} hours")
    print()

    # PROPOSED APPROACH: Tiered validation with realistic iteration counts
    print("PROPOSED APPROACH: Tiered validation (3/6/9/12)")
    print("-" * 80)
    print()

    # Define when concepts pass each tier based on calibration score thresholds
    # Tier thresholds map to approximate iteration requirements
    tiers = [
        {'name': 'strict', 'max_iter': 3, 'score_threshold': 0.70, 'pass_at_iters': [3]},
        {'name': 'high', 'max_iter': 6, 'score_threshold': 0.60, 'pass_at_iters': [4, 5, 6]},
        {'name': 'medium', 'max_iter': 9, 'score_threshold': 0.50, 'pass_at_iters': [7, 8, 9]},
        {'name': 'relaxed', 'max_iter': 12, 'score_threshold': 0.40, 'pass_at_iters': list(range(10, 51))},
    ]

    # Map each iteration count to which tier it would pass
    def get_graduation_tier(strict_iter):
        """Determine which tier and iteration a concept graduates at."""
        if strict_iter <= 3:
            return ('strict', 3)
        elif strict_iter <= 6:
            return ('high', min(strict_iter, 6))
        elif strict_iter <= 9:
            return ('medium', min(strict_iter, 9))
        else:
            return ('relaxed', min(strict_iter, 12))

    # Calculate actual iterations per concept
    tier_stats = {
        'strict': {'concepts': [], 'iters': []},
        'high': {'concepts': [], 'iters': []},
        'medium': {'concepts': [], 'iters': []},
        'relaxed': {'concepts': [], 'iters': []},
    }

    for strict_iter, count in strict_pass_distribution.items():
        tier_name, actual_iter = get_graduation_tier(strict_iter)
        tier_stats[tier_name]['concepts'].append(count)
        tier_stats[tier_name]['iters'].append(actual_iter * count)

    # Summarize
    proposed_total_iters = 0
    print("Graduation breakdown:")
    for tier_name, stats in tier_stats.items():
        total_concepts_in_tier = sum(stats['concepts'])
        total_iters_in_tier = sum(stats['iters'])
        if total_concepts_in_tier > 0:
            avg_iter = total_iters_in_tier / total_concepts_in_tier
            pct = (total_concepts_in_tier / total_concepts) * 100
            print(f"  {tier_name:8s}: {total_concepts_in_tier:3d} concepts ({pct:5.1f}%) "
                  f"× {avg_iter:.1f} avg iters = {total_iters_in_tier:6,} total iterations")
            proposed_total_iters += total_iters_in_tier

    proposed_time_hours = (proposed_total_iters * 2) / 3600

    print()
    print(f"  Total iterations: {proposed_total_iters:,}")
    print(f"  Estimated time: {proposed_time_hours:.1f} hours")
    print()

    # COMPARISON
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    slowdown_iters = proposed_total_iters - current_total_iters
    slowdown_pct = (slowdown_iters / current_total_iters) * 100
    slowdown_hours = proposed_time_hours - current_time_hours

    print(f"Current:   {current_total_iters:6,} iterations ({current_time_hours:5.1f} hours)")
    print(f"Proposed:  {proposed_total_iters:6,} iterations ({proposed_time_hours:5.1f} hours)")
    print(f"Slowdown:  {slowdown_iters:+6,} iterations ({slowdown_pct:+6.1f}%) | {slowdown_hours:+5.1f} hours")
    print()

    # QUALITY IMPROVEMENT
    print("QUALITY IMPROVEMENT")
    print("-" * 80)

    # Grade distribution
    # Current: everything gets the quality it would naturally have at iteration 3
    # Proposed: concepts train until they pass their tier

    def quality_score_at_iteration(target_iter, actual_iter):
        """Estimate quality score based on how close to target iteration."""
        if actual_iter >= target_iter:
            return 1.0  # Full quality
        else:
            # Diminishing quality the earlier we stop
            return 0.5 + 0.5 * (actual_iter / target_iter)

    current_quality = 0
    proposed_quality = 0

    for strict_iter, count in strict_pass_distribution.items():
        # Current: all stop at 3
        current_quality += count * quality_score_at_iteration(strict_iter, 3)

        # Proposed: train until passing tier
        tier_name, actual_iter = get_graduation_tier(strict_iter)
        proposed_quality += count * quality_score_at_iteration(strict_iter, actual_iter)

    current_quality /= total_concepts
    proposed_quality /= total_concepts

    quality_improvement = proposed_quality - current_quality
    quality_improvement_pct = (quality_improvement / current_quality) * 100

    print(f"Current quality score:  {current_quality:.3f}")
    print(f"Proposed quality score: {proposed_quality:.3f}")
    print(f"Quality improvement:    {quality_improvement:+.3f} ({quality_improvement_pct:+.1f}%)")
    print()

    # EFFICIENCY
    efficiency = quality_improvement_pct / slowdown_pct if slowdown_pct > 0 else 0
    print(f"Efficiency: {efficiency:.3f} (quality % gain per % slowdown)")
    print()

    # RECOMMENDATION
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    print(f"📊 Trade-off: {slowdown_pct:+.0f}% training time for {quality_improvement_pct:+.1f}% quality")
    print()

    if slowdown_pct < 100 and quality_improvement_pct > 15:
        print("✅ STRONGLY RECOMMENDED: Modest slowdown (<2×) with significant quality gain (>15%)")
    elif slowdown_pct < 150 and quality_improvement_pct > 10:
        print("✓ RECOMMENDED: Acceptable slowdown for measurable quality improvement")
    elif slowdown_pct < 200:
        print("⚠ MODERATE: Consider if quality is priority over speed")
    else:
        print("❌ NOT RECOMMENDED: Excessive slowdown relative to quality gain")

    print()
    print("Key benefits:")
    print(f"  • Prevents {sum(strict_pass_distribution.get(i, 0) for i in [20, 30, 50])} concepts from needing 20-50 iterations")
    print(f"  • Caps worst case at 12 iterations (vs 50)")
    print(f"  • {tier_stats['strict']['concepts'][0] if tier_stats['strict']['concepts'] else 0} concepts graduate at 3 (no slowdown)")
    print(f"  • Natural quality stratification (A/B+/B/C+ tiers)")
    print()


if __name__ == '__main__':
    main()
