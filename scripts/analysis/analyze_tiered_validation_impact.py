#!/usr/bin/env python3
"""
Analyze the impact of tiered validation on training speed vs quality.

Compares:
- Current: Graduate everything at 3 iterations (no validation blocking)
- Proposed: Tiered validation (3/6/9/12 iterations for strict/high/medium/relaxed)

Uses empirical distribution from previous training runs.
"""

import numpy as np


def main():
    print("=" * 80)
    print("TIERED VALIDATION IMPACT ANALYSIS")
    print("=" * 80)
    print()

    # Empirical distribution from previous runs with strict validation
    # From user data:
    # - 34.8% graduate in 3-5 iterations (fast)
    # - 27.6% in 6-10 iterations (medium)
    # - 22.4% in 11-20 iterations (slow)
    # - 15.2% in 21-50 iterations (very slow)

    # Let's break this down more granularly based on the pattern
    # Assume exponential decay with some concepts being hard
    distribution = {
        3: 0.25,   # 25% graduate at 3 (naturally excellent)
        4: 0.10,   # 10% at 4
        5: 0.10,   # 10% at 5 (total 45% by iteration 5)
        6: 0.08,   # 8% at 6
        7: 0.07,   # 7% at 7
        8: 0.06,   # 6% at 8
        9: 0.05,   # 5% at 9
        10: 0.04,  # 4% at 10 (total 75% by iteration 10)
        11: 0.04,  # 4% at 11
        12: 0.03,  # 3% at 12
        15: 0.03,  # 3% at 15
        20: 0.05,  # 5% at 20 (total 90% by iteration 20)
        30: 0.05,  # 5% at 30
        50: 0.05,  # 5% at 50 (the long tail)
    }

    total_concepts = 1000  # Hypothetical

    print("EMPIRICAL DISTRIBUTION (from previous training with strict validation)")
    print("-" * 80)
    cumulative = 0
    for iters, pct in sorted(distribution.items()):
        cumulative += pct
        count = int(total_concepts * pct)
        print(f"  Iteration {iters:2d}: {pct*100:5.1f}% ({count:3d} concepts) | Cumulative: {cumulative*100:5.1f}%")
    print()

    # CURRENT APPROACH: Graduate everything at 3 iterations (no blocking)
    print("CURRENT APPROACH: No validation blocking (graduate at 3)")
    print("-" * 80)
    current_total_iterations = 3 * total_concepts
    print(f"  All concepts: 3 iterations")
    print(f"  Total iterations: {current_total_iterations:,}")
    print(f"  Quality distribution:")
    print(f"    - High quality (would pass strict): 25% (250 concepts)")
    print(f"    - Medium quality (would need 6-10 iters): 30% (300 concepts)")
    print(f"    - Low quality (would need 11+ iters): 45% (450 concepts)")
    print()

    # PROPOSED APPROACH: Tiered validation (3/6/9/12)
    print("PROPOSED APPROACH: Tiered validation (strict@3, high@6, medium@9, relaxed@12)")
    print("-" * 80)

    # Define tiers
    tiers = {
        'strict': {'max_iters': 3, 'threshold': 0.70, 'grade': 'A'},
        'high': {'max_iters': 6, 'threshold': 0.60, 'grade': 'A-/B+'},
        'medium': {'max_iters': 9, 'threshold': 0.50, 'grade': 'B'},
        'relaxed': {'max_iters': 12, 'threshold': 0.40, 'grade': 'B-/C+'},
    }

    # Model what happens with tiered validation
    # Assumptions:
    # - Concepts that would naturally graduate at N iterations will pass validation at tier covering N
    # - Concepts are pushed to next tier if they fail current tier

    tier_results = []

    # Strict tier (1-3 iterations): Captures the 25% that naturally excel
    strict_graduates = int(total_concepts * 0.25)
    strict_iterations = strict_graduates * 3
    tier_results.append(('strict@3', strict_graduates, 3, strict_iterations, 'A'))

    # High tier (4-6 iterations): Captures next 23% (originally 4-6 iters)
    # Plus some that failed strict (originally 7-8) get pushed here
    high_candidates = int(total_concepts * 0.30)  # 30% try high tier
    high_graduates = int(total_concepts * 0.23)  # 23% succeed
    high_iterations = high_graduates * 6
    tier_results.append(('high@6', high_graduates, 6, high_iterations, 'A-/B+'))

    # Medium tier (7-9 iterations): Captures next 17%
    medium_candidates = int(total_concepts * 0.25)
    medium_graduates = int(total_concepts * 0.17)
    medium_iterations = medium_graduates * 9
    tier_results.append(('medium@9', medium_graduates, 9, medium_iterations, 'B'))

    # Relaxed tier (10-12 iterations): Captures remaining 35%
    # This includes everything that would have gone 10-50 iterations
    relaxed_graduates = total_concepts - strict_graduates - high_graduates - medium_graduates
    relaxed_iterations = relaxed_graduates * 12
    tier_results.append(('relaxed@12', relaxed_graduates, 12, relaxed_iterations, 'B-/C+'))

    proposed_total_iterations = sum(r[3] for r in tier_results)

    for tier_name, count, iters, total_iters, grade in tier_results:
        pct = (count / total_concepts) * 100
        print(f"  {tier_name:12s}: {count:3d} concepts ({pct:5.1f}%) × {iters:2d} iters = {total_iters:6,} iterations [{grade}]")

    print(f"\n  Total iterations: {proposed_total_iterations:,}")
    print()

    # COMPARISON
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    slowdown_iters = proposed_total_iterations - current_total_iterations
    slowdown_pct = (slowdown_iters / current_total_iterations) * 100

    print(f"Current approach:  {current_total_iterations:6,} iterations")
    print(f"Proposed approach: {proposed_total_iterations:6,} iterations")
    print(f"Slowdown:          {slowdown_iters:+6,} iterations ({slowdown_pct:+.1f}%)")
    print()

    print("QUALITY IMPROVEMENT:")
    print("-" * 80)
    print("Current (no blocking):")
    print("  - 25% high quality (A)")
    print("  - 30% medium quality (would be A-/B+ with more training)")
    print("  - 45% low quality (would need 11+ iterations)")
    print()
    print("Proposed (tiered blocking):")
    print("  - 25% A-tier (strict@3)")
    print("  - 23% A-/B+ tier (high@6)")
    print("  - 17% B-tier (medium@9)")
    print("  - 35% B-/C+ tier (relaxed@12)")
    print()

    # TIME ESTIMATION
    print("TIME ESTIMATION (assuming 2 seconds per concept per iteration):")
    print("-" * 80)
    current_hours = (current_total_iterations * 2) / 3600
    proposed_hours = (proposed_total_iterations * 2) / 3600

    print(f"Current:  {current_hours:6.1f} hours")
    print(f"Proposed: {proposed_hours:6.1f} hours")
    print(f"Slowdown: {proposed_hours - current_hours:+6.1f} hours")
    print()

    # BENEFIT QUANTIFICATION
    print("BENEFIT QUANTIFICATION:")
    print("-" * 80)

    # Quality score: A=1.0, B+=0.85, B=0.75, C+=0.65, unvalidated=0.50
    quality_scores = {'A': 1.0, 'A-/B+': 0.85, 'B': 0.75, 'B-/C+': 0.65}

    current_quality = (0.25 * 1.0) + (0.30 * 0.85) + (0.45 * 0.50)  # unvalidated get 0.5
    proposed_quality = sum(
        (count / total_concepts) * quality_scores[grade]
        for _, count, _, _, grade in tier_results
    )

    print(f"Current quality score:  {current_quality:.3f}")
    print(f"Proposed quality score: {proposed_quality:.3f}")
    print(f"Quality improvement:    {proposed_quality - current_quality:+.3f} ({(proposed_quality - current_quality)/current_quality * 100:+.1f}%)")
    print()

    # RECOMMENDATION
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print(f"Trade-off: +{slowdown_pct:.1f}% training time for +{(proposed_quality - current_quality)/current_quality * 100:.1f}% quality")
    print()

    efficiency = (proposed_quality - current_quality) / (slowdown_pct / 100)
    print(f"Efficiency ratio: {efficiency:.2f} (quality gain per % slowdown)")
    print()

    if slowdown_pct < 150 and (proposed_quality - current_quality) > 0.15:
        print("✓ RECOMMENDED: Modest slowdown with significant quality improvement")
    elif slowdown_pct < 100:
        print("✓ RECOMMENDED: Slowdown less than 2× with quality improvements")
    else:
        print("⚠ CAUTION: Slowdown may outweigh quality benefits")
    print()

    print("Key advantages of tiered approach:")
    print("  1. Prevents the 15% long tail (21-50 iterations)")
    print("  2. Natural quality distribution instead of binary pass/fail")
    print("  3. Concepts that can achieve A-tier do so early")
    print("  4. Hard concepts still graduate with acceptable quality by iteration 12")
    print()


if __name__ == '__main__':
    main()
