#!/usr/bin/env python3
"""
Quick test of tiered validation thresholds.

Tests the new progressive validation logic:
- Iterations 1-5: Strict (push for A-tier, score >= 0.70)
- Iterations 6-10: Moderate (accept B-tier, score >= 0.50)
- Iterations 11+: Relaxed (accept C-tier, score >= 0.30)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.map.training.dual_adaptive_trainer import DualAdaptiveTrainer
import torch
import numpy as np


def test_tier_logic():
    """Test the tier selection logic."""
    trainer = DualAdaptiveTrainer(
        validation_tier1_iterations=5,
        validation_tier2_iterations=10,
    )

    print("Testing tier selection logic:")
    print("=" * 60)

    test_iterations = [1, 3, 5, 6, 8, 10, 11, 15, 20]
    for iter_num in test_iterations:
        tier_info = trainer._get_validation_tier(iter_num)
        print(f"Iteration {iter_num:2d}: {tier_info['tier']:8s} tier "
              f"(min_score={tier_info['min_score']:.2f}, "
              f"target_grade={tier_info['target_grade']}, "
              f"max_rank={tier_info['max_target_rank']})")

    print()


def simulate_validation_scenarios():
    """Simulate different validation scenarios."""
    trainer = DualAdaptiveTrainer(
        validation_tier1_iterations=5,
        validation_tier2_iterations=10,
        validation_blocking=True,  # Enable blocking for demonstration
    )

    print("Simulating validation scenarios:")
    print("=" * 60)

    # Scenario 1: B-tier score at different iterations
    score = 0.45  # B-tier (between 0.2 and 0.5)

    for iteration in [3, 7, 12]:
        tier_info = trainer._get_validation_tier(iteration)

        # Mock validation results
        target_rank = 4
        avg_other_rank = 8.0

        passed = (target_rank <= tier_info['max_target_rank']) and \
                (avg_other_rank >= tier_info['min_other_rank']) and \
                (score >= tier_info['min_score'])

        # Assign grade
        if score >= 0.5:
            grade = 'A'
        elif score >= 0.2:
            grade = 'B'
        else:
            grade = 'C'

        print(f"\nIteration {iteration} ({tier_info['tier']} tier):")
        print(f"  Score: {score:.2f}, Grade: {grade}")
        print(f"  Target rank: #{target_rank}, Others: {avg_other_rank:.1f}")
        print(f"  Thresholds: score>={tier_info['min_score']:.2f}, "
              f"target_rank<={tier_info['max_target_rank']}, "
              f"other_rank>={tier_info['min_other_rank']:.1f}")
        print(f"  Result: {'✓ PASS' if passed else '✗ FAIL (continue training)'}")

    print()
    print("Expected behavior:")
    print("  - Iteration 3 (strict): FAIL → continue training for A-tier")
    print("  - Iteration 7 (moderate): PASS → B-tier acceptable")
    print("  - Iteration 12 (relaxed): PASS → B-tier more than acceptable")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("TIERED VALIDATION TEST")
    print("=" * 60)
    print()

    test_tier_logic()
    simulate_validation_scenarios()

    print("=" * 60)
    print("Test complete!")
    print()
    print("Key insights:")
    print("  1. Early iterations (1-5) push for A-tier quality")
    print("  2. Middle iterations (6-10) accept B-tier to avoid diminishing returns")
    print("  3. Late iterations (11+) accept C-tier to prevent long tail blowout")
    print("  4. This balances quality vs training speed efficiently")
    print("=" * 60)
