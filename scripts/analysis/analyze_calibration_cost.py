#!/usr/bin/env python3
"""
Analyze how many extra samples/iterations are spent due to calibration validation.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

def parse_calibration_impact(log_path: Path):
    """Parse log to see iteration where F1 target met vs iteration where calibration passed."""

    with open(log_path) as f:
        lines = f.readlines()

    results = {
        'concepts': [],
        'extra_iterations_total': 0,
        'extra_samples_total': 0,
    }

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for concept training start
        concept_match = re.match(r'\[(\d+)/\d+\] Training: (.+)', line)
        if concept_match:
            concept_name = concept_match.group(2)

            # Track when concept first graduates on F1/gap criteria
            first_graduate_iter = None
            first_graduate_samples = None

            # Track when it actually finishes (after calibration)
            final_iter = None
            final_samples = None

            # Scan forward through iterations
            j = i + 1
            while j < len(lines):
                iter_line = lines[j]

                # Check for next concept (end of this one)
                if re.match(r'\[(\d+)/\d+\] Training:', iter_line):
                    break

                # Check for graduation on F1 criteria
                iter_match = re.search(r'\[Iter\s+(\d+)\] Activation:\s+(\d+) samples.*✓ GRADUATED', iter_line)
                if iter_match and first_graduate_iter is None:
                    first_graduate_iter = int(iter_match.group(1))
                    first_graduate_samples = int(iter_match.group(2))

                # Check for final completion
                if '✓ Adaptive training complete:' in iter_line:
                    # Look back for the final iteration
                    k = j - 1
                    while k > i:
                        final_iter_match = re.search(r'\[Iter\s+(\d+)\] Activation:\s+(\d+) samples', lines[k])
                        if final_iter_match:
                            final_iter = int(final_iter_match.group(1))
                            final_samples = int(final_iter_match.group(2))
                            break
                        k -= 1
                    break

                j += 1

            # Calculate extra cost
            if first_graduate_iter and final_iter:
                extra_iters = final_iter - first_graduate_iter
                extra_samples = final_samples - first_graduate_samples

                if extra_iters > 0:
                    results['concepts'].append({
                        'name': concept_name,
                        'first_graduate_iter': first_graduate_iter,
                        'first_graduate_samples': first_graduate_samples,
                        'final_iter': final_iter,
                        'final_samples': final_samples,
                        'extra_iterations': extra_iters,
                        'extra_samples': extra_samples,
                    })
                    results['extra_iterations_total'] += extra_iters
                    results['extra_samples_total'] += extra_samples

        i += 1

    return results

def print_analysis(results):
    """Print calibration cost analysis."""

    print("=" * 80)
    print("CALIBRATION VALIDATION COST ANALYSIS")
    print("=" * 80)
    print()

    concepts_with_extra = results['concepts']

    if not concepts_with_extra:
        print("No concepts required extra iterations for calibration validation.")
        return

    print(f"Concepts requiring extra iterations: {len(concepts_with_extra)}")
    print(f"Total extra iterations: {results['extra_iterations_total']}")
    print(f"Total extra samples: {results['extra_samples_total']}")
    print()

    # Sort by extra iterations
    sorted_concepts = sorted(concepts_with_extra, key=lambda x: x['extra_iterations'], reverse=True)

    print("TOP 20 CONCEPTS WITH HIGHEST CALIBRATION COST:")
    print("-" * 80)
    print(f"{'Concept':<40} {'1st Grad':<10} {'Final':<10} {'Extra Iters':<12} {'Extra Samples':<15}")
    print("-" * 80)

    for concept in sorted_concepts[:20]:
        print(f"{concept['name']:<40} "
              f"Iter {concept['first_graduate_iter']:<7} "
              f"Iter {concept['final_iter']:<7} "
              f"+{concept['extra_iterations']:<11} "
              f"+{concept['extra_samples']:<14}")

    if len(sorted_concepts) > 20:
        print(f"... and {len(sorted_concepts) - 20} more")

    print()
    print("STATISTICS:")
    print("-" * 80)

    # Calculate stats
    extra_iters = [c['extra_iterations'] for c in concepts_with_extra]
    extra_samples = [c['extra_samples'] for c in concepts_with_extra]

    print(f"Average extra iterations per concept: {sum(extra_iters) / len(extra_iters):.1f}")
    print(f"Max extra iterations: {max(extra_iters)}")
    print(f"Average extra samples per concept: {sum(extra_samples) / len(extra_samples):.1f}")
    print(f"Max extra samples: {max(extra_samples)}")
    print()

    # Buckets
    print("EXTRA ITERATIONS DISTRIBUTION:")
    print("-" * 80)

    buckets = [
        (1, 1, "1 extra iteration"),
        (2, 2, "2 extra iterations"),
        (3, 3, "3 extra iterations"),
        (4, 5, "4-5 extra iterations"),
        (6, 10, "6-10 extra iterations"),
        (11, 20, "11-20 extra iterations"),
        (21, 100, "21+ extra iterations"),
    ]

    for min_val, max_val, label in buckets:
        count = sum(1 for val in extra_iters if min_val <= val <= max_val)
        if count > 0:
            pct = 100 * count / len(extra_iters)
            print(f"  {label:25s}: {count:4d} ({pct:5.1f}%)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_calibration_cost.py <log_file>")
        return 1

    log_path = Path(sys.argv[1])

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return 1

    print(f"Analyzing: {log_path}")
    print()

    results = parse_calibration_impact(log_path)
    print_analysis(results)

    return 0

if __name__ == '__main__':
    sys.exit(main())
