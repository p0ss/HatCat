#!/usr/bin/env python3
"""
Analyze training log to understand iteration distribution and failures.
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

def parse_training_log(log_path: Path):
    """Parse training log and extract iteration counts per concept."""

    with open(log_path) as f:
        content = f.read()

    # Pattern to match concept training sections
    # Look for "Training: ConceptName" followed by iterations
    concept_pattern = r'\[\d+/\d+\] Training: (.+?)\n.*?(?:✓ Adaptive training complete|⚠️  Activation lens did not graduate)'

    results = {
        'graduated': [],
        'failed': [],
        'by_layer': defaultdict(lambda: {'graduated': [], 'failed': []}),
    }

    # Track current layer
    current_layer = None
    layer_pattern = r'TRAINING LAYER (\d+)'

    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Track layer transitions
        layer_match = re.search(layer_pattern, line)
        if layer_match:
            current_layer = int(layer_match.group(1))

        # Look for concept training start
        concept_match = re.match(r'\[(\d+)/(\d+)\] Training: (.+)', line)
        if concept_match:
            concept_num = concept_match.group(1)
            total = concept_match.group(2)
            concept_name = concept_match.group(3)

            # Scan forward to find iterations
            iterations = []
            graduated = False

            j = i + 1
            while j < len(lines):
                iter_line = lines[j]

                # Check for iteration
                iter_match = re.search(r'\[Iter\s+(\d+)\] Activation:', iter_line)
                if iter_match:
                    iterations.append(int(iter_match.group(1)))

                # Check for graduation
                if '✓ Adaptive training complete' in iter_line:
                    graduated = True
                    break

                # Check for failure
                if '⚠️  Activation lens did not graduate' in iter_line:
                    graduated = False
                    break

                # Stop at next concept
                if re.match(r'\[\d+/\d+\] Training:', iter_line):
                    break

                j += 1

            if iterations:
                max_iter = max(iterations)
                entry = {
                    'concept': concept_name,
                    'iterations': max_iter,
                    'layer': current_layer,
                }

                if graduated:
                    results['graduated'].append(entry)
                    if current_layer is not None:
                        results['by_layer'][current_layer]['graduated'].append(entry)
                else:
                    results['failed'].append(entry)
                    if current_layer is not None:
                        results['by_layer'][current_layer]['failed'].append(entry)

        i += 1

    return results

def print_statistics(results):
    """Print training statistics."""

    print("=" * 80)
    print("TRAINING LOG ANALYSIS")
    print("=" * 80)
    print()

    # Overall statistics
    total_graduated = len(results['graduated'])
    total_failed = len(results['failed'])
    total = total_graduated + total_failed

    print(f"Total concepts: {total}")
    print(f"  Graduated: {total_graduated} ({100*total_graduated/total:.1f}%)")
    print(f"  Failed: {total_failed} ({100*total_failed/total:.1f}%)")
    print()

    # Iteration buckets for graduated
    if results['graduated']:
        print("GRADUATED CONCEPTS - Iteration Distribution")
        print("-" * 80)

        iter_counts = [entry['iterations'] for entry in results['graduated']]

        buckets = [
            (3, 3, "3 iterations (minimum)"),
            (4, 5, "4-5 iterations"),
            (6, 7, "6-7 iterations"),
            (8, 10, "8-10 iterations"),
            (11, 15, "11-15 iterations"),
            (16, 20, "16-20 iterations"),
            (21, 30, "21-30 iterations"),
            (31, 50, "31-50 iterations"),
        ]

        for min_iter, max_iter, label in buckets:
            count = sum(1 for i in iter_counts if min_iter <= i <= max_iter)
            if count > 0:
                pct = 100 * count / len(iter_counts)
                print(f"  {label:30s}: {count:4d} ({pct:5.1f}%)")

        print()
        print(f"  Average iterations: {sum(iter_counts) / len(iter_counts):.1f}")
        print(f"  Median iterations: {sorted(iter_counts)[len(iter_counts)//2]}")
        print()

    # Failed concepts
    if results['failed']:
        print("FAILED CONCEPTS")
        print("-" * 80)
        print(f"Total failed: {len(results['failed'])}")
        print()

        # Group by layer
        by_layer = defaultdict(list)
        for entry in results['failed']:
            by_layer[entry['layer']].append(entry['concept'])

        for layer in sorted(by_layer.keys()):
            concepts = by_layer[layer]
            print(f"Layer {layer}: {len(concepts)} failed")
            for concept in concepts[:10]:  # Show first 10
                print(f"  - {concept}")
            if len(concepts) > 10:
                print(f"  ... and {len(concepts) - 10} more")
            print()

    # Per-layer breakdown
    print("PER-LAYER STATISTICS")
    print("-" * 80)

    for layer in sorted(results['by_layer'].keys()):
        layer_data = results['by_layer'][layer]
        grad = len(layer_data['graduated'])
        fail = len(layer_data['failed'])
        total = grad + fail

        if total == 0:
            continue

        print(f"Layer {layer}:")
        print(f"  Total: {total}, Graduated: {grad} ({100*grad/total:.1f}%), Failed: {fail} ({100*fail/total:.1f}%)")

        if layer_data['graduated']:
            iters = [e['iterations'] for e in layer_data['graduated']]
            print(f"  Avg iterations (graduated): {sum(iters)/len(iters):.1f}")

        print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_training_log.py <log_file>")
        print("\nExample: python analyze_training_log.py overnight_training_20251114_004130.log")
        return 1

    log_path = Path(sys.argv[1])

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return 1

    print(f"Analyzing: {log_path}")
    print()

    results = parse_training_log(log_path)
    print_statistics(results)

    return 0

if __name__ == '__main__':
    sys.exit(main())
