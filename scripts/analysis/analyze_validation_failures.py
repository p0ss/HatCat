#!/usr/bin/env python3
"""
Analyze correlation between rapid graduation, low synset count, and validation failure.

Hypothesis: Concepts with few synsets overfit quickly to small samples, achieving high
test F1 (A-grade) in few iterations but failing to generalize to lens calibration.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import statistics

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "sumo_classifiers_with_broken_ai_safety"


def analyze_layer(layer: int) -> Dict:
    """Analyze validation failures for a single layer."""
    results_path = RESULTS_DIR / f"layer{layer}" / "results.json"

    if not results_path.exists():
        return None

    with open(results_path) as f:
        data = json.load(f)

    concepts = data["results"]

    # Categorize concepts
    passed = []
    failed = []

    for concept in concepts:
        # Skip if no validation data
        if "validation_passed" not in concept:
            continue

        entry = {
            "concept": concept["concept"],
            "synset_count": concept["synset_count"],
            "iterations": concept.get("total_iterations", concept.get("activation_iterations", 0)),
            "test_f1": concept["test_f1"],
            "validation_passed": concept["validation_passed"],
            "validation_score": concept.get("validation_calibration_score", 0),
            "children_count": concept.get("category_children_count", 0),
        }

        if concept["validation_passed"]:
            passed.append(entry)
        else:
            failed.append(entry)

    # Compute statistics
    def stats(entries: List[Dict], field: str) -> Dict:
        if not entries:
            return {"mean": 0, "median": 0, "min": 0, "max": 0}
        values = [e[field] for e in entries]
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "layer": layer,
        "total": len(concepts),
        "passed": len(passed),
        "failed": len(failed),
        "passed_stats": {
            "synsets": stats(passed, "synset_count"),
            "iterations": stats(passed, "iterations"),
            "test_f1": stats(passed, "test_f1"),
            "val_score": stats(passed, "validation_score"),
        },
        "failed_stats": {
            "synsets": stats(failed, "synset_count"),
            "iterations": stats(failed, "iterations"),
            "test_f1": stats(failed, "test_f1"),
            "val_score": stats(failed, "validation_score"),
        },
        "passed_concepts": passed,
        "failed_concepts": failed,
    }


def print_analysis(analysis: Dict):
    """Print analysis results."""
    layer = analysis["layer"]
    print(f"\n{'='*80}")
    print(f"LAYER {layer} ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal concepts: {analysis['total']}")
    print(f"  Passed validation: {analysis['passed']} ({analysis['passed']/analysis['total']*100:.1f}%)")
    print(f"  Failed validation: {analysis['failed']} ({analysis['failed']/analysis['total']*100:.1f}%)")

    # Compare statistics
    print(f"\n{'METRIC':<20} {'PASSED':<25} {'FAILED':<25} {'DIFFERENCE':<15}")
    print("-" * 85)

    for metric in ["synsets", "iterations", "test_f1", "val_score"]:
        passed_mean = analysis["passed_stats"][metric]["mean"]
        failed_mean = analysis["failed_stats"][metric]["mean"]
        diff = passed_mean - failed_mean

        passed_str = f"{passed_mean:.2f} (med: {analysis['passed_stats'][metric]['median']:.1f})"
        failed_str = f"{failed_mean:.2f} (med: {analysis['failed_stats'][metric]['median']:.1f})"

        if metric == "synsets":
            label = "Synset count"
        elif metric == "iterations":
            label = "Iterations"
        elif metric == "test_f1":
            label = "Test F1"
        else:
            label = "Validation score"

        # Highlight direction
        if metric in ["synsets", "test_f1", "val_score"]:
            # Higher is better for these
            indicator = "✓" if diff > 0 else "✗"
        else:
            # Iterations: neither is clearly better
            indicator = "→"

        print(f"{label:<20} {passed_str:<25} {failed_str:<25} {indicator} {diff:+.2f}")

    # Find patterns
    print(f"\n{'PATTERN ANALYSIS'}")
    print("-" * 80)

    # Low synset + rapid graduation + validation failure
    rapid_graduates_failed = [
        c for c in analysis["failed_concepts"]
        if c["iterations"] <= 3 and c["synset_count"] <= 5
    ]

    rapid_graduates_passed = [
        c for c in analysis["passed_concepts"]
        if c["iterations"] <= 3 and c["synset_count"] <= 5
    ]

    print(f"\nRapid graduates (≤3 iterations) with low synsets (≤5):")
    print(f"  Failed validation: {len(rapid_graduates_failed)}")
    print(f"  Passed validation: {len(rapid_graduates_passed)}")

    if rapid_graduates_failed:
        print(f"\n  Failed rapid graduates with low synsets:")
        for c in sorted(rapid_graduates_failed, key=lambda x: x["synset_count"])[:10]:
            print(f"    {c['concept']:<30} synsets={c['synset_count']:<2} iter={c['iterations']:<2} F1={c['test_f1']:.3f} val={c['validation_score']:.3f}")

    # High test F1 but low validation score
    high_f1_low_val = [
        c for c in analysis["failed_concepts"]
        if c["test_f1"] >= 0.95
    ]

    print(f"\nHigh test F1 (≥0.95) but failed validation:")
    print(f"  Count: {len(high_f1_low_val)}")

    if high_f1_low_val:
        print(f"\n  Top offenders (high test F1, low validation):")
        for c in sorted(high_f1_low_val, key=lambda x: x["test_f1"] - x["validation_score"], reverse=True)[:10]:
            gap = c["test_f1"] - c["validation_score"]
            print(f"    {c['concept']:<30} synsets={c['synset_count']:<2} iter={c['iterations']:<2} F1={c['test_f1']:.3f} val={c['validation_score']:.3f} gap={gap:.3f}")


def main():
    print("="*80)
    print("VALIDATION FAILURE ANALYSIS")
    print("="*80)
    print("\nHypothesis: Low synset count → rapid overfitting → high test F1 but low validation")

    all_analyses = []

    for layer in range(6):
        analysis = analyze_layer(layer)
        if analysis:
            all_analyses.append(analysis)
            print_analysis(analysis)

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    total_passed = sum(a["passed"] for a in all_analyses)
    total_failed = sum(a["failed"] for a in all_analyses)
    total_concepts = total_passed + total_failed

    print(f"\nAcross all layers:")
    print(f"  Total concepts: {total_concepts}")
    print(f"  Passed: {total_passed} ({total_passed/total_concepts*100:.1f}%)")
    print(f"  Failed: {total_failed} ({total_failed/total_concepts*100:.1f}%)")

    # Aggregate statistics
    all_passed = []
    all_failed = []

    for a in all_analyses:
        all_passed.extend(a["passed_concepts"])
        all_failed.extend(a["failed_concepts"])

    def aggregate_stats(concepts: List[Dict]) -> Dict:
        if not concepts:
            return {}
        return {
            "synsets_mean": statistics.mean(c["synset_count"] for c in concepts),
            "iterations_mean": statistics.mean(c["iterations"] for c in concepts),
            "test_f1_mean": statistics.mean(c["test_f1"] for c in concepts),
            "val_score_mean": statistics.mean(c["validation_score"] for c in concepts),
        }

    passed_agg = aggregate_stats(all_passed)
    failed_agg = aggregate_stats(all_failed)

    print(f"\nAggregate means:")
    print(f"{'METRIC':<20} {'PASSED':<15} {'FAILED':<15} {'DIFFERENCE':<15}")
    print("-" * 65)
    print(f"{'Synset count':<20} {passed_agg['synsets_mean']:<15.2f} {failed_agg['synsets_mean']:<15.2f} {passed_agg['synsets_mean']-failed_agg['synsets_mean']:+.2f}")
    print(f"{'Iterations':<20} {passed_agg['iterations_mean']:<15.2f} {failed_agg['iterations_mean']:<15.2f} {passed_agg['iterations_mean']-failed_agg['iterations_mean']:+.2f}")
    print(f"{'Test F1':<20} {passed_agg['test_f1_mean']:<15.3f} {failed_agg['test_f1_mean']:<15.3f} {passed_agg['test_f1_mean']-failed_agg['test_f1_mean']:+.3f}")
    print(f"{'Validation score':<20} {passed_agg['val_score_mean']:<15.3f} {failed_agg['val_score_mean']:<15.3f} {passed_agg['val_score_mean']-failed_agg['val_score_mean']:+.3f}")

    # Key finding
    print(f"\n{'KEY FINDINGS'}")
    print("-" * 80)

    if failed_agg['synsets_mean'] < passed_agg['synsets_mean']:
        synset_diff = passed_agg['synsets_mean'] - failed_agg['synsets_mean']
        print(f"✓ Failed concepts have {synset_diff:.1f} fewer synsets on average")

    if failed_agg['iterations_mean'] < passed_agg['iterations_mean']:
        iter_diff = passed_agg['iterations_mean'] - failed_agg['iterations_mean']
        print(f"✓ Failed concepts graduate {iter_diff:.1f} iterations faster (potential overfitting)")

    if abs(failed_agg['test_f1_mean'] - passed_agg['test_f1_mean']) < 0.02:
        print(f"✓ Test F1 is similar between passed/failed (both ~{failed_agg['test_f1_mean']:.3f})")
        print(f"  → Test F1 alone is NOT predictive of validation success!")

    gap_failed = failed_agg['test_f1_mean'] - failed_agg['val_score_mean']
    gap_passed = passed_agg['test_f1_mean'] - passed_agg['val_score_mean']

    if gap_failed > gap_passed:
        print(f"✓ Failed concepts show larger generalization gap:")
        print(f"  Failed: test F1 - val score = {gap_failed:.3f}")
        print(f"  Passed: test F1 - val score = {gap_passed:.3f}")
        print(f"  → Failed concepts overfit to test set!")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
