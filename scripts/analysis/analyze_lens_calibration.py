#!/usr/bin/env python3
"""
Analyze Lens Calibration Results

Generates visualizations and detailed reports from calibration test data.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_calibration_results(results_file: Path) -> Dict:
    """Load calibration results from JSON."""
    with open(results_file) as f:
        return json.load(f)


def plot_category_distribution(summary: Dict, output_dir: Path):
    """Plot pie chart of lens categories."""
    categories = summary['categories']

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {
        'well_calibrated': '#4CAF50',
        'marginal': '#FFC107',
        'over_firing': '#FF5722',
        'under_firing': '#2196F3',
        'broken': '#9E9E9E'
    }

    labels = []
    sizes = []
    plot_colors = []

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        labels.append(f"{cat} ({count})")
        sizes.append(count)
        plot_colors.append(colors.get(cat, '#CCCCCC'))

    ax.pie(sizes, labels=labels, colors=plot_colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Lens Calibration Categories\n{summary['lens_pack']}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'category_distribution.png', dpi=150)
    print(f"✓ Saved category_distribution.png")
    plt.close()


def plot_fp_tp_scatter(results: Dict, output_dir: Path):
    """Scatter plot of FP rate vs TP rate."""
    fp_rates = []
    tp_rates = []
    categories = []
    names = []

    for name, data in results.items():
        fp_rates.append(data['metrics']['fp_rate'])
        tp_rates.append(data['metrics']['tp_rate'])
        categories.append(data['category'])
        names.append(name)

    fig, ax = plt.subplots(figsize=(12, 10))

    category_colors = {
        'well_calibrated': '#4CAF50',
        'marginal': '#FFC107',
        'over_firing': '#FF5722',
        'under_firing': '#2196F3',
        'broken': '#9E9E9E'
    }

    for cat in set(categories):
        mask = [c == cat for c in categories]
        fp_cat = [fp for fp, m in zip(fp_rates, mask) if m]
        tp_cat = [tp for tp, m in zip(tp_rates, mask) if m]
        ax.scatter(fp_cat, tp_cat, label=cat, alpha=0.6, s=50, color=category_colors.get(cat, '#CCCCCC'))

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Lens Performance: FP Rate vs TP Rate', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add ideal region
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, label='High TP threshold')
    ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.3, label='Low FP threshold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fp_tp_scatter.png', dpi=150)
    print(f"✓ Saved fp_tp_scatter.png")
    plt.close()


def plot_f1_histogram(results: Dict, output_dir: Path):
    """Histogram of F1 scores by category."""
    category_f1 = {}

    for name, data in results.items():
        cat = data['category']
        f1 = data['metrics']['f1']

        if cat not in category_f1:
            category_f1[cat] = []
        category_f1[cat].append(f1)

    fig, ax = plt.subplots(figsize=(12, 6))

    bins = np.linspace(0, 1, 21)
    category_colors = {
        'well_calibrated': '#4CAF50',
        'marginal': '#FFC107',
        'over_firing': '#FF5722',
        'under_firing': '#2196F3',
        'broken': '#9E9E9E'
    }

    for cat, f1_scores in sorted(category_f1.items()):
        ax.hist(f1_scores, bins=bins, alpha=0.6, label=cat, color=category_colors.get(cat, '#CCCCCC'))

    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_ylabel('Number of Lenses', fontsize=12)
    ax.set_title('F1 Score Distribution by Category', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'f1_histogram.png', dpi=150)
    print(f"✓ Saved f1_histogram.png")
    plt.close()


def plot_score_distributions(results: Dict, output_dir: Path, sample_size: int = 20):
    """Box plots of score distributions (positive, negative, irrelevant)."""
    # Sample random lenses for visualization
    sampled_names = list(results.keys())[:sample_size]

    positive_scores = []
    negative_scores = []
    irrelevant_scores = []
    labels = []

    for name in sampled_names:
        data = results[name]
        positive_scores.append(data['metrics']['avg_positive_score'])
        negative_scores.append(data['metrics']['avg_negative_score'])
        irrelevant_scores.append(data['metrics']['avg_irrelevant_score'])
        labels.append(name.split('_')[0][:15])  # Truncate long names

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Positive scores
    axes[0].barh(labels, positive_scores, color='green', alpha=0.6)
    axes[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Average Probability')
    axes[0].set_title('Positive Sample Scores (should be high)', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Negative scores
    axes[1].barh(labels, negative_scores, color='orange', alpha=0.6)
    axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Average Probability')
    axes[1].set_title('Negative Sample Scores (should be low)', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    # Irrelevant scores
    axes[2].barh(labels, irrelevant_scores, color='blue', alpha=0.6)
    axes[2].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Average Probability')
    axes[2].set_title('Irrelevant Sample Scores (should be low)', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'score_distributions.png', dpi=150)
    print(f"✓ Saved score_distributions.png")
    plt.close()


def generate_detailed_report(summary: Dict, output_dir: Path):
    """Generate detailed text report."""
    report_file = output_dir / 'calibration_report.txt'

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LENS PACK CALIBRATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Lens Pack: {summary['lens_pack']}\n")
        f.write(f"Model: {summary['model']}\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Total Lenses: {summary['total_lenses']}\n\n")

        f.write("="*80 + "\n")
        f.write("CATEGORY BREAKDOWN\n")
        f.write("="*80 + "\n\n")

        for cat, count in sorted(summary['categories'].items(), key=lambda x: -x[1]):
            pct = 100 * count / summary['total_lenses']
            f.write(f"{cat:20s}: {count:5d} lenses ({pct:5.1f}%)\n")

        # Well-calibrated lenses
        f.write("\n" + "="*80 + "\n")
        f.write("WELL-CALIBRATED LENSS (Top 20)\n")
        f.write("="*80 + "\n\n")

        well_calibrated = [
            (name, data['metrics']['f1'])
            for name, data in summary['results'].items()
            if data['category'] == 'well_calibrated'
        ]
        well_calibrated.sort(key=lambda x: -x[1])

        for name, f1 in well_calibrated[:20]:
            data = summary['results'][name]
            m = data['metrics']
            f.write(f"{name:40s}: F1={f1:.3f}, TP={m['tp_rate']:.3f}, FP={m['fp_rate']:.3f}\n")

        # Over-firing lenses
        f.write("\n" + "="*80 + "\n")
        f.write("OVER-FIRING LENSS (Top 20 by FP Rate)\n")
        f.write("="*80 + "\n\n")

        over_firing = sorted(
            [(name, data['metrics']['fp_rate'], data['metrics'])
             for name, data in summary['results'].items()],
            key=lambda x: -x[1]
        )[:20]

        for name, fp_rate, m in over_firing:
            f.write(f"{name:40s}: FP={fp_rate:.3f}, TP={m['tp_rate']:.3f}, "
                   f"F1={m['f1']:.3f}\n")

        # Under-firing lenses
        f.write("\n" + "="*80 + "\n")
        f.write("UNDER-FIRING LENSS (Top 20 by Low TP Rate)\n")
        f.write("="*80 + "\n\n")

        under_firing = sorted(
            [(name, data['metrics']['tp_rate'], data['metrics'])
             for name, data in summary['results'].items()],
            key=lambda x: x[1]
        )[:20]

        for name, tp_rate, m in under_firing:
            f.write(f"{name:40s}: TP={tp_rate:.3f}, FP={m['fp_rate']:.3f}, "
                   f"F1={m['f1']:.3f}\n")

        # Broken lenses
        f.write("\n" + "="*80 + "\n")
        f.write("BROKEN LENSS (F1 < 0.3)\n")
        f.write("="*80 + "\n\n")

        broken = [
            (name, data['metrics'])
            for name, data in summary['results'].items()
            if data['category'] == 'broken'
        ]

        for name, m in broken:
            f.write(f"{name:40s}: F1={m['f1']:.3f}, TP={m['tp_rate']:.3f}, "
                   f"FP={m['fp_rate']:.3f}\n")

    print(f"✓ Saved calibration_report.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze lens calibration results"
    )
    parser.add_argument('results_file', type=Path,
                       help='Calibration results JSON file')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: same as results file)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_file.parent

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("LENS CALIBRATION ANALYSIS")
    print("="*80)
    print(f"Results file: {args.results_file}")
    print(f"Output dir: {args.output_dir}")
    print("="*80)

    # Load results
    print("\nLoading results...")
    summary = load_calibration_results(args.results_file)
    print(f"✓ Loaded {summary['total_lenses']} lens results")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_category_distribution(summary, args.output_dir)
    plot_fp_tp_scatter(summary['results'], args.output_dir)
    plot_f1_histogram(summary['results'], args.output_dir)
    plot_score_distributions(summary['results'], args.output_dir)

    # Generate report
    print("\nGenerating detailed report...")
    generate_detailed_report(summary, args.output_dir)

    print(f"\n{'='*80}")
    print(f"Analysis complete! Results in: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
