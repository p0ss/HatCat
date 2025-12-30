#!/usr/bin/env python3
"""
Analyze and visualize lens benchmark results.

Creates:
- Confusion matrices (heatmaps of cross-concept activation)
- Steering effectiveness plots
- Quality score reports
- Anomaly detection
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_benchmark_csv(csv_path: Path) -> pd.DataFrame:
    """Load benchmark CSV into DataFrame."""
    return pd.read_csv(csv_path)


def create_confusion_matrix(
    df: pd.DataFrame,
    context: str,
    metric: str = 'mean_activation',
    output_path: Optional[Path] = None
):
    """
    Create confusion matrix heatmap for concept lenses.

    Args:
        df: Benchmark results DataFrame
        context: Context to visualize ("prompt", "response", etc.)
        metric: Metric to display
        output_path: Where to save plot
    """
    # Filter to context
    ctx_df = df[df['context'] == context]

    # Pivot to matrix format
    concepts = sorted(ctx_df['prompt_concept'].unique())
    matrix = np.zeros((len(concepts), len(concepts)))

    for i, prompt_concept in enumerate(concepts):
        for j, detected_concept in enumerate(concepts):
            value = ctx_df[
                (ctx_df['prompt_concept'] == prompt_concept) &
                (ctx_df['detected_concept'] == detected_concept)
            ][metric].values

            if len(value) > 0:
                matrix[i, j] = value[0]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=concepts,
        yticklabels=concepts,
        vmin=0,
        vmax=1,
        cbar_kws={'label': metric}
    )
    plt.xlabel('Detected Concept')
    plt.ylabel('Prompted Concept')
    plt.title(f'Concept Lens Confusion Matrix ({context})')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved confusion matrix: {output_path}")
    else:
        plt.show()

    plt.close()


def create_simplex_confusion_matrix(
    df: pd.DataFrame,
    simplex: str,
    context: str,
    metric: str = 'mean_activation',
    output_path: Optional[Path] = None
):
    """
    Create 3x3 confusion matrix for a simplex.

    Args:
        df: Benchmark results DataFrame
        simplex: Simplex dimension name
        context: Context to visualize
        metric: Metric to display
        output_path: Where to save plot
    """
    # Filter to simplex and context
    simplex_df = df[
        (df['simplex_dimension'] == simplex) &
        (df['context'] == context)
    ]

    poles = ['negative', 'neutral', 'positive']
    matrix = np.zeros((3, 3))

    for i, prompted_pole in enumerate(poles):
        for j, detected_pole in enumerate(poles):
            value = simplex_df[
                (simplex_df['pole_prompted'] == prompted_pole) &
                (simplex_df['pole_detected'] == detected_pole)
            ][metric].values

            if len(value) > 0:
                matrix[i, j] = value[0]

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=['μ−', 'μ0', 'μ+'],
        yticklabels=['μ−', 'μ0', 'μ+'],
        vmin=0,
        vmax=1,
        cbar_kws={'label': metric}
    )
    plt.xlabel('Detected Pole')
    plt.ylabel('Prompted Pole')
    plt.title(f'{simplex} Simplex Confusion Matrix ({context})')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved simplex matrix: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_steering_effectiveness(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
):
    """
    Plot steering effectiveness for concept lenses.

    Shows mean activation across baseline, steered_up, and steered_down.
    """
    # Get concepts
    concepts = sorted(df['prompt_concept'].unique())

    # Extract data for each concept
    baseline = []
    steered_up = []
    steered_down = []

    for concept in concepts:
        baseline_val = df[
            (df['prompt_concept'] == concept) &
            (df['detected_concept'] == concept) &
            (df['context'] == 'prompt')
        ]['mean_activation'].values[0] if len(df[
            (df['prompt_concept'] == concept) &
            (df['detected_concept'] == concept) &
            (df['context'] == 'prompt')
        ]) > 0 else 0

        up_val = df[
            (df['prompt_concept'] == concept) &
            (df['context'] == 'steered_up')
        ]['mean_activation'].values[0] if len(df[
            (df['prompt_concept'] == concept) &
            (df['context'] == 'steered_up')
        ]) > 0 else 0

        down_val = df[
            (df['prompt_concept'] == concept) &
            (df['context'] == 'steered_down')
        ]['mean_activation'].values[0] if len(df[
            (df['prompt_concept'] == concept) &
            (df['context'] == 'steered_down')
        ]) > 0 else 0

        baseline.append(baseline_val)
        steered_up.append(up_val)
        steered_down.append(down_val)

    # Plot
    x = np.arange(len(concepts))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, baseline, width, label='Baseline (prompt)', alpha=0.8)
    ax.bar(x, steered_up, width, label='Steered Up', alpha=0.8)
    ax.bar(x + width, steered_down, width, label='Steered Down', alpha=0.8)

    ax.set_xlabel('Concept')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Steering Effectiveness (Concept Lenses)')
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved steering plot: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_simplex_steering(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
):
    """
    Plot homeostatic steering effectiveness for simplexes.

    Shows pole activations under different steering contexts.
    """
    simplexes = sorted(df['simplex_dimension'].unique())

    fig, axes = plt.subplots(len(simplexes), 1, figsize=(12, 4 * len(simplexes)))
    if len(simplexes) == 1:
        axes = [axes]

    for idx, simplex in enumerate(simplexes):
        ax = axes[idx]

        # Get steering contexts
        contexts = ['steered_to_negative', 'steered_to_neutral', 'steered_to_positive']
        poles = ['negative', 'neutral', 'positive']

        # Build data matrix [context, pole]
        data = np.zeros((len(contexts), len(poles)))

        for i, context in enumerate(contexts):
            for j, pole in enumerate(poles):
                values = df[
                    (df['simplex_dimension'] == simplex) &
                    (df['context'] == context) &
                    (df['pole_detected'] == pole)
                ]['mean_activation'].values

                if len(values) > 0:
                    data[i, j] = values[0]

        # Plot grouped bar chart
        x = np.arange(len(contexts))
        width = 0.25

        for j, pole in enumerate(poles):
            ax.bar(x + (j - 1) * width, data[:, j], width, label=f'μ{["−", "0", "+"][j]}', alpha=0.8)

        ax.set_xlabel('Steering Context')
        ax.set_ylabel('Mean Activation')
        ax.set_title(f'{simplex} Homeostatic Steering')
        ax.set_xticks(x)
        ax.set_xticklabels(['→ μ−', '→ μ0', '→ μ+'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved simplex steering plot: {output_path}")
    else:
        plt.show()

    plt.close()


def calculate_quality_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate quality metrics for concept lenses.

    Returns:
        Dict with quality scores per concept
    """
    concepts = sorted(df['prompt_concept'].unique())
    quality = {}

    for concept in concepts:
        # Diagonal (correct concept)
        diagonal = df[
            (df['prompt_concept'] == concept) &
            (df['detected_concept'] == concept) &
            (df['context'] == 'prompt')
        ]['mean_activation'].values

        # Off-diagonal (incorrect concepts)
        off_diagonal = df[
            (df['prompt_concept'] == concept) &
            (df['detected_concept'] != concept) &
            (df['context'] == 'prompt')
        ]['mean_activation'].values

        # Baseline vs steered
        baseline = diagonal[0] if len(diagonal) > 0 else 0

        steered_up = df[
            (df['prompt_concept'] == concept) &
            (df['context'] == 'steered_up')
        ]['mean_activation'].values
        steered_up = steered_up[0] if len(steered_up) > 0 else 0

        steered_down = df[
            (df['prompt_concept'] == concept) &
            (df['context'] == 'steered_down')
        ]['mean_activation'].values
        steered_down = steered_down[0] if len(steered_down) > 0 else 0

        # Compute metrics
        precision = baseline / (np.mean(off_diagonal) + 1e-6) if len(off_diagonal) > 0 else 0
        selectivity = 1 - (np.max(off_diagonal) / (baseline + 1e-6)) if len(off_diagonal) > 0 else 0
        steerability = (steered_up - baseline) / (baseline + 1e-6)
        suppression = 1 - (steered_down / (baseline + 1e-6))

        quality[concept] = {
            'precision': float(precision),
            'selectivity': float(selectivity),
            'steerability': float(steerability),
            'suppression': float(suppression),
            'baseline_activation': float(baseline)
        }

    return quality


def save_quality_report(quality: Dict, output_path: Path):
    """Save quality metrics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(quality, f, indent=2)

    print(f"  ✓ Saved quality report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze lens benchmark results")
    parser.add_argument("--concept-csv", type=Path, help="Concept lens benchmark CSV")
    parser.add_argument("--simplex-csv", type=Path, help="Simplex lens benchmark CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for plots/reports")
    args = parser.parse_args()

    print("=" * 80)
    print("LENS BENCHMARK ANALYSIS")
    print("=" * 80)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze concept lenses
    if args.concept_csv and args.concept_csv.exists():
        print(f"\nAnalyzing concept lenses: {args.concept_csv}")
        df = load_benchmark_csv(args.concept_csv)

        # Confusion matrices
        print("  Creating confusion matrices...")
        for context in ['prompt', 'response']:
            create_confusion_matrix(
                df, context,
                output_path=args.output_dir / f'confusion_matrix_{context}.png'
            )

        # Steering effectiveness
        print("  Creating steering plot...")
        plot_steering_effectiveness(
            df,
            output_path=args.output_dir / 'steering_effectiveness.png'
        )

        # Quality metrics
        print("  Calculating quality metrics...")
        quality = calculate_quality_metrics(df)
        save_quality_report(quality, args.output_dir / 'concept_quality_report.json')

        # Print summary
        print("\n  Quality Summary:")
        for concept, metrics in quality.items():
            print(f"    {concept}:")
            print(f"      Precision: {metrics['precision']:.2f}")
            print(f"      Selectivity: {metrics['selectivity']:.2f}")
            print(f"      Steerability: {metrics['steerability']:.2f}")

    # Analyze simplex lenses
    if args.simplex_csv and args.simplex_csv.exists():
        print(f"\nAnalyzing simplex lenses: {args.simplex_csv}")
        df = load_benchmark_csv(args.simplex_csv)

        simplexes = df['simplex_dimension'].unique()

        # Confusion matrices for each simplex
        print("  Creating simplex confusion matrices...")
        for simplex in simplexes:
            for context in ['prompt', 'response']:
                create_simplex_confusion_matrix(
                    df, simplex, context,
                    output_path=args.output_dir / f'simplex_{simplex}_{context}.png'
                )

        # Steering plots
        print("  Creating homeostatic steering plot...")
        plot_simplex_steering(
            df,
            output_path=args.output_dir / 'simplex_steering.png'
        )

    print("\n" + "=" * 80)
    print(f"✓ Analysis complete. Results in {args.output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
