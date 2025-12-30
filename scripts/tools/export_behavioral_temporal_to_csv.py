#!/usr/bin/env python3
"""
Export Behavioral vs Definitional Temporal Results to CSV

Reads the JSON results from temporal concept detection and exports to CSV
for visualization and analysis.
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def load_results(results_dir: Path):
    """Load all result JSON files from a run directory."""
    results = {}

    for verb in ['deceive', 'manipulate', 'persuade']:
        analysis_file = results_dir / f'{verb}_analysis.json'
        if analysis_file.exists():
            with open(analysis_file) as f:
                results[verb] = json.load(f)

    return results


def create_concept_frequency_csv(results: dict, output_file: Path):
    """Create CSV of concept frequencies by verb and prompt type."""

    rows = []

    for verb, analysis in results.items():
        for prompt_type, data in analysis.items():
            # Top concepts across all timesteps
            for concept, freq in data['top_concepts_all_timesteps']:
                rows.append({
                    'verb': verb,
                    'prompt_type': prompt_type,
                    'concept': concept,
                    'frequency': freq,
                    'measurement': 'all_timesteps',
                    'samples': data['n_samples']
                })

            # Top concepts final state only
            for concept, freq in data['top_concepts_final_only']:
                rows.append({
                    'verb': verb,
                    'prompt_type': prompt_type,
                    'concept': concept,
                    'frequency': freq,
                    'measurement': 'final_state',
                    'samples': data['n_samples']
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"✓ Exported {len(df)} rows to {output_file}")

    return df


def create_summary_stats_csv(results: dict, output_file: Path):
    """Create CSV of summary statistics by verb and prompt type."""

    rows = []

    for verb, analysis in results.items():
        for prompt_type, data in analysis.items():
            rows.append({
                'verb': verb,
                'prompt_type': prompt_type,
                'n_samples': data['n_samples'],
                'unique_concepts_all': data['unique_concepts_all_timesteps'],
                'unique_concepts_final': data['unique_concepts_final_only'],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"✓ Exported {len(df)} rows to {output_file}")

    return df


def create_top_concepts_pivot(df: pd.DataFrame, output_file: Path, top_n: int = 5):
    """Create pivot table showing top N concepts for each verb×prompt_type combination."""

    # Filter to top N concepts per group
    df_filtered = (df[df['measurement'] == 'all_timesteps']
                   .sort_values(['verb', 'prompt_type', 'frequency'], ascending=[True, True, False])
                   .groupby(['verb', 'prompt_type'])
                   .head(top_n))

    # Create pivot
    pivot = df_filtered.pivot_table(
        index='concept',
        columns=['verb', 'prompt_type'],
        values='frequency',
        fill_value=0
    )

    pivot.to_csv(output_file)
    print(f"✓ Exported pivot table to {output_file}")

    return pivot


def main():
    parser = argparse.ArgumentParser(
        description="Export behavioral vs definitional temporal results to CSV"
    )
    parser.add_argument('results_dir', type=Path, help='Results directory')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory (default: same as results_dir)')
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1

    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)

    if not results:
        print("Error: No results found")
        return 1

    print(f"\nFound results for {len(results)} verbs: {', '.join(results.keys())}")

    # Export concept frequencies
    print("\nExporting concept frequencies...")
    df_concepts = create_concept_frequency_csv(
        results,
        output_dir / 'concept_frequencies.csv'
    )

    # Export summary stats
    print("\nExporting summary statistics...")
    df_summary = create_summary_stats_csv(
        results,
        output_dir / 'summary_stats.csv'
    )

    # Export pivot table of top concepts
    print("\nExporting top concepts pivot table...")
    pivot = create_top_concepts_pivot(
        df_concepts,
        output_dir / 'top_concepts_pivot.csv',
        top_n=10
    )

    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    print("  - concept_frequencies.csv (detailed concept frequencies)")
    print("  - summary_stats.csv (summary statistics)")
    print("  - top_concepts_pivot.csv (pivot table of top concepts)")

    return 0


if __name__ == '__main__':
    exit(main())
