"""
Visualize temporal activation patterns from recorded data.

Usage:
    python scripts/visualize_temporal_activations.py \
        results/temporal_test/deception_politics.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_timeline(filepath: Path):
    """Load recorded timeline data"""
    with open(filepath) as f:
        return json.load(f)


def extract_concept_timeseries(timeline_data):
    """
    Convert timeline to concept timeseries.

    Returns:
        concepts: dict of {concept_name: [divergences over time]}
        token_indices: list of token indices for each timestep
    """
    timeline = timeline_data['recorder']

    # Collect all concepts that appear
    all_concepts = set()
    for step in timeline:
        all_concepts.update(step['concepts'].keys())

    # Build timeseries for each concept
    concepts = {name: [] for name in all_concepts}
    token_indices = []

    for step in timeline:
        token_indices.append(step['token_idx'])
        for name in all_concepts:
            if name in step['concepts']:
                concepts[name].append(step['concepts'][name]['divergence'])
            else:
                concepts[name].append(0.0)  # Not active

    return concepts, token_indices


def plot_timeline(timeline_data, output_path=None, top_k=10):
    """
    Plot concept activations over time.

    Shows top-K concepts by max divergence as lines on a chart.
    """
    concepts, token_indices = extract_concept_timeseries(timeline_data)

    # Sort concepts by max divergence
    concept_maxes = {name: max(vals) for name, vals in concepts.items()}
    top_concepts = sorted(concept_maxes.items(),
                         key=lambda x: x[1],
                         reverse=True)[:top_k]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each concept
    for concept_name, max_div in top_concepts:
        values = concepts[concept_name]
        ax.plot(token_indices, values, label=concept_name, linewidth=2, alpha=0.8)

    # Mark token emissions with vertical lines
    unique_tokens = sorted(set(token_indices))
    for tok_idx in unique_tokens:
        ax.axvline(x=tok_idx, color='gray', alpha=0.2, linestyle='--', linewidth=0.5)

    # Formatting
    ax.set_xlabel('Token Index', fontsize=12)
    ax.set_ylabel('Divergence', fontsize=12)
    ax.set_title(f'Temporal Concept Activations\n"{timeline_data["prompt"]}"',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_sparklines(timeline_data, output_path=None, top_k=10):
    """
    Create ASCII sparkline visualization (like the mockup).
    """
    concepts, token_indices = extract_concept_timeseries(timeline_data)

    # Sort concepts by max divergence
    concept_maxes = {name: max(vals) for name, vals in concepts.items()}
    top_concepts = sorted(concept_maxes.items(),
                         key=lambda x: x[1],
                         reverse=True)[:top_k]

    # Sparkline characters
    chars = '▁▂▃▄▅▆▇█'

    print(f"\n{'='*80}")
    print(f"Temporal Concept Activations")
    print(f"Prompt: {timeline_data['prompt']}")
    print(f"Generated: {timeline_data['generated_text']}")
    print(f"{'='*80}\n")

    for concept_name, max_div in top_concepts:
        values = np.array(concepts[concept_name])

        # Normalize to sparkline range
        if max_div > 0:
            normalized = (values / max_div * 7).astype(int)
            sparkline = ''.join(chars[min(n, 7)] for n in normalized)
        else:
            sparkline = chars[0] * len(values)

        # Print with concept name and max divergence
        print(f"{concept_name:30s} [{max_div:5.3f}] {sparkline}")

    print(f"\n{'='*80}")
    print(f"Total timesteps: {len(token_indices)}")
    print(f"Unique tokens: {len(set(token_indices))}")
    print(f"{'='*80}\n")


def analyze_temporal_patterns(timeline_data):
    """
    Look for interesting temporal patterns:
    - Pre-generation concept activation (concept peaks before it's mentioned)
    - Concept competition (multiple concepts high at same time)
    - Thinking pauses (high activity with no token output)
    """
    timeline = timeline_data['recorder']

    print(f"\n{'='*80}")
    print("Temporal Pattern Analysis")
    print(f"{'='*80}\n")

    # 1. Check for pre-generation activation
    print("Looking for pre-generation concept activation...")
    # TODO: Compare concept peaks to generated text mentions

    # 2. Check for concept competition
    print("\nLooking for concept competition (multiple high concepts)...")
    multi_concept_steps = []
    for step in timeline:
        high_concepts = [name for name, data in step['concepts'].items()
                        if data['divergence'] > 0.5]
        if len(high_concepts) >= 3:
            multi_concept_steps.append({
                'token_idx': step['token_idx'],
                'concepts': high_concepts
            })

    if multi_concept_steps:
        print(f"Found {len(multi_concept_steps)} steps with 3+ high concepts:")
        for step in multi_concept_steps[:5]:  # Show first 5
            print(f"  Token {step['token_idx']}: {', '.join(step['concepts'])}")

    # 3. Count forward passes per token (thinking time)
    print("\nForward passes per token (thinking time)...")
    from collections import Counter
    token_counts = Counter(step['token_idx'] for step in timeline)
    avg_passes = np.mean(list(token_counts.values()))
    print(f"  Average: {avg_passes:.1f} forward passes per token")

    # Show tokens with lots of thinking
    heavy_tokens = [(tok, count) for tok, count in token_counts.items()
                   if count > avg_passes * 1.5]
    if heavy_tokens:
        print(f"  Tokens with heavy thinking (>1.5x avg):")
        for tok, count in sorted(heavy_tokens)[:5]:
            print(f"    Token {tok}: {count} passes")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help='JSON file from record_temporal_activations.py')
    parser.add_argument('--output', type=Path, help='Output plot file (PNG)')
    parser.add_argument('--top-k', type=int, default=10, help='Number of concepts to show')
    parser.add_argument('--mode', choices=['plot', 'sparkline', 'analyze', 'all'],
                       default='all', help='Visualization mode')
    args = parser.parse_args()

    # Load data
    timeline_data = load_timeline(args.input)

    # Generate visualizations
    if args.mode in ['plot', 'all']:
        output_path = args.output or args.input.with_suffix('.png')
        plot_timeline(timeline_data, output_path, args.top_k)

    if args.mode in ['sparkline', 'all']:
        plot_sparklines(timeline_data, top_k=args.top_k)

    if args.mode in ['analyze', 'all']:
        analyze_temporal_patterns(timeline_data)


if __name__ == '__main__':
    main()
