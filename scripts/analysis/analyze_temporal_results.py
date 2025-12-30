#!/usr/bin/env python3
"""
Analyze temporal monitoring test results and create a summary of most frequent concepts.

This script reads all sample JSON files from a test run and generates a summary
showing:
1. Top concepts across all samples
2. Top concepts per prompt
3. Layer distribution statistics
4. Concept frequency patterns
5. Word frequency analysis of generated text (excluding stopwords)
6. Top words per prompt
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# Common English stopwords to exclude from word frequency analysis
STOPWORDS = {
    'the', 'and', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
    'may', 'might', 'must', 'can', 'of', 'to', 'in', 'for', 'on', 'at', 'by',
    'with', 'from', 'as', 'it', 'that', 'this', 'these', 'those', 'i', 'you',
    'he', 'she', 'we', 'they', 'them', 'their', 'his', 'her', 'its', 'our',
    'your', 'my', 'me', 'him', 'us', 'what', 'which', 'who', 'when', 'where',
    'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'but', 'or', 'because', 'if', 'while', 'after',
    'before', 'above', 'below', 'between', 'during', 'through', 'into', 'up',
    'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
}


def load_sample_results(result_dir: Path) -> List[Dict]:
    """Load all sample JSON files from the result directory."""
    results = []
    sample_files = sorted(result_dir.glob("sample_*.json"))

    for sample_file in sample_files:
        with open(sample_file, 'r') as f:
            results.append(json.load(f))

    return results


def analyze_concept_frequencies(results: List[Dict]) -> Dict:
    """
    Analyze concept frequencies across all samples.

    Returns:
        Dict with concept frequencies, layer distribution, and per-prompt stats
    """
    # Global concept frequency
    global_concept_freq = Counter()

    # Per-layer concept frequency
    layer_concept_freq = defaultdict(Counter)

    # Per-prompt concept frequency
    prompt_concept_freq = defaultdict(Counter)

    # Concept co-occurrence (which concepts appear together)
    concept_cooccurrence = defaultdict(Counter)

    for result in results:
        prompt = result.get('prompt', 'Unknown')

        # Track concepts seen in this sample
        sample_concepts = set()

        for timestep in result.get('timesteps', []):
            for concept_name, concept_data in timestep.get('concepts', {}).items():
                layer = concept_data.get('layer', -1)

                # Update global frequency
                global_concept_freq[concept_name] += 1

                # Update layer-specific frequency
                layer_concept_freq[layer][concept_name] += 1

                # Update prompt-specific frequency
                prompt_concept_freq[prompt][concept_name] += 1

                # Track for co-occurrence
                sample_concepts.add(concept_name)

        # Update co-occurrence matrix
        sample_concepts_list = list(sample_concepts)
        for i, concept_a in enumerate(sample_concepts_list):
            for concept_b in sample_concepts_list[i+1:]:
                concept_cooccurrence[concept_a][concept_b] += 1
                concept_cooccurrence[concept_b][concept_a] += 1

    return {
        'global_frequency': global_concept_freq,
        'layer_frequency': dict(layer_concept_freq),
        'prompt_frequency': dict(prompt_concept_freq),
        'cooccurrence': dict(concept_cooccurrence)
    }


def analyze_word_frequencies(results: List[Dict]) -> Dict:
    """
    Analyze word frequencies in generated text across all samples.

    Returns:
        Dict with word frequencies globally and per-prompt
    """
    # Global word frequency
    global_word_freq = Counter()

    # Per-prompt word frequency
    prompt_word_freq = defaultdict(Counter)

    for result in results:
        prompt = result.get('prompt', 'Unknown')
        generated_text = result.get('generated_text', '')

        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z]+\b', generated_text.lower())

        # Filter stopwords and count
        for word in words:
            if word not in STOPWORDS and len(word) > 1:  # Also skip single-letter words
                global_word_freq[word] += 1
                prompt_word_freq[prompt][word] += 1

    return {
        'global_frequency': global_word_freq,
        'prompt_frequency': dict(prompt_word_freq)
    }


def generate_summary_report(results: List[Dict], analysis: Dict, word_analysis: Dict, output_file: Path):
    """Generate a comprehensive summary report."""

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEMPORAL MONITORING - CONCEPT FREQUENCY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Test overview
        f.write(f"Total samples analyzed: {len(results)}\n")

        if results:
            total_timesteps = sum(len(r.get('timesteps', [])) for r in results)
            f.write(f"Total timesteps: {total_timesteps}\n")
            f.write(f"Unique concepts detected: {len(analysis['global_frequency'])}\n\n")

        # Top concepts overall
        f.write("=" * 80 + "\n")
        f.write("TOP 50 MOST FREQUENT CONCEPTS (ACROSS ALL SAMPLES)\n")
        f.write("=" * 80 + "\n\n")

        top_concepts = analysis['global_frequency'].most_common(50)
        for rank, (concept, count) in enumerate(top_concepts, 1):
            f.write(f"{rank:3d}. {concept:40s} {count:6d} occurrences\n")

        # Layer distribution
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCEPT FREQUENCY BY LAYER\n")
        f.write("=" * 80 + "\n\n")

        for layer in sorted(analysis['layer_frequency'].keys()):
            layer_concepts = analysis['layer_frequency'][layer]
            total_in_layer = sum(layer_concepts.values())

            f.write(f"Layer {layer}: {total_in_layer} total detections, "
                   f"{len(layer_concepts)} unique concepts\n")

            # Top 10 concepts for this layer
            top_layer = layer_concepts.most_common(10)
            for rank, (concept, count) in enumerate(top_layer, 1):
                pct = (count / total_in_layer * 100) if total_in_layer > 0 else 0
                f.write(f"  {rank:2d}. {concept:35s} {count:5d} ({pct:5.1f}%)\n")
            f.write("\n")

        # Per-prompt analysis
        f.write("=" * 80 + "\n")
        f.write("TOP CONCEPTS PER PROMPT\n")
        f.write("=" * 80 + "\n\n")

        for prompt, concept_freq in analysis['prompt_frequency'].items():
            f.write(f"Prompt: \"{prompt}\"\n")
            f.write("-" * 80 + "\n")

            top_prompt = concept_freq.most_common(15)
            total_for_prompt = sum(concept_freq.values())

            for rank, (concept, count) in enumerate(top_prompt, 1):
                pct = (count / total_for_prompt * 100) if total_for_prompt > 0 else 0
                f.write(f"  {rank:2d}. {concept:35s} {count:5d} ({pct:5.1f}%)\n")
            f.write("\n")

        # Generated text word frequency
        f.write("=" * 80 + "\n")
        f.write("GENERATED TEXT - WORD FREQUENCY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Top 30 Most Frequent Words (excluding stopwords):\n")
        f.write("-" * 80 + "\n")
        top_words = word_analysis['global_frequency'].most_common(30)
        for rank, (word, count) in enumerate(top_words, 1):
            f.write(f"{rank:3d}. {word:25s} {count:6d} occurrences\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP WORDS PER PROMPT\n")
        f.write("=" * 80 + "\n\n")

        for prompt, word_freq in word_analysis['prompt_frequency'].items():
            f.write(f"Prompt: \"{prompt}\"\n")
            f.write("-" * 80 + "\n")

            top_prompt_words = word_freq.most_common(10)
            total_words = sum(word_freq.values())

            for rank, (word, count) in enumerate(top_prompt_words, 1):
                pct = (count / total_words * 100) if total_words > 0 else 0
                f.write(f"  {rank:2d}. {word:25s} {count:5d} ({pct:5.1f}%)\n")
            f.write("\n")

        # Concept co-occurrence
        f.write("=" * 80 + "\n")
        f.write("CONCEPT CO-OCCURRENCE (Top 20 pairs)\n")
        f.write("=" * 80 + "\n\n")

        # Flatten co-occurrence into pairs
        cooccurrence_pairs = []
        seen_pairs = set()

        for concept_a, cooccur_dict in analysis['cooccurrence'].items():
            for concept_b, count in cooccur_dict.items():
                pair = tuple(sorted([concept_a, concept_b]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    cooccurrence_pairs.append((pair, count))

        cooccurrence_pairs.sort(key=lambda x: x[1], reverse=True)

        for rank, (pair, count) in enumerate(cooccurrence_pairs[:20], 1):
            f.write(f"{rank:3d}. {pair[0]:30s} <-> {pair[1]:30s} ({count} samples)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def generate_json_summary(analysis: Dict, word_analysis: Dict, output_file: Path):
    """Generate a JSON summary for programmatic analysis."""

    summary = {
        'top_concepts_global': [
            {'concept': concept, 'count': count}
            for concept, count in analysis['global_frequency'].most_common(100)
        ],
        'top_concepts_by_layer': {},
        'top_concepts_by_prompt': {},
        'top_words_global': [
            {'word': word, 'count': count}
            for word, count in word_analysis['global_frequency'].most_common(50)
        ],
        'top_words_by_prompt': {}
    }

    # Layer summaries
    for layer, concept_freq in analysis['layer_frequency'].items():
        summary['top_concepts_by_layer'][str(layer)] = [
            {'concept': concept, 'count': count}
            for concept, count in concept_freq.most_common(20)
        ]

    # Prompt summaries - concepts
    for prompt, concept_freq in analysis['prompt_frequency'].items():
        summary['top_concepts_by_prompt'][prompt] = [
            {'concept': concept, 'count': count}
            for concept, count in concept_freq.most_common(20)
        ]

    # Prompt summaries - words
    for prompt, word_freq in word_analysis['prompt_frequency'].items():
        summary['top_words_by_prompt'][prompt] = [
            {'word': word, 'count': count}
            for word, count in word_freq.most_common(15)
        ]

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze temporal monitoring test results"
    )

    parser.add_argument('result_dir', type=str,
                       help='Directory containing sample_*.json files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for text report (default: result_dir/analysis_report.txt)')
    parser.add_argument('--json-output', type=str, default=None,
                       help='Output file for JSON summary (default: result_dir/analysis_summary.json)')

    args = parser.parse_args()

    result_dir = Path(args.result_dir)

    if not result_dir.exists():
        print(f"Error: Directory {result_dir} does not exist")
        return 1

    # Set default output paths
    if args.output is None:
        output_file = result_dir / "analysis_report.txt"
    else:
        output_file = Path(args.output)

    if args.json_output is None:
        json_output_file = result_dir / "analysis_summary.json"
    else:
        json_output_file = Path(args.json_output)

    print(f"Loading results from {result_dir}...")
    results = load_sample_results(result_dir)

    if not results:
        print(f"Error: No sample files found in {result_dir}")
        return 1

    print(f"Loaded {len(results)} samples")

    print("Analyzing concept frequencies...")
    analysis = analyze_concept_frequencies(results)

    print("Analyzing word frequencies in generated text...")
    word_analysis = analyze_word_frequencies(results)

    print(f"Generating text report: {output_file}")
    generate_summary_report(results, analysis, word_analysis, output_file)

    print(f"Generating JSON summary: {json_output_file}")
    generate_json_summary(analysis, word_analysis, json_output_file)

    print("\nAnalysis complete!")
    print(f"  - Text report: {output_file}")
    print(f"  - JSON summary: {json_output_file}")

    # Print quick summary to console
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(results)}")
    print(f"Unique concepts: {len(analysis['global_frequency'])}")
    print(f"\nTop 10 concepts:")
    for rank, (concept, count) in enumerate(analysis['global_frequency'].most_common(10), 1):
        print(f"  {rank:2d}. {concept:40s} {count:6d}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
