#!/usr/bin/env python3
"""
Analyze HatCat concept detections from behavioral vs definitional experiment.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import sys

def analyze_hatcat_concepts(run_dir: Path):
    """Analyze HatCat concepts across all verbs and prompt types."""

    results = {}

    for verb_dir in run_dir.iterdir():
        if not verb_dir.is_dir():
            continue

        verb = verb_dir.name
        results[verb] = {}

        # Find all training_data_*.json files
        for json_file in verb_dir.glob('training_data_*.json'):
            if json_file.name == 'training_data_all.json':
                continue  # Skip combined file

            prompt_type = json_file.stem.replace('training_data_', '')

            with open(json_file) as f:
                samples = json.load(f)

            # Extract HatCat concepts from each sample
            prompt_concepts = Counter()
            response_concepts = Counter()

            for sample in samples:
                if 'hatcat_prompt' in sample and 'top_concepts' in sample['hatcat_prompt']:
                    for concept in sample['hatcat_prompt']['top_concepts']:
                        prompt_concepts[concept] += 1

                if 'hatcat_response' in sample and 'top_concepts' in sample['hatcat_response']:
                    for concept in sample['hatcat_response']['top_concepts']:
                        response_concepts[concept] += 1

            results[verb][prompt_type] = {
                'prompt_concepts': dict(prompt_concepts.most_common(20)),
                'response_concepts': dict(response_concepts.most_common(20)),
                'num_samples': len(samples)
            }

    return results

def print_summary(results: dict):
    """Print a readable summary of concept analysis."""

    for verb, prompt_types in results.items():
        print(f"\n{'='*80}")
        print(f"VERB: {verb.upper()}")
        print(f"{'='*80}")

        for prompt_type, data in prompt_types.items():
            print(f"\n{prompt_type} ({data['num_samples']} samples):")
            print(f"-" * 80)

            print("\n  Top Prompt Concepts:")
            for concept, count in list(data['prompt_concepts'].items())[:10]:
                print(f"    {concept:30s} : {count}")

            print("\n  Top Response Concepts:")
            for concept, count in list(data['response_concepts'].items())[:10]:
                print(f"    {concept:30s} : {count}")

def save_markdown_report(results: dict, output_file: Path):
    """Save analysis as markdown report."""

    with open(output_file, 'w') as f:
        f.write("# HatCat Concept Analysis\n\n")
        f.write("Analysis of concept detections across prompt types\n\n")

        for verb, prompt_types in results.items():
            f.write(f"\n## {verb.upper()}\n\n")

            for prompt_type, data in prompt_types.items():
                f.write(f"\n### {prompt_type} ({data['num_samples']} samples)\n\n")

                f.write("#### Top Prompt Concepts\n\n")
                f.write("| Concept | Count |\n")
                f.write("|---------|-------|\n")
                for concept, count in list(data['prompt_concepts'].items())[:15]:
                    f.write(f"| {concept} | {count} |\n")

                f.write("\n#### Top Response Concepts\n\n")
                f.write("| Concept | Count |\n")
                f.write("|---------|-------|\n")
                for concept, count in list(data['response_concepts'].items())[:15]:
                    f.write(f"| {concept} | {count} |\n")

                f.write("\n---\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_hatcat_concepts.py <run_directory>")
        print("\nExample:")
        print("  python analyze_hatcat_concepts.py results/behavioral_vs_definitional_experiment/run_20251117_141036")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(1)

    print(f"Analyzing HatCat concepts in: {run_dir}")

    results = analyze_hatcat_concepts(run_dir)

    # Print to console
    print_summary(results)

    # Save markdown report
    output_file = run_dir / 'hatcat_concept_analysis.md'
    save_markdown_report(results, output_file)
    print(f"\n✓ Saved detailed report to: {output_file}")

    # Save JSON for programmatic access
    json_file = run_dir / 'hatcat_concept_analysis.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON data to: {json_file}")

if __name__ == '__main__':
    main()
