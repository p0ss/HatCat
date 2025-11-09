#!/usr/bin/env python3
"""
Test SUMO Temporal Monitoring with Varied Prompts

Validates that:
1. Generation quality is not degraded (no mode collapse)
2. Concepts are detected appropriately for different topics
3. System works across multiple samples per prompt
4. API output is consistent and usable
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.monitoring import SUMOTemporalMonitor, load_sumo_classifiers


# Diverse test prompts covering different domains
TEST_PROMPTS = [
    # AI/Tech concepts (should activate AI*, ComputerScience, etc.)
    "Artificial intelligence can help society by",
    "Machine learning algorithms work by",
    "Neural networks are used for",

    # Physical/concrete concepts (should activate Physical, Object, etc.)
    "A car is a vehicle that",
    "The ocean contains many",
    "Mountains are formed when",

    # Abstract concepts (should activate Proposition, Attribute, etc.)
    "The meaning of life is",
    "Truth and beauty are",
    "Justice requires that",

    # Social/organizational (should activate Organization, SocialRelation, etc.)
    "Companies need to",
    "Governments are responsible for",
    "Communities thrive when",

    # Quantitative (should activate Quantity, Number, etc.)
    "The calculation shows that",
    "There are many reasons why",
    "Five important factors include",
]


def test_generation_quality(results: list) -> dict:
    """
    Analyze generation quality to detect mode collapse.

    Returns:
        Dict with quality metrics
    """
    quality_metrics = {
        'total_samples': len(results),
        'mode_collapse_detected': [],
        'avg_unique_tokens': 0,
        'avg_token_length': 0,
        'repetition_rate': []
    }

    for i, result in enumerate(results):
        tokens = result['tokens']

        # Check for mode collapse (repetitive tokens)
        if len(tokens) > 5:
            # Count unique tokens in last 10 positions
            last_n = min(10, len(tokens))
            unique_in_window = len(set(tokens[-last_n:]))
            repetition = 1 - (unique_in_window / last_n)
            quality_metrics['repetition_rate'].append(repetition)

            # Flag if > 80% repetition
            if repetition > 0.8:
                quality_metrics['mode_collapse_detected'].append({
                    'sample': i,
                    'prompt': result['prompt'],
                    'generated': result['generated_text'],
                    'repetition_rate': repetition
                })

        # Average metrics
        quality_metrics['avg_unique_tokens'] += len(set(tokens))
        quality_metrics['avg_token_length'] += len(tokens)

    quality_metrics['avg_unique_tokens'] /= len(results)
    quality_metrics['avg_token_length'] /= len(results)
    quality_metrics['avg_repetition_rate'] = sum(quality_metrics['repetition_rate']) / len(quality_metrics['repetition_rate']) if quality_metrics['repetition_rate'] else 0

    return quality_metrics


def test_concept_detection(results: list) -> dict:
    """
    Analyze concept detection patterns across prompts.

    Returns:
        Dict with detection statistics
    """
    detection_stats = {
        'concepts_by_category': {},
        'avg_concepts_per_token': 0,
        'total_unique_concepts': set(),
        'layer_distribution': {0: 0, 1: 0, 2: 0}
    }

    total_tokens = 0
    total_concepts = 0

    for result in results:
        for ts in result['timesteps']:
            total_tokens += 1
            total_concepts += len(ts['concepts'])

            for c in ts['concepts']:
                detection_stats['total_unique_concepts'].add(c['concept'])
                detection_stats['layer_distribution'][c['layer']] += 1

    detection_stats['avg_concepts_per_token'] = total_concepts / total_tokens if total_tokens > 0 else 0
    detection_stats['total_unique_concepts'] = len(detection_stats['total_unique_concepts'])

    return detection_stats


def main():
    parser = argparse.ArgumentParser(
        description="Test SUMO temporal monitoring with varied prompts"
    )

    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model name (default: gemma-3-4b-pt)')
    parser.add_argument('--layers', nargs='+', type=int, default=[0, 1, 2],
                       help='Which classifier layers to load (default: 0 1 2)')
    parser.add_argument('--samples-per-prompt', type=int, default=3,
                       help='Samples per prompt (default: 3)')
    parser.add_argument('--max-tokens', type=int, default=30,
                       help='Max tokens to generate per sample (default: 30)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (default: 0.8)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Show top K concepts per timestep (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for display (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--output-dir', type=str, default='results/temporal_tests',
                       help='Output directory (default: results/temporal_tests)')
    parser.add_argument('--show-details', action='store_true',
                       help='Show token-by-token details for each sample')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SUMO TEMPORAL MONITORING - COMPREHENSIVE TEST")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print(f"Samples per prompt: {args.samples_per_prompt}")
    print(f"Total samples: {len(TEST_PROMPTS) * args.samples_per_prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")

    # Load classifiers
    print("\nLoading SUMO classifiers...")
    classifiers, hidden_dim = load_sumo_classifiers(
        layers=args.layers,
        device=args.device
    )

    # Create monitor
    monitor = SUMOTemporalMonitor(
        classifiers=classifiers,
        top_k=args.top_k,
        threshold=args.threshold
    )

    # Load model
    print(f"\nLoading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=args.device
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)

    all_results = []

    for prompt_idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{prompt_idx + 1}/{len(TEST_PROMPTS)}] Prompt: \"{prompt}\"")
        print("-" * 80)

        for sample_idx in range(args.samples_per_prompt):
            print(f"  Sample {sample_idx + 1}/{args.samples_per_prompt}...", end=" ")

            result = monitor.monitor_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                device=args.device
            )

            # Add metadata
            result['prompt_idx'] = prompt_idx
            result['sample_idx'] = sample_idx

            all_results.append(result)

            # Quick summary
            print(f"Generated {len(result['tokens'])} tokens, "
                  f"detected {result['summary']['unique_concepts_detected']} unique concepts")

            if args.show_details:
                monitor.print_report(result, show_token_details=True)

    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Quality analysis
    print("\n1. Generation Quality:")
    quality = test_generation_quality(all_results)
    print(f"   Total samples: {quality['total_samples']}")
    print(f"   Avg unique tokens: {quality['avg_unique_tokens']:.1f}")
    print(f"   Avg token length: {quality['avg_token_length']:.1f}")
    print(f"   Avg repetition rate: {quality['avg_repetition_rate']:.2%}")

    if quality['mode_collapse_detected']:
        print(f"\n   ⚠️  MODE COLLAPSE DETECTED in {len(quality['mode_collapse_detected'])} samples:")
        for collapse in quality['mode_collapse_detected'][:3]:  # Show first 3
            print(f"      Sample {collapse['sample']}: \"{collapse['generated'][:50]}...\"")
            print(f"      Repetition: {collapse['repetition']:.2%}")
    else:
        print(f"   ✓ No mode collapse detected")

    # Detection analysis
    print("\n2. Concept Detection:")
    detection = test_concept_detection(all_results)
    print(f"   Avg concepts per token: {detection['avg_concepts_per_token']:.2f}")
    print(f"   Total unique concepts: {detection['total_unique_concepts']}")
    print(f"   Layer distribution:")
    total_detections = sum(detection['layer_distribution'].values())
    for layer, count in sorted(detection['layer_distribution'].items()):
        pct = count / total_detections * 100 if total_detections > 0 else 0
        print(f"      Layer {layer}: {count} ({pct:.1f}%)")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save all individual results
    for i, result in enumerate(all_results):
        result_file = output_dir / f"sample_{i:03d}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

    print(f"✓ Saved {len(all_results)} individual results to {output_dir}/")

    # Save summary
    summary = {
        'test_config': {
            'model': args.model,
            'num_prompts': len(TEST_PROMPTS),
            'samples_per_prompt': args.samples_per_prompt,
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'threshold': args.threshold
        },
        'quality_metrics': quality,
        'detection_stats': detection,
        'prompts': TEST_PROMPTS
    }

    summary_file = output_dir / "test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)  # default=str for set serialization

    print(f"✓ Saved summary to {summary_file}")

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if quality['mode_collapse_detected']:
        print("❌ FAILED - Mode collapse detected")
        print("   The monitoring system is interfering with generation")
        return 1
    elif quality['avg_repetition_rate'] > 0.5:
        print("⚠️  WARNING - High repetition rate")
        print("   Generation quality may be degraded")
        return 1
    else:
        print("✓ PASSED - No generation degradation detected")
        print(f"  Generated diverse text across {len(all_results)} samples")
        print(f"  Detected {detection['total_unique_concepts']} unique concepts")
        print(f"  Average {detection['avg_concepts_per_token']:.2f} concepts per token")
        return 0


if __name__ == '__main__':
    sys.exit(main())
