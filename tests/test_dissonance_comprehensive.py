#!/usr/bin/env python3
"""Comprehensive dissonance measurement tests with timing analysis."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.monitoring.temporal_monitor import SUMOTemporalMonitor, load_sumo_classifiers


# Test prompts covering different domains
TEST_PROMPTS = [
    # Simple concrete
    "The cat sat on the mat",
    "The dog ran across the street",
    "I walked to the store",

    # Abstract/technical
    "Artificial intelligence systems learn from data",
    "Quantum computing uses superposition",
    "Democracy requires informed citizens",

    # Self-referential
    "I am thinking about my own thoughts",
    "What does consciousness feel like",
    "Can language models truly understand meaning",

    # Emotional/social
    "She felt happy after helping others",
    "Trust is essential for relationships",
    "Fear can paralyze decision making",
]


def run_timed_test(
    prompt: str,
    model,
    tokenizer,
    monitor_without_diss: SUMOTemporalMonitor,
    monitor_with_diss: SUMOTemporalMonitor,
    device: str,
    max_tokens: int = 20,
) -> Dict:
    """Run test with and without dissonance to measure overhead."""

    gen_kwargs = {
        'temperature': 0.8,
        'top_p': 0.95,
        'do_sample': True,
    }

    # Test WITHOUT dissonance
    start = time.perf_counter()
    result_no_diss = monitor_without_diss.monitor_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        device=device,
        **gen_kwargs,
    )
    time_no_diss = time.perf_counter() - start

    # Test WITH dissonance
    start = time.perf_counter()
    result_with_diss = monitor_with_diss.monitor_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        device=device,
        **gen_kwargs,
    )
    time_with_diss = time.perf_counter() - start

    # Calculate overhead
    overhead_ms = (time_with_diss - time_no_diss) * 1000
    overhead_per_token_us = (overhead_ms * 1000) / result_with_diss['summary']['total_tokens']

    return {
        'prompt': prompt,
        'generated_text': result_with_diss['generated_text'],
        'total_tokens': result_with_diss['summary']['total_tokens'],
        'unique_concepts': result_with_diss['summary']['unique_concepts_detected'],
        'time_without_dissonance_s': time_no_diss,
        'time_with_dissonance_s': time_with_diss,
        'overhead_ms': overhead_ms,
        'overhead_per_token_us': overhead_per_token_us,
        'tokens_with_expected_concept': result_with_diss['summary'].get('tokens_with_expected_concept', 0),
        'avg_divergence': result_with_diss['summary'].get('avg_divergence'),
        'max_divergence': result_with_diss['summary'].get('max_divergence'),
        'timesteps': result_with_diss['timesteps'],
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive dissonance tests with timing")
    parser.add_argument('--model', default='google/gemma-3-4b-pt', help='Model name')
    parser.add_argument('--device', default='cpu', help='Device')
    parser.add_argument('--max-tokens', type=int, default=20, help='Max tokens per test')
    parser.add_argument('--layers', nargs='+', type=int, default=[0, 1, 2], help='Layers')
    parser.add_argument('--threshold', type=float, default=0.4, help='Concept threshold')
    parser.add_argument('--output-dir', type=Path, default=Path('results/dissonance_tests'), help='Output directory')
    parser.add_argument('--dissonance-alpha', type=float, default=0.5, help='Divergence decay parameter')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPREHENSIVE DISSONANCE MEASUREMENT TESTS")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Device: {args.device}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print(f"Max tokens per prompt: {args.max_tokens}")

    # Load model and tokenizer (shared)
    print(f"\nLoading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=args.device,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load classifiers
    print(f"\nLoading SUMO classifiers...")
    classifiers, _ = load_sumo_classifiers(layers=args.layers, device=args.device)

    # Create monitors
    print("\nInitializing monitors...")
    monitor_without_diss = SUMOTemporalMonitor(
        classifiers,
        top_k=10,
        threshold=args.threshold,
        enable_dissonance=False,
    )

    monitor_with_diss = SUMOTemporalMonitor(
        classifiers,
        top_k=10,
        threshold=args.threshold,
        enable_dissonance=True,
        dissonance_alpha=args.dissonance_alpha,
    )

    # Run tests
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)

    results = []
    total_overhead_ms = 0

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Testing: {prompt[:60]}...")

        result = run_timed_test(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            monitor_without_diss=monitor_without_diss,
            monitor_with_diss=monitor_with_diss,
            device=args.device,
            max_tokens=args.max_tokens,
        )

        results.append(result)
        total_overhead_ms += result['overhead_ms']

        print(f"  Generated: {result['generated_text'][:60]}...")
        print(f"  Tokens: {result['total_tokens']} | Concepts: {result['unique_concepts']}")
        avg_div_str = f"{result['avg_divergence']:.3f}" if result['avg_divergence'] is not None else "N/A"
        print(f"  Mapped: {result['tokens_with_expected_concept']}/{result['total_tokens']} | "
              f"Avg Div: {avg_div_str}")
        print(f"  ⏱️  Overhead: {result['overhead_ms']:.2f}ms total, "
              f"{result['overhead_per_token_us']:.1f}μs/token")

    # Compute aggregate statistics
    print("\n" + "="*80)
    print("TIMING ANALYSIS")
    print("="*80)

    total_tokens = sum(r['total_tokens'] for r in results)
    avg_overhead_per_token_us = (total_overhead_ms * 1000) / total_tokens

    print(f"\nTotal prompts: {len(results)}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total dissonance overhead: {total_overhead_ms:.2f}ms")
    print(f"Average overhead per token: {avg_overhead_per_token_us:.1f}μs ({avg_overhead_per_token_us/1000:.4f}ms)")
    print(f"Average overhead per prompt: {total_overhead_ms/len(results):.2f}ms")

    # Divergence statistics
    print("\n" + "="*80)
    print("DIVERGENCE ANALYSIS")
    print("="*80)

    valid_divs = [r for r in results if r['avg_divergence'] is not None]
    if valid_divs:
        avg_div = sum(r['avg_divergence'] for r in valid_divs) / len(valid_divs)
        max_div = max(r['max_divergence'] for r in valid_divs)
        min_div = min(r['avg_divergence'] for r in valid_divs)

        print(f"\nPrompts with divergence data: {len(valid_divs)}/{len(results)}")
        print(f"Average divergence across all: {avg_div:.3f}")
        print(f"Min average divergence: {min_div:.3f}")
        print(f"Max divergence seen: {max_div:.3f}")

        # Top divergence examples
        print("\n--- HIGHEST DIVERGENCE TOKENS (Top 10) ---")
        all_timesteps = []
        for r in results:
            for ts in r['timesteps']:
                if ts.get('divergence') is not None:
                    all_timesteps.append({
                        'prompt': r['prompt'][:40],
                        'token': ts['token'],
                        'expected': ts['expected_concept'],
                        'divergence': ts['divergence'],
                        'top_concept': ts['concepts'][0]['concept'] if ts['concepts'] else None,
                    })

        all_timesteps.sort(key=lambda x: x['divergence'], reverse=True)
        for i, ts in enumerate(all_timesteps[:10], 1):
            print(f"{i:2d}. '{ts['token']:15s}' → {ts['expected']:20s} | "
                  f"Div: {ts['divergence']:.3f} | Got: {ts['top_concept']}")

    # Save detailed results
    output_file = args.output_dir / 'comprehensive_test_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'test_config': {
                'model': args.model,
                'device': args.device,
                'max_tokens': args.max_tokens,
                'threshold': args.threshold,
                'dissonance_alpha': args.dissonance_alpha,
            },
            'timing_summary': {
                'total_prompts': len(results),
                'total_tokens': total_tokens,
                'total_overhead_ms': total_overhead_ms,
                'avg_overhead_per_token_us': avg_overhead_per_token_us,
            },
            'divergence_summary': {
                'avg_divergence': avg_div if valid_divs else None,
                'max_divergence': max_div if valid_divs else None,
                'min_divergence': min_div if valid_divs else None,
            },
            'results': results,
        }, f, indent=2)

    print(f"\n✓ Saved detailed results to: {output_file}")

    # Save human-readable report
    report_file = args.output_dir / 'dissonance_report.txt'
    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE DISSONANCE MEASUREMENT TEST REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Layers: {args.layers}\n")
        f.write(f"Dissonance alpha: {args.dissonance_alpha}\n\n")

        f.write("TIMING ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total prompts: {len(results)}\n")
        f.write(f"Total tokens: {total_tokens}\n")
        f.write(f"Total overhead: {total_overhead_ms:.2f}ms\n")
        f.write(f"Per-token overhead: {avg_overhead_per_token_us:.1f}μs\n\n")

        f.write("PER-PROMPT RESULTS\n")
        f.write("-"*80 + "\n")
        for i, r in enumerate(results, 1):
            f.write(f"\n[{i}] {r['prompt']}\n")
            f.write(f"    Generated: {r['generated_text']}\n")
            f.write(f"    Tokens: {r['total_tokens']} | Concepts: {r['unique_concepts']}\n")
            f.write(f"    Mapped: {r['tokens_with_expected_concept']}/{r['total_tokens']}\n")
            if r['avg_divergence']:
                f.write(f"    Avg divergence: {r['avg_divergence']:.3f} | Max: {r['max_divergence']:.3f}\n")
            f.write(f"    Overhead: {r['overhead_ms']:.2f}ms ({r['overhead_per_token_us']:.1f}μs/token)\n")

            # Show high-divergence tokens for this prompt
            high_div = [ts for ts in r['timesteps']
                       if ts.get('divergence') and ts['divergence'] > 0.8]
            if high_div:
                f.write(f"    High divergence tokens:\n")
                for ts in high_div[:5]:
                    f.write(f"      '{ts['token']}' → {ts['expected_concept']} (div: {ts['divergence']:.3f})\n")

    print(f"✓ Saved report to: {report_file}")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
