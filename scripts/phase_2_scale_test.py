"""
Phase 2: Scale Test with Minimal Training
==========================================

Goal: Test 1×1 training at different concept scales to understand
the impact of concept count vs training depth.

Test at: 1, 10, 100, 1000, 10000 concepts
Training: 1×1 (minimal) for all concepts
Evaluation: Fixed test set accuracy

This tells us:
- How does accuracy degrade with scale at minimal training?
- Should we prioritize depth (10×10) or breadth (1×1 for more concepts)?
- Where do we hit diminishing returns?

Expected runtime:
- 10 concepts × 1×1: ~3 minutes
- 100 concepts × 1×1: ~30 minutes
- 1000 concepts × 1×1: ~5 hours
- 10000 concepts × 1×1: ~50 hours (overnight)
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import argparse
import json
import time
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.stage_1_5_temporal import get_activation_sequence
from scripts.phase_1_find_curve_v2 import sample_test_set, sample_train_sequences, train_and_evaluate


def test_scale(
    model,
    tokenizer,
    concept_graph_path: Path,
    output_dir: Path,
    n_concepts: int,
    n_defs: int = 1,
    n_rels: int = 1,
    device: str = "cuda",
    model_name: str = "unknown"
):
    """
    Test a specific scale with fixed training config.

    Args:
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        n_concepts: Number of concepts to test
        n_defs: Definitions per concept (default: 1)
        n_rels: Relationships per concept (default: 1)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PHASE 2: SCALE TEST")
    print(f"{'='*70}")
    print(f"Concepts: {n_concepts}")
    print(f"Training: {n_defs}×{n_rels} per concept")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Load concept graph
    with open(concept_graph_path) as f:
        concept_data = json.load(f)

    all_concepts = list(concept_data.keys())
    concepts = all_concepts[:n_concepts]

    print(f"Testing {len(concepts)} concepts from {len(all_concepts)} available\n")

    # Get hidden dim
    with torch.inference_mode():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_output = model(**test_input, output_hidden_states=True)
        hidden_dim = test_output.hidden_states[-1].shape[-1]

    print(f"Hidden dim: {hidden_dim}\n")

    # Test each concept
    results = []

    for i, concept in enumerate(concepts):
        print(f"[{i+1}/{len(concepts)}] {concept}...", end=" ", flush=True)

        concept_start = time.time()

        try:
            negatives = concept_data[concept].get('negatives', [])
            related_structured = concept_data[concept].get('related_structured', {})

            if len(negatives) == 0:
                print("✗ No negatives, skipping", flush=True)
                continue

            # Generate test set (fixed)
            test_pos, test_neg = sample_test_set(
                model, tokenizer, concept, negatives, related_structured,
                n_samples=10, layer_idx=-1, device=device
            )

            # Generate training data
            train_pos, train_neg = sample_train_sequences(
                model, tokenizer, concept, negatives, related_structured,
                n_defs, n_rels, layer_idx=-1, device=device
            )

            # Train and evaluate
            train_acc, test_acc = train_and_evaluate(
                train_pos, train_neg, test_pos, test_neg, hidden_dim
            )

            concept_elapsed = time.time() - concept_start

            result = {
                'concept': concept,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'elapsed_seconds': concept_elapsed
            }

            results.append(result)

            print(f"✓ {test_acc:.1%} ({concept_elapsed:.1f}s)", flush=True)

        except Exception as e:
            print(f"✗ Failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

        # Clear cache
        torch.cuda.empty_cache()

        # Save checkpoint every 10 concepts
        if (i + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_{i+1}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'n_completed': len(results),
                    'n_total': len(concepts),
                    'results': results
                }, f, indent=2)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")

    # Compute statistics
    total_elapsed = time.time() - start_time

    if results:
        test_accs = [r['test_acc'] for r in results]
        train_accs = [r['train_acc'] for r in results]

        stats = {
            'mean_test_acc': np.mean(test_accs),
            'median_test_acc': np.median(test_accs),
            'std_test_acc': np.std(test_accs),
            'min_test_acc': np.min(test_accs),
            'max_test_acc': np.max(test_accs),
            'mean_train_acc': np.mean(train_accs),
            'n_above_80': sum(1 for acc in test_accs if acc >= 0.80),
            'n_above_90': sum(1 for acc in test_accs if acc >= 0.90),
            'n_above_95': sum(1 for acc in test_accs if acc >= 0.95),
        }
    else:
        stats = {}

    # Save final results
    output = {
        'config': {
            'n_concepts': n_concepts,
            'n_defs': n_defs,
            'n_rels': n_rels,
            'model': model_name
        },
        'statistics': stats,
        'results': results,
        'total_time_seconds': total_elapsed,
        'successful_concepts': len(results),
        'failed_concepts': n_concepts - len(results)
    }

    output_path = output_dir / f"scale_test_n{n_concepts}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("SCALE TEST COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Successful: {len(results)}/{n_concepts} concepts")
    print()

    if results:
        print("Test Accuracy Statistics:")
        print(f"  Mean:   {stats['mean_test_acc']:.1%}")
        print(f"  Median: {stats['median_test_acc']:.1%}")
        print(f"  Std:    {stats['std_test_acc']:.1%}")
        print(f"  Range:  {stats['min_test_acc']:.1%} - {stats['max_test_acc']:.1%}")
        print()
        print("Accuracy Distribution:")
        print(f"  ≥80%: {stats['n_above_80']}/{len(results)} ({stats['n_above_80']/len(results):.1%})")
        print(f"  ≥90%: {stats['n_above_90']}/{len(results)} ({stats['n_above_90']/len(results):.1%})")
        print(f"  ≥95%: {stats['n_above_95']}/{len(results)} ({stats['n_above_95']/len(results):.1%})")

    print()
    print(f"Results saved to: {output_path}")
    print()


def run_all_scales(
    concept_graph_path: Path,
    model_name: str,
    output_root: Path,
    device: str = "cuda"
):
    """Run tests at all scales: 1, 10, 100, 1000, 10000."""
    scales = [1, 10, 100, 1000, 10000]

    print(f"\n{'='*70}")
    print("PHASE 2: MULTI-SCALE TEST")
    print(f"{'='*70}")
    print(f"Scales: {scales}")
    print(f"Training: 1×1 per concept")
    print(f"{'='*70}\n")

    # Load model once for all scales
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded\n")

    for n in scales:
        output_dir = output_root / f"scale_n{n}"

        print(f"\n{'='*70}")
        print(f"Starting scale test: {n} concepts")
        print(f"{'='*70}\n")

        try:
            test_scale(
                model=model,
                tokenizer=tokenizer,
                concept_graph_path=concept_graph_path,
                output_dir=output_dir,
                n_concepts=n,
                n_defs=1,
                n_rels=1,
                device=device,
                model_name=model_name
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print(f"Completed up to {n} concepts scale")
            break
        except Exception as e:
            print(f"\n\n✗ Scale test {n} failed: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Phase 2: Test 1×1 training at different concept scales"
    )

    parser.add_argument('--concept-graph', type=str, required=True,
                       help='Path to concept graph JSON')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model to use')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--n-concepts', type=int, default=None,
                       help='Number of concepts (default: run all scales)')
    parser.add_argument('--n-defs', type=int, default=1,
                       help='Definitions per concept (default: 1)')
    parser.add_argument('--n-rels', type=int, default=1,
                       help='Relationships per concept (default: 1)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    concept_graph_path = Path(args.concept_graph)
    output_dir = Path(args.output_dir)

    if args.n_concepts:
        # Single scale test - load model once
        print(f"Loading {args.model}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map=args.device
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✓ Model loaded\n")

        test_scale(
            model=model,
            tokenizer=tokenizer,
            concept_graph_path=concept_graph_path,
            output_dir=output_dir,
            n_concepts=args.n_concepts,
            n_defs=args.n_defs,
            n_rels=args.n_rels,
            device=args.device,
            model_name=args.model
        )
    else:
        # Run all scales
        run_all_scales(
            concept_graph_path=concept_graph_path,
            model_name=args.model,
            output_root=output_dir,
            device=args.device
        )
