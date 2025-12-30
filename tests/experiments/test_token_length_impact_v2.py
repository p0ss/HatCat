#!/usr/bin/env python3
"""
Test token length impact on validation failures - Simplified version.

Uses existing training infrastructure to test Carnivore at 3 token lengths.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from training.sumo_classifiers import (
    load_layer_concepts,
    load_all_concepts,
    extract_activations,
)
from training.sumo_data_generation import (
    create_sumo_training_dataset,
    build_sumo_negative_pool,
)
from training.dual_adaptive_trainer import DualAdaptiveTrainer
from training.falloff_validation import FalloffValidator


def train_concept_at_token_length(
    concept_name: str,
    layer: int,
    max_new_tokens: int,
    model_name: str = "google/gemma-2-2b-it",
    device: str = "cuda",
):
    """Train a single concept with specified token length."""

    print(f"\n{'='*80}")
    print(f"TRAINING: {concept_name} with max_new_tokens={max_new_tokens}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load concepts
    print(f"Loading Layer {layer} concepts...")
    layer_concepts, concept_map = load_layer_concepts(layer)
    all_concepts = load_all_concepts()

    target_concept = concept_map[concept_name]
    print(f"Found: {concept_name}")
    print(f"  Synsets: {len(target_concept.get('synsets', []))}")
    print(f"  Children: {len(target_concept.get('category_children', []))}")

    # Build negative pool
    print("\nBuilding negative pool...")
    negative_pool = build_sumo_negative_pool(
        target_concept=target_concept,
        all_concepts=all_concepts,
    )
    print(f"  Negative pool: {len(negative_pool)} concepts")

    # Setup trainer
    print("\nInitializing adaptive trainer...")
    trainer = DualAdaptiveTrainer(
        input_dim=model.config.hidden_size,
        learning_rate=0.001,
        max_cycles=10,
        min_samples=10,
        first_increment=20,
        subsequent_increment=30,
    )

    # Setup validator
    print("Initializing falloff validator...")
    validator = FalloffValidator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=-1,
        max_new_tokens=max_new_tokens,  # Use specified token length
    )

    # Training loop
    cycle_results = []
    graduated = False

    for cycle in range(trainer.max_cycles):
        required_samples = trainer.get_required_samples(cycle)

        print(f"\n{'-'*80}")
        print(f"Cycle {cycle}: {required_samples} samples")
        cycle_start = time.time()

        # Generate training data
        print(f"  Generating data...")
        dataset = create_sumo_training_dataset(
            target_concept=target_concept,
            negative_pool=negative_pool,
            n_positive=required_samples,
            n_negative=required_samples,
        )

        # Extract activations
        print(f"  Extracting activations...")
        extract_start = time.time()

        pos_acts = extract_activations(
            model=model,
            tokenizer=tokenizer,
            prompts=dataset["positive_prompts"],
            device=device,
            layer_idx=-1,
            max_new_tokens=max_new_tokens,  # Use specified token length
        )

        neg_acts = extract_activations(
            model=model,
            tokenizer=tokenizer,
            prompts=dataset["negative_prompts"],
            device=device,
            layer_idx=-1,
            max_new_tokens=max_new_tokens,  # Use specified token length
        )

        extract_time = time.time() - extract_start
        print(f"  Extraction: {extract_time:.1f}s")

        # Train
        print(f"  Training lens...")
        train_start = time.time()
        train_result = trainer.train_cycle(
            activations_pos=pos_acts,
            activations_neg=neg_acts,
        )
        train_time = time.time() - train_start
        print(f"  Training: {train_time:.3f}s")

        # Test
        print(f"  Testing...")
        test_result = trainer.test(
            activations_pos=pos_acts[:5],  # Use subset for test
            activations_neg=neg_acts[:5],
        )
        print(f"  Test F1: {test_result.get('f1', 0):.3f}")

        # Validate
        print(f"  Validating...")
        val_result = validator.validate_lens(
            lens=trainer.get_lens(),
            concept=target_concept,
            all_concepts=all_concepts,
            cycle=cycle,
        )
        print(f"  Validation: {val_result['score']:.3f} (threshold: {val_result['threshold']:.3f})")

        cycle_time = time.time() - cycle_start

        cycle_results.append({
            "cycle": cycle,
            "samples": required_samples,
            "test_f1": test_result.get("f1", 0),
            "validation_score": val_result["score"],
            "validation_passed": val_result["passed"],
            "extract_time": extract_time,
            "train_time": train_time,
            "cycle_time": cycle_time,
        })

        if val_result["passed"]:
            print(f"\n✓ Graduated at cycle {cycle}!")
            graduated = True
            break

    total_time = time.time() - start_time
    final = cycle_results[-1] if cycle_results else {}

    result = {
        "concept": concept_name,
        "layer": layer,
        "max_new_tokens": max_new_tokens,
        "graduated": graduated,
        "total_cycles": len(cycle_results),
        "final_test_f1": final.get("test_f1", 0),
        "final_validation": final.get("validation_score", 0),
        "generalization_gap": final.get("test_f1", 0) - final.get("validation_score", 0),
        "total_time": total_time,
        "cycle_history": cycle_results,
    }

    print(f"\n{'='*80}")
    print(f"RESULTS: max_new_tokens={max_new_tokens}")
    print(f"{'='*80}")
    print(f"Graduated: {graduated}")
    print(f"Cycles: {len(cycle_results)}")
    print(f"Test F1: {result['final_test_f1']:.3f}")
    print(f"Validation: {result['final_validation']:.3f}")
    print(f"Gap: {result['generalization_gap']:.3f}")
    print(f"Total time: {total_time:.1f}s")

    return result


def main():
    print("="*80)
    print("TOKEN LENGTH EXPERIMENT")
    print("="*80)
    print("\nTest subject: Carnivore (Layer 2)")
    print("Variants: [10, 20, 40] tokens")
    print("\nExpected: Gap should decrease as tokens increase")

    concept_name = "Carnivore"
    layer = 2
    token_lengths = [10, 20, 40]

    results = []

    for max_tokens in token_lengths:
        print(f"\n\n{'#'*80}")
        print(f"VARIANT: {max_tokens} tokens")
        print(f"{'#'*80}")

        try:
            result = train_concept_at_token_length(
                concept_name=concept_name,
                layer=layer,
                max_new_tokens=max_tokens,
                device="cuda",
            )
            results.append(result)
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_dir = PROJECT_ROOT / "results" / "token_length_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "concept": concept_name,
            "layer": layer,
            "timestamp": timestamp,
            "results": results,
        }, f, indent=2)

    # Summary
    print(f"\n\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Results: {output_file}")

    print(f"\n{'Tokens':<10} {'Grad':<8} {'Cycles':<10} {'Test F1':<10} {'Val':<10} {'Gap':<10} {'Time (s)':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['max_new_tokens']:<10} "
              f"{'Yes' if r['graduated'] else 'No':<8} "
              f"{r['total_cycles']:<10} "
              f"{r['final_test_f1']:<10.3f} "
              f"{r['final_validation']:<10.3f} "
              f"{r['generalization_gap']:<10.3f} "
              f"{r['total_time']:<12.1f}")

    # Analysis
    if len(results) >= 2:
        print(f"\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}")

        baseline = results[1]  # 20 tokens

        for i, r in enumerate(results):
            if r['max_new_tokens'] == 20:
                continue

            gap_change = baseline['generalization_gap'] - r['generalization_gap']
            time_ratio = r['total_time'] / baseline['total_time']

            print(f"\n{r['max_new_tokens']} vs 20 tokens:")
            print(f"  Gap change: {gap_change:+.3f}")
            print(f"  Time ratio: {time_ratio:.2f}x")

            if abs(gap_change) > 0.1:
                print(f"  ✓ SIGNIFICANT difference")
            elif abs(gap_change) > 0.05:
                print(f"  ? MODERATE difference")
            else:
                print(f"  ✗ MINIMAL difference")


if __name__ == "__main__":
    main()
