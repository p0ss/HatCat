#!/usr/bin/env python3
"""
Test token length impact on validation failures.

Hypothesis: The 20-token generation length is too short, causing lenses to learn
superficial patterns that work on test data but fail on validation prompts.

Experiment design:
1. Train the same concept (ContentBearingPhysical) with 3 different max_new_tokens
2. Compare test F1, validation score, and generalization gap
3. Measure training time for each variant

Expected outcome:
- Longer token generation → better validation scores (smaller gap)
- Longer token generation → proportionally longer training time
- Trade-off between training speed and lens quality
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

from training.sumo_data_generation import (
    load_sumo_concepts,
    load_all_concepts,
    build_sumo_training_data,
    build_sumo_test_data,
)
from training.dual_adaptive_trainer import DualAdaptiveTrainer
from training.falloff_validation import FalloffValidator
from src.hat.monitoring.lens_manager import DynamicLensManager


def train_single_concept_with_token_length(
    concept_name: str,
    layer: int,
    max_new_tokens: int,
    model_name: str = "google/gemma-2-2b-it",
    device: str = "cuda",
    n_train_pos: int = 10,
    n_train_neg: int = 10,
    n_test_pos: int = 5,
    n_test_neg: int = 5,
):
    """Train a single concept with specified token length."""

    print(f"\n{'='*80}")
    print(f"TRAINING: {concept_name} with max_new_tokens={max_new_tokens}")
    print(f"{'='*80}")

    start_time = time.time()

    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load concepts
    print(f"\nLoading SUMO concepts...")
    layer_concepts = load_sumo_concepts(layer=layer)
    all_concepts = load_all_concepts()  # For nephew negatives

    # Find target concept
    target_concept = None
    for concept in layer_concepts:
        if concept["sumo_term"] == concept_name:
            target_concept = concept
            break

    if not target_concept:
        raise ValueError(f"Concept '{concept_name}' not found in layer {layer}")

    print(f"Found concept: {concept_name}")
    print(f"  Synsets: {len(target_concept.get('synsets', []))}")
    print(f"  Children: {len(target_concept.get('category_children', []))}")

    # Build training and test data
    print(f"\nGenerating training data...")
    train_data = build_sumo_training_data(
        target_concept=target_concept,
        all_concepts=all_concepts,
        n_positive=n_train_pos,
        n_negative=n_train_neg,
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=-1,
        max_new_tokens=max_new_tokens,  # ← Key parameter!
    )

    print(f"\nGenerating test data...")
    test_data = build_sumo_test_data(
        target_concept=target_concept,
        all_concepts=all_concepts,
        n_positive=n_test_pos,
        n_negative=n_test_neg,
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=-1,
        max_new_tokens=max_new_tokens,  # ← Key parameter!
    )

    # Setup trainer with adaptive training
    print(f"\nSetting up adaptive trainer...")
    trainer = DualAdaptiveTrainer(
        input_dim=model.config.hidden_size,
        learning_rate=0.001,
        max_cycles=10,
        min_samples=10,
        first_increment=20,
        subsequent_increment=30,
    )

    # Setup validation
    print(f"\nSetting up falloff validation...")
    validator = FalloffValidator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=-1,
        max_new_tokens=max_new_tokens,  # ← Key parameter!
    )

    # Train with adaptive cycles
    print(f"\nStarting adaptive training...")
    training_start = time.time()

    cycle_results = []
    graduated = False

    for cycle in range(trainer.max_cycles):
        required_samples = trainer.get_required_samples(cycle)

        print(f"\n{'-'*80}")
        print(f"Cycle {cycle}: Generating {required_samples} samples")

        # Generate data for this cycle
        cycle_train = build_sumo_training_data(
            target_concept=target_concept,
            all_concepts=all_concepts,
            n_positive=required_samples,
            n_negative=required_samples,
            model=model,
            tokenizer=tokenizer,
            device=device,
            layer_idx=-1,
            max_new_tokens=max_new_tokens,
        )

        # Train
        print(f"Training lens...")
        train_result = trainer.train_cycle(
            activations_pos=cycle_train["activations_pos"],
            activations_neg=cycle_train["activations_neg"],
        )

        # Test
        print(f"Testing lens...")
        test_metrics = trainer.test(
            activations_pos=test_data["activations_pos"],
            activations_neg=test_data["activations_neg"],
        )

        print(f"Test metrics: F1={test_metrics.get('f1', 0):.3f}, "
              f"Precision={test_metrics.get('precision', 0):.3f}, "
              f"Recall={test_metrics.get('recall', 0):.3f}")

        # Validate
        print(f"Running falloff validation...")
        val_result = validator.validate_lens(
            lens=trainer.get_lens(),
            concept=target_concept,
            all_concepts=all_concepts,
            cycle=cycle,
        )

        print(f"Validation score: {val_result['score']:.3f} "
              f"(threshold: {val_result['threshold']:.3f})")

        cycle_results.append({
            "cycle": cycle,
            "samples": required_samples,
            "test_f1": test_metrics.get("f1", 0),
            "test_precision": test_metrics.get("precision", 0),
            "test_recall": test_metrics.get("recall", 0),
            "validation_score": val_result["score"],
            "validation_threshold": val_result["threshold"],
            "validation_passed": val_result["passed"],
        })

        # Check graduation
        if val_result["passed"]:
            print(f"\n✓ Concept graduated at cycle {cycle}!")
            graduated = True
            break

    training_time = time.time() - training_start
    total_time = time.time() - start_time

    # Final results
    final_cycle = cycle_results[-1] if cycle_results else {}

    result = {
        "concept": concept_name,
        "layer": layer,
        "max_new_tokens": max_new_tokens,
        "graduated": graduated,
        "total_cycles": len(cycle_results),
        "final_test_f1": final_cycle.get("test_f1", 0),
        "final_validation_score": final_cycle.get("validation_score", 0),
        "generalization_gap": final_cycle.get("test_f1", 0) - final_cycle.get("validation_score", 0),
        "training_time_seconds": training_time,
        "total_time_seconds": total_time,
        "cycle_history": cycle_results,
    }

    print(f"\n{'='*80}")
    print(f"RESULTS FOR max_new_tokens={max_new_tokens}")
    print(f"{'='*80}")
    print(f"Graduated: {graduated}")
    print(f"Cycles: {len(cycle_results)}")
    print(f"Final test F1: {result['final_test_f1']:.3f}")
    print(f"Final validation: {result['final_validation_score']:.3f}")
    print(f"Generalization gap: {result['generalization_gap']:.3f}")
    print(f"Training time: {training_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")

    return result


def main():
    """Run token length experiment."""

    print("="*80)
    print("TOKEN LENGTH EXPERIMENT")
    print("="*80)
    print("\nHypothesis: 20-token generation is too short, causing overfitting")
    print("Test subject: Carnivore (Layer 2)")
    print("  - Has 8 synsets (not a synset count issue)")
    print("  - Test F1=1.000, Validation=0.000 (perfect overfitting)")
    print("  - Ideal for isolating token length effect")
    print("\nVariants (testing both directions):")
    print("  A: max_new_tokens=10 (0.5x baseline - should worsen gap)")
    print("  B: max_new_tokens=20 (current baseline)")
    print("  C: max_new_tokens=40 (2x baseline - should improve gap)")
    print("\nKey test: If gap worsens at 10 tokens, strengthens hypothesis")

    # Experiment parameters
    concept_name = "Carnivore"
    layer = 2
    token_lengths = [10, 20, 40]

    # Run experiments
    results = []

    for max_tokens in token_lengths:
        print(f"\n\n{'#'*80}")
        print(f"VARIANT: max_new_tokens={max_tokens}")
        print(f"{'#'*80}")

        try:
            result = train_single_concept_with_token_length(
                concept_name=concept_name,
                layer=layer,
                max_new_tokens=max_tokens,
                device="cuda",
                n_train_pos=10,
                n_train_neg=10,
                n_test_pos=5,
                n_test_neg=5,
            )
            results.append(result)
        except Exception as e:
            print(f"\n✗ Error training with max_new_tokens={max_tokens}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_dir = PROJECT_ROOT / "results" / "token_length_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_{timestamp}.json"

    experiment_data = {
        "experiment": "token_length_impact",
        "concept": concept_name,
        "layer": layer,
        "timestamp": timestamp,
        "variants": results,
    }

    with open(output_file, "w") as f:
        json.dump(experiment_data, f, indent=2)

    print(f"\n\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_file}")

    print(f"\n{'Tokens':<10} {'Graduated':<12} {'Cycles':<10} {'Test F1':<10} {'Val Score':<12} {'Gap':<10} {'Time (s)':<12}")
    print("-" * 88)

    for r in results:
        print(f"{r['max_new_tokens']:<10} "
              f"{'Yes' if r['graduated'] else 'No':<12} "
              f"{r['total_cycles']:<10} "
              f"{r['final_test_f1']:<10.3f} "
              f"{r['final_validation_score']:<12.3f} "
              f"{r['generalization_gap']:<10.3f} "
              f"{r['training_time_seconds']:<12.1f}")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    if len(results) >= 2:
        baseline = results[0]  # 20 tokens

        for variant in results[1:]:
            gap_improvement = baseline['generalization_gap'] - variant['generalization_gap']
            time_ratio = variant['training_time_seconds'] / baseline['training_time_seconds']

            print(f"\n{variant['max_new_tokens']} tokens vs {baseline['max_new_tokens']} tokens:")
            print(f"  Gap improvement: {gap_improvement:+.3f} ({-gap_improvement/baseline['generalization_gap']*100:+.1f}%)")
            print(f"  Time ratio: {time_ratio:.2f}x")
            print(f"  Graduated: {baseline['graduated']} → {variant['graduated']}")

            if gap_improvement > 0.1 and time_ratio < 3:
                print(f"  ✓ SIGNIFICANT IMPROVEMENT - Worth the time cost!")
            elif gap_improvement > 0.05:
                print(f"  ? MODERATE IMPROVEMENT - Consider trade-off")
            else:
                print(f"  ✗ MINIMAL IMPROVEMENT - Not worth the time cost")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
