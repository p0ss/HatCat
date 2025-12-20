#!/usr/bin/env python3
"""
Test adaptive training with integrated validation.

Quick test to verify the validation integration works correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.map.training.dual_adaptive_trainer import DualAdaptiveTrainer
from src.map.training.sumo_data_generation import (
    create_sumo_training_dataset,
    build_sumo_negative_pool,
)
from src.map.training.sumo_classifiers import load_layer_concepts, extract_activations


def main():
    print("=" * 80)
    print("TESTING ADAPTIVE TRAINING WITH VALIDATION")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model_name = "google/gemma-3-4b-pt"
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print("✓ Model loaded")

    # Load a single concept from layer 1
    print("\nLoading layer 1 concepts...")
    concepts, concept_map = load_layer_concepts(layer=1)

    # Pick a specific concept to test (e.g., GeologicalProcess)
    test_concept = None
    for concept in concepts:
        if concept['sumo_term'] == 'GeologicalProcess':
            test_concept = concept
            break

    if test_concept is None:
        print("GeologicalProcess not found, using first concept")
        test_concept = concepts[0]

    concept_name = test_concept['sumo_term']
    print(f"Testing with: {concept_name}")

    # Generate training data
    print("\nGenerating training data...")
    negative_pool = build_sumo_negative_pool(concepts, test_concept)

    train_prompts, train_labels = create_sumo_training_dataset(
        concept=test_concept,
        all_concepts=concept_map,
        negative_pool=negative_pool,
        n_positives=20,
        n_negatives=20,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    test_negative_pool = negative_pool[len(negative_pool) // 2:]
    test_prompts, test_labels = create_sumo_training_dataset(
        concept=test_concept,
        all_concepts=concept_map,
        negative_pool=test_negative_pool,
        n_positives=10,
        n_negatives=10,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    print(f"✓ Generated {len(train_prompts)} train, {len(test_prompts)} test prompts")

    # Extract activations
    print("\nExtracting activations...")
    X_train = extract_activations(model, tokenizer, train_prompts, device, layer_idx=15)
    X_test = extract_activations(model, tokenizer, test_prompts, device, layer_idx=15)
    print(f"✓ Extracted activations: train {X_train.shape}, test {X_test.shape}")

    # Create adaptive trainer WITH validation
    print("\nInitializing adaptive trainer with validation...")
    trainer = DualAdaptiveTrainer(
        activation_target_accuracy=0.95,
        activation_baseline=10,
        activation_increment=2,
        activation_max_samples=40,  # Smaller for quick test
        model=model,
        tokenizer=tokenizer,
        validate_lenses=True,
        validation_threshold=0.5,
        validation_layer_idx=15,
        train_activation=True,
        train_text=False,
    )
    print("✓ Trainer initialized")

    # Train concept
    print("\nTraining concept with validation...")
    results = trainer.train_concept(
        concept_name=concept_name,
        train_activations=X_train,
        train_labels=np.array(train_labels),
        test_activations=X_test,
        test_labels=np.array(test_labels),
        train_texts=None,
        test_texts=None,
    )

    # Report results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if results['activation']:
        act = results['activation']
        print(f"\nActivation lens:")
        print(f"  Samples: {act['samples']}")
        print(f"  Iterations: {act['iterations']}")
        print(f"  Test F1: {act['test_f1']:.3f}")
        print(f"  Test Precision: {act['test_precision']:.3f}")
        print(f"  Test Recall: {act['test_recall']:.3f}")

        if 'validation' in act:
            val = act['validation']
            print(f"\n  Validation:")
            print(f"    Passed: {val['passed']}")
            print(f"    Calibration score: {val['calibration_score']:.3f}")
            print(f"    Target rank: #{val['target_rank']}")
            print(f"    Avg other rank: {val['avg_other_rank']:.1f}")
            print(f"    Expected domain: {val['expected_domain']}")

            if val['passed']:
                print(f"    ✓ Lens is well-calibrated")
            else:
                print(f"    ✗ Lens failed calibration (may fire universally)")
        else:
            print("\n  ⚠️  No validation results (validation may have been skipped)")
    else:
        print("\n⚠️  Activation lens did not graduate")

    print(f"\nTotal iterations: {results['total_iterations']}")
    print(f"Total time: {results['total_time']:.2f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
