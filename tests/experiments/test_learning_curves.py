#!/usr/bin/env python3
"""
Quick test to verify learning curve improvements work correctly.
Tests on one concept from Layer 0.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.map.training.sumo_classifiers import load_layer_concepts, build_sumo_negative_pool, create_sumo_training_dataset, extract_activations
from src.map.training.dual_adaptive_trainer import DualAdaptiveTrainer
import numpy as np

def main():
    print("=" * 80)
    print("TESTING LEARNING CURVE IMPROVEMENTS")
    print("=" * 80)
    print()

    # Load model
    model_name = "google/gemma-3-4b-pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    )
    print("✓ Model loaded")
    print()

    # Load Layer 0, test on Physical
    concepts, concept_map = load_layer_concepts(0)
    concept = next(c for c in concepts if c['sumo_term'] == 'Physical')

    print(f"Testing concept: Physical")
    print()

    # Generate training data
    negative_pool = build_sumo_negative_pool(concepts, concept)
    train_prompts, train_labels = create_sumo_training_dataset(
        concept=concept,
        all_concepts=concept_map,
        negative_pool=negative_pool,
        n_positives=10,
        n_negatives=9,  # One less than needed
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    test_prompts, test_labels = create_sumo_training_dataset(
        concept=concept,
        all_concepts=concept_map,
        negative_pool=negative_pool[len(negative_pool)//2:],
        n_positives=20,
        n_negatives=20,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    print(f"Generated {len(train_prompts)} train, {len(test_prompts)} test prompts")
    print()

    # Extract activations
    print("Extracting activations...")
    X_train = extract_activations(model, tokenizer, train_prompts, device)
    X_test = extract_activations(model, tokenizer, test_prompts, device)
    print("✓ Activations extracted")
    print()

    # Train with adaptive trainer
    print("Training with learning curve tracking...")
    print()

    adaptive_trainer = DualAdaptiveTrainer(
        activation_target_accuracy=0.95,
        activation_baseline=10,
        activation_increment=1,
        activation_max_samples=30,
        max_iterations=10,
        model=model,
        tokenizer=tokenizer,
        validate_lenses=False,  # Disable for quick test
        train_activation=True,
        train_text=False,
    )

    results = adaptive_trainer.train_concept(
        concept_name='Physical',
        train_activations=X_train,
        train_labels=np.array(train_labels),
        test_activations=X_test,
        test_labels=np.array(test_labels),
        train_texts=None,
        test_texts=None,
    )

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    if results['activation']:
        act = results['activation']
        print(f"✓ Graduated after {act['iterations']} iterations")
        print(f"  Samples: {act['samples']}")
        print(f"  Train F1: {act.get('train_f1', 'N/A'):.3f}")
        print(f"  Test F1: {act['test_f1']:.3f}")
        print(f"  Overfit gap: {act.get('overfit_gap', 'N/A'):.3f}")
    else:
        print("✗ Did not graduate")

    print()
    print("Expected output:")
    print("  - Iter 1-2: Should NOT graduate (min 3 iterations)")
    print("  - Iter 3+: Can graduate if test_f1 ≥ 0.95 and gap ≤ 0.10")
    print("  - Learning curve: train_f1, test_f1, gap printed each iteration")

    return 0

if __name__ == '__main__':
    sys.exit(main())
