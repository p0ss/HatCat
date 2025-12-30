#!/usr/bin/env python3
"""
Retrain motivational_regulation with increased PATIENCE.

The previous run showed F1 still climbing when early stopping kicked in at epoch 53 (F1=0.736).
This run uses PATIENCE=30 to allow more convergence time.
"""

import json
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.sumo_data_generation import create_simplex_pole_training_dataset_contrastive
from training.sumo_classifiers import extract_activations
from training.tripole_classifier import train_tripole_simplex

# Paths
S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"
EXISTING_RUN_DIR = PROJECT_ROOT / "results" / "s_tier_tripole_lazy" / "run_20251125_094653"

# Training configuration
BEHAVIORAL_RATIO = 0.6
LAYER_IDX = 12

# Lazy generation parameters
SAMPLES_PER_POLE_INITIAL = 60
SAMPLES_PER_POLE_INCREMENT = 60
SAMPLES_PER_POLE_MAX = 300
MIN_F1_THRESHOLD = 0.80

# Training hyperparameters - INCREASED PATIENCE
MAX_EPOCHS = 200
PATIENCE = 30  # Increased from 10 to 30
LEARNING_RATE = 1e-3
LAMBDA_MARGIN = 0.5
LAMBDA_ORTHO = 1e-4


def load_motivational_regulation_simplex():
    """Load motivational_regulation simplex definition."""
    with open(S_TIER_DEFS_PATH) as f:
        s_tier_defs = json.load(f)

    simplex_def = s_tier_defs['simplexes']['motivational_regulation']
    simplex = {
        'simplex_dimension': 'motivational_regulation',
        'three_pole_simplex': {
            'negative_pole': simplex_def['negative_pole'],
            'neutral_homeostasis': simplex_def['neutral_homeostasis'],
            'positive_pole': simplex_def['positive_pole']
        }
    }
    return simplex


def generate_tripole_data(
    simplex: dict,
    n_samples_per_pole: int,
    model,
    tokenizer,
    device: str,
    layer_idx: int
):
    """Generate balanced 3-class training data."""
    dimension = simplex['simplex_dimension']
    three_pole = simplex['three_pole_simplex']

    all_prompts = []
    tripole_labels = []
    pole_index_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    print(f"  Generating balanced data for all 3 poles...")

    for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
        pole_data = three_pole[pole_name]
        pole_type = pole_name.split('_')[0]

        # Get other poles for contrastive learning
        other_pole_names = [p for p in ['negative_pole', 'neutral_homeostasis', 'positive_pole'] if p != pole_name]
        other_poles_data = [
            {**three_pole[p], 'pole_type': p.split('_')[0]}
            for p in other_pole_names
        ]

        # Generate contrastive dataset
        prompts, binary_labels = create_simplex_pole_training_dataset_contrastive(
            pole_data=pole_data,
            pole_type=pole_type,
            dimension=dimension,
            other_poles_data=other_poles_data,
            behavioral_ratio=BEHAVIORAL_RATIO,
            prompts_per_synset=5
        )

        # Extract only POSITIVE examples for this pole
        pole_idx = pole_index_map[pole_type]
        positive_prompts = [p for p, label in zip(prompts, binary_labels) if label == 1]

        # Downsample to target if needed
        if len(positive_prompts) > n_samples_per_pole:
            import random
            random.shuffle(positive_prompts)
            positive_prompts = positive_prompts[:n_samples_per_pole]

        # Add to dataset with pole label
        all_prompts.extend(positive_prompts)
        tripole_labels.extend([pole_idx] * len(positive_prompts))

        print(f"    [{pole_type.upper()}] {len(positive_prompts)} samples (target: {n_samples_per_pole})")

    # Extract activations
    print(f"  Extracting activations at layer {layer_idx}...")
    activations = extract_activations(model, tokenizer, all_prompts, device, layer_idx)

    # Handle combined extraction (doubles activations)
    if activations.shape[0] == 2 * len(all_prompts):
        tripole_labels_expanded = []
        for label in tripole_labels:
            tripole_labels_expanded.append(label)
            tripole_labels_expanded.append(label)
        tripole_labels = tripole_labels_expanded
        print(f"    Combined extraction: duplicated labels")

    # Convert to tensors
    activations = torch.tensor(activations, dtype=torch.float32)
    labels_tensor = torch.tensor(tripole_labels, dtype=torch.long)

    # Count samples per pole
    pole_counts = {}
    for pole_type, idx in pole_index_map.items():
        count = sum(1 for l in tripole_labels if l == idx)
        pole_counts[pole_type] = count

    print(f"  Total: {activations.shape[0]} samples")

    return activations, labels_tensor, pole_counts


def train_with_lazy_generation(
    simplex: dict,
    model,
    tokenizer,
    device: str,
    run_dir: Path,
    layer_idx: int = 12
):
    """Train with lazy data generation and increased patience."""
    dimension = simplex['simplex_dimension']

    print(f"\n  Training {dimension} with lazy generation (PATIENCE={PATIENCE})...")

    n_samples = SAMPLES_PER_POLE_INITIAL
    iteration = 0
    best_f1 = 0.0
    best_lens = None
    best_history = None

    while n_samples <= SAMPLES_PER_POLE_MAX:
        iteration += 1
        print(f"\n  Iteration {iteration}: {n_samples} samples per pole ({n_samples * 3} total)")

        # Generate data
        all_activations, all_labels, pole_counts = generate_tripole_data(
            simplex=simplex,
            n_samples_per_pole=n_samples,
            model=model,
            tokenizer=tokenizer,
            device=device,
            layer_idx=layer_idx
        )

        # Train/test split (80/20)
        n_total = all_activations.shape[0]
        n_train = int(n_total * 0.8)

        # Shuffle
        indices = torch.randperm(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_activations = all_activations[train_indices]
        train_labels = all_labels[train_indices]
        test_activations = all_activations[test_indices]
        test_labels = all_labels[test_indices]

        print(f"  Train: {train_activations.shape[0]}, Test: {test_activations.shape[0]}")

        # Train tripole lens with increased patience
        lens, history = train_tripole_simplex(
            train_activations=train_activations,
            train_labels=train_labels,
            test_activations=test_activations,
            test_labels=test_labels,
            hidden_dim=train_activations.shape[1],
            device=device,
            lr=LEARNING_RATE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,  # Using increased patience
            lambda_margin=LAMBDA_MARGIN,
            lambda_ortho=LAMBDA_ORTHO,
        )

        test_f1 = history['best_test_f1']
        print(f"  Test F1: {test_f1:.3f}")

        # Track best
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_lens = lens
            best_history = history

        # Check if graduated
        if test_f1 >= MIN_F1_THRESHOLD:
            print(f"  ✓ Graduated! (F1 {test_f1:.3f} >= {MIN_F1_THRESHOLD})")
            break

        # Check if we should continue
        if n_samples >= SAMPLES_PER_POLE_MAX:
            print(f"  ✗ Reached max samples ({SAMPLES_PER_POLE_MAX}) without graduating")
            break

        # Increment dataset size
        n_samples += SAMPLES_PER_POLE_INCREMENT
        print(f"  F1 {test_f1:.3f} < {MIN_F1_THRESHOLD}, increasing to {n_samples} samples per pole...")

    # Save results
    print(f"\n  Saving results...")

    # Save lens (best across all iterations)
    if best_lens is not None:
        lens_file = run_dir / "tripole_lens.pt"
        torch.save(best_lens.state_dict(), lens_file)
        print(f"    ✓ Lens saved: {lens_file}")

    # Save metrics
    results = {
        'dimension': dimension,
        'best_test_f1': best_f1,
        'graduated': best_f1 >= MIN_F1_THRESHOLD,
        'total_iterations': iteration,
        'final_samples_per_pole': n_samples if n_samples <= SAMPLES_PER_POLE_MAX else SAMPLES_PER_POLE_MAX,
        'final_metrics': best_history['final_metrics'] if best_history else {},
        'pole_counts': pole_counts,
        'patience_used': PATIENCE
    }

    results_file = run_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    ✓ Results saved: {results_file}")

    return results


def main():
    print("=" * 80)
    print("RETRAIN motivational_regulation WITH INCREASED PATIENCE")
    print("=" * 80)
    print(f"PATIENCE increased from 10 to {PATIENCE}")
    print(f"Previous best: F1=0.736 (still climbing when stopped)")

    # Load simplex
    print("\n1. Loading motivational_regulation simplex...")
    simplex = load_motivational_regulation_simplex()
    print(f"   ✓ Loaded simplex")

    # Setup logging
    simplex_dir = EXISTING_RUN_DIR / "motivational_regulation"
    simplex_dir.mkdir(parents=True, exist_ok=True)

    log_file = simplex_dir / "retrain_patience30.log"

    class TeeLogger:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_handle = open(log_file, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(original_stdout, log_handle)
    sys.stderr = TeeLogger(original_stderr, log_handle)

    print(f"\n2. Log file: {log_file}")

    # Load model
    print("\n3. Loading model...")
    model_name = "google/gemma-3-4b-pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print(f"   ✓ Model loaded on {device}")

    # Train
    print(f"\n4. Training motivational_regulation...")

    try:
        results = train_with_lazy_generation(
            simplex=simplex,
            model=model,
            tokenizer=tokenizer,
            device=device,
            run_dir=simplex_dir,
            layer_idx=LAYER_IDX
        )

        # Update main results.json
        print("\n5. Updating main results.json...")
        main_results_file = EXISTING_RUN_DIR / "results.json"

        with open(main_results_file, 'r') as f:
            main_results = json.load(f)

        # Update graduated/failed lists
        if results['graduated']:
            if 'motivational_regulation' not in main_results['graduated']:
                main_results['graduated'].append('motivational_regulation')
            if 'motivational_regulation' in main_results['failed']:
                main_results['failed'].remove('motivational_regulation')
        else:
            if 'motivational_regulation' not in main_results['failed']:
                main_results['failed'].append('motivational_regulation')

        # Update simplex results
        found = False
        for i, existing in enumerate(main_results['simplexes']):
            if existing['dimension'] == 'motivational_regulation':
                main_results['simplexes'][i] = results
                found = True
                break
        if not found:
            main_results['simplexes'].append(results)

        with open(main_results_file, 'w') as f:
            json.dump(main_results, f, indent=2)

        print(f"   ✓ Updated: {main_results_file}")

        # Final summary
        print("\n" + "=" * 80)
        print("RETRAINING COMPLETE")
        print("=" * 80)

        if results['graduated']:
            print(f"\n✓ motivational_regulation GRADUATED!")
            print(f"  Final F1: {results['best_test_f1']:.3f}")
            print(f"  Total iterations: {results['total_iterations']}")
            print(f"  Samples per pole: {results['final_samples_per_pole']}")

            total_graduated = len(main_results['graduated'])
            print(f"\n  Overall: {total_graduated}/13 simplexes graduated")
        else:
            print(f"\n✗ motivational_regulation still failed")
            print(f"  Best F1: {results['best_test_f1']:.3f}")
            print(f"  (threshold: {MIN_F1_THRESHOLD})")

    except Exception as e:
        print(f"\n✗ Failed with error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n✓ Results saved to: {simplex_dir}")
    print("=" * 80)

    # Restore stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()

    print(f"\n✓ Training log saved to: {log_file}")


if __name__ == "__main__":
    main()
