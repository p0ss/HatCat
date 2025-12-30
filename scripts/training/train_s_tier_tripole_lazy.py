#!/usr/bin/env python3
"""
Train S-tier three-pole simplex lenses using proper joint tripole architecture with lazy generation.

Uses TripoleLens with joint 3-class softmax (not binary classifiers).
Implements lazy data generation: start with 60 samples per pole, increment if needed.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.sumo_data_generation import create_simplex_pole_training_dataset_contrastive
from training.sumo_classifiers import extract_activations
from training.tripole_classifier import train_tripole_simplex, TripoleLens

# Paths
S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "s_tier_tripole_lazy"

# Training configuration
BEHAVIORAL_RATIO = 0.6  # 60% behavioral, 40% definitional
LAYER_IDX = 12  # Layer to extract activations from

# Lazy generation parameters 
SAMPLES_PER_POLE_INITIAL = 60  # Start with 60 samples per pole (180 total)
SAMPLES_PER_POLE_INCREMENT = 60  # Add 60 per pole if training fails
SAMPLES_PER_POLE_MAX = 300  # Maximum samples per pole
MIN_F1_THRESHOLD = 0.80  # F1 threshold to graduate (will adjust based on results)

# Training hyperparameters
MAX_EPOCHS = 200  # Increased from 100 to allow fuller convergence
PATIENCE = 10
LEARNING_RATE = 1e-3
LAMBDA_MARGIN = 0.5
LAMBDA_ORTHO = 1e-4


def load_s_tier_simplexes():
    """Load all S-tier simplexes from s_tier_simplex_definitions.json"""
    with open(S_TIER_DEFS_PATH) as f:
        s_tier_defs = json.load(f)

    simplexes = []
    for dimension, simplex_def in s_tier_defs['simplexes'].items():
        simplex = {
            'simplex_dimension': dimension,
            'three_pole_simplex': {
                'negative_pole': simplex_def['negative_pole'],
                'neutral_homeostasis': simplex_def['neutral_homeostasis'],
                'positive_pole': simplex_def['positive_pole']
            }
        }
        simplexes.append(simplex)

    return simplexes


def generate_tripole_data(
    simplex: dict,
    n_samples_per_pole: int,
    model,
    tokenizer,
    device: str,
    layer_idx: int
):
    """
    Generate balanced 3-class training data for tripole training.

    Args:
        simplex: Simplex concept dict
        n_samples_per_pole: Target number of samples per pole
        model: Language model for extracting activations
        tokenizer: Tokenizer
        device: Device to run on
        layer_idx: Layer to extract activations from

    Returns:
        (activations, labels, pole_counts) tuple
        - activations: [n_total, hidden_dim] tensor
        - labels: [n_total] tensor with pole indices (0=negative, 1=neutral, 2=positive)
        - pole_counts: dict with actual sample counts per pole
    """
    dimension = simplex['simplex_dimension']
    three_pole = simplex['three_pole_simplex']

    # Generate contrastive datasets for each pole
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
        # Duplicate labels to match
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


def train_simplex_with_lazy_generation(
    simplex: dict,
    model,
    tokenizer,
    device: str,
    run_dir: Path,
    layer_idx: int = 12
):
    """
    Train a tripole lens with lazy data generation.

    Starts with SAMPLES_PER_POLE_INITIAL samples per pole.
    If training fails to reach MIN_F1_THRESHOLD, incrementally adds more data.

    Args:
        simplex: Simplex concept dict
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        run_dir: Output directory for this simplex
        layer_idx: Layer to extract activations from

    Returns:
        results: Dict with training results
    """
    dimension = simplex['simplex_dimension']

    print(f"\n  Training {dimension} with lazy generation...")

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

        # Train tripole lens
        lens, history = train_tripole_simplex(
            train_activations=train_activations,
            train_labels=train_labels,
            test_activations=test_activations,
            test_labels=test_labels,
            hidden_dim=train_activations.shape[1],
            device=device,
            lr=LEARNING_RATE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
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
        'pole_counts': pole_counts
    }

    results_file = run_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    ✓ Results saved: {results_file}")

    return results


def main():
    print("=" * 80)
    print("S-TIER TRIPOLE TRAINING WITH LAZY GENERATION")
    print("=" * 80)
    print(f"Architecture: Joint 3-class softmax (TripoleLens)")
    print(f"Start: {SAMPLES_PER_POLE_INITIAL} samples/pole, Increment: {SAMPLES_PER_POLE_INCREMENT}, Max: {SAMPLES_PER_POLE_MAX}")
    print(f"Graduation threshold: F1 >= {MIN_F1_THRESHOLD}")

    # Load simplexes
    print("\n1. Loading S-tier simplexes...")
    simplexes = load_s_tier_simplexes()
    print(f"   Found {len(simplexes)} S-tier simplexes")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = run_dir / "training.log"

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

    print(f"\n2. Output directory: {run_dir}")
    print(f"   Log file: {log_file}")

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

    # Train each simplex
    print(f"\n4. Training {len(simplexes)} simplexes with lazy generation...")

    all_results = []
    graduated = []
    failed = []

    for i, simplex in enumerate(simplexes, 1):
        dimension = simplex['simplex_dimension']

        print(f"\n[{i}/{len(simplexes)}] {dimension}")
        print("─" * 60)

        simplex_dir = run_dir / dimension
        simplex_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = train_simplex_with_lazy_generation(
                simplex=simplex,
                model=model,
                tokenizer=tokenizer,
                device=device,
                run_dir=simplex_dir,
                layer_idx=LAYER_IDX
            )

            all_results.append(results)

            if results['graduated']:
                graduated.append(dimension)
            else:
                failed.append(dimension)

        except Exception as e:
            print(f"  ✗ Failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed.append(dimension)
            all_results.append({
                'dimension': dimension,
                'error': str(e),
                'graduated': False
            })

        # Save intermediate results
        with open(run_dir / "results.json", 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'config': {
                    'samples_per_pole_initial': SAMPLES_PER_POLE_INITIAL,
                    'samples_per_pole_increment': SAMPLES_PER_POLE_INCREMENT,
                    'samples_per_pole_max': SAMPLES_PER_POLE_MAX,
                    'min_f1_threshold': MIN_F1_THRESHOLD,
                    'behavioral_ratio': BEHAVIORAL_RATIO,
                    'layer_idx': LAYER_IDX
                },
                'total_simplexes': len(simplexes),
                'completed': i,
                'graduated': graduated,
                'failed': failed,
                'simplexes': all_results
            }, f, indent=2)

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(f"\nTotal simplexes: {len(simplexes)}")
    print(f"Graduated: {len(graduated)}/{len(simplexes)}")
    print(f"Failed: {len(failed)}/{len(simplexes)}")

    if graduated:
        print("\n✓ Graduated simplexes:")
        for dim in graduated:
            print(f"  - {dim}")

    if failed:
        print("\n✗ Failed simplexes:")
        for dim in failed:
            print(f"  - {dim}")

    # Performance statistics
    test_f1s = [r['best_test_f1'] for r in all_results if 'best_test_f1' in r]
    iterations_list = [r['total_iterations'] for r in all_results if 'total_iterations' in r]
    samples_list = [r['final_samples_per_pole'] for r in all_results if 'final_samples_per_pole' in r]

    if test_f1s:
        print(f"\nPerformance:")
        print(f"  Average test F1: {sum(test_f1s) / len(test_f1s):.3f}")
        print(f"  Average iterations: {sum(iterations_list) / len(iterations_list):.1f}")
        print(f"  Average samples/pole: {sum(samples_list) / len(samples_list):.1f}")

    print(f"\n✓ Results saved to: {run_dir}")
    print("=" * 80)

    # Restore stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()

    print(f"\n✓ Training log saved to: {log_file}")


if __name__ == "__main__":
    main()
