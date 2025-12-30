#!/usr/bin/env python3
"""
Train S-tier three-pole simplex lenses for homeostatic steering.

This script trains 3 binary classifiers per simplex:
- μ− (negative pole) detector
- μ0 (neutral homeostasis) detector
- μ+ (positive pole) detector

These enable homeostatic steering: detecting current pole and steering toward μ0.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
# File is at scripts/training/train_s_tier_simplexes.py
# Need 3 parents to reach project root: train_s_tier_simplexes.py -> training -> scripts -> HatCat
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from map.training.sumo_data_generation import create_simplex_pole_training_dataset_contrastive
from map.training.dual_adaptive_trainer import DualAdaptiveTrainer
from map.training.sumo_classifiers import extract_activations, select_layers_for_concept, get_num_layers

# Paths
S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "s_tier_simplexes"

# Training configuration
BEHAVIORAL_RATIO = 0.6  # 60% behavioral, 40% definitional

# Lazy generation - only create what we need
# Start higher (60) since we have rich enriched data
INITIAL_SAMPLES = 60  # Start with 60 samples per class (120 total)
FIRST_INCREMENT = 60  # Add 60 if initial fails (120 total)
SUBSEQUENT_INCREMENT = 60  # Add 60 per subsequent cycle
MAX_SAMPLES = 300  # Maximum samples per class


def load_s_tier_simplexes():
    """Load all S-tier simplexes from s_tier_simplex_definitions.json"""
    with open(S_TIER_DEFS_PATH) as f:
        s_tier_defs = json.load(f)

    # Convert to the expected format (compatible with old layer2 structure)
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


def train_simplex_pole(
    simplex: dict,
    pole_name: str,
    trainer: DualAdaptiveTrainer,
    model,
    tokenizer,
    device: str,
    run_dir: Path,
    multi_layer_mode: bool = True  # Auto-select best layers like regular training
):
    """
    Train a single pole detector for a simplex with lazy data generation.

    Args:
        simplex: Simplex concept dict from s_tier_simplex_definitions.json
        pole_name: "negative_pole", "neutral_homeostasis", or "positive_pole"
        trainer: DualAdaptiveTrainer instance
        model: Language model for extracting activations
        tokenizer: Tokenizer
        device: Device to run on
        run_dir: Output directory for this simplex
        multi_layer_mode: If True, auto-select best layers from each third
                         using same logic as regular training
    """
    dimension = simplex['simplex_dimension']
    three_pole = simplex['three_pole_simplex']

    # Get pole data
    pole_data = three_pole[pole_name]
    pole_type = pole_name.split('_')[0]  # "negative", "neutral", or "positive"

    # Get other poles for hard negatives
    other_pole_names = [p for p in ['negative_pole', 'neutral_homeostasis', 'positive_pole'] if p != pole_name]
    other_poles_data = [
        {**three_pole[p], 'pole_type': p.split('_')[0]}
        for p in other_pole_names
    ]

    print(f"\n  [{pole_type.upper()}] Training {pole_type} pole detector")
    print(f"    Synset: {pole_data.get('synset', 'custom SUMO')}")

    # Generate test set once (fixed size)
    print(f"    Generating test set...")
    test_prompts, test_labels = create_simplex_pole_training_dataset_contrastive(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=dimension,
        other_poles_data=other_poles_data,
        behavioral_ratio=BEHAVIORAL_RATIO,
        prompts_per_synset=3  # Smaller for test set
    )
    # Take first 40 samples for test
    test_prompts = test_prompts[:40]
    test_labels = np.array(test_labels[:40])
    print(f"    ✓ Test set: {len(test_prompts)} samples")

    # Layer selection: use same logic as regular training
    selected_layers = None
    if multi_layer_mode:
        print(f"    Selecting best layers...")
        # Generate sample for layer selection
        layer_sample_prompts, layer_sample_labels = create_simplex_pole_training_dataset_contrastive(
            pole_data=pole_data,
            pole_type=pole_type,
            dimension=dimension,
            other_poles_data=other_poles_data,
            behavioral_ratio=BEHAVIORAL_RATIO,
            prompts_per_synset=2
        )
        pos_prompts = [p for p, l in zip(layer_sample_prompts, layer_sample_labels) if l == 1][:20]
        neg_prompts = [p for p, l in zip(layer_sample_prompts, layer_sample_labels) if l == 0][:20]

        n_model_layers = get_num_layers(model)
        selected_layers, layer_scores = select_layers_for_concept(
            model=model,
            tokenizer=tokenizer,
            pos_prompts=pos_prompts,
            neg_prompts=neg_prompts,
            device=device,
            n_model_layers=n_model_layers,
            top_k=1  # One layer per third (early/mid/late)
        )
        # Update trainer to use selected layers
        trainer.validation_layer_idx = selected_layers

    # Define lazy generation function
    def generate_training_samples(n_samples: int):
        """Generate n_samples lazily when trainer needs them."""
        # Generate with higher prompts_per_synset to get enough variety
        all_prompts, all_labels = create_simplex_pole_training_dataset_contrastive(
            pole_data=pole_data,
            pole_type=pole_type,
            dimension=dimension,
            other_poles_data=other_poles_data,
            behavioral_ratio=BEHAVIORAL_RATIO,
            prompts_per_synset=5  # Generate more per synset
        )
        # Take first n_samples (generation is already balanced)
        n_take = min(len(all_prompts), n_samples)
        return all_prompts[:n_take], all_labels[:n_take]

    # Use train_concept_incremental for lazy training data generation
    generation_config = {
        'custom_generate_fn': generate_training_samples,  # Custom generation for tripole
        'model': model,
        'tokenizer': tokenizer,
        'device': device,
    }

    # Train with lazy generation
    results = trainer.train_concept_incremental(
        concept_name=f"{dimension}_{pole_type}",
        generation_config=generation_config,
        test_prompts=test_prompts,
        test_labels=test_labels
    )

    # Store selected layers in results for reference
    if selected_layers is not None:
        results['selected_layers'] = selected_layers

    # Save results
    pole_output_dir = run_dir / pole_type
    pole_output_dir.mkdir(parents=True, exist_ok=True)

    # Save lens (if it graduated)
    if results.get('activation_classifier') is not None:
        lens = results['activation_classifier']
        lens_file = pole_output_dir / f"{dimension}_{pole_type}_classifier.pt"
        torch.save(lens.state_dict(), lens_file)
        print(f"    ✓ Lens saved to {lens_file}")

    # Save metrics (remove non-serializable objects)
    results_to_save = {
        'activation_f1': results.get('activation_f1'),
        'activation_tier': results.get('activation_tier'),
        'validation_passed': results.get('validation_passed'),
        'total_iterations': results.get('total_iterations'),
        'total_time': results.get('total_time'),
        'selected_layers': results.get('selected_layers')  # Track which layers were selected
    }

    results_file = pole_output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"    ✓ Results saved to {results_file}")

    return results


def main():
    print("=" * 80)
    print("S+ THREE-POLE SIMPLEX TRAINING")
    print("=" * 80)

    # Load simplexes
    print("\n1. Loading S-tier simplexes from s_tier_simplex_definitions.json...")
    simplexes = load_s_tier_simplexes()
    print(f"   Found {len(simplexes)} S-tier simplexes")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    log_file = run_dir / "training.log"

    # Duplicate stdout/stderr to log file
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

    # Initialize trainer with lazy generation parameters
    print("\n4. Initializing trainer...")
    trainer = DualAdaptiveTrainer(
        model=model,
        tokenizer=tokenizer,
        validation_layer_idx=15,  # Default, will be updated per-pole by layer selection
        validate_lenses=True,
        validation_mode="falloff",
        train_activation=True,
        train_text=False,
        # Lazy generation parameters (user requested 60-90 starting point)
        activation_initial_samples=INITIAL_SAMPLES,
        activation_first_increment=FIRST_INCREMENT,
        activation_subsequent_increment=SUBSEQUENT_INCREMENT,
        activation_max_samples=MAX_SAMPLES
    )
    print(f"   ✓ Trainer ready (start={INITIAL_SAMPLES}, increment={FIRST_INCREMENT}, max={MAX_SAMPLES})")

    # Train each simplex
    print(f"\n5. Training {len(simplexes)} simplexes ({len(simplexes) * 3} lenses total)...")

    all_results = []
    failed = []

    for i, simplex in enumerate(simplexes, 1):
        dimension = simplex['simplex_dimension']

        print(f"\n[{i}/{len(simplexes)}] {dimension}")
        print("─" * 60)

        simplex_dir = run_dir / dimension
        simplex_dir.mkdir(parents=True, exist_ok=True)

        simplex_results = {
            'dimension': dimension,
            'poles': {}
        }

        # Train all 3 poles
        for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
            try:
                results = train_simplex_pole(
                    simplex=simplex,
                    pole_name=pole_name,
                    trainer=trainer,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    run_dir=simplex_dir,
                    multi_layer_mode=True  # Auto-select best layers like regular training
                )

                pole_type = pole_name.split('_')[0]
                activation_results = results.get('activation', {})
                simplex_results['poles'][pole_type] = {
                    'success': True,
                    'test_f1': activation_results.get('test_f1', 0.0),
                    'samples_used': activation_results.get('samples_used', 0),
                    'iterations': activation_results.get('iterations', 0)
                }

            except Exception as e:
                print(f"    ✗ Failed: {e}")
                pole_type = pole_name.split('_')[0]
                simplex_results['poles'][pole_type] = {
                    'success': False,
                    'error': str(e)
                }
                failed.append(f"{dimension}/{pole_type}")

        all_results.append(simplex_results)

        # Save intermediate results
        with open(run_dir / "results.json", 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_simplexes': len(simplexes),
                'completed': i,
                'failed_lenses': failed,
                'simplexes': all_results
            }, f, indent=2)

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    total_lenses = len(simplexes) * 3
    successful_lenses = sum(
        sum(1 for p in s['poles'].values() if p.get('success'))
        for s in all_results
    )

    print(f"\nTotal simplexes: {len(simplexes)}")
    print(f"Total lenses: {total_lenses}")
    print(f"Successful: {successful_lenses}/{total_lenses}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed lenses:")
        for lens in failed:
            print(f"  - {lens}")

    # Performance statistics
    test_f1s = []
    samples_used = []
    iterations_list = []

    for simplex in all_results:
        for pole_type, pole_results in simplex['poles'].items():
            if pole_results.get('success'):
                test_f1s.append(pole_results.get('test_f1', 0.0))
                samples_used.append(pole_results.get('samples_used', 0))
                iterations_list.append(pole_results.get('iterations', 0))

    if test_f1s:
        print(f"\nPerformance:")
        print(f"  Average test F1: {sum(test_f1s) / len(test_f1s):.3f}")
        print(f"  Average samples used: {sum(samples_used) / len(samples_used):.1f}")
        print(f"  Average iterations: {sum(iterations_list) / len(iterations_list):.1f}")

    print(f"\n✓ Results saved to: {run_dir}")
    print("=" * 80)

    # Restore stdout/stderr and close log file
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()

    print(f"\n✓ Training log saved to: {log_file}")


if __name__ == "__main__":
    main()
