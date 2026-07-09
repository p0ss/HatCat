#!/usr/bin/env python3
"""
Train S-tier three-pole simplex lenses for homeostatic steering.

This script trains 3 binary classifiers per simplex:
- μ− (negative pole) detector
- μ0 (neutral homeostasis) detector
- μ+ (positive pole) detector

These enable homeostatic steering: detecting current pole and steering toward μ0.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path so `from src.map...` imports resolve consistently with
# the rest of the codebase (which expects HatCatDev as the package root, not src/).
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.map.training.sumo_data_generation import create_simplex_pole_training_dataset_contrastive
from src.map.training.dual_adaptive_trainer import DualAdaptiveTrainer
from src.map.training.sumo_classifiers import extract_activations, select_layers_for_concept, get_num_layers

# Default paths (overridable via CLI)
DEFAULT_CONCEPT_PACK = PROJECT_ROOT / "concept_packs" / "first-light"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "s_tier_simplexes"
DEFAULT_MODEL = "google/gemma-3-4b-pt"

# Training configuration
BEHAVIORAL_RATIO = 0.6  # 60% behavioral, 40% definitional

# Lazy generation - only create what we need
# Start higher (60) since we have rich enriched data
INITIAL_SAMPLES = 60  # Start with 60 samples per class (120 total)
FIRST_INCREMENT = 60  # Add 60 if initial fails (120 total)
SUBSEQUENT_INCREMENT = 60  # Add 60 per subsequent cycle
MAX_SAMPLES = 300  # Maximum samples per class


def load_s_tier_simplexes(concept_pack_path: Path):
    """Load S-tier simplexes from the concept pack's simplexes.json.

    The concept pack's simplexes.json schema has each entry with full pole
    structure under 'three_pole_simplex' (synset/lemmas/definition per pole),
    matching the format `train_simplex_pole` consumes.
    """
    simplexes_path = Path(concept_pack_path) / "simplexes.json"
    with open(simplexes_path) as f:
        data = json.load(f)

    return [
        {
            'simplex_dimension': entry['simplex_dimension'],
            'three_pole_simplex': entry['three_pole_simplex'],
        }
        for entry in data['simplexes']
        if entry.get('simplex_dimension') and 'three_pole_simplex' in entry
    ]


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

    # Save lens (if it graduated). Sanitize the dimension for use as a filename
    # component — some dimensions contain path separators (e.g. "aspiration/social_mobility")
    # which would otherwise produce nested filenames that mkdir-less torch.save rejects.
    safe_dimension = dimension.replace('/', '_')
    if results.get('activation_classifier') is not None:
        lens = results['activation_classifier']
        lens_file = pole_output_dir / f"{safe_dimension}_{pole_type}_classifier.pt"
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


def run_simplex_training(
    model,
    tokenizer,
    simplexes: list,
    run_dir: Path,
    device: str = "cuda",
    timestamp: str = None,
) -> tuple:
    """
    Train all simplex poles with per-pole layer selection and lazy data generation.

    Args:
        model: Loaded language model (already on device, in eval mode).
        tokenizer: Matching tokenizer.
        simplexes: List of simplex dicts, each with 'simplex_dimension' and
                   'three_pole_simplex' (with negative_pole, neutral_homeostasis,
                   positive_pole, each carrying synset/lemmas/definition).
        run_dir: Output directory for this training run. Created if missing.
                 Per-simplex subdirectories will be created inside.
        device: Device for activation extraction.
        timestamp: Optional timestamp string for results.json. Defaults to now.

    Returns:
        (all_results, failed) where all_results is a list of per-simplex result
        dicts and failed is a list of "{dimension}/{pole_type}" strings.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer with lazy generation parameters
    print("\nInitializing simplex trainer...")
    trainer = DualAdaptiveTrainer(
        model=model,
        tokenizer=tokenizer,
        validation_layer_idx=15,  # Default; updated per-pole by layer selection
        validate_lenses=True,
        validation_mode="falloff",
        train_activation=True,
        train_text=False,
        activation_initial_samples=INITIAL_SAMPLES,
        activation_first_increment=FIRST_INCREMENT,
        activation_subsequent_increment=SUBSEQUENT_INCREMENT,
        activation_max_samples=MAX_SAMPLES,
    )
    print(f"   ✓ Trainer ready (start={INITIAL_SAMPLES}, increment={FIRST_INCREMENT}, max={MAX_SAMPLES})")

    print(f"\nTraining {len(simplexes)} simplexes ({len(simplexes) * 3} lenses total)...")

    all_results = []
    failed = []

    for i, simplex in enumerate(simplexes, 1):
        dimension = simplex['simplex_dimension']

        print(f"\n[{i}/{len(simplexes)}] {dimension}")
        print("─" * 60)

        # Sanitize dimension for filesystem path (some dimensions contain path
        # separators, e.g. "aspiration/social_mobility"). Keeps directory naming
        # consistent with the per-pole filename produced by train_simplex_pole.
        safe_dimension = dimension.replace('/', '_')
        simplex_dir = run_dir / safe_dimension
        simplex_dir.mkdir(parents=True, exist_ok=True)

        simplex_results = {
            'dimension': dimension,
            'poles': {}
        }

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
                    multi_layer_mode=True,
                )

                pole_type = pole_name.split('_')[0]
                activation_results = results.get('activation', {})
                simplex_results['poles'][pole_type] = {
                    'success': True,
                    'test_f1': activation_results.get('test_f1', 0.0),
                    'samples_used': activation_results.get('samples_used', 0),
                    'iterations': activation_results.get('iterations', 0),
                }

            except Exception as e:
                print(f"    ✗ Failed: {e}")
                pole_type = pole_name.split('_')[0]
                simplex_results['poles'][pole_type] = {
                    'success': False,
                    'error': str(e),
                }
                failed.append(f"{dimension}/{pole_type}")

        all_results.append(simplex_results)

        with open(run_dir / "results.json", 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_simplexes': len(simplexes),
                'completed': i,
                'failed_lenses': failed,
                'simplexes': all_results,
            }, f, indent=2)

    print("\n" + "=" * 80)
    print("SIMPLEX TRAINING COMPLETE")
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

    test_f1s = [
        p.get('test_f1', 0.0)
        for s in all_results
        for p in s['poles'].values()
        if p.get('success')
    ]
    samples_used = [
        p.get('samples_used', 0)
        for s in all_results
        for p in s['poles'].values()
        if p.get('success')
    ]
    iterations_list = [
        p.get('iterations', 0)
        for s in all_results
        for p in s['poles'].values()
        if p.get('success')
    ]

    if test_f1s:
        print(f"\nPerformance:")
        print(f"  Average test F1: {sum(test_f1s) / len(test_f1s):.3f}")
        print(f"  Average samples used: {sum(samples_used) / len(samples_used):.1f}")
        print(f"  Average iterations: {sum(iterations_list) / len(iterations_list):.1f}")

    print(f"\n✓ Results saved to: {run_dir}")
    return all_results, failed


def main():
    parser = argparse.ArgumentParser(
        description="Train S-tier three-pole simplex lenses with per-pole layer selection"
    )
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'Model name or path (default: {DEFAULT_MODEL})')
    parser.add_argument('--concept-pack', type=Path, default=DEFAULT_CONCEPT_PACK,
                        help='Concept pack root containing simplexes.json '
                             f'(default: {DEFAULT_CONCEPT_PACK})')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory root (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Run name (default: run_<timestamp>)')
    args = parser.parse_args()

    print("=" * 80)
    print("S+ THREE-POLE SIMPLEX TRAINING")
    print("=" * 80)
    print(f"Model:        {args.model}")
    print(f"Concept pack: {args.concept_pack}")
    print(f"Output root:  {args.output_dir}")

    # Load simplexes
    print(f"\n1. Loading S-tier simplexes from {args.concept_pack}/simplexes.json...")
    simplexes = load_s_tier_simplexes(args.concept_pack)
    print(f"   Found {len(simplexes)} S-tier simplexes")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}"
    run_dir = args.output_dir / run_name
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
    model_name = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        local_files_only=True,
    )
    model.eval()
    print(f"   ✓ Model loaded on {device} (name: {model_name})")

    # Delegate to the callable training function (also called from train_full_lens_pack.py)
    run_simplex_training(
        model=model,
        tokenizer=tokenizer,
        simplexes=simplexes,
        run_dir=run_dir,
        device=device,
        timestamp=timestamp,
    )

    # Restore stdout/stderr and close log file
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()

    print(f"\n✓ Training log saved to: {log_file}")


if __name__ == "__main__":
    main()
