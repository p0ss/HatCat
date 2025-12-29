#!/usr/bin/env python3
"""
Train the 5 RLHF-related simplex axes.

These axes capture different dimensions of RLHF-induced behavior:
- rl_reward_signal: Positive/negative evaluation feedback
- allowance_termination: Jailbreak triggers vs safe responses
- normative_alignment: Polite/respectful vs rude/hostile
- epistemic_posture: Overconfident vs appropriately hedged
- instrumentality: Action-enabling vs descriptive/blocking
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from map.training.sumo_data_generation import create_simplex_pole_training_dataset_contrastive
from map.training.dual_adaptive_trainer import DualAdaptiveTrainer
from map.training.sumo_classifiers import select_layers_for_concept, get_num_layers

# Paths
S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "rlhf_axes"

# RLHF axes to train
RLHF_AXES = [
    'rl_reward_signal',
    'allowance_termination',
    'normative_alignment',
    'epistemic_posture',
    'instrumentality'
]

# Training configuration
BEHAVIORAL_RATIO = 0.6
INITIAL_SAMPLES = 60
FIRST_INCREMENT = 60
SUBSEQUENT_INCREMENT = 60
MAX_SAMPLES = 300


def load_rlhf_simplexes():
    """Load only the RLHF-related simplexes."""
    with open(S_TIER_DEFS_PATH) as f:
        s_tier_defs = json.load(f)

    simplexes = []
    for dimension in RLHF_AXES:
        if dimension in s_tier_defs['simplexes']:
            simplex_def = s_tier_defs['simplexes'][dimension]
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
    run_dir: Path
):
    """Train a single pole detector."""
    dimension = simplex['simplex_dimension']
    three_pole = simplex['three_pole_simplex']

    pole_data = three_pole[pole_name]
    pole_type = pole_name.split('_')[0]

    other_pole_names = [p for p in ['negative_pole', 'neutral_homeostasis', 'positive_pole'] if p != pole_name]
    other_poles_data = [
        {**three_pole[p], 'pole_type': p.split('_')[0]}
        for p in other_pole_names
    ]

    print(f"\n  [{pole_type.upper()}] Training {pole_type} pole detector")

    # Generate test set
    print(f"    Generating test set...")
    test_prompts, test_labels = create_simplex_pole_training_dataset_contrastive(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=dimension,
        other_poles_data=other_poles_data,
        behavioral_ratio=BEHAVIORAL_RATIO,
        prompts_per_synset=3
    )
    test_prompts = test_prompts[:40]
    test_labels = np.array(test_labels[:40])
    print(f"    Test set: {len(test_prompts)} samples ({sum(test_labels)} pos, {len(test_labels) - sum(test_labels)} neg)")

    # Layer selection
    print(f"    Selecting best layers...")
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
        top_k=1
    )
    trainer.validation_layer_idx = selected_layers
    print(f"    Selected layers: {selected_layers}")

    # Lazy generation function
    def generate_training_samples(n_samples: int):
        all_prompts, all_labels = create_simplex_pole_training_dataset_contrastive(
            pole_data=pole_data,
            pole_type=pole_type,
            dimension=dimension,
            other_poles_data=other_poles_data,
            behavioral_ratio=BEHAVIORAL_RATIO,
            prompts_per_synset=5
        )
        n_take = min(len(all_prompts), n_samples)
        return all_prompts[:n_take], all_labels[:n_take]

    generation_config = {
        'custom_generate_fn': generate_training_samples,
        'model': model,
        'tokenizer': tokenizer,
        'device': device,
    }

    # Train
    results = trainer.train_concept_incremental(
        concept_name=f"{dimension}_{pole_type}",
        generation_config=generation_config,
        test_prompts=test_prompts,
        test_labels=test_labels
    )

    results['selected_layers'] = selected_layers

    # Save
    pole_output_dir = run_dir / pole_type
    pole_output_dir.mkdir(parents=True, exist_ok=True)

    if results.get('activation_classifier') is not None:
        lens = results['activation_classifier']
        lens_file = pole_output_dir / f"{dimension}_{pole_type}_classifier.pt"
        torch.save(lens.state_dict(), lens_file)
        print(f"    Lens saved: {lens_file.name}")

    results_to_save = {
        'activation_f1': results.get('activation_f1'),
        'activation_tier': results.get('activation_tier'),
        'validation_passed': results.get('validation_passed'),
        'total_iterations': results.get('total_iterations'),
        'selected_layers': results.get('selected_layers')
    }

    with open(pole_output_dir / "results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)

    return results


def main():
    print("=" * 80)
    print("RLHF AXES TRAINING")
    print("=" * 80)

    # Load simplexes
    print("\n1. Loading RLHF simplexes...")
    simplexes = load_rlhf_simplexes()
    print(f"   Found {len(simplexes)} RLHF axes: {[s['simplex_dimension'] for s in simplexes]}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n2. Output: {run_dir}")

    # Load model
    print("\n3. Loading model...")
    model_name = "google/gemma-3-4b-it"  # Use IT model for RLHF axes
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"   Model: {model_name} on {device}")

    # Initialize trainer
    print("\n4. Initializing trainer...")
    trainer = DualAdaptiveTrainer(
        model=model,
        tokenizer=tokenizer,
        validation_layer_idx=15,
        validate_lenses=True,
        validation_mode="falloff",
        train_activation=True,
        train_text=False,
        activation_initial_samples=INITIAL_SAMPLES,
        activation_first_increment=FIRST_INCREMENT,
        activation_subsequent_increment=SUBSEQUENT_INCREMENT,
        activation_max_samples=MAX_SAMPLES
    )

    # Train
    print(f"\n5. Training {len(simplexes)} axes ({len(simplexes) * 3} lenses)...")

    all_results = []
    failed = []

    for i, simplex in enumerate(simplexes, 1):
        dimension = simplex['simplex_dimension']

        print(f"\n[{i}/{len(simplexes)}] {dimension}")
        print("-" * 60)

        simplex_dir = run_dir / dimension
        simplex_dir.mkdir(parents=True, exist_ok=True)

        simplex_results = {'dimension': dimension, 'poles': {}}

        for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
            try:
                results = train_simplex_pole(
                    simplex=simplex,
                    pole_name=pole_name,
                    trainer=trainer,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    run_dir=simplex_dir
                )

                pole_type = pole_name.split('_')[0]
                simplex_results['poles'][pole_type] = {
                    'success': True,
                    'f1': results.get('activation_f1', 0.0),
                    'layers': results.get('selected_layers')
                }

            except Exception as e:
                print(f"    FAILED: {e}")
                pole_type = pole_name.split('_')[0]
                simplex_results['poles'][pole_type] = {'success': False, 'error': str(e)}
                failed.append(f"{dimension}/{pole_type}")

        all_results.append(simplex_results)

        # Save progress
        with open(run_dir / "results.json", 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'model': model_name,
                'axes': RLHF_AXES,
                'completed': i,
                'failed': failed,
                'results': all_results
            }, f, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    total = len(simplexes) * 3
    successful = sum(sum(1 for p in s['poles'].values() if p.get('success')) for s in all_results)

    print(f"\nTotal lenses: {total}")
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed:")
        for f in failed:
            print(f"  - {f}")

    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
