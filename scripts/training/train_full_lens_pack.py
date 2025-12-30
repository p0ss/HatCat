#!/usr/bin/env python3
"""
Train complete SUMO lens pack including all layers and simplexes.

This is the comprehensive training script that:
1. Trains all SUMO hierarchy layers (0-5) with nephew negative sampling
2. Trains all S-tier three-pole simplexes (13 in Layer 2)
3. Uses adaptive training with falloff validation
4. Generates lens pack ready for deployment

Architecture:
- Layers 0-5: Binary classifiers for hierarchical SUMO concepts
- Layer 2 simplexes: 3 binary lenses per simplex (negative/neutral/positive poles)
- Total: ~5,665 regular lenses + 39 simplex lenses (3 per simplex × 13 simplexes)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.map.training.sumo_classifiers import train_sumo_classifiers
from src.map.training.sumo_data_generation import create_simplex_pole_training_dataset
from src.map.training.dual_adaptive_trainer import DualAdaptiveTrainer
from src.map.training.sumo_classifiers import extract_activations
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train complete SUMO lens pack with all layers and simplexes"
    )

    # Model configuration
    parser.add_argument('--model', default="google/gemma-3-4b-pt",
                        help='Model name (default: gemma-3-4b-pt)')
    parser.add_argument('--device', default="cuda",
                        help='Device (default: cuda)')

    # Layer selection
    parser.add_argument('--layers', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5],
                        help='Which layers to train (default: 0 1 2 3 4 5)')
    parser.add_argument('--skip-simplexes', action='store_true',
                        help='Skip simplex training (default: train simplexes)')

    # Training configuration
    parser.add_argument('--n-train-pos', type=int, default=50,
                        help='Positive training samples per concept (default: 50)')
    parser.add_argument('--n-train-neg', type=int, default=50,
                        help='Negative training samples per concept (default: 50)')
    parser.add_argument('--n-test-pos', type=int, default=20,
                        help='Positive test samples per concept (default: 20)')
    parser.add_argument('--n-test-neg', type=int, default=20,
                        help='Negative test samples per concept (default: 20)')

    # Adaptive training
    parser.add_argument('--validation-mode', type=str, default='falloff',
                        choices=['loose', 'falloff', 'strict'],
                        help='Validation mode (default: falloff)')

    # Output
    parser.add_argument('--output-dir', type=str,
                        default="results/full_lens_pack",
                        help='Output directory (default: results/full_lens_pack)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Run name (default: timestamp)')
    parser.add_argument('--all-layers', action='store_true',
                        help='Extract from all layers (experimental). Classifier learns which layers matter.')

    return parser.parse_args()


def load_s_tier_simplexes():
    """Load all S-tier simplexes from layer2.json"""
    layer2_path = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers" / "layer2.json"
    with open(layer2_path) as f:
        layer2 = json.load(f)

    simplexes = []
    for concept in layer2['concepts']:
        if concept.get('s_tier') and concept.get('simplex_dimension'):
            simplexes.append(concept)

    return simplexes


def train_simplex_pole(
    simplex: dict,
    pole_name: str,
    trainer: DualAdaptiveTrainer,
    model,
    tokenizer,
    device: str,
    run_dir: Path,
    layer_idx: int = 15
):
    """
    Train a single pole detector for a simplex.

    Args:
        simplex: Simplex concept dict from layer2.json
        pole_name: "negative_pole", "neutral_homeostasis", or "positive_pole"
        trainer: DualAdaptiveTrainer instance
        model: Language model for extracting activations
        tokenizer: Tokenizer
        device: Device to run on
        run_dir: Output directory for this simplex
        layer_idx: Layer to extract activations from
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

    # Generate training data (80/20 split)
    all_prompts, all_labels = create_simplex_pole_training_dataset(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=dimension,
        other_poles_data=other_poles_data,
        n_positives=125,  # Will give us 100 train positives
        n_negatives=125,  # Will give us 100 train negatives
        behavioral_ratio=0.6,
    )

    # Split into train/test
    n_samples = len(all_prompts)
    split_idx = int(n_samples * 0.8)

    train_prompts = all_prompts[:split_idx]
    train_labels = all_labels[:split_idx]
    test_prompts = all_prompts[split_idx:]
    test_labels = all_labels[split_idx:]

    print(f"    Generated {len(train_prompts)} train, {len(test_prompts)} test prompts")

    # Extract activations for training
    train_activations = extract_activations(
        model, tokenizer, train_prompts, device=device, layer_idx=layer_idx
    )
    test_activations = extract_activations(
        model, tokenizer, test_prompts, device=device, layer_idx=layer_idx
    )

    # Train with adaptive trainer (activation lens only)
    result = trainer.train_activation_lens_adaptive(
        concept_name=f"{dimension}_{pole_type}",
        train_activations=train_activations,
        train_labels=train_labels,
        test_activations=test_activations,
        test_labels=test_labels,
    )

    # Save classifier
    if result['classifier'] is not None:
        lens_path = run_dir / f"{dimension}_{pole_type}_pole.pt"
        torch.save(result['classifier'].state_dict(), lens_path)
        print(f"    ✓ Saved lens to {lens_path.name}")

    return result


def train_simplexes(
    model,
    tokenizer,
    device: str,
    output_dir: Path,
    validation_mode: str = 'falloff',
    all_layers: bool = False,
):
    """Train all S-tier three-pole simplexes."""
    print("\n" + "=" * 80)
    print("TRAINING S-TIER SIMPLEXES")
    if all_layers:
        print("MODE: All-layers extraction (experimental)")
    print("=" * 80)

    simplexes = load_s_tier_simplexes()
    print(f"\nFound {len(simplexes)} S-tier simplexes to train")

    # Create output directory
    simplex_dir = output_dir / "simplexes"
    simplex_dir.mkdir(parents=True, exist_ok=True)

    # Initialize adaptive trainer
    trainer = DualAdaptiveTrainer(
        activation_target_accuracy=0.95,
        activation_initial_samples=10,
        activation_first_increment=20,
        activation_subsequent_increment=30,
        activation_max_samples=200,
        text_target_accuracy=0.80,
        text_initial_samples=10,
        text_first_increment=20,
        text_subsequent_increment=30,
        text_max_samples=200,
        model=model,
        tokenizer=tokenizer,
        max_response_tokens=100,
        validate_lenses=True,
        validation_mode=validation_mode,
        validation_threshold=0.5,
        validation_layer_idx=15,
        validation_tier1_iterations=3,
        validation_tier2_iterations=6,
        validation_tier3_iterations=9,
        validation_tier4_iterations=12,
        train_activation=True,
        train_text=False,
        all_layers=all_layers,
    )

    all_results = []

    for i, simplex in enumerate(simplexes):
        dimension = simplex['simplex_dimension']
        sumo_term = simplex['sumo_term']

        print(f"\n[{i+1}/{len(simplexes)}] Training simplex: {dimension} ({sumo_term})")

        # Create directory for this simplex
        run_dir = simplex_dir / dimension
        run_dir.mkdir(exist_ok=True)

        simplex_results = {
            'dimension': dimension,
            'sumo_term': sumo_term,
            'poles': {}
        }

        # Train each pole
        for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
            try:
                result = train_simplex_pole(
                    simplex=simplex,
                    pole_name=pole_name,
                    trainer=trainer,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    run_dir=run_dir,
                )
                simplex_results['poles'][pole_name] = result
            except Exception as e:
                print(f"    ✗ Failed to train {pole_name}: {e}")
                simplex_results['poles'][pole_name] = {'error': str(e)}

        all_results.append(simplex_results)

    # Save summary
    summary_path = simplex_dir / "simplex_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Simplex training complete. Summary saved to {summary_path}")

    return all_results


def main():
    args = parse_args()

    # Create run directory
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FULL LENS PACK TRAINING")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Device: {args.device}")
    print(f"Layers: {args.layers}")
    print(f"Simplexes: {'Skip' if args.skip_simplexes else 'Train'}")
    print(f"Training samples: {args.n_train_pos} pos, {args.n_train_neg} neg")
    print(f"Test samples: {args.n_test_pos} pos, {args.n_test_neg} neg")
    print(f"Validation mode: {args.validation_mode}")
    print(f"Output: {output_dir}")
    print()

    # Save configuration
    config = {
        'model': args.model,
        'device': args.device,
        'layers': args.layers,
        'skip_simplexes': args.skip_simplexes,
        'n_train_pos': args.n_train_pos,
        'n_train_neg': args.n_train_neg,
        'n_test_pos': args.n_test_pos,
        'n_test_neg': args.n_test_neg,
        'validation_mode': args.validation_mode,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Train regular SUMO layers
    print("STEP 1: Training SUMO hierarchy layers")
    print("-" * 80)

    train_sumo_classifiers(
        layers=args.layers,
        model_name=args.model,
        device=args.device,
        n_train_pos=args.n_train_pos,
        n_train_neg=args.n_train_neg,
        n_test_pos=args.n_test_pos,
        n_test_neg=args.n_test_neg,
        output_dir=str(output_dir / "layers"),
        train_text_lenses=False,
        use_adaptive_training=True,
        validation_mode=args.validation_mode,
    )

    # Train simplexes
    if not args.skip_simplexes:
        print("\n" + "=" * 80)
        print("STEP 2: Training S-tier simplexes")
        print("-" * 80)

        # Load model for simplex training
        print(f"\nLoading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            device_map=args.device,
        )
        model.eval()

        train_simplexes(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            output_dir=output_dir,
            validation_mode=args.validation_mode,
            all_layers=args.all_layers,
        )

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nAll lenses saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Assemble lens pack: python scripts/assemble_lens_pack.py")
    print("2. Calibrate lenses: python scripts/calibrate_lens_pack.py")
    print("3. Deploy for inference")
    print()


if __name__ == '__main__':
    main()
