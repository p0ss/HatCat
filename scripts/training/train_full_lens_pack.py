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
from src.map.training.sample_quality import SampleSaver
from transformers import AutoTokenizer, AutoModelForCausalLM

# Sibling script in scripts/training/ — provides the canonical simplex training
# logic with per-pole layer selection, contrastive sampling, and lazy data
# generation. The script dir is on sys.path when this is run as a script.
from train_s_tier_simplexes import run_simplex_training, load_s_tier_simplexes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train complete SUMO lens pack with all layers and simplexes"
    )

    # Model configuration
    parser.add_argument('--model', default="google/gemma-3-4b-pt",
                        help='Model name (default: gemma-3-4b-pt)')
    parser.add_argument('--device', default="cuda",
                        help='Device (default: cuda)')
    parser.add_argument('--concept-pack', default="concept_packs/first-light",
                        help='Path to concept pack (default: concept_packs/first-light)')

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
    parser.add_argument('--save-samples', action='store_true',
                        help='Save generated training samples (with quality checks) to output_dir/samples/. '
                             'Useful for auditing prompt/response quality and for downstream CAT fine-tuning.')

    return parser.parse_args()


# load_s_tier_simplexes is imported from train_s_tier_simplexes (sibling script)
# to keep a single source of truth for the loader semantics.


def train_simplexes(
    model,
    tokenizer,
    device: str,
    output_dir: Path,
    concept_pack_path: Path,
    validation_mode: str = 'falloff',
    all_layers: bool = False,
):
    """Train all S-tier three-pole simplexes by delegating to the canonical
    implementation in train_s_tier_simplexes.run_simplex_training (which uses
    per-pole layer selection, contrastive sampling, and lazy data generation).
    """
    print("\n" + "=" * 80)
    print("TRAINING S-TIER SIMPLEXES")
    print("=" * 80)

    simplexes = load_s_tier_simplexes(concept_pack_path)
    print(f"\nFound {len(simplexes)} S-tier simplexes to train")

    simplex_dir = output_dir / "simplexes"
    return run_simplex_training(
        model=model,
        tokenizer=tokenizer,
        simplexes=simplexes,
        run_dir=simplex_dir,
        device=device,
    )


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
    print(f"Concept pack: {args.concept_pack}")
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
        'concept_pack': args.concept_pack,
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

    # Resolve hierarchy directory from concept pack
    concept_pack_path = Path(args.concept_pack)
    hierarchy_dir = concept_pack_path / "hierarchy"

    sample_saver = None
    if args.save_samples:
        sample_saver = SampleSaver(output_dir, args.concept_pack)
        print(f"Sample saving enabled: {sample_saver.output_dir}")

    train_sumo_classifiers(
        layers=args.layers,
        hierarchy_dir=hierarchy_dir,
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
        sample_saver=sample_saver,
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
            concept_pack_path=concept_pack_path,
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
