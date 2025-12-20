#!/usr/bin/env python3
"""
Validate all trained lenses to identify poorly-calibrated classifiers.

Usage:
    python scripts/validate_trained_lenses.py --layer 1
    python scripts/validate_trained_lenses.py --lens-pack gemma-3-4b-pt_sumo-wordnet-v1 --layer 1
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.map.training.lens_validation import validate_lens_set


def main():
    parser = argparse.ArgumentParser(
        description="Validate trained lenses for calibration"
    )
    parser.add_argument('--lens-pack', type=str, default='gemma-3-4b-pt_sumo-wordnet-v1',
                       help='Lens pack ID')
    parser.add_argument('--layer', type=int, required=True,
                       help='Layer to validate (e.g., 1)')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--layer-idx', type=int, default=15,
                       help='Model layer index for activations')

    args = parser.parse_args()

    print("=" * 80)
    print("LENS VALIDATION")
    print("=" * 80)
    print(f"Lens pack: {args.lens_pack}")
    print(f"Layer: {args.layer}")
    print(f"Model: {args.model}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=args.device,
        local_files_only=True
    )
    model.eval()
    print("✓ Model loaded")
    print()

    # Load lens pack metadata
    from src.map.lens_pack import load_lens_pack
    try:
        lens_pack = load_lens_pack(args.lens_pack, auto_pull=False)
    except Exception as e:
        print(f"ERROR: Lens pack '{args.lens_pack}' not found: {e}")
        return 1

    # Get lenses directory (all lenses in one dir, we'll filter by layer)
    lens_dir = lens_pack.activation_lenses_dir

    if not lens_dir.exists():
        print(f"ERROR: Lens directory not found: {lens_dir}")
        return 1

    # Load layer metadata
    layer_data_file = Path("data/concept_graph/abstraction_layers") / f"layer{args.layer}.json"

    if not layer_data_file.exists():
        print(f"ERROR: Layer metadata not found: {layer_data_file}")
        return 1

    with open(layer_data_file) as f:
        layer_data = json.load(f)

    # Get concepts list
    concepts = layer_data['concepts']
    print(f"Loaded {len(concepts)} concept definitions for layer {args.layer}")
    print()

    # Run validation
    results = validate_lens_set(
        lens_dir=lens_dir,
        concepts=concepts,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        layer_idx=args.layer_idx,
        save_results=True,
    )

    # Exit with failure if pass rate is too low
    if results['pass_rate'] < 0.75:
        print(f"\n⚠️  WARNING: Pass rate {results['pass_rate']:.1%} is below 75%")
        print("Consider retraining with better negative examples")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
