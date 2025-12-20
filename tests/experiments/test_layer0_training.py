#!/usr/bin/env python3
"""
Test training layer 0 concepts with validation.

Since layer 0 concepts ARE the validation domains, this tests
self-consistency (a Process lens should fire on Process prompts).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.map.training.sumo_classifiers import train_layer


def main():
    print("=" * 80)
    print("TESTING LAYER 0 TRAINING WITH VALIDATION")
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

    # Train a few layer 0 concepts with adaptive training + validation
    print("\nTraining layer 0 with adaptive training and validation...")
    output_dir = Path("results/test_layer0_validation")

    summary = train_layer(
        layer=0,
        model=model,
        tokenizer=tokenizer,
        n_train_pos=10,
        n_train_neg=10,
        n_test_pos=20,
        n_test_neg=20,
        device=device,
        output_dir=output_dir,
        save_text_samples=False,
        use_adaptive_training=True,
        train_text_lenses=False,  # Skip text lenses for quick test
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Layer {summary['layer']}: {summary['n_successful']}/{summary['n_concepts']} concepts")
    print(f"Average Test F1: {summary['avg_test_f1']:.3f}")

    # Show validation results
    import json
    results_file = output_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            results_data = json.load(f)

        print("\nValidation Results:")
        for result in results_data['results']:
            if 'validation_passed' in result:
                status = "✓" if result['validation_passed'] else "✗"
                print(f"  {status} {result['concept']:20s} "
                      f"score={result['validation_calibration_score']:.2f} "
                      f"target=#{result['validation_target_rank']} "
                      f"others={result['validation_avg_other_rank']:.1f}")
            else:
                print(f"  ? {result['concept']:20s} (no validation)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
