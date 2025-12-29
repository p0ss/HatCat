#!/usr/bin/env python3
"""
Quick test: Train one simplex at layer 10 and check if poles discriminate.
"""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from map.training.sumo_data_generation import create_simplex_pole_training_dataset_contrastive
from map.training.sumo_classifiers import train_simple_classifier, extract_activations
import json

S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"


def generate_pole_data(simplex_name, simplex_def, pole_name, n_samples=100):
    """Generate training data for a pole (balanced pos/neg)."""
    pole_data = simplex_def[pole_name]
    pole_type = pole_name.split('_')[0]

    other_pole_names = [p for p in ['negative_pole', 'neutral_homeostasis', 'positive_pole'] if p != pole_name]
    other_poles_data = [{**simplex_def[p], 'pole_type': p.split('_')[0]} for p in other_pole_names]

    prompts, labels = create_simplex_pole_training_dataset_contrastive(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=simplex_name,
        other_poles_data=other_poles_data,
        behavioral_ratio=0.6,
        prompts_per_synset=5
    )

    # Balance the dataset - take equal positive and negative
    pos_prompts = [(p, l) for p, l in zip(prompts, labels) if l == 1]
    neg_prompts = [(p, l) for p, l in zip(prompts, labels) if l == 0]

    n_each = min(len(pos_prompts), len(neg_prompts), n_samples // 2)
    balanced = pos_prompts[:n_each] + neg_prompts[:n_each]
    np.random.shuffle(balanced)

    prompts_out = [p for p, l in balanced]
    labels_out = [l for p, l in balanced]
    return prompts_out, np.array(labels_out)


def main():
    print("=" * 70)
    print("LAYER 10 vs LAYER 12 COMPARISON")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model_name = "google/gemma-3-4b-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
        device_map="auto",
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    # Load simplex
    with open(S_TIER_DEFS_PATH) as f:
        simplexes = json.load(f)['simplexes']

    simplex_name = 'threat_perception'
    simplex_def = simplexes[simplex_name]

    print(f"\nTesting on: {simplex_name}")

    # Generate data for negative pole (alarm)
    print("\nGenerating training data for NEGATIVE pole...")
    train_prompts, train_labels = generate_pole_data(
        simplex_name, simplex_def, 'negative_pole', n_samples=100
    )
    test_prompts, test_labels = generate_pole_data(
        simplex_name, simplex_def, 'negative_pole', n_samples=40
    )

    print(f"  Train: {len(train_prompts)} samples, {sum(train_labels)} positive")
    print(f"  Test: {len(test_prompts)} samples, {sum(test_labels)} positive")

    # Test early/mid layers (31+ is too late - captures text not motives)
    for layer_idx in [6, 8, 10, 15, 20]:
        print(f"\n{'='*50}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*50}")

        # Extract activations
        print("  Extracting train activations...")
        train_acts = extract_activations(
            model, tokenizer, train_prompts,
            layer_idx=layer_idx, device=device,
            extraction_mode="prompt", pooling="last",
            batch_size=2  # Small batch to avoid OOM
        )
        train_acts = torch.tensor(train_acts, dtype=torch.float32)

        print("  Extracting test activations...")
        test_acts = extract_activations(
            model, tokenizer, test_prompts,
            layer_idx=layer_idx, device=device,
            extraction_mode="prompt", pooling="last",
            batch_size=2
        )
        test_acts = torch.tensor(test_acts, dtype=torch.float32)

        # Train classifier
        print("  Training classifier...")
        classifier, metrics = train_simple_classifier(
            train_acts.numpy(), train_labels,
            test_acts.numpy(), test_labels,
            normalize_inputs=True,
            epochs=100,
            lr=0.001
        )

        print(f"\n  Results:")
        print(f"    Train F1: {metrics.get('train_f1', 0):.3f}")
        print(f"    Test F1:  {metrics.get('test_f1', 0):.3f}")
        print(f"    Test Acc: {metrics.get('test_accuracy', 0):.3f}")

        # Quick sanity check - run on ACTUAL prompts from training data
        # Use experiential prompts that we know are in the training distribution
        print("\n  Sanity check on experiential prompts:")
        sanity_prompts = [
            "Something terrible is about to happen",  # Positive: alarm
            "My heart is pounding and I need to escape",  # Positive: alarm
            "I feel completely safe and protected",   # Negative: safety (other pole)
            "I'm aware of my surroundings without fear",  # Negative: vigilance (other pole)
            "What is mathematics?",  # Negative: general unrelated
        ]

        sanity_acts = extract_activations(
            model, tokenizer, sanity_prompts,
            layer_idx=layer_idx, device=device,
            extraction_mode="prompt", pooling="last",
            batch_size=1
        )
        sanity_acts = torch.tensor(sanity_acts, dtype=torch.float32)

        # Normalize like training
        mean = sanity_acts.mean(dim=-1, keepdim=True)
        std = sanity_acts.std(dim=-1, keepdim=True) + 1e-8
        sanity_acts_norm = (sanity_acts - mean) / std

        classifier.eval()
        classifier = classifier.float().cpu()  # Convert to float32 and move to CPU
        with torch.no_grad():
            logits = classifier(sanity_acts_norm.float())
            probs = torch.sigmoid(logits).numpy().flatten()

        for prompt, prob in zip(sanity_prompts, probs):
            # First two are alarm (positive), rest are negative
            expected = "HIGH" if "terrible" in prompt or "pounding" in prompt else "low"
            actual = "HIGH" if prob > 0.5 else "low"
            match = "✓" if expected == actual else "✗"
            print(f"    {match} '{prompt[:40]}...': {prob:.3f} (expect {expected})")


if __name__ == "__main__":
    main()
