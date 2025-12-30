#!/usr/bin/env python3
"""
Test Impact of Prompt+Generation Extraction on Lens Training

Hypothesis: Extracting activations from BOTH prompt processing and generation
phases can nearly double training data without additional generation time,
potentially improving lens quality.

Experiment:
    Train the same concept 3 ways:
    1. Generation-only (current method)
    2. Prompt-only (new capability)
    3. Prompt+Generation combined (hypothesis: best performance)

    Compare test F1, validation score, and sample efficiency.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.sumo_classifiers import load_layer_concepts, load_all_concepts
from training.sumo_data_generation import (
    create_sumo_training_dataset,
    build_sumo_negative_pool,
)
from training.dual_adaptive_trainer import DualAdaptiveTrainer


def extract_activations_generation_only(
    model,
    tokenizer,
    prompts,
    device="cuda",
    layer_idx=-1,
    max_new_tokens=20,
    temperature=0.7,
):
    """
    Extract activations from GENERATION phase only (current method).

    Returns: np.ndarray [n_prompts, hidden_dim]
    """
    activations = []
    model.eval()

    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Pool across all generation steps
            step_activations = []
            for step_hidden in outputs.hidden_states:
                target_layer = step_hidden[layer_idx]  # [1, seq_len, hidden_dim]
                pooled = target_layer[0].mean(dim=0)  # [hidden_dim]
                step_activations.append(pooled)

            # Mean pool across steps
            final = torch.stack(step_activations).mean(dim=0)
            activations.append(final.float().cpu().numpy())

    return np.array(activations)


def extract_activations_prompt_only(
    model,
    tokenizer,
    prompts,
    device="cuda",
    layer_idx=-1,
):
    """
    Extract activations from PROMPT phase only (new method).

    Returns: np.ndarray [n_prompts, hidden_dim]
    """
    activations = []
    model.eval()

    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Forward pass through prompt (no generation)
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            # Extract last layer states for prompt
            last_layer_states = outputs.hidden_states[layer_idx]  # [1, prompt_len, hidden_dim]

            # Mean pool across prompt positions
            pooled = last_layer_states[0].mean(dim=0)  # [hidden_dim]
            activations.append(pooled.float().cpu().numpy())

    return np.array(activations)


def extract_activations_combined(
    model,
    tokenizer,
    prompts,
    device="cuda",
    layer_idx=-1,
    max_new_tokens=20,
    temperature=0.7,
):
    """
    Extract activations from BOTH prompt and generation phases.

    For each prompt, we get TWO activation vectors:
    - One from prompt processing
    - One from generation

    This doubles our training data at minimal cost (generation already needed).

    Returns: np.ndarray [n_prompts * 2, hidden_dim]
    """
    activations = []
    model.eval()

    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]

            # PHASE 1: Extract from prompt processing
            prompt_outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            prompt_states = prompt_outputs.hidden_states[layer_idx]  # [1, prompt_len, hidden_dim]
            prompt_pooled = prompt_states[0].mean(dim=0)  # [hidden_dim]
            activations.append(prompt_pooled.float().cpu().numpy())

            # PHASE 2: Extract from generation
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Pool across generation steps
            step_activations = []
            for step_hidden in gen_outputs.hidden_states:
                target_layer = step_hidden[layer_idx]  # [1, seq_len, hidden_dim]
                pooled = target_layer[0].mean(dim=0)  # [hidden_dim]
                step_activations.append(pooled)

            gen_pooled = torch.stack(step_activations).mean(dim=0)
            activations.append(gen_pooled.float().cpu().numpy())

    return np.array(activations)


def train_concept_with_extraction_mode(
    concept_name: str,
    layer: int,
    extraction_mode: str,
    model,
    tokenizer,
    all_concepts,
    n_samples: int = 30,
    device: str = "cuda",
):
    """
    Train a concept using specified extraction mode.

    Args:
        extraction_mode: 'generation-only', 'prompt-only', or 'combined'
    """
    print(f"\n{'='*80}")
    print(f"TRAINING: {concept_name} with {extraction_mode}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Load concept
    layer_concepts, concept_map = load_layer_concepts(layer)
    target_concept = concept_map[concept_name]

    print(f"Concept: {concept_name}")
    print(f"  Synsets: {len(target_concept.get('synsets', []))}")
    print(f"  Children: {len(target_concept.get('category_children', []))}")

    # Build negative pool
    negative_pool = build_sumo_negative_pool(
        target_concept=target_concept,
        all_concepts=all_concepts,
    )
    print(f"  Negative pool: {len(negative_pool)} concepts")

    # Generate training data
    print(f"\nGenerating {n_samples} positive and negative samples...")
    prompts, labels = create_sumo_training_dataset(
        concept=target_concept,
        all_concepts=all_concepts,
        negative_pool=negative_pool,
        n_positives=n_samples,
        n_negatives=n_samples,
    )

    # Split into positive and negative
    dataset = {
        "positive_prompts": [p for p, l in zip(prompts, labels) if l == 1],
        "negative_prompts": [p for p, l in zip(prompts, labels) if l == 0],
    }

    # Extract activations based on mode
    print(f"Extracting activations ({extraction_mode})...")
    extract_start = time.time()

    if extraction_mode == "generation-only":
        pos_acts = extract_activations_generation_only(
            model, tokenizer, dataset["positive_prompts"], device=device
        )
        neg_acts = extract_activations_generation_only(
            model, tokenizer, dataset["negative_prompts"], device=device
        )
        actual_pos_samples = n_samples
        actual_neg_samples = n_samples

    elif extraction_mode == "prompt-only":
        pos_acts = extract_activations_prompt_only(
            model, tokenizer, dataset["positive_prompts"], device=device
        )
        neg_acts = extract_activations_prompt_only(
            model, tokenizer, dataset["negative_prompts"], device=device
        )
        actual_pos_samples = n_samples
        actual_neg_samples = n_samples

    elif extraction_mode == "combined":
        # Get 2x samples per prompt (prompt + generation)
        pos_acts = extract_activations_combined(
            model, tokenizer, dataset["positive_prompts"], device=device
        )
        neg_acts = extract_activations_combined(
            model, tokenizer, dataset["negative_prompts"], device=device
        )
        actual_pos_samples = n_samples * 2  # Doubled!
        actual_neg_samples = n_samples * 2

    extract_time = time.time() - extract_start

    print(f"  Extracted {len(pos_acts)} positive, {len(neg_acts)} negative")
    print(f"  Extraction time: {extract_time:.1f}s")

    # Train lens
    print("\nTraining lens...")
    trainer = DualAdaptiveTrainer(
        input_dim=pos_acts.shape[1],
        learning_rate=0.001,
        max_cycles=1,  # Single cycle for fair comparison
        min_samples=actual_pos_samples,
    )

    train_start = time.time()
    train_result = trainer.train_cycle(
        activations_pos=pos_acts,
        activations_neg=neg_acts,
    )
    train_time = time.time() - train_start

    # Test on held-out data
    print("Testing...")
    test_prompts, test_labels = create_sumo_training_dataset(
        concept=target_concept,
        all_concepts=all_concepts,
        negative_pool=negative_pool,
        n_positives=10,
        n_negatives=10,
    )

    test_pos_prompts = [p for p, l in zip(test_prompts, test_labels) if l == 1]
    test_neg_prompts = [p for p, l in zip(test_prompts, test_labels) if l == 0]

    # Test with generation-only for fair comparison
    test_pos = extract_activations_generation_only(
        model, tokenizer, test_pos_prompts, device=device
    )
    test_neg = extract_activations_generation_only(
        model, tokenizer, test_neg_prompts, device=device
    )

    test_result = trainer.test(
        activations_pos=test_pos,
        activations_neg=test_neg,
    )

    total_time = time.time() - start_time

    result = {
        "concept": concept_name,
        "layer": layer,
        "extraction_mode": extraction_mode,
        "training_samples": {
            "positive": actual_pos_samples,
            "negative": actual_neg_samples,
            "total": actual_pos_samples + actual_neg_samples,
        },
        "test_metrics": {
            "f1": test_result.get("f1", 0),
            "accuracy": test_result.get("accuracy", 0),
            "precision": test_result.get("precision", 0),
            "recall": test_result.get("recall", 0),
        },
        "timing": {
            "extraction": extract_time,
            "training": train_time,
            "total": total_time,
        },
    }

    print(f"\n{'='*80}")
    print(f"RESULTS: {extraction_mode}")
    print(f"{'='*80}")
    print(f"Samples: {actual_pos_samples + actual_neg_samples}")
    print(f"Test F1: {result['test_metrics']['f1']:.3f}")
    print(f"Test Acc: {result['test_metrics']['accuracy']:.3f}")
    print(f"Time: {total_time:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test prompt+generation extraction impact"
    )
    parser.add_argument('--concept', type=str, default='Carnivore',
                       help='Concept to test (default: Carnivore)')
    parser.add_argument('--layer', type=int, default=2,
                       help='Layer number (default: 2)')
    parser.add_argument('--n-samples', type=int, default=30,
                       help='Number of samples per mode (default: 30)')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it',
                       help='Model name (default: gemma-2-2b-it)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')

    args = parser.parse_args()

    print("=" * 80)
    print("PROMPT+GENERATION EXTRACTION EXPERIMENT")
    print("=" * 80)
    print(f"Concept: {args.concept} (Layer {args.layer})")
    print(f"Samples: {args.n_samples} per mode")
    print(f"Model: {args.model}")
    print()
    print("Testing 3 extraction modes:")
    print("  1. generation-only: Current method")
    print("  2. prompt-only: Extract from prompt processing")
    print("  3. combined: Extract from BOTH (2x data!)")

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/prompt_generation_extraction/run_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load concepts
    print("Loading concepts...")
    all_concepts = load_all_concepts()

    # Test all three modes
    modes = ["generation-only", "prompt-only", "combined"]
    results = []

    for mode in modes:
        result = train_concept_with_extraction_mode(
            concept_name=args.concept,
            layer=args.layer,
            extraction_mode=mode,
            model=model,
            tokenizer=tokenizer,
            all_concepts=all_concepts,
            n_samples=args.n_samples,
            device=args.device,
        )
        results.append(result)

        # Save individual result
        result_file = output_dir / f"{mode}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

    # Comparative analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    print(f"\n{'Mode':<20} {'Samples':<10} {'F1':<10} {'Acc':<10} {'Time (s)':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['extraction_mode']:<20} "
              f"{r['training_samples']['total']:<10} "
              f"{r['test_metrics']['f1']:<10.3f} "
              f"{r['test_metrics']['accuracy']:<10.3f} "
              f"{r['timing']['total']:<12.1f}")

    # Calculate improvements
    baseline = results[0]  # generation-only
    combined = results[2]  # combined

    f1_improvement = combined['test_metrics']['f1'] - baseline['test_metrics']['f1']
    sample_increase = combined['training_samples']['total'] / baseline['training_samples']['total']
    time_overhead = (combined['timing']['total'] - baseline['timing']['total']) / baseline['timing']['total']

    print("\n" + "=" * 80)
    print("COMBINED vs GENERATION-ONLY")
    print("=" * 80)
    print(f"Sample increase: {sample_increase:.1f}x ({combined['training_samples']['total']} vs {baseline['training_samples']['total']})")
    print(f"F1 change: {f1_improvement:+.3f} ({combined['test_metrics']['f1']:.3f} vs {baseline['test_metrics']['f1']:.3f})")
    print(f"Time overhead: {time_overhead:+.1%}")

    if f1_improvement > 0.05:
        print("\n✓ SIGNIFICANT IMPROVEMENT")
        print("  Combined extraction improves lens quality")
        print("  Recommendation: Use combined extraction in training pipeline")
    elif f1_improvement > 0.01:
        print("\n⚠️  MODEST IMPROVEMENT")
        print("  Combined extraction helps somewhat")
        print("  Consider using if time overhead acceptable")
    elif f1_improvement > -0.01:
        print("\n→ NO SIGNIFICANT DIFFERENCE")
        print("  Combined extraction doesn't hurt, gives more data")
        print("  May help with sample efficiency in adaptive training")
    else:
        print("\n✗ DEGRADATION")
        print("  Combined extraction may introduce noise")
        print("  Stick with generation-only for now")

    # Save summary
    summary = {
        "experiment_config": {
            "concept": args.concept,
            "layer": args.layer,
            "n_samples": args.n_samples,
            "model": args.model,
        },
        "results": results,
        "comparison": {
            "sample_increase": float(sample_increase),
            "f1_improvement": float(f1_improvement),
            "time_overhead": float(time_overhead),
        }
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
