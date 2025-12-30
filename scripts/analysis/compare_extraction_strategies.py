#!/usr/bin/env python3
"""
Compare Extraction Strategies: Prompt+Generation vs Longer Tokens

Tests TWO strategies for getting more training data:
  A) Extract from prompt + generation (2x data, same tokens)
  B) Generate longer (40 tokens instead of 20)

Tests on TWO types of concepts:
  1) Abstract concept (e.g., Attribute - Layer 0)
  2) Specific concept (e.g., Carnivore - Layer 2)

Hypothesis: Prompt+generation should outperform longer tokens because
core concepts activate early (during prompt), not in extended generation.
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


def extract_generation_only(model, tokenizer, prompts, max_tokens=20, device="cuda"):
    """Extract from generation only."""
    acts = []
    model.eval()

    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Pool across generation steps
            step_acts = []
            for step_hidden in outputs.hidden_states:
                pooled = step_hidden[-1][0].mean(dim=0)
                step_acts.append(pooled)

            final = torch.stack(step_acts).mean(dim=0)
            acts.append(final.float().cpu().numpy())

    return np.array(acts)


def extract_prompt_and_generation(model, tokenizer, prompts, max_tokens=20, device="cuda"):
    """Extract from BOTH prompt and generation (2x samples)."""
    acts = []
    model.eval()

    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # PROMPT PHASE
            prompt_outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            prompt_pooled = prompt_outputs.hidden_states[-1][0].mean(dim=0)
            acts.append(prompt_pooled.float().cpu().numpy())

            # GENERATION PHASE
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            step_acts = []
            for step_hidden in gen_outputs.hidden_states:
                pooled = step_hidden[-1][0].mean(dim=0)
                step_acts.append(pooled)

            gen_pooled = torch.stack(step_acts).mean(dim=0)
            acts.append(gen_pooled.float().cpu().numpy())

    return np.array(acts)


def train_with_strategy(
    concept_name,
    layer,
    strategy,  # 'baseline-20', 'combined-20', or 'long-40'
    model,
    tokenizer,
    all_concepts,
    n_samples=30,
    device="cuda",
):
    """Train concept with specified strategy."""
    print(f"\n{'='*80}")
    print(f"{concept_name} - {strategy}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Load concept
    layer_concepts, concept_map = load_layer_concepts(layer)
    concept = concept_map[concept_name]
    negative_pool = build_sumo_negative_pool(all_concepts, concept)

    print(f"Concept: {concept_name} (Layer {layer})")
    print(f"  Synsets: {len(concept.get('synsets', []))}")
    print(f"  Negatives: {len(negative_pool)}")

    # Generate prompts
    prompts, labels = create_sumo_training_dataset(
        concept=concept,
        all_concepts=all_concepts,
        negative_pool=negative_pool,
        n_positives=n_samples,
        n_negatives=n_samples,
    )

    pos_prompts = [p for p, l in zip(prompts, labels) if l == 1]
    neg_prompts = [p for p, l in zip(prompts, labels) if l == 0]

    # Extract based on strategy
    print(f"Extracting with {strategy}...")
    extract_start = time.time()

    if strategy == 'baseline-20':
        pos_acts = extract_generation_only(model, tokenizer, pos_prompts, max_tokens=20, device=device)
        neg_acts = extract_generation_only(model, tokenizer, neg_prompts, max_tokens=20, device=device)

    elif strategy == 'combined-20':
        pos_acts = extract_prompt_and_generation(model, tokenizer, pos_prompts, max_tokens=20, device=device)
        neg_acts = extract_prompt_and_generation(model, tokenizer, neg_prompts, max_tokens=20, device=device)

    elif strategy == 'long-40':
        pos_acts = extract_generation_only(model, tokenizer, pos_prompts, max_tokens=40, device=device)
        neg_acts = extract_generation_only(model, tokenizer, neg_prompts, max_tokens=40, device=device)

    extract_time = time.time() - extract_start

    print(f"  Samples: {len(pos_acts)} pos, {len(neg_acts)} neg")
    print(f"  Extract time: {extract_time:.1f}s")

    # Train simple classifier
    print("Training...")
    from training.sumo_classifiers import train_simple_classifier

    # Split for train/val
    split = int(len(pos_acts) * 0.8)
    X_train = np.vstack([pos_acts[:split], neg_acts[:split]])
    y_train = np.array([1] * split + [0] * split)
    X_val = np.vstack([pos_acts[split:], neg_acts[split:]])
    y_val = np.array([1] * (len(pos_acts) - split) + [0] * (len(neg_acts) - split))

    train_start = time.time()
    lens, train_metrics = train_simple_classifier(
        X_train, y_train, X_val, y_val,
        hidden_dim=128, epochs=50, lr=0.001
    )
    train_time = time.time() - train_start

    # Test (always use baseline-20 for fair comparison)
    print("Testing...")
    test_prompts, test_labels = create_sumo_training_dataset(
        concept=concept,
        all_concepts=all_concepts,
        negative_pool=negative_pool,
        n_positives=10,
        n_negatives=10,
    )

    test_pos = [p for p, l in zip(test_prompts, test_labels) if l == 1]
    test_neg = [p for p, l in zip(test_prompts, test_labels) if l == 0]

    test_pos_acts = extract_generation_only(model, tokenizer, test_pos, max_tokens=20, device=device)
    test_neg_acts = extract_generation_only(model, tokenizer, test_neg, max_tokens=20, device=device)

    # Test
    X_test = np.vstack([test_pos_acts, test_neg_acts])
    y_test = np.array([1] * len(test_pos_acts) + [0] * len(test_neg_acts))

    # Evaluate
    lens.eval()
    lens = lens.cpu()  # Move to CPU for simplicity
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        y_tensor = torch.LongTensor(y_test)
        outputs = lens(X_tensor)
        preds = (outputs.squeeze() > 0.5).long()

        from sklearn.metrics import f1_score, accuracy_score
        test_result = {
            "f1": f1_score(y_test, preds.numpy()),
            "accuracy": accuracy_score(y_test, preds.numpy())
        }

    total_time = time.time() - start_time

    result = {
        "concept": concept_name,
        "layer": layer,
        "strategy": strategy,
        "samples": len(pos_acts) + len(neg_acts),
        "test_f1": test_result.get("f1", 0),
        "test_acc": test_result.get("accuracy", 0),
        "extract_time": extract_time,
        "train_time": train_time,
        "total_time": total_time,
    }

    print(f"Test F1: {result['test_f1']:.3f}")
    print(f"Test Acc: {result['test_acc']:.3f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--abstract-concept', type=str, default='Attribute',
                       help='Abstract concept to test (default: Attribute)')
    parser.add_argument('--abstract-layer', type=int, default=0,
                       help='Layer for abstract concept (default: 0)')
    parser.add_argument('--specific-concept', type=str, default='Carnivore',
                       help='Specific concept to test (default: Carnivore)')
    parser.add_argument('--specific-layer', type=int, default=2,
                       help='Layer for specific concept (default: 2)')
    parser.add_argument('--n-samples', type=int, default=30,
                       help='Training samples (default: 30)')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("=" * 80)
    print("EXTRACTION STRATEGY COMPARISON")
    print("=" * 80)
    print("\nStrategies:")
    print("  baseline-20: Generation only, 20 tokens")
    print("  combined-20: Prompt + generation, 20 tokens (2x samples!)")
    print("  long-40: Generation only, 40 tokens")
    print("\nConcepts:")
    print(f"  Abstract: {args.abstract_concept} (L{args.abstract_layer})")
    print(f"  Specific: {args.specific_concept} (L{args.specific_layer})")

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/extraction_strategy_comparison/run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading concepts...")
    all_concepts = load_all_concepts()

    # Run all combinations
    concepts = [
        (args.abstract_concept, args.abstract_layer, "abstract"),
        (args.specific_concept, args.specific_layer, "specific"),
    ]

    strategies = ['baseline-20', 'combined-20', 'long-40']

    all_results = []

    for concept_name, layer, concept_type in concepts:
        print(f"\n{'#'*80}")
        print(f"{concept_type.upper()} CONCEPT: {concept_name}")
        print(f"{'#'*80}")

        for strategy in strategies:
            result = train_with_strategy(
                concept_name=concept_name,
                layer=layer,
                strategy=strategy,
                model=model,
                tokenizer=tokenizer,
                all_concepts=all_concepts,
                n_samples=args.n_samples,
                device=args.device,
            )
            result['concept_type'] = concept_type
            all_results.append(result)

    # Analysis
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for concept_type in ['abstract', 'specific']:
        print(f"\n{concept_type.upper()} CONCEPT:")
        print("-" * 80)
        print(f"{'Strategy':<15} {'Samples':<10} {'F1':<10} {'Acc':<10} {'Time (s)':<10}")
        print("-" * 80)

        concept_results = [r for r in all_results if r['concept_type'] == concept_type]
        baseline = [r for r in concept_results if r['strategy'] == 'baseline-20'][0]

        for r in concept_results:
            print(f"{r['strategy']:<15} "
                  f"{r['samples']:<10} "
                  f"{r['test_f1']:<10.3f} "
                  f"{r['test_acc']:<10.3f} "
                  f"{r['total_time']:<10.1f}")

        # Compare to baseline
        combined = [r for r in concept_results if r['strategy'] == 'combined-20'][0]
        long = [r for r in concept_results if r['strategy'] == 'long-40'][0]

        print(f"\nvs baseline-20:")
        print(f"  combined-20: F1 {combined['test_f1'] - baseline['test_f1']:+.3f}, "
              f"Time {(combined['total_time'] / baseline['total_time'] - 1) * 100:+.0f}%")
        print(f"  long-40: F1 {long['test_f1'] - baseline['test_f1']:+.3f}, "
              f"Time {(long['total_time'] / baseline['total_time'] - 1) * 100:+.0f}%")

    # Overall winner
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    abstract_results = [r for r in all_results if r['concept_type'] == 'abstract']
    specific_results = [r for r in all_results if r['concept_type'] == 'specific']

    abstract_combined_delta = [r for r in abstract_results if r['strategy'] == 'combined-20'][0]['test_f1'] - \
                               [r for r in abstract_results if r['strategy'] == 'baseline-20'][0]['test_f1']

    specific_combined_delta = [r for r in specific_results if r['strategy'] == 'combined-20'][0]['test_f1'] - \
                               [r for r in specific_results if r['strategy'] == 'baseline-20'][0]['test_f1']

    avg_combined_delta = (abstract_combined_delta + specific_combined_delta) / 2

    if avg_combined_delta > 0.05:
        print("✓ STRONG WIN for combined-20")
        print(f"  Average F1 improvement: +{avg_combined_delta:.3f}")
        print("  Recommendation: Use prompt+generation extraction")
    elif avg_combined_delta > 0.01:
        print("⚠️  MODEST WIN for combined-20")
        print(f"  Average F1 improvement: +{avg_combined_delta:.3f}")
        print("  Recommendation: Consider prompt+generation for sample efficiency")
    else:
        print("→ NO CLEAR WINNER")
        print(f"  Average F1 delta: {avg_combined_delta:+.3f}")
        print("  Recommendation: Stick with baseline-20 for simplicity")

    # Save
    summary = {
        "config": vars(args),
        "results": all_results,
        "timestamp": timestamp,
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved to {output_dir}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
