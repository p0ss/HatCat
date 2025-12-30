#!/usr/bin/env python3
"""
Cross-Strategy Extraction Comparison

Tests all lenses against all extraction methods to find:
1. Which training strategy produces the best lenses?
2. Which lenses generalize best across extraction methods?

Creates a 3x3 matrix of results:
- Rows: Training strategies (baseline-20, combined-20, long-40)
- Cols: Test extraction methods (baseline-20, combined-20, long-40)

Uses 100 test samples for statistical power.
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


def extract_with_strategy(model, tokenizer, prompts, strategy, device="cuda"):
    """Extract activations using specified strategy."""
    if strategy == 'baseline-20':
        return extract_generation_only(model, tokenizer, prompts, max_tokens=20, device=device)
    elif strategy == 'combined-20':
        return extract_prompt_and_generation(model, tokenizer, prompts, max_tokens=20, device=device)
    elif strategy == 'long-40':
        return extract_generation_only(model, tokenizer, prompts, max_tokens=40, device=device)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def train_lens(
    concept_name,
    layer,
    train_strategy,
    model,
    tokenizer,
    all_concepts,
    n_samples=30,
    device="cuda",
):
    """Train a lens using specified training strategy."""
    print(f"\nTraining {concept_name} with {train_strategy}...")

    # Load concept
    layer_concepts, concept_map = load_layer_concepts(layer)
    concept = concept_map[concept_name]
    negative_pool = build_sumo_negative_pool(all_concepts, concept)

    # Generate training prompts
    prompts, labels = create_sumo_training_dataset(
        concept=concept,
        all_concepts=all_concepts,
        negative_pool=negative_pool,
        n_positives=n_samples,
        n_negatives=n_samples,
    )

    pos_prompts = [p for p, l in zip(prompts, labels) if l == 1]
    neg_prompts = [p for p, l in zip(prompts, labels) if l == 0]

    # Extract training activations
    pos_acts = extract_with_strategy(model, tokenizer, pos_prompts, train_strategy, device)
    neg_acts = extract_with_strategy(model, tokenizer, neg_prompts, train_strategy, device)

    print(f"  Training samples: {len(pos_acts)} pos, {len(neg_acts)} neg")

    # Train classifier
    from training.sumo_classifiers import train_simple_classifier

    # Split for train/val
    split = int(len(pos_acts) * 0.8)
    X_train = np.vstack([pos_acts[:split], neg_acts[:split]])
    y_train = np.array([1] * split + [0] * split)
    X_val = np.vstack([pos_acts[split:], neg_acts[split:]])
    y_val = np.array([1] * (len(pos_acts) - split) + [0] * (len(neg_acts) - split))

    lens, train_metrics = train_simple_classifier(
        X_train, y_train, X_val, y_val,
        hidden_dim=128, epochs=50, lr=0.001
    )

    return lens, concept, negative_pool


def test_lens(
    lens,
    concept,
    negative_pool,
    test_strategy,
    model,
    tokenizer,
    all_concepts,
    n_test=100,
    device="cuda",
):
    """Test lens using specified test extraction strategy."""
    # Generate test prompts
    test_prompts, test_labels = create_sumo_training_dataset(
        concept=concept,
        all_concepts=all_concepts,
        negative_pool=negative_pool,
        n_positives=n_test,
        n_negatives=n_test,
    )

    test_pos = [p for p, l in zip(test_prompts, test_labels) if l == 1]
    test_neg = [p for p, l in zip(test_prompts, test_labels) if l == 0]

    # Extract test activations with specified strategy
    test_pos_acts = extract_with_strategy(model, tokenizer, test_pos, test_strategy, device)
    test_neg_acts = extract_with_strategy(model, tokenizer, test_neg, test_strategy, device)

    X_test = np.vstack([test_pos_acts, test_neg_acts])
    y_test = np.array([1] * len(test_pos_acts) + [0] * len(test_neg_acts))

    # Evaluate
    lens.eval()
    lens = lens.cpu()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        outputs = lens(X_tensor)
        preds = (outputs.squeeze() > 0.5).long()

        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(y_test, preds.numpy())
        acc = accuracy_score(y_test, preds.numpy())

    return f1, acc


def run_cross_strategy_test(
    concept_name,
    layer,
    model,
    tokenizer,
    all_concepts,
    n_train=30,
    n_test=100,
    device="cuda",
):
    """Run full cross-strategy comparison for one concept."""
    print(f"\n{'='*80}")
    print(f"CONCEPT: {concept_name} (Layer {layer})")
    print(f"{'='*80}")

    strategies = ['baseline-20', 'combined-20', 'long-40']

    # Train all lenses
    lenses = {}
    concept_data = None
    negative_pool = None

    for train_strategy in strategies:
        lens, concept, neg_pool = train_lens(
            concept_name=concept_name,
            layer=layer,
            train_strategy=train_strategy,
            model=model,
            tokenizer=tokenizer,
            all_concepts=all_concepts,
            n_samples=n_train,
            device=device,
        )
        lenses[train_strategy] = lens
        concept_data = concept
        negative_pool = neg_pool

    # Test all lenses against all extraction methods
    print(f"\nTesting all combinations (3x3 = 9 tests)...")
    results = []

    for train_strategy in strategies:
        for test_strategy in strategies:
            print(f"  Testing {train_strategy} lens on {test_strategy} extraction...")
            f1, acc = test_lens(
                lens=lenses[train_strategy],
                concept=concept_data,
                negative_pool=negative_pool,
                test_strategy=test_strategy,
                model=model,
                tokenizer=tokenizer,
                all_concepts=all_concepts,
                n_test=n_test,
                device=device,
            )

            results.append({
                "train_strategy": train_strategy,
                "test_strategy": test_strategy,
                "f1": f1,
                "accuracy": acc,
            })

            print(f"    F1: {f1:.3f}, Acc: {acc:.3f}")

    return results


def print_results_matrix(results, concept_name):
    """Print results as a matrix."""
    print(f"\n{'='*80}")
    print(f"RESULTS MATRIX: {concept_name}")
    print(f"{'='*80}")

    strategies = ['baseline-20', 'combined-20', 'long-40']

    # Create matrix
    matrix = {}
    for r in results:
        key = (r['train_strategy'], r['test_strategy'])
        matrix[key] = r['f1']

    # Print header
    print(f"\n{'Train \\ Test':<20} ", end="")
    for test_strat in strategies:
        print(f"{test_strat:<15}", end="")
    print("  Avg")
    print("-" * 80)

    # Print rows
    for train_strat in strategies:
        print(f"{train_strat:<20} ", end="")
        row_scores = []
        for test_strat in strategies:
            score = matrix[(train_strat, test_strat)]
            row_scores.append(score)
            print(f"{score:<15.3f}", end="")
        avg = np.mean(row_scores)
        print(f"{avg:.3f}")

    # Print column averages
    print("-" * 80)
    print(f"{'Avg':<20} ", end="")
    for test_strat in strategies:
        col_scores = [matrix[(ts, test_strat)] for ts in strategies]
        print(f"{np.mean(col_scores):<15.3f}", end="")
    print()


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
    parser.add_argument('--n-train', type=int, default=30,
                       help='Training samples per class (default: 30)')
    parser.add_argument('--n-test', type=int, default=100,
                       help='Test samples per class (default: 100)')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("=" * 80)
    print("CROSS-STRATEGY EXTRACTION COMPARISON")
    print("=" * 80)
    print("\nTraining Strategies:")
    print("  baseline-20: Generation only, 20 tokens")
    print("  combined-20: Prompt + generation, 20 tokens (2x samples)")
    print("  long-40: Generation only, 40 tokens")
    print("\nTest Strategies:")
    print("  Each lens tested against all 3 extraction methods")
    print(f"\nTest size: {args.n_test} positives + {args.n_test} negatives = {args.n_test * 2} samples")
    print("\nConcepts:")
    print(f"  Abstract: {args.abstract_concept} (L{args.abstract_layer})")
    print(f"  Specific: {args.specific_concept} (L{args.specific_layer})")

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/extraction_strategy_cross/run_{timestamp}")
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

    # Test both concepts
    all_results = {}

    # Abstract concept
    print(f"\n{'#'*80}")
    print(f"ABSTRACT CONCEPT: {args.abstract_concept}")
    print(f"{'#'*80}")
    abstract_results = run_cross_strategy_test(
        concept_name=args.abstract_concept,
        layer=args.abstract_layer,
        model=model,
        tokenizer=tokenizer,
        all_concepts=all_concepts,
        n_train=args.n_train,
        n_test=args.n_test,
        device=args.device,
    )
    all_results['abstract'] = abstract_results
    print_results_matrix(abstract_results, args.abstract_concept)

    # Specific concept
    print(f"\n{'#'*80}")
    print(f"SPECIFIC CONCEPT: {args.specific_concept}")
    print(f"{'#'*80}")
    specific_results = run_cross_strategy_test(
        concept_name=args.specific_concept,
        layer=args.specific_layer,
        model=model,
        tokenizer=tokenizer,
        all_concepts=all_concepts,
        n_train=args.n_train,
        n_test=args.n_test,
        device=args.device,
    )
    all_results['specific'] = specific_results
    print_results_matrix(specific_results, args.specific_concept)

    # Overall analysis
    print(f"\n{'='*80}")
    print("OVERALL ANALYSIS")
    print(f"{'='*80}")

    strategies = ['baseline-20', 'combined-20', 'long-40']

    # Which training strategy produces best lenses? (average across all test methods)
    print("\nBest Training Strategy (by average F1 across test methods):")
    for strat in strategies:
        abstract_scores = [r['f1'] for r in abstract_results if r['train_strategy'] == strat]
        specific_scores = [r['f1'] for r in specific_results if r['train_strategy'] == strat]
        avg = np.mean(abstract_scores + specific_scores)
        print(f"  {strat}: {avg:.3f}")

    # Which lenses generalize best? (smallest variance across test methods)
    print("\nMost Generalizable (lowest variance across test methods):")
    for strat in strategies:
        abstract_scores = [r['f1'] for r in abstract_results if r['train_strategy'] == strat]
        specific_scores = [r['f1'] for r in specific_results if r['train_strategy'] == strat]
        variance = np.var(abstract_scores + specific_scores)
        print(f"  {strat}: variance = {variance:.4f}")

    # Diagonal performance (matched test/train)
    print("\nDiagonal Performance (matched test/train extraction):")
    for strat in strategies:
        abstract_diag = [r['f1'] for r in abstract_results
                        if r['train_strategy'] == strat and r['test_strategy'] == strat][0]
        specific_diag = [r['f1'] for r in specific_results
                        if r['train_strategy'] == strat and r['test_strategy'] == strat][0]
        avg = (abstract_diag + specific_diag) / 2
        print(f"  {strat}: {avg:.3f}")

    # Save results
    summary = {
        "config": vars(args),
        "results": {
            "abstract": abstract_results,
            "specific": specific_results,
        },
        "timestamp": timestamp,
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
