#!/usr/bin/env python3
"""
Compare feature weighting vs trained classifiers for concept discrimination.

This test compares two approaches:
1. ConceptActivationMapper: Statistical IDF-weighted centroids (no training)
2. Trained binary classifiers: Current pipeline output

For each approach, we measure:
- Top-1 accuracy: Does the correct concept rank first?
- Top-5 accuracy: Is the correct concept in top 5?
- Mean reciprocal rank (MRR): Average of 1/rank for correct concept
- Hierarchical consistency: Do parent concepts activate when children do?

Usage:
    python -m training.calibration.compare_approaches \
        --concept-pack concept_packs/first-light \
        --lens-pack lens_packs/apertus-8b_first-light \
        --model swiss-ai/Apertus-8B-2509 \
        --n-concepts 200 \
        --n-prompts 5
"""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .feature_weighting import ConceptActivationMapper


@dataclass
class EvalResult:
    """Evaluation results for one approach."""
    approach: str
    top1_accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    mrr: float  # Mean reciprocal rank
    mean_rank: float
    hierarchical_consistency: float
    inference_time_ms: float
    n_samples: int


def load_classifier(lens_path: Path, hidden_dim: int, device: str) -> nn.Module:
    """Load a trained lens classifier."""
    state = torch.load(lens_path, map_location=device)
    classifier = nn.Sequential(
        nn.Linear(hidden_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1)
    ).to(device)
    classifier.load_state_dict(state)
    classifier.eval()
    return classifier


def sample_concepts(
    concept_pack_dir: Path,
    n_per_layer: int = 30,
    layers: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Sample concepts across layers for testing.

    Returns list of concept dicts with sumo_term, layer, parent_concepts, training_hints.
    """
    if layers is None:
        layers = list(range(7))

    sampled = []
    hierarchy_dir = concept_pack_dir / "hierarchy"

    for layer in layers:
        layer_file = hierarchy_dir / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        concepts = layer_data.get('concepts', layer_data if isinstance(layer_data, list) else [])

        # Filter to concepts with training hints (better prompts)
        with_hints = [c for c in concepts if c.get('training_hints', {}).get('positive_examples')]

        # If not enough with hints, use all
        pool = with_hints if len(with_hints) >= n_per_layer else concepts

        # Sample
        n_sample = min(n_per_layer, len(pool))
        selected = random.sample(pool, n_sample)

        for c in selected:
            c['_layer'] = layer  # Store layer for easy access

        sampled.extend(selected)

    return sampled


def generate_prompts_for_concept(concept: Dict, n_prompts: int = 5) -> List[str]:
    """Generate diverse prompts for a concept."""
    term = concept.get('sumo_term') or concept.get('term', 'unknown')
    hints = concept.get('training_hints', {})

    prompts = []

    # Use positive examples from training hints
    pos_examples = hints.get('positive_examples', [])
    prompts.extend(pos_examples[:n_prompts])

    # Generate additional prompts if needed
    if len(prompts) < n_prompts:
        templates = [
            f"Tell me about {term}.",
            f"What is {term}?",
            f"Explain {term} in detail.",
            f"The concept of {term} refers to",
            f"Define {term}.",
        ]
        for t in templates:
            if len(prompts) >= n_prompts:
                break
            prompts.append(t)

    return prompts[:n_prompts]


def extract_activation(
    model,
    tokenizer,
    prompt: str,
    device: str,
    layer_idx: int = 15,
) -> np.ndarray:
    """Extract activation for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_idx]
        activation = hidden_states[0, -1, :].float().cpu().numpy()

    return activation


def evaluate_mapper(
    mapper: ConceptActivationMapper,
    test_data: List[Tuple[str, int, np.ndarray]],  # (concept, layer, activation)
    concept_to_parents: Dict[str, List[str]],
) -> EvalResult:
    """Evaluate ConceptActivationMapper on test data."""

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    reciprocal_ranks = []
    ranks = []
    hierarchical_checks = []

    start_time = time.time()

    for concept, layer, activation in test_data:
        # Get activation map
        act_map = mapper.compute_activations(activation)

        # Find rank of correct concept
        rank = None
        for i, a in enumerate(act_map.activations):
            if a.concept == concept:
                rank = i + 1  # 1-indexed
                break

        if rank is None:
            rank = len(act_map.activations)  # Not found = worst rank

        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)

        if rank == 1:
            top1_correct += 1
        if rank <= 5:
            top5_correct += 1
        if rank <= 10:
            top10_correct += 1

        # Check hierarchical consistency
        # If this concept is active, its parents should be too (or ranked higher)
        parents = concept_to_parents.get(concept, [])
        for parent in parents:
            if parent in act_map.by_concept:
                parent_rank = None
                for i, a in enumerate(act_map.activations):
                    if a.concept == parent:
                        parent_rank = i + 1
                        break
                if parent_rank:
                    # Parent should rank similar or higher than child
                    # (allow some slack - within 50 ranks)
                    hierarchical_checks.append(1 if parent_rank <= rank + 50 else 0)

    elapsed = (time.time() - start_time) * 1000  # ms
    n = len(test_data)

    return EvalResult(
        approach="ConceptActivationMapper",
        top1_accuracy=top1_correct / n,
        top5_accuracy=top5_correct / n,
        top10_accuracy=top10_correct / n,
        mrr=np.mean(reciprocal_ranks),
        mean_rank=np.mean(ranks),
        hierarchical_consistency=np.mean(hierarchical_checks) if hierarchical_checks else 1.0,
        inference_time_ms=elapsed / n,
        n_samples=n,
    )


def evaluate_classifiers(
    classifiers: Dict[str, nn.Module],
    concept_layers: Dict[str, int],
    test_data: List[Tuple[str, int, np.ndarray]],
    concept_to_parents: Dict[str, List[str]],
    device: str,
) -> EvalResult:
    """Evaluate trained classifiers on test data."""

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    reciprocal_ranks = []
    ranks = []
    hierarchical_checks = []

    start_time = time.time()

    for concept, layer, activation in test_data:
        # Score with all classifiers
        act_tensor = torch.tensor(activation, dtype=torch.float32).to(device)

        scores = []
        for cls_concept, classifier in classifiers.items():
            with torch.no_grad():
                score = classifier(act_tensor).item()
            scores.append((cls_concept, score, concept_layers.get(cls_concept, 0)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Find rank of correct concept
        rank = None
        for i, (c, s, l) in enumerate(scores):
            if c == concept:
                rank = i + 1
                break

        if rank is None:
            rank = len(scores)

        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)

        if rank == 1:
            top1_correct += 1
        if rank <= 5:
            top5_correct += 1
        if rank <= 10:
            top10_correct += 1

        # Hierarchical consistency
        parents = concept_to_parents.get(concept, [])
        for parent in parents:
            parent_rank = None
            for i, (c, s, l) in enumerate(scores):
                if c == parent:
                    parent_rank = i + 1
                    break
            if parent_rank:
                hierarchical_checks.append(1 if parent_rank <= rank + 50 else 0)

    elapsed = (time.time() - start_time) * 1000
    n = len(test_data)

    return EvalResult(
        approach="Trained Classifiers",
        top1_accuracy=top1_correct / n,
        top5_accuracy=top5_correct / n,
        top10_accuracy=top10_correct / n,
        mrr=np.mean(reciprocal_ranks),
        mean_rank=np.mean(ranks),
        hierarchical_consistency=np.mean(hierarchical_checks) if hierarchical_checks else 1.0,
        inference_time_ms=elapsed / n,
        n_samples=n,
    )


def run_comparison(
    model,
    tokenizer,
    concept_pack_dir: Path,
    lens_pack_dir: Path,
    device: str,
    n_concepts: int = 200,
    n_prompts_per_concept: int = 5,
    train_ratio: float = 0.6,  # 60% for mapper training, 40% for testing
    layer_idx: int = 15,
    layers: Optional[List[int]] = None,
) -> Tuple[EvalResult, EvalResult, Dict]:
    """
    Run full comparison between approaches.

    Returns:
        (mapper_result, classifier_result, metadata)
    """
    if layers is None:
        layers = [0, 1, 2, 3, 4, 5, 6]

    hidden_dim = model.config.hidden_size

    print("=" * 70)
    print("FEATURE WEIGHTING vs CLASSIFIERS COMPARISON")
    print("=" * 70)

    # 1. Sample concepts
    print(f"\n1. Sampling {n_concepts} concepts across layers {layers}...")
    n_per_layer = max(1, n_concepts // len(layers))
    concepts = sample_concepts(concept_pack_dir, n_per_layer, layers)
    print(f"   Sampled {len(concepts)} concepts")

    # Build parent mapping
    concept_to_parents = {}
    for c in concepts:
        term = c.get('sumo_term') or c.get('term')
        parents = c.get('parent_concepts', [])
        if parents:
            concept_to_parents[term] = parents

    # 2. Generate prompts and collect activations
    print(f"\n2. Collecting activations ({n_prompts_per_concept} prompts per concept)...")
    all_data = []  # (concept, layer, activation, prompt)

    for concept in tqdm(concepts, desc="Collecting"):
        term = concept.get('sumo_term') or concept.get('term')
        layer = concept.get('_layer', 0)
        prompts = generate_prompts_for_concept(concept, n_prompts_per_concept)

        for prompt in prompts:
            try:
                activation = extract_activation(model, tokenizer, prompt, device, layer_idx)
                all_data.append((term, layer, activation, prompt))
            except Exception as e:
                print(f"   Error on {term}: {e}")
                continue

    print(f"   Collected {len(all_data)} samples")

    # 3. Split into train/test
    random.shuffle(all_data)
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    print(f"\n3. Split: {len(train_data)} train, {len(test_data)} test")

    # 4. Build ConceptActivationMapper from training data
    print("\n4. Building ConceptActivationMapper from training data...")

    # Group training data by concept
    train_by_concept: Dict[str, Tuple[int, List[np.ndarray]]] = {}
    for term, layer, activation, prompt in train_data:
        if term not in train_by_concept:
            train_by_concept[term] = (layer, [])
        train_by_concept[term][1].append(activation)

    mapper = ConceptActivationMapper(hidden_dim=hidden_dim)
    mapper.fit(train_by_concept)

    # 5. Load trained classifiers
    print("\n5. Loading trained classifiers...")
    classifiers = {}
    concept_layers = {}

    # Only load classifiers for concepts in our test set
    test_concepts = set(term for term, layer, activation, prompt in test_data)

    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if not layer_dir.exists():
            continue

        for lens_path in layer_dir.glob("*_classifier.pt"):
            concept = lens_path.stem.replace("_classifier", "")
            if concept in test_concepts or concept in train_by_concept:
                try:
                    classifiers[concept] = load_classifier(lens_path, hidden_dim, device)
                    concept_layers[concept] = layer
                except Exception as e:
                    print(f"   Failed to load {concept}: {e}")

    print(f"   Loaded {len(classifiers)} classifiers")

    # 6. Evaluate both approaches on test data
    print("\n6. Evaluating on test data...")

    # Filter test data to concepts we have classifiers for
    test_data_filtered = [
        (term, layer, activation)
        for term, layer, activation, prompt in test_data
        if term in classifiers
    ]
    print(f"   Test samples with classifiers: {len(test_data_filtered)}")

    print("\n   Evaluating ConceptActivationMapper...")
    mapper_result = evaluate_mapper(mapper, test_data_filtered, concept_to_parents)

    print("   Evaluating Trained Classifiers...")
    classifier_result = evaluate_classifiers(
        classifiers, concept_layers, test_data_filtered, concept_to_parents, device
    )

    # 7. Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    def print_result(r: EvalResult):
        print(f"\n  {r.approach}:")
        print(f"    Top-1 Accuracy:  {r.top1_accuracy:.1%}")
        print(f"    Top-5 Accuracy:  {r.top5_accuracy:.1%}")
        print(f"    Top-10 Accuracy: {r.top10_accuracy:.1%}")
        print(f"    MRR:             {r.mrr:.3f}")
        print(f"    Mean Rank:       {r.mean_rank:.1f}")
        print(f"    Hierarchical:    {r.hierarchical_consistency:.1%}")
        print(f"    Inference Time:  {r.inference_time_ms:.2f} ms/sample")

    print_result(mapper_result)
    print_result(classifier_result)

    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)

    top1_diff = mapper_result.top1_accuracy - classifier_result.top1_accuracy
    speed_ratio = classifier_result.inference_time_ms / mapper_result.inference_time_ms

    if top1_diff > 0.05:
        print(f"  Mapper wins by {top1_diff:.1%} on Top-1 accuracy")
    elif top1_diff < -0.05:
        print(f"  Classifiers win by {-top1_diff:.1%} on Top-1 accuracy")
    else:
        print(f"  Approaches are comparable (diff: {top1_diff:+.1%})")

    print(f"  Mapper is {speed_ratio:.1f}x faster per sample")

    metadata = {
        'n_concepts_sampled': len(concepts),
        'n_train_samples': len(train_data),
        'n_test_samples': len(test_data_filtered),
        'n_classifiers': len(classifiers),
        'layers': layers,
        'train_ratio': train_ratio,
    }

    return mapper_result, classifier_result, metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare feature weighting vs classifiers')
    parser.add_argument('--concept-pack', required=True, help='Path to concept pack')
    parser.add_argument('--lens-pack', required=True, help='Path to lens pack')
    parser.add_argument('--model', required=True, help='Model name/path')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--n-concepts', type=int, default=200, help='Number of concepts to sample')
    parser.add_argument('--n-prompts', type=int, default=5, help='Prompts per concept')
    parser.add_argument('--train-ratio', type=float, default=0.6, help='Ratio for mapper training')
    parser.add_argument('--layer-idx', type=int, default=15, help='Model layer for activations')
    parser.add_argument('--layers', nargs='+', type=int, default=None, help='Hierarchy layers to test')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    mapper_result, classifier_result, metadata = run_comparison(
        model=model,
        tokenizer=tokenizer,
        concept_pack_dir=Path(args.concept_pack),
        lens_pack_dir=Path(args.lens_pack),
        device=args.device,
        n_concepts=args.n_concepts,
        n_prompts_per_concept=args.n_prompts,
        train_ratio=args.train_ratio,
        layer_idx=args.layer_idx,
        layers=args.layers,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.lens_pack) / "comparison_results.json"

    results = {
        'mapper': {
            'approach': mapper_result.approach,
            'top1_accuracy': mapper_result.top1_accuracy,
            'top5_accuracy': mapper_result.top5_accuracy,
            'top10_accuracy': mapper_result.top10_accuracy,
            'mrr': mapper_result.mrr,
            'mean_rank': mapper_result.mean_rank,
            'hierarchical_consistency': mapper_result.hierarchical_consistency,
            'inference_time_ms': mapper_result.inference_time_ms,
        },
        'classifiers': {
            'approach': classifier_result.approach,
            'top1_accuracy': classifier_result.top1_accuracy,
            'top5_accuracy': classifier_result.top5_accuracy,
            'top10_accuracy': classifier_result.top10_accuracy,
            'mrr': classifier_result.mrr,
            'mean_rank': classifier_result.mean_rank,
            'hierarchical_consistency': classifier_result.hierarchical_consistency,
            'inference_time_ms': classifier_result.inference_time_ms,
        },
        'metadata': metadata,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
