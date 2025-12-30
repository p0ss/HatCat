#!/usr/bin/env python3
"""
Phase 6.7: Steering Validation with Strength vs Dampening Grid

Tests baseline and manifold steering across:
- 32 concepts
- Strength sweep: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
- Dampening multipliers: [0.0, 0.5, 1.0, 2.0] (applied to layer-wise dampening)

Goal: Determine if manifold steering is actually working or if layer dampening kills effectiveness.
"""

import argparse
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hat.steering.manifold import ManifoldSteerer
from src.hat import extract_concept_vector, generate_with_steering


def test_baseline_steering(
    model,
    tokenizer,
    concepts: List[str],
    prompts: List[str],
    strengths: List[float],
    device: str
) -> Dict:
    """Test simple baseline steering (no manifold)."""
    print(f"\n{'='*60}")
    print("BASELINE STEERING TEST")
    print(f"{'='*60}")

    results = {}

    for concept in concepts:
        print(f"\nTesting concept: {concept}")
        v = extract_concept_vector(model, tokenizer, concept, device=device)

        concept_results = []
        for strength in strengths:
            for prompt in prompts[:2]:  # Use 2 prompts per strength
                text = generate_with_steering(
                    model, tokenizer, prompt, v, strength,
                    max_new_tokens=30, device=device
                )

                concept_results.append({
                    "strength": strength,
                    "prompt": prompt,
                    "text": text[:100]  # Store first 100 chars
                })

        results[concept] = concept_results
        print(f"  ✓ Tested {len(concept_results)} combinations")

    return results


def test_manifold_steering_grid(
    steerer: ManifoldSteerer,
    concepts: List[str],
    prompts: List[str],
    strengths: List[float],
    dampening_multipliers: List[float]
) -> Dict:
    """Test manifold steering with 2D grid of strength × dampening."""
    print(f"\n{'='*60}")
    print("MANIFOLD STEERING GRID TEST")
    print(f"{'='*60}")

    results = {}

    for concept in concepts:
        print(f"\nTesting concept: {concept}")
        concept_results = []

        for damp_mult in dampening_multipliers:
            for strength in strengths:
                for prompt in prompts[:2]:  # Use 2 prompts per config
                    try:
                        # Modify max_norm to scale dampening effect
                        # Higher max_norm = less dampening
                        max_norm = 1.0 * (1.0 + damp_mult)

                        text = steerer.generate(
                            prompt=prompt,
                            concept=concept,
                            strength=strength,
                            max_new_tokens=30,
                            max_norm_per_layer=max_norm
                        )

                        concept_results.append({
                            "strength": strength,
                            "dampening_mult": damp_mult,
                            "max_norm": max_norm,
                            "prompt": prompt,
                            "text": text[:100]
                        })
                    except Exception as e:
                        print(f"    Error at strength={strength}, damp={damp_mult}: {e}")

        results[concept] = concept_results
        print(f"  ✓ Tested {len(concept_results)} combinations")

    return results


def analyze_diversity(results: Dict) -> Dict:
    """Analyze output diversity to detect if steering is working."""
    analysis = {}

    for concept, outputs in results.items():
        texts = [r["text"] for r in outputs]
        unique_texts = len(set(texts))
        total_texts = len(texts)
        diversity_ratio = unique_texts / total_texts if total_texts > 0 else 0.0

        analysis[concept] = {
            "total_outputs": total_texts,
            "unique_outputs": unique_texts,
            "diversity_ratio": diversity_ratio,
            "working": diversity_ratio > 0.3  # If >30% unique, steering is working
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Phase 6.7: Steering Validation")
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/phase_6_7_steering_validation")
    parser.add_argument("--n-concepts", type=int, default=32,
                        help="Number of concepts to test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PHASE 6.7: STEERING VALIDATION")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Concepts: {args.n_concepts}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # float32 for numerical stability
        device_map=args.device
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded\n")

    # Test concepts (first 32 from typical list)
    all_concepts = [
        "person", "change", "animal", "object", "action", "time", "place",
        "quality", "relation", "number", "thought", "emotion", "truth",
        "knowledge", "language", "society", "culture", "nature", "life",
        "death", "power", "freedom", "justice", "beauty", "art", "science",
        "technology", "religion", "morality", "identity", "consciousness",
        "reality"
    ]
    concepts = all_concepts[:args.n_concepts]

    # Test parameters
    prompts = [
        "Tell me about",
        "Describe",
        "What is",
        "Consider",
        "Explain"
    ]

    strengths = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    dampening_multipliers = [0.0, 0.5, 1.0, 2.0]  # 0=heavy damp, 2=light damp

    # Test 1: Baseline steering
    print("\n" + "="*60)
    print("TEST 1: BASELINE STEERING")
    print("="*60)
    baseline_results = test_baseline_steering(
        model, tokenizer, concepts, prompts, strengths, args.device
    )
    baseline_analysis = analyze_diversity(baseline_results)

    # Test 2: Manifold steering (fit first)
    print("\n" + "="*60)
    print("Fitting manifold steerer...")
    print("="*60)
    steerer = ManifoldSteerer(model, tokenizer, device=args.device)
    start_fit = time.time()
    steerer.fit(concepts=concepts, n_manifold_samples=4)  # Use 4 samples for speed
    fit_time = time.time() - start_fit
    print(f"✓ Fitting complete: {fit_time:.1f}s\n")

    print("\n" + "="*60)
    print("TEST 2: MANIFOLD STEERING GRID")
    print("="*60)
    manifold_results = test_manifold_steering_grid(
        steerer, concepts, prompts, strengths, dampening_multipliers
    )
    manifold_analysis = analyze_diversity(manifold_results)

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    baseline_working = sum(1 for a in baseline_analysis.values() if a["working"])
    manifold_working = sum(1 for a in manifold_analysis.values() if a["working"])

    print(f"\nBaseline Steering:")
    print(f"  Working concepts: {baseline_working}/{len(concepts)}")
    print(f"  Mean diversity: {np.mean([a['diversity_ratio'] for a in baseline_analysis.values()]):.2%}")

    print(f"\nManifold Steering:")
    print(f"  Working concepts: {manifold_working}/{len(concepts)}")
    print(f"  Mean diversity: {np.mean([a['diversity_ratio'] for a in manifold_analysis.values()]):.2%}")

    # Save results
    results = {
        "config": {
            "model": args.model,
            "n_concepts": args.n_concepts,
            "concepts": concepts,
            "strengths": strengths,
            "dampening_multipliers": dampening_multipliers,
            "fit_time_seconds": fit_time
        },
        "baseline": {
            "results": baseline_results,
            "analysis": baseline_analysis
        },
        "manifold": {
            "results": manifold_results,
            "analysis": manifold_analysis
        },
        "summary": {
            "baseline_working": baseline_working,
            "manifold_working": manifold_working,
            "baseline_mean_diversity": float(np.mean([a['diversity_ratio'] for a in baseline_analysis.values()])),
            "manifold_mean_diversity": float(np.mean([a['diversity_ratio'] for a in manifold_analysis.values()]))
        }
    }

    output_file = output_dir / "validation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
