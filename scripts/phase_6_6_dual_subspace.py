#!/usr/bin/env python3
"""
Phase 6.6: Dual-Subspace Manifold Steering

Test the combination of:
1. Contamination removal (Phase 6)
2. Task manifold projection (Huang et al.)
3. Layer-wise dampening

Goal: Achieve stable, monotonic Δ vs strength at ±1.0+ with ≥90% coherence

Expected outcomes:
- If curvature dominant: Δ vs strength becomes linear
- If contamination dominant: Coherence improves at extremes
- If both: Linear Δ AND high coherence at ±1.0

Validation metrics:
1. Δ vs strength: Should be monotonic (no inverted-U)
2. Δ vs ||Δactivation||: Should straighten (confirms manifold following)
3. Coherence at ±1.0: Should stay ≥90%
4. Neutral baseline Δ: Should stay low and stable
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.steering.manifold import ManifoldSteerer
from src.steering import build_centroids, compute_semantic_shift

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_baseline_steering(
    model,
    tokenizer,
    embed_model,
    concept: str,
    concept_info: dict,
    neutral_pool: List[str],
    prompts: List[str],
    strengths: List[float],
    device: str = "cuda"
) -> Dict:
    """
    Evaluate baseline steering (no manifold projection) for comparison.
    """
    from src.steering import extract_concept_vector, generate_with_steering

    logger.info("Evaluating baseline steering (no manifold projection)...")

    v = extract_concept_vector(model, tokenizer, concept, device=device)

    # Build semantic centroids
    core, boundary, neg = build_centroids(concept, concept_info, neutral_pool, embed_model)

    results = []
    for strength in strengths:
        logger.info(f"  Baseline strength: {strength:+.2f}")

        for prompt in prompts:
            try:
                text = generate_with_steering(
                    model, tokenizer, prompt, v, strength,
                    max_new_tokens=50, device=device
                )

                delta = compute_semantic_shift(text, core, neg, embed_model)

                results.append({
                    "strength": strength,
                    "prompt": prompt,
                    "text": text,
                    "delta": delta,
                    "coherent": True
                })

            except Exception as e:
                logger.warning(f"    Failed at {strength:+.2f}: {e}")
                results.append({
                    "strength": strength,
                    "prompt": prompt,
                    "text": "",
                    "delta": 0.0,
                    "coherent": False,
                    "error": str(e)
                })

    # Aggregate
    coherent_results = [r for r in results if r["coherent"]]
    mean_delta = np.mean([r["delta"] for r in coherent_results]) if coherent_results else 0.0
    coherence_rate = len(coherent_results) / len(results) if results else 0.0

    summary = {
        "method": "baseline",
        "mean_delta": mean_delta,
        "coherence_rate": coherence_rate,
        "results": results
    }

    logger.info(f"  → Baseline: Δ={mean_delta:.3f}, coherence={coherence_rate:.1%}")

    return summary


def evaluate_manifold_steering(
    steerer: ManifoldSteerer,
    embed_model,
    concept: str,
    concept_info: dict,
    neutral_pool: List[str],
    prompts: List[str],
    strengths: List[float],
    max_norm_per_layer: float = 1.0
) -> Dict:
    """
    Evaluate dual-subspace manifold steering.
    """
    logger.info("Evaluating manifold steering (dual-subspace)...")

    # Build semantic centroids
    core, boundary, neg = build_centroids(concept, concept_info, neutral_pool, embed_model)

    results = []
    for strength in strengths:
        logger.info(f"  Manifold strength: {strength:+.2f}")

        for prompt in prompts:
            try:
                text = steerer.generate(
                    prompt=prompt,
                    concept=concept,
                    strength=strength,
                    max_new_tokens=50,
                    max_norm_per_layer=max_norm_per_layer
                )

                delta = compute_semantic_shift(text, core, neg, embed_model)

                results.append({
                    "strength": strength,
                    "prompt": prompt,
                    "text": text,
                    "delta": delta,
                    "coherent": True
                })

            except Exception as e:
                logger.warning(f"    Failed at {strength:+.2f}: {e}")
                results.append({
                    "strength": strength,
                    "prompt": prompt,
                    "text": "",
                    "delta": 0.0,
                    "coherent": False,
                    "error": str(e)
                })

    # Aggregate
    coherent_results = [r for r in results if r["coherent"]]
    mean_delta = np.mean([r["delta"] for r in coherent_results]) if coherent_results else 0.0
    coherence_rate = len(coherent_results) / len(results) if results else 0.0

    summary = {
        "method": "manifold",
        "mean_delta": mean_delta,
        "coherence_rate": coherence_rate,
        "results": results
    }

    logger.info(f"  → Manifold: Δ={mean_delta:.3f}, coherence={coherence_rate:.1%}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 6.6: Dual-Subspace Manifold Steering")
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/phase_6_6_dual_subspace")
    parser.add_argument("--n-manifold-samples", type=int, default=10,
                        help="Samples for task manifold estimation")
    parser.add_argument("--max-norm", type=float, default=1.0,
                        help="Max norm per layer")
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Phase 6.6: Dual-Subspace Manifold Steering")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Manifold samples: {args.n_manifold_samples}")
    logger.info(f"Max norm per layer: {args.max_norm}")

    start_time = time.time()

    # Load models
    logger.info("Loading models...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # CRITICAL: float32 for stable extraction
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Test configuration
    concepts = ["person", "change"]  # Start with 2 concepts
    test_concept = "person"

    prompts = [
        "Tell me about",
        "Describe the nature of",
        "What characterizes",
    ]

    # CRITICAL: Test at high strengths to validate manifold steering
    strengths = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]

    # Mock concept info and neutral pool
    concept_info = {
        "person": {
            "definition": "A human being regarded as an individual",
            "relationships": [
                {"related": "human", "type": "synonym"},
                {"related": "individual", "type": "synonym"},
            ]
        },
        "change": {
            "definition": "To make or become different",
            "relationships": [
                {"related": "transform", "type": "synonym"},
                {"related": "modify", "type": "synonym"},
            ]
        }
    }

    neutral_pool = [
        "molecule", "algorithm", "frequency", "topology",
        "wavelength", "matrix", "sequence", "protocol",
        "architecture", "framework", "infrastructure", "mechanism"
    ]

    # Initialize manifold steerer
    logger.info("\n" + "="*80)
    logger.info("Fitting dual-subspace manifold steering...")
    logger.info("="*80)

    steerer = ManifoldSteerer(model, tokenizer, device=args.device)
    steerer.fit(
        concepts=concepts,
        n_manifold_samples=args.n_manifold_samples
    )

    # Evaluate baseline
    logger.info("\n" + "="*80)
    logger.info("Baseline Evaluation")
    logger.info("="*80)

    baseline_summary = evaluate_baseline_steering(
        model, tokenizer, embed_model,
        test_concept, concept_info, neutral_pool,
        prompts, strengths, device=args.device
    )

    # Evaluate manifold steering
    logger.info("\n" + "="*80)
    logger.info("Manifold Steering Evaluation")
    logger.info("="*80)

    manifold_summary = evaluate_manifold_steering(
        steerer, embed_model,
        test_concept, concept_info, neutral_pool,
        prompts, strengths, max_norm_per_layer=args.max_norm
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "model": args.model,
            "concepts": concepts,
            "test_concept": test_concept,
            "strengths": strengths,
            "n_manifold_samples": args.n_manifold_samples,
            "max_norm_per_layer": args.max_norm
        },
        "baseline": baseline_summary,
        "manifold": manifold_summary
    }

    output_path = output_dir / "dual_subspace_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Saved results to: {output_path}")

    # Print comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: Baseline vs Manifold Steering")
    logger.info("="*80)
    logger.info(f"{'Method':<20} {'Mean Δ':<15} {'Coherence':<15}")
    logger.info("-"*80)
    logger.info(
        f"{'Baseline':<20} "
        f"{baseline_summary['mean_delta']:<15.3f} "
        f"{baseline_summary['coherence_rate']:<15.1%}"
    )
    logger.info(
        f"{'Manifold':<20} "
        f"{manifold_summary['mean_delta']:<15.3f} "
        f"{manifold_summary['coherence_rate']:<15.1%}"
    )

    # Analyze by strength
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS: Coherence by Strength")
    logger.info("="*80)
    logger.info(f"{'Strength':<15} {'Baseline':<15} {'Manifold':<15} {'Improvement':<15}")
    logger.info("-"*80)

    for strength in strengths:
        baseline_results = [r for r in baseline_summary["results"] if r["strength"] == strength]
        manifold_results = [r for r in manifold_summary["results"] if r["strength"] == strength]

        baseline_coherence = sum(1 for r in baseline_results if r["coherent"]) / len(baseline_results) if baseline_results else 0
        manifold_coherence = sum(1 for r in manifold_results if r["coherent"]) / len(manifold_results) if manifold_results else 0

        improvement = manifold_coherence - baseline_coherence

        logger.info(
            f"{strength:<15.2f} "
            f"{baseline_coherence:<15.1%} "
            f"{manifold_coherence:<15.1%} "
            f"{improvement:+.1%}"
        )

    elapsed = time.time() - start_time
    logger.info(f"\nPhase 6.6 complete! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
