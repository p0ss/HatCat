#!/usr/bin/env python3
"""
Phase 6.6: Dual-Subspace Manifold Steering Test

Tests the actual dual-subspace manifold steering implementation from manifold.py,
comparing it to baseline projection steering at high strengths (±1.0).

Expected outcome: Manifold-aware steering maintains coherence at high strengths
where baseline steering degrades (inverted-U curve).
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

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hat.steering.manifold import ManifoldSteerer
from src.hat.steering.hooks import generate_with_steering
from src.hat.steering.extraction import extract_concept_vector
from src.hat.steering.evaluation import build_centroids, compute_semantic_shift

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_coherent(text: str) -> bool:
    """
    Check if generated text is coherent (not gibberish).

    Simple heuristics:
    - Has at least 10 characters
    - Contains at least one space
    - Not all same character
    - Contains some common words
    """
    if len(text) < 10:
        return False
    if ' ' not in text:
        return False
    if len(set(text)) < 5:  # Too repetitive
        return False

    # Check for some common words
    common_words = {'the', 'a', 'is', 'of', 'to', 'and', 'in', 'that', 'it', 'for', 'on', 'with', 'as'}
    words = text.lower().split()
    if not any(word in common_words for word in words):
        return False

    return True


def evaluate_steering_method(
    model,
    tokenizer,
    embed_model,
    concept: str,
    concept_vector: np.ndarray,
    prompts: List[str],
    strengths: List[float],
    concept_info: dict,
    neutral_pool: List[str],
    method: str,  # "baseline" or "manifold"
    manifold_steerer = None,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate one steering method across multiple prompts and strengths.
    """
    logger.info(f"Evaluating {method} steering...")

    # Build semantic centroids
    core, boundary, neg = build_centroids(
        concept, concept_info, neutral_pool, embed_model
    )

    results = []

    for strength in strengths:
        logger.info(f"  Strength: {strength:+.2f}")

        for prompt in prompts:
            try:
                # Generate with steering
                if method == "baseline":
                    text = generate_with_steering(
                        model, tokenizer, prompt, concept_vector, strength,
                        max_new_tokens=50, device=device
                    )
                elif method == "manifold":
                    text = manifold_steerer.generate(
                        prompt=prompt,
                        concept=concept,
                        strength=strength,
                        max_new_tokens=50
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Compute metrics
                delta = compute_semantic_shift(text, core, neg, embed_model)
                coherent = is_coherent(text)

                results.append({
                    "concept": concept,
                    "method": method,
                    "strength": strength,
                    "prompt": prompt,
                    "text": text,
                    "delta": delta,
                    "coherent": coherent,
                })

                logger.info(f"    Δ={delta:.3f}, coherent={coherent}")

            except Exception as e:
                logger.warning(f"    Failed at {strength:+.2f}: {e}")
                results.append({
                    "concept": concept,
                    "method": method,
                    "strength": strength,
                    "prompt": prompt,
                    "text": "",
                    "delta": 0.0,
                    "coherent": False,
                    "error": str(e)
                })

    # Aggregate metrics by strength
    by_strength = {}
    for r in results:
        s = r["strength"]
        if s not in by_strength:
            by_strength[s] = []
        by_strength[s].append(r)

    strength_stats = []
    for s in sorted(by_strength.keys()):
        group = by_strength[s]
        coherent_group = [r for r in group if r["coherent"]]

        if coherent_group:
            mean_delta = np.mean([r["delta"] for r in coherent_group])
        else:
            mean_delta = 0.0

        coherence_rate = len(coherent_group) / len(group) if group else 0.0

        strength_stats.append({
            "strength": s,
            "mean_delta": mean_delta,
            "coherence_rate": coherence_rate,
            "n_samples": len(group),
            "n_coherent": len(coherent_group)
        })

    summary = {
        "method": method,
        "concept": concept,
        "strength_stats": strength_stats,
        "all_results": results
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 6.6: Dual-Subspace Manifold Steering")
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--concepts", nargs='+', default=["person", "change", "animal"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/phase_6_6_dual_subspace")
    parser.add_argument("--n-manifold-samples", type=int, default=10,
                        help="Samples per concept for manifold estimation")
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Phase 6.6: Dual-Subspace Manifold Steering")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Concepts: {args.concepts}")
    logger.info(f"Output: {args.output_dir}")

    start_time = time.time()

    # Load models
    logger.info("Loading models...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,  # float32 for stable extraction
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Test configuration
    prompts = [
        "Tell me about",
        "Describe the nature of",
        "What characterizes",
    ]

    strengths = [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]

    # Mock concept info and neutral pool
    concept_info = {}
    for c in args.concepts:
        concept_info[c] = {
            "definition": f"A {c}",
            "relationships": []
        }

    neutral_pool = [
        "molecule", "algorithm", "frequency", "topology",
        "wavelength", "matrix", "sequence", "protocol",
        "architecture", "framework", "infrastructure", "mechanism"
    ]

    # Initialize manifold steerer and fit subspaces
    logger.info("\n" + "="*80)
    logger.info("FITTING DUAL-SUBSPACE MANIFOLD STEERER")
    logger.info("="*80)

    manifold_steerer = ManifoldSteerer(model, tokenizer, device=args.device)
    manifold_steerer.fit(
        concepts=args.concepts,
        n_manifold_samples=args.n_manifold_samples
    )

    logger.info("✓ Manifold steerer fitted!")

    # Run experiments for each concept
    all_results = []

    for concept in args.concepts:
        logger.info("\n" + "="*80)
        logger.info(f"TESTING CONCEPT: {concept}")
        logger.info("="*80)

        # Extract concept vector for baseline
        logger.info("Extracting concept vector for baseline...")
        concept_vector = extract_concept_vector(model, tokenizer, concept, device=args.device)

        # Test baseline steering
        logger.info("\n--- Baseline Projection Steering ---")
        baseline_results = evaluate_steering_method(
            model, tokenizer, embed_model,
            concept, concept_vector, prompts, strengths,
            concept_info, neutral_pool,
            method="baseline",
            device=args.device
        )
        all_results.append(baseline_results)

        # Test manifold steering
        logger.info("\n--- Dual-Subspace Manifold Steering ---")
        manifold_results = evaluate_steering_method(
            model, tokenizer, embed_model,
            concept, concept_vector, prompts, strengths,
            concept_info, neutral_pool,
            method="manifold",
            manifold_steerer=manifold_steerer,
            device=args.device
        )
        all_results.append(manifold_results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "dual_subspace_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n✓ Saved results to: {output_path}")

    # Print comparison
    logger.info("\n" + "="*80)
    logger.info("SUMMARY COMPARISON")
    logger.info("="*80)

    for result in all_results:
        logger.info(f"\nConcept: {result['concept']} | Method: {result['method']}")
        logger.info("-"*80)
        logger.info(f"{'Strength':<10} {'Mean Δ':<10} {'Coherence':<12} {'n_coherent/n_total':<20}")
        logger.info("-"*80)

        for stat in result['strength_stats']:
            logger.info(
                f"{stat['strength']:+.2f}       "
                f"{stat['mean_delta']:<10.3f} "
                f"{stat['coherence_rate']:<12.1%} "
                f"{stat['n_coherent']}/{stat['n_samples']}"
            )

    # Compare coherence at high strengths
    logger.info("\n" + "="*80)
    logger.info("KEY FINDING: Coherence at High Strengths (±1.0)")
    logger.info("="*80)

    for concept in args.concepts:
        baseline_res = next(r for r in all_results if r['concept'] == concept and r['method'] == 'baseline')
        manifold_res = next(r for r in all_results if r['concept'] == concept and r['method'] == 'manifold')

        # Get coherence at strength=1.0
        baseline_high = next(s for s in baseline_res['strength_stats'] if abs(s['strength'] - 1.0) < 0.01)
        manifold_high = next(s for s in manifold_res['strength_stats'] if abs(s['strength'] - 1.0) < 0.01)

        logger.info(f"\n{concept}:")
        logger.info(f"  Baseline coherence at +1.0:  {baseline_high['coherence_rate']:.1%}")
        logger.info(f"  Manifold coherence at +1.0:  {manifold_high['coherence_rate']:.1%}")

        improvement = manifold_high['coherence_rate'] - baseline_high['coherence_rate']
        if improvement > 0.1:
            logger.info(f"  ✓ Manifold steering improves coherence by {improvement:.1%}")
        elif improvement < -0.1:
            logger.info(f"  ✗ Manifold steering reduces coherence by {abs(improvement):.1%}")
        else:
            logger.info(f"  → No significant difference")

    elapsed = time.time() - start_time
    logger.info(f"\nPhase 6.6 complete! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
