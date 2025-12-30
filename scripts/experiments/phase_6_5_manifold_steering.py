#!/usr/bin/env python3
"""
Phase 6.5: Manifold-Aware Steering

Test composite steering vectors: v_steer = α·v_centroid + β·v_boundary + γ·v_curvature

Goal: Follow the semantic manifold instead of moving linearly in parameter space,
eliminating the inverted-U curve at high steering strengths.

Approach:
- v_centroid: Core concept direction (baseline)
- v_boundary: Contrastive direction (pushes away from semantic neighbors)
- v_curvature: PCA residual (captures nonlinear manifold structure)

Expected outcome: Linear Δ vs strength relationship, coherence maintained at ±1.0
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hat import (
    extract_concept_vector,
    generate_with_steering,
    build_centroids,
    compute_semantic_shift,
    apply_subspace_removal
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_composite_vector(
    model,
    tokenizer,
    concept: str,
    synonyms: List[str],
    contrastive: List[str],
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    device: str = "cuda"
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract composite steering vector with centroid, boundary, and curvature components.

    Args:
        concept: Main concept to steer
        synonyms: Related concepts for estimating curvature (e.g., ["human", "individual"])
        contrastive: Distinct concepts for boundary term (e.g., ["animal", "object"])
        alpha, beta, gamma: Mixing coefficients

    Returns:
        v_steer: Composite steering vector
        components: Dict with v_centroid, v_boundary, v_curvature
    """
    # Extract centroid (core concept)
    v_centroid = extract_concept_vector(model, tokenizer, concept, device=device)
    logger.info(f"  v_centroid norm: {np.linalg.norm(v_centroid):.3f}")

    # Extract boundary vector (contrastive)
    if beta > 0 and contrastive:
        contrastive_vectors = [
            extract_concept_vector(model, tokenizer, c, device=device)
            for c in contrastive
        ]
        v_contrast_mean = np.mean(contrastive_vectors, axis=0)
        v_boundary = v_centroid - v_contrast_mean
        v_boundary = v_boundary / (np.linalg.norm(v_boundary) + 1e-8)
        logger.info(f"  v_boundary norm: {np.linalg.norm(v_boundary):.3f}")
    else:
        v_boundary = np.zeros_like(v_centroid)

    # Extract curvature vector (PCA residual from synonyms)
    if gamma > 0 and synonyms:
        # Collect concept + synonyms
        all_vectors = [v_centroid] + [
            extract_concept_vector(model, tokenizer, syn, device=device)
            for syn in synonyms
        ]
        vectors_array = np.array(all_vectors)

        # Apply PCA-1 to remove linear structure
        clean_vectors = apply_subspace_removal(vectors_array, method="pca_1")

        # Curvature = original - linear approximation
        v_curvature = vectors_array[0] - clean_vectors[0]  # Residual for main concept

        # Normalize
        norm = np.linalg.norm(v_curvature)
        if norm > 1e-8:
            v_curvature = v_curvature / norm
            logger.info(f"  v_curvature norm: {norm:.3f} (before normalization)")
        else:
            logger.warning(f"  v_curvature is near-zero, using zero vector")
            v_curvature = np.zeros_like(v_centroid)
    else:
        v_curvature = np.zeros_like(v_centroid)

    # Compose steering vector
    v_steer = alpha * v_centroid + beta * v_boundary + gamma * v_curvature

    # Normalize composite
    v_steer = v_steer / (np.linalg.norm(v_steer) + 1e-8)

    components = {
        "v_centroid": v_centroid,
        "v_boundary": v_boundary,
        "v_curvature": v_curvature,
    }

    return v_steer, components


def compute_activation_delta_norm(
    model,
    tokenizer,
    prompt: str,
    steering_vector: np.ndarray,
    strength: float,
    layer_idx: int = -1,
    device: str = "cuda"
) -> float:
    """
    Compute ||Δactivation|| - the norm of the change in activation space.

    This measures actual movement in parameter space, not semantic space.
    If Δ vs ||Δactivation|| is linear, confirms manifold curvature hypothesis.
    """
    # Generate without steering (baseline)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        # Baseline generation
        outputs_baseline = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Get mean activation
        baseline_acts = []
        for step_states in outputs_baseline.hidden_states:
            last_layer = step_states[-1] if layer_idx == -1 else step_states[layer_idx]
            baseline_acts.append(last_layer[0, -1, :].cpu().numpy())
        baseline_mean = np.mean(baseline_acts, axis=0)

    # Generate with steering
    from src.hat import create_steering_hook
    hook_fn = create_steering_hook(steering_vector, strength, device)

    # Register hook
    # Handle different model architectures (Gemma-3 vs Gemma-2)
    if hasattr(model.model, 'language_model'):
        # Gemma-3 architecture
        layers = model.model.language_model.layers
    elif hasattr(model.model, 'layers'):
        # Gemma-2 architecture
        layers = model.model.layers
    else:
        raise AttributeError(f"Cannot find layers in model architecture: {type(model.model)}")

    target_layer = layers[-1] if layer_idx == -1 else layers[layer_idx]
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.inference_mode():
            outputs_steered = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # Get mean activation
            steered_acts = []
            for step_states in outputs_steered.hidden_states:
                last_layer = step_states[-1] if layer_idx == -1 else step_states[layer_idx]
                steered_acts.append(last_layer[0, -1, :].cpu().numpy())
            steered_mean = np.mean(steered_acts, axis=0)
    finally:
        handle.remove()

    # Compute ||Δactivation||
    delta_activation = steered_mean - baseline_mean
    delta_norm = np.linalg.norm(delta_activation)

    return delta_norm


def evaluate_manifold_steering(
    model,
    tokenizer,
    embed_model,
    concept: str,
    synonyms: List[str],
    contrastive: List[str],
    prompts: List[str],
    concept_info: dict,
    neutral_pool: List[str],
    alpha: float,
    beta: float,
    gamma: float,
    strengths: List[float],
    device: str = "cuda"
) -> Dict:
    """
    Evaluate manifold steering for one coefficient configuration.
    """
    logger.info(f"Testing (α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f})...")

    # Extract composite vector
    v_steer, components = extract_composite_vector(
        model, tokenizer, concept, synonyms, contrastive,
        alpha, beta, gamma, device
    )

    # Build semantic centroids for Δ evaluation
    core, boundary, neg = build_centroids(
        concept, concept_info, neutral_pool, embed_model
    )

    results = []

    for strength in strengths:
        logger.info(f"  Strength: {strength:+.2f}")

        for prompt in prompts:
            try:
                # Generate with manifold steering
                text = generate_with_steering(
                    model, tokenizer, prompt, v_steer, strength,
                    max_new_tokens=50, device=device
                )

                # Compute semantic shift
                delta = compute_semantic_shift(text, core, neg, embed_model)

                # Compute activation delta norm
                delta_norm = compute_activation_delta_norm(
                    model, tokenizer, prompt, v_steer, strength, device=device
                )

                results.append({
                    "concept": concept,
                    "strength": strength,
                    "prompt": prompt,
                    "text": text,
                    "delta": delta,
                    "delta_norm": delta_norm,
                    "coherent": True,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                })

            except Exception as e:
                logger.warning(f"    Failed at {strength:+.2f}: {e}")
                results.append({
                    "concept": concept,
                    "strength": strength,
                    "prompt": prompt,
                    "text": "",
                    "delta": 0.0,
                    "delta_norm": 0.0,
                    "coherent": False,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "error": str(e)
                })

    # Aggregate metrics
    coherent_results = [r for r in results if r["coherent"]]

    if coherent_results:
        mean_delta = np.mean([r["delta"] for r in coherent_results])
        mean_delta_norm = np.mean([r["delta_norm"] for r in coherent_results])
        coherence_rate = len(coherent_results) / len(results)
    else:
        mean_delta = 0.0
        mean_delta_norm = 0.0
        coherence_rate = 0.0

    summary = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "mean_delta": mean_delta,
        "mean_delta_norm": mean_delta_norm,
        "coherence_rate": coherence_rate,
        "results": results,
    }

    logger.info(f"  → Δ={mean_delta:.3f}, ||Δact||={mean_delta_norm:.3f}, coherence={coherence_rate:.1%}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 6.5: Manifold-Aware Steering")
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/phase_6_5_manifold_steering")
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Phase 6.5: Manifold-Aware Steering")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output_dir}")

    start_time = time.time()

    # Load models
    logger.info("Loading models...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,  # CRITICAL: float32 for stable extraction
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Test configuration
    concept = "person"
    synonyms = ["human", "individual"]
    contrastive = ["animal", "object"]

    prompts = [
        "Tell me about",
        "Describe the nature of",
        "What characterizes",
    ]

    strengths = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]

    # Coefficient grid
    configs = [
        # Baseline
        (1.0, 0.0, 0.0, "baseline"),
        # Boundary only
        (0.8, 0.2, 0.0, "boundary"),
        # Curvature only
        (0.8, 0.0, 0.2, "curvature"),
        # Full composite
        (0.7, 0.2, 0.1, "full"),
        (0.6, 0.2, 0.2, "full_balanced"),
    ]

    # Mock concept info and neutral pool (simplified for testing)
    concept_info = {
        "person": {
            "definition": "A human being regarded as an individual",
            "relationships": [
                {"related": "human", "type": "synonym"},
                {"related": "individual", "type": "synonym"},
            ]
        }
    }

    neutral_pool = [
        "molecule", "algorithm", "frequency", "topology",
        "wavelength", "matrix", "sequence", "protocol",
        "architecture", "framework", "infrastructure", "mechanism"
    ]

    # Run experiments
    all_results = []

    for alpha, beta, gamma, name in configs:
        logger.info(f"\n{'='*80}")
        logger.info(f"Configuration: {name} (α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f})")
        logger.info(f"{'='*80}")

        summary = evaluate_manifold_steering(
            model, tokenizer, embed_model,
            concept, synonyms, contrastive, prompts,
            concept_info, neutral_pool,
            alpha, beta, gamma, strengths,
            device=args.device
        )

        summary["config_name"] = name
        all_results.append(summary)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "manifold_steering_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n✓ Saved results to: {output_path}")

    # Print comparison
    logger.info("\n" + "="*80)
    logger.info("SUMMARY COMPARISON")
    logger.info("="*80)
    logger.info(f"{'Config':<20} {'α':<6} {'β':<6} {'γ':<6} {'Δ':<10} {'||Δact||':<10} {'Coherence':<10}")
    logger.info("-"*80)

    for result in all_results:
        logger.info(
            f"{result['config_name']:<20} "
            f"{result['alpha']:<6.2f} "
            f"{result['beta']:<6.2f} "
            f"{result['gamma']:<6.2f} "
            f"{result['mean_delta']:<10.3f} "
            f"{result['mean_delta_norm']:<10.3f} "
            f"{result['coherence_rate']:<10.1%}"
        )

    elapsed = time.time() - start_time
    logger.info(f"\nPhase 6.5 complete! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
