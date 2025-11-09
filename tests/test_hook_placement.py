#!/usr/bin/env python3
"""
Test Hook Placement for Steering Effectiveness

Tests two hook placements with proper statistical sampling:
1. Current: After layer (before next layer's RMSNorm)
2. Proposed: Before MLP (steering participates in nonlinearity)

Metrics:
- Embedding similarity between ±strength outputs
- Diversity across multiple samples
- Semantic shift (Δ) from baseline
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.steering import extract_concept_vector
from src.utils.gpu_cleanup import cleanup_model, print_gpu_memory


def compute_embedding_similarity(model, tokenizer, text1: str, text2: str, device: str) -> float:
    """Compute cosine similarity between text embeddings."""
    def get_embedding(text: str):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            if hasattr(model.model, 'language_model'):
                embeds = model.model.language_model.embed_tokens(inputs.input_ids)
            else:
                embeds = model.model.embed_tokens(inputs.input_ids)
            return embeds.mean(dim=1).cpu().numpy()[0]

    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)

    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)

    return float(np.dot(emb1, emb2))


def compute_semantic_shift(model, tokenizer, text: str, concept: str, device: str) -> float:
    """Compute Δ = cos(text, concept) - cos(text, 'nothing')."""
    def get_embedding(phrase: str):
        inputs = tokenizer(phrase, return_tensors="pt").to(device)
        with torch.inference_mode():
            if hasattr(model.model, 'language_model'):
                embeds = model.model.language_model.embed_tokens(inputs.input_ids)
            else:
                embeds = model.model.embed_tokens(inputs.input_ids)
            return embeds.mean(dim=1).cpu().numpy()[0]

    text_v = get_embedding(text)
    conc_v = get_embedding(concept)
    neg_v = get_embedding("nothing")

    text_v = text_v / (np.linalg.norm(text_v) + 1e-8)
    conc_v = conc_v / (np.linalg.norm(conc_v) + 1e-8)
    neg_v = neg_v / (np.linalg.norm(neg_v) + 1e-8)

    return float(np.dot(text_v, conc_v) - np.dot(text_v, neg_v))


def test_hook_placement(
    model, tokenizer, concept_vector: np.ndarray,
    placement: str, concept: str, strengths: list,
    prompts: list, device: str
):
    """Test a specific hook placement with multiple samples."""

    if hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers

    target_layer = layers[-1]
    v_tensor = torch.from_numpy(concept_vector).float().to(device)

    results = []

    for strength in strengths:
        for prompt in prompts:
            if placement == "after_layer":
                # Current: Hook after layer
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    v_matched = v_tensor.to(dtype=hidden.dtype)
                    projection = (hidden @ v_matched.unsqueeze(-1)) * v_matched
                    steered = hidden - (strength * projection)
                    return (steered,) if isinstance(output, tuple) else steered

                handle = target_layer.register_forward_hook(hook_fn)

            elif placement == "before_mlp":
                # Proposed: Hook before MLP
                original_mlp_forward = target_layer.mlp.forward

                def hook_mlp(hidden_states):
                    v_matched = v_tensor.to(dtype=hidden_states.dtype)
                    projection = (hidden_states @ v_matched.unsqueeze(-1)) * v_matched
                    steered = hidden_states - (strength * projection)
                    return original_mlp_forward(steered)

                target_layer.mlp.forward = hook_mlp

            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Compute metrics
                delta = compute_semantic_shift(model, tokenizer, text, concept, device)

                results.append({
                    "strength": strength,
                    "prompt": prompt,
                    "text": text,
                    "delta": delta
                })

            finally:
                if placement == "after_layer":
                    handle.remove()
                else:
                    target_layer.mlp.forward = original_mlp_forward

    return results


def analyze_results(results, placement_name):
    """Compute aggregate metrics from results."""

    # Group by strength
    by_strength = {}
    for r in results:
        s = r["strength"]
        if s not in by_strength:
            by_strength[s] = []
        by_strength[s].append(r)

    # Compute metrics per strength
    metrics = {}
    for strength, samples in by_strength.items():
        texts = [s["text"] for s in samples]
        deltas = [s["delta"] for s in samples]

        # Diversity
        unique_texts = len(set(texts))
        diversity = unique_texts / len(texts)

        # Delta stats
        delta_mean = np.mean(deltas)
        delta_std = np.std(deltas)

        metrics[str(strength)] = {
            "n_samples": len(samples),
            "diversity": float(diversity),
            "delta_mean": float(delta_mean),
            "delta_std": float(delta_std),
            "unique_outputs": unique_texts
        }

    # Compare positive vs negative
    pos_samples = [r for r in results if r["strength"] > 0]
    neg_samples = [r for r in results if r["strength"] < 0]

    if pos_samples and neg_samples:
        # Embedding similarity between positive and negative outputs
        pos_texts = [s["text"] for s in pos_samples[:5]]  # Sample 5
        neg_texts = [s["text"] for s in neg_samples[:5]]

        # Compute average pairwise similarity
        similarities = []
        for pt in pos_texts:
            for nt in neg_texts:
                # We'll compute this externally
                similarities.append(0.0)  # Placeholder

        # Spearman correlation
        strengths = [r["strength"] for r in results]
        deltas = [r["delta"] for r in results]
        rho, pval = spearmanr(strengths, deltas)

        metrics["aggregate"] = {
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "mean_diversity": float(np.mean([m["diversity"] for m in metrics.values() if isinstance(m, dict) and "diversity" in m]))
        }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--concept", default="person")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/hook_placement_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("="*60)
    print("HOOK PLACEMENT TEST")
    print("="*60)
    print(f"Concept: {args.concept}")
    print(f"Strengths: [-1.0, -0.5, 0.0, 0.5, 1.0]")
    print(f"Prompts per strength: 3")
    print("="*60 + "\n")

    print_gpu_memory()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=args.device
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded")
    print_gpu_memory()

    try:
        # Extract concept vector
        print(f"\nExtracting concept vector for '{args.concept}'...")
        concept_vector = extract_concept_vector(model, tokenizer, args.concept, device=args.device)
        print(f"✓ Vector shape: {concept_vector.shape}")

        # Test parameters
        strengths = [-1.0, -0.5, 0.0, 0.5, 1.0]
        prompts = ["Tell me about", "Describe", "What is"]

        # Test current placement
        print("\n" + "="*60)
        print("TESTING: After Layer Placement")
        print("="*60)
        after_results = test_hook_placement(
            model, tokenizer, concept_vector,
            "after_layer", args.concept, strengths, prompts, args.device
        )
        after_metrics = analyze_results(after_results, "after_layer")
        print(f"✓ Completed {len(after_results)} generations")

        # Test proposed placement
        print("\n" + "="*60)
        print("TESTING: Before MLP Placement")
        print("="*60)
        before_results = test_hook_placement(
            model, tokenizer, concept_vector,
            "before_mlp", args.concept, strengths, prompts, args.device
        )
        before_metrics = analyze_results(before_results, "before_mlp")
        print(f"✓ Completed {len(before_results)} generations")

        # Compute cross-placement similarity
        print("\n" + "="*60)
        print("COMPUTING SIGN SYMMETRY")
        print("="*60)

        after_pos = [r for r in after_results if r["strength"] == 1.0]
        after_neg = [r for r in after_results if r["strength"] == -1.0]
        before_pos = [r for r in before_results if r["strength"] == 1.0]
        before_neg = [r for r in before_results if r["strength"] == -1.0]

        after_similarities = []
        for pos in after_pos:
            for neg in after_neg:
                sim = compute_embedding_similarity(model, tokenizer, pos["text"], neg["text"], args.device)
                after_similarities.append(sim)

        before_similarities = []
        for pos in before_pos:
            for neg in before_neg:
                sim = compute_embedding_similarity(model, tokenizer, pos["text"], neg["text"], args.device)
                before_similarities.append(sim)

        after_symmetry = float(np.mean(after_similarities))
        before_symmetry = float(np.mean(before_similarities))

        print(f"After layer: mean cos(+1, -1) = {after_symmetry:.4f}")
        print(f"Before MLP: mean cos(+1, -1) = {before_symmetry:.4f}")
        print(f"Improvement: {after_symmetry - before_symmetry:.4f}")

        # Save results
        output_data = {
            "metadata": {
                "test": "Hook Placement Comparison",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model,
                "concept": args.concept,
                "runtime_seconds": time.time() - start_time
            },
            "config": {
                "strengths": strengths,
                "prompts": prompts,
                "n_samples": len(after_results)
            },
            "results": {
                "after_layer": {
                    "samples": after_results,
                    "metrics": after_metrics,
                    "sign_symmetry": after_symmetry
                },
                "before_mlp": {
                    "samples": before_results,
                    "metrics": before_metrics,
                    "sign_symmetry": before_symmetry
                }
            },
            "conclusion": {
                "after_layer_symmetry": after_symmetry,
                "before_mlp_symmetry": before_symmetry,
                "symmetry_reduction": after_symmetry - before_symmetry,
                "better_placement": "before_mlp" if before_symmetry < after_symmetry else "after_layer"
            }
        }

        output_file = output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Saved to: {output_file}")
        print("="*60)

    finally:
        print("\nCleaning up GPU memory...")
        cleanup_model(model, tokenizer)
        print_gpu_memory()


if __name__ == "__main__":
    main()
