#!/usr/bin/env python3
"""
Phase 6.7: Steering Ablation Study with Full Metrics

Design Matrix:
Variant              | PCA removal | Manifold proj | Dampening | Expected
---------------------|-------------|---------------|-----------|------------------
① Raw baseline       | ✗           | ✗             | ✗         | Noisy but steered
② Contamination-only | ✓           | ✗             | ✗         | Clean but weak
③ Manifold-only      | ✗           | ✓             | varies    | Clean & responsive
④ Dual-subspace      | ✓           | ✓             | varies    | TBD

For variant ③ and ④, test 4 dampening multipliers: [0.0, 0.5, 1.0, 2.0]

Metrics per concept:
- Δ mean/std (semantic shift)
- Output diversity ratio
- Coherence % (ppl < 1.5× baseline)
- Spearman ρ (Δ vs strength)
- Δ range width (max - min)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hat import extract_concept_vector, generate_with_steering

# Simple test config for speed
N_CONCEPTS = 8  # Reduced for faster iteration
CONCEPTS = ["person", "animal", "object", "action", "time", "place", "quality", "emotion"]
STRENGTHS = [-2.0, -1.0, 0.0, 1.0, 2.0]  # 5 key strengths
PROMPTS = ["Tell me about", "Describe"]  # 2 prompts
DAMPENING_MULTS = [0.0, 0.5, 1.0, 2.0]  # Skip dampening, half, normal, double


def compute_delta(model, tokenizer, text: str, concept: str, device: str) -> float:
    """Compute Δ = cos(text, concept) - cos(text, 'nothing')."""
    def get_emb(phrase: str):
        inputs = tokenizer(phrase, return_tensors="pt").to(device)
        with torch.inference_mode():
            if hasattr(model.model, 'language_model'):
                embeds = model.model.language_model.embed_tokens(inputs.input_ids)
            elif hasattr(model.model, 'embed_tokens'):
                embeds = model.model.embed_tokens(inputs.input_ids)
            else:
                embeds = model.model.get_input_embeddings()(inputs.input_ids)
            return embeds.mean(dim=1).cpu().numpy()[0]

    text_v = get_emb(text)
    conc_v = get_emb(concept)
    neg_v = get_emb("nothing")

    text_v = text_v / (np.linalg.norm(text_v) + 1e-8)
    conc_v = conc_v / (np.linalg.norm(conc_v) + 1e-8)
    neg_v = neg_v / (np.linalg.norm(neg_v) + 1e-8)

    return float(np.dot(text_v, conc_v) - np.dot(text_v, neg_v))


def compute_ppl(model, tokenizer, text: str, device: str) -> float:
    """Compute perplexity."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, labels=inputs.input_ids)
        return torch.exp(outputs.loss).item()


def test_raw_baseline(model, tokenizer, device):
    """Variant ①: Raw baseline (no processing)."""
    print(f"\n{'='*60}\nVARIANT ①: RAW BASELINE\n{'='*60}")

    results = {}
    for concept in CONCEPTS:
        print(f"  {concept}...", end=" ")
        v = extract_concept_vector(model, tokenizer, concept, device=device)

        outputs = []
        for strength in STRENGTHS:
            for prompt in PROMPTS:
                text = generate_with_steering(
                    model, tokenizer, prompt, v, strength,
                    max_new_tokens=30, device=device
                )
                delta = compute_delta(model, tokenizer, text, concept, device)
                ppl = compute_ppl(model, tokenizer, text, device)

                outputs.append({
                    "strength": strength,
                    "text": text[:80],
                    "delta": delta,
                    "ppl": ppl
                })

        results[concept] = outputs
        print(f"✓ {len(outputs)}")

    return results


def analyze_results(results, baseline_ppl=10.0):
    """Compute metrics for each concept."""
    analysis = {}

    for concept, outputs in results.items():
        deltas = [o["delta"] for o in outputs]
        strengths = [o["strength"] for o in outputs]
        ppls = [o["ppl"] for o in outputs]
        texts = [o["text"] for o in outputs]

        # Diversity
        unique_texts = len(set(texts))
        diversity_ratio = unique_texts / len(texts)

        # Coherence
        coherent = [p < 1.5 * baseline_ppl for p in ppls]
        coherence_rate = sum(coherent) / len(coherent)

        # Δ stats
        delta_mean = np.mean(deltas)
        delta_std = np.std(deltas)
        delta_range = max(deltas) - min(deltas)

        # Spearman correlation
        rho, pval = spearmanr(strengths, deltas)

        analysis[concept] = {
            "delta_mean": float(delta_mean),
            "delta_std": float(delta_std),
            "delta_range": float(delta_range),
            "diversity_ratio": float(diversity_ratio),
            "coherence_rate": float(coherence_rate),
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "working": bool(diversity_ratio > 0.3 and abs(rho) > 0.2)  # Convert numpy.bool_ to bool
        }

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/phase_6_7_ablation_v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PHASE 6.7: STEERING ABLATION (FAST)")
    print(f"{'='*60}")
    print(f"Concepts: {N_CONCEPTS}")
    print(f"Strengths: {STRENGTHS}")
    print(f"Dampening multipliers: {DAMPENING_MULTS}")
    print(f"{'='*60}\n")

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
    print("✓ Model loaded\n")

    # Compute baseline perplexity
    baseline_text = generate_with_steering(
        model, tokenizer, "Hello", None, 0.0,
        max_new_tokens=30, device=args.device
    )
    baseline_ppl = compute_ppl(model, tokenizer, baseline_text, args.device)
    print(f"Baseline perplexity: {baseline_ppl:.2f}\n")

    # Test variant ① (raw baseline)
    raw_results = test_raw_baseline(model, tokenizer, args.device)
    raw_analysis = analyze_results(raw_results, baseline_ppl)

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")

    raw_working = sum(1 for a in raw_analysis.values() if a["working"])
    raw_diversity = np.mean([a["diversity_ratio"] for a in raw_analysis.values()])
    raw_rho = np.mean([abs(a["spearman_rho"]) for a in raw_analysis.values()])

    print(f"\nRaw Baseline:")
    print(f"  Working: {raw_working}/{N_CONCEPTS}")
    print(f"  Mean diversity: {raw_diversity:.2%}")
    print(f"  Mean |ρ|: {raw_rho:.3f}")

    # Save
    output_data = {
        "config": {
            "model": args.model,
            "concepts": CONCEPTS,
            "strengths": STRENGTHS,
            "baseline_ppl": baseline_ppl
        },
        "raw_baseline": {
            "results": raw_results,
            "analysis": raw_analysis,
            "summary": {
                "working": raw_working,
                "mean_diversity": float(raw_diversity),
                "mean_rho": float(raw_rho)
            }
        }
    }

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved to: {output_dir}/ablation_results.json")
    print("="*60)


if __name__ == "__main__":
    main()
