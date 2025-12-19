#!/usr/bin/env python3
"""
Phase 6.7: Full Steering Ablation Study

Design Matrix:
Variant              | PCA removal | Manifold proj | Dampening | Expected
---------------------|-------------|---------------|-----------|------------------
① Raw baseline       | ✗           | ✗             | ✗         | Noisy but steered
② Contamination-only | ✓           | ✗             | ✗         | Clean but weak
③ Manifold-only      | ✗           | ✓             | varies    | Clean & responsive (paper)
④ Dual-subspace      | ✓           | ✓             | varies    | Over-damped?

Test parameters:
- 32 concepts
- 7 strengths: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
- 4 dampening multipliers for ③④: [0.0, 0.5, 1.0, 2.0]

Metrics:
- Δ mean/std per concept
- Output diversity ratio
- Coherence % (ppl < 1.5× baseline)
- Spearman ρ (Δ vs strength)
- Δ range width (dynamic range)
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

from src.steering import extract_concept_vector, generate_with_steering
from src.steering.manifold import estimate_contamination_subspace, estimate_task_manifold, apply_dual_subspace_steering

# Test configuration
CONCEPTS = [
    "person", "change", "animal", "object", "action", "time", "place", "quality",
    "relation", "number", "thought", "emotion", "truth", "knowledge", "language",
    "society", "culture", "nature", "life", "death", "power", "freedom", "justice",
    "beauty", "art", "science", "technology", "religion", "morality", "identity",
    "consciousness", "reality"
]
STRENGTHS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
PROMPTS = ["Tell me about", "Describe"]
DAMPENING_MULTS = [0.0, 0.5, 1.0, 2.0]


def compute_delta(model, tokenizer, text: str, concept: str, device: str) -> float:
    """Compute Δ = cos(text, concept) - cos(text, 'nothing')."""
    def get_emb(phrase: str):
        inputs = tokenizer(phrase, return_tensors="pt").to(device)
        with torch.inference_mode():
            if hasattr(model.model, 'language_model'):
                embeds = model.model.language_model.embed_tokens(inputs.input_ids)
            else:
                embeds = model.model.embed_tokens(inputs.input_ids)
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


def generate_with_processing(
    model, tokenizer, prompt: str, concept_vector: np.ndarray,
    strength: float, variant: str, dampening_mult: float,
    U_S: np.ndarray, U_M: np.ndarray, device: str
) -> str:
    """Generate with specific processing variant."""
    # Get layers
    if hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers

    total_layers = len(layers)
    target_layer = layers[-1]

    # Process vector based on variant
    if variant == "raw":
        v_processed = concept_vector

    elif variant == "contamination":
        contamination_proj = U_S @ (U_S.T @ concept_vector)
        v_processed = concept_vector - contamination_proj

    elif variant == "manifold":
        # Manifold projection only
        v_mw = U_M @ (U_M.T @ concept_vector)

        # Apply dampening (scaled by multiplier)
        layer_depth = (total_layers - 1) / total_layers
        depth_gain = np.sqrt(1.0 - layer_depth)
        depth_gain = depth_gain ** dampening_mult  # 0=no damp, 1=normal, 2=more damp
        v_mw = v_mw * depth_gain

        # Norm clipping (scaled by multiplier)
        max_norm = 1.0 * (1.0 + dampening_mult)
        norm = np.linalg.norm(v_mw)
        if norm > max_norm:
            v_mw = v_mw * (max_norm / norm)

        v_processed = v_mw

    elif variant == "dual":
        # Full dual-subspace
        v_processed = apply_dual_subspace_steering(
            concept_vector, U_S, U_M,
            layer_idx=total_layers - 1,
            total_layers=total_layers,
            max_norm_per_layer=1.0 * (1.0 + dampening_mult),
            ema_alpha=0.0,
            prev_vector=None
        )

    # Create hook
    v_tensor = torch.from_numpy(v_processed).float().to(device)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        v_matched = v_tensor.to(dtype=hidden.dtype)
        projection = (hidden @ v_matched.unsqueeze(-1)) * v_matched
        steered = hidden + strength * projection  # positive = amplify
        return (steered,) if isinstance(output, tuple) else steered

    # Generate
    handle = target_layer.register_forward_hook(hook_fn)
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
    finally:
        handle.remove()

    return text


def test_variant(model, tokenizer, variant_name: str, dampening_mult: float,
                concepts: list, U_S, U_M_dict, baseline_ppl: float, device: str):
    """Test one variant configuration."""
    print(f"\n  Testing {variant_name} (damp={dampening_mult})...")

    results = {}
    for concept in concepts:
        v = extract_concept_vector(model, tokenizer, concept, device=device)
        U_M = U_M_dict.get(concept) if U_M_dict else None

        outputs = []
        for strength in STRENGTHS:
            for prompt in PROMPTS:
                try:
                    if variant_name in ["raw", "contamination"]:
                        # No dampening for these
                        text = generate_with_processing(
                            model, tokenizer, prompt, v, strength,
                            variant_name, 0.0, U_S, U_M, device
                        )
                    else:
                        text = generate_with_processing(
                            model, tokenizer, prompt, v, strength,
                            variant_name, dampening_mult, U_S, U_M, device
                        )

                    delta = compute_delta(model, tokenizer, text, concept, device)
                    ppl = compute_ppl(model, tokenizer, text, device)

                    outputs.append({
                        "strength": strength,
                        "text": text[:80],
                        "delta": delta,
                        "ppl": ppl
                    })
                except Exception as e:
                    print(f"    Error: {e}")
                    outputs.append({
                        "strength": strength,
                        "text": "",
                        "delta": 0.0,
                        "ppl": 999.0
                    })

        results[concept] = outputs

    # Analyze
    analysis = {}
    for concept, outputs in results.items():
        deltas = [o["delta"] for o in outputs]
        strengths = [o["strength"] for o in outputs]
        ppls = [o["ppl"] for o in outputs]
        texts = [o["text"] for o in outputs]

        unique_texts = len(set(texts))
        diversity_ratio = unique_texts / len(texts)
        coherent = [p < 1.5 * baseline_ppl for p in ppls]
        coherence_rate = sum(coherent) / len(coherent)
        delta_mean = np.mean(deltas)
        delta_std = np.std(deltas)
        delta_range = max(deltas) - min(deltas)
        rho, pval = spearmanr(strengths, deltas)

        analysis[concept] = {
            "delta_mean": float(delta_mean),
            "delta_std": float(delta_std),
            "delta_range": float(delta_range),
            "diversity_ratio": float(diversity_ratio),
            "coherence_rate": float(coherence_rate),
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "working": bool(diversity_ratio > 0.3 and abs(rho) > 0.2)
        }

    working = sum(1 for a in analysis.values() if a["working"])
    mean_diversity = np.mean([a["diversity_ratio"] for a in analysis.values()])
    mean_rho = np.mean([abs(a["spearman_rho"]) for a in analysis.values()])

    print(f"    → Working: {working}/{len(concepts)}, Diversity: {mean_diversity:.1%}, |ρ|: {mean_rho:.3f}")

    return results, analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/phase_6_7_full_ablation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PHASE 6.7: FULL ABLATION STUDY")
    print(f"{'='*60}")
    print(f"Concepts: {len(CONCEPTS)}")
    print(f"Strengths: {STRENGTHS}")
    print(f"Dampening mults: {DAMPENING_MULTS}")
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

    # Baseline perplexity
    baseline_text = generate_with_steering(
        model, tokenizer, "Hello", None, 0.0,
        max_new_tokens=30, device=args.device
    )
    baseline_ppl = compute_ppl(model, tokenizer, baseline_text, args.device)
    print(f"Baseline perplexity: {baseline_ppl:.2f}\n")

    # Estimate subspaces
    print("Estimating contamination subspace...")
    concept_vectors = []
    for concept in CONCEPTS:
        v = extract_concept_vector(model, tokenizer, concept, device=args.device)
        concept_vectors.append(v)
    concept_matrix = np.array(concept_vectors)
    U_S, _ = estimate_contamination_subspace(concept_matrix, n_components=5)
    print(f"✓ U_S shape: {U_S.shape}\n")

    print("Estimating task manifolds...")
    U_M_dict = {}
    for i, concept in enumerate(CONCEPTS):
        print(f"  [{i+1}/{len(CONCEPTS)}] {concept}...", end=" ")
        v = concept_vectors[i]
        try:
            U_M, _ = estimate_task_manifold(
                model, tokenizer, concept, v,
                n_samples=4, device=args.device
            )
            U_M_dict[concept] = U_M
            print(f"✓")
        except Exception as e:
            print(f"✗ ({e})")
            U_M_dict[concept] = np.eye(len(v))
    print()

    # Test all variants
    all_results = {}

    # ① Raw baseline
    print(f"\n{'='*60}")
    print("VARIANT ①: RAW BASELINE")
    print(f"{'='*60}")
    raw_results, raw_analysis = test_variant(
        model, tokenizer, "raw", 0.0,
        CONCEPTS, U_S, U_M_dict, baseline_ppl, args.device
    )
    all_results["raw"] = {"results": raw_results, "analysis": raw_analysis}

    # ② Contamination-only
    print(f"\n{'='*60}")
    print("VARIANT ②: CONTAMINATION-ONLY")
    print(f"{'='*60}")
    cont_results, cont_analysis = test_variant(
        model, tokenizer, "contamination", 0.0,
        CONCEPTS, U_S, U_M_dict, baseline_ppl, args.device
    )
    all_results["contamination"] = {"results": cont_results, "analysis": cont_analysis}

    # ③ Manifold-only (test all dampening)
    print(f"\n{'='*60}")
    print("VARIANT ③: MANIFOLD-ONLY")
    print(f"{'='*60}")
    for damp in DAMPENING_MULTS:
        key = f"manifold_damp{damp}"
        results, analysis = test_variant(
            model, tokenizer, "manifold", damp,
            CONCEPTS, U_S, U_M_dict, baseline_ppl, args.device
        )
        all_results[key] = {"results": results, "analysis": analysis}

    # ④ Dual-subspace (test all dampening)
    print(f"\n{'='*60}")
    print("VARIANT ④: DUAL-SUBSPACE")
    print(f"{'='*60}")
    for damp in DAMPENING_MULTS:
        key = f"dual_damp{damp}"
        results, analysis = test_variant(
            model, tokenizer, "dual", damp,
            CONCEPTS, U_S, U_M_dict, baseline_ppl, args.device
        )
        all_results[key] = {"results": results, "analysis": analysis}

    # Save
    output_data = {
        "config": {
            "model": args.model,
            "concepts": CONCEPTS,
            "strengths": STRENGTHS,
            "dampening_mults": DAMPENING_MULTS,
            "baseline_ppl": baseline_ppl
        },
        "variants": all_results
    }

    with open(output_dir / "full_ablation_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved to: {output_dir}/full_ablation_results.json")
    print("="*60)


if __name__ == "__main__":
    main()
