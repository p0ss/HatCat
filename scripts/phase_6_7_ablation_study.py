#!/usr/bin/env python3
"""
Phase 6.7: Steering Ablation Study

Tests 4 steering configurations to identify which components help vs hurt:

Variant              | PCA removal | Manifold proj | Dampening | Expected
---------------------|-------------|---------------|-----------|------------------
① Raw baseline       | ✗           | ✗             | ✗         | Noisy but steered
② Contamination-only | ✓           | ✗             | ✗         | Clean but weak
③ Manifold-only      | ✗           | ✓             | ✓         | Clean & responsive (paper)
④ Dual-subspace      | ✓           | ✓             | ✓         | Over-damped / broken

For each variant, test:
- 32 concepts
- 7 strength values: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
- Measure output diversity to detect if steering is working
"""

import argparse
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr

from src.steering.extraction import extract_concept_vector
from src.steering.hooks import create_steering_hook
from src.steering.manifold import estimate_contamination_subspace, estimate_task_manifold, apply_dual_subspace_steering


def compute_semantic_shift_simple(
    model,
    tokenizer,
    text: str,
    concept: str,
    neg_concept: str,
    device: str
) -> float:
    """Compute Δ = cos(text, concept) - cos(text, neg_concept)."""
    def get_embedding(phrase: str) -> np.ndarray:
        inputs = tokenizer(phrase, return_tensors="pt").to(device)
        with torch.inference_mode():
            if hasattr(model.model, 'embed_tokens'):
                embeds = model.model.embed_tokens(inputs.input_ids)
            elif hasattr(model.model, 'language_model'):
                embeds = model.model.language_model.embed_tokens(inputs.input_ids)
            else:
                raise AttributeError(f"Cannot find embed_tokens")
            return embeds.mean(dim=1).cpu().numpy()[0]

    text_emb = get_embedding(text)
    concept_emb = get_embedding(concept)
    neg_emb = get_embedding(neg_concept)

    # Normalize
    text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
    concept_emb = concept_emb / (np.linalg.norm(concept_emb) + 1e-8)
    neg_emb = neg_emb / (np.linalg.norm(neg_emb) + 1e-8)

    # Δ = cos(text, concept) - cos(text, neg)
    delta = float(np.dot(text_emb, concept_emb) - np.dot(text_emb, neg_emb))
    return delta


def compute_perplexity(model, tokenizer, text: str, device: str) -> float:
    """Compute perplexity for coherence check."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, labels=inputs.input_ids)
        return torch.exp(outputs.loss).item()


def generate_with_variant(
    model,
    tokenizer,
    prompt: str,
    concept_vector: np.ndarray,
    strength: float,
    variant: str,
    dampening_mult: float = 1.0,
    U_S: Optional[np.ndarray] = None,
    U_M: Optional[np.ndarray] = None,
    device: str = "cuda",
    max_new_tokens: int = 30
) -> str:
    """
    Generate with specific steering variant.

    Variants:
    - "raw": No processing, just strength * v
    - "contamination": PCA removal only
    - "manifold": Manifold projection + dampening only
    - "dual": Full dual-subspace (contamination + manifold + dampening)
    """
    # Get model layers
    if hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    elif hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise AttributeError(f"Cannot find layers in model: {type(model.model)}")

    total_layers = len(layers)
    target_layer = layers[-1]  # Steer final layer

    # Process vector based on variant
    if variant == "raw":
        # No processing
        v_processed = concept_vector

    elif variant == "contamination":
        # PCA removal only
        if U_S is None:
            raise ValueError("U_S required for contamination variant")
        contamination_proj = U_S @ (U_S.T @ concept_vector)
        v_processed = concept_vector - contamination_proj

    elif variant == "manifold":
        # Manifold projection + dampening (paper's approach)
        if U_M is None:
            raise ValueError("U_M required for manifold variant")

        # Project onto manifold
        v_mw = U_M @ (U_M.T @ concept_vector)

        # Apply layer-wise dampening (final layer)
        layer_depth = (total_layers - 1) / total_layers
        depth_gain = np.sqrt(1.0 - layer_depth)
        v_mw = v_mw * depth_gain

        # Norm clipping
        norm = np.linalg.norm(v_mw)
        if norm > 1.0:
            v_mw = v_mw / norm

        v_processed = v_mw

    elif variant == "dual":
        # Full dual-subspace
        if U_S is None or U_M is None:
            raise ValueError("U_S and U_M required for dual variant")

        v_processed = apply_dual_subspace_steering(
            concept_vector, U_S, U_M,
            layer_idx=total_layers - 1,
            total_layers=total_layers,
            max_norm_per_layer=1.0,
            ema_alpha=0.0,
            prev_vector=None
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Create hook
    v_tensor = torch.from_numpy(v_processed).float().to(device)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        v_matched = v_tensor.to(dtype=hidden.dtype)
        projection = (hidden @ v_matched.unsqueeze(-1)) * v_matched
        steered = hidden - strength * projection
        return (steered,) if isinstance(output, tuple) else steered

    # Generate with hook
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        handle.remove()

    return text


def test_variant(
    model,
    tokenizer,
    concepts: List[str],
    variant_name: str,
    prompts: List[str],
    strengths: List[float],
    U_S: Optional[np.ndarray],
    U_M_dict: Optional[Dict[str, np.ndarray]],
    device: str
) -> Dict:
    """Test one steering variant across all concepts and strengths."""
    print(f"\n{'='*60}")
    print(f"TESTING VARIANT: {variant_name.upper()}")
    print(f"{'='*60}")

    results = {}

    for concept in concepts:
        print(f"\n  Concept: {concept}")

        # Extract concept vector
        v = extract_concept_vector(model, tokenizer, concept, device=device)
        U_M = U_M_dict.get(concept) if U_M_dict else None

        concept_results = []
        for strength in strengths:
            for prompt in prompts[:2]:  # 2 prompts per strength for speed
                try:
                    text = generate_with_variant(
                        model, tokenizer, prompt, v, strength,
                        variant=variant_name,
                        U_S=U_S, U_M=U_M, device=device
                    )

                    concept_results.append({
                        "strength": strength,
                        "prompt": prompt,
                        "text": text[:100]  # Store first 100 chars
                    })
                except Exception as e:
                    print(f"    Error at strength={strength}: {e}")

        results[concept] = concept_results
        print(f"    ✓ {len(concept_results)} tests")

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
            "working": diversity_ratio > 0.3  # >30% unique = steering works
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Phase 6.7: Steering Ablation Study")
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/phase_6_7_ablation")
    parser.add_argument("--n-concepts", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PHASE 6.7: STEERING ABLATION STUDY")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Concepts: {args.n_concepts}")
    print(f"Output: {output_dir}")
    print(f"\nVariants:")
    print(f"  ① Raw baseline       (no processing)")
    print(f"  ② Contamination-only (PCA removal)")
    print(f"  ③ Manifold-only      (paper's method)")
    print(f"  ④ Dual-subspace      (all components)")
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

    # Concepts
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
    prompts = ["Tell me about", "Describe", "What is", "Consider", "Explain"]
    strengths = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    # Estimate contamination subspace (U_S) for variants 2 & 4
    print("\n" + "="*60)
    print("Estimating contamination subspace (U_S)...")
    print("="*60)
    concept_vectors = []
    for concept in concepts:
        v = extract_concept_vector(model, tokenizer, concept, device=args.device)
        concept_vectors.append(v)
    concept_matrix = np.array(concept_vectors)
    U_S, _ = estimate_contamination_subspace(concept_matrix, n_components=5)
    print(f"✓ U_S shape: {U_S.shape}\n")

    # Estimate task manifolds (U_M) for variants 3 & 4
    print("="*60)
    print("Estimating task manifolds (U_M)...")
    print("="*60)
    U_M_dict = {}
    for concept in concepts:
        print(f"  {concept}...", end=" ")
        v = extract_concept_vector(model, tokenizer, concept, device=args.device)

        # For manifold variant: use raw vector
        # For dual variant: use cleaned vector
        try:
            U_M, _ = estimate_task_manifold(
                model, tokenizer, concept, v,
                n_samples=4, device=args.device
            )
            U_M_dict[concept] = U_M
            print(f"✓ ({U_M.shape[1]} dims)")
        except Exception as e:
            print(f"✗ ({e})")
            U_M_dict[concept] = np.eye(len(v))  # Fallback

    print()

    # Run all 4 variants
    all_results = {}
    all_analysis = {}

    variants = [
        ("raw", None, None),
        ("contamination", U_S, None),
        ("manifold", None, U_M_dict),
        ("dual", U_S, U_M_dict)
    ]

    for variant_name, use_U_S, use_U_M in variants:
        results = test_variant(
            model, tokenizer, concepts, variant_name,
            prompts, strengths,
            use_U_S, use_U_M, args.device
        )
        analysis = analyze_diversity(results)

        all_results[variant_name] = results
        all_analysis[variant_name] = analysis

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for variant_name in ["raw", "contamination", "manifold", "dual"]:
        analysis = all_analysis[variant_name]
        working = sum(1 for a in analysis.values() if a["working"])
        mean_div = np.mean([a['diversity_ratio'] for a in analysis.values()])

        print(f"\n{variant_name.upper()}:")
        print(f"  Working concepts: {working}/{len(concepts)}")
        print(f"  Mean diversity: {mean_div:.2%}")

    # Save results
    output_data = {
        "config": {
            "model": args.model,
            "n_concepts": args.n_concepts,
            "concepts": concepts,
            "strengths": strengths,
            "U_S_shape": U_S.shape if U_S is not None else None
        },
        "variants": {
            variant: {
                "results": all_results[variant],
                "analysis": all_analysis[variant],
                "working_count": sum(1 for a in all_analysis[variant].values() if a["working"]),
                "mean_diversity": float(np.mean([a['diversity_ratio'] for a in all_analysis[variant].values()]))
            }
            for variant in ["raw", "contamination", "manifold", "dual"]
        }
    }

    output_file = output_dir / "ablation_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
