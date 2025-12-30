"""
Validate: Do activation signatures converge within 10 samples?

Tests the hypothesis that E[f_θ(x)] stabilizes quickly across samples.
Uses proper attention-masked pooling and hold-out validation.
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_activation(model, tokenizer, text, layer_idx=-1, device="cuda"):
    """Extract activation vector with proper attention masking."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    model.config.output_hidden_states = True
    
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
    
    hs = out.hidden_states[layer_idx]                     # [B,T,D]
    mask = inputs["attention_mask"].unsqueeze(-1)         # [B,T,1]
    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.squeeze(0).float().cpu().numpy()

def generate_contexts(concept, n_samples):
    """Generate diverse contexts for a concept."""
    templates = [
        f"{concept}",
        f"The concept of {concept}",
        f"An example of {concept} is",
        f"{concept} refers to",
        f"When discussing {concept}",
        f"The meaning of {concept}",
        f"In the context of {concept}",
        f"Understanding {concept}",
        f"{concept} can be described as",
        f"The nature of {concept}",
        f"Regarding {concept}",
        f"With respect to {concept}",
        f"{concept} involves",
        f"The essence of {concept}",
        f"Considering {concept}",
        f"{concept} encompasses",
        f"The significance of {concept}",
        f"Exploring {concept}",
        f"{concept} manifests as",
        f"The role of {concept}",
    ]
    return templates[:n_samples]

def measure_convergence(model, tokenizer, concept, max_samples=20, layer_idx=-1, device="cuda"):
    """
    Measure how activation signature converges with more samples.
    Uses second half as hold-out reference as n grows.
    """
    ctxs = generate_contexts(concept, max_samples)
    np.random.shuffle(ctxs)

    print(f"\n  Collecting activations for '{concept}'...")
    print(f"  Contexts (first 3):")
    for i, ctx in enumerate(ctxs[:3], 1):
        print(f"    {i}. {ctx}")

    # Collect all activations
    acts = np.stack([
        get_activation(model, tokenizer, c, layer_idx, device)
        for c in ctxs
    ])

    # Log activation statistics
    print(f"\n  Activation statistics:")
    print(f"    Shape: {acts.shape}")
    print(f"    Mean: {acts.mean():.6f}")
    print(f"    Std: {acts.std():.6f}")
    print(f"    Min: {acts.min():.6f}, Max: {acts.max():.6f}")
    print(f"    Non-zero: {(acts != 0).sum()}/{acts.size} ({100*(acts != 0).sum()/acts.size:.1f}%)")

    # Use second half as hold-out reference
    split = max_samples // 2
    A_ref = acts[split:].mean(axis=0)

    print(f"\n  Reference (from samples {split+1}-{max_samples}):")
    print(f"    ||A_ref||: {np.linalg.norm(A_ref):.6f}")

    rows = []
    for n in range(1, split + 1):
        A_n = acts[:n].mean(axis=0)
        sigma = acts[:n].std(axis=0).mean()

        # Compute differences
        diff = A_n - A_ref
        rel = np.linalg.norm(diff) / np.linalg.norm(A_ref)
        cos = 1 - cosine(A_n, A_ref)

        rows.append({
            "n_samples": n,
            "rel_difference": rel,
            "cosine_similarity": cos,
            "sigma": sigma,
            "ci95": 1.96 * sigma / np.sqrt(n),
            "A_n_norm": np.linalg.norm(A_n),
            "diff_norm": np.linalg.norm(diff)
        })

        # Log key checkpoints
        if n in [1, 5, 10]:
            print(f"\n  At N={n}:")
            print(f"    ||A_{n}||: {np.linalg.norm(A_n):.6f}")
            print(f"    ||A_{n} - A_ref||: {np.linalg.norm(diff):.6f}")
            print(f"    Relative diff: {rel:.4f} ({'✓' if rel < 0.05 else '✗'})")
            print(f"    Cosine sim: {cos:.6f}")

    return rows

def test_convergence_hypothesis(concepts, model_name="google/gemma-3-270m", device="cuda"):
    """Test convergence across multiple concepts."""
    
    print(f"Loading {model_name}...")
    model = AutoModel.from_pretrained(
        model_name, 
        dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    model.to(device)
    model.config.output_hidden_states = True
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    
    for concept in tqdm(concepts, desc="Testing concepts"):
        convergence = measure_convergence(
            model, tokenizer, concept, 
            max_samples=20, 
            layer_idx=-1,
            device=device
        )
        results[concept] = convergence
        
        # Check 10-sample performance
        data_at_10 = convergence[9]  # 0-indexed
        print(f"\n{concept}:")
        print(f"  Relative difference: {data_at_10['rel_difference']:.4f}")
        print(f"  Cosine similarity: {data_at_10['cosine_similarity']:.6f}")
        print(f"  Within 5%: {'✓' if data_at_10['rel_difference'] < 0.05 else '✗'}")
    
    return results

def plot_convergence(results, output_path="data/processed/convergence_analysis.png"):
    """Visualize convergence across concepts."""

    from pathlib import Path

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for concept, data in results.items():
        n_samples = [d['n_samples'] for d in data]
        rel_diffs = [d['rel_difference'] for d in data]
        cos_sims = [d['cosine_similarity'] for d in data]
        sigmas = [d['sigma'] for d in data]
        ci_widths = [d['ci95'] for d in data]
        
        # Plot relative difference
        axes[0, 0].plot(n_samples, rel_diffs, marker='o', label=concept, linewidth=2)
        
        # Plot cosine similarity
        axes[0, 1].plot(n_samples, cos_sims, marker='o', label=concept, linewidth=2)
        
        # Plot variance
        axes[1, 0].plot(n_samples, sigmas, marker='o', label=concept, linewidth=2)
        
        # Plot confidence interval width
        axes[1, 1].plot(n_samples, ci_widths, marker='o', label=concept, linewidth=2)
    
    # Reference lines
    axes[0, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    axes[0, 0].axvline(x=10, color='green', linestyle='--', alpha=0.5, label='N=10')
    axes[0, 1].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
    axes[0, 1].axvline(x=10, color='green', linestyle='--', alpha=0.5, label='N=10')
    axes[1, 0].axvline(x=10, color='green', linestyle='--', alpha=0.5, label='N=10')
    axes[1, 1].axvline(x=10, color='green', linestyle='--', alpha=0.5, label='N=10')
    
    # Formatting
    axes[0, 0].set_xlabel('Number of Samples', fontsize=11)
    axes[0, 0].set_ylabel('Relative Difference', fontsize=11)
    axes[0, 0].set_title('Convergence: ||A_n - A_ref|| / ||A_ref||', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Number of Samples', fontsize=11)
    axes[0, 1].set_ylabel('Cosine Similarity', fontsize=11)
    axes[0, 1].set_title('Cosine Similarity to Hold-out Reference', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Number of Samples', fontsize=11)
    axes[1, 0].set_ylabel('Standard Deviation', fontsize=11)
    axes[1, 0].set_title('Activation Variance (σ)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Number of Samples', fontsize=11)
    axes[1, 1].set_ylabel('CI Width (95%)', fontsize=11)
    axes[1, 1].set_title('Confidence Interval Width', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate convergence hypothesis")
    parser.add_argument("--model", type=str, default="google/gemma-3-270m",
                       help="Model to test")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--concepts", type=str, nargs="+",
                       default=["democracy", "dog", "running", "happiness", "gravity"],
                       help="Concepts to test")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CONVERGENCE VALIDATION EXPERIMENT")
    print("=" * 70)
    print("\nHypothesis: Activation signatures converge within 10 samples")
    print("Criterion: ||A_10 - A_ref|| / ||A_ref|| < 0.05")
    print(f"Testing {len(args.concepts)} concepts\n")
    
    results = test_convergence_hypothesis(
        concepts=args.concepts,
        model_name=args.model,
        device=args.device
    )
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    within_5pct = 0
    within_10pct = 0
    avg_rel_diff = 0
    avg_cos_sim = 0

    for concept, data in results.items():
        rel_diff = data[9]['rel_difference']  # N=10
        cos_sim = data[9]['cosine_similarity']
        avg_rel_diff += rel_diff
        avg_cos_sim += cos_sim

        if rel_diff < 0.05:
            within_5pct += 1
            status = "✓ PASS (5%)"
        elif rel_diff < 0.10:
            within_10pct += 1
            status = "⚠ MARGINAL (10%)"
        else:
            status = "✗ FAIL"
        print(f"{concept:20s}: rel_diff={rel_diff:.4f}, cos_sim={cos_sim:.6f} {status}")

    n_concepts = len(results)
    avg_rel_diff /= n_concepts
    avg_cos_sim /= n_concepts

    success_rate = 100 * within_5pct / n_concepts
    marginal_rate = 100 * within_10pct / n_concepts

    print(f"\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"Average relative difference at N=10: {avg_rel_diff:.4f}")
    print(f"Average cosine similarity at N=10:  {avg_cos_sim:.6f}")
    print(f"\nConcepts within 5% at N=10:  {within_5pct}/{n_concepts} ({success_rate:.1f}%)")
    print(f"Concepts within 10% at N=10: {within_10pct}/{n_concepts} ({marginal_rate:.1f}%)")

    print(f"\nWhat this means:")
    if avg_rel_diff < 0.05:
        print("  ✓ Excellent: Simple templates provide stable representations")
    elif avg_rel_diff < 0.10:
        print("  ⚠ Moderate: Templates help but more diversity needed")
    elif avg_rel_diff < 0.25:
        print("  ⚠ High variance: Stage 0 → Stage 1 → Stage 2 refinement essential")
    else:
        print("  ✗ Very high variance: Single samples unreliable, multi-stage critical")

    if avg_cos_sim > 0.99:
        print("  ✓ Strong alignment: Activation patterns are consistent")
    elif avg_cos_sim > 0.95:
        print("  ⚠ Good alignment but significant angular differences remain")
    else:
        print("  ✗ Poor alignment: Contexts produce different activation directions")

    if success_rate >= 60:
        print("\n✓ Hypothesis SUPPORTED: Most concepts converge within 10 samples")
    else:
        print("\n✗ Hypothesis REJECTED: Need more samples for convergence")
        print(f"  → Recommendation: Use Stage 1 (5-10 templates) + Stage 2 (20+ diverse)")
        print(f"  → Progressive refinement is justified by these results")
    
    plot_convergence(results)