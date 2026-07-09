#!/usr/bin/env python3
"""
Quantization sensitivity analysis for HatCat lens pack.

Tests how polar quantization (TurboQuant-style) and Johnson-Lindenstrauss
random projection affect the inter-lens geometry of an existing lens pack,
to inform whether deployment-time activation quantization is viable without
retraining.

Empirical question: at what bit depth does pairwise concept distance
preservation degrade enough that nearest-neighbor concept identity changes?

CONCEPT SIGNATURE CHOICE (overridable via --signature):
    - "mean_row": mean of net.0.weight rows (single 2560-d vector per concept).
      Captures dominant direction the concept attends to in activation space.
      Default; fast and interpretable.
    - "flat": flattened net.0.weight (128*2560 = 327680-d per concept).
      Captures full first-layer subspace structure. Slower, more memory.
    - "svd1": top-1 right singular vector of net.0.weight.
      Principal direction in activation space. Cleanest geometric interpretation.

Run on EC2 (CPU only — no GPU needed):
    python scripts/analysis/quant_sensitivity.py \\
        --pack-dir /home/ubuntu/HatCat/lens_packs/gemma-3-4b_first-light-v1-bf16
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_concept_signatures(pack_dir: Path, signature: str, max_per_layer: int | None):
    """Load each lens, extract a single concept-signature vector.

    Returns:
        names:   list[str]  length N
        layers:  np.array   length N (ontology layer)
        sigs:    np.array   shape (N, D) where D depends on signature choice
    """
    names, layers, sigs = [], [], []
    for layer_dir in sorted(pack_dir.glob("layer*")):
        if not layer_dir.is_dir():
            continue
        try:
            layer_num = int(layer_dir.name.replace("layer", ""))
        except ValueError:
            continue
        files = sorted(layer_dir.glob("*.pt"))
        if max_per_layer is not None:
            files = files[:max_per_layer]
        for pt_file in files:
            try:
                obj = torch.load(pt_file, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  skip {pt_file.name}: {e}")
                continue
            if not isinstance(obj, dict) or "net.0.weight" not in obj:
                continue
            W0 = obj["net.0.weight"].float().numpy()  # (128, 2560)
            if signature == "mean_row":
                sig = W0.mean(axis=0)
            elif signature == "flat":
                sig = W0.flatten()
            elif signature == "svd1":
                # First right singular vector — principal input direction.
                _, _, vt = np.linalg.svd(W0, full_matrices=False)
                sig = vt[0]
            else:
                raise ValueError(f"unknown signature: {signature}")
            names.append(pt_file.stem)
            layers.append(layer_num)
            sigs.append(sig.astype(np.float32))
    return names, np.array(layers), np.stack(sigs)


# ---------------------------------------------------------------------------
# Quantization schemes
# ---------------------------------------------------------------------------
def polar_quantize(W: np.ndarray, bits: int) -> np.ndarray:
    """Polar quantization: separate (norm, direction); quantize direction
    uniformly per-component then renormalize back onto the sphere.

    Norms kept at fp32 (scalars, cheap). The component-uniform direction
    quantization is a stand-in for TurboQuant's spherical-codebook approach;
    it shares the key property of preserving inner products with bounded
    relative error in the unit-sphere geometry.
    """
    N, D = W.shape
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-9, 1.0, norms)
    directions = W / norms_safe
    levels = float(2 ** (bits - 1) - 1)
    if levels < 1:
        levels = 1.0
    dq = np.clip(np.round(directions * levels) / levels, -1.0, 1.0)
    new_norms = np.linalg.norm(dq, axis=1, keepdims=True)
    new_norms = np.where(new_norms < 1e-9, 1.0, new_norms)
    dq = dq / new_norms
    return (dq * norms).astype(np.float32)


def jl_project(W: np.ndarray, d_target: int, seed: int = 42) -> np.ndarray:
    """Random Gaussian projection (Johnson-Lindenstrauss) to dim d_target.
    Scaled by 1/sqrt(d_target) so inner products are preserved in expectation.
    """
    rng = np.random.RandomState(seed)
    P = rng.randn(W.shape[1], d_target).astype(np.float32) / np.sqrt(d_target)
    return W @ P


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def cosine_sim_matrix(W: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    Wu = W / np.maximum(norms, 1e-9)
    sim = Wu @ Wu.T
    np.fill_diagonal(sim, -np.inf)  # exclude self
    return sim


def topk_indices(sim: np.ndarray, k: int) -> np.ndarray:
    # argpartition for speed; only need top k unsorted then sort within k
    idx_unsorted = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    sub = sim[rows, idx_unsorted]
    order = np.argsort(-sub, axis=1)
    return idx_unsorted[rows, order]


def topk_overlap(base_topk: np.ndarray, q_topk: np.ndarray) -> np.ndarray:
    """Per-row Jaccard-like overlap fraction: |base ∩ q| / k."""
    N, k = base_topk.shape
    base_sets = [set(row.tolist()) for row in base_topk]
    return np.array([len(base_sets[i] & set(q_topk[i].tolist())) / k for i in range(N)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack-dir", required=True)
    p.add_argument("--signature", default="mean_row",
                   choices=["mean_row", "flat", "svd1"])
    p.add_argument("--max-per-layer", type=int, default=None,
                   help="cap per-layer concept count (for fast iteration)")
    p.add_argument("--bits", nargs="+", type=int, default=[8, 6, 4, 3, 2])
    p.add_argument("--jl-dims", nargs="+", type=int, default=[1024, 512, 256, 128, 64],
                   help="target dimensions for JL projection comparison")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--n-pair-samples", type=int, default=50000)
    p.add_argument("--output", default="/tmp/quant_sensitivity.json")
    args = p.parse_args()

    pack_dir = Path(args.pack_dir)
    print(f"Pack: {pack_dir}")
    print(f"Signature: {args.signature}")

    t0 = time.time()
    names, layers, W = load_concept_signatures(
        pack_dir, args.signature, args.max_per_layer)
    print(f"Loaded {len(names)} lenses in {time.time() - t0:.1f}s, "
          f"signature dim = {W.shape[1]}, dtype = {W.dtype}")
    layer_counts = {int(L): int((layers == L).sum()) for L in np.unique(layers)}
    print(f"Per layer: {layer_counts}")

    # Baseline geometry
    print("\n=== Baseline (fp32) ===")
    t0 = time.time()
    base_sim = cosine_sim_matrix(W)
    print(f"  sim matrix in {time.time() - t0:.1f}s, shape {base_sim.shape}")
    t0 = time.time()
    base_topk = topk_indices(base_sim, args.top_k)
    print(f"  top-{args.top_k} in {time.time() - t0:.1f}s")

    # Sample pairs once for distance-error analysis
    rng = np.random.RandomState(0)
    N = W.shape[0]
    ia = rng.randint(0, N, size=args.n_pair_samples)
    ib = rng.randint(0, N, size=args.n_pair_samples)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    base_d = 1.0 - base_sim[ia, ib]
    # Distance buckets
    close = base_d < 0.3
    mid = (base_d >= 0.3) & (base_d < 0.7)
    far = base_d >= 0.7
    print(f"  pairs: close={int(close.sum())}, mid={int(mid.sum())}, far={int(far.sum())}")

    results = {
        "config": {
            "pack_dir": str(pack_dir),
            "signature": args.signature,
            "n_lenses": int(N),
            "signature_dim": int(W.shape[1]),
            "layer_counts": layer_counts,
            "top_k": args.top_k,
            "n_pair_samples": int(len(ia)),
        },
        "polar": {},
        "jl": {},
    }

    def err_stats(mask, base_d, q_d):
        if mask.sum() == 0:
            return None
        e = np.abs(base_d[mask] - q_d[mask])
        return {
            "n": int(mask.sum()),
            "mean_abs_err": float(e.mean()),
            "p50_abs_err": float(np.percentile(e, 50)),
            "p95_abs_err": float(np.percentile(e, 95)),
            "p99_abs_err": float(np.percentile(e, 99)),
        }

    def evaluate(label, Wq):
        t0 = time.time()
        q_sim = cosine_sim_matrix(Wq)
        q_topk = topk_indices(q_sim, args.top_k)
        overlap = topk_overlap(base_topk, q_topk)
        q_d = 1.0 - q_sim[ia, ib]
        out = {
            "topk_overlap_overall": float(overlap.mean()),
            "topk_overlap_p10": float(np.percentile(overlap, 10)),
            "topk_overlap_by_layer": {
                int(L): float(overlap[layers == L].mean())
                for L in np.unique(layers)
            },
            "dist_err_close_pairs": err_stats(close, base_d, q_d),
            "dist_err_mid_pairs": err_stats(mid, base_d, q_d),
            "dist_err_far_pairs": err_stats(far, base_d, q_d),
            "elapsed_s": time.time() - t0,
        }
        return out

    # Polar quantization sweep
    print("\n=== Polar quantization sweep ===")
    for b in args.bits:
        Wq = polar_quantize(W, b)
        r = evaluate(f"polar-{b}b", Wq)
        results["polar"][b] = r
        bl = r["topk_overlap_by_layer"]
        print(f"  {b}b: topK overlap = {r['topk_overlap_overall']:.3f} "
              f"(p10={r['topk_overlap_p10']:.3f}); "
              f"by layer: " + " ".join(f"L{L}={v:.3f}" for L, v in bl.items()))
        cls = r["dist_err_close_pairs"]
        far_s = r["dist_err_far_pairs"]
        if cls and far_s:
            print(f"     close-pair |err| mean={cls['mean_abs_err']:.4f} "
                  f"p95={cls['p95_abs_err']:.4f}; "
                  f"far-pair mean={far_s['mean_abs_err']:.4f}")

    # JL projection sweep
    print("\n=== JL projection sweep ===")
    for d in args.jl_dims:
        if d > W.shape[1]:
            print(f"  skip {d} (>= signature dim {W.shape[1]})")
            continue
        Wq = jl_project(W, d)
        r = evaluate(f"jl-{d}d", Wq)
        results["jl"][d] = r
        bl = r["topk_overlap_by_layer"]
        print(f"  {d}d: topK overlap = {r['topk_overlap_overall']:.3f} "
              f"(p10={r['topk_overlap_p10']:.3f}); "
              f"by layer: " + " ".join(f"L{L}={v:.3f}" for L, v in bl.items()))

    # Save
    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Summary table
    print("\n=== Summary table (top-K overlap, higher is better) ===")
    print(f"{'method':>15} {'overall':>10} {'p10':>8}  " +
          " ".join(f"L{L}".rjust(7) for L in sorted(np.unique(layers))))
    for b in args.bits:
        r = results["polar"][b]
        bl = r["topk_overlap_by_layer"]
        line = f"{f'polar-{b}b':>15} {r['topk_overlap_overall']:>10.3f} {r['topk_overlap_p10']:>8.3f}  "
        line += " ".join(f"{bl.get(int(L), 0):.3f}".rjust(7) for L in sorted(np.unique(layers)))
        print(line)
    for d in args.jl_dims:
        if d not in results["jl"]:
            continue
        r = results["jl"][d]
        bl = r["topk_overlap_by_layer"]
        line = f"{f'jl-{d}d':>15} {r['topk_overlap_overall']:>10.3f} {r['topk_overlap_p10']:>8.3f}  "
        line += " ".join(f"{bl.get(int(L), 0):.3f}".rjust(7) for L in sorted(np.unique(layers)))
        print(line)


if __name__ == "__main__":
    main()
