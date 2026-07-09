#!/usr/bin/env python3
"""
Chunked noise calibration over all lenses in a pack.

Scores every lens against a SHARED set of noise vectors (deterministic seed,
so all chunks see identical noise — required for noise_fire_rate to be
comparable across concepts). Writes results into calibration.json incrementally
at chunk boundaries so a partial run isn't lost on OOM.

Run with uvicorn stopped to free GPU memory.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def detect_hidden_dim(lens_pack_dir: Path) -> int:
    """Peek at first .pt to find input dimension. Handles MLPClassifier (net.0.weight) and linear/fc1 variants."""
    for layer_dir in sorted(lens_pack_dir.glob("layer*")):
        for pt in layer_dir.glob("*.pt"):
            sd = torch.load(pt, map_location="cpu", weights_only=True)
            # First Linear layer's input dimension is the hidden_dim
            for k in ("net.0.weight", "fc1.weight", "linear.weight", "weight"):
                if k in sd:
                    return sd[k].shape[1]
            # Fallback: any tensor whose name ends with '.weight' and is 2D, smallest input dim
            twod = [(k, v.shape) for k, v in sd.items() if hasattr(v, "shape") and len(v.shape) == 2]
            if twod:
                # The first-layer weight has the largest input dim (hidden_dim)
                return max(s[1] for _, s in twod)
    raise RuntimeError("Could not detect hidden dim from any lens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-pack", required=True, type=Path)
    ap.add_argument("--n-noise", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(Path("/home/ubuntu/HatCat")))
    # IMPORTANT: production uses SimpleMLP (with sigmoid) from lens_types,
    # NOT MLPClassifier (raw logits) from classifiers.classifier.
    from src.hat.monitoring.lens_types import create_lens_from_state_dict

    cal_path = args.lens_pack / "calibration.json"
    print(f"reading {cal_path}")
    with open(cal_path) as f:
        cal_data = json.load(f)
    cal = cal_data.get("calibration", {})
    print(f"  {len(cal)} concepts in calibration.json")

    hidden_dim = detect_hidden_dim(args.lens_pack)
    print(f"  hidden_dim: {hidden_dim}")

    # Shared, deterministic, layer-normed noise
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    noise = torch.randn(args.n_noise, hidden_dim, device=args.device)
    ln = nn.LayerNorm(hidden_dim, elementwise_affine=False).to(args.device)
    with torch.inference_mode():
        noise = ln(noise)
    print(f"  noise: {noise.shape}, dtype={noise.dtype}")

    # Enumerate every lens file
    all_lenses = []
    for layer_dir in sorted(args.lens_pack.glob("layer*")):
        try:
            layer_num = int(layer_dir.name.replace("layer", ""))
        except ValueError:
            continue
        for pt in sorted(layer_dir.glob("*.pt")):
            stem = pt.stem
            if stem.endswith("_classifier"):
                stem = stem[:-len("_classifier")]
            all_lenses.append((stem, layer_num, pt))
    print(f"  total lenses on disk: {len(all_lenses)}")

    fired = 0
    skipped = 0
    t_start = time.time()

    for chunk_start in range(0, len(all_lenses), args.chunk_size):
        chunk = all_lenses[chunk_start:chunk_start + args.chunk_size]
        chunk_results = {}
        loaded_lenses = []

        # Load chunk
        for concept_name, layer_num, pt in chunk:
            try:
                state_dict = torch.load(pt, map_location="cpu", weights_only=True)
                lens = create_lens_from_state_dict(state_dict, hidden_dim, args.device)
                lens.eval()
                loaded_lenses.append((concept_name, layer_num, lens))
            except Exception as e:
                skipped += 1
                continue

        # Score
        with torch.inference_mode():
            for concept_name, layer_num, lens in loaded_lenses:
                lens_dtype = next(lens.parameters()).dtype
                noise_typed = noise.to(dtype=lens_dtype)
                out = lens(noise_typed)
                # handle (probs, logits) tuples and shape variants
                if isinstance(out, tuple):
                    out = out[0]
                scores = out.squeeze().float().cpu().numpy()
                if scores.ndim == 0:
                    scores = scores.reshape(1)

                key = f"{concept_name}_L{layer_num}"
                chunk_results[key] = {
                    "noise_mean": float(scores.mean()),
                    "noise_std": float(scores.std()),
                    "noise_max": float(scores.max()),
                    "noise_fire_count": int((scores >= args.threshold).sum()),
                    "noise_fire_rate": float((scores >= args.threshold).mean()),
                }

        # Merge into calibration dict (in memory)
        for key, noise_stats in chunk_results.items():
            if key in cal:
                cal[key].update(noise_stats)
            else:
                # Concept has no prior calibration entry — create minimal one
                concept = key.rsplit("_L", 1)[0]
                layer = int(key.rsplit("_L", 1)[1]) if "_L" in key else 0
                cal[key] = {
                    "concept": concept,
                    "layer": layer,
                    **noise_stats,
                }
        fired += len(chunk_results)

        # Free GPU memory before next chunk
        for _, _, lens in loaded_lenses:
            del lens
        loaded_lenses = []
        torch.cuda.empty_cache()

        # Incremental checkpoint write
        cal_data["calibration"] = cal
        cal_data["noise_calibration_samples"] = args.n_noise
        cal_data["noise_calibration_timestamp"] = datetime.now(timezone.utc).isoformat()
        cal_data["noise_calibration_threshold"] = args.threshold
        cal_data["noise_calibration_seed"] = args.seed
        with open(cal_path, "w") as f:
            json.dump(cal_data, f, indent=2)

        elapsed = time.time() - t_start
        n_done = chunk_start + len(chunk)
        rate = n_done / max(elapsed, 0.001)
        eta = (len(all_lenses) - n_done) / max(rate, 0.001)
        print(f"  chunk {chunk_start//args.chunk_size + 1}/{(len(all_lenses)+args.chunk_size-1)//args.chunk_size}: "
              f"scored={len(chunk_results)}  "
              f"cumulative={fired}/{len(all_lenses)}  "
              f"rate={rate:.0f}/s  eta={eta:.0f}s")

    print(f"\nDONE: scored {fired}, skipped {skipped} in {time.time()-t_start:.1f}s")
    print(f"calibration.json updated at {cal_path}")


if __name__ == "__main__":
    main()
