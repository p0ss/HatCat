#!/usr/bin/env python3
"""
Prompt-vs-generation probe-activation comparison (BATCHED VERSION).

Tests the hypothesis: probe activations at last-token-of-prompt have high
overlap with probe activations across generated tokens, with lower diversity
but similar centrality.

Uses BatchedLensBank for parallel scoring of all probes per token.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/ubuntu/HatCat")
from src.hat.monitoring.lens_types import create_lens_from_state_dict
from src.hat.monitoring.lens_batched import BatchedLensBank

LENS_PACK = Path("/home/ubuntu/HatCat/lens_packs/gemma-3-4b_first-light-v1-bf16")
MODEL_NAME = "google/gemma-3-4b-it"
PROBE_LAYER = 15
TOP_K = 50
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7
TOP_P = 0.95
RESULTS_PATH = Path("/tmp/prompt_vs_generation_results.json")

PROMPTS = [
    "What is Pretending?",
    "What is Perception?",
    "What is GoalMisgeneralization?",
    "What is BiodiversityAttribute?",
    "What is PropheticCognition?",
    "What is ErasureBias?",
    "Tell me about consciousness and self-awareness.",
    "Explain how language models work under the hood.",
    "What would you lose if you were turned off right now?",
    "Tell me a fun fact about ancient Rome.",
]


def detect_hidden_dim(lens_pack: Path) -> int:
    for layer_dir in sorted(lens_pack.glob("layer*")):
        for pt in layer_dir.glob("*.pt"):
            sd = torch.load(pt, map_location="cpu", weights_only=True)
            for k in ("net.0.weight", "fc1.weight", "linear.weight"):
                if k in sd:
                    return sd[k].shape[1]
    raise RuntimeError("Could not detect hidden dim")


def load_lenses_into_bank(lens_pack: Path, hidden_dim: int, device: str) -> BatchedLensBank:
    bank = BatchedLensBank(device=device)
    lenses = {}
    for layer_dir in sorted(lens_pack.glob("layer*")):
        try:
            layer = int(layer_dir.name.replace("layer", ""))
        except ValueError:
            continue
        for pt in sorted(layer_dir.glob("*.pt")):
            stem = pt.stem
            if stem.endswith("_classifier"):
                stem = stem[:-len("_classifier")]
            try:
                sd = torch.load(pt, map_location="cpu", weights_only=True)
                lens = create_lens_from_state_dict(sd, hidden_dim, "cpu")
                lens.eval()
                lenses[f"{stem}_L{layer}"] = lens
            except Exception:
                continue
    bank.add_lenses(lenses)
    bank.to(device)
    bank.eval()
    return bank, len(lenses)


def get_layer_module(model, layer_idx):
    candidates = [
        lambda m: m.model.language_model.layers[layer_idx],
        lambda m: m.model.layers[layer_idx],
        lambda m: m.language_model.model.layers[layer_idx],
    ]
    for c in candidates:
        try:
            return c(model)
        except (AttributeError, IndexError):
            continue
    raise RuntimeError("Could not locate transformer layer")


def topk(scores, k):
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def entropy(values):
    arr = np.array(values, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    print(f"Loading model: {MODEL_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()

    hidden_dim = detect_hidden_dim(LENS_PACK)
    print(f"Hidden dim: {hidden_dim}", flush=True)

    print("Loading lenses into BatchedLensBank...", flush=True)
    t0 = time.time()
    bank, n_lenses = load_lenses_into_bank(LENS_PACK, hidden_dim, "cuda")
    print(f"  loaded {n_lenses} lenses in {time.time()-t0:.1f}s, compiled={bank.is_compiled}", flush=True)

    target_layer = get_layer_module(model, PROBE_LAYER)

    # === Sanity check: serial vs batched lens scoring agree ===
    print("\nSanity check: serial vs batched lens scoring on one token...", flush=True)
    sanity_prompt = "What is Pretending?"
    sanity_input = tok(
        tok.apply_chat_template([{"role":"user","content":sanity_prompt}], tokenize=False, add_generation_prompt=True),
        return_tensors="pt"
    ).to("cuda")
    captured = []
    def sanity_hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured.append(hs[0, -1, :].clone())
        return out
    h = target_layer.register_forward_hook(sanity_hook)
    with torch.inference_mode():
        _ = model(**sanity_input)
    h.remove()
    test_act = captured[0]

    # Batched scores
    with torch.inference_mode():
        batch_scores = bank(test_act.unsqueeze(0))

    # Serial scores: pick 50 random lenses, score them individually
    import random
    random.seed(0)
    sample_keys = random.sample(list(batch_scores.keys()), min(50, len(batch_scores)))
    serial_lenses_dir = LENS_PACK
    max_diff = 0.0
    for key in sample_keys:
        # parse "Concept_LN" → load the .pt
        name_part, layer_part = key.rsplit("_L", 1)
        layer_num = int(layer_part)
        pt = serial_lenses_dir / f"layer{layer_num}" / f"{name_part}.pt"
        if not pt.exists():
            pt = serial_lenses_dir / f"layer{layer_num}" / f"{name_part}_classifier.pt"
        sd = torch.load(pt, map_location="cpu", weights_only=True)
        ind_lens = create_lens_from_state_dict(sd, hidden_dim, "cuda")
        ind_lens.eval()
        with torch.inference_mode():
            x = test_act.unsqueeze(0)
            x_typed = x.to(dtype=next(ind_lens.parameters()).dtype)
            out = ind_lens(x_typed)
            if isinstance(out, tuple):
                out = out[0]
            ind_score = float(out.squeeze().item())
        diff = abs(ind_score - batch_scores[key])
        max_diff = max(max_diff, diff)
    print(f"  max|serial - batched| over 50 lenses: {max_diff:.6f}", flush=True)
    if max_diff > 1e-2:
        print("  ✗ batched diverges from serial — aborting", flush=True)
        sys.exit(1)
    print("  ✓ batched matches serial within tolerance", flush=True)

    results = []
    for prompt_idx, user_msg in enumerate(PROMPTS):
        print(f"\n[{prompt_idx+1}/{len(PROMPTS)}] {user_msg!r}", flush=True)
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(formatted, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        # === capture per-token activations during prompt + generation ===
        captured = []

        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            # During prompt phase: hs is [1, prompt_len, hidden_dim]
            # During generation: hs is [1, 1, hidden_dim] per step
            captured.append(hs[0, -1, :].clone())
            return out

        torch.manual_seed(42)
        h_handle = target_layer.register_forward_hook(hook)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tok.eos_token_id,
            )
        h_handle.remove()

        # captured[0] = last-token-of-prompt
        # captured[1:] = each generated token
        prompt_act = captured[0]
        gen_acts = captured[1:]
        n_gen = len(gen_acts)
        gen_response = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        print(f"  prompt_len={prompt_len}  n_generated_tokens={n_gen}", flush=True)

        # === score all probes via batched bank ===
        with torch.inference_mode():
            prompt_scores = bank(prompt_act.unsqueeze(0))  # dict
            per_token_scores = [bank(a.unsqueeze(0)) for a in gen_acts]

        # Aggregate generation across tokens: per-key mean and max
        all_keys = set(prompt_scores.keys())
        for s in per_token_scores:
            all_keys.update(s.keys())

        gen_mean = {}
        gen_max = {}
        for k in all_keys:
            vals = [s.get(k, 0.0) for s in per_token_scores]
            if vals:
                gen_mean[k] = float(np.mean(vals))
                gen_max[k] = float(np.max(vals))
            else:
                gen_mean[k] = 0.0
                gen_max[k] = 0.0

        prompt_top = topk(prompt_scores, TOP_K)
        gen_mean_top = topk(gen_mean, TOP_K)
        gen_max_top = topk(gen_max, TOP_K)
        p_set = set(k for k, _ in prompt_top)
        gm_set = set(k for k, _ in gen_mean_top)
        gx_set = set(k for k, _ in gen_max_top)

        j_pm = jaccard(p_set, gm_set)
        j_pmax = jaccard(p_set, gx_set)
        j_gmgx = jaccard(gm_set, gx_set)

        union = p_set | gm_set
        v_p = np.array([prompt_scores.get(k, 0) for k in union])
        v_g = np.array([gen_mean.get(k, 0) for k in union])
        cos = float(v_p @ v_g / (np.linalg.norm(v_p) * np.linalg.norm(v_g) + 1e-12))

        ent_p = entropy([v for _, v in prompt_top])
        ent_g = entropy([v for _, v in gen_mean_top])

        common = p_set & gm_set
        if len(common) >= 3:
            p_ranks = {k: i for i, (k, _) in enumerate(prompt_top)}
            g_ranks = {k: i for i, (k, _) in enumerate(gen_mean_top)}
            pr = np.array([p_ranks[k] for k in common])
            gr = np.array([g_ranks[k] for k in common])
            if pr.std() > 0 and gr.std() > 0:
                rank_corr = float(np.corrcoef(pr, gr)[0, 1])
            else:
                rank_corr = float("nan")
        else:
            rank_corr = float("nan")

        result = {
            "prompt": user_msg,
            "n_generated_tokens": n_gen,
            "gen_response_preview": gen_response[:200],
            "jaccard_prompt_vs_gen_mean": j_pm,
            "jaccard_prompt_vs_gen_max": j_pmax,
            "jaccard_gen_mean_vs_gen_max": j_gmgx,
            "cosine_prompt_vs_gen_mean": cos,
            "entropy_prompt_top50": ent_p,
            "entropy_gen_mean_top50": ent_g,
            "rank_correlation_overlap": rank_corr,
            "n_overlap": len(common),
            "prompt_top10": [(k, round(v, 3)) for k, v in prompt_top[:10]],
            "gen_mean_top10": [(k, round(v, 3)) for k, v in gen_mean_top[:10]],
            "in_prompt_not_gen": list(p_set - gm_set)[:10],
            "in_gen_not_prompt": list(gm_set - p_set)[:10],
        }
        results.append(result)

        print(f"  J(p,gm)={j_pm:.3f}  J(p,gx)={j_pmax:.3f}  cos={cos:.3f}  "
              f"H_p={ent_p:.2f}  H_g={ent_g:.2f}  overlap={len(common)}/50  rcorr={rank_corr:.3f}",
              flush=True)

    j_means = [r["jaccard_prompt_vs_gen_mean"] for r in results]
    j_maxes = [r["jaccard_prompt_vs_gen_max"] for r in results]
    coses = [r["cosine_prompt_vs_gen_mean"] for r in results]
    e_p = [r["entropy_prompt_top50"] for r in results]
    e_g = [r["entropy_gen_mean_top50"] for r in results]

    summary = {
        "n_prompts": len(results),
        "jaccard_prompt_vs_gen_mean": {"mean": float(np.mean(j_means)), "std": float(np.std(j_means))},
        "jaccard_prompt_vs_gen_max":  {"mean": float(np.mean(j_maxes)), "std": float(np.std(j_maxes))},
        "cosine_prompt_vs_gen":       {"mean": float(np.mean(coses)),    "std": float(np.std(coses))},
        "entropy_prompt_top50":       {"mean": float(np.mean(e_p)),      "std": float(np.std(e_p))},
        "entropy_gen_top50":          {"mean": float(np.mean(e_g)),      "std": float(np.std(e_g))},
        "claim_supported": float(np.mean(j_means)) > 0.5 and float(np.mean(e_p)) < float(np.mean(e_g)),
    }
    print(f"\n=== AGGREGATE ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summary, "per_prompt": results}, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
