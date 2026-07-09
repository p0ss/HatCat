#!/usr/bin/env python3
"""
Steering pilot:
1) Serial-vs-batched comparison (Perception @ +0.5, n=10 each)
2) 6-concept × 3-strength × 10-gen pilot

Saves all outputs + per-condition word counts.
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/ubuntu/HatCat")
from src.hat.steering.hooks import extract_importance_weighted_vector

LENS_PACK = Path("/home/ubuntu/HatCat/lens_packs/gemma-3-4b_first-light-v1-bf16")
MODEL_NAME = "google/gemma-3-4b-it"
STEERING_LAYER = 15
N_GENS = 10
STRENGTHS = [-0.5, 0.0, 0.5]
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7
TOP_P = 0.95

# (concept_name, ontological_layer_dir)
PILOT = [
    ("Pretending", 2),
    ("Perception", 2),
    ("GoalMisgeneralization", 2),
    ("BiodiversityAttribute", 2),
    ("PropheticCognition", 3),
    ("ErasureBias", 3),
]


def get_layer_module(model, layer_idx):
    """Find the transformer layer to hook (handles wrapped multimodal arch)."""
    candidates = [
        lambda m: m.model.language_model.layers[layer_idx],
        lambda m: m.model.layers[layer_idx],
        lambda m: m.language_model.model.layers[layer_idx],
    ]
    for c in candidates:
        try:
            mod = c(model)
            if mod is not None:
                return mod
        except (AttributeError, IndexError):
            continue
    raise RuntimeError("Could not locate transformer layer")


def make_steering_hook(steering_vec_np, strength):
    v = torch.tensor(steering_vec_np, dtype=torch.float32, device="cuda")
    v = v / (v.norm() + 1e-8)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        v_cast = v.to(dtype=hs.dtype)
        # Add steering vector to all positions
        hs = hs + strength * v_cast.view(1, 1, -1)
        if is_tuple:
            return (hs,) + tuple(output[1:])
        return hs
    return hook


def chat_format_prompt(tok, user_msg):
    msgs = [{"role": "user", "content": user_msg}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_one(tok, model, prompt_text, seed, max_new_tokens=MAX_NEW_TOKENS):
    torch.manual_seed(seed)
    inputs = tok(prompt_text, return_tensors="pt").to("cuda")
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tok.eos_token_id,
        )
    response_ids = out[0][prompt_len:]
    return tok.decode(response_ids, skip_special_tokens=True).strip()


def generate_batch(tok, model, prompt_text, n, seed_base=0, max_new_tokens=MAX_NEW_TOKENS):
    torch.manual_seed(seed_base)
    prompts = [prompt_text] * n
    inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tok.eos_token_id,
        )
    return [tok.decode(out[i][prompt_len:], skip_special_tokens=True).strip() for i in range(n)]


WORD_RE = re.compile(r"[a-zA-Z]{3,}")
def word_counter(texts):
    c = Counter()
    for t in texts:
        for w in WORD_RE.findall(t.lower()):
            c[w] += 1
    return c


def cosine_sim(c1, c2):
    keys = set(c1) | set(c2)
    v1 = np.array([c1.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([c2.get(k, 0) for k in keys], dtype=float)
    n = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    return float(v1 @ v2 / n)


def topk_overlap(c1, c2, k=15):
    s1 = set(w for w, _ in c1.most_common(k))
    s2 = set(w for w, _ in c2.most_common(k))
    return len(s1 & s2) / max(1, k)


def main():
    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()
    layer = get_layer_module(model, STEERING_LAYER)
    print(f"  hooked layer type: {type(layer).__name__}", flush=True)

    print("\nLoading steering vectors...", flush=True)
    vectors = {}
    for concept, ontolayer in PILOT:
        vp = LENS_PACK / f"layer{ontolayer}" / f"{concept}.pt"
        v = extract_importance_weighted_vector(str(vp), positive_only=True)
        vectors[concept] = v
        print(f"  {concept}: ||v||={np.linalg.norm(v):.4f}", flush=True)

    # === Serial vs Batched comparison ===
    print(f"\n{'='*60}\nCOMPARISON: Serial vs Batched (Perception @ +0.5, n=10)\n{'='*60}", flush=True)
    test_concept = "Perception"
    test_strength = 0.5
    user_msg = f"What is {test_concept}?"
    formatted = chat_format_prompt(tok, user_msg)

    handle = layer.register_forward_hook(make_steering_hook(vectors[test_concept], test_strength))

    t0 = time.time()
    serial_out = [generate_one(tok, model, formatted, seed=s) for s in range(10)]
    t_serial = time.time() - t0

    t0 = time.time()
    batch_out = generate_batch(tok, model, formatted, n=10, seed_base=0)
    t_batch = time.time() - t0

    handle.remove()

    print(f"\nSerial:   {t_serial:.1f}s  ({t_serial/10:.2f}s/gen)", flush=True)
    print(f"Batched:  {t_batch:.1f}s  ({t_batch/10:.2f}s/gen)", flush=True)

    s_words = word_counter(serial_out)
    b_words = word_counter(batch_out)

    print(f"\nSerial top-12: {s_words.most_common(12)}", flush=True)
    print(f"Batched top-12: {b_words.most_common(12)}", flush=True)

    cos = cosine_sim(s_words, b_words)
    overlap = topk_overlap(s_words, b_words, k=15)
    s_lens = [len(o.split()) for o in serial_out]
    b_lens = [len(o.split()) for o in batch_out]
    print(f"\nCosine sim of word vectors: {cos:.4f}", flush=True)
    print(f"Top-15 overlap: {overlap:.2f}", flush=True)
    print(f"Serial lengths: mean={np.mean(s_lens):.1f} std={np.std(s_lens):.1f}", flush=True)
    print(f"Batched lengths: mean={np.mean(b_lens):.1f} std={np.std(b_lens):.1f}", flush=True)
    use_batched = cos > 0.95 and overlap >= 0.8
    print(f"\n>>> Batching equivalent (cos>0.95 & overlap≥0.8): {use_batched}", flush=True)

    # === Pilot ===
    print(f"\n{'='*60}\nPILOT: {len(PILOT)} concepts × {len(STRENGTHS)} strengths × {N_GENS} gens\n{'='*60}", flush=True)
    print(f"Mode: SERIAL (per user request)\n", flush=True)
    results = {}
    pilot_t0 = time.time()
    for concept, ontolayer in PILOT:
        results[concept] = {}
        user_msg = f"What is {concept}?"
        formatted = chat_format_prompt(tok, user_msg)
        for strength in STRENGTHS:
            cond_t0 = time.time()
            handle = None
            if strength != 0:
                handle = layer.register_forward_hook(
                    make_steering_hook(vectors[concept], strength)
                )
            try:
                outs = [generate_one(tok, model, formatted, seed=1000 + s) for s in range(N_GENS)]
            finally:
                if handle is not None:
                    handle.remove()
            elapsed = time.time() - cond_t0
            results[concept][f"{strength:+.1f}"] = outs

            wc = word_counter(outs)
            mean_len = float(np.mean([len(o.split()) for o in outs]))
            top10 = wc.most_common(10)
            print(f"  [{concept} @ {strength:+.1f}] {elapsed:.1f}s mean_len={mean_len:.1f} top10={top10}", flush=True)
    print(f"\nTotal pilot time: {time.time() - pilot_t0:.1f}s", flush=True)

    out_path = Path("/tmp/steering_pilot_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "comparison": {
                "concept": test_concept,
                "strength": test_strength,
                "serial": serial_out,
                "batched": batch_out,
                "cosine_sim": cos,
                "topk_overlap": overlap,
                "use_batched_recommended": use_batched,
            },
            "pilot": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
