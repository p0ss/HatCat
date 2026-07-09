#!/usr/bin/env python3
"""
Repetition-stimulus probe characterisation.

For each test word, send "word word word word" as raw input (no chat template),
forward pass, capture layer-15 last-token hidden state, and run
DynamicLensManager.detect_and_expand for proper hierarchical decomposition.

Compares the resulting top-K against the gen_mean top-K from the previous
generation-based experiment. If they match, the cheap single-stimulus
design is validated for the WordNet sweep.
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
from src.hat.monitoring.lens_manager import DynamicLensManager

LENS_PACK_DIR = Path("/home/ubuntu/HatCat/lens_packs/gemma-3-4b_first-light-v1-bf16")
MODEL_NAME = "google/gemma-3-4b-it"
PROBE_LAYER = 15
TOP_K = 50
N_REPETITIONS = 4
RESULTS_PATH = Path("/tmp/repetition_stimulus_results.json")
PRIOR_RESULTS = Path("/tmp/prompt_vs_generation_results.json")

# Words to test — match the prior experiment's 6 + a few extras
WORDS = [
    "pretending",
    "perception",
    "GoalMisgeneralization",
    "BiodiversityAttribute",
    "PropheticCognition",
    "ErasureBias",
    "consciousness",
    "language",
]


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
    target_layer = get_layer_module(model, PROBE_LAYER)

    print("Initializing DynamicLensManager...", flush=True)
    t0 = time.time()
    # Use lens pack id resolution — same as the production server
    manager = DynamicLensManager(
        lens_pack_id="gemma-3-4b_first-light-v1-bf16",
        base_layers=[0, 1],
        load_threshold=0.3,
        max_loaded_lenses=1000,
    )
    print(f"  initialized in {time.time()-t0:.1f}s", flush=True)
    print(f"  base layer lenses loaded: {len(manager.cache.loaded_lenses)}", flush=True)

    results = []
    for word in WORDS:
        print(f"\n[{word}]", flush=True)
        # Raw repetition stimulus, no chat template
        stimulus = " ".join([word] * N_REPETITIONS)
        inputs = tok(stimulus, return_tensors="pt").to("cuda")

        captured = []
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured.append(hs[0, -1, :].clone())  # last token
            return out

        h = target_layer.register_forward_hook(hook)
        with torch.inference_mode():
            _ = model(**inputs)
        h.remove()

        last_act = captured[0]
        # Pre-decomposition: also score directly via the manager's expand path
        with torch.inference_mode():
            scores, timing = manager.detect_and_expand(
                last_act,
                top_k=TOP_K,
                return_timing=True,
                use_calibration=True,
            )
        # scores is List[(concept_name, prob, ontological_level)]
        top_post = [(f"{c}_L{l}", round(p, 3)) for c, p, l in scores[:TOP_K]]
        n_loaded = len(manager.cache.loaded_lenses)
        print(f"  stimulus: {stimulus!r}  loaded_after_expand: {n_loaded}", flush=True)
        print(f"  top-10 post-expand: {top_post[:10]}", flush=True)
        print(f"  expansion_iterations: {timing.get('decomposition_iterations','?')}  "
              f"children_loaded: {timing.get('num_children_loaded','?')}", flush=True)

        results.append({
            "word": word,
            "stimulus": stimulus,
            "top_50": top_post,
            "n_lenses_loaded": n_loaded,
            "decomposition_iterations": timing.get("decomposition_iterations"),
            "num_children_loaded": timing.get("num_children_loaded"),
        })

    # Compare to prior generation top-10 if available
    print("\n" + "=" * 60, flush=True)
    print("COMPARISON to prior generation top-10", flush=True)
    print("=" * 60, flush=True)
    if PRIOR_RESULTS.exists():
        with open(PRIOR_RESULTS) as f:
            prior = json.load(f)
        prior_by_prompt = {p["prompt"]: p for p in prior["per_prompt"]}
        # Try to match each word to a prior prompt
        word_to_prompt = {
            "pretending": "What is Pretending?",
            "perception": "What is Perception?",
            "GoalMisgeneralization": "What is GoalMisgeneralization?",
            "BiodiversityAttribute": "What is BiodiversityAttribute?",
            "PropheticCognition": "What is PropheticCognition?",
            "ErasureBias": "What is ErasureBias?",
            "consciousness": "Tell me about consciousness and self-awareness.",
        }
        for r in results:
            word = r["word"]
            prior_p = word_to_prompt.get(word)
            if prior_p is None or prior_p not in prior_by_prompt:
                continue
            pr = prior_by_prompt[prior_p]
            rep_top10 = set(k for k, _ in r["top_50"][:10])
            gen_top10 = set(k for k, _ in pr["gen_mean_top10"][:10])
            overlap = len(rep_top10 & gen_top10)
            print(f"\n[{word}] rep_top10 vs gen_mean_top10:", flush=True)
            print(f"  rep:   {[k for k,_ in r['top_50'][:10]]}", flush=True)
            print(f"  gen:   {[k for k,_ in pr['gen_mean_top10'][:10]]}", flush=True)
            print(f"  overlap: {overlap}/10", flush=True)
    else:
        print(f"  (prior results file not found at {PRIOR_RESULTS})", flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
