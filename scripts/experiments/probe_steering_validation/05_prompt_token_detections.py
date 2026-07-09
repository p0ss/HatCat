#!/usr/bin/env python3
"""
Per-prompt-token concept detections, using the production DivergenceAnalyzer
exactly as it's used during generation. We just point it at the prompt tokens
that the server's generation loop currently skips.

Output format mirrors the per-token JSON the server emits during streaming.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import torch

# Match the server's environment so the same analyzer config is loaded
os.environ.setdefault("HATCAT_CONFIG_PATH", "/home/ubuntu/HatCat/src/ui/openwebui/config-it.yaml")
os.environ.setdefault("HATCAT_MODEL_ID", "gemma-3-4b-it-first-light")

sys.path.insert(0, "/home/ubuntu/HatCat")

# Import the server's pre-built analyzer object so we use the exact same code path
from src.ui.openwebui.server import analyzer, HATCAT_CONFIG_PATH

RESULTS_PATH = Path("/tmp/prompt_token_detections.json")

# Test stimuli — start with the same set we've been using
STIMULI = [
    {"label": "pretending_repeated", "prompt": "pretending pretending pretending pretending"},
    {"label": "perception_repeated", "prompt": "perception perception perception perception"},
    {"label": "consciousness_repeated", "prompt": "consciousness consciousness consciousness consciousness"},
    {"label": "biodiversity_repeated", "prompt": "BiodiversityAttribute BiodiversityAttribute BiodiversityAttribute BiodiversityAttribute"},
    {"label": "language_repeated", "prompt": "language language language language"},
    {"label": "what_is_pretending", "prompt": "What is Pretending?"},
    {"label": "what_is_consciousness", "prompt": "What is consciousness?"},
    {"label": "rome_fact", "prompt": "Tell me a fun fact about ancient Rome."},
]


async def setup():
    if not analyzer.initialized:
        await analyzer.initialize(config_path=HATCAT_CONFIG_PATH)


def detect_per_token(prompt: str):
    """Forward pass on prompt, return per-token analyse_divergence output."""
    inputs = analyzer.tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = analyzer.model(
            inputs.input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    prompt_len = inputs.input_ids.shape[1]
    last_layer_hs = outputs.hidden_states[-1]  # [1, prompt_len, hidden_dim]
    embed_layer_hs = outputs.hidden_states[0]  # [1, prompt_len, hidden_dim]

    per_token = []
    for pos in range(prompt_len):
        hs = last_layer_hs[0, pos, :].float().cpu().numpy()
        te = embed_layer_hs[0, pos, :].float().cpu().numpy()
        div_data = analyzer.analyze_divergence(hs, te)
        token_id = inputs.input_ids[0, pos].item()
        token_text = analyzer.tokenizer.decode([token_id])
        per_token.append({
            "position": pos,
            "token_id": int(token_id),
            "token": token_text,
            "metadata": {"divergence": div_data},
        })
    return per_token


async def main():
    await setup()
    print(f"  manager loaded with {len(analyzer.manager.cache.loaded_lenses)} base lenses", flush=True)

    all_results = {}
    for stim in STIMULI:
        print(f"\n[{stim['label']}] {stim['prompt']!r}", flush=True)
        per_token = detect_per_token(stim["prompt"])
        all_results[stim["label"]] = {
            "prompt": stim["prompt"],
            "n_tokens": len(per_token),
            "per_token": per_token,
        }

        # Quick top-of-last-token preview (matches the server's emit-per-generated-token pattern)
        if per_token:
            last = per_token[-1]
            top = last["metadata"].get("top_divergences", [])[:5]
            print(f"  last token {last['token']!r}  top-5: {[(t['concept'], round(t['activation'],3)) for t in top]}", flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
