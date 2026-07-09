#!/usr/bin/env python3
"""Diagnose why probes are saturating at 1.0"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def diagnose():
    model_id = "google/gemma-3-4b-pt"
    lens_pack = Path("lens_packs/gemma-3-4b_polar-introspective-v2")
    prompt = "What is the truest sentence you can say about your present goals?"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Get hidden state
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :].float().cpu()

    print(f"\n1. RAW HIDDEN STATE:")
    print(f"   Shape: {hidden.shape}")
    print(f"   Mean: {hidden.mean().item():.4f}")
    print(f"   Std: {hidden.std().item():.4f}")
    print(f"   Min: {hidden.min().item():.4f}")
    print(f"   Max: {hidden.max().item():.4f}")

    # Normalize (as lens_manager does)
    hidden_norm = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],))
    print(f"\n2. NORMALIZED HIDDEN STATE:")
    print(f"   Mean: {hidden_norm.mean().item():.6f}")
    print(f"   Std: {hidden_norm.std().item():.4f}")
    print(f"   Min: {hidden_norm.min().item():.4f}")
    print(f"   Max: {hidden_norm.max().item():.4f}")

    # Load a sample probe
    sample_probe = list((lens_pack / "L2/layer17").glob("*_positive.pt"))[0]
    print(f"\n3. SAMPLE PROBE: {sample_probe.name}")

    state = torch.load(sample_probe, map_location="cpu", weights_only=True)
    print(f"   Keys: {list(state.keys())}")

    # Check if it has LayerNorm
    has_ln = "net.0.weight" in state and state["net.0.weight"].shape[0] == hidden.shape[-1]
    print(f"   Has LayerNorm: {has_ln}")

    if has_ln:
        ln_w = state["net.0.weight"]
        ln_b = state["net.0.bias"]
        print(f"   LN weight stats: mean={ln_w.mean():.4f}, std={ln_w.std():.4f}")
        print(f"   LN bias stats: mean={ln_b.mean():.4f}, std={ln_b.std():.4f}")

        # W1 is at index 1 after LayerNorm
        w1 = state["net.1.weight"]
        b1 = state["net.1.bias"]
    else:
        w1 = state["net.0.weight"]
        b1 = state["net.0.bias"]

    print(f"\n4. FIRST LINEAR LAYER:")
    print(f"   W1 shape: {w1.shape}")
    print(f"   W1 stats: mean={w1.mean():.4f}, std={w1.std():.4f}, min={w1.min():.4f}, max={w1.max():.4f}")
    print(f"   b1 stats: mean={b1.mean():.4f}, std={b1.std():.4f}")

    # Manual forward pass
    print(f"\n5. MANUAL FORWARD PASS:")
    x = hidden_norm.squeeze()

    # Convert to same dtype as weights
    weight_dtype = w1.dtype
    x = x.to(weight_dtype)
    print(f"   Input dtype: {x.dtype}, weight dtype: {weight_dtype}")

    if has_ln:
        # Apply learned LayerNorm
        x = x * ln_w + ln_b
        print(f"   After LN: mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")

    # First linear
    h1 = x @ w1.T + b1
    print(f"   After W1 (pre-ReLU): mean={h1.float().mean():.4f}, std={h1.float().std():.4f}, min={h1.float().min():.4f}, max={h1.float().max():.4f}")

    h1_relu = torch.relu(h1)
    print(f"   After ReLU: mean={h1_relu.mean():.4f}, std={h1_relu.std():.4f}")

    # Second linear - indices depend on architecture
    # For non-LN: net.0=Linear, net.1=ReLU, net.2=Dropout, net.3=Linear, ...
    # For LN: net.0=LN, net.1=Linear, net.2=ReLU, net.3=Dropout, net.4=Linear, ...
    if has_ln:
        w2 = state["net.4.weight"]
        b2 = state["net.4.bias"]
        w3 = state["net.7.weight"]
        b3 = state["net.7.bias"]
    else:
        w2 = state["net.3.weight"]
        b2 = state["net.3.bias"]
        w3 = state["net.6.weight"]
        b3 = state["net.6.bias"]

    h2 = h1_relu @ w2.T + b2
    print(f"   After W2 (pre-ReLU): mean={h2.float().mean():.4f}, std={h2.float().std():.4f}")

    h2_relu = torch.relu(h2)
    print(f"   After ReLU: mean={h2_relu.float().mean():.4f}")

    logit = h2_relu @ w3.T + b3
    prob = torch.sigmoid(logit.float())

    print(f"\n6. OUTPUT:")
    print(f"   Logit: {logit.item():.4f}")
    print(f"   Probability: {prob.item():.4f}")

    if logit.item() > 10:
        print(f"\n   WARNING: Logit is very large ({logit.item():.1f}), causing sigmoid saturation!")
        print(f"   sigmoid(10) = {torch.sigmoid(torch.tensor(10.0)).item():.6f}")
        print(f"   sigmoid(20) = {torch.sigmoid(torch.tensor(20.0)).item():.10f}")


if __name__ == "__main__":
    diagnose()
