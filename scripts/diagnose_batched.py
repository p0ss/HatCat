#!/usr/bin/env python3
"""Diagnose BatchedLensBank forward pass"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hat.monitoring.lens_manager import DynamicLensManager


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

    print(f"Loading lens manager...")
    lens_manager = DynamicLensManager(
        lenses_dir=lens_pack,
        device="cuda",
        max_loaded_lenses=500,
    )

    lens_bank = lens_manager.cache.get_lens_bank()
    print(f"\nBatchedLensBank state:")
    print(f"  has_layer_norm: {lens_bank.has_layer_norm}")
    print(f"  has_polar_lenses: {lens_bank.has_polar_lenses}")
    print(f"  num concepts: {len(lens_bank.concept_keys)}")
    print(f"  W1 shape: {lens_bank.W1.shape}")
    print(f"  LN_w: {lens_bank.LN_w}")
    if lens_bank.neg_W1 is not None:
        print(f"  neg_W1 shape: {lens_bank.neg_W1.shape}")
        print(f"  neg_LN_w: {lens_bank.neg_LN_w}")

    # Get hidden state
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :].float()

    print(f"\nHidden state before normalization:")
    print(f"  shape: {hidden.shape}")
    print(f"  mean: {hidden.mean():.4f}, std: {hidden.std():.4f}")

    # Normalize as lens_manager does
    hidden_norm = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],))
    print(f"\nHidden state after normalization:")
    print(f"  mean: {hidden_norm.mean():.6f}, std: {hidden_norm.std():.4f}")

    # Convert to bank dtype
    x = hidden_norm.to(lens_bank.W1.device, dtype=lens_bank.W1.dtype)
    print(f"  dtype after conversion: {x.dtype}")

    # Manual forward through bank
    N = lens_bank.W1.shape[0]
    print(f"\nManual forward pass (N={N} concepts):")

    if lens_bank.has_layer_norm and lens_bank.LN_w is not None:
        print("  Using LayerNorm path")
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        x_expanded = x_norm * lens_bank.LN_w + lens_bank.LN_b
        x_expanded = x_expanded.unsqueeze(1)
    else:
        print("  Using expand path (no LayerNorm)")
        x_expanded = x.expand(N, -1, -1)

    print(f"  x_expanded shape: {x_expanded.shape}")

    # First layer
    h1 = torch.bmm(x_expanded, lens_bank.W1.transpose(1, 2))
    h1 = h1.squeeze(1) + lens_bank.b1
    print(f"  h1 (pre-ReLU): mean={h1.float().mean():.4f}, std={h1.float().std():.4f}, min={h1.float().min():.4f}, max={h1.float().max():.4f}")

    h1 = torch.relu(h1)
    print(f"  h1 (post-ReLU): mean={h1.float().mean():.4f}")

    # Second layer
    h2 = torch.bmm(h1.unsqueeze(1), lens_bank.W2.transpose(1, 2))
    h2 = h2.squeeze(1) + lens_bank.b2
    print(f"  h2 (pre-ReLU): mean={h2.float().mean():.4f}, std={h2.float().std():.4f}")

    h2 = torch.relu(h2)

    # Output
    logits = torch.bmm(h2.unsqueeze(1), lens_bank.W3.transpose(1, 2))
    logits = logits.squeeze(-1).squeeze(-1) + lens_bank.b3.squeeze(-1)
    probs = torch.sigmoid(logits.float())

    print(f"\n  Logits: mean={logits.float().mean():.4f}, std={logits.float().std():.4f}, min={logits.float().min():.4f}, max={logits.float().max():.4f}")
    print(f"  Probs: mean={probs.mean():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")

    # Check distribution of probs
    prob_vals = probs.cpu().numpy()
    print(f"\n  Prob distribution:")
    print(f"    < 0.1: {(prob_vals < 0.1).sum()}")
    print(f"    0.1-0.5: {((prob_vals >= 0.1) & (prob_vals < 0.5)).sum()}")
    print(f"    0.5-0.9: {((prob_vals >= 0.5) & (prob_vals < 0.9)).sum()}")
    print(f"    > 0.9: {(prob_vals > 0.9).sum()}")
    print(f"    = 1.0: {(prob_vals >= 0.9999).sum()}")

    # Now run through the actual bank and compare
    print("\n\nActual lens_bank forward:")
    probs_dict, logits_dict, polar_details = lens_bank(x, return_logits=True, return_polar_details=True)

    first_key = list(probs_dict.keys())[0]
    print(f"  First concept: {first_key}")
    print(f"  Score: {probs_dict[first_key]}")
    if first_key in polar_details:
        pd = polar_details[first_key]
        print(f"  Polar details: pos={pd.get('positive_prob')}, neg={pd.get('negative_prob')}")


if __name__ == "__main__":
    diagnose()
