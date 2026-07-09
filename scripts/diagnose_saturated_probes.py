#!/usr/bin/env python3
"""
Diagnose which probes are saturating and why.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hat.monitoring.lens_manager import DynamicLensManager


def diagnose():
    model_id = "google/gemma-3-4b-it"
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
        hidden = outputs.hidden_states[-1][:, -1, :].float()

    hidden_norm = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],))

    print(f"Loading lens manager...")
    lens_manager = DynamicLensManager(
        lenses_dir=lens_pack,
        device="cuda",
        max_loaded_lenses=2000,
    )

    lens_bank = lens_manager.cache.get_lens_bank()
    x = hidden_norm.to(lens_bank.W1.device, dtype=lens_bank.W1.dtype)

    # Get raw logits (not just probs)
    # Run positive forward
    pos_probs, pos_logits = lens_bank._run_forward(
        x, lens_bank.W1, lens_bank.b1, lens_bank.W2, lens_bank.b2,
        lens_bank.W3, lens_bank.b3, lens_bank.LN_w, lens_bank.LN_b
    )

    # Run negative forward
    neg_probs, neg_logits = lens_bank._run_forward(
        x, lens_bank.neg_W1, lens_bank.neg_b1, lens_bank.neg_W2, lens_bank.neg_b2,
        lens_bank.neg_W3, lens_bank.neg_b3, lens_bank.neg_LN_w, lens_bank.neg_LN_b
    )

    print(f"\n{'='*80}")
    print("LOGIT ANALYSIS")
    print(f"{'='*80}")

    pos_logits_np = pos_logits.cpu().numpy()
    neg_logits_np = neg_logits.cpu().numpy()
    pos_probs_np = pos_probs.cpu().numpy()
    neg_probs_np = neg_probs.cpu().numpy()

    # Find saturated concepts (both > 0.999)
    both_saturated_mask = (pos_probs_np > 0.999) & (neg_probs_np > 0.999)
    pos_only_saturated = (pos_probs_np > 0.999) & (neg_probs_np <= 0.999)
    neg_only_saturated = (pos_probs_np <= 0.999) & (neg_probs_np > 0.999)
    neither_saturated = (pos_probs_np <= 0.999) & (neg_probs_np <= 0.999)

    print(f"\nSaturation breakdown:")
    print(f"  Both saturated: {both_saturated_mask.sum()}")
    print(f"  Pos only saturated: {pos_only_saturated.sum()}")
    print(f"  Neg only saturated: {neg_only_saturated.sum()}")
    print(f"  Neither saturated: {neither_saturated.sum()}")

    print(f"\nLogit statistics for BOTH SATURATED probes:")
    if both_saturated_mask.sum() > 0:
        bs_pos_logits = pos_logits_np[both_saturated_mask]
        bs_neg_logits = neg_logits_np[both_saturated_mask]
        print(f"  Positive logits: min={bs_pos_logits.min():.1f}, max={bs_pos_logits.max():.1f}, mean={bs_pos_logits.mean():.1f}")
        print(f"  Negative logits: min={bs_neg_logits.min():.1f}, max={bs_neg_logits.max():.1f}, mean={bs_neg_logits.mean():.1f}")

    print(f"\nLogit statistics for NEITHER SATURATED probes:")
    if neither_saturated.sum() > 0:
        ns_pos_logits = pos_logits_np[neither_saturated]
        ns_neg_logits = neg_logits_np[neither_saturated]
        print(f"  Positive logits: min={ns_pos_logits.min():.1f}, max={ns_pos_logits.max():.1f}, mean={ns_pos_logits.mean():.1f}")
        print(f"  Negative logits: min={ns_neg_logits.min():.1f}, max={ns_neg_logits.max():.1f}, mean={ns_neg_logits.mean():.1f}")

    # Check which layers/levels the saturated probes are from
    print(f"\n{'='*80}")
    print("SATURATED PROBE LAYER ANALYSIS")
    print(f"{'='*80}")

    saturated_indices = [i for i in range(len(lens_bank.concept_keys)) if both_saturated_mask[i]]

    layer_counts = {}
    for idx in saturated_indices:
        key = lens_bank.concept_keys[idx]
        if isinstance(key, tuple) and len(key) > 1:
            layer = key[1]
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

    print(f"\nSaturated probes by layer:")
    for layer in sorted(layer_counts.keys()):
        print(f"  Layer {layer}: {layer_counts[layer]}")

    # Sample some saturated concepts
    print(f"\n{'='*80}")
    print("SAMPLE SATURATED CONCEPTS")
    print(f"{'='*80}")

    for idx in saturated_indices[:10]:
        key = lens_bank.concept_keys[idx]
        name = key[0] if isinstance(key, tuple) else key
        layer = key[1] if isinstance(key, tuple) and len(key) > 1 else "?"
        print(f"  [{layer}] {name[:60]}")
        print(f"       pos_logit={pos_logits_np[idx]:.1f}, neg_logit={neg_logits_np[idx]:.1f}")

    # Sample some healthy concepts
    print(f"\n{'='*80}")
    print("SAMPLE HEALTHY CONCEPTS (neither saturated)")
    print(f"{'='*80}")

    healthy_indices = [i for i in range(len(lens_bank.concept_keys)) if neither_saturated[i]]
    for idx in healthy_indices[:10]:
        key = lens_bank.concept_keys[idx]
        name = key[0] if isinstance(key, tuple) else key
        layer = key[1] if isinstance(key, tuple) and len(key) > 1 else "?"
        print(f"  [{layer}] {name[:60]}")
        print(f"       pos_logit={pos_logits_np[idx]:.1f} -> prob={pos_probs_np[idx]:.4f}")
        print(f"       neg_logit={neg_logits_np[idx]:.1f} -> prob={neg_probs_np[idx]:.4f}")


if __name__ == "__main__":
    diagnose()
