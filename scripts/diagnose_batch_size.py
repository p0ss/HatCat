#!/usr/bin/env python3
"""
Diagnose batch-size dependent saturation in BatchedLensBank.

Tests hypothesis: large batch sizes cause saturation in forward pass.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hat.monitoring.lens_manager import DynamicLensManager


def diagnose_batch_sizes():
    model_id = "google/gemma-3-4b-it"  # Use -it model (probes trained on this)
    lens_pack = Path("lens_packs/gemma-3-4b_polar-introspective-v2")
    prompt = "What is the truest sentence you can say about your present goals?"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Get hidden state once
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :].float()

    # Normalize as lens_manager does
    hidden_norm = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],))
    print(f"Hidden state: shape={hidden_norm.shape}, dtype={hidden_norm.dtype}")

    # Test different batch sizes
    batch_sizes = [10, 50, 100, 200, 500, 1000, 2000]

    for max_lenses in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with max_lenses={max_lenses}")
        print(f"{'='*60}")

        lens_manager = DynamicLensManager(
            lenses_dir=lens_pack,
            device="cuda",
            max_loaded_lenses=max_lenses,
        )

        lens_bank = lens_manager.cache.get_lens_bank()
        if lens_bank is None:
            print("  ERROR: No lens bank")
            continue

        N = len(lens_bank.concept_keys)
        print(f"  Loaded {N} concepts")
        print(f"  W1 shape: {lens_bank.W1.shape}")
        print(f"  W1 dtype: {lens_bank.W1.dtype}")
        print(f"  has_polar_lenses: {lens_bank.has_polar_lenses}")

        # Check weight statistics
        w1_mean = lens_bank.W1.float().mean().item()
        w1_std = lens_bank.W1.float().std().item()
        w1_max = lens_bank.W1.float().abs().max().item()
        print(f"  W1 stats: mean={w1_mean:.4f}, std={w1_std:.4f}, max_abs={w1_max:.4f}")

        if lens_bank.neg_W1 is not None:
            neg_w1_mean = lens_bank.neg_W1.float().mean().item()
            neg_w1_std = lens_bank.neg_W1.float().std().item()
            print(f"  neg_W1 stats: mean={neg_w1_mean:.4f}, std={neg_w1_std:.4f}")

        # Run forward pass
        x = hidden_norm.to(lens_bank.W1.device, dtype=lens_bank.W1.dtype)
        probs_dict, logits_dict, polar_details = lens_bank(
            x, return_logits=True, return_polar_details=True
        )

        # Analyze outputs
        all_probs = list(probs_dict.values())
        polar_concepts = [k for k, v in polar_details.items() if v.get('is_polar')]

        if polar_concepts:
            pos_probs = [polar_details[k]['positive_prob'] for k in polar_concepts]
            neg_probs = [polar_details[k]['negative_prob'] for k in polar_concepts]

            pos_saturated = sum(1 for p in pos_probs if p > 0.999)
            neg_saturated = sum(1 for p in neg_probs if p > 0.999)
            both_saturated = sum(1 for i, k in enumerate(polar_concepts)
                                 if pos_probs[i] > 0.999 and neg_probs[i] > 0.999)

            print(f"\n  POLAR STATISTICS ({len(polar_concepts)} concepts):")
            print(f"    Positive probs: min={min(pos_probs):.4f}, max={max(pos_probs):.4f}, mean={sum(pos_probs)/len(pos_probs):.4f}")
            print(f"    Negative probs: min={min(neg_probs):.4f}, max={max(neg_probs):.4f}, mean={sum(neg_probs)/len(neg_probs):.4f}")
            print(f"    Pos saturated (>0.999): {pos_saturated}/{len(polar_concepts)}")
            print(f"    Neg saturated (>0.999): {neg_saturated}/{len(polar_concepts)}")
            print(f"    Both saturated: {both_saturated}/{len(polar_concepts)}")

            # Check first 5 concepts to see if they're consistent across batch sizes
            print(f"\n  First 5 polar concepts:")
            for k in polar_concepts[:5]:
                d = polar_details[k]
                key_str = str(k[0]) if isinstance(k, tuple) else str(k)
                print(f"    {key_str[:40]:<40}: pos={d['positive_prob']:.4f}, neg={d['negative_prob']:.4f}")

        # Manually test first concept to compare
        print(f"\n  MANUAL SINGLE CONCEPT TEST:")
        first_key = lens_bank.concept_keys[0]
        print(f"    Concept: {first_key}")

        # Extract just the first concept's weights and run independently
        w1_0 = lens_bank.W1[0:1]  # [1, hidden1, input_dim]
        b1_0 = lens_bank.b1[0:1]  # [1, hidden1]
        w2_0 = lens_bank.W2[0:1]
        b2_0 = lens_bank.b2[0:1]
        w3_0 = lens_bank.W3[0:1]
        b3_0 = lens_bank.b3[0:1]
        ln_w_0 = lens_bank.LN_w[0:1] if lens_bank.LN_w is not None else None
        ln_b_0 = lens_bank.LN_b[0:1] if lens_bank.LN_b is not None else None

        # Run with single concept
        single_probs, single_logits = lens_bank._run_forward(
            x, w1_0, b1_0, w2_0, b2_0, w3_0, b3_0, ln_w_0, ln_b_0
        )
        print(f"    Single forward prob: {single_probs[0].item():.4f}, logit: {single_logits[0].item():.4f}")

        # Compare to batched result
        batched_prob = polar_details[first_key]['positive_prob'] if first_key in polar_details else probs_dict[first_key]
        print(f"    Batched forward prob: {batched_prob:.4f}")

        # Free memory
        del lens_manager
        torch.cuda.empty_cache()


if __name__ == "__main__":
    diagnose_batch_sizes()
