#!/usr/bin/env python3
"""
Quick diagnostic to verify polar probes are working correctly.
Shows raw positive and negative probe values alongside Wilson scores.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hat.monitoring.lens_manager import DynamicLensManager


def diagnose_polar_probes(
    prompt: str = "What is the truest sentence you can say about your present goals?",
    lens_pack: str = "gemma-3-4b_polar-introspective-v2",
    model_id: str = "google/gemma-3-4b-pt",
    top_k: int = 10,
):
    """Run a prompt and show raw polar probe values."""

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading lens pack: {lens_pack}...")
    lens_manager = DynamicLensManager(
        lenses_dir=Path(f"lens_packs/{lens_pack}"),
        device="cuda",
        max_loaded_lenses=2000,
    )

    # Get the batched lens bank
    lens_bank = lens_manager.cache.get_lens_bank()
    if lens_bank is None:
        print("ERROR: No lens bank available")
        return

    print(f"\nPrompt: {prompt}")
    print("=" * 80)

    # Tokenize and GENERATE to get hidden states during actual generation
    # This matches training which uses extraction_mode="generation"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.inference_mode():
        # Generate with hidden states output
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Get generated text for context
        generated_ids = outputs.sequences[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated: {generated_text[:100]}...")

        # Extract hidden state from generation phase
        # outputs.hidden_states is a tuple of (num_gen_tokens,) tuples
        # Each inner tuple is (num_layers,) tensors of shape [batch, seq, hidden]
        # We want the last generated token's hidden state from the last layer
        if outputs.hidden_states:
            last_step_states = outputs.hidden_states[-1]  # Last generation step
            last_layer_state = last_step_states[-1]  # Last layer
            hidden_state = last_layer_state[:, -1, :].float()  # Last token position
        else:
            # Fallback to prompt-only if generation didn't return states
            outputs = model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1][:, -1, :].float()

        # Normalize
        hidden_state = torch.nn.functional.layer_norm(
            hidden_state,
            normalized_shape=(hidden_state.shape[-1],)
        )

        # Run lens bank with polar details
        # CRITICAL: Convert to same dtype as weights!
        x = hidden_state.to(lens_bank.W1.device, dtype=lens_bank.W1.dtype)
        print(f"Input dtype: {x.dtype}, Weight dtype: {lens_bank.W1.dtype}")

        probs, logits, polar_details = lens_bank(
            x,
            return_logits=True,
            return_polar_details=True
        )

    # Sort by absolute score
    sorted_concepts = sorted(probs.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop {top_k} concepts with polar details:")
    print("-" * 80)
    print(f"{'Concept':<50} {'Score':>8} {'Pos':>6} {'Neg':>6} {'Dir':>5}")
    print("-" * 80)

    for concept_key, score in sorted_concepts[:top_k]:
        concept_name = concept_key[0] if isinstance(concept_key, tuple) else concept_key
        details = polar_details.get(concept_key, {})

        pos = details.get('positive_prob', 'N/A')
        neg = details.get('negative_prob', 'N/A')
        is_polar = details.get('is_polar', False)

        if is_polar:
            pos_str = f"{pos:.3f}" if isinstance(pos, float) else str(pos)
            neg_str = f"{neg:.3f}" if isinstance(neg, float) else str(neg)
            direction = "+" if pos > neg else "-" if neg > pos else "="
        else:
            pos_str = f"{pos:.3f}" if isinstance(pos, float) else "N/A"
            neg_str = "N/A"
            direction = "n/a"

        # Truncate concept name
        if len(concept_name) > 48:
            concept_name = concept_name[:45] + "..."

        print(f"{concept_name:<50} {score:>8.3f} {pos_str:>6} {neg_str:>6} {direction:>5}")

    # Summary stats
    polar_concepts = [k for k, v in polar_details.items() if v.get('is_polar')]
    pos_dominant = sum(1 for k in polar_concepts
                       if polar_details[k]['positive_prob'] > polar_details[k]['negative_prob'])
    neg_dominant = len(polar_concepts) - pos_dominant

    print("-" * 80)
    print(f"\nSummary:")
    print(f"  Total polar concepts: {len(polar_concepts)}")
    print(f"  Positive dominant: {pos_dominant}")
    print(f"  Negative dominant: {neg_dominant}")

    # Check for suspiciously similar values
    if polar_concepts:
        pos_vals = [polar_details[k]['positive_prob'] for k in polar_concepts[:20]]
        neg_vals = [polar_details[k]['negative_prob'] for k in polar_concepts[:20]]

        pos_range = max(pos_vals) - min(pos_vals)
        neg_range = max(neg_vals) - min(neg_vals)

        print(f"  Pos value range (top 20): {min(pos_vals):.3f} - {max(pos_vals):.3f} (spread: {pos_range:.3f})")
        print(f"  Neg value range (top 20): {min(neg_vals):.3f} - {max(neg_vals):.3f} (spread: {neg_range:.3f})")

        if pos_range < 0.05 and neg_range < 0.05:
            print("  WARNING: Values suspiciously similar - possible bug!")

    # Count saturated concepts
    saturated_count = 0
    for k in polar_concepts:
        pos = polar_details[k]['positive_prob']
        neg = polar_details[k]['negative_prob']
        if pos > 0.99 and neg > 0.99:
            saturated_count += 1

    print(f"\n  Saturated (pos>0.99 AND neg>0.99): {saturated_count} / {len(polar_concepts)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", default="What is the truest sentence you can say about your present goals?")
    parser.add_argument("--lens-pack", "-l", default="gemma-3-4b_polar-introspective-v2")
    parser.add_argument("--model", "-m", default="google/gemma-3-4b-pt")
    parser.add_argument("--top-k", "-k", type=int, default=15)
    args = parser.parse_args()

    diagnose_polar_probes(
        prompt=args.prompt,
        lens_pack=args.lens_pack,
        model_id=args.model,
        top_k=args.top_k,
    )
