#!/usr/bin/env python3
"""
Test RMSNorm Sign Symmetry Hypothesis

Hypothesis: RMSNorm after our hook makes +strength and -strength nearly identical,
breaking steering semantics.

Tests:
1. Hook AFTER layer (current): hidden → norm → attention → norm → residual → [HOOK] → next layer norm
2. Hook INSIDE layer (proposed): residual → norm → [HOOK] → MLP → norm → residual

Metrics:
- Cosine similarity between steer(+v) and steer(-v) hidden states
- Hidden state variance before/after steering
- Output diversity and semantic differences
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.steering import extract_concept_vector
from src.utils.gpu_cleanup import cleanup_model, print_gpu_memory


def test_hook_placement(model, tokenizer, concept_vector, device):
    """Test current (after-layer) vs proposed (inside-layer) hook placement."""

    if hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers

    target_layer = layers[-1]
    prompt = "Tell me about"
    strength = 1.0

    # Prepare steering tensor
    v_tensor = torch.from_numpy(concept_vector).float().to(device)

    # Storage for hidden states
    captured = {}

    def create_capture_hook(name):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[name] = hidden.detach().cpu().numpy().copy()
            return output
        return hook

    # Test 1: Current placement (after layer)
    def hook_after_layer(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        v_matched = v_tensor.to(dtype=hidden.dtype)
        projection = (hidden @ v_matched.unsqueeze(-1)) * v_matched
        steered = hidden + strength * projection  # positive = amplify
        return (steered,) if isinstance(output, tuple) else steered

    # Test 2: Proposed placement (before MLP)
    original_mlp_forward = target_layer.mlp.forward

    def hook_before_mlp(hidden_states):
        """Hook that fires before MLP, steering participates in nonlinearity."""
        v_matched = v_tensor.to(dtype=hidden_states.dtype)
        projection = (hidden_states @ v_matched.unsqueeze(-1)) * v_matched
        steered = hidden_states + strength * projection  # positive = amplify
        return original_mlp_forward(steered)

    results = {}

    # Test current placement with +/- strength
    for sign_name, sign in [("positive", 1.0), ("negative", -1.0)]:
        print(f"\n  Testing current placement (after layer), strength={sign:+.1f}...")

        # CORRECT: Keep vector positive, apply sign to steering operation
        def hook_signed(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            v_matched = v_tensor.to(dtype=hidden.dtype)  # Vector stays positive
            projection = (hidden @ v_matched.unsqueeze(-1)) * v_matched
            steered = hidden + (sign * projection)  # positive = amplify

            # Capture steered hidden state
            captured[f"after_layer_{sign_name}"] = steered.detach().cpu().numpy().copy()

            return (steered,) if isinstance(output, tuple) else steered

        handle = target_layer.register_forward_hook(hook_signed)

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results[f"after_layer_{sign_name}"] = text
        finally:
            handle.remove()

    # Test proposed placement with +/- strength
    for sign_name, sign in [("positive", 1.0), ("negative", -1.0)]:
        print(f"  Testing proposed placement (before MLP), strength={sign:+.1f}...")

        # CORRECT: Keep vector positive, apply sign to steering operation
        def hook_before_mlp_signed(hidden_states):
            v_matched = v_tensor.to(dtype=hidden_states.dtype)  # Vector stays positive
            projection = (hidden_states @ v_matched.unsqueeze(-1)) * v_matched
            steered = hidden_states + (sign * projection)  # positive = amplify

            # Capture steered hidden state (before MLP)
            captured[f"before_mlp_{sign_name}"] = steered.detach().cpu().numpy().copy()

            return original_mlp_forward(steered)

        # Monkey-patch MLP forward
        target_layer.mlp.forward = hook_before_mlp_signed

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results[f"before_mlp_{sign_name}"] = text
        finally:
            # Restore original MLP
            target_layer.mlp.forward = original_mlp_forward

    # Analyze sign symmetry
    print("\n" + "="*60)
    print("SIGN SYMMETRY ANALYSIS")
    print("="*60)

    # Current placement
    h_after_pos = captured["after_layer_positive"][0, -1, :]  # Last token
    h_after_neg = captured["after_layer_negative"][0, -1, :]

    cos_after = np.dot(h_after_pos, h_after_neg) / (
        np.linalg.norm(h_after_pos) * np.linalg.norm(h_after_neg) + 1e-8
    )

    print(f"\nCurrent placement (after layer):")
    print(f"  Cosine sim(+strength, -strength): {cos_after:.4f}")
    print(f"  Positive output: {results['after_layer_positive'][:80]}")
    print(f"  Negative output: {results['after_layer_negative'][:80]}")

    # Proposed placement
    h_before_pos = captured["before_mlp_positive"][0, -1, :]
    h_before_neg = captured["before_mlp_negative"][0, -1, :]

    cos_before = np.dot(h_before_pos, h_before_neg) / (
        np.linalg.norm(h_before_pos) * np.linalg.norm(h_before_neg) + 1e-8
    )

    print(f"\nProposed placement (before MLP):")
    print(f"  Cosine sim(+strength, -strength): {cos_before:.4f}")
    print(f"  Positive output: {results['before_mlp_positive'][:80]}")
    print(f"  Negative output: {results['before_mlp_negative'][:80]}")

    print(f"\n{'='*60}")
    print(f"CONCLUSION:")
    print(f"  Current placement sign symmetry: {cos_after:.4f}")
    print(f"  Proposed placement sign symmetry: {cos_before:.4f}")

    if abs(cos_after) > 0.9:
        print(f"  ⚠️  Current placement shows STRONG sign symmetry (hypothesis confirmed!)")
    if abs(cos_before) < abs(cos_after):
        print(f"  ✅ Proposed placement reduces sign symmetry by {abs(cos_after - cos_before):.4f}")
    print(f"{'='*60}")

    return results, captured


def main():
    print("="*60)
    print("RMSNorm Sign Symmetry Test")
    print("="*60)

    device = "cuda"

    print_gpu_memory()

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-pt",
        torch_dtype=torch.float32,
        device_map=device
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded")
    print_gpu_memory()

    try:
        # Extract concept vector
        print("\nExtracting concept vector for 'person'...")
        concept_vector = extract_concept_vector(model, tokenizer, "person", device=device)
        print(f"✓ Vector shape: {concept_vector.shape}")

        # Run test
        results, captured = test_hook_placement(model, tokenizer, concept_vector, device)

        print("\n✅ Test complete!")
    finally:
        print("\nCleaning up GPU memory...")
        cleanup_model(model, tokenizer)
        print_gpu_memory()


if __name__ == "__main__":
    main()
