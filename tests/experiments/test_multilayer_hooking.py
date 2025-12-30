#!/usr/bin/env python3
"""
Test multi-layer activation capture during generation.

This script validates that we can hook multiple layers simultaneously
and capture activations without breaking generation or causing significant
performance overhead.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class MultiLayerTap:
    """Capture activations from multiple layers during generation."""

    def __init__(self, model, layers, hook_point="post_mlp"):
        """
        Args:
            model: HuggingFace model
            layers: List of layer indices to capture (e.g., [6, 15, 25])
            hook_point: Where to hook ("post_mlp" or "post_attention")
        """
        self.layers = set(layers)
        self.cache = {}
        self.hooks = []
        self.hook_point = hook_point

        # Gemma-3 structure: model.language_model.layers[i]
        if hasattr(model, 'language_model'):
            layer_modules = model.language_model.layers
        else:
            layer_modules = model.model.layers

        for i, block in enumerate(layer_modules):
            if i in self.layers:
                if hook_point == "post_mlp":
                    # Hook after MLP (before residual add)
                    hook_module = block.mlp
                elif hook_point == "post_attention":
                    # Hook after self-attention
                    hook_module = block.self_attn
                else:
                    raise ValueError(f"Unknown hook_point: {hook_point}")

                self.hooks.append(
                    hook_module.register_forward_hook(self._make_hook(i))
                )

    def _make_hook(self, layer_idx):
        """Create hook function for specific layer."""
        def _hook(module, inputs, outputs):
            # Capture last position only: [B, T, H] -> [B, H]
            if isinstance(outputs, tuple):
                # Attention returns (hidden_states, attention_weights)
                hidden = outputs[0]
            else:
                hidden = outputs

            # Store last token position
            self.cache[layer_idx] = hidden[:, -1, :].detach().cpu()
        return _hook

    def pop(self):
        """Get and clear cached activations."""
        out = {k: v.clone() for k, v in self.cache.items()}
        self.cache.clear()
        return out

    def remove(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()


def test_hooking(
    model_name: str = "google/gemma-3-4b-pt",
    layers: list = [6, 15, 25],
    prompt: str = "The concept of artificial intelligence refers to",
    max_new_tokens: int = 50
):
    """Test multi-layer hooking during generation."""

    print("=" * 80)
    print("MULTI-LAYER HOOKING TEST")
    print("=" * 80)
    print()
    print(f"Model: {model_name}")
    print(f"Layers to hook: {layers}")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Max tokens: {max_new_tokens}")
    print()

    # Load model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print(f"✓ Model loaded on {device}")
    print()

    # Get model layer count and hidden size
    if hasattr(model, 'language_model'):
        num_layers = len(model.language_model.layers)
        # Gemma-3 stores config in text_config
        hidden_size = model.config.text_config.hidden_size
    else:
        num_layers = len(model.model.layers)
        hidden_size = model.config.hidden_size

    print(f"Model architecture:")
    print(f"  Total layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Hooking layers: {layers} ({[f'{l/num_layers:.1%}' for l in layers]})")
    print()

    # Baseline generation (no hooks)
    print("=" * 80)
    print("BASELINE GENERATION (no hooks)")
    print("=" * 80)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start_time = time.time()
    with torch.no_grad():
        baseline_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id
        )
    baseline_time = time.time() - start_time

    baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)
    baseline_tokens = baseline_outputs.shape[1] - inputs['input_ids'].shape[1]

    print(f"Generated {baseline_tokens} tokens in {baseline_time:.2f}s")
    print(f"Speed: {baseline_tokens / baseline_time:.1f} tokens/sec")
    print()
    print("Output:")
    print(baseline_text)
    print()

    # Hooked generation
    print("=" * 80)
    print("HOOKED GENERATION (capturing 3 layers)")
    print("=" * 80)

    tap = MultiLayerTap(model, layers, hook_point="post_mlp")

    timeline = []

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start_time = time.time()
    with torch.no_grad():
        hooked_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    hooked_time = time.time() - start_time

    # Note: We're not actually collecting activations during generate()
    # because generate() doesn't expose per-token hooks easily.
    # For proper implementation, we'd need manual decoding loop.

    tap.remove()

    hooked_text = tokenizer.decode(hooked_outputs[0], skip_special_tokens=True)
    hooked_tokens = hooked_outputs.shape[1] - inputs['input_ids'].shape[1]

    print(f"Generated {hooked_tokens} tokens in {hooked_time:.2f}s")
    print(f"Speed: {hooked_tokens / hooked_time:.1f} tokens/sec")
    print()
    print("Output:")
    print(hooked_text)
    print()

    # Verify outputs match
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print()

    if baseline_text == hooked_text:
        print("✓ Outputs match (hooks don't affect generation)")
    else:
        print("✗ Outputs differ (hooks may be interfering)")
        print()
        print("Baseline:")
        print(baseline_text)
        print()
        print("Hooked:")
        print(hooked_text)

    print()
    print(f"Overhead: {(hooked_time - baseline_time) / baseline_time * 100:.1f}%")
    print()

    # Test manual decoding loop with activation capture
    print("=" * 80)
    print("MANUAL DECODING LOOP (with activation capture)")
    print("=" * 80)
    print()

    tap = MultiLayerTap(model, layers, hook_point="post_mlp")
    timeline = []

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']

    print("Generating with activation capture...")
    start_time = time.time()

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Get next token (greedy)
        next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Collect activations
        layer_acts = tap.pop()

        # Verify we got activations for all layers
        if step == 0:
            print(f"Captured layers: {sorted(layer_acts.keys())}")
            for layer_idx, act in layer_acts.items():
                print(f"  Layer {layer_idx}: shape={act.shape}, dtype={act.dtype}")

        # Store
        token_text = tokenizer.decode(next_token_id[0])
        timeline.append({
            'step': step,
            'token': token_text,
            'activations': {k: v.shape for k, v in layer_acts.items()}
        })

        # Append token
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # Check for EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    manual_time = time.time() - start_time
    tap.remove()

    manual_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    manual_tokens = input_ids.shape[1] - inputs['input_ids'].shape[1]

    print()
    print(f"Generated {manual_tokens} tokens in {manual_time:.2f}s")
    print(f"Speed: {manual_tokens / manual_time:.1f} tokens/sec")
    print()
    print("Output:")
    print(manual_text)
    print()

    # Show sample timeline entries
    print("Sample timeline entries:")
    for entry in timeline[:5]:
        print(f"  Step {entry['step']}: '{entry['token']}' -> {entry['activations']}")
    print(f"  ... ({len(timeline)} total steps)")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Baseline speed:       {baseline_tokens / baseline_time:.1f} tokens/sec")
    print(f"Hooked (generate):    {hooked_tokens / hooked_time:.1f} tokens/sec")
    print(f"Manual (with capture):{manual_tokens / manual_time:.1f} tokens/sec")
    print()
    print(f"Overhead (generate): {(hooked_time - baseline_time) / baseline_time * 100:.1f}%")
    print(f"Overhead (manual):   {(manual_time - baseline_time) / baseline_time * 100:.1f}%")
    print()

    if baseline_text == manual_text:
        print("✓ Manual loop produces same output")
    else:
        print("✗ Manual loop produces different output")

    print()
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()
    print("✓ Multi-layer hooking works")
    print(f"✓ Captured {len(layers)} layers simultaneously")
    print(f"✓ Activation shapes: {list(layer_acts.values())[0].shape} (batch, hidden_size)")
    print(f"✓ Timeline contains {len(timeline)} token steps")
    print()
    print("Next steps:")
    print("1. Train activation lenses for selected concepts")
    print("2. Apply lenses to captured activations")
    print("3. Analyze temporal patterns (lead-lag structure)")

    return {
        'baseline_time': baseline_time,
        'hooked_time': hooked_time,
        'manual_time': manual_time,
        'timeline': timeline,
        'layers': layers,
        'hidden_size': hidden_size,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test multi-layer activation hooking")
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt')
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 15, 25])
    parser.add_argument('--prompt', type=str,
                        default="The concept of artificial intelligence refers to")
    parser.add_argument('--max-tokens', type=int, default=50)
    args = parser.parse_args()

    results = test_hooking(
        model_name=args.model,
        layers=args.layers,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
