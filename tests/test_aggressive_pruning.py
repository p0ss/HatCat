#!/usr/bin/env python3
"""Test aggressive pruning in per-token simulation."""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.dynamic_probe_manager import DynamicProbeManager


def test_with_and_without_pruning():
    """Compare per-token performance with/without aggressive pruning."""
    prompt = "The cat sat on the mat"
    model_name = "google/gemma-3-4b-pt"
    device = "cuda"
    num_tokens = 10

    # Load model once
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    # Generate tokens once
    print(f"Generating {num_tokens} tokens...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # Test 1: WITHOUT aggressive pruning
    print("\n" + "=" * 80)
    print("TEST 1: WITHOUT AGGRESSIVE PRUNING (baseline)")
    print("=" * 80)

    manager1 = DynamicProbeManager(
        device=device,
        base_layers=[0],
        load_threshold=0.3,
        max_loaded_probes=500,
        keep_top_k=50,
        aggressive_pruning=False,  # Disabled
    )

    token_times1 = []
    for i, step_states in enumerate(outputs.hidden_states[:num_tokens]):
        last_layer = step_states[-1]
        hidden_state = last_layer[:, -1, :]

        results, timing = manager1.detect_and_expand(hidden_state, return_timing=True)
        token_times1.append(timing['total'])

        top_concept = results[0][0] if results else "N/A"
        print(f"  Token {i+1:2d}: {timing['total']:6.2f}ms  "
              f"(children={timing['num_children_loaded']:3d}, "
              f"loaded={timing['loaded_probes']:3d}, "
              f"unloaded={manager1.stats['total_unloads']:3d}) "
              f"→ {top_concept}")

    avg1 = sum(token_times1) / len(token_times1)
    max1 = max(token_times1)
    print(f"\nAverage: {avg1:.2f}ms, Max: {max1:.2f}ms")
    print(f"Total unloads: {manager1.stats['total_unloads']}")
    print(f"Final loaded probes: {len(manager1.loaded_probes)}")

    # Test 2: WITH aggressive pruning
    print("\n" + "=" * 80)
    print("TEST 2: WITH AGGRESSIVE PRUNING (keep_top_k=50)")
    print("=" * 80)

    manager2 = DynamicProbeManager(
        device=device,
        base_layers=[0],
        load_threshold=0.3,
        max_loaded_probes=500,
        keep_top_k=50,  # Only keep top 50
        aggressive_pruning=True,  # Enabled
    )

    token_times2 = []
    for i, step_states in enumerate(outputs.hidden_states[:num_tokens]):
        last_layer = step_states[-1]
        hidden_state = last_layer[:, -1, :]

        results, timing = manager2.detect_and_expand(hidden_state, return_timing=True)
        token_times2.append(timing['total'])

        top_concept = results[0][0] if results else "N/A"
        print(f"  Token {i+1:2d}: {timing['total']:6.2f}ms  "
              f"(children={timing['num_children_loaded']:3d}, "
              f"loaded={timing['loaded_probes']:3d}, "
              f"unloaded={manager2.stats['total_unloads']:3d}) "
              f"→ {top_concept}")

    avg2 = sum(token_times2) / len(token_times2)
    max2 = max(token_times2)
    print(f"\nAverage: {avg2:.2f}ms, Max: {max2:.2f}ms")
    print(f"Total unloads: {manager2.stats['total_unloads']}")
    print(f"Final loaded probes: {len(manager2.loaded_probes)}")

    # Test 3: VERY aggressive pruning (keep_top_k=30)
    print("\n" + "=" * 80)
    print("TEST 3: VERY AGGRESSIVE PRUNING (keep_top_k=30)")
    print("=" * 80)

    manager3 = DynamicProbeManager(
        device=device,
        base_layers=[0],
        load_threshold=0.3,
        max_loaded_probes=500,
        keep_top_k=30,  # Even more aggressive
        aggressive_pruning=True,
    )

    token_times3 = []
    for i, step_states in enumerate(outputs.hidden_states[:num_tokens]):
        last_layer = step_states[-1]
        hidden_state = last_layer[:, -1, :]

        results, timing = manager3.detect_and_expand(hidden_state, return_timing=True)
        token_times3.append(timing['total'])

        top_concept = results[0][0] if results else "N/A"
        print(f"  Token {i+1:2d}: {timing['total']:6.2f}ms  "
              f"(children={timing['num_children_loaded']:3d}, "
              f"loaded={timing['loaded_probes']:3d}, "
              f"unloaded={manager3.stats['total_unloads']:3d}) "
              f"→ {top_concept}")

    avg3 = sum(token_times3) / len(token_times3)
    max3 = max(token_times3)
    print(f"\nAverage: {avg3:.2f}ms, Max: {max3:.2f}ms")
    print(f"Total unloads: {manager3.stats['total_unloads']}")
    print(f"Final loaded probes: {len(manager3.loaded_probes)}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"{'Configuration':<30s} {'Avg (ms)':>10s} {'Max (ms)':>10s} {'Final Loaded':>15s}")
    print("-" * 80)
    print(f"{'No pruning':<30s} {avg1:>10.2f} {max1:>10.2f} {len(manager1.loaded_probes):>15d}")
    print(f"{'Aggressive (top-50)':<30s} {avg2:>10.2f} {max2:>10.2f} {len(manager2.loaded_probes):>15d}")
    print(f"{'Very aggressive (top-30)':<30s} {avg3:>10.2f} {max3:>10.2f} {len(manager3.loaded_probes):>15d}")
    print()
    print(f"Speedup (top-50 vs baseline):  {avg1/avg2:.2f}x")
    print(f"Speedup (top-30 vs baseline):  {avg1/avg3:.2f}x")
    print(f"\n100-token overhead estimates:")
    print(f"  No pruning:         {avg1 * 100:.0f}ms ({avg1 * 100 / 1000:.1f}s)")
    print(f"  Aggressive (top-50): {avg2 * 100:.0f}ms ({avg2 * 100 / 1000:.1f}s)")
    print(f"  Very aggressive (top-30): {avg3 * 100:.0f}ms ({avg3 * 100 / 1000:.1f}s)")


if __name__ == '__main__':
    test_with_and_without_pruning()
