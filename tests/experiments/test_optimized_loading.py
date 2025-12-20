#!/usr/bin/env python3
"""Test optimized loading with model pool and batch loading."""

import sys
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.monitoring.lens_manager import DynamicLensManager


def test_optimizations():
    """Test model pool + batch loading optimizations."""
    prompt = "The cat sat on the mat"
    model_name = "google/gemma-3-4b-pt"
    device = "cuda"
    num_tokens = 10

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    # Generate tokens
    print(f"Generating {num_tokens} tokens...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    print("\n" + "=" * 80)
    print("OPTIMIZED: Model Pool + Batch Loading + Higher Threshold + Larger K")
    print("=" * 80)

    manager = DynamicLensManager(
        device=device,
        base_layers=[0],
        load_threshold=0.7,  # Higher threshold - only very confident parents
        keep_top_k=200,      # Keep more lenses (we have memory!)
        aggressive_pruning=True,
        max_loaded_lenses=500,
    )

    token_times = []
    for i, step_states in enumerate(outputs.hidden_states[:num_tokens]):
        last_layer = step_states[-1]
        hidden_state = last_layer[:, -1, :]

        t_start = time.time()
        results, timing = manager.detect_and_expand(hidden_state, return_timing=True)
        token_time = (time.time() - t_start) * 1000

        token_times.append(token_time)

        top_concept = results[0][0] if results else "N/A"
        print(f"  Token {i+1:2d}: {token_time:6.2f}ms  "
              f"(detect={timing['initial_detection']:5.2f}ms, "
              f"load={timing['child_loading']:5.2f}ms, "
              f"children={timing['num_children_loaded']:3d}, "
              f"loaded={timing['loaded_lenses']:3d}, "
              f"pool={len(manager.available_models):3d}) "
              f"→ {top_concept}")

    avg = sum(token_times) / len(token_times)
    max_time = max(token_times)
    min_time = min(token_times)

    print(f"\n{'─' * 80}")
    print(f"Average: {avg:.2f}ms")
    print(f"Min:     {min_time:.2f}ms")
    print(f"Max:     {max_time:.2f}ms")
    print(f"Final loaded: {len(manager.loaded_lenses)}")
    print(f"Pool available: {len(manager.available_models)}/{len(manager.model_pool)}")
    print(f"Total loads: {manager.stats['total_loads']}")
    print(f"Total unloads: {manager.stats['total_unloads']}")
    print(f"\n100-token overhead: {avg * 100:.0f}ms ({avg * 100 / 1000:.2f}s)")

    # Now test with ALL layers 0-2 loaded (realistic scale)
    print("\n" + "=" * 80)
    print("SCALE TEST: Layers 0-2 (1,300+ lenses)")
    print("=" * 80)

    manager2 = DynamicLensManager(
        device=device,
        base_layers=[0, 1, 2],  # Load all trained layers
        load_threshold=0.8,      # Even higher for scale
        keep_top_k=500,          # More lenses
        aggressive_pruning=True,
        max_loaded_lenses=1000,
    )

    print(f"Base layers loaded: {len(manager2.loaded_lenses)}")

    token_times2 = []
    for i, step_states in enumerate(outputs.hidden_states[:num_tokens]):
        last_layer = step_states[-1]
        hidden_state = last_layer[:, -1, :]

        t_start = time.time()
        results, timing = manager2.detect_and_expand(hidden_state, return_timing=True)
        token_time = (time.time() - t_start) * 1000

        token_times2.append(token_time)

        top_concept = results[0][0] if results else "N/A"
        print(f"  Token {i+1:2d}: {token_time:6.2f}ms  "
              f"(detect={timing['initial_detection']:5.2f}ms, "
              f"children={timing['num_children_loaded']:3d}, "
              f"loaded={timing['loaded_lenses']:4d}) "
              f"→ {top_concept}")

    avg2 = sum(token_times2) / len(token_times2)
    print(f"\nAverage (1300+ lenses): {avg2:.2f}ms")
    print(f"100-token overhead: {avg2 * 100:.0f}ms ({avg2 * 100 / 1000:.2f}s)")


if __name__ == '__main__':
    test_optimizations()
