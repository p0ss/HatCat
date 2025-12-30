#!/usr/bin/env python3
"""
Benchmark torch.compile() impact on lens inference speed.

Tests:
1. Uncompiled lens inference (baseline)
2. Compiled individual lenses
3. Compiled batch inference (if feasible)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hat.monitoring.lens_manager import DynamicLensManager


def benchmark_uncompiled(lens_manager, hidden_state, num_runs=100):
    """Baseline: current implementation."""
    times = []

    with torch.inference_mode():
        for _ in range(num_runs):
            start = time.perf_counter()
            for concept_key, lens in lens_manager.loaded_lenses.items():
                prob = lens(hidden_state).item()
            times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times) * 1000
    return avg_time, len(lens_manager.loaded_lenses)


def benchmark_compiled_individual(lens_manager, hidden_state, num_runs=100):
    """Compile each lens individually."""
    print("  Compiling individual lenses...")
    compile_start = time.perf_counter()

    compiled_lenses = {}
    for concept_key, lens in lens_manager.loaded_lenses.items():
        compiled_lenses[concept_key] = torch.compile(lens, mode='reduce-overhead')

    compile_time = time.perf_counter() - compile_start

    # Warmup to trigger actual compilation
    print("  Warming up compiled lenses...")
    with torch.inference_mode():
        for lens in compiled_lenses.values():
            _ = lens(hidden_state).item()

    # Benchmark
    times = []
    with torch.inference_mode():
        for _ in range(num_runs):
            start = time.perf_counter()
            for lens in compiled_lenses.values():
                prob = lens(hidden_state).item()
            times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times) * 1000
    return avg_time, len(compiled_lenses), compile_time


def benchmark_compiled_batch(lens_manager, hidden_state, num_runs=100):
    """
    Try to batch all lenses into a single forward pass.

    This stacks all lens weight matrices and runs one matmul.
    Only works if all lenses have identical architecture.
    """
    print("  Attempting batched compilation...")

    # Extract all lens weights
    lenses = list(lens_manager.loaded_lenses.values())

    # Check if all lenses have compatible structure
    try:
        # Get first lens's structure
        first_lens = lenses[0]
        if not hasattr(first_lens, 'state_dict'):
            print("  ✗ Lenses don't have state_dict, cannot batch")
            return None, None, None

        # Try to stack weights
        weights = []
        biases = []

        for lens in lenses:
            state = lens.state_dict()
            # Assuming linear layer structure: weight and bias
            if 'weight' in state and 'bias' in state:
                weights.append(state['weight'])
                biases.append(state['bias'])
            else:
                print(f"  ✗ Lens structure incompatible: {list(state.keys())}")
                return None, None, None

        # Stack into batch
        batch_weight = torch.stack(weights, dim=0)  # [num_lenses, out_features, in_features]
        batch_bias = torch.stack(biases, dim=0)      # [num_lenses, out_features]

        print(f"  Batch shape: weight={batch_weight.shape}, bias={batch_bias.shape}")

        # Create batched forward function
        def batched_forward(x):
            # x: [1, hidden_dim]
            # batch_weight: [num_lenses, 1, hidden_dim]
            # Output: [num_lenses, 1]
            return torch.baddbmm(
                batch_bias.unsqueeze(1),  # [num_lenses, 1, 1]
                batch_weight,              # [num_lenses, 1, hidden_dim]
                x.unsqueeze(0).transpose(1, 2)  # [1, hidden_dim, 1] -> [1, 1, hidden_dim]
            ).squeeze(-1).squeeze(-1)  # [num_lenses]

        # Compile batched function
        compile_start = time.perf_counter()
        compiled_batched = torch.compile(batched_forward, mode='reduce-overhead')
        compile_time = time.perf_counter() - compile_start

        # Warmup
        print("  Warming up batched compiled function...")
        with torch.inference_mode():
            _ = compiled_batched(hidden_state)

        # Benchmark
        times = []
        with torch.inference_mode():
            for _ in range(num_runs):
                start = time.perf_counter()
                probs = compiled_batched(hidden_state)
                times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times) * 1000
        return avg_time, len(lenses), compile_time

    except Exception as e:
        print(f"  ✗ Batching failed: {e}")
        return None, None, None


def main():
    print("=" * 80)
    print("TORCH.COMPILE() LENS INFERENCE BENCHMARK")
    print("=" * 80)

    device = "cuda"
    model_name = "google/gemma-3-4b-pt"
    prompt = "Artificial intelligence can help society by"

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Prompt: \"{prompt}\"")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model loaded")

    # Initialize lens manager
    print("\nInitializing DynamicLensManager...")
    lens_manager = DynamicLensManager(
        lens_pack_id="gemma-3-4b-pt_sumo-wordnet-v3",
        base_layers=[0, 1],
        max_loaded_lenses=500,
        load_threshold=0.3,
        device=device
    )
    print(f"✓ Loaded {len(lens_manager.loaded_lenses)} base lenses")

    # Get hidden state
    print("\nExtracting hidden state...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        hidden_state = outputs.hidden_states[0][-1][:, -1, :].float()

    print(f"✓ Hidden state shape: {hidden_state.shape}")

    # ========================================================================
    # BENCHMARK 1: Uncompiled (baseline)
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. BASELINE: Uncompiled Lens Inference")
    print("=" * 80)

    avg_time, num_lenses = benchmark_uncompiled(lens_manager, hidden_state, num_runs=100)

    print(f"\nResults:")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Per-lens:    {avg_time/num_lenses:.4f}ms")
    print(f"  Num lenses:   {num_lenses}")

    baseline_time = avg_time

    # ========================================================================
    # BENCHMARK 2: Compiled individual lenses
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. COMPILED: Individual Lens Compilation")
    print("=" * 80)

    avg_time, num_lenses, compile_time = benchmark_compiled_individual(
        lens_manager, hidden_state, num_runs=100
    )

    speedup = baseline_time / avg_time if avg_time > 0 else 0

    print(f"\nResults:")
    print(f"  Compilation time: {compile_time:.2f}s")
    print(f"  Average time:     {avg_time:.2f}ms")
    print(f"  Per-lens:        {avg_time/num_lenses:.4f}ms")
    print(f"  Speedup:          {speedup:.2f}x")
    print(f"  Improvement:      {(1 - avg_time/baseline_time)*100:.1f}%")

    # ========================================================================
    # BENCHMARK 3: Batched compilation (if feasible)
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. COMPILED: Batched Lens Inference")
    print("=" * 80)

    avg_time, num_lenses, compile_time = benchmark_compiled_batch(
        lens_manager, hidden_state, num_runs=100
    )

    if avg_time is not None:
        speedup = baseline_time / avg_time if avg_time > 0 else 0

        print(f"\nResults:")
        print(f"  Compilation time: {compile_time:.2f}s")
        print(f"  Average time:     {avg_time:.2f}ms")
        print(f"  Speedup:          {speedup:.2f}x")
        print(f"  Improvement:      {(1 - avg_time/baseline_time)*100:.1f}%")
    else:
        print("\n✗ Batched compilation not feasible for this lens architecture")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nBaseline: {baseline_time:.2f}ms ({baseline_time/num_lenses:.4f}ms per lens)")
    print(f"\nRecommendation will depend on actual speedup achieved.")
    print(f"If speedup < 1.5x: torch.compile() not worth the complexity")
    print(f"If speedup > 2x: torch.compile() is a clear win")

    return 0


if __name__ == '__main__':
    sys.exit(main())
