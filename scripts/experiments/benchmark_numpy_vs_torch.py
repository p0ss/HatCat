#!/usr/bin/env python3
"""
Benchmark numpy vs torch for hot-path operations.

Measures actual overhead of:
1. .numpy() conversions
2. np.dot vs torch matmul
3. np.linalg.norm vs tensor.norm()
4. Full steering computation (hooks.py pattern)

Run: python scripts/experiments/benchmark_numpy_vs_torch.py
"""

import time
import numpy as np
import torch
from typing import Dict, Tuple
import statistics

# Simulate realistic dimensions
HIDDEN_DIM = 4096  # Apertus-8B hidden dim
NUM_CONCEPTS = 10  # Typical number of steering concepts
NUM_ITERATIONS = 1000  # Iterations for timing


def benchmark_numpy_conversion():
    """Measure overhead of .numpy() and torch.from_numpy()"""

    # GPU tensor
    x_gpu = torch.randn(HIDDEN_DIM, device='cuda', dtype=torch.bfloat16)
    x_cpu = torch.randn(HIDDEN_DIM, dtype=torch.float32)
    x_np = np.random.randn(HIDDEN_DIM).astype(np.float32)

    results = {}

    # GPU -> CPU -> numpy
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        arr = x_gpu.float().cpu().numpy()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['gpu_to_numpy'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    # numpy -> GPU tensor
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t = torch.from_numpy(x_np).to(device='cuda', dtype=torch.bfloat16)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['numpy_to_gpu'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    # Round-trip: GPU -> numpy -> compute -> GPU
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        arr = x_gpu.float().cpu().numpy()
        arr = arr / (np.linalg.norm(arr) + 1e-8)  # Simple op
        t = torch.from_numpy(arr).to(device='cuda', dtype=torch.bfloat16)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['round_trip_with_op'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    # Same op purely in torch (no conversion)
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t = x_gpu / (x_gpu.norm() + 1e-8)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['torch_native_op'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    return results


def benchmark_dot_product():
    """Measure np.dot vs torch matmul"""

    # Vectors
    a_np = np.random.randn(HIDDEN_DIM).astype(np.float32)
    b_np = np.random.randn(HIDDEN_DIM).astype(np.float32)
    a_torch_cpu = torch.from_numpy(a_np)
    b_torch_cpu = torch.from_numpy(b_np)
    a_torch_gpu = a_torch_cpu.to('cuda', dtype=torch.bfloat16)
    b_torch_gpu = b_torch_cpu.to('cuda', dtype=torch.bfloat16)

    results = {}

    # numpy dot (CPU)
    times = []
    for _ in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        _ = np.dot(a_np, b_np)
        times.append(time.perf_counter() - t0)
    results['np_dot_cpu'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    # torch dot (CPU)
    times = []
    for _ in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        _ = torch.dot(a_torch_cpu, b_torch_cpu)
        times.append(time.perf_counter() - t0)
    results['torch_dot_cpu'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    # torch dot (GPU)
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = torch.dot(a_torch_gpu, b_torch_gpu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['torch_dot_gpu'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    # torch @ operator (GPU)
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = a_torch_gpu @ b_torch_gpu
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['torch_matmul_gpu'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    return results


def benchmark_norm():
    """Measure np.linalg.norm vs tensor.norm()"""

    x_np = np.random.randn(HIDDEN_DIM).astype(np.float32)
    x_torch_cpu = torch.from_numpy(x_np)
    x_torch_gpu = x_torch_cpu.to('cuda', dtype=torch.bfloat16)

    results = {}

    # numpy norm
    times = []
    for _ in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        _ = np.linalg.norm(x_np)
        times.append(time.perf_counter() - t0)
    results['np_norm'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    # torch norm CPU
    times = []
    for _ in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        _ = x_torch_cpu.norm()
        times.append(time.perf_counter() - t0)
    results['torch_norm_cpu'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    # torch norm GPU
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = x_torch_gpu.norm()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['torch_norm_gpu'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    return results


def benchmark_steering_pattern():
    """
    Benchmark the actual steering computation pattern from hooks.py:

    Current (numpy):
        proj_coef = np.dot(target_vector, reference_vector)
        contrastive = target_vector - proj_coef * reference_vector
        magnitude = np.linalg.norm(contrastive)
        contrastive = contrastive / magnitude

    Proposed (torch):
        proj_coef = target @ reference
        contrastive = target - proj_coef * reference
        contrastive = contrastive / contrastive.norm()
    """

    # Setup - multiple concepts like real steering
    target_np = [np.random.randn(HIDDEN_DIM).astype(np.float32) for _ in range(NUM_CONCEPTS)]
    ref_np = [np.random.randn(HIDDEN_DIM).astype(np.float32) for _ in range(NUM_CONCEPTS)]

    target_gpu = [torch.from_numpy(t).to('cuda', dtype=torch.bfloat16) for t in target_np]
    ref_gpu = [torch.from_numpy(r).to('cuda', dtype=torch.bfloat16) for r in ref_np]

    results = {}

    # Current pattern: numpy on CPU
    times = []
    for _ in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        for t, r in zip(target_np, ref_np):
            proj_coef = np.dot(t, r)
            contrastive = t - proj_coef * r
            magnitude = np.linalg.norm(contrastive)
            if magnitude > 1e-8:
                contrastive = contrastive / magnitude
        times.append(time.perf_counter() - t0)
    results['numpy_steering'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
        'per_concept_us': statistics.mean(times) * 1e6 / NUM_CONCEPTS,
    }

    # Proposed pattern: torch on GPU
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t, r in zip(target_gpu, ref_gpu):
            proj_coef = t @ r
            contrastive = t - proj_coef * r
            contrastive = contrastive / (contrastive.norm() + 1e-8)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['torch_steering_gpu'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
        'per_concept_us': statistics.mean(times) * 1e6 / NUM_CONCEPTS,
    }

    # Batched torch (stack all concepts, single matmul)
    target_stacked = torch.stack(target_gpu)  # [NUM_CONCEPTS, HIDDEN_DIM]
    ref_stacked = torch.stack(ref_gpu)

    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # Batched dot products
        proj_coefs = (target_stacked * ref_stacked).sum(dim=1, keepdim=True)
        contrastive = target_stacked - proj_coefs * ref_stacked
        norms = contrastive.norm(dim=1, keepdim=True)
        contrastive = contrastive / (norms + 1e-8)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['torch_steering_batched'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
        'per_concept_us': statistics.mean(times) * 1e6 / NUM_CONCEPTS,
    }

    return results


def benchmark_full_hook_simulation():
    """
    Simulate full steering hook overhead including:
    - Hidden state extraction
    - Vector application
    - Result injection back
    """

    # Simulate hidden states from model (batch=1, seq_len=1, hidden_dim)
    hidden_states = torch.randn(1, 1, HIDDEN_DIM, device='cuda', dtype=torch.bfloat16)

    # Steering vectors (as currently stored - numpy)
    vectors_np = {f"concept_{i}": np.random.randn(HIDDEN_DIM).astype(np.float32)
                  for i in range(NUM_CONCEPTS)}

    # Steering vectors (proposed - torch on GPU)
    vectors_gpu = {k: torch.from_numpy(v).to('cuda', dtype=torch.bfloat16)
                   for k, v in vectors_np.items()}

    results = {}
    strength = 0.3

    # Current: numpy vectors, convert each time
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        h = hidden_states.clone()
        for name, vec in vectors_np.items():
            # Convert numpy to torch (current overhead)
            vec_tensor = torch.from_numpy(vec).to(device=h.device, dtype=h.dtype)
            h = h + strength * vec_tensor

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['current_numpy_vectors'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    # Proposed: torch vectors, no conversion
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        h = hidden_states.clone()
        for name, vec in vectors_gpu.items():
            h = h + strength * vec

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['proposed_torch_vectors'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    # Batched: single operation
    all_vecs = torch.stack(list(vectors_gpu.values()))  # [N, hidden_dim]

    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        h = hidden_states.clone()
        combined = all_vecs.sum(dim=0)  # Sum all vectors
        h = h + strength * combined

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['batched_torch'] = {
        'mean_us': statistics.mean(times) * 1e6,
        'std_us': statistics.stdev(times) * 1e6,
    }

    return results


def main():
    print("=" * 70)
    print("NUMPY VS TORCH PERFORMANCE BENCHMARK")
    print(f"Hidden dim: {HIDDEN_DIM}, Concepts: {NUM_CONCEPTS}, Iterations: {NUM_ITERATIONS}")
    print("=" * 70)

    # Warm up GPU
    _ = torch.randn(1000, 1000, device='cuda') @ torch.randn(1000, 1000, device='cuda')
    torch.cuda.synchronize()

    print("\n1. CONVERSION OVERHEAD (.numpy() round-trips)")
    print("-" * 50)
    results = benchmark_numpy_conversion()
    for name, data in results.items():
        print(f"  {name:30s}: {data['mean_us']:8.2f} ± {data['std_us']:5.2f} µs")

    speedup = results['round_trip_with_op']['mean_us'] / results['torch_native_op']['mean_us']
    print(f"\n  Speedup (native vs round-trip): {speedup:.1f}x")

    print("\n2. DOT PRODUCT")
    print("-" * 50)
    results = benchmark_dot_product()
    for name, data in results.items():
        print(f"  {name:30s}: {data['mean_us']:8.2f} ± {data['std_us']:5.2f} µs")

    print("\n3. NORM")
    print("-" * 50)
    results = benchmark_norm()
    for name, data in results.items():
        print(f"  {name:30s}: {data['mean_us']:8.2f} ± {data['std_us']:5.2f} µs")

    print("\n4. STEERING PATTERN (contrastive projection)")
    print("-" * 50)
    results = benchmark_steering_pattern()
    for name, data in results.items():
        extra = f" ({data['per_concept_us']:.2f} µs/concept)" if 'per_concept_us' in data else ""
        print(f"  {name:30s}: {data['mean_us']:8.2f} ± {data['std_us']:5.2f} µs{extra}")

    speedup_loop = results['numpy_steering']['mean_us'] / results['torch_steering_gpu']['mean_us']
    speedup_batch = results['numpy_steering']['mean_us'] / results['torch_steering_batched']['mean_us']
    print(f"\n  Speedup (torch loop vs numpy): {speedup_loop:.1f}x")
    print(f"  Speedup (batched vs numpy):    {speedup_batch:.1f}x")

    print("\n5. FULL HOOK SIMULATION (hidden state modification)")
    print("-" * 50)
    results = benchmark_full_hook_simulation()
    for name, data in results.items():
        print(f"  {name:30s}: {data['mean_us']:8.2f} ± {data['std_us']:5.2f} µs")

    speedup = results['current_numpy_vectors']['mean_us'] / results['proposed_torch_vectors']['mean_us']
    speedup_batch = results['current_numpy_vectors']['mean_us'] / results['batched_torch']['mean_us']
    print(f"\n  Speedup (torch vs numpy):   {speedup:.1f}x")
    print(f"  Speedup (batched vs numpy): {speedup_batch:.1f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Per-token overhead estimates")
    print("=" * 70)

    # Typical steering uses ~10 concepts with contrastive projection
    numpy_per_token = results['current_numpy_vectors']['mean_us']
    torch_per_token = results['proposed_torch_vectors']['mean_us']
    batch_per_token = results['batched_torch']['mean_us']

    print(f"\n  Current (numpy):     {numpy_per_token:.1f} µs/token")
    print(f"  Proposed (torch):    {torch_per_token:.1f} µs/token  ({numpy_per_token/torch_per_token:.1f}x faster)")
    print(f"  Proposed (batched):  {batch_per_token:.1f} µs/token  ({numpy_per_token/batch_per_token:.1f}x faster)")

    # At 50 tokens/sec generation, how much overhead?
    tokens_per_sec = 50
    print(f"\n  At {tokens_per_sec} tokens/sec generation:")
    print(f"    Current overhead:  {numpy_per_token * tokens_per_sec / 1000:.2f} ms/sec ({numpy_per_token * tokens_per_sec / 10000:.1f}% of time)")
    print(f"    Proposed overhead: {torch_per_token * tokens_per_sec / 1000:.2f} ms/sec ({torch_per_token * tokens_per_sec / 10000:.1f}% of time)")


if __name__ == "__main__":
    main()
