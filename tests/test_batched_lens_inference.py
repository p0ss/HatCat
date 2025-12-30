#!/usr/bin/env python3
"""
Test that batched lens inference produces identical results to sequential inference.

Validates P2: Verify batched lens accuracy matches sequential.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from typing import Dict

from src.hat.monitoring.lens_manager import SimpleMLP, BatchedLensBank


def create_test_lenses(n_lenses: int, input_dim: int = 2048, device: str = "cuda") -> Dict[str, nn.Module]:
    """Create n test lenses with random weights."""
    lenses = {}
    for i in range(n_lenses):
        lens = SimpleMLP(input_dim).to(device)
        lens.eval()
        lenses[f"concept_{i}"] = lens
    return lenses


def test_batched_matches_sequential():
    """Test that batched inference matches sequential for various lens counts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = 2048

    for n_lenses in [1, 5, 10, 50, 100]:
        print(f"\nTesting with {n_lenses} lenses...")

        # Create lenses
        lenses = create_test_lenses(n_lenses, input_dim, device)

        # Create random hidden state
        hidden_state = torch.randn(1, input_dim, device=device)

        # Sequential inference
        sequential_scores = {}
        sequential_logits = {}
        with torch.inference_mode():
            for key, lens in lenses.items():
                prob, logit = lens(hidden_state, return_logits=True)
                sequential_scores[key] = prob.item()
                sequential_logits[key] = logit.item()

        # Batched inference
        bank = BatchedLensBank(device=device)
        bank.add_lenses(lenses)

        with torch.inference_mode():
            batched_scores, batched_logits = bank(hidden_state, return_logits=True)

        # Compare results
        max_prob_diff = 0.0
        max_logit_diff = 0.0

        for key in lenses.keys():
            prob_diff = abs(sequential_scores[key] - batched_scores[key])
            logit_diff = abs(sequential_logits[key] - batched_logits[key])
            max_prob_diff = max(max_prob_diff, prob_diff)
            max_logit_diff = max(max_logit_diff, logit_diff)

            if prob_diff > 1e-5:
                print(f"  {key}: seq={sequential_scores[key]:.6f}, batch={batched_scores[key]:.6f}, diff={prob_diff:.2e}")

        # Tolerance for floating point differences
        assert max_prob_diff < 1e-5, f"Probability difference too large: {max_prob_diff}"
        assert max_logit_diff < 1e-4, f"Logit difference too large: {max_logit_diff}"

        print(f"  ✓ Max probability diff: {max_prob_diff:.2e}")
        print(f"  ✓ Max logit diff: {max_logit_diff:.2e}")


def test_batched_dtype_handling():
    """Test that batched inference handles different dtypes correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = 2048

    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\nTesting with dtype={dtype}...")

        # Create lenses with specific dtype
        lenses = {}
        for i in range(10):
            lens = SimpleMLP(input_dim, dtype=dtype).to(device)
            lens.eval()
            lenses[f"concept_{i}"] = lens

        # Create hidden state with different dtype (simulating model output)
        hidden_state = torch.randn(1, input_dim, device=device, dtype=torch.bfloat16)

        # Batched inference should handle dtype conversion
        bank = BatchedLensBank(device=device)
        bank.add_lenses(lenses)

        with torch.inference_mode():
            batched_scores = bank(hidden_state)

        assert len(batched_scores) == 10, f"Expected 10 scores, got {len(batched_scores)}"

        for key, prob in batched_scores.items():
            assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"

        print(f"  ✓ All {len(batched_scores)} probabilities valid")


def test_empty_bank():
    """Test that empty bank returns empty dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bank = BatchedLensBank(device=device)
    hidden_state = torch.randn(1, 2048, device=device)

    with torch.inference_mode():
        scores = bank(hidden_state)
        scores_with_logits, logits = bank(hidden_state, return_logits=True)

    assert scores == {}, f"Expected empty dict, got {scores}"
    assert scores_with_logits == {}, f"Expected empty dict, got {scores_with_logits}"
    assert logits == {}, f"Expected empty dict, got {logits}"
    print("\n✓ Empty bank returns empty dicts")


def benchmark_speedup():
    """Benchmark batched vs sequential inference."""
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = 2048
    n_lenses = 50
    n_iterations = 100

    print(f"\nBenchmarking {n_lenses} lenses, {n_iterations} iterations...")

    # Create lenses
    lenses = create_test_lenses(n_lenses, input_dim, device)
    hidden_state = torch.randn(1, input_dim, device=device)

    # Warm up
    with torch.inference_mode():
        for _ in range(10):
            for lens in lenses.values():
                _ = lens(hidden_state)

    if device == "cuda":
        torch.cuda.synchronize()

    # Sequential timing
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_iterations):
            for lens in lenses.values():
                _ = lens(hidden_state)
    if device == "cuda":
        torch.cuda.synchronize()
    sequential_time = (time.perf_counter() - start) / n_iterations * 1000

    # Batched timing
    bank = BatchedLensBank(device=device)
    bank.add_lenses(lenses)

    # Warm up
    with torch.inference_mode():
        for _ in range(10):
            _ = bank(hidden_state)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_iterations):
            _ = bank(hidden_state)
    if device == "cuda":
        torch.cuda.synchronize()
    batched_time = (time.perf_counter() - start) / n_iterations * 1000

    speedup = sequential_time / batched_time

    print(f"  Sequential: {sequential_time:.3f} ms")
    print(f"  Batched:    {batched_time:.3f} ms")
    print(f"  Speedup:    {speedup:.1f}x")

    return speedup


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Batched Lens Inference")
    print("=" * 60)

    test_empty_bank()
    test_batched_matches_sequential()
    test_batched_dtype_handling()
    speedup = benchmark_speedup()

    print("\n" + "=" * 60)
    if speedup >= 5:
        print("✓ All tests passed! Batched inference verified.")
    else:
        print(f"⚠ Tests passed but speedup only {speedup:.1f}x (expected ~10x)")
    print("=" * 60)
