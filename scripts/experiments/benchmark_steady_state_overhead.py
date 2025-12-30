#!/usr/bin/env python3
"""
Benchmark steady-state monitoring overhead.

Measures per-token overhead AFTER the lens hierarchy is warmed up,
to separate "first-time loading" from "steady-state inference".

This is the realistic overhead for production use after warm-up.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hat.monitoring.lens_manager import DynamicLensManager


# Warm-up prompts to load diverse concept branches
WARMUP_PROMPTS = [
    "The computer algorithm processes data",
    "The philosophical meaning of truth",
    "A car drives down the highway",
    "The company employs many workers",
    "Mathematics involves abstract reasoning",
    "The ocean contains marine life",
    "Social relationships form communities",
    "Neural networks learn patterns",
    "Physical objects have mass",
    "Organizations make decisions",
]

# Test prompts for steady-state measurement
TEST_PROMPTS = [
    "Machine learning algorithms work by",
    "The human brain processes information through",
    "To implement a system, you need to",
    "The meaning of consciousness is",
    "Technology has changed how we",
]


def warm_up_hierarchy(
    model,
    tokenizer,
    lens_manager: DynamicLensManager,
    prompts: List[str],
    tokens_per_prompt: int = 20,
    device: str = "cuda",
):
    """Warm up the lens hierarchy by processing diverse prompts."""
    print(f"\nWarming up hierarchy with {len(prompts)} prompts...")

    total_lenses_before = len(lens_manager.loaded_lenses)

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.inference_mode():
            for _ in range(tokens_per_prompt):
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][:, -1, :]
                lens_manager.detect_and_expand(hidden, top_k=10)

                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                inputs = {"input_ids": torch.cat([inputs["input_ids"], next_token], dim=-1)}

        lenses_now = len(lens_manager.loaded_lenses)
        print(f"  [{i+1}/{len(prompts)}] {prompt[:40]}... ({lenses_now} lenses)")

    total_lenses_after = len(lens_manager.loaded_lenses)
    print(f"\n✓ Warm-up complete: {total_lenses_before} → {total_lenses_after} lenses loaded")

    return total_lenses_after


def benchmark_steady_state(
    model,
    tokenizer,
    lens_manager: DynamicLensManager,
    prompts: List[str],
    max_tokens: int = 30,
    samples: int = 3,
    device: str = "cuda",
):
    """Benchmark steady-state detection overhead."""
    print(f"\nBenchmarking steady-state ({len(prompts)} prompts × {samples} samples)...")

    all_detection_times = []
    all_token_times = []
    all_initial_times = []
    all_child_times = []

    for prompt in prompts:
        for s in range(samples):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Warm cache for this prompt
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][:, -1, :]
                lens_manager.detect_and_expand(hidden, top_k=10)

            if device == "cuda":
                torch.cuda.synchronize()

            # Measure generation
            detection_times = []
            token_times = []
            initial_times = []
            child_times = []

            with torch.inference_mode():
                for _ in range(max_tokens):
                    start_token = time.perf_counter()

                    outputs = model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1][:, -1, :]

                    if device == "cuda":
                        torch.cuda.synchronize()

                    start_detect = time.perf_counter()
                    concepts, timing = lens_manager.detect_and_expand(
                        hidden, top_k=10, return_timing=True
                    )

                    if device == "cuda":
                        torch.cuda.synchronize()

                    detect_time = (time.perf_counter() - start_detect) * 1000
                    detection_times.append(detect_time)
                    initial_times.append(timing.get('initial_detection', 0))
                    child_times.append(timing.get('child_loading', 0))

                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                    inputs = {"input_ids": torch.cat([inputs["input_ids"], next_token], dim=-1)}

                    if device == "cuda":
                        torch.cuda.synchronize()

                    token_times.append((time.perf_counter() - start_token) * 1000)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

            all_detection_times.extend(detection_times)
            all_token_times.extend(token_times)
            all_initial_times.extend(initial_times)
            all_child_times.extend(child_times)

            avg_detect = sum(detection_times) / len(detection_times)
            avg_initial = sum(initial_times) / len(initial_times)
            print(f"  {prompt[:35]}... detect:{avg_detect:.2f}ms (init:{avg_initial:.2f}ms)")

    return {
        "detection_times": all_detection_times,
        "token_times": all_token_times,
        "initial_times": all_initial_times,
        "child_times": all_child_times,
    }


def benchmark_baseline(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 30,
    samples: int = 3,
    device: str = "cuda",
):
    """Benchmark baseline generation without monitoring."""
    print(f"\nBenchmarking baseline (no monitoring)...")

    all_token_times = []

    for prompt in prompts:
        for s in range(samples):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Warm cache
            with torch.inference_mode():
                _ = model(**inputs)

            if device == "cuda":
                torch.cuda.synchronize()

            token_times = []

            with torch.inference_mode():
                for _ in range(max_tokens):
                    start = time.perf_counter()

                    outputs = model(**inputs)
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                    inputs = {"input_ids": torch.cat([inputs["input_ids"], next_token], dim=-1)}

                    if device == "cuda":
                        torch.cuda.synchronize()

                    token_times.append((time.perf_counter() - start) * 1000)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

            all_token_times.extend(token_times)
            avg = sum(token_times) / len(token_times)
            print(f"  {prompt[:35]}... avg:{avg:.2f}ms")

    return all_token_times


def main():
    parser = argparse.ArgumentParser(description="Benchmark steady-state monitoring overhead")
    parser.add_argument("--model", type=str, default="swiss-ai/Apertus-8B-2509")
    parser.add_argument("--lens-pack", type=str, default="apertus-8b_first-light_calibration-test-2")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="results/steady_state_benchmark")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("=" * 70)
    print("STEADY-STATE MONITORING OVERHEAD BENCHMARK")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Lens pack: {args.lens_pack}")
    print(f"Warm-up: {len(WARMUP_PROMPTS)} prompts × {args.warmup_tokens} tokens")
    print(f"Test: {len(TEST_PROMPTS)} prompts × {args.samples} samples × {args.max_tokens} tokens")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()

    # Load lens manager
    print("Loading lens manager...")
    lens_pack_path = Path("lens_packs") / args.lens_pack
    concept_pack_info = json.load(open(lens_pack_path / "pack_info.json"))
    source_pack = concept_pack_info.get("source_pack", "first-light")
    hierarchy_path = Path("concept_packs") / source_pack / "hierarchy"

    lens_manager = DynamicLensManager(
        layers_data_dir=hierarchy_path,
        lenses_dir=lens_pack_path,
        device=args.device,
        base_layers=[0, 1],
        max_loaded_lenses=500,
        keep_top_k=50,
        normalize_hidden_states=True,
    )

    print(f"Initial lenses: {len(lens_manager.loaded_lenses)}")
    print(f"Batched inference: {lens_manager._use_batched_inference}")

    # Warm up hierarchy
    lenses_after_warmup = warm_up_hierarchy(
        model, tokenizer, lens_manager, WARMUP_PROMPTS,
        tokens_per_prompt=args.warmup_tokens, device=args.device
    )

    # Benchmark baseline
    baseline_times = benchmark_baseline(
        model, tokenizer, TEST_PROMPTS,
        max_tokens=args.max_tokens, samples=args.samples, device=args.device
    )

    # Benchmark with monitoring
    monitored = benchmark_steady_state(
        model, tokenizer, lens_manager, TEST_PROMPTS,
        max_tokens=args.max_tokens, samples=args.samples, device=args.device
    )

    # Results
    print("\n" + "=" * 70)
    print("STEADY-STATE RESULTS")
    print("=" * 70)

    baseline_avg = sum(baseline_times) / len(baseline_times)
    monitored_avg = sum(monitored["token_times"]) / len(monitored["token_times"])
    detect_avg = sum(monitored["detection_times"]) / len(monitored["detection_times"])
    initial_avg = sum(monitored["initial_times"]) / len(monitored["initial_times"])
    child_avg = sum(monitored["child_times"]) / len(monitored["child_times"])

    overhead = monitored_avg - baseline_avg
    overhead_pct = (overhead / baseline_avg) * 100

    # Percentiles
    det_sorted = sorted(monitored["detection_times"])
    p50 = det_sorted[len(det_sorted) // 2]
    p95 = det_sorted[int(len(det_sorted) * 0.95)]
    p99 = det_sorted[int(len(det_sorted) * 0.99)]

    print(f"\nLenses loaded after warm-up: {lenses_after_warmup}")

    print(f"\nBASELINE (no monitoring):")
    print(f"  Avg per-token: {baseline_avg:.2f} ms")

    print(f"\nWITH MONITORING (steady-state):")
    print(f"  Avg per-token:     {monitored_avg:.2f} ms")
    print(f"  Avg detection:     {detect_avg:.2f} ms")
    print(f"    - Initial (batched lens): {initial_avg:.2f} ms")
    print(f"    - Child loading:          {child_avg:.2f} ms")

    print(f"\nOVERHEAD:")
    print(f"  Per-token overhead: {overhead:.2f} ms ({overhead_pct:.1f}%)")

    print(f"\nDETECTION LATENCY DISTRIBUTION:")
    print(f"  Min:  {min(monitored['detection_times']):.2f} ms")
    print(f"  P50:  {p50:.2f} ms")
    print(f"  P95:  {p95:.2f} ms")
    print(f"  P99:  {p99:.2f} ms")
    print(f"  Max:  {max(monitored['detection_times']):.2f} ms")

    # Save results
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "metadata": {
                "model": args.model,
                "lens_pack": args.lens_pack,
                "lenses_after_warmup": lenses_after_warmup,
                "batched_inference": lens_manager._use_batched_inference,
                "timestamp": datetime.now().isoformat(),
            },
            "summary": {
                "baseline_avg_ms": baseline_avg,
                "monitored_avg_ms": monitored_avg,
                "detection_avg_ms": detect_avg,
                "initial_detection_avg_ms": initial_avg,
                "child_loading_avg_ms": child_avg,
                "overhead_ms": overhead,
                "overhead_percent": overhead_pct,
                "detection_p50_ms": p50,
                "detection_p95_ms": p95,
                "detection_p99_ms": p99,
            },
            "baseline_times": baseline_times,
            "monitored": monitored,
        }

        with open(output_path / "steady_state_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
