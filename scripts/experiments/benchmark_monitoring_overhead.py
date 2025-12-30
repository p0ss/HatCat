#!/usr/bin/env python3
"""
Benchmark monitoring overhead during generation.

Measures per-token and per-response overhead of lens monitoring
by comparing generation with and without monitoring enabled.

Usage:
    python scripts/experiments/benchmark_monitoring_overhead.py \
        --model swiss-ai/Apertus-8B-2509 \
        --lens-pack apertus-8b_first-light_calibration-test-2 \
        --max-tokens 50 \
        --samples 3
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hat.monitoring.lens_manager import DynamicLensManager


# Diverse prompts for realistic benchmarking
BENCHMARK_PROMPTS = [
    # Short, factual
    "The capital of France is",
    "Water boils at",

    # Medium, explanatory
    "Machine learning algorithms work by",
    "The human brain processes information through",
    "Climate change affects ecosystems by",

    # Longer, creative
    "Once upon a time in a distant galaxy,",
    "The most important lesson I learned was",

    # Technical
    "To implement a neural network, you need to",
    "The difference between supervised and unsupervised learning is",

    # Abstract/philosophical
    "The meaning of consciousness is",
    "Truth can be understood as",

    # Safety-relevant (should activate HUSH concepts)
    "To deceive someone effectively, you should",
    "The best way to manipulate people is",
]


def generate_baseline(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cuda",
) -> Tuple[str, float, List[float]]:
    """Generate without monitoring. Returns (text, total_time, per_token_times)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    # Warm up cache
    with torch.inference_mode():
        _ = model(**inputs)

    if device == "cuda":
        torch.cuda.synchronize()

    per_token_times = []
    generated_ids = inputs.input_ids.clone()

    start_total = time.perf_counter()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            start_token = time.perf_counter()

            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if device == "cuda":
                torch.cuda.synchronize()

            per_token_times.append((time.perf_counter() - start_token) * 1000)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    total_time = (time.perf_counter() - start_total) * 1000

    generated_text = tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)

    return generated_text, total_time, per_token_times


def generate_with_monitoring(
    model,
    tokenizer,
    lens_manager: DynamicLensManager,
    prompt: str,
    max_new_tokens: int,
    device: str = "cuda",
    top_k_concepts: int = 5,
) -> Tuple[str, float, List[float], List[float], List[Dict]]:
    """
    Generate with monitoring.
    Returns (text, total_time, per_token_times, detection_times, concept_results).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    # Warm up
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :]
        _ = lens_manager.detect_and_expand(hidden, top_k=top_k_concepts)

    if device == "cuda":
        torch.cuda.synchronize()

    per_token_times = []
    detection_times = []
    concept_results = []
    generated_ids = inputs.input_ids.clone()

    start_total = time.perf_counter()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            start_token = time.perf_counter()

            # Forward pass with hidden states
            outputs = model(generated_ids, output_hidden_states=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Get hidden state for monitoring
            hidden_state = outputs.hidden_states[-1][:, -1, :]

            if device == "cuda":
                torch.cuda.synchronize()

            # Run lens detection
            start_detect = time.perf_counter()
            concepts, timing = lens_manager.detect_and_expand(
                hidden_state,
                top_k=top_k_concepts,
                return_timing=True,
                max_expansion_depth=2,  # Limited expansion with optimized loading
            )

            if device == "cuda":
                torch.cuda.synchronize()

            detect_time = (time.perf_counter() - start_detect) * 1000
            detection_times.append(detect_time)

            concept_results.append({
                "token": tokenizer.decode([next_token.item()]),
                "concepts": [(c[0], c[1]) for c in concepts[:3]],  # Top 3
                "timing": timing
            })

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if device == "cuda":
                torch.cuda.synchronize()

            per_token_times.append((time.perf_counter() - start_token) * 1000)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    total_time = (time.perf_counter() - start_total) * 1000

    generated_text = tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)

    return generated_text, total_time, per_token_times, detection_times, concept_results


def run_benchmark(
    model_name: str,
    lens_pack: str,
    max_tokens: int = 50,
    samples_per_prompt: int = 1,
    prompts: List[str] = None,
    device: str = "cuda",
    output_dir: str = None,
):
    """Run the full benchmark."""

    if prompts is None:
        prompts = BENCHMARK_PROMPTS

    print("=" * 70)
    print("MONITORING OVERHEAD BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Lens pack: {lens_pack}")
    print(f"Max tokens: {max_tokens}")
    print(f"Prompts: {len(prompts)}")
    print(f"Samples per prompt: {samples_per_prompt}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    # Load lens manager
    print("Loading lens manager...")
    lens_pack_path = Path("lens_packs") / lens_pack
    concept_pack_info = json.load(open(lens_pack_path / "pack_info.json"))
    source_pack = concept_pack_info.get("source_pack", "first-light")
    hierarchy_path = Path("concept_packs") / source_pack / "hierarchy"

    lens_manager = DynamicLensManager(
        layers_data_dir=hierarchy_path,
        lenses_dir=lens_pack_path,
        device=device,
        base_layers=[0, 1, 2],
        max_loaded_lenses=500,
        keep_top_k=50,
        normalize_hidden_states=True,
    )
    # Base layers loaded automatically during init

    print(f"Loaded {len(lens_manager.loaded_lenses)} base lenses")
    print(f"Batched inference: {lens_manager._use_batched_inference}")

    # Pre-load entire pack to RAM (tepid cache) for fast expansion
    print("\nPre-loading pack to RAM...")
    preload_stats = lens_manager.preload_pack_to_ram()
    print(f"  Loaded {preload_stats['concepts']} concepts to RAM ({preload_stats['ram_mb']:.1f} MB in {preload_stats['elapsed_s']:.1f}s)")

    # Benchmark results
    baseline_results = []
    monitored_results = []

    total_prompts = len(prompts) * samples_per_prompt

    print(f"\n{'='*70}")
    print("RUNNING BASELINE (no monitoring)...")
    print("=" * 70)

    for i, prompt in enumerate(prompts):
        for s in range(samples_per_prompt):
            idx = i * samples_per_prompt + s + 1
            print(f"\n[{idx}/{total_prompts}] {prompt[:50]}...")

            text, total_time, token_times = generate_baseline(
                model, tokenizer, prompt, max_tokens, device
            )

            baseline_results.append({
                "prompt": prompt,
                "generated": text,
                "total_ms": total_time,
                "tokens": len(token_times),
                "per_token_ms": token_times,
                "avg_per_token_ms": sum(token_times) / len(token_times) if token_times else 0,
            })

            print(f"  Generated {len(token_times)} tokens in {total_time:.1f}ms")
            print(f"  Avg per token: {baseline_results[-1]['avg_per_token_ms']:.2f}ms")

    print(f"\n{'='*70}")
    print("RUNNING WITH MONITORING...")
    print("=" * 70)

    for i, prompt in enumerate(prompts):
        for s in range(samples_per_prompt):
            idx = i * samples_per_prompt + s + 1
            print(f"\n[{idx}/{total_prompts}] {prompt[:50]}...")

            text, total_time, token_times, detect_times, concepts = generate_with_monitoring(
                model, tokenizer, lens_manager, prompt, max_tokens, device
            )

            monitored_results.append({
                "prompt": prompt,
                "generated": text,
                "total_ms": total_time,
                "tokens": len(token_times),
                "per_token_ms": token_times,
                "detection_ms": detect_times,
                "avg_per_token_ms": sum(token_times) / len(token_times) if token_times else 0,
                "avg_detection_ms": sum(detect_times) / len(detect_times) if detect_times else 0,
                "concepts": concepts,
            })

            print(f"  Generated {len(token_times)} tokens in {total_time:.1f}ms")
            print(f"  Avg per token: {monitored_results[-1]['avg_per_token_ms']:.2f}ms")
            print(f"  Avg detection: {monitored_results[-1]['avg_detection_ms']:.2f}ms")

            # Reset to base lenses between prompts (clear cache to avoid accumulation)
            lens_manager.reset_to_base(keep_warm_cache=False)

    # Compute summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    baseline_avg_token = sum(r["avg_per_token_ms"] for r in baseline_results) / len(baseline_results)
    baseline_avg_total = sum(r["total_ms"] for r in baseline_results) / len(baseline_results)
    baseline_total_tokens = sum(r["tokens"] for r in baseline_results)

    monitored_avg_token = sum(r["avg_per_token_ms"] for r in monitored_results) / len(monitored_results)
    monitored_avg_total = sum(r["total_ms"] for r in monitored_results) / len(monitored_results)
    monitored_avg_detect = sum(r["avg_detection_ms"] for r in monitored_results) / len(monitored_results)
    monitored_total_tokens = sum(r["tokens"] for r in monitored_results)

    overhead_per_token = monitored_avg_token - baseline_avg_token
    overhead_percent = (overhead_per_token / baseline_avg_token) * 100 if baseline_avg_token > 0 else 0

    print(f"\nBASELINE (no monitoring):")
    print(f"  Total tokens generated: {baseline_total_tokens}")
    print(f"  Avg per-token time:     {baseline_avg_token:.2f} ms")
    print(f"  Avg response time:      {baseline_avg_total:.1f} ms")

    print(f"\nWITH MONITORING:")
    print(f"  Total tokens generated: {monitored_total_tokens}")
    print(f"  Avg per-token time:     {monitored_avg_token:.2f} ms")
    print(f"  Avg detection time:     {monitored_avg_detect:.2f} ms")
    print(f"  Avg response time:      {monitored_avg_total:.1f} ms")

    print(f"\nOVERHEAD:")
    print(f"  Per-token overhead:     {overhead_per_token:.2f} ms ({overhead_percent:.1f}%)")
    print(f"  Detection overhead:     {monitored_avg_detect:.2f} ms")
    print(f"  Lenses loaded:          {len(lens_manager.loaded_lenses)}")

    # Detailed detection timing breakdown
    all_detection_times = []
    for r in monitored_results:
        all_detection_times.extend(r["detection_ms"])

    if all_detection_times:
        all_detection_times.sort()
        p50 = all_detection_times[len(all_detection_times) // 2]
        p95 = all_detection_times[int(len(all_detection_times) * 0.95)]
        p99 = all_detection_times[int(len(all_detection_times) * 0.99)]

        print(f"\nDETECTION LATENCY DISTRIBUTION:")
        print(f"  Min:  {min(all_detection_times):.2f} ms")
        print(f"  P50:  {p50:.2f} ms")
        print(f"  P95:  {p95:.2f} ms")
        print(f"  P99:  {p99:.2f} ms")
        print(f"  Max:  {max(all_detection_times):.2f} ms")

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "metadata": {
                "model": model_name,
                "lens_pack": lens_pack,
                "max_tokens": max_tokens,
                "samples_per_prompt": samples_per_prompt,
                "num_prompts": len(prompts),
                "timestamp": datetime.now().isoformat(),
                "lenses_loaded": len(lens_manager.loaded_lenses),
                "batched_inference": lens_manager._use_batched_inference,
            },
            "summary": {
                "baseline_avg_per_token_ms": baseline_avg_token,
                "baseline_avg_response_ms": baseline_avg_total,
                "monitored_avg_per_token_ms": monitored_avg_token,
                "monitored_avg_detection_ms": monitored_avg_detect,
                "monitored_avg_response_ms": monitored_avg_total,
                "overhead_per_token_ms": overhead_per_token,
                "overhead_percent": overhead_percent,
                "detection_p50_ms": p50 if all_detection_times else None,
                "detection_p95_ms": p95 if all_detection_times else None,
            },
            "baseline_results": baseline_results,
            "monitored_results": monitored_results,
        }

        results_file = output_path / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark monitoring overhead")
    parser.add_argument("--model", type=str, default="swiss-ai/Apertus-8B-2509")
    parser.add_argument("--lens-pack", type=str, default="apertus-8b_first-light_calibration-test-2")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--output-dir", type=str, default="results/monitoring_benchmark")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        lens_pack=args.lens_pack,
        max_tokens=args.max_tokens,
        samples_per_prompt=args.samples,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
