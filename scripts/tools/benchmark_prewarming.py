#!/usr/bin/env python3
"""
Benchmark pre-warming strategy: load child lenses during prompt processing.

Tests:
1. Baseline: Cold start (load children during generation)
2. Pre-warming: Load children during prompt eval, reuse during generation

Hypothesis: 75% concept overlap means we can pre-load most children with zero
generation latency cost.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hat.monitoring.lens_manager import DynamicLensManager


def benchmark_baseline(model, tokenizer, lens_manager, prompt: str, device: str = "cuda", n_tokens: int = 10):
    """Baseline: cold start, load children during generation."""
    model.eval()

    # Reset lens manager to base layers only
    lens_manager.reset_to_base()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    timings = {
        'generation': 0,
        'lens_detection': 0,
        'child_loading': 0,
    }

    children_loaded = 0

    with torch.inference_mode():
        start_gen = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=True,
            temperature=0.8,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        timings['generation'] = time.perf_counter() - start_gen

        # Process hidden states
        for step_idx, step_states in enumerate(outputs.hidden_states):
            last_layer = step_states[-1]
            hidden_state = last_layer[:, -1, :].float()

            start_detect = time.perf_counter()
            detected, timing_info = lens_manager.detect_and_expand(
                hidden_state,
                top_k=10,
                return_timing=True
            )
            timings['lens_detection'] += time.perf_counter() - start_detect
            timings['child_loading'] += timing_info.get('child_loading', 0)
            children_loaded += timing_info.get('num_children_loaded', 0)

    return {
        'generation_ms': timings['generation'] * 1000,
        'lens_detection_ms': timings['lens_detection'] * 1000,
        'child_loading_ms': timings['child_loading'],
        'children_loaded': children_loaded,
        'per_token_ms': timings['lens_detection'] * 1000 / n_tokens,
        'child_loading_per_token_ms': timings['child_loading'] / n_tokens,
    }


def benchmark_prewarming(model, tokenizer, lens_manager, prompt: str, device: str = "cuda", n_tokens: int = 10):
    """Pre-warming: load children during prompt eval, reuse during generation."""
    model.eval()

    # Reset lens manager to base layers only
    lens_manager.reset_to_base()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    timings = {
        'prompt_eval': 0,
        'prompt_lens_detection': 0,
        'prompt_child_loading': 0,
        'generation': 0,
        'generation_lens_detection': 0,
        'generation_child_loading': 0,
    }

    prompt_children = 0
    generation_children = 0

    with torch.inference_mode():
        # ====================================================================
        # PHASE 1: Prompt processing with pre-warming
        # ====================================================================
        start_prompt = time.perf_counter()

        # Get prompt hidden states
        prompt_outputs = model(**inputs, output_hidden_states=True)
        prompt_hidden = prompt_outputs.hidden_states[-1][:, -1, :].float()

        # Run lens detection on prompt to pre-load children
        start_detect = time.perf_counter()
        prompt_detected, timing_info = lens_manager.detect_and_expand(
            prompt_hidden,
            top_k=10,
            return_timing=True
        )
        timings['prompt_lens_detection'] = time.perf_counter() - start_detect
        timings['prompt_child_loading'] = timing_info.get('child_loading', 0)
        prompt_children = timing_info.get('num_children_loaded', 0)

        timings['prompt_eval'] = time.perf_counter() - start_prompt

        # ====================================================================
        # PHASE 2: Generation with warm cache
        # ====================================================================
        start_gen = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=True,
            temperature=0.8,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        timings['generation'] = time.perf_counter() - start_gen

        # Process hidden states
        for step_idx, step_states in enumerate(outputs.hidden_states):
            last_layer = step_states[-1]
            hidden_state = last_layer[:, -1, :].float()

            start_detect = time.perf_counter()
            detected, timing_info = lens_manager.detect_and_expand(
                hidden_state,
                top_k=10,
                return_timing=True
            )
            timings['generation_lens_detection'] += time.perf_counter() - start_detect
            timings['generation_child_loading'] += timing_info.get('child_loading', 0)
            generation_children += timing_info.get('num_children_loaded', 0)

    return {
        'prompt_eval_ms': timings['prompt_eval'] * 1000,
        'prompt_lens_detection_ms': timings['prompt_lens_detection'] * 1000,
        'prompt_child_loading_ms': timings['prompt_child_loading'],
        'prompt_children': prompt_children,
        'generation_ms': timings['generation'] * 1000,
        'generation_lens_detection_ms': timings['generation_lens_detection'] * 1000,
        'generation_child_loading_ms': timings['generation_child_loading'],
        'generation_children': generation_children,
        'per_token_ms': timings['generation_lens_detection'] * 1000 / n_tokens,
        'child_loading_per_token_ms': timings['generation_child_loading'] / n_tokens,
    }


def main():
    print("=" * 80)
    print("PRE-WARMING STRATEGY BENCHMARK")
    print("=" * 80)

    device = "cuda"
    model_name = "google/gemma-3-4b-pt"
    prompt = "Artificial intelligence can help society by"
    n_tokens = 10

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Generation tokens: {n_tokens}")

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

    # ========================================================================
    # BENCHMARK 1: Baseline (cold start)
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. BASELINE: Cold Start (load children during generation)")
    print("=" * 80)

    baseline = benchmark_baseline(model, tokenizer, lens_manager, prompt, device, n_tokens)

    print(f"\nResults:")
    print(f"  Generation time:         {baseline['generation_ms']:.2f}ms")
    print(f"  Lens detection:         {baseline['lens_detection_ms']:.2f}ms")
    print(f"    └─ Per token:          {baseline['per_token_ms']:.2f}ms/token")
    print(f"  Child loading (disk I/O):{baseline['child_loading_ms']:.2f}ms")
    print(f"    └─ Per token:          {baseline['child_loading_per_token_ms']:.2f}ms/token")
    print(f"  Total children loaded:   {baseline['children_loaded']}")
    print(f"    └─ Per token:          {baseline['children_loaded']/n_tokens:.1f} children/token")

    # ========================================================================
    # BENCHMARK 2: Pre-warming
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. PRE-WARMING: Load children during prompt processing")
    print("=" * 80)

    prewarmed = benchmark_prewarming(model, tokenizer, lens_manager, prompt, device, n_tokens)

    print(f"\nPrompt Phase:")
    print(f"  Total time:              {prewarmed['prompt_eval_ms']:.2f}ms")
    print(f"  Lens detection:         {prewarmed['prompt_lens_detection_ms']:.2f}ms")
    print(f"  Child loading:           {prewarmed['prompt_child_loading_ms']:.2f}ms")
    print(f"  Children loaded:         {prewarmed['prompt_children']}")

    print(f"\nGeneration Phase:")
    print(f"  Generation time:         {prewarmed['generation_ms']:.2f}ms")
    print(f"  Lens detection:         {prewarmed['generation_lens_detection_ms']:.2f}ms")
    print(f"    └─ Per token:          {prewarmed['per_token_ms']:.2f}ms/token")
    print(f"  Child loading (disk I/O):{prewarmed['generation_child_loading_ms']:.2f}ms")
    print(f"    └─ Per token:          {prewarmed['child_loading_per_token_ms']:.2f}ms/token")
    print(f"  Children loaded:         {prewarmed['generation_children']}")
    print(f"    └─ Per token:          {prewarmed['generation_children']/n_tokens:.1f} children/token")

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    # Child loading reduction
    baseline_child_loading = baseline['child_loading_per_token_ms']
    prewarmed_child_loading = prewarmed['child_loading_per_token_ms']
    child_loading_reduction = baseline_child_loading - prewarmed_child_loading
    child_loading_pct = (child_loading_reduction / baseline_child_loading * 100) if baseline_child_loading > 0 else 0

    print(f"\nChild Loading (disk I/O) per token:")
    print(f"  Baseline:    {baseline_child_loading:.2f}ms/token")
    print(f"  Pre-warming: {prewarmed_child_loading:.2f}ms/token")
    print(f"  Reduction:   {child_loading_reduction:.2f}ms/token ({child_loading_pct:.1f}% improvement)")

    # Total lens overhead per token
    baseline_overhead = baseline['per_token_ms']
    prewarmed_overhead = prewarmed['per_token_ms']
    overhead_reduction = baseline_overhead - prewarmed_overhead
    overhead_pct = (overhead_reduction / baseline_overhead * 100) if baseline_overhead > 0 else 0

    print(f"\nTotal Lens Overhead per token:")
    print(f"  Baseline:    {baseline_overhead:.2f}ms/token")
    print(f"  Pre-warming: {prewarmed_overhead:.2f}ms/token")
    print(f"  Reduction:   {overhead_reduction:.2f}ms/token ({overhead_pct:.1f}% improvement)")

    # Cache hit rate (proxy)
    baseline_children_per_token = baseline['children_loaded'] / n_tokens
    prewarmed_children_per_token = prewarmed['generation_children'] / n_tokens

    print(f"\nChildren loaded during generation:")
    print(f"  Baseline:    {baseline_children_per_token:.1f} children/token")
    print(f"  Pre-warming: {prewarmed_children_per_token:.1f} children/token")
    print(f"  Reduction:   {baseline_children_per_token - prewarmed_children_per_token:.1f} children/token")

    # Concept overlap validation
    if prewarmed['prompt_children'] > 0:
        overlap_estimate = (baseline['children_loaded'] - prewarmed['generation_children']) / prewarmed['prompt_children'] * 100
        print(f"\nEstimated concept overlap:")
        print(f"  Prompt pre-loaded {prewarmed['prompt_children']} children")
        print(f"  Generation avoided loading ~{baseline['children_loaded'] - prewarmed['generation_children']} children")
        print(f"  Overlap: ~{min(overlap_estimate, 100):.0f}%")

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if overhead_reduction > 5.0:
        print(f"\n✓ Pre-warming provides significant benefit: {overhead_reduction:.2f}ms/token reduction")
        print(f"  Current overhead: {prewarmed_overhead:.2f}ms/token (vs {baseline_overhead:.2f}ms baseline)")
        print(f"  Gap to target (<10ms): {max(0, prewarmed_overhead - 10.0):.2f}ms")
    elif overhead_reduction > 1.0:
        print(f"\n~ Pre-warming provides modest benefit: {overhead_reduction:.2f}ms/token reduction")
        print(f"  May be worth implementing depending on use case")
    else:
        print(f"\n✗ Pre-warming provides minimal benefit: {overhead_reduction:.2f}ms/token reduction")
        print(f"  Not worth the implementation complexity")

    print(f"\nNote: Prompt processing overhead ({prewarmed['prompt_eval_ms']:.2f}ms) is one-time cost")
    print(f"      amortized over {n_tokens} generation tokens = {prewarmed['prompt_eval_ms']/n_tokens:.2f}ms/token")

    return 0


if __name__ == '__main__':
    sys.exit(main())
