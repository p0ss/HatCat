#!/usr/bin/env python3
"""
Detailed per-millisecond breakdown of DynamicLensManager overhead.

Profiles:
1. Base lens inference (per lens)
2. Child loading (disk I/O + initialization)
3. Child lens inference
4. Cache management
5. Python overhead (sorting, dict operations)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hat.monitoring.lens_manager import DynamicLensManager


def profile_single_token_detailed(model, tokenizer, lens_manager, prompt: str, device: str = "cuda"):
    """Get extremely detailed timing breakdown for a single token."""
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        # Generate one token and extract hidden state
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        hidden_state = outputs.hidden_states[0][-1][:, -1, :].float()

    # Run detect_and_expand with detailed timing
    start_total = time.perf_counter()
    detected, timing_info = lens_manager.detect_and_expand(
        hidden_state,
        top_k=10,
        return_timing=True
    )
    total_time = time.perf_counter() - start_total

    return {
        'total_ms': total_time * 1000,
        'initial_detection_ms': timing_info.get('initial_detection', 0),
        'child_loading_ms': timing_info.get('child_loading', 0),
        'child_detection_ms': timing_info.get('child_detection', 0),
        'cache_management_ms': timing_info.get('cache_management', 0),
        'num_children_loaded': timing_info.get('num_children_loaded', 0),
        'num_base_lenses': len(lens_manager.loaded_lenses) - timing_info.get('num_children_loaded', 0),
        'num_total_lenses': len(lens_manager.loaded_lenses),
        'concepts_detected': len(detected),
    }


def profile_base_lens_inference(lens_manager, hidden_state, num_runs=100):
    """Profile time to run inference on all base lenses."""
    times = []

    with torch.inference_mode():
        for _ in range(num_runs):
            start = time.perf_counter()
            for concept_key, lens in lens_manager.loaded_lenses.items():
                prob = lens(hidden_state).item()
            times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times) * 1000
    per_lens = avg_time / len(lens_manager.loaded_lenses)

    return {
        'avg_total_ms': avg_time,
        'per_lens_ms': per_lens,
        'num_lenses': len(lens_manager.loaded_lenses),
    }


def profile_python_overhead(current_scores, top_k=10):
    """Profile Python overhead: sorting, dict operations."""
    times = {
        'sorting': 0,
        'dict_lookup': 0,
        'list_append': 0,
    }

    num_runs = 1000

    # Sorting overhead
    for _ in range(num_runs):
        start = time.perf_counter()
        sorted_concepts = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_concepts = sorted_concepts[:top_k]
        times['sorting'] += time.perf_counter() - start

    # Dict lookup overhead
    parent_to_children = {k: [f"child_{i}" for i in range(3)] for k in current_scores.keys()}
    for _ in range(num_runs):
        start = time.perf_counter()
        for concept_key, prob in top_k_concepts:
            child_keys = parent_to_children.get(concept_key, [])
        times['dict_lookup'] += time.perf_counter() - start

    # List operations
    for _ in range(num_runs):
        child_keys_to_load = []
        start = time.perf_counter()
        for concept_key, prob in top_k_concepts:
            child_keys = parent_to_children.get(concept_key, [])
            for child_key in child_keys:
                if child_key not in current_scores:
                    child_keys_to_load.append(child_key)
        times['list_append'] += time.perf_counter() - start

    return {k: (v / num_runs) * 1000 for k, v in times.items()}


def main():
    print("=" * 80)
    print("DETAILED LENS OVERHEAD ANALYSIS")
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

    # Initialize DynamicLensManager
    print("\n" + "-" * 80)
    print("INITIALIZING LENS MANAGER (base_layers=[0,1])")
    print("-" * 80)
    lens_manager = DynamicLensManager(
        lens_pack_id="gemma-3-4b-pt_sumo-wordnet-v3",
        base_layers=[0, 1],
        max_loaded_lenses=500,
        load_threshold=0.3,
        device=device
    )
    print(f"✓ Loaded {len(lens_manager.loaded_lenses)} base lenses")

    # Get a hidden state for testing
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

    # ========================================================================
    # 1. SINGLE TOKEN DETAILED BREAKDOWN
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. SINGLE TOKEN DETAILED BREAKDOWN (top_k=10)")
    print("=" * 80)

    detailed = profile_single_token_detailed(model, tokenizer, lens_manager, prompt, device)

    print(f"\nTotal time: {detailed['total_ms']:.2f}ms")
    print(f"\nBreakdown:")
    print(f"  Initial detection:  {detailed['initial_detection_ms']:.2f}ms ({detailed['initial_detection_ms']/detailed['total_ms']*100:.1f}%)")
    print(f"    └─ {detailed['num_base_lenses']} base lenses @ {detailed['initial_detection_ms']/detailed['num_base_lenses']:.3f}ms/lens")
    print(f"  Child loading:      {detailed['child_loading_ms']:.2f}ms ({detailed['child_loading_ms']/detailed['total_ms']*100:.1f}%)")
    print(f"    └─ {detailed['num_children_loaded']} children loaded")
    if detailed['num_children_loaded'] > 0:
        print(f"    └─ {detailed['child_loading_ms']/detailed['num_children_loaded']:.2f}ms per child")
    print(f"  Child detection:    {detailed['child_detection_ms']:.2f}ms ({detailed['child_detection_ms']/detailed['total_ms']*100:.1f}%)")
    if detailed['num_children_loaded'] > 0:
        print(f"    └─ {detailed['child_detection_ms']/detailed['num_children_loaded']:.3f}ms/lens")
    print(f"  Cache management:   {detailed['cache_management_ms']:.2f}ms ({detailed['cache_management_ms']/detailed['total_ms']*100:.1f}%)")

    accounted = (detailed['initial_detection_ms'] + detailed['child_loading_ms'] +
                 detailed['child_detection_ms'] + detailed['cache_management_ms'])
    unaccounted = detailed['total_ms'] - accounted
    print(f"  Python overhead:    {unaccounted:.2f}ms ({unaccounted/detailed['total_ms']*100:.1f}%) [estimated]")

    print(f"\nResults:")
    print(f"  Concepts detected: {detailed['concepts_detected']}")
    print(f"  Total lenses run: {detailed['num_base_lenses'] + detailed['num_children_loaded']}")

    # ========================================================================
    # 2. BASE LENS INFERENCE MICROBENCHMARK
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. BASE LENS INFERENCE MICROBENCHMARK")
    print("=" * 80)

    print(f"\nRunning 100 iterations of {len(lens_manager.loaded_lenses)} lenses...")
    base_timing = profile_base_lens_inference(lens_manager, hidden_state, num_runs=100)

    print(f"\nResults:")
    print(f"  Average total time: {base_timing['avg_total_ms']:.2f}ms")
    print(f"  Per-lens time:     {base_timing['per_lens_ms']:.4f}ms")
    print(f"  Num lenses:         {base_timing['num_lenses']}")

    # ========================================================================
    # 3. PYTHON OVERHEAD MICROBENCHMARK
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. PYTHON OVERHEAD MICROBENCHMARK")
    print("=" * 80)

    # Create mock scores
    mock_scores = {(f"concept_{i}", i % 6): 0.5 + (i * 0.01) % 0.5
                   for i in range(len(lens_manager.loaded_lenses))}

    print(f"\nRunning 1000 iterations with {len(mock_scores)} concepts...")
    py_overhead = profile_python_overhead(mock_scores, top_k=10)

    print(f"\nResults:")
    print(f"  Sorting (top-k):    {py_overhead['sorting']:.4f}ms")
    print(f"  Dict lookups:       {py_overhead['dict_lookup']:.4f}ms")
    print(f"  List operations:    {py_overhead['list_append']:.4f}ms")
    print(f"  Total Python:       {sum(py_overhead.values()):.4f}ms")

    # ========================================================================
    # 4. MULTI-TOKEN AVERAGE
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. MULTI-TOKEN AVERAGE (10 tokens)")
    print("=" * 80)

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    timings = {
        'generation': 0,
        'hidden_extraction': 0,
        'lens_detection': 0,
        'initial_detection': 0,
        'child_loading': 0,
        'child_detection': 0,
        'cache_management': 0,
    }

    n_tokens = 10
    total_children_loaded = 0

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
            start_extract = time.perf_counter()
            last_layer = step_states[-1]
            hidden_state = last_layer[:, -1, :]
            hidden_state_f32 = hidden_state.float()
            timings['hidden_extraction'] += time.perf_counter() - start_extract

            # Lens detection
            start_detect = time.perf_counter()
            detected, timing_info = lens_manager.detect_and_expand(
                hidden_state_f32,
                top_k=10,
                return_timing=True
            )
            timings['lens_detection'] += time.perf_counter() - start_detect
            timings['initial_detection'] += timing_info.get('initial_detection', 0)
            timings['child_loading'] += timing_info.get('child_loading', 0)
            timings['child_detection'] += timing_info.get('child_detection', 0)
            timings['cache_management'] += timing_info.get('cache_management', 0)
            total_children_loaded += timing_info.get('num_children_loaded', 0)

    print(f"\nGeneration: {timings['generation']*1000:.2f}ms ({timings['generation']*1000/n_tokens:.2f}ms/token)")
    print(f"Hidden extraction: {timings['hidden_extraction']*1000:.2f}ms ({timings['hidden_extraction']*1000/n_tokens:.2f}ms/token)")
    print(f"\nLens detection: {timings['lens_detection']*1000:.2f}ms ({timings['lens_detection']*1000/n_tokens:.2f}ms/token)")
    print(f"  Initial detection:  {timings['initial_detection']:.2f}ms ({timings['initial_detection']/n_tokens:.2f}ms/token)")
    print(f"  Child loading:      {timings['child_loading']:.2f}ms ({timings['child_loading']/n_tokens:.2f}ms/token)")
    print(f"  Child detection:    {timings['child_detection']:.2f}ms ({timings['child_detection']/n_tokens:.2f}ms/token)")
    print(f"  Cache management:   {timings['cache_management']:.2f}ms ({timings['cache_management']/n_tokens:.2f}ms/token)")

    total_accounted = (timings['initial_detection'] + timings['child_loading'] +
                      timings['child_detection'] + timings['cache_management'])
    python_overhead = timings['lens_detection']*1000 - total_accounted
    print(f"  Python overhead:    {python_overhead:.2f}ms ({python_overhead/n_tokens:.2f}ms/token) [estimated]")

    print(f"\nTotal children loaded: {total_children_loaded} ({total_children_loaded/n_tokens:.1f}/token)")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY & OPTIMIZATION TARGETS")
    print("=" * 80)

    per_token_overhead = timings['lens_detection'] * 1000 / n_tokens

    print(f"\nCurrent per-token overhead: {per_token_overhead:.2f}ms")
    print(f"Target overhead: 10.0ms")
    print(f"Gap: {per_token_overhead - 10.0:.2f}ms ({(per_token_overhead - 10.0)/per_token_overhead*100:.1f}%)")

    print(f"\nOptimization opportunities (ranked by impact):")
    components = [
        ('Initial detection', timings['initial_detection']/n_tokens),
        ('Child loading (disk I/O)', timings['child_loading']/n_tokens),
        ('Child detection', timings['child_detection']/n_tokens),
        ('Python overhead', python_overhead/n_tokens),
        ('Cache management', timings['cache_management']/n_tokens),
    ]
    components.sort(key=lambda x: x[1], reverse=True)

    for i, (name, time_ms) in enumerate(components, 1):
        pct = (time_ms / per_token_overhead) * 100
        print(f"  {i}. {name:30s} {time_ms:6.2f}ms ({pct:5.1f}%)")

    print(f"\nRecommendations:")
    if components[0][0] == 'Initial detection':
        avg_per_lens = timings['initial_detection'] / n_tokens / len(lens_manager.loaded_lenses)
        print(f"  • Initial detection: Batching could help (currently {avg_per_lens*1000:.3f}ms/lens)")
        print(f"    - Run all {len(lens_manager.loaded_lenses)} lenses in single batch instead of loop")

    if components[0][0] == 'Child loading (disk I/O)' or components[1][0] == 'Child loading (disk I/O)':
        avg_per_child = timings['child_loading'] / total_children_loaded if total_children_loaded > 0 else 0
        print(f"  • Child loading: Implement caching/preloading ({avg_per_child:.2f}ms per child)")
        print(f"    - Consider LRU cache for frequently accessed children")
        print(f"    - Or reduce base layers further (currently [0,1])")

    if python_overhead / n_tokens > 5.0:
        print(f"  • Python overhead: Consider Cython/numba for hot paths")

    return 0


if __name__ == '__main__':
    sys.exit(main())
