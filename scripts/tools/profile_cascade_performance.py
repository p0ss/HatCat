#!/usr/bin/env python3
"""
Profile cascade detection performance to identify bottlenecks.

Measures:
1. Lens loading time (file I/O, state dict loading, GPU transfer)
2. Inference time (forward pass through lenses)
3. Child lookup time (parent-child mapping traversal)
4. Sorting/filtering time (top-K selection)
5. Overall memory allocation

Goal: Identify slowest operations for optimization.
"""

import sys
import time
from pathlib import Path
import cProfile
import pstats
from io import StringIO

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.monitoring.lens_manager import DynamicLensManager


def profile_single_cascade():
    """Profile a single cascade with detailed timing."""
    print("=" * 80)
    print("DETAILED CASCADE PROFILING")
    print("=" * 80)

    prompt = "The cat sat on the mat"
    model_name = "google/gemma-3-4b-pt"
    device = "cuda"

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    # Initialize manager
    print("\nInitializing manager...")
    t_start = time.time()
    manager = DynamicLensManager(
        device=device,
        base_layers=[0],
        load_threshold=0.3,
        max_loaded_lenses=1000,
    )
    init_time = (time.time() - t_start) * 1000
    print(f"✓ Manager initialization: {init_time:.2f}ms")

    # Extract hidden state
    print(f"\nExtracting hidden state from: \"{prompt}\"")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_state = outputs.hidden_states[-1].mean(dim=1)

    # Profile 3 cascade levels with detailed breakdown
    print("\n" + "=" * 80)
    print("PROFILING CASCADE LEVELS")
    print("=" * 80)

    for level in range(1, 4):
        print(f"\n{'=' * 80}")
        print(f"LEVEL {level}")
        print(f"{'=' * 80}")

        # Manual detailed timing
        t_total_start = time.time()

        # Step 1: Run loaded lenses
        t1 = time.time()
        current_scores = {}
        with torch.inference_mode():
            for concept_key, lens in manager.loaded_lenses.items():
                prob = lens(hidden_state).item()
                current_scores[concept_key] = prob
        inference_time = (time.time() - t1) * 1000

        # Step 2: Identify high-confidence parents
        t2 = time.time()
        child_keys_to_load = []
        high_confidence_parents = []
        for concept_key, prob in current_scores.items():
            if prob > manager.load_threshold:
                high_confidence_parents.append(concept_key)
                child_keys = manager.parent_to_children.get(concept_key, [])
                for child_key in child_keys:
                    if child_key not in manager.loaded_lenses:
                        child_keys_to_load.append(child_key)
        lookup_time = (time.time() - t2) * 1000

        # Step 3: Load children (detailed breakdown)
        t3 = time.time()
        load_times = []
        for child_key in child_keys_to_load:
            t_load_start = time.time()
            metadata = manager.concept_metadata.get(child_key)
            if not metadata or not metadata.activation_lens_path:
                continue

            # File I/O
            t_io = time.time()
            state_dict = torch.load(metadata.activation_lens_path, map_location='cpu')
            io_time = (time.time() - t_io) * 1000

            # Model creation + state dict loading
            t_model = time.time()
            from src.hat.monitoring.lens_manager import SimpleMLP
            lens = SimpleMLP(manager.hidden_dim)

            # Handle key mismatch
            model_keys = set(lens.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            if model_keys != loaded_keys:
                new_state_dict = {}
                for key, value in state_dict.items():
                    if not key.startswith('net.'):
                        new_state_dict[f'net.{key}'] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict

            lens.load_state_dict(state_dict)
            model_time = (time.time() - t_model) * 1000

            # GPU transfer
            t_gpu = time.time()
            lens = lens.to(device)
            lens.eval()
            gpu_time = (time.time() - t_gpu) * 1000

            manager.loaded_lenses[child_key] = lens
            total_load = (time.time() - t_load_start) * 1000

            load_times.append({
                'io': io_time,
                'model': model_time,
                'gpu': gpu_time,
                'total': total_load,
            })

        loading_time = (time.time() - t3) * 1000

        # Step 4: Run newly loaded lenses
        t4 = time.time()
        with torch.inference_mode():
            for child_key in child_keys_to_load:
                if child_key in manager.loaded_lenses:
                    lens = manager.loaded_lenses[child_key]
                    prob = lens(hidden_state).item()
                    current_scores[child_key] = prob
        new_inference_time = (time.time() - t4) * 1000

        # Step 5: Sort and select top K
        t5 = time.time()
        results = []
        for concept_key, prob in current_scores.items():
            concept_name, layer = concept_key
            results.append((concept_name, prob, layer))
        results.sort(key=lambda x: x[1], reverse=True)
        top_k = results[:10]
        sorting_time = (time.time() - t5) * 1000

        total_time = (time.time() - t_total_start) * 1000

        # Print detailed breakdown
        print(f"\nLoaded lenses before: {len(manager.loaded_lenses) - len(child_keys_to_load)}")
        print(f"High-confidence parents: {len(high_confidence_parents)}")
        print(f"Children to load: {len(child_keys_to_load)}")
        print(f"Loaded lenses after: {len(manager.loaded_lenses)}")

        print(f"\n{'Timing Breakdown:'}")
        print(f"  1. Inference (existing lenses):    {inference_time:8.2f}ms  ({inference_time/total_time*100:5.1f}%)")
        print(f"  2. Parent-child lookup:            {lookup_time:8.2f}ms  ({lookup_time/total_time*100:5.1f}%)")
        print(f"  3. Loading children:               {loading_time:8.2f}ms  ({loading_time/total_time*100:5.1f}%)")
        print(f"  4. Inference (new lenses):         {new_inference_time:8.2f}ms  ({new_inference_time/total_time*100:5.1f}%)")
        print(f"  5. Sorting/filtering:              {sorting_time:8.2f}ms  ({sorting_time/total_time*100:5.1f}%)")
        print(f"  {'─' * 60}")
        print(f"  TOTAL:                             {total_time:8.2f}ms")

        # Detailed loading breakdown
        if load_times:
            avg_io = sum(lt['io'] for lt in load_times) / len(load_times)
            avg_model = sum(lt['model'] for lt in load_times) / len(load_times)
            avg_gpu = sum(lt['gpu'] for lt in load_times) / len(load_times)
            avg_total = sum(lt['total'] for lt in load_times) / len(load_times)

            print(f"\n{'Per-Lens Loading Breakdown:'}")
            print(f"  File I/O (torch.load):             {avg_io:8.2f}ms  ({avg_io/avg_total*100:5.1f}%)")
            print(f"  Model creation + state_dict:       {avg_model:8.2f}ms  ({avg_model/avg_total*100:5.1f}%)")
            print(f"  GPU transfer (.to(device)):        {avg_gpu:8.2f}ms  ({avg_gpu/avg_total*100:5.1f}%)")
            print(f"  {'─' * 60}")
            print(f"  Average per lens:                 {avg_total:8.2f}ms")
            print(f"  Total loading time:                {loading_time:8.2f}ms")

        print(f"\n{'Top 5 concepts:'}")
        for i, (concept, prob, layer) in enumerate(top_k[:5], 1):
            print(f"  {i}. [L{layer}] {concept:30s} {prob:.3f}")

    # Final statistics
    print("\n" + "=" * 80)
    print("FINAL MEMORY STATISTICS")
    print("=" * 80)
    print(f"Total lenses loaded: {len(manager.loaded_lenses)}")
    print(f"Total concepts available: {len(manager.concept_metadata)}")
    print(f"Memory footprint: {len(manager.loaded_lenses)/len(manager.concept_metadata)*100:.2f}%")
    print(f"Estimated memory: ~{len(manager.loaded_lenses) * 1.3:.0f}MB")


def profile_per_token_simulation():
    """Simulate per-token cascade during generation."""
    print("\n" + "=" * 80)
    print("PER-TOKEN CASCADE SIMULATION")
    print("=" * 80)

    prompt = "The cat sat on the mat"
    model_name = "google/gemma-3-4b-pt"
    device = "cuda"
    num_tokens = 10

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    # Initialize manager
    manager = DynamicLensManager(
        device=device,
        base_layers=[0],
        load_threshold=0.3,
        max_loaded_lenses=500,  # Tighter memory budget
    )

    # Generate tokens
    print(f"\nGenerating {num_tokens} tokens...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # Simulate per-token cascade
    print(f"\nSimulating cascade for each token...")
    print(f"{'─' * 80}")

    token_times = []
    for i, step_states in enumerate(outputs.hidden_states[:num_tokens]):
        # Extract hidden state for this token
        last_layer = step_states[-1]
        hidden_state = last_layer[:, -1, :]

        # Run single cascade level (typical for per-token)
        t_start = time.time()
        results, timing = manager.detect_and_expand(
            hidden_state,
            top_k=10,
            return_timing=True,
        )
        token_time = (time.time() - t_start) * 1000

        token_times.append(token_time)

        top_concept = results[0][0] if results else "N/A"
        print(f"  Token {i+1:2d}: {token_time:6.2f}ms  "
              f"(detection={timing['initial_detection']:5.2f}ms, "
              f"loading={timing['child_loading']:5.2f}ms, "
              f"children={timing['num_children_loaded']:3d}, "
              f"loaded={timing['loaded_lenses']:3d}) "
              f"→ {top_concept}")

    print(f"{'─' * 80}")
    avg_time = sum(token_times) / len(token_times)
    max_time = max(token_times)
    min_time = min(token_times)
    print(f"\nPer-token statistics:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min:     {min_time:.2f}ms")
    print(f"  Max:     {max_time:.2f}ms")
    print(f"\nEstimated overhead for 100-token generation: {avg_time * 100:.0f}ms ({avg_time * 100 / 1000:.1f}s)")


def profile_with_cprofile():
    """Use cProfile for detailed Python profiling."""
    print("\n" + "=" * 80)
    print("PYTHON PROFILING (cProfile)")
    print("=" * 80)

    prompt = "The cat sat on the mat"
    model_name = "google/gemma-3-4b-pt"
    device = "cuda"

    # Load model and manager
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    manager = DynamicLensManager(
        device=device,
        base_layers=[0],
        load_threshold=0.3,
        max_loaded_lenses=1000,
    )

    # Extract hidden state
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_state = outputs.hidden_states[-1].mean(dim=1)

    # Profile cascade
    print("\nProfiling 3 cascade levels...")

    profiler = cProfile.Profile()
    profiler.enable()

    # Run 3 cascade levels
    for _ in range(3):
        results, timing = manager.detect_and_expand(hidden_state, return_timing=True)

    profiler.disable()

    # Print stats
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    print("\nTop 20 functions by cumulative time:")
    stats.print_stats(20)
    print(s.getvalue())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Profile cascade detection performance")
    parser.add_argument(
        '--mode',
        choices=['detailed', 'per-token', 'cprofile', 'all'],
        default='detailed',
        help='Profiling mode'
    )
    args = parser.parse_args()

    if args.mode in ['detailed', 'all']:
        profile_single_cascade()

    if args.mode in ['per-token', 'all']:
        profile_per_token_simulation()

    if args.mode in ['cprofile', 'all']:
        profile_with_cprofile()


if __name__ == '__main__':
    main()
