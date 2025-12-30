#!/usr/bin/env python3
"""
Profile DynamicLensManager overhead to identify bottlenecks.

Measures timing for:
1. Model forward pass (baseline)
2. Lens detection only
3. Dynamic lens loading/unloading
4. Overall per-token overhead
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hat.monitoring.lens_manager import DynamicLensManager


def profile_model_baseline(model, tokenizer, prompt: str, n_tokens: int = 30, device: str = "cuda"):
    """Profile pure model generation without any lens overhead."""
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        start = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=True,
            temperature=0.8,
            output_hidden_states=False,  # Don't even extract hidden states
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        end = time.perf_counter()

    elapsed = end - start
    per_token = elapsed / n_tokens * 1000  # milliseconds

    return {
        'total_time': elapsed,
        'per_token_ms': per_token,
        'tokens': n_tokens
    }


def profile_with_hidden_states(model, tokenizer, prompt: str, n_tokens: int = 30, device: str = "cuda"):
    """Profile model generation with hidden state extraction."""
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.inference_mode():
        start = time.perf_counter()
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

        # Extract hidden states like we do in tests
        for step_idx, step_states in enumerate(outputs.hidden_states):
            last_layer = step_states[-1]
            hidden_state = last_layer[:, -1, :]
            hidden_state_f32 = hidden_state.float()

        end = time.perf_counter()

    elapsed = end - start
    per_token = elapsed / n_tokens * 1000

    return {
        'total_time': elapsed,
        'per_token_ms': per_token,
        'tokens': n_tokens
    }


def profile_with_lens_detection(model, tokenizer, lens_manager, prompt: str, n_tokens: int = 30, device: str = "cuda"):
    """Profile with lens detection but minimal dynamic loading."""
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    timings = {
        'generation': 0,
        'hidden_extraction': 0,
        'lens_detection': 0,
        'total': 0
    }

    start_total = time.perf_counter()

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

    timings['total'] = time.perf_counter() - start_total

    per_token = timings['total'] / n_tokens * 1000

    return {
        'total_time': timings['total'],
        'generation_time': timings['generation'],
        'hidden_extraction_time': timings['hidden_extraction'],
        'lens_detection_time': timings['lens_detection'],
        'per_token_ms': per_token,
        'tokens': n_tokens,
        'breakdown': {
            'generation_pct': timings['generation'] / timings['total'] * 100,
            'hidden_extraction_pct': timings['hidden_extraction'] / timings['total'] * 100,
            'lens_detection_pct': timings['lens_detection'] / timings['total'] * 100,
        }
    }


def profile_detect_and_expand_internal(lens_manager, hidden_state):
    """Profile internal detect_and_expand breakdown."""
    # Get detailed timing from lens_manager
    with torch.inference_mode():
        start = time.perf_counter()
        detected, timing_info = lens_manager.detect_and_expand(
            hidden_state,
            top_k=10,
            return_timing=True
        )
        total = time.perf_counter() - start

    return {
        'total_ms': total * 1000,
        'detection_ms': timing_info.get('detection_time', 0) * 1000,
        'expansion_ms': timing_info.get('expansion_time', 0) * 1000,
        'loading_ms': timing_info.get('loading_time', 0) * 1000,
        'pruning_ms': timing_info.get('pruning_time', 0) * 1000,
        'lenses_loaded': timing_info.get('lenses_loaded', 0),
        'concepts_detected': len(detected)
    }


def main():
    print("=" * 80)
    print("DYNAMIC LENS MANAGER - PERFORMANCE PROFILING")
    print("=" * 80)

    device = "cuda"
    model_name = "google/gemma-3-4b-pt"
    prompt = "Artificial intelligence can help society by"
    n_tokens = 30

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Tokens: {n_tokens}")
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

    # Baseline: pure generation
    print("\n" + "-" * 80)
    print("1. BASELINE: Pure generation (no hidden states)")
    print("-" * 80)
    baseline = profile_model_baseline(model, tokenizer, prompt, n_tokens, device)
    print(f"Total time: {baseline['total_time']:.3f}s")
    print(f"Per-token: {baseline['per_token_ms']:.2f}ms")

    # With hidden states
    print("\n" + "-" * 80)
    print("2. WITH HIDDEN STATES: Generation + extraction")
    print("-" * 80)
    with_hidden = profile_with_hidden_states(model, tokenizer, prompt, n_tokens, device)
    print(f"Total time: {with_hidden['total_time']:.3f}s")
    print(f"Per-token: {with_hidden['per_token_ms']:.2f}ms")
    overhead = with_hidden['per_token_ms'] - baseline['per_token_ms']
    print(f"Overhead from hidden state extraction: {overhead:.2f}ms per token")

    # Initialize DynamicLensManager
    print("\n" + "-" * 80)
    print("3. INITIALIZING LENS MANAGER")
    print("-" * 80)
    print("Loading base layers [0, 1]...")
    start_init = time.perf_counter()
    lens_manager = DynamicLensManager(
        lens_pack_id="gemma-3-4b-pt_sumo-wordnet-v3",
        base_layers=[0, 1],
        max_loaded_lenses=500,
        load_threshold=0.3,
        device=device
    )
    init_time = time.perf_counter() - start_init
    print(f"✓ Initialized in {init_time:.2f}s")
    print(f"  Base lenses loaded: {len(lens_manager.loaded_lenses)}")

    # With lens detection
    print("\n" + "-" * 80)
    print("4. WITH LENS DETECTION: Full pipeline")
    print("-" * 80)
    with_lenses = profile_with_lens_detection(model, tokenizer, lens_manager, prompt, n_tokens, device)
    print(f"Total time: {with_lenses['total_time']:.3f}s")
    print(f"Per-token: {with_lenses['per_token_ms']:.2f}ms")
    print(f"\nBreakdown:")
    print(f"  Generation:        {with_lenses['generation_time']:.3f}s ({with_lenses['breakdown']['generation_pct']:.1f}%)")
    print(f"  Hidden extraction: {with_lenses['hidden_extraction_time']:.3f}s ({with_lenses['breakdown']['hidden_extraction_pct']:.1f}%)")
    print(f"  Lens detection:   {with_lenses['lens_detection_time']:.3f}s ({with_lenses['breakdown']['lens_detection_pct']:.1f}%)")

    lens_overhead = with_lenses['per_token_ms'] - with_hidden['per_token_ms']
    print(f"\nLens detection overhead: {lens_overhead:.2f}ms per token")

    # Profile single detect_and_expand call in detail
    print("\n" + "-" * 80)
    print("5. DETAILED LENS DETECTION (single token)")
    print("-" * 80)

    # Get a hidden state
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

    internal = profile_detect_and_expand_internal(lens_manager, hidden_state)
    print(f"Total detect_and_expand: {internal['total_ms']:.2f}ms")
    print(f"  Detection: {internal['detection_ms']:.2f}ms")
    print(f"  Expansion: {internal['expansion_ms']:.2f}ms")
    print(f"  Loading:   {internal['loading_ms']:.2f}ms")
    print(f"  Pruning:   {internal['pruning_ms']:.2f}ms")
    print(f"  Lenses loaded: {internal['lenses_loaded']}")
    print(f"  Concepts detected: {internal['concepts_detected']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline generation:        {baseline['per_token_ms']:.2f}ms/token")
    print(f"+ Hidden state extraction:  {with_hidden['per_token_ms']:.2f}ms/token (+{overhead:.2f}ms)")
    print(f"+ Lens detection:          {with_lenses['per_token_ms']:.2f}ms/token (+{lens_overhead:.2f}ms)")
    print(f"\nTotal HatCat overhead:      {with_lenses['per_token_ms'] - baseline['per_token_ms']:.2f}ms/token")

    target_overhead = 10.0  # Target: <10ms per token
    current_overhead = with_lenses['per_token_ms'] - baseline['per_token_ms']

    if current_overhead <= target_overhead:
        print(f"✓ ACCEPTABLE: Overhead is within {target_overhead}ms target")
    else:
        print(f"✗ UNACCEPTABLE: Overhead of {current_overhead:.2f}ms exceeds {target_overhead}ms target")
        print(f"  Need to reduce by: {current_overhead - target_overhead:.2f}ms")

    return 0


if __name__ == '__main__':
    sys.exit(main())
