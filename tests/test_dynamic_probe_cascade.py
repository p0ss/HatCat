#!/usr/bin/env python3
"""
Test Dynamic Hierarchical Probe Cascade

Benchmark test:
1. Detect parent (Layer 0)
2. Load and detect children (Layer 1)
3. Load and detect grandchildren (Layer 2)

Measures speed at each level and total memory footprint.
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.dynamic_probe_manager import DynamicProbeManager


def test_cascade_detection(
    prompt: str = "The cat sat on the mat",
    model_name: str = "google/gemma-3-4b-pt",
    device: str = "cuda",
):
    """
    Test hierarchical cascade detection.

    Flow:
    1. Run base layers (0-1) on hidden state
    2. Identify highest scoring parent
    3. Load its children
    4. Identify highest scoring child
    5. Load its children (if any)

    Measures timing at each step.
    """
    print("=" * 80)
    print("DYNAMIC PROBE CASCADE TEST")
    print("=" * 80)
    print(f"Prompt: \"{prompt}\"")
    print(f"Model: {model_name}")
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    t_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()
    print(f"✓ Model loaded ({time.time() - t_start:.1f}s)")

    # Initialize dynamic probe manager
    print("\nInitializing DynamicProbeManager...")
    t_start = time.time()
    manager = DynamicProbeManager(
        device=device,
        base_layers=[0, 1],
        load_threshold=0.5,
        max_loaded_probes=500,
    )
    print(f"✓ Manager initialized ({time.time() - t_start:.1f}s)")

    # Extract hidden state from prompt
    print(f"\nExtracting hidden state from prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # Use last layer, mean pooled
        hidden_state = outputs.hidden_states[-1].mean(dim=1)  # [1, hidden_dim]

    print(f"✓ Hidden state shape: {hidden_state.shape}")

    # Run cascade detection
    print("\n" + "=" * 80)
    print("HIERARCHICAL CASCADE")
    print("=" * 80)

    # Level 1: Initial detection (base layers)
    print("\n[LEVEL 1] Initial detection with base layers (0-1)")
    print("-" * 80)

    results_1, timing_1 = manager.detect_and_expand(
        hidden_state,
        top_k=10,
        return_timing=True,
    )

    print(f"Initial detection: {timing_1['initial_detection']:.2f}ms")
    print(f"Loaded probes: {timing_1['loaded_probes']}")
    print(f"Children loaded: {timing_1['num_children_loaded']}")
    print(f"Total time: {timing_1['total']:.2f}ms")

    print(f"\nTop 10 concepts (Level 1):")
    for i, (concept, prob, layer) in enumerate(results_1, 1):
        bar = "█" * int(prob * 20)
        metadata = manager.concept_metadata.get(concept)
        children_info = f" ({len(metadata.category_children)} children)" if metadata and metadata.category_children else ""
        print(f"  {i:2d}. [L{layer}] {concept:30s} {prob:.3f} {bar}{children_info}")

    # Level 2: Second expansion (should load more children)
    print("\n[LEVEL 2] Second expansion")
    print("-" * 80)

    results_2, timing_2 = manager.detect_and_expand(
        hidden_state,
        top_k=10,
        return_timing=True,
    )

    print(f"Initial detection: {timing_2['initial_detection']:.2f}ms")
    print(f"Loaded probes: {timing_2['loaded_probes']}")
    print(f"Children loaded: {timing_2['num_children_loaded']}")
    print(f"Total time: {timing_2['total']:.2f}ms")

    print(f"\nTop 10 concepts (Level 2):")
    for i, (concept, prob, layer) in enumerate(results_2, 1):
        bar = "█" * int(prob * 20)
        metadata = manager.concept_metadata.get(concept)
        children_info = f" ({len(metadata.category_children)} children)" if metadata and metadata.category_children else ""
        print(f"  {i:2d}. [L{layer}] {concept:30s} {prob:.3f} {bar}{children_info}")

    # Level 3: Third expansion (convergence)
    print("\n[LEVEL 3] Third expansion")
    print("-" * 80)

    results_3, timing_3 = manager.detect_and_expand(
        hidden_state,
        top_k=10,
        return_timing=True,
    )

    print(f"Initial detection: {timing_3['initial_detection']:.2f}ms")
    print(f"Loaded probes: {timing_3['loaded_probes']}")
    print(f"Children loaded: {timing_3['num_children_loaded']}")
    print(f"Total time: {timing_3['total']:.2f}ms")

    print(f"\nTop 10 concepts (Level 3):")
    for i, (concept, prob, layer) in enumerate(results_3, 1):
        bar = "█" * int(prob * 20)
        metadata = manager.concept_metadata.get(concept)
        path = manager.get_concept_path(concept)
        path_str = " → ".join(path)
        print(f"  {i:2d}. [L{layer}] {concept:30s} {prob:.3f} {bar}")
        print(f"       Path: {path_str}")

    # Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    total_detection_time = timing_1['total'] + timing_2['total'] + timing_3['total']
    avg_per_level = total_detection_time / 3

    print(f"\nTiming per level:")
    print(f"  Level 1: {timing_1['total']:.2f}ms")
    print(f"  Level 2: {timing_2['total']:.2f}ms")
    print(f"  Level 3: {timing_3['total']:.2f}ms")
    print(f"  Average: {avg_per_level:.2f}ms/level")
    print(f"  Total:   {total_detection_time:.2f}ms")

    print(f"\nMemory efficiency:")
    print(f"  Final loaded probes: {timing_3['loaded_probes']}")
    print(f"  Total concepts available: {len(manager.concept_metadata)}")
    print(f"  Memory usage: {timing_3['loaded_probes']}/{len(manager.concept_metadata)} "
          f"({100 * timing_3['loaded_probes'] / len(manager.concept_metadata):.1f}%)")

    # Detailed stats
    manager.print_stats()

    # Test multiple prompts for realistic performance
    print("\n" + "=" * 80)
    print("MULTI-PROMPT BENCHMARK")
    print("=" * 80)

    test_prompts = [
        "The dog ran through the park",
        "Scientists discovered a new particle",
        "The computer processed the data",
        "She wrote a beautiful poem",
        "The economy is growing rapidly",
    ]

    print(f"\nTesting {len(test_prompts)} diverse prompts...")
    prompt_timings = []

    for i, test_prompt in enumerate(test_prompts, 1):
        # Extract hidden state
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_state = outputs.hidden_states[-1].mean(dim=1)

        # Run 3 cascade levels
        t_start = time.time()
        for _ in range(3):
            results, timing = manager.detect_and_expand(hidden_state, return_timing=True)
        total_time = (time.time() - t_start) * 1000

        prompt_timings.append(total_time)

        top_concept = results[0][0] if results else "N/A"
        print(f"  {i}. \"{test_prompt[:40]:40s}\" → {top_concept:20s} ({total_time:.1f}ms)")

    avg_time = sum(prompt_timings) / len(prompt_timings)
    print(f"\nAverage cascade time (3 levels): {avg_time:.2f}ms")
    print(f"Speed per level: {avg_time / 3:.2f}ms")

    print("\n" + "=" * 80)
    print("✓ CASCADE TEST COMPLETE")
    print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test dynamic hierarchical probe cascade")
    parser.add_argument(
        '--prompt',
        type=str,
        default="The cat sat on the mat",
        help='Test prompt'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='google/gemma-3-4b-pt',
        help='Model name'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda/cpu)'
    )
    args = parser.parse_args()

    test_cascade_detection(
        prompt=args.prompt,
        model_name=args.model,
        device=args.device,
    )


if __name__ == '__main__':
    main()
