#!/usr/bin/env python3
"""
Simple cascade test showing Layer 0 → 1 → 2 expansion.

Clear demonstration:
1. Start with ONLY Layer 0 (14 concepts)
2. Detect highest scoring parent
3. Load its children from Layer 1
4. Detect highest scoring child
5. Load its children from Layer 2
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.dynamic_probe_manager import DynamicProbeManager


def main():
    print("=" * 80)
    print("HIERARCHICAL PROBE CASCADE: Layer 0 → 1 → 2")
    print("=" * 80)

    prompt = "The cat sat on the mat"
    model_name = "google/gemma-3-4b-pt"
    device = "cuda"

    print(f"\nPrompt: \"{prompt}\"")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()
    print("✓ Model loaded")

    # Initialize with ONLY layer 0 as base
    print("\nInitializing manager with ONLY Layer 0 as base...")
    manager = DynamicProbeManager(
        device=device,
        base_layers=[0],  # Only layer 0
        load_threshold=0.3,  # Lower threshold to see more expansion
        max_loaded_probes=1000,
    )
    print(f"✓ Base layer loaded: {len(manager.loaded_probes)} probes (Layer 0 only)")

    # Extract hidden state
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_state = outputs.hidden_states[-1].mean(dim=1)

    print("\n" + "=" * 80)
    print("STEP 1: Detect with Layer 0 only")
    print("=" * 80)

    results, timing = manager.detect_and_expand(hidden_state, top_k=5, return_timing=True)

    print(f"\nDetection time: {timing['initial_detection']:.2f}ms")
    print(f"Children loaded: {timing['num_children_loaded']}")
    print(f"Total loaded probes: {timing['loaded_probes']}")
    print(f"Total time: {timing['total']:.2f}ms")

    print(f"\nTop 5 concepts:")
    for i, (concept, prob, layer) in enumerate(results, 1):
        metadata = manager.concept_metadata.get(concept)
        children = metadata.category_children if metadata else []
        print(f"  {i}. [L{layer}] {concept:30s} {prob:.3f}  ({len(children)} children)")

    print("\n" + "=" * 80)
    print("STEP 2: Second detection (should load Layer 1 children)")
    print("=" * 80)

    results, timing = manager.detect_and_expand(hidden_state, top_k=10, return_timing=True)

    print(f"\nDetection time: {timing['initial_detection']:.2f}ms")
    print(f"Children loaded: {timing['num_children_loaded']}")
    print(f"Total loaded probes: {timing['loaded_probes']}")
    print(f"Total time: {timing['total']:.2f}ms")

    print(f"\nTop 10 concepts:")
    for i, (concept, prob, layer) in enumerate(results, 1):
        metadata = manager.concept_metadata.get(concept)
        children = metadata.category_children if metadata else []
        path = manager.get_concept_path(concept)
        print(f"  {i}. [L{layer}] {concept:30s} {prob:.3f}  ({len(children)} children)")
        if len(path) > 1:
            print(f"       Path: {' → '.join(path)}")

    print("\n" + "=" * 80)
    print("STEP 3: Third detection (should load Layer 2 children)")
    print("=" * 80)

    results, timing = manager.detect_and_expand(hidden_state, top_k=10, return_timing=True)

    print(f"\nDetection time: {timing['initial_detection']:.2f}ms")
    print(f"Children loaded: {timing['num_children_loaded']}")
    print(f"Total loaded probes: {timing['loaded_probes']}")
    print(f"Total time: {timing['total']:.2f}ms")

    print(f"\nTop 10 concepts:")
    for i, (concept, prob, layer) in enumerate(results, 1):
        metadata = manager.concept_metadata.get(concept)
        children = metadata.category_children if metadata else []
        path = manager.get_concept_path(concept)
        print(f"  {i}. [L{layer}] {concept:30s} {prob:.3f}  ({len(children)} children)")
        if len(path) > 1:
            print(f"       Path: {' → '.join(path)}")

    # Show final stats
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total concepts in metadata: {len(manager.concept_metadata):,}")
    print(f"Loaded in memory: {len(manager.loaded_probes)}")
    print(f"Memory footprint: {len(manager.loaded_probes) / len(manager.concept_metadata) * 100:.2f}%")
    print(f"Estimated memory: ~{len(manager.loaded_probes) * 1.3:.0f}MB (activation probes)")

    # Show loaded concepts by layer
    loaded_by_layer = {}
    for concept_name in manager.loaded_probes.keys():
        metadata = manager.concept_metadata.get(concept_name)
        if metadata:
            layer = metadata.layer
            loaded_by_layer[layer] = loaded_by_layer.get(layer, 0) + 1

    print(f"\nLoaded probes by layer:")
    for layer in sorted(loaded_by_layer.keys()):
        count = loaded_by_layer[layer]
        print(f"  Layer {layer}: {count:4d} probes")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
