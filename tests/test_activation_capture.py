"""
Test script for activation capture system.
Validates that we can capture activations from Gemma-3 270M.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activation_capture import ActivationCapture, ActivationConfig, BaselineGenerator
from src.activation_capture.model_loader import ModelLoader
from src.utils import ActivationStorage


def test_model_loading():
    """Test that we can load the model."""
    print("=" * 80)
    print("TEST 1: Model Loading")
    print("=" * 80)

    model, tokenizer = ModelLoader.load_gemma_270m()

    print(f"\nModel type: {type(model).__name__}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Print layer structure
    ModelLoader.print_layer_structure(model, filter_str="layer")

    return model, tokenizer


def test_activation_capture(model, tokenizer):
    """Test basic activation capture."""
    print("\n" + "=" * 80)
    print("TEST 2: Basic Activation Capture")
    print("=" * 80)

    config = ActivationConfig(
        top_k=100,
        storage_dtype=torch.float16
    )

    with ActivationCapture(model, config) as capturer:
        capturer.register_hooks()

        # Test prompt
        prompt = "The concept of justice"
        inputs = tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        activations = capturer.get_activations()

        print(f"\nCaptured activations from {len(activations)} layers")
        print("\nSample layers:")
        for i, (name, tensor) in enumerate(list(activations.items())[:5]):
            print(f"  {name}: {tensor.shape}, dtype={tensor.dtype}")

            # Check sparsity
            if len(tensor.shape) > 1:
                sparsity = (tensor == 0).sum().item() / tensor.numel()
                print(f"    Sparsity: {sparsity:.2%}")

    return activations


def test_baseline_generation(model, tokenizer):
    """Test baseline activation generation."""
    print("\n" + "=" * 80)
    print("TEST 3: Baseline Generation")
    print("=" * 80)

    config = ActivationConfig(top_k=100, storage_dtype=torch.float16)
    baseline_gen = BaselineGenerator(model, tokenizer, config)

    baseline = baseline_gen.generate_baseline(num_samples=5)

    print(f"\nGenerated baseline from {len(baseline)} layers")
    print("\nSample baseline activations:")
    for i, (name, tensor) in enumerate(list(baseline.items())[:3]):
        print(f"  {name}: {tensor.shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f}")

    return baseline


def test_activation_diff(model, tokenizer, baseline):
    """Test activation difference computation."""
    print("\n" + "=" * 80)
    print("TEST 4: Activation Difference (Concept vs Baseline)")
    print("=" * 80)

    config = ActivationConfig(top_k=100, storage_dtype=torch.float16)

    # Capture activations for a concept
    concept_prompt = "The concept of justice is important for society"

    with ActivationCapture(model, config) as capturer:
        capturer.register_hooks()

        inputs = tokenizer(concept_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            _ = model(**inputs)

        concept_activations = capturer.get_activations()

        # Compute difference
        diffs = capturer.compute_activation_diff(baseline, concept_activations)

        print(f"\nComputed differences for {len(diffs)} layers")
        print("\nSample differences:")
        for i, (name, tensor) in enumerate(list(diffs.items())[:3]):
            print(f"  {name}:")
            print(f"    Shape: {tensor.shape}")
            print(f"    Mean abs diff: {tensor.abs().mean():.6f}")
            print(f"    Max abs diff: {tensor.abs().max():.6f}")
            print(f"    Non-zero: {(tensor != 0).sum().item()}/{tensor.numel()}")

    return diffs


def test_storage(baseline, diffs):
    """Test HDF5 storage."""
    print("\n" + "=" * 80)
    print("TEST 5: HDF5 Storage")
    print("=" * 80)

    storage_path = Path("data/test_activations.h5")
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy
    baseline_np = {k: v.cpu().numpy() for k, v in baseline.items()}
    diffs_np = {k: v.cpu().numpy() for k, v in diffs.items()}

    # Store
    with ActivationStorage(storage_path, mode='w') as storage:
        storage.store_baseline(baseline_np)
        storage.store_concept_activations(
            "justice",
            diffs_np,
            metadata={"prompt": "The concept of justice", "category": "abstract"}
        )

        # Get stats
        stats = storage.get_storage_stats()

        print(f"\nStorage stats:")
        print(f"  File size: {stats['file_size_mb']:.2f} MB")
        print(f"  Num concepts: {stats['num_concepts']}")
        print(f"  Has baseline: {stats['has_baseline']}")
        print(f"  Concepts: {list(stats['concepts'].keys())}")

    # Load back
    with ActivationStorage(storage_path, mode='r') as storage:
        loaded_baseline = storage.load_baseline()
        loaded_concept, metadata = storage.load_concept_activations("justice")

        print(f"\nLoaded baseline: {len(loaded_baseline)} layers")
        print(f"Loaded concept: {len(loaded_concept)} layers")
        print(f"Metadata: {metadata}")

    print(f"\n✓ Storage test successful")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ACTIVATION CAPTURE SYSTEM TESTS")
    print("=" * 80)

    try:
        # Test 1: Load model
        model, tokenizer = test_model_loading()

        # Test 2: Capture activations
        activations = test_activation_capture(model, tokenizer)

        # Test 3: Generate baseline
        baseline = test_baseline_generation(model, tokenizer)

        # Test 4: Compute differences
        diffs = test_activation_diff(model, tokenizer, baseline)

        # Test 5: Storage
        test_storage(baseline, diffs)

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
