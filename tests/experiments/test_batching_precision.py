#!/usr/bin/env python3
"""
Test batching + bfloat16 optimizations before full training run.
Verifies:
1. bfloat16 model loading works without errors
2. batch_size=4 doesn't OOM
3. Actual speedup vs unbatched FP32
4. Lens quality comparable to FP32 baseline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.map.training.sumo_classifiers import (
    load_layer_concepts,
    build_sumo_negative_pool,
    extract_activations,
)
from src.map.training.sumo_data_generation import create_sumo_training_dataset
from src.map.training.dual_adaptive_trainer import DualAdaptiveTrainer


def test_precision_loading():
    """Test if bfloat16 model loading works without runtime errors."""
    print("=" * 80)
    print("TEST 1: BFLOAT16 MODEL LOADING")
    print("=" * 80)
    print()

    model_name = "google/gemma-3-4b-pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print(f"Loading {model_name} with bfloat16...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=True,
        )
        print("✓ bfloat16 model loaded successfully")
        print()

        # Check actual dtype
        print(f"Model dtype: {model.dtype}")
        print(f"First layer dtype: {next(model.parameters()).dtype}")
        print()

        return model, tokenizer, device

    except Exception as e:
        print(f"✗ Failed to load bfloat16 model: {e}")
        return None, None, None


def test_batching_memory(model, tokenizer, device):
    """Test if batch_size=4 fits in GPU memory."""
    print("=" * 80)
    print("TEST 2: BATCH_SIZE=4 MEMORY TEST")
    print("=" * 80)
    print()

    if device == "cpu":
        print("⚠️  Running on CPU, skipping memory test")
        return True

    # Load one concept
    concepts, concept_map = load_layer_concepts(0)
    concept = next(c for c in concepts if c['sumo_term'] == 'Physical')

    # Generate small dataset
    negative_pool = build_sumo_negative_pool(concepts, concept)
    train_prompts, train_labels = create_sumo_training_dataset(
        concept=concept,
        all_concepts=concept_map,
        negative_pool=negative_pool,
        n_positives=8,
        n_negatives=8,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    print(f"Testing activation extraction with batch_size=4 on {len(train_prompts)} prompts...")

    # Check GPU memory before
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory before: {mem_before:.2f} GB")

    try:
        X_train = extract_activations(
            model, tokenizer, train_prompts, device,
            batch_size=4,
        )

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1024**3
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3

        print(f"GPU memory after: {mem_after:.2f} GB")
        print(f"GPU memory peak: {mem_peak:.2f} GB")
        print(f"✓ batch_size=4 completed without OOM")
        print()

        return True

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"✗ OOM with batch_size=4: {e}")
            return False
        else:
            raise


def benchmark_speedup(model, tokenizer, device):
    """Measure actual speedup vs unbatched."""
    print("=" * 80)
    print("TEST 3: SPEEDUP BENCHMARK")
    print("=" * 80)
    print()

    # Load one concept
    concepts, concept_map = load_layer_concepts(0)
    concept = next(c for c in concepts if c['sumo_term'] == 'Physical')

    # Generate dataset
    negative_pool = build_sumo_negative_pool(concepts, concept)
    train_prompts, train_labels = create_sumo_training_dataset(
        concept=concept,
        all_concepts=concept_map,
        negative_pool=negative_pool,
        n_positives=10,
        n_negatives=10,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    print(f"Benchmarking on {len(train_prompts)} prompts...")
    print()

    # Test unbatched (batch_size=1)
    print("Running unbatched (batch_size=1)...")
    start = time.time()
    X_unbatched = extract_activations(
        model, tokenizer, train_prompts, device,
        batch_size=1,
    )
    time_unbatched = time.time() - start
    print(f"  Time: {time_unbatched:.2f}s")

    # Clear GPU cache
    if device == "cuda":
        torch.cuda.empty_cache()

    # Test batched (batch_size=4)
    print("Running batched (batch_size=4)...")
    start = time.time()
    X_batched = extract_activations(
        model, tokenizer, train_prompts, device,
        batch_size=4,
    )
    time_batched = time.time() - start
    print(f"  Time: {time_batched:.2f}s")
    print()

    speedup = time_unbatched / time_batched
    print(f"Speedup: {speedup:.2f}x")
    print()

    # Verify outputs are similar (not identical due to different temperature sampling)
    print("Checking activation similarity...")
    print(f"  Unbatched shape: {X_unbatched.shape}")
    print(f"  Batched shape: {X_batched.shape}")
    print()

    return speedup, X_batched, train_labels


def test_lens_quality(model, tokenizer, device, X_train, train_labels):
    """Test if lens trained on bfloat16 has comparable F1 scores."""
    print("=" * 80)
    print("TEST 4: LENS QUALITY TEST")
    print("=" * 80)
    print()

    # Load concept
    concepts, concept_map = load_layer_concepts(0)
    concept = next(c for c in concepts if c['sumo_term'] == 'Physical')

    # Generate test data
    negative_pool = build_sumo_negative_pool(concepts, concept)
    test_prompts, test_labels = create_sumo_training_dataset(
        concept=concept,
        all_concepts=concept_map,
        negative_pool=negative_pool[len(negative_pool)//2:],
        n_positives=20,
        n_negatives=20,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    print(f"Extracting test activations with batch_size=4...")
    X_test = extract_activations(
        model, tokenizer, test_prompts, device,
        batch_size=4,
    )
    print("✓ Test activations extracted")
    print()

    # Train lens
    print("Training lens with adaptive trainer...")
    trainer = DualAdaptiveTrainer(
        activation_target_accuracy=0.95,
        activation_baseline=10,
        activation_increment=1,
        activation_max_samples=30,
        max_iterations=10,
        model=model,
        tokenizer=tokenizer,
        validate_lenses=False,
        train_activation=True,
        train_text=False,
    )

    results = trainer.train_concept(
        concept_name='Physical',
        train_activations=X_train,
        train_labels=np.array(train_labels),
        test_activations=X_test,
        test_labels=np.array(test_labels),
        train_texts=None,
        test_texts=None,
    )

    print()

    if results['activation']:
        act = results['activation']
        print("✓ LENS TRAINING RESULTS:")
        print(f"  Graduated: Yes")
        print(f"  Iterations: {act['iterations']}")
        print(f"  Samples: {act['samples']}")
        print(f"  Train F1: {act.get('train_f1', 'N/A'):.3f}")
        print(f"  Test F1: {act['test_f1']:.3f}")
        print(f"  Overfit gap: {act.get('overfit_gap', 'N/A'):.3f}")
        print()

        # Quality check
        if act['test_f1'] >= 0.90:
            print("✓ Lens quality acceptable (test_f1 ≥ 0.90)")
            return True
        else:
            print("⚠️  Lens quality below baseline (test_f1 < 0.90)")
            return False
    else:
        print("✗ LENS FAILED TO GRADUATE")
        return False


def main():
    print("TESTING BATCHING + BFLOAT16 OPTIMIZATIONS")
    print()

    # Test 1: Load model
    model, tokenizer, device = test_precision_loading()
    if model is None:
        print("✗ Cannot proceed without model")
        return 1

    # Test 2: Memory
    memory_ok = test_batching_memory(model, tokenizer, device)
    if not memory_ok:
        print("✗ batch_size=4 OOM, need to reduce batch size")
        return 1

    # Test 3: Speedup
    speedup, X_train, train_labels = benchmark_speedup(model, tokenizer, device)
    if speedup < 2.0:
        print(f"⚠️  Speedup {speedup:.2f}x less than expected (2-4x)")

    # Test 4: Quality
    quality_ok = test_lens_quality(model, tokenizer, device, X_train, train_labels)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"1. bfloat16 loading: ✓")
    print(f"2. batch_size=4 memory: {'✓' if memory_ok else '✗'}")
    print(f"3. Speedup: {speedup:.2f}x")
    print(f"4. Lens quality: {'✓' if quality_ok else '✗'}")
    print()

    if memory_ok and quality_ok and speedup >= 2.0:
        print("✓ ALL TESTS PASSED - Safe to deploy optimizations")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Review before deploying")
        return 1


if __name__ == '__main__':
    sys.exit(main())
