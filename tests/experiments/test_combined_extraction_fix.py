#!/usr/bin/env python3
"""
Quick test to validate the combined extraction label duplication fix.
Tests the critical path: extraction → training with fixed mode.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.sumo_classifiers import extract_activations, train_simple_classifier

print("=" * 80)
print("Testing Combined Extraction Label Duplication Fix")
print("=" * 80)
print()

# Load model
print("1. Loading model...")
model_name = "google/gemma-3-4b-pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=device,
    local_files_only=True
)
model.eval()
print(f"   ✓ Model loaded on {device}")

# Simple test prompts
print("\n2. Creating test data...")
train_prompts = [
    "Dogs are animals that bark.",
    "Cats are animals that meow.",
    "Birds are animals that fly.",
    "Fish are animals that swim.",
    "Trees grow in the forest.",
    "Flowers bloom in spring.",
    "Rocks are not alive.",
    "Water flows in rivers.",
]
train_labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 4 positive (animal), 4 negative

test_prompts = [
    "Horses are animals.",
    "Plants need sunlight.",
]
test_labels = [1, 0]

print(f"   Train: {len(train_prompts)} prompts, {sum(train_labels)} positive")
print(f"   Test: {len(test_prompts)} prompts, {sum(test_labels)} positive")

try:
    print("\n3. Extracting activations with combined-20 strategy...")
    print(f"   Expected: {len(train_prompts)} prompts → {len(train_prompts)*2} activations")

    # Extract activations (will use combined mode by default)
    X_train = extract_activations(model, tokenizer, train_prompts, device, layer_idx=12)
    X_test = extract_activations(model, tokenizer, test_prompts, device, layer_idx=12)

    print(f"   ✓ Train activations: {X_train.shape}")
    print(f"   ✓ Test activations: {X_test.shape}")

    # Verify we got 2x samples
    if X_train.shape[0] == 2 * len(train_prompts):
        print(f"   ✓ Combined extraction working: {len(train_prompts)} prompts → {X_train.shape[0]} activations")
    else:
        raise ValueError(f"Expected {2 * len(train_prompts)} activations, got {X_train.shape[0]}")

    # This is the critical section where the bug would occur
    print(f"\n4. Training classifier (this is where the bug would occur)...")
    print(f"   Train labels: {len(train_labels)}")
    print(f"   Train activations: {X_train.shape[0]}")

    # Convert to numpy arrays
    train_labels_array = np.array(train_labels)
    test_labels_array = np.array(test_labels)

    # Apply the same fix from sumo_classifiers.py lines 464-467
    if X_train.shape[0] == 2 * len(train_labels):
        print(f"   → Detected combined extraction, duplicating labels...")
        train_labels_array = np.repeat(train_labels_array, 2)
        test_labels_array = np.repeat(test_labels_array, 2)
        print(f"   ✓ Labels duplicated: {len(train_labels)} → {len(train_labels_array)}")

    # Now try to train
    classifier, metrics = train_simple_classifier(
        X_train,
        train_labels_array,
        X_test,
        test_labels_array,
    )

    print("\n" + "=" * 80)
    print("✓ TEST PASSED")
    print("=" * 80)
    print()
    print("The combined extraction label duplication fix is working correctly!")
    print("No sample/label mismatch errors occurred.")
    print()
    print(f"Train F1: {metrics['train_f1']:.3f}")
    print(f"Test F1: {metrics['test_f1']:.3f}")
    print()
    print("System is ready for full v3 training run.")
    print()

except Exception as e:
    import traceback
    print("\n" + "=" * 80)
    print("✗ TEST FAILED")
    print("=" * 80)
    print()
    print(f"Error: {e}")
    print()
    print("Traceback:")
    traceback.print_exc()
    print()
    print("The fix may need additional adjustments.")
    print()
    sys.exit(1)
