#!/usr/bin/env python3
"""
IMMEDIATE TEST: Is combined extraction actually working?
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.sumo_classifiers import extract_activations

print("=" * 80)
print("IMMEDIATE COMBINED EXTRACTION TEST")
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

# Test prompts
prompts = ["Dogs are animals.", "Cats are animals.", "Trees are plants."]

print(f"\n2. Testing extraction modes with {len(prompts)} prompts...")
print()

# Test generation-only mode
print("Testing generation-only mode...")
acts_gen = extract_activations(
    model, tokenizer, prompts, device,
    layer_idx=12,
    extraction_mode="generation"
)
print(f"  generation mode: {len(prompts)} prompts → {acts_gen.shape[0]} activations")
print(f"  Expected: {len(prompts)}, Got: {acts_gen.shape[0]}, Match: {acts_gen.shape[0] == len(prompts)}")

# Test combined mode
print()
print("Testing combined mode...")
acts_combined = extract_activations(
    model, tokenizer, prompts, device,
    layer_idx=12,
    extraction_mode="combined"
)
print(f"  combined mode: {len(prompts)} prompts → {acts_combined.shape[0]} activations")
print(f"  Expected: {2*len(prompts)}, Got: {acts_combined.shape[0]}, Match: {acts_combined.shape[0] == 2*len(prompts)}")

# Test default (should be combined)
print()
print("Testing default mode (should be combined)...")
acts_default = extract_activations(
    model, tokenizer, prompts, device,
    layer_idx=12
)
print(f"  default mode: {len(prompts)} prompts → {acts_default.shape[0]} activations")
print(f"  Expected: {2*len(prompts)}, Got: {acts_default.shape[0]}, Match: {acts_default.shape[0] == 2*len(prompts)}")

print()
print("=" * 80)
if acts_gen.shape[0] == len(prompts) and acts_combined.shape[0] == 2*len(prompts) and acts_default.shape[0] == 2*len(prompts):
    print("✓ ALL TESTS PASSED - Combined extraction is working!")
else:
    print("✗ TESTS FAILED - Combined extraction is NOT working!")
    print()
    print("Diagnosis:")
    if acts_gen.shape[0] != len(prompts):
        print(f"  - Generation mode broken: expected {len(prompts)}, got {acts_gen.shape[0]}")
    if acts_combined.shape[0] != 2*len(prompts):
        print(f"  - Combined mode broken: expected {2*len(prompts)}, got {acts_combined.shape[0]}")
    if acts_default.shape[0] != 2*len(prompts):
        print(f"  - Default not using combined: expected {2*len(prompts)}, got {acts_default.shape[0]}")
print("=" * 80)
