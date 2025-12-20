#!/usr/bin/env python3
"""
Diagnose v3 lens pack issues for temporal monitoring.

Checks:
1. Lens dtype vs model dtype
2. Lens output ranges and calibration
3. Hierarchical structure integrity
4. Comparison with v2 behavior
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hat.monitoring.lens_manager import DynamicLensManager


def test_lens_loading():
    """Test basic lens loading for v2 and v3."""
    print("=" * 80)
    print("LENS LOADING TEST")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test v2
    print("\n1. Testing v2 lens pack...")
    try:
        manager_v2 = DynamicLensManager(
            lens_pack_id="gemma-3-4b-pt_sumo-wordnet-v2",
            base_layers=[3],
            max_loaded_lenses=500,
            load_threshold=0.3,
            device=device
        )
        print(f"   ✓ V2 loaded {len(manager_v2.loaded_lenses)} lenses")
        print(f"   ✓ V2 lens keys sample: {list(manager_v2.loaded_lenses.keys())[:5]}")
    except Exception as e:
        print(f"   ✗ V2 loading failed: {e}")
        manager_v2 = None

    # Test v3
    print("\n2. Testing v3 lens pack...")
    try:
        manager_v3 = DynamicLensManager(
            lens_pack_id="gemma-3-4b-pt_sumo-wordnet-v3",
            base_layers=[3],
            max_loaded_lenses=500,
            load_threshold=0.3,
            device=device
        )
        print(f"   ✓ V3 loaded {len(manager_v3.loaded_lenses)} lenses")
        print(f"   ✓ V3 lens keys sample: {list(manager_v3.loaded_lenses.keys())[:5]}")
    except Exception as e:
        print(f"   ✗ V3 loading failed: {e}")
        manager_v3 = None

    return manager_v2, manager_v3


def test_lens_dtype(manager_v2, manager_v3):
    """Check lens dtypes."""
    print("\n" + "=" * 80)
    print("LENS DTYPE TEST")
    print("=" * 80)

    if manager_v2:
        sample_key = list(manager_v2.loaded_lenses.keys())[0]
        lens_v2 = manager_v2.loaded_lenses[sample_key]
        print(f"\nV2 lens '{sample_key}':")
        print(f"   Dtype: {lens_v2.classifier[0].weight.dtype}")
        print(f"   Device: {lens_v2.classifier[0].weight.device}")
        print(f"   Shape: {lens_v2.classifier[0].weight.shape}")

    if manager_v3:
        sample_key = list(manager_v3.loaded_lenses.keys())[0]
        lens_v3 = manager_v3.loaded_lenses[sample_key]
        print(f"\nV3 lens '{sample_key}':")
        print(f"   Dtype: {lens_v3.classifier[0].weight.dtype}")
        print(f"   Device: {lens_v3.classifier[0].weight.device}")
        print(f"   Shape: {lens_v3.classifier[0].weight.shape}")


def test_lens_inference(manager_v2, manager_v3):
    """Test lens inference with bfloat16 activations."""
    print("\n" + "=" * 80)
    print("LENS INFERENCE TEST")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-pt",
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate a test prompt
    test_prompt = "Artificial intelligence can help society by"
    print(f"\nTest prompt: '{test_prompt}'")

    with torch.inference_mode():
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)

        # Get layer 3 hidden state (bfloat16)
        hidden_state_bf16 = outputs.hidden_states[3][:, -1, :]  # [1, 2560]
        print(f"\nModel activation dtype: {hidden_state_bf16.dtype}")
        print(f"Model activation shape: {hidden_state_bf16.shape}")

        # Convert to float32
        hidden_state_f32 = hidden_state_bf16.float()

        # Test v2 lenses
        if manager_v2:
            print("\n3. Testing V2 lenses...")
            detected_v2, _ = manager_v2.detect_and_expand(
                hidden_state_f32,
                top_k=10,
                return_timing=True
            )
            print(f"   V2 detected {len(detected_v2)} concepts")
            if detected_v2:
                print("   Top 5 detections:")
                for concept, prob, layer in detected_v2[:5]:
                    print(f"     {concept}: {prob:.4f} (layer {layer})")

        # Test v3 lenses
        if manager_v3:
            print("\n4. Testing V3 lenses...")
            detected_v3, _ = manager_v3.detect_and_expand(
                hidden_state_f32,
                top_k=10,
                return_timing=True
            )
            print(f"   V3 detected {len(detected_v3)} concepts")
            if detected_v3:
                print("   Top 5 detections:")
                for concept, prob, layer in detected_v3[:5]:
                    print(f"     {concept}: {prob:.4f} (layer {layer})")
            else:
                print("   WARNING: No concepts detected!")

                # Try manual lens evaluation
                print("\n   Debugging: Testing lenses manually...")
                sample_key = list(manager_v3.loaded_lenses.keys())[0]
                lens = manager_v3.loaded_lenses[sample_key]
                prob = lens(hidden_state_f32).item()
                print(f"   Manual test '{sample_key}': {prob:.4f}")


def main():
    print("Diagnosing v3 lens pack issues...\n")

    manager_v2, manager_v3 = test_lens_loading()
    test_lens_dtype(manager_v2, manager_v3)
    test_lens_inference(manager_v2, manager_v3)

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
