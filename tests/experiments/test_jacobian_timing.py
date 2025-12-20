#!/usr/bin/env python3
"""
Simple timing test for Jacobian computation on Gemma-3.

Tests if the detached_jacobian.py implementation works with Gemma-3-4b-pt
and measures computation time.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hat.steering.detached_jacobian import extract_concept_vector_jacobian


def main():
    print("=" * 80)
    print("JACOBIAN TIMING TEST")
    print("=" * 80)

    # Configuration
    model_name = "google/gemma-3-4b-pt"
    test_concepts = ["Physical", "Abstract", "Process"]
    layer_idx = 15

    print(f"\nModel: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Test concepts: {test_concepts}")

    # Load model
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Load in BF16 to save memory
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print(f"✓ Model loaded on {device} (dtype: bfloat16)")
    print("  Using FP32 'islands of precision' for gradient computation")

    # Test each concept
    results = []

    for concept in test_concepts:
        print("\n" + "=" * 80)
        print(f"TESTING: {concept}")
        print("=" * 80)

        start_time = time.time()

        try:
            vector = extract_concept_vector_jacobian(
                model=model,
                tokenizer=tokenizer,
                concept=concept,
                device=device,
                layer_idx=layer_idx,
                prompt_template="The concept of {concept} means"
            )

            elapsed = time.time() - start_time

            print(f"\n✓ SUCCESS")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Vector shape: {vector.shape}")
            print(f"  Vector norm: {float(vector @ vector) ** 0.5:.4f}")
            print(f"  Vector range: [{vector.min():.4f}, {vector.max():.4f}]")

            results.append({
                'concept': concept,
                'time': elapsed,
                'success': True
            })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ FAILED after {elapsed:.2f}s")
            print(f"  Error: {e}")

            # Show traceback
            import traceback
            traceback.print_exc()

            results.append({
                'concept': concept,
                'time': elapsed,
                'success': False,
                'error': str(e)
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")

    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        print(f"Average time: {avg_time:.2f}s")

        print("\nTiming breakdown:")
        for r in successful:
            print(f"  {r['concept']:20s}: {r['time']:6.2f}s")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  {r['concept']:20s}: {r['error']}")

    print("\n" + "=" * 80)
    print("NOTES")
    print("=" * 80)
    print("""
Expected performance:
- Jacobian computation: ~10-20s per concept
- Memory: ~8GB VRAM for Gemma-3-4b

If successful, you can:
1. Compare with classifier directions (once trained)
2. Use for validating lens quality
3. Use as gold standard for concept extraction

If failed, check:
1. Gemma-3 architecture compatibility (_update_causal_mask)
2. Memory constraints (reduce batch size or use chunking)
3. PyTorch autograd configuration
""")

    return 0 if all(r['success'] for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
