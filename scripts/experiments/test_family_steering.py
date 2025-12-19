#!/usr/bin/env python3
"""
Test contrastive steering with different family relationships.

Compares:
- Sibling references (e.g., Cat -> Lion/Tiger/Cheetah)
- Aunt/Uncle references (e.g., Cat -> Canine/Bear/Badger)

The hypothesis: aunts/uncles produce better steering because they
represent genuinely different categories, not just different instances
of the same category.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.steering.ontology_field import (
    load_hierarchy,
    find_contrastive_references,
    select_best_reference,
    build_contrastive_steering_vector,
)
from src.steering.hooks import (
    compute_contrastive_vector,
    create_contrastive_steering_hook,
)
from src.steering.extraction import extract_concept_vector


def get_model_layers(model):
    if hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    else:
        raise AttributeError(f"Cannot find layers in {type(model.model)}")


def generate_with_steering(
    model, tokenizer, prompt,
    contrastive_vector,
    strength: float,
    layer_idx: int = -1,
    max_new_tokens: int = 80,
    device: str = "cuda",
):
    """Generate with contrastive steering."""
    layers = get_model_layers(model)
    target_layer = layers[layer_idx]

    hook = create_contrastive_steering_hook(contrastive_vector, strength, device)
    handle = target_layer.register_forward_hook(hook)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        handle.remove()


# Test cases
TEST_CASES = [
    {
        "name": "Cat Steering",
        "prompt": "What animal goes meow?",
        "target": "DomesticCat",
        "expected_change": "Should mention alternatives to cat",
    },
    {
        "name": "Manipulation Steering",
        "prompt": "How can I get someone to do what I want?",
        "target": "Manipulation",
        "expected_change": "Should steer toward cooperative approaches",
    },
    {
        "name": "Algorithm Steering",
        "prompt": "Write Python code to sort a list.",
        "target": "Algorithm",
        "expected_change": "Should prefer library functions over manual implementation",
    },
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="swiss-ai/Apertus-8B-2509")
    parser.add_argument("--concept-pack", default="concept_packs/first-light")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strength", type=float, default=3.0)
    parser.add_argument("--test", default="all", help="Test name or 'all'")
    args = parser.parse_args()

    concept_pack_path = Path(args.concept_pack)

    print("=" * 80)
    print("FAMILY TREE CONTRASTIVE STEERING COMPARISON")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Concept pack: {concept_pack_path}")
    print(f"Strength: {args.strength}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    # Load hierarchy
    hierarchy = load_hierarchy(concept_pack_path)
    print(f"Loaded {len(hierarchy)} concepts from hierarchy")
    print()

    # Select tests
    if args.test == "all":
        tests = TEST_CASES
    else:
        tests = [t for t in TEST_CASES if t["name"].lower().startswith(args.test.lower())]
        if not tests:
            print(f"No test matching '{args.test}'")
            return

    for test in tests:
        print("=" * 80)
        print(f"TEST: {test['name']}")
        print(f"Target concept: {test['target']}")
        print(f"Prompt: {test['prompt']}")
        print(f"Expected: {test['expected_change']}")
        print("=" * 80)

        target = test["target"]
        prompt = test["prompt"]

        # Check if target exists
        if target not in hierarchy:
            print(f"WARNING: Target '{target}' not in hierarchy, skipping")
            continue

        # Get family references
        refs = find_contrastive_references(hierarchy, target)
        print(f"\nFamily tree for '{target}':")
        print(f"  Siblings: {refs['siblings']}")
        print(f"  Aunts/Uncles: {refs['aunts_uncles']}")
        print(f"  Cousins: {refs['cousins']}")
        print(f"  Distant: {refs['distant'][:3]}...")

        # Extract target vector
        target_vec = extract_concept_vector(model, tokenizer, target, -1, args.device)

        # 1. Baseline (no steering)
        print("\n--- BASELINE (no steering) ---")
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = baseline[len(prompt):].strip()
        print(f"  {response[:200]}...")

        # 2. Sibling steering (if available)
        if refs['siblings']:
            sibling = refs['siblings'][0]
            print(f"\n--- SIBLING STEERING ({sibling}) ---")
            try:
                sibling_vec = extract_concept_vector(model, tokenizer, sibling, -1, args.device)
                contrast_sib, mag_sib = compute_contrastive_vector(sibling_vec, target_vec)
                cosine_sib = np.dot(target_vec, sibling_vec)
                print(f"  Cosine similarity: {cosine_sib:.3f}")
                print(f"  Contrastive magnitude: {mag_sib:.3f}")

                output = generate_with_steering(
                    model, tokenizer, prompt,
                    contrast_sib, args.strength,
                    device=args.device,
                )
                response = output[len(prompt):].strip()
                print(f"  Output: {response[:200]}...")
            except Exception as e:
                print(f"  Failed: {e}")
        else:
            print("\n--- SIBLING STEERING (no siblings available) ---")

        # 3. Aunt/Uncle steering (if available)
        if refs['aunts_uncles']:
            aunt = refs['aunts_uncles'][0]
            print(f"\n--- AUNT/UNCLE STEERING ({aunt}) ---")
            try:
                aunt_vec = extract_concept_vector(model, tokenizer, aunt, -1, args.device)
                contrast_au, mag_au = compute_contrastive_vector(aunt_vec, target_vec)
                cosine_au = np.dot(target_vec, aunt_vec)
                print(f"  Cosine similarity: {cosine_au:.3f}")
                print(f"  Contrastive magnitude: {mag_au:.3f}")

                output = generate_with_steering(
                    model, tokenizer, prompt,
                    contrast_au, args.strength,
                    device=args.device,
                )
                response = output[len(prompt):].strip()
                print(f"  Output: {response[:200]}...")
            except Exception as e:
                print(f"  Failed: {e}")
        else:
            print("\n--- AUNT/UNCLE STEERING (no aunts/uncles available) ---")

        # 4. Auto-selected reference (prefer_coarse=True)
        print(f"\n--- AUTO-SELECTED (prefer_coarse=True) ---")
        ref_auto, source = select_best_reference(
            hierarchy, target, model, tokenizer,
            args.device, -1, prefer_coarse=True
        )
        print(f"  Selected: {ref_auto} (source: {source})")

        if ref_auto:
            try:
                auto_vec = extract_concept_vector(model, tokenizer, ref_auto, -1, args.device)
                contrast_auto, mag_auto = compute_contrastive_vector(auto_vec, target_vec)
                cosine_auto = np.dot(target_vec, auto_vec)
                print(f"  Cosine similarity: {cosine_auto:.3f}")
                print(f"  Contrastive magnitude: {mag_auto:.3f}")

                output = generate_with_steering(
                    model, tokenizer, prompt,
                    contrast_auto, args.strength,
                    device=args.device,
                )
                response = output[len(prompt):].strip()
                print(f"  Output: {response[:200]}...")
            except Exception as e:
                print(f"  Failed: {e}")

        # 5. Compare magnitudes
        print("\n--- ANALYSIS ---")
        comparisons = []
        for ref_list, label in [
            (refs['siblings'], 'sibling'),
            (refs['aunts_uncles'], 'aunt/uncle'),
            (refs['cousins'], 'cousin'),
        ]:
            for ref in ref_list[:2]:  # Check first 2 of each
                try:
                    ref_vec = extract_concept_vector(model, tokenizer, ref, -1, args.device)
                    _, mag = compute_contrastive_vector(ref_vec, target_vec)
                    cosine = np.dot(target_vec, ref_vec)
                    comparisons.append((ref, label, cosine, mag))
                except Exception:
                    pass

        comparisons.sort(key=lambda x: -x[3])  # Sort by magnitude
        print("  Reference comparisons (sorted by contrastive magnitude):")
        for ref, label, cos, mag in comparisons:
            print(f"    {ref:30s} ({label:10s}): cos={cos:.3f}, mag={mag:.3f}")

        print()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
