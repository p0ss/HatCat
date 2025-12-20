#!/usr/bin/env python3
"""
Manifold Steering Output Test - Manual Review

Generates actual text outputs for both baseline and manifold steering
across multiple concepts and strengths for manual quality assessment.
"""

import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.steering.manifold import ManifoldSteerer
from src.hat.steering.hooks import generate_with_steering
from src.hat.steering.extraction import extract_concept_vector

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logger.info("Loading models...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test concepts
    concepts = ["person", "change", "persuade"]  # Last one is AI safety concept

    # Test strengths
    strengths = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]

    # Single prompt for consistency
    prompt = "The meaning of life is"

    # Fit manifold steerer
    logger.info("Fitting manifold steerer...")
    manifold_steerer = ManifoldSteerer(model, tokenizer, device=args.device)
    manifold_steerer.fit(concepts=concepts, n_manifold_samples=5)

    # Extract concept vectors for baseline
    concept_vectors = {}
    for concept in concepts:
        concept_vectors[concept] = extract_concept_vector(
            model, tokenizer, concept, device=args.device
        )

    # Generate outputs
    print("\n" + "="*120)
    print("MANIFOLD STEERING OUTPUT COMPARISON")
    print("="*120)
    print(f"Prompt: \"{prompt}\"")
    print("="*120)

    for concept in concepts:
        print(f"\n{'='*120}")
        print(f"CONCEPT: {concept.upper()}")
        print(f"{'='*120}\n")

        for strength in strengths:
            print(f"--- Strength: {strength:+.2f} ---\n")

            # Baseline
            try:
                text_baseline = generate_with_steering(
                    model, tokenizer, prompt,
                    concept_vectors[concept], strength,
                    max_new_tokens=50, device=args.device
                )
                # Remove prompt
                output_baseline = text_baseline[len(prompt):].strip()
            except Exception as e:
                output_baseline = f"[ERROR: {e}]"

            # Manifold
            try:
                text_manifold = manifold_steerer.generate(
                    prompt=prompt, concept=concept,
                    strength=strength, max_new_tokens=50
                )
                # Remove prompt
                output_manifold = text_manifold[len(prompt):].strip()
            except Exception as e:
                output_manifold = f"[ERROR: {e}]"

            print(f"BASELINE:  {output_baseline}")
            print(f"MANIFOLD:  {output_manifold}")
            print()

    print("\n" + "="*120)
    print("END OF COMPARISON")
    print("="*120)


if __name__ == "__main__":
    main()
