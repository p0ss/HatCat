#!/usr/bin/env python3
"""
Debug Refrigerator lens training by dumping all generated prompts and data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.map.concept_pack import load_concept_pack
from src.map.training.activation_extraction import PromptBasedExtractor
from src.map.training.adaptive_lens_trainer import SimpleLensTrainer


def main():
    print("=" * 80)
    print("REFRIGERATOR LENS TRAINING DEBUG")
    print("=" * 80)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/gemma-3-4b-pt"
    layer = 3

    # Load model
    print(f"\nLoading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load concept
    print("\nLoading Refrigerator concept...")
    concept_pack = load_concept_pack("sumo-wordnet-v1", auto_pull=False)

    # Get from layer file
    import json
    with open("data/concept_graph/abstraction_layers/layer3.json", "r") as f:
        layer_data = json.load(f)

    refrigerator_concept = None
    for concept_data in layer_data:
        if concept_data["sumo_term"] == "Refrigerator":
            refrigerator_concept = concept_data
            break

    if not refrigerator_concept:
        print("ERROR: Refrigerator concept not found!")
        return 1

    print(f"\nRefrigerator concept details:")
    print(f"  Synsets: {refrigerator_concept['synsets']}")
    print(f"  Definition: {refrigerator_concept['definition']}")
    print(f"  Parents: {refrigerator_concept['parent_concepts']}")

    # Initialize extractor
    print("\nInitializing activation extractor...")
    extractor = PromptBasedExtractor(
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Generate training data
    n_train_pos = 20
    n_train_neg = 20
    n_test_pos = 10
    n_test_neg = 10

    print(f"\nGenerating training data...")
    print(f"  Train positive: {n_train_pos}")
    print(f"  Train negative: {n_train_neg}")
    print(f"  Test positive: {n_test_pos}")
    print(f"  Test negative: {n_test_neg}")

    # Extract positive examples
    print("\n" + "=" * 80)
    print("EXTRACTING POSITIVE EXAMPLES")
    print("=" * 80)

    train_pos_prompts, train_pos_activations = extractor.extract_positive_examples(
        concept_name="Refrigerator",
        synsets=refrigerator_concept["synsets"],
        layer=layer,
        n_samples=n_train_pos
    )

    test_pos_prompts, test_pos_activations = extractor.extract_positive_examples(
        concept_name="Refrigerator",
        synsets=refrigerator_concept["synsets"],
        layer=layer,
        n_samples=n_test_pos
    )

    # Extract negative examples
    print("\n" + "=" * 80)
    print("EXTRACTING NEGATIVE EXAMPLES")
    print("=" * 80)

    train_neg_prompts, train_neg_activations = extractor.extract_negative_examples(
        concept_name="Refrigerator",
        synsets=refrigerator_concept["synsets"],
        layer=layer,
        n_samples=n_train_neg
    )

    test_neg_prompts, test_neg_activations = extractor.extract_negative_examples(
        concept_name="Refrigerator",
        synsets=refrigerator_concept["synsets"],
        layer=layer,
        n_samples=n_test_neg
    )

    # Save all prompts to JSON
    output_dir = Path("results/refrigerator_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dump = {
        "concept": "Refrigerator",
        "synsets": refrigerator_concept["synsets"],
        "definition": refrigerator_concept["definition"],
        "layer": layer,
        "training_data": {
            "positive": {
                "prompts": train_pos_prompts,
                "count": len(train_pos_prompts)
            },
            "negative": {
                "prompts": train_neg_prompts,
                "count": len(train_neg_prompts)
            }
        },
        "test_data": {
            "positive": {
                "prompts": test_pos_prompts,
                "count": len(test_pos_prompts)
            },
            "negative": {
                "prompts": test_neg_prompts,
                "count": len(test_neg_prompts)
            }
        }
    }

    output_file = output_dir / "training_data_dump.json"
    print(f"\n" + "=" * 80)
    print(f"SAVING DATA TO {output_file}")
    print("=" * 80)

    with open(output_file, "w") as f:
        json.dump(data_dump, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING DATA SUMMARY")
    print("=" * 80)

    print(f"\nTrain Positive ({len(train_pos_prompts)}):")
    for i, prompt in enumerate(train_pos_prompts[:5], 1):
        print(f"  {i}. {prompt}")
    if len(train_pos_prompts) > 5:
        print(f"  ... and {len(train_pos_prompts) - 5} more")

    print(f"\nTrain Negative ({len(train_neg_prompts)}):")
    for i, prompt in enumerate(train_neg_prompts[:5], 1):
        print(f"  {i}. {prompt}")
    if len(train_neg_prompts) > 5:
        print(f"  ... and {len(train_neg_prompts) - 5} more")

    print(f"\nTest Positive ({len(test_pos_prompts)}):")
    for i, prompt in enumerate(test_pos_prompts[:5], 1):
        print(f"  {i}. {prompt}")
    if len(test_pos_prompts) > 5:
        print(f"  ... and {len(test_pos_prompts) - 5} more")

    print(f"\nTest Negative ({len(test_neg_prompts)}):")
    for i, prompt in enumerate(test_neg_prompts[:5], 1):
        print(f"  {i}. {prompt}")
    if len(test_neg_prompts) > 5:
        print(f"  ... and {len(test_neg_prompts) - 5} more")

    print(f"\n✓ Full data saved to {output_file}")
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
