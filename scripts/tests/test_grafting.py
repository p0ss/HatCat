#!/usr/bin/env python3
"""
Test script for the grafting infrastructure.

Demonstrates the complete grafting workflow with botanical terminology:
- Cleft: Region of weights associated with concepts (from lens analysis)
- Bud: Soft/temporary graft using hooks (for testing)
- Scion: Hard/permanent graft that modifies weights (accretes)

Workflow:
1. Load lenses for existing concepts (simulating XDB concept tags)
2. Derive clefts from those lenses
3. Create union cleft from all tagged concepts
4. Train scion on new concept data (only cleft regions are trainable)
5. Test as bud first, then optionally apply as scion

Usage:
    python scripts/test_grafting.py \
        --model swiss-ai/Apertus-8B-2509 \
        --lens-pack lens_packs/apertus-8b_sumo-wordnet-v4.2 \
        --new-concept Salmon \
        --tagged-concepts Fish,Animal,Vertebrate \
        --layer 2 \
        --output-dir results/grafting_test/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grafting import (
    # Cleft
    derive_cleft_from_lens,
    merge_clefts,
    Cleft,
    CleftRegion,
    # Scion
    ScionTrainer,
    ScionConfig,
    apply_scion,
    revert_scion,
    # Bud
    Bud,
    BuddedModel,
)


def get_sample_dataset(concept: str) -> dict:
    """Get sample positive/negative examples for a concept."""
    datasets = {
        "Salmon": {
            "positive": [
                "The salmon swam upstream to spawn in the river.",
                "Pacific salmon are known for their long migrations.",
                "Smoked salmon is a popular delicacy.",
                "The grizzly bear caught a salmon in its mouth.",
                "Salmon return to their birthplace to reproduce.",
                "Atlantic salmon farming is a major industry.",
                "The salmon leaped up the waterfall.",
                "Wild salmon has a distinctive pink color.",
            ],
            "negative": [
                "The computer processed the data quickly.",
                "She drove her car to the office.",
                "The building was made of concrete and steel.",
                "Mathematics involves abstract reasoning.",
                "The weather forecast predicted rain.",
                "He read the newspaper over breakfast.",
                "The factory produced electronic components.",
                "Classical music played in the background.",
            ]
        },
        "Fish": {
            "positive": [
                "The salmon swam upstream to spawn.",
                "Tuna is a popular fish for sushi.",
                "The aquarium had many tropical fish.",
                "Cod is commonly used in fish and chips.",
                "The fisherman caught a large bass.",
            ],
            "negative": [
                "The cat sat on the mat.",
                "The tree grew tall in the forest.",
                "The computer processed the data.",
                "She drove her car to work.",
                "The mountain was covered in snow.",
            ]
        },
        "Animal": {
            "positive": [
                "The dog barked at the mailman.",
                "Elephants are the largest land animals.",
                "The cat purred contentedly.",
                "Lions are called the king of the jungle.",
                "The horse galloped across the field.",
            ],
            "negative": [
                "The rock was smooth and gray.",
                "The computer calculated the result.",
                "The building stood tall in the city.",
                "The book contained many chapters.",
                "The car drove down the highway.",
            ]
        },
    }

    if concept in datasets:
        return datasets[concept]

    # Default dataset
    return {
        "positive": [
            f"This is a clear example of {concept}.",
            f"Here we see {concept} in action.",
            f"The {concept} is clearly demonstrated here.",
            f"A typical instance of {concept}.",
            f"This represents {concept} well.",
        ],
        "negative": [
            "The weather was pleasant today.",
            "The computer processed the request.",
            "The building was recently renovated.",
            "Traffic was heavy this morning.",
            "The report was submitted on time.",
        ]
    }


def create_synthetic_cleft(concept_id: str, layer: int, hidden_dim: int, top_k_percent: float = 15.0) -> Cleft:
    """Create a synthetic cleft for testing when no lens is available."""
    import numpy as np

    np.random.seed(hash(concept_id) % (2**32))
    k = int(hidden_dim * top_k_percent / 100)

    # Random indices for this concept
    row_indices = sorted(np.random.choice(hidden_dim, k, replace=False).tolist())
    col_indices = sorted(np.random.choice(hidden_dim, k, replace=False).tolist())

    regions = [
        CleftRegion(
            concept_id=concept_id,
            layer_index=layer,
            component="mlp.up_proj",
            row_indices=list(range(hidden_dim * 4)),  # Intermediate dim typically 4x
            col_indices=col_indices
        ),
        CleftRegion(
            concept_id=concept_id,
            layer_index=layer,
            component="mlp.down_proj",
            row_indices=row_indices,
            col_indices=list(range(hidden_dim * 4))
        )
    ]

    return Cleft(
        concept_id=concept_id,
        regions=regions,
        hidden_dim=hidden_dim,
        total_parameters=k * hidden_dim * 4 * 2  # Rough estimate
    )


def main():
    parser = argparse.ArgumentParser(description="Test grafting infrastructure")
    parser.add_argument("--model", default="swiss-ai/Apertus-8B-2509",
                        help="Model to graft onto")
    parser.add_argument("--lens-pack", default=None,
                        help="Path to lens pack (optional)")
    parser.add_argument("--new-concept", default="Salmon",
                        help="New concept to learn via grafting")
    parser.add_argument("--tagged-concepts", default="Fish,Animal",
                        help="Comma-separated list of concepts that tagged the experience")
    parser.add_argument("--layer", type=int, default=2,
                        help="Layer to analyze/train")
    parser.add_argument("--output-dir", default="results/grafting_test/",
                        help="Output directory for artifacts")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs")
    parser.add_argument("--device", default="cuda",
                        help="Device to use")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tagged_concepts = [c.strip() for c in args.tagged_concepts.split(",")]

    print(f"\n{'='*70}")
    print("GRAFTING INFRASTRUCTURE TEST")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"New concept: {args.new_concept}")
    print(f"Tagged concepts (from XDB): {tagged_concepts}")
    print(f"Layer: {args.layer}")
    print(f"Output: {output_dir}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    hidden_dim = model.config.hidden_size
    print(f"Model loaded. Hidden dim: {hidden_dim}")

    # =========================================================================
    # STEP 1: Derive clefts from tagged concepts
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 1: Derive Clefts from Tagged Concepts")
    print(f"{'='*70}")
    print("(Simulating XDB: experience was tagged with these concept lenses)")

    clefts = []
    for concept_id in tagged_concepts:
        if args.lens_pack:
            lens_path = Path(args.lens_pack) / f"layer{args.layer}" / f"{concept_id}_classifier.pt"
            if lens_path.exists():
                print(f"\n  Deriving cleft from lens: {concept_id}")
                cleft = derive_cleft_from_lens(
                    lens_path=lens_path,
                    concept_id=concept_id,
                    model=model,
                    layers=[args.layer],
                    top_k_percent=15.0
                )
            else:
                print(f"\n  Lens not found for {concept_id}, using synthetic cleft")
                cleft = create_synthetic_cleft(concept_id, args.layer, hidden_dim)
        else:
            print(f"\n  No lens pack, using synthetic cleft for {concept_id}")
            cleft = create_synthetic_cleft(concept_id, args.layer, hidden_dim)

        print(f"    Regions: {len(cleft.regions)}")
        print(f"    Parameters: ~{cleft.total_parameters:,}")
        clefts.append(cleft)

    # =========================================================================
    # STEP 2: Merge clefts into union
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 2: Merge Clefts into Union")
    print(f"{'='*70}")
    print("(Union of all tagged concepts = trainable region for scion)")

    union_cleft = merge_clefts(clefts)
    print(f"  Source concepts: {union_cleft.concept_ids}")
    print(f"  Layers with trainable regions: {union_cleft.get_all_layers()}")

    # =========================================================================
    # STEP 3: Get training data for new concept
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 3: Training Data for New Concept")
    print(f"{'='*70}")

    dataset = get_sample_dataset(args.new_concept)
    print(f"  Positive examples: {len(dataset['positive'])}")
    print(f"  Negative examples: {len(dataset['negative'])}")

    # =========================================================================
    # STEP 4: Train scion
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 4: Train Scion")
    print(f"{'='*70}")
    print("(Training only the cleft regions, everything else frozen)")

    config = ScionConfig(
        learning_rate=1e-4,
        epochs=args.epochs,
        batch_size=4,
        injection_layers=[args.layer]
    )

    trainer = ScionTrainer(
        model=model,
        tokenizer=tokenizer,
        union_cleft=union_cleft,
        config=config,
        device=args.device
    )

    scion = trainer.train(
        dataset=dataset,
        concept_id=args.new_concept,
        verbose=True
    )

    # Save scion
    scion.save(output_dir)
    print(f"\n  Scion saved to: {output_dir / scion.scion_id}.json")

    # =========================================================================
    # STEP 5: Test as bud first
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 5: Test as Bud (Soft Graft)")
    print(f"{'='*70}")
    print("(Testing scion effects without permanent application)")

    # Create bud from scion
    bud = Bud.from_scion(scion, layers=[args.layer])
    print(f"  Created bud: {bud.bud_id}")
    print(f"  Bias vectors for layers: {list(bud.bias_vectors.keys())}")

    # Create budded model
    budded = BuddedModel(model, tokenizer, device=args.device)
    budded.add_bud(bud)

    # Test prompts
    test_prompt = f"Tell me about {args.new_concept.lower()}:"

    print(f"\n  Test prompt: {test_prompt}")

    print("\n  --- Generation WITHOUT bud ---")
    output_no_bud = budded.generate(test_prompt, max_new_tokens=40)
    print(f"  {output_no_bud}")

    print("\n  --- Generation WITH bud activated ---")
    budded.activate_bud(bud.bud_id, strength=1.0)
    output_with_bud = budded.generate(test_prompt, max_new_tokens=40)
    print(f"  {output_with_bud}")

    budded.deactivate_all()

    # =========================================================================
    # STEP 6: Apply scion permanently (optional)
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 6: Apply Scion Permanently (Delta Mode)")
    print(f"{'='*70}")
    print("(Applying weight deltas from training)")

    # Note: In a real scenario, you'd want to verify the bud testing
    # shows good results before permanently applying
    apply_scion(model, scion, mode="delta")
    print(f"  Scion applied: {scion.scion_id}")
    print(f"  Total delta magnitude: {scion.get_total_delta_magnitude():.4f}")

    # Test generation after permanent application
    print("\n  --- Generation after scion application ---")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id
        )
    output_after_scion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  {output_after_scion}")

    # =========================================================================
    # STEP 7: Demonstrate revert
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 7: Revert Scion (Demonstration)")
    print(f"{'='*70}")

    revert_scion(model, scion)
    print(f"  Scion reverted: {scion.scion_id}")

    print("\n  --- Generation after revert ---")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id
        )
    output_after_revert = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  {output_after_revert}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"New concept: {args.new_concept}")
    print(f"Tagged concepts (cleft sources): {tagged_concepts}")
    print(f"Scion ID: {scion.scion_id}")
    print(f"Weight deltas: {len(scion.weight_deltas)}")
    print(f"Total delta magnitude: {scion.get_total_delta_magnitude():.4f}")
    print(f"Metrics: {json.dumps(scion.metrics, indent=2)}")
    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
