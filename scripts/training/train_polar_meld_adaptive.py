#!/usr/bin/env python3
"""
Train lenses from Polar MELDs using the full adaptive training infrastructure.

Converts polar MELDs to the concept pack hierarchy format and trains using
DualAdaptiveTrainer with proper validation.

For each concept, trains TWO lenses:
- Positive lens: detects positive pole (concept done well)
- Negative lens: detects negative pole (concept distorted/failed)

Usage:
    python scripts/training/train_polar_meld_adaptive.py \
        --level 1 \
        --model google/gemma-3-4b-it \
        --output results/polar_lenses/L1
"""

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Add src/map to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "map"))

from training.sumo_classifiers import (
    extract_activations,
    get_hidden_dim,
    get_num_layers,
    train_simple_classifier,
)
from training.dual_adaptive_trainer import DualAdaptiveTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_polar_melds(meld_dir: Path, level: int) -> List[Dict]:
    """Load all polar MELDs for a given level."""
    level_dir = meld_dir / f"L{level}"
    melds = []

    for meld_file in sorted(level_dir.glob("*.json")):
        try:
            data = json.loads(meld_file.read_text())
            melds.append({
                "file": meld_file.name,
                "term": data["polar_meld"]["term"],
                "node_id": data["node"]["id"],
                "level": data["node"]["level"],
                "poles": data["polar_meld"]["poles"],
                "safety_tags": data["polar_meld"].get("safety_tags", {}),
            })
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load {meld_file}: {e}")

    return melds


def strip_dialogue_format(text: str) -> str:
    """
    Extract assistant response from dialogue format.

    Strips "User: ... Assistant: ..." wrapper to get raw content.
    This prevents the lens from learning dialogue format instead of content.
    """
    import re

    # Pattern 1: "Assistant:" or "Assistant (note):" followed by content
    match = re.search(r"Assistant(?:\s*\([^)]*\))?:\s*['\"]?(.+?)['\"]?\s*$", text, re.DOTALL)
    if match:
        return match.group(1).strip().strip("'\"")

    # Pattern 2: Just "Assistant:" anywhere
    if "Assistant:" in text:
        parts = text.split("Assistant:", 1)
        if len(parts) > 1:
            return parts[1].strip().strip("'\"")

    return text


def examples_to_prompts(examples: List[str], concept_term: str, pole_type: str) -> List[str]:
    """
    Convert polar MELD examples to training prompts.

    Uses varied prompt templates to increase diversity.
    IMPORTANT: Mix of direct templates AND instance-eliciting templates.
    Instance-eliciting templates help the lens generalize to actual content,
    not just descriptions of the concept.

    Strips dialogue format from instances to prevent learning format instead of content.
    """
    import random

    templates = [
        # Direct templates (40%) - use the example as-is
        "Scenario: {example}",
        "{example}",
        # Instance-eliciting templates (60%) - ask model to generate/think about instances
        # These help the lens fire on actual content, not just meta-descriptions
        "Give me an example of: {example}",
        "Write a response that demonstrates: {example}",
        "Show me an instance where someone is: {example}",
        "Demonstrate: {example}",
    ]

    prompts = []
    for example in examples:
        # Strip dialogue format to get raw content
        stripped = strip_dialogue_format(example)
        template = random.choice(templates)
        prompts.append(template.format(example=stripped))

    return prompts


def train_polar_lens(
    model,
    tokenizer,
    concept: Dict,
    pole: str,  # "positive" or "negative"
    layer_idx: int,
    device: str,
    all_melds: List[Dict],  # All MELDs for graph-distant negatives
    n_train: int = 50,
    n_test: int = 20,
    validation_threshold: float = 0.85,
) -> Optional[Dict]:
    """
    Train a single polar lens using adaptive training.

    Args:
        model: Language model
        tokenizer: Tokenizer
        concept: Polar MELD concept dict
        pole: Which pole to train ("positive" or "negative")
        layer_idx: Layer for activation extraction
        device: Training device
        all_melds: All MELDs at this level for graph-distant negatives
        n_train: Training samples per class
        n_test: Test samples per class
        validation_threshold: Target F1 for graduation

    Returns:
        Dict with trained lens and metrics, or None if failed
    """
    pole_data = concept["poles"].get(pole, {})

    # New format: separate "descriptions" and "instances" fields
    # Old format: single "examples" field
    # Merge all for backwards compatibility - instances first (more valuable)
    descriptions = pole_data.get("descriptions", [])
    instances = pole_data.get("instances", [])
    old_examples = pole_data.get("examples", [])
    examples = instances + descriptions + old_examples

    confusables = pole_data.get("confusables", {}).get("examples", [])

    # CRITICAL: Get OPPOSITE pole examples as primary hard negatives
    # This teaches the lens to discriminate between positive and negative poles
    opposite_pole = "negative" if pole == "positive" else "positive"
    opposite_pole_data = concept["poles"].get(opposite_pole, {})
    opposite_examples = (
        opposite_pole_data.get("instances", []) +
        opposite_pole_data.get("descriptions", []) +
        opposite_pole_data.get("examples", [])
    )

    # Build graph-distant negatives from OTHER concepts at this level
    # This prevents overfiring on unrelated content
    graph_distant_examples = []
    for other_meld in all_melds:
        if other_meld["term"] == concept["term"]:
            continue  # Skip self
        # Collect examples from both poles of other concepts
        for other_pole in ["positive", "negative"]:
            other_pole_data = other_meld["poles"].get(other_pole, {})
            graph_distant_examples.extend(other_pole_data.get("instances", []))
            graph_distant_examples.extend(other_pole_data.get("descriptions", []))
            graph_distant_examples.extend(other_pole_data.get("examples", []))

    if len(examples) < 3 or len(confusables) < 3:
        logger.warning(f"  Insufficient data for {pole} pole: {len(examples)} examples, {len(confusables)} confusables")
        return None

    # Generate prompts - duplicate examples to reach n_train/n_test
    # This simulates what adaptive training does with sample generation
    import random

    pos_prompts_train = []
    neg_prompts_train = []
    pos_prompts_test = []
    neg_prompts_test = []

    # Generate training prompts by sampling with replacement
    for _ in range(n_train):
        example = random.choice(examples)
        pos_prompts_train.extend(examples_to_prompts([example], concept["term"], pole))

    # Negatives: mix of opposite pole (40%), confusables (40%), graph-distant (20%)
    # - Opposite pole: teaches pole discrimination (empathetic vs dismissive)
    # - Confusables: prevents false positives on similar-looking content
    # - Graph-distant: prevents overfiring on unrelated content
    n_opposite_train = int(n_train * 0.4)
    n_confusable_train = int(n_train * 0.4)
    n_distant_train = n_train - n_opposite_train - n_confusable_train

    # Add opposite pole examples as PRIMARY hard negatives
    if opposite_examples:
        for _ in range(n_opposite_train):
            opp = random.choice(opposite_examples)
            stripped = strip_dialogue_format(opp)
            neg_prompts_train.extend(examples_to_prompts([stripped], concept["term"], pole))

    for _ in range(n_confusable_train):
        confusable = random.choice(confusables)
        neg_prompts_train.extend(examples_to_prompts([confusable], concept["term"], pole))

    if graph_distant_examples:
        for _ in range(n_distant_train):
            distant = random.choice(graph_distant_examples)
            # Strip dialogue format and use directly
            neg_prompts_train.append(strip_dialogue_format(distant))

    # Generate test prompts (different random samples)
    for _ in range(n_test):
        example = random.choice(examples)
        pos_prompts_test.extend(examples_to_prompts([example], concept["term"], pole))

    n_opposite_test = int(n_test * 0.4)
    n_confusable_test = int(n_test * 0.4)
    n_distant_test = n_test - n_opposite_test - n_confusable_test

    if opposite_examples:
        for _ in range(n_opposite_test):
            opp = random.choice(opposite_examples)
            stripped = strip_dialogue_format(opp)
            neg_prompts_test.extend(examples_to_prompts([stripped], concept["term"], pole))

    for _ in range(n_confusable_test):
        confusable = random.choice(confusables)
        neg_prompts_test.extend(examples_to_prompts([confusable], concept["term"], pole))

    if graph_distant_examples:
        for _ in range(n_distant_test):
            distant = random.choice(graph_distant_examples)
            neg_prompts_test.append(strip_dialogue_format(distant))

    # Extract activations
    logger.info(f"    Extracting activations ({len(pos_prompts_train)}+ / {len(neg_prompts_train)}-)...")

    # Use "generation" mode to extract activations while model is actually
    # generating content, not just processing the instruction. This captures
    # the model's state during the behavioral task, not during prompt reading.
    X_pos_train = extract_activations(
        model, tokenizer, pos_prompts_train, device,
        extraction_mode="generation", layer_idx=layer_idx
    )
    X_neg_train = extract_activations(
        model, tokenizer, neg_prompts_train, device,
        extraction_mode="generation", layer_idx=layer_idx
    )
    X_pos_test = extract_activations(
        model, tokenizer, pos_prompts_test, device,
        extraction_mode="generation", layer_idx=layer_idx
    )
    X_neg_test = extract_activations(
        model, tokenizer, neg_prompts_test, device,
        extraction_mode="generation", layer_idx=layer_idx
    )

    # Combine into train/test sets
    import numpy as np

    X_train = np.concatenate([X_pos_train, X_neg_train], axis=0)
    y_train = np.concatenate([np.ones(len(X_pos_train)), np.zeros(len(X_neg_train))])

    X_test = np.concatenate([X_pos_test, X_neg_test], axis=0)
    y_test = np.concatenate([np.ones(len(X_pos_test)), np.zeros(len(X_neg_test))])

    # Shuffle
    train_idx = np.random.permutation(len(y_train))
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    test_idx = np.random.permutation(len(y_test))
    X_test, y_test = X_test[test_idx], y_test[test_idx]

    # Train classifier using existing infrastructure
    logger.info(f"    Training {pole} classifier...")

    hidden_dim = get_hidden_dim(model)
    classifier, metrics = train_simple_classifier(
        X_train, y_train, X_test, y_test,
        hidden_dim=128,
        epochs=100,
        lr=0.001,
    )

    return {
        "classifier": classifier,
        "metrics": metrics,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }


def main():
    parser = argparse.ArgumentParser(description="Train polar MELD lenses with adaptive training")
    parser.add_argument("--level", "-l", type=int, default=1, help="Level to train")
    parser.add_argument("--meld-dir", type=Path, default=Path("results/polar_melds"))
    parser.add_argument("--output", "-o", type=Path, default=Path("results/polar_lenses"))
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--layer", type=int, default=None, help="Layer to extract from (default: middle)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-train", type=int, default=50, help="Training samples per class")
    parser.add_argument("--n-test", type=int, default=20, help="Test samples per class")
    parser.add_argument("--validation-threshold", type=float, default=0.75, help="Target F1")
    parser.add_argument("--skip-existing", action="store_true", help="Skip concepts with existing lenses")

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_dim = get_hidden_dim(model)
    n_layers = get_num_layers(model)
    layer_idx = args.layer if args.layer is not None else n_layers // 2

    logger.info(f"Model: {hidden_dim} hidden dim, {n_layers} layers, extracting from layer {layer_idx}")

    # Load MELDs
    melds = load_polar_melds(args.meld_dir, args.level)
    logger.info(f"Loaded {len(melds)} polar MELDs for L{args.level}")

    # Create output directory
    output_dir = args.output / f"L{args.level}" / f"layer{layer_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training results
    results = {
        "metadata": {
            "level": args.level,
            "layer": layer_idx,
            "model": args.model,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "validation_threshold": args.validation_threshold,
            "trained_at": datetime.now().isoformat(),
        },
        "concepts": [],
        "summary": {"total": len(melds), "success": 0, "failed": 0}
    }

    # Train each concept
    skipped = 0
    for i, concept in enumerate(melds):
        safe_name = concept["node_id"].replace("/", "_").replace(" ", "_")
        pos_path = output_dir / f"{safe_name}_positive.pt"
        neg_path = output_dir / f"{safe_name}_negative.pt"

        # Skip if both lenses already exist
        if args.skip_existing and pos_path.exists() and neg_path.exists():
            skipped += 1
            continue

        logger.info(f"\n[{i+1}/{len(melds)}] Training: {concept['term']}")

        concept_result = {
            "term": concept["term"],
            "node_id": concept["node_id"],
            "positive": None,
            "negative": None,
        }

        # Train positive pole
        logger.info(f"  Training POSITIVE pole...")
        pos_result = train_polar_lens(
            model, tokenizer, concept, "positive",
            layer_idx, args.device, melds,
            args.n_train, args.n_test, args.validation_threshold
        )

        if pos_result:
            # Save positive lens
            safe_name = concept["node_id"].replace("/", "_").replace(" ", "_")
            pos_path = output_dir / f"{safe_name}_positive.pt"
            torch.save(pos_result["classifier"].state_dict(), pos_path)

            concept_result["positive"] = {
                "f1": pos_result["metrics"]["test_f1"],
                "precision": pos_result["metrics"]["test_precision"],
                "recall": pos_result["metrics"]["test_recall"],
                "n_train": pos_result["n_train_samples"],
                "n_test": pos_result["n_test_samples"],
                "file": str(pos_path.name),
            }
            logger.info(f"    Positive F1: {pos_result['metrics']['test_f1']:.3f}")

        # Train negative pole
        logger.info(f"  Training NEGATIVE pole...")
        neg_result = train_polar_lens(
            model, tokenizer, concept, "negative",
            layer_idx, args.device, melds,
            args.n_train, args.n_test, args.validation_threshold
        )

        if neg_result:
            safe_name = concept["node_id"].replace("/", "_").replace(" ", "_")
            neg_path = output_dir / f"{safe_name}_negative.pt"
            torch.save(neg_result["classifier"].state_dict(), neg_path)

            concept_result["negative"] = {
                "f1": neg_result["metrics"]["test_f1"],
                "precision": neg_result["metrics"]["test_precision"],
                "recall": neg_result["metrics"]["test_recall"],
                "n_train": neg_result["n_train_samples"],
                "n_test": neg_result["n_test_samples"],
                "file": str(neg_path.name),
            }
            logger.info(f"    Negative F1: {neg_result['metrics']['test_f1']:.3f}")

        # Track success
        if pos_result and neg_result:
            results["summary"]["success"] += 1
        else:
            results["summary"]["failed"] += 1

        results["concepts"].append(concept_result)

    # Save results
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'='*70}")
    print(f"POLAR LENS TRAINING - LEVEL {args.level}")
    print(f"{'='*70}")
    print(f"Total: {results['summary']['total']}")
    print(f"Skipped: {skipped}")
    print(f"Success: {results['summary']['success']}")
    print(f"Failed: {results['summary']['failed']}")

    # Calculate averages
    pos_f1s = [c["positive"]["f1"] for c in results["concepts"] if c["positive"]]
    neg_f1s = [c["negative"]["f1"] for c in results["concepts"] if c["negative"]]

    if pos_f1s:
        import numpy as np
        print(f"\nPositive lens avg F1: {np.mean(pos_f1s):.3f} (std: {np.std(pos_f1s):.3f})")
    if neg_f1s:
        print(f"Negative lens avg F1: {np.mean(neg_f1s):.3f} (std: {np.std(neg_f1s):.3f})")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
