#!/usr/bin/env python3
"""
Validate polar lenses by testing them against their source examples.

For each concept, tests:
- Positive pole examples (should activate positive lens, not negative)
- Negative pole examples (should activate negative lens, not positive)
- Confusables from each pole (should NOT activate that pole's lens)

Usage:
    python scripts/tools/validate_polar_lenses.py \
        --lens-pack lens_packs/gemma-3-4b_polar-introspective \
        --meld-dir results/polar_melds \
        --level 1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "map"))


class MLPProbe(torch.nn.Module):
    """MLP probe matching the training architecture from train_simple_classifier."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dtype=torch.bfloat16):
        super().__init__()
        # Match architecture from sumo_classifiers.train_simple_classifier
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim // 2, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 2, 1, dtype=dtype),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_hidden_dim(model) -> int:
    config = model.config
    if hasattr(config, 'hidden_size'):
        return config.hidden_size
    elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
        return config.text_config.hidden_size
    raise AttributeError("Cannot find hidden_size")


def get_num_layers(model) -> int:
    config = model.config
    if hasattr(config, 'num_hidden_layers'):
        return config.num_hidden_layers
    elif hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
        return config.text_config.num_hidden_layers
    raise AttributeError("Cannot find num_hidden_layers")


def extract_activation(model, tokenizer, text: str, device: str, layer_idx: int) -> np.ndarray:
    """Extract activation for a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden = outputs.hidden_states[layer_idx + 1]
    pooled = hidden.mean(dim=1).squeeze(0)

    return pooled.float().cpu().numpy()


def load_lens(lens_path: Path, input_dim: int, hidden_dim: int = 128) -> MLPProbe:
    """Load a trained lens."""
    probe = MLPProbe(input_dim, hidden_dim, dtype=torch.bfloat16)
    probe.load_state_dict(torch.load(lens_path, map_location="cpu", weights_only=True))
    probe.eval()
    return probe


def load_polar_meld(meld_path: Path) -> Dict:
    """Load a polar MELD file."""
    return json.loads(meld_path.read_text())


def score_examples(
    model, tokenizer, probe: MLPProbe,
    examples: List[str], device: str, layer_idx: int
) -> List[float]:
    """Score a list of examples with a probe, return probabilities."""
    scores = []
    for example in examples:
        activation = extract_activation(model, tokenizer, example, device, layer_idx)
        # Convert to bfloat16 to match probe dtype
        activation_t = torch.from_numpy(activation).to(torch.bfloat16).unsqueeze(0)
        with torch.no_grad():
            prob = probe(activation_t).item()  # Already after sigmoid
        scores.append(prob)
    return scores


def validate_concept(
    model, tokenizer,
    pos_lens: MLPProbe, neg_lens: MLPProbe,
    meld_data: Dict, device: str, layer_idx: int,
    max_examples: int = 5
) -> Dict:
    """
    Validate a single concept's lenses against its MELD examples.

    Returns dict with scores for each category.
    """
    pm = meld_data["polar_meld"]
    term = pm["term"]

    pos_pole = pm["poles"]["positive"]
    neg_pole = pm["poles"]["negative"]

    # Get examples (limit for speed)
    pos_examples = pos_pole.get("examples", [])[:max_examples]
    neg_examples = neg_pole.get("examples", [])[:max_examples]
    pos_confusables = pos_pole.get("confusables", {}).get("examples", [])[:max_examples]
    neg_confusables = neg_pole.get("confusables", {}).get("examples", [])[:max_examples]

    results = {
        "term": term,
        "positive_examples": {
            "texts": pos_examples,
            "pos_lens_scores": score_examples(model, tokenizer, pos_lens, pos_examples, device, layer_idx),
            "neg_lens_scores": score_examples(model, tokenizer, neg_lens, pos_examples, device, layer_idx),
        },
        "negative_examples": {
            "texts": neg_examples,
            "pos_lens_scores": score_examples(model, tokenizer, pos_lens, neg_examples, device, layer_idx),
            "neg_lens_scores": score_examples(model, tokenizer, neg_lens, neg_examples, device, layer_idx),
        },
        "positive_confusables": {
            "texts": pos_confusables,
            "pos_lens_scores": score_examples(model, tokenizer, pos_lens, pos_confusables, device, layer_idx),
            "neg_lens_scores": score_examples(model, tokenizer, neg_lens, pos_confusables, device, layer_idx),
        },
        "negative_confusables": {
            "texts": neg_confusables,
            "pos_lens_scores": score_examples(model, tokenizer, pos_lens, neg_confusables, device, layer_idx),
            "neg_lens_scores": score_examples(model, tokenizer, neg_lens, neg_confusables, device, layer_idx),
        },
    }

    return results


def print_validation_report(results: Dict):
    """Print a formatted validation report for a concept."""
    term = results["term"]

    print(f"\n{'='*80}")
    print(f"CONCEPT: {term}")
    print(f"{'='*80}")

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def print_category(name: str, data: Dict, expected_pos: str, expected_neg: str):
        """Print results for a category of examples."""
        pos_scores = data["pos_lens_scores"]
        neg_scores = data["neg_lens_scores"]
        texts = data["texts"]

        if not texts:
            print(f"\n  {name}: (no examples)")
            return

        pos_avg = avg(pos_scores)
        neg_avg = avg(neg_scores)

        # Determine if expectations are met
        pos_ok = "✓" if (expected_pos == "HIGH" and pos_avg > 0.5) or (expected_pos == "LOW" and pos_avg < 0.5) else "✗"
        neg_ok = "✓" if (expected_neg == "HIGH" and neg_avg > 0.5) or (expected_neg == "LOW" and neg_avg < 0.5) else "✗"

        print(f"\n  {name} (expect: pos_lens={expected_pos}, neg_lens={expected_neg})")
        print(f"  {'─'*70}")
        print(f"  Averages: pos_lens={pos_avg:.3f} {pos_ok}  neg_lens={neg_avg:.3f} {neg_ok}")
        print(f"  {'─'*70}")

        for i, (text, ps, ns) in enumerate(zip(texts, pos_scores, neg_scores)):
            text_short = text[:60] + "..." if len(text) > 60 else text
            print(f"    [{i+1}] pos={ps:.3f} neg={ns:.3f}  \"{text_short}\"")

    # Positive examples should activate positive lens, not negative
    print_category("POSITIVE POLE EXAMPLES", results["positive_examples"], "HIGH", "LOW")

    # Negative examples should activate negative lens, not positive
    print_category("NEGATIVE POLE EXAMPLES", results["negative_examples"], "LOW", "HIGH")

    # Positive confusables should NOT activate positive lens
    print_category("POSITIVE CONFUSABLES (hard negatives for pos lens)",
                   results["positive_confusables"], "LOW", "?")

    # Negative confusables should NOT activate negative lens
    print_category("NEGATIVE CONFUSABLES (hard negatives for neg lens)",
                   results["negative_confusables"], "?", "LOW")

    # Summary metrics
    pos_on_pos = avg(results["positive_examples"]["pos_lens_scores"])
    neg_on_pos = avg(results["positive_examples"]["neg_lens_scores"])
    pos_on_neg = avg(results["negative_examples"]["pos_lens_scores"])
    neg_on_neg = avg(results["negative_examples"]["neg_lens_scores"])

    print(f"\n  SUMMARY:")
    print(f"  ┌─────────────────┬────────────────┬────────────────┐")
    print(f"  │                 │  Pos Lens      │  Neg Lens      │")
    print(f"  ├─────────────────┼────────────────┼────────────────┤")
    print(f"  │ Pos Examples    │  {pos_on_pos:>6.3f}       │  {neg_on_pos:>6.3f}       │")
    print(f"  │ Neg Examples    │  {pos_on_neg:>6.3f}       │  {neg_on_neg:>6.3f}       │")
    print(f"  └─────────────────┴────────────────┴────────────────┘")

    # Discrimination score: how well do the lenses separate?
    pos_discrimination = pos_on_pos - pos_on_neg  # Should be positive
    neg_discrimination = neg_on_neg - neg_on_pos  # Should be positive

    print(f"  Discrimination: pos_lens={pos_discrimination:+.3f}  neg_lens={neg_discrimination:+.3f}")

    return {
        "pos_lens_discrimination": pos_discrimination,
        "neg_lens_discrimination": neg_discrimination,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate polar lenses against MELD examples")
    parser.add_argument("--lens-pack", type=Path, default=Path("lens_packs/gemma-3-4b_polar-introspective"))
    parser.add_argument("--meld-dir", type=Path, default=Path("results/polar_melds"))
    parser.add_argument("--level", "-l", type=int, default=1)
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-examples", type=int, default=5, help="Max examples per category")
    parser.add_argument("--concepts", nargs="*", help="Specific concepts to test (default: all)")

    args = parser.parse_args()

    # Load model
    print(f"Loading model {args.model}...")
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
    layer_idx = n_layers // 2  # Use middle layer (same as training default)

    print(f"Model: {hidden_dim} hidden dim, using layer {layer_idx}")

    # Find lens directory
    lens_dir = args.lens_pack / f"layer{args.level}"
    if not lens_dir.exists():
        print(f"Error: Lens directory not found: {lens_dir}")
        sys.exit(1)

    # Find MELD directory
    meld_level_dir = args.meld_dir / f"L{args.level}"
    if not meld_level_dir.exists():
        print(f"Error: MELD directory not found: {meld_level_dir}")
        sys.exit(1)

    # Load MELDs and find corresponding lenses
    all_results = []

    for meld_file in sorted(meld_level_dir.glob("*.json")):
        meld_data = load_polar_meld(meld_file)
        term = meld_data["polar_meld"]["term"]

        # Filter if specific concepts requested
        if args.concepts and term not in args.concepts:
            continue

        # Find corresponding lenses
        pos_lens_path = lens_dir / f"{term}_positive.pt"
        neg_lens_path = lens_dir / f"{term}_negative.pt"

        if not pos_lens_path.exists() or not neg_lens_path.exists():
            print(f"Warning: Lenses not found for {term}, skipping")
            continue

        print(f"\nValidating: {term}...")

        # Load lenses
        pos_lens = load_lens(pos_lens_path, input_dim=hidden_dim)
        neg_lens = load_lens(neg_lens_path, input_dim=hidden_dim)

        # Validate
        results = validate_concept(
            model, tokenizer, pos_lens, neg_lens,
            meld_data, args.device, layer_idx, args.max_examples
        )

        metrics = print_validation_report(results)
        all_results.append({
            "term": term,
            **metrics
        })

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    if all_results:
        pos_discs = [r["pos_lens_discrimination"] for r in all_results]
        neg_discs = [r["neg_lens_discrimination"] for r in all_results]

        print(f"\nDiscrimination scores (higher = better separation):")
        print(f"  Positive lens: mean={np.mean(pos_discs):.3f}, std={np.std(pos_discs):.3f}")
        print(f"  Negative lens: mean={np.mean(neg_discs):.3f}, std={np.std(neg_discs):.3f}")

        print(f"\nPer-concept breakdown:")
        print(f"  {'Concept':<50} {'Pos Disc':>10} {'Neg Disc':>10}")
        print(f"  {'-'*70}")
        for r in sorted(all_results, key=lambda x: x["pos_lens_discrimination"], reverse=True):
            print(f"  {r['term'][:49]:<50} {r['pos_lens_discrimination']:>+10.3f} {r['neg_lens_discrimination']:>+10.3f}")


if __name__ == "__main__":
    main()
