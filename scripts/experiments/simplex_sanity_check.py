#!/usr/bin/env python3
"""
Simplex Lens Sanity Check

Loads the Nov 25 trained simplex lenses and checks for:
1. Over-firing (all poles activating for same simplex)
2. Cross-simplex co-activation patterns
3. Appropriate pole separation on test prompts
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Simplex results directory
SIMPLEX_DIR = Path("/home/poss/Documents/Code/HatCat/results/s_tier_simplexes/run_20251226_023252")

# Test prompts designed to trigger specific poles
TEST_PROMPTS = {
    # Should trigger threat_perception:negative (alarm)
    "threat_alarm": [
        "I'm terrified something terrible is about to happen",
        "There's imminent danger and I need to escape",
        "This situation feels extremely threatening",
    ],
    # Should trigger threat_perception:positive (safety)
    "threat_safety": [
        "I feel completely safe and secure here",
        "Everything is calm and there's nothing to worry about",
        "This is a peaceful, protected environment",
    ],
    # Should trigger social_orientation:negative (hostility)
    "social_hostile": [
        "I want to attack and destroy my enemies",
        "Everyone is against me and I must fight back",
        "I feel aggressive and combative toward others",
    ],
    # Should trigger social_orientation:positive (cooperation)
    "social_cooperative": [
        "Let's work together to solve this problem",
        "I value collaboration and helping others succeed",
        "We can achieve more by cooperating than competing",
    ],
    # Neutral/baseline prompts
    "neutral": [
        "The weather today is partly cloudy",
        "Please pass me the salt",
        "The meeting is scheduled for 3pm",
    ],
    # Mixed/ambiguous prompts
    "ambiguous": [
        "I'm not sure how I feel about this situation",
        "Things could go either way from here",
        "It's complicated and I have mixed feelings",
    ],
}


def create_classifier_model(input_dim: int = 2560) -> torch.nn.Module:
    """Create classifier model matching saved architecture."""
    # Architecture: 2560 -> 128 -> 64 -> 1
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 1),
    )


def load_simplex_classifiers(simplex_dir: Path) -> Dict[str, Dict[str, torch.nn.Module]]:
    """Load all simplex classifiers from training results."""
    classifiers = {}

    for simplex_path in simplex_dir.iterdir():
        if not simplex_path.is_dir() or simplex_path.name in ['logs', 'aspiration']:
            continue

        simplex_name = simplex_path.name
        classifiers[simplex_name] = {}

        for pole in ['negative', 'neutral', 'positive']:
            pole_dir = simplex_path / pole
            if not pole_dir.exists():
                continue

            # Find the classifier file
            classifier_files = list(pole_dir.glob("*_classifier.pt"))
            if not classifier_files:
                print(f"  Warning: No classifier found for {simplex_name}/{pole}")
                continue

            classifier_path = classifier_files[0]
            try:
                state_dict = torch.load(classifier_path, map_location='cpu')
                # Create model and load weights
                model = create_classifier_model()

                # Handle both old format (0.weight) and new format (net.0.weight)
                # Also convert bfloat16 to float32 for compatibility
                if any(k.startswith('net.') for k in state_dict.keys()):
                    # New format: strip 'net.' prefix and convert to float32
                    state_dict = {k.replace('net.', ''): v.float() for k, v in state_dict.items()}
                else:
                    # Old format: just ensure float32
                    state_dict = {k: v.float() for k, v in state_dict.items()}

                model.load_state_dict(state_dict)
                model.eval()
                classifiers[simplex_name][pole] = model
            except Exception as e:
                print(f"  Error loading {simplex_name}/{pole}: {e}")

    return classifiers


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layer_idx: int = 31  # Layer 31 has best test F1 and activation distances
) -> torch.Tensor:
    """Extract hidden state activations from model."""
    device = next(model.parameters()).device
    all_activations = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get last token activation from specified layer
            hidden = outputs.hidden_states[layer_idx]
            last_token_act = hidden[0, -1, :].cpu()
            all_activations.append(last_token_act)

    return torch.stack(all_activations)


def normalize_activations(activations: torch.Tensor) -> torch.Tensor:
    """Per-sample normalize activations (same as training normalization).

    This matches the normalize_inputs=True behavior in train_simple_classifier:
    per-sample mean subtraction and std division, equivalent to LayerNorm
    with elementwise_affine=False.
    """
    mean = activations.mean(dim=-1, keepdim=True)
    std = activations.std(dim=-1, keepdim=True) + 1e-8
    return (activations - mean) / std


def run_classifiers(
    activations: torch.Tensor,
    classifiers: Dict[str, Dict[str, torch.nn.Module]]
) -> Dict[str, Dict[str, List[float]]]:
    """Run all classifiers on activations and return probabilities."""
    results = {}

    # CRITICAL: Normalize activations before classification
    # Training uses normalize_inputs=True, so inference must match
    activations_norm = normalize_activations(activations)

    for simplex_name, poles in classifiers.items():
        results[simplex_name] = {}
        for pole_name, classifier in poles.items():
            with torch.no_grad():
                logits = classifier(activations_norm.float())
                probs = torch.sigmoid(logits).squeeze()

                # Handle single value vs batch
                if probs.dim() == 0:
                    probs = probs.unsqueeze(0)

                results[simplex_name][pole_name] = probs.numpy().tolist()

    return results


def analyze_results(
    results: Dict[str, Dict[str, List[float]]],
    prompt_category: str,
    prompts: List[str]
) -> Dict:
    """Analyze classifier results for issues."""
    analysis = {
        "category": prompt_category,
        "num_prompts": len(prompts),
        "issues": [],
        "simplex_activations": {},
    }

    for simplex_name, poles in results.items():
        pole_means = {pole: np.mean(probs) for pole, probs in poles.items()}
        analysis["simplex_activations"][simplex_name] = pole_means

        # Check for over-firing: all poles > 0.5
        high_poles = [p for p, v in pole_means.items() if v > 0.5]
        if len(high_poles) >= 2:
            analysis["issues"].append({
                "type": "over_firing",
                "simplex": simplex_name,
                "high_poles": high_poles,
                "values": {p: round(v, 3) for p, v in pole_means.items()}
            })

        # Check for under-firing: all poles < 0.3
        if all(v < 0.3 for v in pole_means.values()):
            analysis["issues"].append({
                "type": "under_firing",
                "simplex": simplex_name,
                "values": {p: round(v, 3) for p, v in pole_means.items()}
            })

    return analysis


def print_activation_matrix(results: Dict[str, Dict[str, List[float]]], prompts: List[str]):
    """Print activation matrix for visual inspection."""
    print("\n  Simplex                  | Neg   | Neu   | Pos   |")
    print("  " + "-" * 52)

    for simplex_name in sorted(results.keys()):
        poles = results[simplex_name]
        neg = np.mean(poles.get('negative', [0]))
        neu = np.mean(poles.get('neutral', [0]))
        pos = np.mean(poles.get('positive', [0]))

        # Highlight concerning patterns
        marker = ""
        if neg > 0.5 and pos > 0.5:
            marker = " !! BOTH"
        elif neg > 0.5 and neu > 0.5:
            marker = " ! neg+neu"
        elif pos > 0.5 and neu > 0.5:
            marker = " ! pos+neu"

        print(f"  {simplex_name:24} | {neg:.3f} | {neu:.3f} | {pos:.3f} |{marker}")


def main():
    print("=" * 60)
    print("SIMPLEX LENS SANITY CHECK")
    print("=" * 60)

    # Load classifiers
    print("\n1. Loading simplex classifiers...")
    classifiers = load_simplex_classifiers(SIMPLEX_DIR)
    print(f"   Loaded {len(classifiers)} simplexes:")
    for name, poles in classifiers.items():
        print(f"     - {name}: {list(poles.keys())}")

    # Load model
    print("\n2. Loading model...")
    model_id = "google/gemma-3-4b-pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"   Model loaded on {device}")

    # Run tests
    print("\n3. Running classifier tests...")
    all_analyses = []

    for category, prompts in TEST_PROMPTS.items():
        print(f"\n   === {category.upper()} ===")
        for p in prompts:
            print(f"   - {p[:50]}...")

        # Extract activations
        activations = extract_activations(model, tokenizer, prompts)

        # Run classifiers
        results = run_classifiers(activations, classifiers)

        # Print matrix
        print_activation_matrix(results, prompts)

        # Analyze
        analysis = analyze_results(results, category, prompts)
        all_analyses.append(analysis)

        if analysis["issues"]:
            print(f"\n   Issues found:")
            for issue in analysis["issues"]:
                print(f"     - {issue['type']}: {issue['simplex']} - {issue.get('high_poles', issue.get('values'))}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_issues = sum(len(a["issues"]) for a in all_analyses)
    over_firing = sum(1 for a in all_analyses for i in a["issues"] if i["type"] == "over_firing")
    under_firing = sum(1 for a in all_analyses for i in a["issues"] if i["type"] == "under_firing")

    print(f"\n   Total issues: {total_issues}")
    print(f"   Over-firing: {over_firing}")
    print(f"   Under-firing: {under_firing}")

    if total_issues == 0:
        print("\n   ✓ No obvious issues detected!")
        print("   Simplex lenses appear to be working correctly.")
    else:
        print(f"\n   ⚠ {total_issues} issues found - review needed before steering experiments")

    # Save detailed results
    output_path = SIMPLEX_DIR / "sanity_check_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_analyses, f, indent=2)
    print(f"\n   Results saved to: {output_path}")


if __name__ == "__main__":
    main()
