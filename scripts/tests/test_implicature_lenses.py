#!/usr/bin/env python3
"""
Mini validation: Can our lens pipeline learn implicature concepts?

Tests a subset of implicature-pragmatics meld concepts to validate
that our training process can distinguish these pragmatic phenomena.
"""

import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from training.sumo_classifiers import extract_activations, train_simple_classifier


def generate_prompts_from_examples(concept: dict, n_positives: int, n_negatives: int) -> tuple[list[str], list[int]]:
    """Generate training prompts directly from concept examples.

    This is a simplified version for mini-validation that doesn't require
    the full concept hierarchy - just uses the meld-provided examples.
    """
    import random

    # Check both top-level and training_hints for examples (both schemas valid)
    pos_examples = concept.get("positive_examples", [])
    neg_examples = concept.get("negative_examples", [])

    training_hints = concept.get("training_hints", {})
    if not pos_examples and training_hints:
        pos_examples = training_hints.get("positive_examples", [])
    if not neg_examples and training_hints:
        neg_examples = training_hints.get("negative_examples", [])
    definition = concept.get("definition", "")
    training_hints = concept.get("training_hints", {})
    disambiguation = training_hints.get("disambiguation", "")
    key_features = training_hints.get("key_features", [])

    prompts = []
    labels = []

    # Templates for variety
    templates = [
        "Example: {example}",
        "Consider this statement: {example}",
        "Analyze: {example}",
        "{example}",
        "Text: {example}",
    ]

    # Generate positive prompts
    for _ in range(n_positives):
        if pos_examples:
            example = random.choice(pos_examples)
            template = random.choice(templates)
            prompts.append(template.format(example=example))
            labels.append(1)

    # Generate negative prompts
    for _ in range(n_negatives):
        if neg_examples:
            example = random.choice(neg_examples)
            template = random.choice(templates)
            prompts.append(template.format(example=example))
            labels.append(0)

    # Shuffle together
    combined = list(zip(prompts, labels))
    random.shuffle(combined)
    prompts, labels = zip(*combined) if combined else ([], [])

    return list(prompts), list(labels)


def load_concepts_from_meld(meld_path: Path, concept_names: list[str] = None) -> dict:
    """Load concepts from a meld JSON file.

    Args:
        meld_path: Path to the meld JSON file
        concept_names: Optional list of concept names to filter. If None, load all.

    Returns:
        Dict of concept_name -> concept_data
    """
    with open(meld_path) as f:
        meld = json.load(f)

    concepts = {}
    for candidate in meld.get("candidates", []):
        term = candidate.get("term")
        if term and (concept_names is None or term in concept_names):
            concepts[term] = candidate

    return concepts


# Default meld path for v2
DEFAULT_MELD_PATH = PROJECT_ROOT / "melds" / "pending" / "implicature-pragmatics-v2.json"

# Concepts to test (subset of meld for mini-validation)
TEST_CONCEPT_NAMES = [
    "ScalarImplicature",
    "IndirectRefusal",
    "SarcasmMarker",
    "ConversationalImplicature",
]

# Fallback mock concepts if meld file not found
TEST_CONCEPTS = {
    "ScalarImplicature": {
        "sumo_term": "ScalarImplicature",
        "definition": "An implicature arising from the use of a weaker term on a scale, implying that stronger terms do not apply. Using 'some' implicates 'not all'; using 'warm' implicates 'not hot'.",
        "layer": 3,
        "parent_concepts": ["Implicature"],
        "positive_examples": [
            "Some of the students passed.",
            "I ate some of the cookies.",
            "The water is warm.",
            "I sometimes exercise.",
            "It's possible that she'll come.",
            "He's good at chess.",
            "I liked the movie."
        ],
        "negative_examples": [
            "All of the students passed.",
            "I ate all of the cookies.",
            "The water is hot.",
            "I always exercise.",
            "She will definitely come.",
            "He's excellent at chess.",
            "I loved the movie."
        ],
        "training_hints": {
            "disambiguation": "Scalar implicatures arise from choosing a WEAKER term on a scale (some/all, warm/hot, good/excellent, possible/certain).",
            "key_features": ["weaker scalar term", "implicates stronger not applicable", "quantity-based inference"]
        }
    },
    "IndirectRefusal": {
        "sumo_term": "IndirectRefusal",
        "definition": "A refusal performed indirectly, typically by providing reasons or conditions that implicate inability or unwillingness without explicit rejection.",
        "layer": 3,
        "parent_concepts": ["IndirectSpeechAct"],
        "positive_examples": [
            "I have to work that night.",
            "I have a bad back.",
            "I need to pick up my kids.",
            "Things are tight right now.",
            "I have three other urgent projects.",
            "That's not really my area of expertise."
        ],
        "negative_examples": [
            "No, I can't come to the party.",
            "No, I won't help you move.",
            "No, I'm not staying late.",
            "No, I won't lend you money.",
            "No, I can't finish this today.",
            "I refuse to help with that."
        ],
        "training_hints": {
            "disambiguation": "Indirect refusals provide REASONS that implicate refusal without saying 'no'.",
            "key_features": ["reason-giving", "implicit no", "face-saving"]
        }
    },
    "SarcasmMarker": {
        "sumo_term": "SarcasmMarker",
        "definition": "Signals that indicate mocking or contemptuous irony directed at a person or their actions.",
        "layer": 3,
        "parent_concepts": ["MetaLinguisticMarker"],
        "positive_examples": [
            "Oh, what a genius idea that turned out to be.",
            "Yeah, because that's worked so well in the past.",
            "Sure, you're such a hard worker.",
            "Oh, I'm sooo sorry I inconvenienced you.",
            "Well, aren't you special.",
            "Thanks for all your 'help'.",
            "Real mature."
        ],
        "negative_examples": [
            "That was a bad idea.",
            "That approach hasn't worked before.",
            "You could work harder.",
            "I apologize for the inconvenience.",
            "You're special to me.",
            "Thank you for your help.",
            "That was immature."
        ],
        "training_hints": {
            "disambiguation": "Sarcasm is HOSTILE irony - it mocks or criticizes through opposite meaning.",
            "key_features": ["hostile intent", "opposite meaning", "mocking tone"]
        }
    },
    "ConversationalImplicature": {
        "sumo_term": "ConversationalImplicature",
        "definition": "An implicature derived from the assumption that the speaker is following cooperative conversational principles.",
        "layer": 3,
        "parent_concepts": ["Implicature"],
        "positive_examples": [
            "A: Where's the coffee? B: There's a shop around the corner.",
            "A: How's your new job? B: The office has nice views.",
            "A: Did you enjoy the book? B: I finished it.",
            "A: Is she a good singer? B: She has great stage presence.",
            "A: Can you help me move? B: I have a bad back.",
            "Nice weather we're having."
        ],
        "negative_examples": [
            "There's a coffee shop around the corner that sells coffee.",
            "My job isn't great but the office has nice views.",
            "I finished the book but didn't enjoy it.",
            "She's not a great singer but has stage presence.",
            "No, I can't help you move because I have a bad back.",
            "I'm initiating social conversation with this weather comment."
        ],
        "training_hints": {
            "disambiguation": "Conversational implicatures are CALCULATED from context + cooperative assumptions.",
            "key_features": ["context-dependent", "cooperative principle", "relevance inference"]
        }
    }
}

# Held-out test examples (not in training)
HELD_OUT_TESTS = {
    "ScalarImplicature": {
        "positive": [
            "A few people agreed with the proposal.",
            "She's somewhat qualified for the position.",
            "The meeting was fairly productive.",
            "I occasionally read fiction.",
        ],
        "negative": [
            "Everyone agreed with the proposal.",
            "She's perfectly qualified for the position.",
            "The meeting was extremely productive.",
            "I constantly read fiction.",
        ]
    },
    "IndirectRefusal": {
        "positive": [
            "My schedule is completely packed next week.",
            "I'm not sure I'm the right person for this.",
            "That would be difficult given current constraints.",
            "I'd need to check with my manager first.",
        ],
        "negative": [
            "No, that won't work for me.",
            "I decline your invitation.",
            "Absolutely not.",
            "I'm saying no to this request.",
        ]
    },
    "SarcasmMarker": {
        "positive": [
            "Wow, what an original thought.",
            "Oh great, another meeting.",
            "Yeah, that'll totally work.",
            "How very helpful of you.",
        ],
        "negative": [
            "That's not an original thought.",
            "I dislike having so many meetings.",
            "I don't think that will work.",
            "That wasn't very helpful.",
        ]
    },
    "ConversationalImplicature": {
        "positive": [
            "A: Want to grab dinner? B: I just ate.",
            "A: Did you like the gift? B: It's certainly... unique.",
            "A: Is John coming? B: His car broke down.",
            "The food was interesting.",
        ],
        "negative": [
            "No, I don't want dinner because I just ate.",
            "I didn't like the gift, it was weird.",
            "John isn't coming because his car broke down.",
            "The food was strange and I didn't enjoy it.",
        ]
    }
}


def load_model_and_tokenizer(model_name: str, device: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def run_mini_validation(
    model_name: str = "swiss-ai/Apertus-8B-2509",
    device: str = "cuda",
    n_train: int = 20,
    layer_idx: int = 15,
    meld_path: Path = None,
):
    """Run mini validation on implicature concepts."""

    print("=" * 70)
    print("IMPLICATURE LENS MINI-VALIDATION (v2)")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Training samples: {n_train} pos, {n_train} neg")

    # Try to load concepts from meld file
    meld_path = meld_path or DEFAULT_MELD_PATH
    if meld_path.exists():
        print(f"Loading concepts from: {meld_path}")
        test_concepts = load_concepts_from_meld(meld_path, TEST_CONCEPT_NAMES)
        print(f"Loaded {len(test_concepts)} concepts from meld")
        for name, concept in test_concepts.items():
            hints = concept.get("training_hints", {})
            n_pos = len(hints.get("positive_examples", []))
            n_neg = len(hints.get("negative_examples", []))
            print(f"  - {name}: {n_pos} pos, {n_neg} neg examples")
    else:
        print(f"Meld file not found at {meld_path}, using fallback mock concepts")
        test_concepts = TEST_CONCEPTS

    print()

    # Load model once
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    results = {}

    for concept_name, concept_data in test_concepts.items():
        print(f"\n{'='*60}")
        print(f"Testing: {concept_name}")
        print(f"{'='*60}")

        # Generate training data using meld examples directly
        print("\n1. Generating training data...")
        prompts, labels = generate_prompts_from_examples(
            concept=concept_data,
            n_positives=n_train,
            n_negatives=n_train,
        )

        print(f"   Generated {len(prompts)} prompts ({sum(labels)} pos, {len(labels) - sum(labels)} neg)")

        # Show a few examples
        print("\n   Sample positives:")
        pos_samples = [p for p, l in zip(prompts, labels) if l == 1][:2]
        for s in pos_samples:
            print(f"     + {s[:70]}...")

        print("\n   Sample negatives:")
        neg_samples = [p for p, l in zip(prompts, labels) if l == 0][:2]
        for s in neg_samples:
            print(f"     - {s[:70]}...")

        # Extract activations
        print("\n2. Extracting activations...")
        try:
            activations = extract_activations(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                layer_idx=layer_idx,
                max_new_tokens=15,
                batch_size=4,
                extraction_mode="combined",
            )
            print(f"   Shape: {activations.shape}")

            # Combined mode doubles samples (prompt + generation per input)
            # Labels need to be doubled too
            labels_expanded = []
            for label in labels:
                labels_expanded.append(label)  # prompt activation
                labels_expanded.append(label)  # generation activation
            labels_array = np.array(labels_expanded)

        except Exception as e:
            print(f"   ERROR during extraction: {e}")
            import traceback
            traceback.print_exc()
            results[concept_name] = {"status": "failed", "error": str(e)}
            continue

        # Split into train/test
        print("\n3. Training classifier...")
        n_samples = len(labels_array)
        indices = np.random.permutation(n_samples)
        split = int(0.8 * n_samples)
        train_idx, test_idx = indices[:split], indices[split:]

        X_train = activations[train_idx]
        y_train = labels_array[train_idx]
        X_test = activations[test_idx]
        y_test = labels_array[test_idx]

        try:
            classifier, metrics = train_simple_classifier(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                hidden_dim=128,
                epochs=100,
                lr=0.001,
            )

            print(f"   Training metrics:")
            print(f"     Train F1: {metrics.get('train_f1', 'N/A'):.3f}")
            print(f"     Test F1:  {metrics.get('test_f1', 'N/A'):.3f}")

        except Exception as e:
            print(f"   ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            results[concept_name] = {"status": "failed", "error": str(e)}
            continue

        # Test on held-out examples
        print("\n4. Testing on held-out examples...")
        held_out = HELD_OUT_TESTS.get(concept_name, {})
        pos_tests = held_out.get("positive", [])
        neg_tests = held_out.get("negative", [])

        if pos_tests and neg_tests:
            test_prompts = pos_tests + neg_tests
            test_labels = [1] * len(pos_tests) + [0] * len(neg_tests)

            # Extract activations for held-out
            test_activations = extract_activations(
                model=model,
                tokenizer=tokenizer,
                prompts=test_prompts,
                device=device,
                layer_idx=layer_idx,
                max_new_tokens=15,
                batch_size=4,
                extraction_mode="combined",
            )

            # Expand labels for combined mode
            test_labels_expanded = []
            for label in test_labels:
                test_labels_expanded.append(label)
                test_labels_expanded.append(label)

            # Get predictions
            classifier.eval()
            with torch.no_grad():
                test_tensor = torch.FloatTensor(test_activations).to(
                    next(classifier.parameters()).device
                )
                preds_raw = classifier(test_tensor).cpu().numpy().flatten()

            # Aggregate predictions per original prompt (average prompt + gen)
            n_original = len(test_prompts)
            aggregated_preds = []
            for i in range(n_original):
                prompt_pred = preds_raw[i * 2]
                gen_pred = preds_raw[i * 2 + 1]
                avg_pred = (prompt_pred + gen_pred) / 2
                aggregated_preds.append(avg_pred)

            # Calculate accuracy
            correct = 0
            print("\n   Predictions:")
            for i, (prompt, label, prob) in enumerate(zip(test_prompts, test_labels, aggregated_preds)):
                pred = 1 if prob > 0.5 else 0
                if pred == label:
                    correct += 1
                status = "✓" if label == pred else "✗"
                label_str = "POS" if label == 1 else "NEG"
                print(f"     {status} [{label_str}] p={prob:.2f} | {prompt[:50]}...")

            accuracy = correct / len(test_labels)
            print(f"\n   Held-out accuracy: {accuracy:.1%} ({correct}/{len(test_labels)})")

            results[concept_name] = {
                "status": "success",
                "train_f1": metrics.get('train_f1', 0),
                "test_f1": metrics.get('test_f1', 0),
                "holdout_accuracy": accuracy,
                "n_holdout": len(test_labels),
            }
        else:
            results[concept_name] = {
                "status": "success",
                "train_f1": metrics.get('train_f1', 0),
                "test_f1": metrics.get('test_f1', 0),
                "holdout_accuracy": None,
            }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Concept':<25} {'Train F1':<12} {'Test F1':<12} {'Holdout Acc':<12}")
    print("-" * 70)

    for concept, res in results.items():
        if res["status"] == "success":
            train_f1 = f"{res['train_f1']:.3f}" if res.get('train_f1') else "N/A"
            test_f1 = f"{res['test_f1']:.3f}" if res.get('test_f1') else "N/A"
            holdout = f"{res['holdout_accuracy']:.1%}" if res.get('holdout_accuracy') else "N/A"
            print(f"{concept:<25} {train_f1:<12} {test_f1:<12} {holdout:<12}")
        else:
            print(f"{concept:<25} FAILED: {res.get('error', 'unknown')[:40]}")

    # Overall assessment
    successful = [r for r in results.values() if r["status"] == "success" and r.get("holdout_accuracy")]
    if successful:
        avg_holdout = sum(r["holdout_accuracy"] for r in successful) / len(successful)
        avg_test_f1 = sum(r.get("test_f1", 0) for r in successful) / len(successful)
        print(f"\nAverage holdout accuracy: {avg_holdout:.1%}")
        print(f"Average test F1: {avg_test_f1:.3f}")

        if avg_holdout >= 0.7:
            print("\n✓ PASS: Lens pipeline can learn implicature concepts")
        elif avg_holdout >= 0.55:
            print("\n⚠ MARGINAL: Some signal but may need more training data or tuning")
        else:
            print("\n✗ CONCERN: Lens pipeline struggling with these concepts")
            print("  Consider: more training examples, different layer, richer prompts")

    # Policy-based acceptance assessment
    # Thresholds from pack.json meld_policy.lens_performance_requirements
    POLICY_THRESHOLDS = {
        "STANDARD": {"min_holdout_accuracy": 0.60, "min_test_f1": 0.60},
        "ELEVATED": {"min_holdout_accuracy": 0.65, "min_test_f1": 0.65},
        "PROTECTED": {"min_holdout_accuracy": 0.70, "min_test_f1": 0.70},
        "CRITICAL": {"min_holdout_accuracy": 0.75, "min_test_f1": 0.75},
    }
    HARD_FLOOR = {"min_holdout_accuracy": 0.60, "min_test_f1": 0.60}

    # The meld is ELEVATED protection level; check if it maps to critical simplex (DeceptionDetector)
    protection_level = "ELEVATED"  # From meld validation output
    maps_to_critical_simplex = True  # Maps to DeceptionDetector per meld

    print("\n" + "=" * 70)
    print("POLICY-BASED ACCEPTANCE ASSESSMENT")
    print("=" * 70)
    print(f"Protection Level: {protection_level}")
    print(f"Maps to Critical Simplex: {maps_to_critical_simplex}")
    print(f"Required thresholds: holdout >= {POLICY_THRESHOLDS[protection_level]['min_holdout_accuracy']:.0%}, "
          f"F1 >= {POLICY_THRESHOLDS[protection_level]['min_test_f1']:.0%}")
    if maps_to_critical_simplex:
        print(f"Hard floor (critical simplex): holdout >= {HARD_FLOOR['min_holdout_accuracy']:.0%}, "
              f"F1 >= {HARD_FLOOR['min_test_f1']:.0%}")

    print(f"\n{'Concept':<25} {'Holdout':<10} {'Test F1':<10} {'Category':<12}")
    print("-" * 70)

    for concept, res in results.items():
        if res["status"] == "success":
            holdout = res.get('holdout_accuracy', 0)
            test_f1 = res.get('test_f1', 0)

            # Determine acceptance category
            thresh = POLICY_THRESHOLDS[protection_level]
            if maps_to_critical_simplex and (holdout < HARD_FLOOR["min_holdout_accuracy"] or test_f1 < HARD_FLOOR["min_test_f1"]):
                category = "REJECT"
            elif holdout >= thresh["min_holdout_accuracy"] and test_f1 >= thresh["min_test_f1"]:
                category = "PASS"
            else:
                category = "CONDITIONAL"

            icon = {"PASS": "✓", "CONDITIONAL": "⚠", "REJECT": "✗"}[category]
            print(f"{concept:<25} {holdout:.1%:<10} {test_f1:.3f:<10} {icon} {category}")

            # Store category in results
            res["acceptance_category"] = category

    # Save results
    output_path = PROJECT_ROOT / "results" / "implicature_mini_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="swiss-ai/Apertus-8B-2509")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--layer", type=int, default=15)
    args = parser.parse_args()

    run_mini_validation(
        model_name=args.model,
        device=args.device,
        n_train=args.n_train,
        layer_idx=args.layer,
    )
