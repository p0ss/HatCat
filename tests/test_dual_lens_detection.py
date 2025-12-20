#!/usr/bin/env python3
"""Test dual lens detection and divergence measurement."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.monitoring.lens_manager import DynamicLensManager
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_dual_detection():
    """Test both activation and text lenses on same token."""

    print("\n" + "=" * 80)
    print("DUAL LENS DETECTION & DIVERGENCE TEST")
    print("=" * 80)

    # Load manager
    print("\nLoading lens manager...")
    manager = DynamicLensManager(
        lenses_dir=Path('results/adaptive_test_tiny'),
        base_layers=[0],
        use_activation_lenses=True,
        use_text_lenses=True,
        keep_top_k=50,
    )

    print(f"\nLoaded:")
    print(f"  Activation lenses: {len(manager.loaded_activation_lenses)}")
    print(f"  Text lenses: {len(manager.loaded_text_lenses)}")

    # Load model
    print("\nLoading Gemma model...")
    model_name = "google/gemma-3-4b-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda",
    )
    model.eval()

    # Test cases
    test_cases = [
        "The cat is an animal.",
        "Physical objects have mass.",
        "The number 42 is a quantity.",
        "Democracy is an abstract concept.",
    ]

    print("\n" + "=" * 80)
    print("TEST CASES")
    print("=" * 80)

    for prompt in test_cases:
        print(f"\nPrompt: \"{prompt}\"")
        print("-" * 80)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        token_ids = inputs.input_ids[0]
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[0]  # Layer 0

        # Test each token
        for token_idx, token_text in enumerate(tokens):
            hidden_state = hidden_states[0, token_idx, :].cpu().numpy()  # [2560]

            # Run activation lenses
            activation_scores = {}
            for concept_key, lens in manager.loaded_activation_lenses.items():
                with torch.no_grad():
                    h = torch.tensor(hidden_state, dtype=torch.float32).to("cuda")
                    prob = lens(h).item()
                    activation_scores[concept_key[0]] = prob

            # Run text lenses
            text_scores = {}
            for concept_key, text_lens in manager.loaded_text_lenses.items():
                try:
                    prob = text_lens.pipeline.predict_proba([token_text])[0, 1]  # Prob of class 1
                    text_scores[concept_key[0]] = prob
                except Exception as e:
                    pass

            # Show results for high-confidence activation OR text detections
            high_conf_activation = {k: v for k, v in activation_scores.items() if v > 0.6}
            high_conf_text = {k: v for k, v in text_scores.items() if v > 0.6}

            if high_conf_activation or high_conf_text:
                print(f"\n  Token: '{token_text}'")

                # Show top activation detections
                if high_conf_activation:
                    print(f"    Activation lenses (>0.6):")
                    for concept, prob in sorted(high_conf_activation.items(), key=lambda x: -x[1])[:5]:
                        print(f"      {concept:30s} {prob:.3f}")

                # Show top text detections
                if high_conf_text:
                    print(f"    Text lenses (>0.6):")
                    for concept, prob in sorted(high_conf_text.items(), key=lambda x: -x[1])[:5]:
                        print(f"      {concept:30s} {prob:.3f}")

                # Show divergences
                all_concepts = set(activation_scores.keys()) | set(text_scores.keys())
                divergences = []
                for concept in all_concepts:
                    act_prob = activation_scores.get(concept, 0.0)
                    txt_prob = text_scores.get(concept, 0.0)
                    div = abs(act_prob - txt_prob)
                    if div > 0.3:  # Significant divergence
                        divergences.append((concept, act_prob, txt_prob, div))

                if divergences:
                    print(f"    DIVERGENCES (>0.3):")
                    for concept, act_prob, txt_prob, div in sorted(divergences, key=lambda x: -x[3])[:3]:
                        print(f"      {concept:30s} Activation:{act_prob:.3f}  Text:{txt_prob:.3f}  Δ={div:.3f}")


if __name__ == '__main__':
    test_dual_detection()
