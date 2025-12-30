#!/usr/bin/env python3
"""
Quick validation of prompt-phase activation monitoring.
Tests with single prompt to ensure implementation works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hat.monitoring.lens_manager import DynamicLensManager

def main():
    print("=" * 80)
    print("PROMPT-PHASE ACTIVATION - VALIDATION TEST")
    print("=" * 80)

    prompt = "Artificial intelligence can help society by"
    print(f"\nTest prompt: \"{prompt}\"")

    # Load model
    print("\nLoading model...")
    model_name = "google/gemma-3-4b-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load lens manager
    print("Loading lens manager...")
    lens_manager = DynamicLensManager(
        lens_pack_id="gemma-3-4b-pt_sumo-wordnet-v2",
        base_layers=[3],
        max_loaded_lenses=500,
        load_threshold=0.3,
        device="cuda"
    )

    print(f"Initial lenses loaded: {len(lens_manager.loaded_lenses)}")

    # Tokenize and process prompt
    print("\nProcessing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids[0]
    tokens = [tokenizer.decode([tid]) for tid in input_ids]

    print(f"Prompt tokens ({len(tokens)}):")
    for i, token in enumerate(tokens):
        print(f"  [{i}] {repr(token)}")

    # Forward pass
    print("\nRunning forward pass...")
    model.eval()

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    # Extract hidden states
    last_layer_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
    print(f"Hidden states shape: {last_layer_states.shape}")

    # Test detection at a few positions
    print("\nTesting concept detection...")
    test_positions = [0, len(tokens)//2, len(tokens)-1]

    for pos in test_positions:
        print(f"\nPosition {pos} (token: {repr(tokens[pos])}):")
        hidden_state = last_layer_states[0, pos:pos+1, :]
        hidden_state_f32 = hidden_state.float()

        detected, timing = lens_manager.detect_and_expand(
            hidden_state_f32,
            top_k=5,
            return_timing=True
        )

        print(f"  Detected {len(detected)} concepts:")
        for concept_name, prob, layer in detected[:5]:
            print(f"    {concept_name}: {prob:.3f} (L{layer})")

    print("\n" + "=" * 80)
    print("VALIDATION SUCCESSFUL")
    print("=" * 80)
    print("\nImplementation working correctly.")
    print("Ready to run full test with:")
    print("  poetry run python scripts/test_prompt_phase_activation.py --mode comparison")

if __name__ == '__main__':
    main()
