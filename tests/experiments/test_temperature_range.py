#!/usr/bin/env python3
"""
Test temperature range to find useful bounds for training data generation.

We want diversity without gibberish. This script generates samples across
temperature range and evaluates coherence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_sample(model, tokenizer, prompt, temperature, max_tokens=30):
    """Generate a single sample at given temperature."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("=" * 80)
    print("TEMPERATURE RANGE CHARACTERIZATION")
    print("=" * 80)
    print()

    # Load model
    model_name = "google/gemma-3-4b-pt"
    print(f"Loading {model_name}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print(f"✓ Loaded on {device}")
    print()

    # Test prompts
    prompts = [
        "Define \"Physical\": ",
        "The concept of artificial intelligence refers to ",
        "An example of a Process would be ",
    ]

    # Temperature range to test
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]

    for prompt in prompts:
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        print("=" * 80)
        print()

        for temp in temperatures:
            output = generate_sample(model, tokenizer, prompt, temp)
            # Show only the generated part
            generated = output[len(prompt):].strip()

            # Truncate for display
            if len(generated) > 100:
                generated = generated[:100] + "..."

            print(f"T={temp:0.1f}: {generated}")

        print()

    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    print("Look for:")
    print("  • T < 0.3: Too deterministic, low diversity")
    print("  • T = 0.3-0.5: Coherent, moderate diversity")
    print("  • T = 0.5-0.9: High diversity, still coherent")
    print("  • T = 0.9-1.2: Maximum useful diversity")
    print("  • T > 1.2: Risk of incoherence/gibberish")
    print()
    print("Recommended range: 0.3-0.9 (balanced) or 0.4-1.0 (more diverse)")

    return 0

if __name__ == '__main__':
    sys.exit(main())
