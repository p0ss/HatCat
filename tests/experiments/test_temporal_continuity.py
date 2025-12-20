"""
Test temporal continuity of concept activations.

Extends SUMOTemporalMonitor to record continuous concept dynamics
instead of just per-token aggregations.

Usage:
    ./.venv/bin/python scripts/test_temporal_continuity.py \
        --prompt "Tell me about deception in politics" \
        --output results/temporal_test/deception_politics.json
"""

import argparse
import json
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hat.monitoring.sumo_temporal import load_sumo_classifiers


def record_continuous_timeline(
    model,
    tokenizer,
    classifiers,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,  # Sampling now stable with float32
    device: str = "cuda",
    top_k_concepts: int = 10,
    threshold: float = 0.1
):
    """
    Record concept activations at EVERY forward pass, not just token outputs.

    Returns timeline with continuous concept dynamics.
    """
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    timeline = []

    with torch.inference_mode():
        # Generate with hidden states
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

        # Extract generated tokens
        token_ids = outputs.sequences[0][prompt_len:].cpu().tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Process hidden states for each forward pass
        # outputs.hidden_states is a tuple: one per generated token
        # Each element is a tuple of tensors (one per layer)

        for step_idx, step_states in enumerate(outputs.hidden_states):
            # step_states is tuple of (num_layers,) tensors
            # Each tensor is [batch=1, seq_len, hidden_dim]
            # Use last layer, last position
            last_layer = step_states[-1]  # [1, seq_len, hidden_dim]
            hidden_state = last_layer[:, -1, :]  # [1, hidden_dim]

            # Run all classifiers
            # Convert to float32 to match classifier dtype
            hidden_state_f32 = hidden_state.float()

            concept_scores = {}
            for concept_name, (classifier, layer, idx) in classifiers.items():
                prob = classifier(hidden_state_f32).item()
                if prob > threshold:
                    concept_scores[concept_name] = {
                        'probability': float(prob),
                        'layer': int(layer)
                    }

            # Get token info
            token = tokens[step_idx] if step_idx < len(tokens) else '<eos>'

            # Record timestep
            timeline.append({
                'forward_pass': step_idx,
                'token_idx': step_idx,  # For now, 1:1 with forward passes
                'token': token,
                'position': prompt_len + step_idx,
                'is_output': True,  # All these are token outputs
                'concepts': concept_scores
            })

    # Build result
    generated_text = ''.join(tokens)

    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'tokens': tokens,
        'timeline': timeline,
        'metadata': {
            'total_forward_passes': len(timeline),
            'total_tokens': len(tokens),
            'top_k_concepts': top_k_concepts,
            'threshold': threshold
        }
    }


def visualize_timeline_ascii(result, top_k=10):
    """
    Create ASCII sparkline visualization of concept timelines.
    """
    import numpy as np

    timeline = result['timeline']

    # Collect all unique concepts
    all_concepts = set()
    for step in timeline:
        all_concepts.update(step['concepts'].keys())

    # Build timeseries for each concept
    concept_timeseries = {name: [] for name in all_concepts}

    for step in timeline:
        for name in all_concepts:
            if name in step['concepts']:
                concept_timeseries[name].append(step['concepts'][name]['probability'])
            else:
                concept_timeseries[name].append(0.0)

    # Sort concepts by max activation
    concept_maxes = {name: max(vals) for name, vals in concept_timeseries.items()}
    top_concepts = sorted(concept_maxes.items(),
                         key=lambda x: x[1],
                         reverse=True)[:top_k]

    # Sparkline characters
    chars = '▁▂▃▄▅▆▇█'

    print("\n" + "=" * 80)
    print("TEMPORAL CONCEPT DYNAMICS")
    print("=" * 80)
    print(f"\nPrompt: {result['prompt']}")
    print(f"Generated: {result['generated_text']}")
    print(f"\nForward passes: {result['metadata']['total_forward_passes']}")
    print(f"Tokens: {result['metadata']['total_tokens']}")
    print("\n" + "-" * 80)
    print("Concept Activation Timelines (top {} by max activation)".format(top_k))
    print("-" * 80 + "\n")

    for concept_name, max_activation in top_concepts:
        values = np.array(concept_timeseries[concept_name])

        # Normalize to sparkline range
        if max_activation > 0:
            normalized = (values / max_activation * 7).astype(int)
            sparkline = ''.join(chars[min(n, 7)] for n in normalized)
        else:
            sparkline = chars[0] * len(values)

        print(f"{concept_name:30s} [{max_activation:5.3f}] {sparkline}")

    print("\n" + "=" * 80 + "\n")

    # Show token sequence with positions
    print("Token Sequence:")
    print("-" * 80)
    for step in timeline[:20]:  # Show first 20 tokens
        print(f"[{step['position']:3d}] {step['token']}")
    if len(timeline) > 20:
        print(f"... ({len(timeline) - 20} more tokens)")
    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt')
    parser.add_argument('--layers', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--top-k-concepts', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.1)
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='cuda',
        torch_dtype=torch.float32  # CRITICAL: float32 for numerical stability
    )
    model.eval()

    # Load SUMO classifiers
    print(f"Loading SUMO classifiers from layers {args.layers}")
    classifiers, hidden_dim = load_sumo_classifiers(layers=args.layers)
    print(f"Loaded {len(classifiers)} classifiers")

    # Record continuous timeline
    print(f"\nGenerating with prompt: {args.prompt}")
    result = record_continuous_timeline(
        model=model,
        tokenizer=tokenizer,
        classifiers=classifiers,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        top_k_concepts=args.top_k_concepts,
        threshold=args.threshold
    )

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Saved timeline to: {args.output}")

    # Visualize
    visualize_timeline_ascii(result, top_k=args.top_k_concepts)

    print(f"\nNext steps:")
    print(f"1. Inspect JSON: cat {args.output}")
    print(f"2. Plot timeline: python scripts/visualize_temporal_activations.py {args.output}")


if __name__ == '__main__':
    main()
