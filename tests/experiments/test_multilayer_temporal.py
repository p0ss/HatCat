#!/usr/bin/env python3
"""
Test multi-layer temporal monitoring using existing lenses.

This demonstrates the multi-layer lead-lag analysis by:
1. Loading existing trained lenses
2. Capturing activations from 3 layers during generation
3. Applying lenses to all layers
4. Plotting temporal evolution
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

from scripts.test_multilayer_hooking import MultiLayerTap


def load_lens(concept_path: Path, device='cpu'):
    """Load a trained lens from .pt file."""
    state_dict = torch.load(concept_path, map_location=device)

    # Extract weights (assuming 3-layer MLP: 2560 → 128 → 64 → 1)
    # We'll use the full model for now
    import torch.nn as nn

    lens = nn.Sequential(
        nn.Linear(2560, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )

    lens.load_state_dict(state_dict)
    lens.to(device)
    lens.eval()

    return lens


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test multi-layer temporal monitoring")
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt')
    parser.add_argument('--layers', type=int, nargs=3, default=[6, 15, 25],
                        help='Model layers to sample (early, mid, late)')
    parser.add_argument('--lens-dir', type=str,
                        default='results/adaptive_test_tiny/layer0')
    parser.add_argument('--concepts', type=str, nargs='+',
                        default=['Physical', 'Abstract', 'Process'],
                        help='Concepts to track')
    parser.add_argument('--prompt', type=str,
                        default="The concept of artificial intelligence refers to")
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--output', type=str,
                        default='results/multilayer_temporal_test.json')
    args = parser.parse_args()

    print("=" * 80)
    print("MULTI-LAYER TEMPORAL MONITORING TEST")
    print("=" * 80)
    print()
    print(f"Model: {args.model}")
    print(f"Layers: {args.layers} (early/mid/late)")
    print(f"Concepts: {args.concepts}")
    print(f"Prompt: {args.prompt[:60]}...")
    print()

    # Load model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print(f"✓ Model loaded on {device}")
    print()

    # Load lenses
    print("Loading lenses...")
    lens_dir = Path(args.lens_dir)
    lenses = {}

    for concept in args.concepts:
        lens_path = lens_dir / f"{concept}_classifier.pt"
        if not lens_path.exists():
            print(f"  ✗ {concept}: not found at {lens_path}")
            continue

        try:
            lenses[concept] = load_lens(lens_path, device='cpu')  # Keep on CPU
            print(f"  ✓ {concept}")
        except Exception as e:
            print(f"  ✗ {concept}: failed to load - {e}")

    if not lenses:
        print("\n✗ No lenses loaded")
        return 1

    print(f"\n✓ Loaded {len(lenses)} lenses")
    print()

    # Generate with multi-layer capture
    print("=" * 80)
    print("GENERATING WITH MULTI-LAYER CAPTURE")
    print("=" * 80)
    print()

    tap = MultiLayerTap(model, args.layers, hook_point="post_mlp")
    timeline = []

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']

    print(f"Generating {args.max_tokens} tokens...")
    start_time = time.time()

    for step in range(args.max_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Get next token (greedy)
        next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Collect activations
        layer_acts = tap.pop()

        # Apply lenses to each layer's activations
        token_text = tokenizer.decode(next_token_id[0])

        step_data = {
            'step': step,
            'token': token_text,
            'concepts': {}
        }

        for concept, lens in lenses.items():
            concept_scores = {}

            for layer_idx, activation in layer_acts.items():
                # Convert to float32 for lens
                act_fp32 = activation.float()

                # Run lens
                with torch.no_grad():
                    score = lens(act_fp32).item()

                concept_scores[f'layer_{layer_idx}'] = score

            step_data['concepts'][concept] = concept_scores

        timeline.append(step_data)

        # Append token
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # Check for EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    generation_time = time.time() - start_time
    tap.remove()

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    print(f"✓ Generated {len(timeline)} tokens in {generation_time:.2f}s")
    print()
    print("Output:")
    print(output_text)
    print()

    # Analyze temporal patterns
    print("=" * 80)
    print("TEMPORAL PATTERN ANALYSIS")
    print("=" * 80)
    print()

    for concept in args.concepts:
        if concept not in lenses:
            continue

        print(f"\n{concept}:")
        print("-" * 40)

        # Extract time series for each layer
        series = {
            f'layer_{layer_idx}': []
            for layer_idx in args.layers
        }

        for step_data in timeline:
            if concept in step_data['concepts']:
                for layer_key, score in step_data['concepts'][concept].items():
                    series[layer_key].append(score)

        # Compute statistics
        for layer_key in series:
            scores = series[layer_key]
            if scores:
                print(f"  {layer_key:12s}: mean={np.mean(scores):.3f}, "
                      f"max={np.max(scores):.3f}, "
                      f"std={np.std(scores):.3f}")

        # Look for temporal patterns
        if all(len(series[k]) > 5 for k in series):
            # Simple lead-lag: does early/mid spike predict late spike?
            early_key = f'layer_{args.layers[0]}'
            mid_key = f'layer_{args.layers[1]}'
            late_key = f'layer_{args.layers[2]}'

            early = np.array(series[early_key])
            mid = np.array(series[mid_key])
            late = np.array(series[late_key])

            # Cross-correlation (mid vs late)
            if len(mid) == len(late):
                corr = np.corrcoef(mid[:-3], late[3:])[0, 1] if len(mid) > 3 else 0
                print(f"\n  Lead-lag (mid→late, lag=3): correlation={corr:.3f}")
                if abs(corr) > 0.3:
                    print(f"    → {'Strong' if abs(corr) > 0.5 else 'Moderate'} "
                          f"{'positive' if corr > 0 else 'negative'} lead-lag detected!")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'model': args.model,
            'layers': args.layers,
            'concepts': args.concepts,
            'prompt': args.prompt,
            'generation_time': generation_time,
            'total_tokens': len(timeline),
        },
        'timeline': timeline,
        'output_text': output_text,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print()
    print("=" * 80)
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    # Create visualization
    print()
    print("Creating visualization...")

    fig, axes = plt.subplots(len(lenses), 1, figsize=(12, 3 * len(lenses)), sharex=True)
    if len(lenses) == 1:
        axes = [axes]

    for idx, concept in enumerate(args.concepts):
        if concept not in lenses:
            continue

        ax = axes[idx]

        # Extract time series
        for layer_idx in args.layers:
            layer_key = f'layer_{layer_idx}'
            scores = [step['concepts'][concept][layer_key]
                     for step in timeline if concept in step['concepts']]

            ax.plot(scores, label=f'Layer {layer_idx}', alpha=0.7, linewidth=2)

        ax.set_ylabel(f'{concept}\nActivation', fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Token Position', fontsize=10)
    axes[0].set_title('Multi-Layer Temporal Evolution', fontsize=12, fontweight='bold')

    plt.tight_layout()

    plot_path = output_path.with_suffix('.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_path}")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("Look for:")
    print("  1. Lead-lag patterns: mid-layer peaks 3-8 tokens before late-layer")
    print("  2. Early spikes: retrieval/context recall")
    print("  3. Mid plateaus: sustained reasoning/planning")
    print("  4. Late spikes: verbalization moments")

    return 0


if __name__ == '__main__':
    sys.exit(main())
