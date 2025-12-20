"""
Test temporal continuity with DynamicLensManager for hierarchical loading.

Uses hierarchical lens loading to handle all layers 3-5 (26K+ classifiers)
while staying within memory constraints through dynamic loading/unloading.

Usage:
    ./.venv/bin/python scripts/test_temporal_continuity_dynamic.py \
        --prompt "Tell me about deception in politics" \
        --output results/temporal_test/deception_politics_dynamic.json
"""

import argparse
import json
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hat.monitoring.lens_manager import DynamicLensManager


def record_continuous_timeline_dynamic(
    model,
    tokenizer,
    lens_manager: DynamicLensManager,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,
    device: str = "cuda",
    top_k_concepts: int = 10,
    parent_threshold: float = 0.3,
    concept_threshold: float = 0.1
):
    """
    Record concept activations using DynamicLensManager for hierarchical loading.

    Returns timeline with continuous concept dynamics and manager statistics.
    """
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    timeline = []
    manager_stats_timeline = []

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
        for step_idx, step_states in enumerate(outputs.hidden_states):
            # Use last layer, last position
            last_layer = step_states[-1]  # [1, seq_len, hidden_dim]
            hidden_state = last_layer[:, -1, :]  # [1, hidden_dim]

            # Convert to float32 to match classifier dtype
            hidden_state_f32 = hidden_state.float()

            # Use DynamicLensManager to detect and expand
            detected, timing = lens_manager.detect_and_expand(
                hidden_state_f32,
                top_k=top_k_concepts,
                return_timing=True
            )

            # Filter by threshold and convert to dict
            concept_scores = {}
            for concept_name, prob, layer in detected:
                if prob > concept_threshold:
                    concept_scores[concept_name] = {
                        'probability': float(prob),
                        'layer': int(layer)
                    }

            # Get token info
            token = tokens[step_idx] if step_idx < len(tokens) else '<eos>'

            # Record timestep
            timeline.append({
                'forward_pass': step_idx,
                'token_idx': step_idx,
                'token': token,
                'position': prompt_len + step_idx,
                'is_output': True,
                'concepts': concept_scores
            })

            # Record manager stats
            manager_stats_timeline.append({
                'step': step_idx,
                'timing': timing,
                'loaded_lenses': len(lens_manager.loaded_lenses)
            })

    # Build result
    generated_text = ''.join(tokens)

    # Get final manager statistics
    final_loaded = len(lens_manager.loaded_lenses)
    max_loaded = max(s['loaded_lenses'] for s in manager_stats_timeline) if manager_stats_timeline else final_loaded

    cache_hits = lens_manager.stats.get('cache_hits', 0)
    cache_misses = lens_manager.stats.get('cache_misses', 0)
    total = cache_hits + cache_misses
    cache_hit_rate = cache_hits / total if total > 0 else 0.0

    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'tokens': tokens,
        'timeline': timeline,
        'metadata': {
            'total_forward_passes': len(timeline),
            'total_tokens': len(tokens),
            'top_k_concepts': top_k_concepts,
            'threshold': concept_threshold,
            'parent_threshold': parent_threshold,
            'manager': {
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'cache_hit_rate': cache_hit_rate,
                'loaded_lenses': final_loaded,
                'max_lenses_loaded': max_loaded,
                'total_loads': lens_manager.stats.get('total_loads', 0),
                'total_unloads': lens_manager.stats.get('total_unloads', 0)
            }
        },
        'manager_stats_timeline': manager_stats_timeline
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
    print("TEMPORAL CONCEPT DYNAMICS (Hierarchical Loading)")
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


def print_manager_stats(result):
    """
    Print DynamicLensManager statistics.
    """
    stats = result['metadata']['manager']

    print("=" * 80)
    print("DYNAMIC LENS MANAGER STATISTICS")
    print("=" * 80)
    print(f"\nCache performance:")
    print(f"  - Hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  - Cache hits: {stats['cache_hits']}")
    print(f"  - Cache misses: {stats['cache_misses']}")
    print(f"\nLens loading:")
    print(f"  - Currently loaded: {stats['loaded_lenses']}")
    print(f"  - Max loaded (peak): {stats['max_lenses_loaded']}")
    print(f"  - Total loads: {stats['total_loads']}")
    print(f"  - Total unloads: {stats['total_unloads']}")
    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt')
    parser.add_argument('--base-layer', type=int, default=3,
                       help='Base SUMO layer to keep always loaded')
    parser.add_argument('--max-lenses', type=int, default=500,
                       help='Max lenses to keep loaded at once')
    parser.add_argument('--load-threshold', type=float, default=0.3,
                       help='Confidence threshold to load child lenses')
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--top-k-concepts', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Min probability to record concept')
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='cuda',
        dtype=torch.float32  # CRITICAL: float32 for numerical stability
    )
    model.eval()

    # Initialize DynamicLensManager
    print(f"\nInitializing DynamicLensManager:")
    print(f"  - Base layer: {args.base_layer}")
    print(f"  - Max loaded lenses: {args.max_lenses}")
    print(f"  - Load threshold: {args.load_threshold}")

    lens_manager = DynamicLensManager(
        base_layers=[args.base_layer],
        max_loaded_lenses=args.max_lenses,
        load_threshold=args.load_threshold,
        device='cuda'
    )

    print(f"  - Initial lenses loaded: {len(lens_manager.loaded_lenses)}")

    # Record continuous timeline
    print(f"\nGenerating with prompt: {args.prompt}")
    result = record_continuous_timeline_dynamic(
        model=model,
        tokenizer=tokenizer,
        lens_manager=lens_manager,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        top_k_concepts=args.top_k_concepts,
        parent_threshold=args.load_threshold,
        concept_threshold=args.threshold
    )

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Saved timeline to: {args.output}")

    # Visualize
    visualize_timeline_ascii(result, top_k=args.top_k_concepts)

    # Show manager statistics
    print_manager_stats(result)

    print(f"\nNext steps:")
    print(f"1. Inspect JSON: cat {args.output}")
    print(f"2. Compare dynamic vs static loading memory usage")
    print(f"3. Experiment with parent_threshold to balance detail vs performance")


if __name__ == '__main__':
    main()
