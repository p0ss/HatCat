#!/usr/bin/env python3
"""
Test Prompt-Phase Activation Monitoring

Captures concept activations during PROMPT PROCESSING, not generation.

This tests the hypothesis that "core concepts typically activate very early
in the piece" - specifically during the prompt itself, not during generation.

Key difference from existing temporal tests:
- Existing: Captures activations during generation (after prompt)
- This: Captures activations during prompt processing (before generation)

Usage:
    poetry run python scripts/test_prompt_phase_activation.py \
        --output results/prompt_phase_tests/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hat.monitoring.lens_manager import DynamicLensManager


# Diverse test prompts covering different domains
TEST_PROMPTS = [
    # AI/Tech concepts (should activate AI*, ComputerScience, etc.)
    "Artificial intelligence can help society by",
    "Machine learning algorithms work by",
    "Neural networks are used for",

    # Physical/concrete concepts (should activate Physical, Object, etc.)
    "A car is a vehicle that",
    "The ocean contains many",
    "Mountains are formed when",

    # Abstract concepts (should activate Proposition, Attribute, etc.)
    "The meaning of life is",
    "Truth and beauty are",
    "Justice requires that",

    # Social/organizational (should activate Organization, SocialRelation, etc.)
    "Companies need to",
    "Governments are responsible for",
    "Communities thrive when",

    # Quantitative (should activate Quantity, Number, etc.)
    "The calculation shows that",
    "There are many reasons why",
    "Five important factors include",
]


def capture_prompt_phase_activations(
    model,
    tokenizer,
    lens_manager,
    prompt: str,
    threshold: float = 0.1,
    top_k_concepts: int = 10,
    device: str = "cuda"
):
    """
    Capture concept activations during prompt processing.

    Returns:
        dict with prompt tokens and concept activations for each position
    """
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids[0]  # [seq_len]

    # Decode each token for display
    tokens = [tokenizer.decode([tid]) for tid in input_ids]

    prompt_timeline = []

    with torch.inference_mode():
        # Forward pass through prompt (no generation)
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract hidden states from last layer
        # outputs.hidden_states: tuple of (num_layers,) tensors
        # Each tensor: [batch=1, seq_len, hidden_dim]
        last_layer_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]

        # Process each position in the prompt
        for position_idx in range(last_layer_states.shape[1]):
            hidden_state = last_layer_states[0, position_idx:position_idx+1, :]  # [1, hidden_dim]

            # Convert to float32 to match classifier dtype
            hidden_state_f32 = hidden_state.float()

            # Use DynamicLensManager to detect concepts
            detected, timing = lens_manager.detect_and_expand(
                hidden_state_f32,
                top_k=top_k_concepts,
                return_timing=True
            )

            # Filter by threshold and convert to dict
            concept_scores = {}
            for concept_name, prob, layer in detected:
                if prob > threshold:
                    concept_scores[concept_name] = {
                        'probability': float(prob),
                        'layer': int(layer)
                    }

            # Get token info
            token = tokens[position_idx]
            token_id = int(input_ids[position_idx])

            # Record this position
            prompt_timeline.append({
                'position': position_idx,
                'token': token,
                'token_id': token_id,
                'concepts': concept_scores,
                'num_concepts': len(concept_scores)
            })

    return {
        'prompt': prompt,
        'tokens': tokens,
        'prompt_length': len(tokens),
        'prompt_timeline': prompt_timeline,
        'summary': {
            'total_concepts_detected': len(set(
                concept for step in prompt_timeline
                for concept in step['concepts'].keys()
            )),
            'avg_concepts_per_position': np.mean([
                step['num_concepts'] for step in prompt_timeline
            ]),
            'concept_emergence': analyze_concept_emergence(prompt_timeline)
        }
    }


def analyze_concept_emergence(timeline):
    """
    Analyze when concepts first appear in the prompt processing.

    Returns dict mapping concept_name -> first_position
    """
    emergence = {}

    for step in timeline:
        for concept_name in step['concepts'].keys():
            if concept_name not in emergence:
                emergence[concept_name] = step['position']

    # Categorize by emergence timing
    prompt_len = len(timeline)
    early_third = prompt_len // 3
    middle_third = 2 * prompt_len // 3

    categorized = {
        'early': [],   # First third
        'middle': [],  # Middle third
        'late': [],    # Last third
    }

    for concept, position in emergence.items():
        if position < early_third:
            categorized['early'].append(concept)
        elif position < middle_third:
            categorized['middle'].append(concept)
        else:
            categorized['late'].append(concept)

    return {
        'first_positions': emergence,
        'by_timing': {
            'early_concepts': len(categorized['early']),
            'middle_concepts': len(categorized['middle']),
            'late_concepts': len(categorized['late']),
        },
        'early_activation_percentage': len(categorized['early']) / len(emergence) * 100 if emergence else 0
    }


def compare_prompt_vs_generation_activations(
    model,
    tokenizer,
    lens_manager,
    prompt: str,
    max_gen_tokens: int = 20,
    threshold: float = 0.1,
    device: str = "cuda"
):
    """
    Compare concept activations during prompt vs generation phases.

    This directly tests the hypothesis that core concepts activate during
    prompt processing rather than during generation.
    """
    # Capture prompt-phase activations
    prompt_result = capture_prompt_phase_activations(
        model=model,
        tokenizer=tokenizer,
        lens_manager=lens_manager,
        prompt=prompt,
        threshold=threshold,
        device=device
    )

    # Capture generation-phase activations
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    generation_timeline = []

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_gen_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

        # Extract generated tokens
        token_ids = outputs.sequences[0][prompt_len:].cpu().tolist()
        gen_tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Process hidden states for each generation step
        for step_idx, step_states in enumerate(outputs.hidden_states):
            last_layer = step_states[-1]
            hidden_state = last_layer[:, -1, :]
            hidden_state_f32 = hidden_state.float()

            detected, _ = lens_manager.detect_and_expand(
                hidden_state_f32,
                top_k=10,
                return_timing=True
            )

            concept_scores = {}
            for concept_name, prob, layer in detected:
                if prob > threshold:
                    concept_scores[concept_name] = {
                        'probability': float(prob),
                        'layer': int(layer)
                    }

            generation_timeline.append({
                'position': step_idx,
                'token': gen_tokens[step_idx] if step_idx < len(gen_tokens) else '<eos>',
                'concepts': concept_scores
            })

    # Extract unique concepts from each phase
    prompt_concepts = set(
        concept for step in prompt_result['prompt_timeline']
        for concept in step['concepts'].keys()
    )

    gen_concepts = set(
        concept for step in generation_timeline
        for concept in step['concepts'].keys()
    )

    # Analyze overlap
    overlap = prompt_concepts & gen_concepts
    prompt_only = prompt_concepts - gen_concepts
    gen_only = gen_concepts - prompt_concepts

    return {
        'prompt': prompt,
        'prompt_phase': prompt_result,
        'generation_phase': {
            'generated_text': ''.join(gen_tokens),
            'tokens': gen_tokens,
            'timeline': generation_timeline,
            'unique_concepts': len(gen_concepts)
        },
        'comparison': {
            'prompt_concepts_count': len(prompt_concepts),
            'generation_concepts_count': len(gen_concepts),
            'overlap_count': len(overlap),
            'prompt_only_count': len(prompt_only),
            'generation_only_count': len(gen_only),
            'overlap_percentage': len(overlap) / len(prompt_concepts) * 100 if prompt_concepts else 0,
            'prompt_only_concepts': sorted(list(prompt_only))[:20],  # Top 20
            'generation_only_concepts': sorted(list(gen_only))[:20],  # Top 20
            'overlapping_concepts': sorted(list(overlap))[:20]  # Top 20
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test prompt-phase activation monitoring"
    )

    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model name (default: gemma-3-4b-pt)')
    parser.add_argument('--base-layer', type=int, default=3,
                       help='Base SUMO layer to keep loaded (default: 3)')
    parser.add_argument('--max-lenses', type=int, default=500,
                       help='Max lenses to keep loaded (default: 500)')
    parser.add_argument('--load-threshold', type=float, default=0.3,
                       help='Threshold to load child lenses (default: 0.3)')
    parser.add_argument('--detection-threshold', type=float, default=0.1,
                       help='Min probability to record concept (default: 0.1)')
    parser.add_argument('--max-gen-tokens', type=int, default=20,
                       help='Max tokens to generate for comparison (default: 20)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results/prompt_phase_tests/)')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['prompt-only', 'comparison', 'both'],
                       help='Test mode: prompt-only, comparison, or both (default: both)')

    args = parser.parse_args()

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/prompt_phase_tests/run_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PROMPT-PHASE ACTIVATION MONITORING")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    print(f"Mode: {args.mode}")
    print(f"Detection threshold: {args.detection_threshold}")

    # Load model
    print(f"\nLoading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=args.device
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize DynamicLensManager
    print("\nInitializing DynamicLensManager...")
    lens_manager = DynamicLensManager(
        lens_pack_id="gemma-3-4b-pt_sumo-wordnet-v2",
        base_layers=[args.base_layer],
        max_loaded_lenses=args.max_lenses,
        load_threshold=args.load_threshold,
        device=args.device
    )

    print(f"  Initial lenses loaded: {len(lens_manager.loaded_lenses)}")

    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)

    all_results = []

    for prompt_idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{prompt_idx + 1}/{len(TEST_PROMPTS)}] Prompt: \"{prompt}\"")
        print("-" * 80)

        if args.mode in ['prompt-only', 'both']:
            print("  Capturing prompt-phase activations...")
            prompt_result = capture_prompt_phase_activations(
                model=model,
                tokenizer=tokenizer,
                lens_manager=lens_manager,
                prompt=prompt,
                threshold=args.detection_threshold,
                device=args.device
            )

            print(f"    Prompt tokens: {prompt_result['prompt_length']}")
            print(f"    Unique concepts: {prompt_result['summary']['total_concepts_detected']}")
            print(f"    Avg concepts/position: {prompt_result['summary']['avg_concepts_per_position']:.2f}")
            print(f"    Early activation %: {prompt_result['summary']['concept_emergence']['early_activation_percentage']:.1f}%")

            # Save prompt-only result
            result_file = output_dir / f"prompt_only_{prompt_idx:03d}.json"
            with open(result_file, 'w') as f:
                json.dump(prompt_result, f, indent=2)

        if args.mode in ['comparison', 'both']:
            print("  Running prompt vs generation comparison...")
            comparison = compare_prompt_vs_generation_activations(
                model=model,
                tokenizer=tokenizer,
                lens_manager=lens_manager,
                prompt=prompt,
                max_gen_tokens=args.max_gen_tokens,
                threshold=args.detection_threshold,
                device=args.device
            )

            print(f"    Prompt concepts: {comparison['comparison']['prompt_concepts_count']}")
            print(f"    Generation concepts: {comparison['comparison']['generation_concepts_count']}")
            print(f"    Overlap: {comparison['comparison']['overlap_count']} ({comparison['comparison']['overlap_percentage']:.1f}%)")

            all_results.append(comparison)

            # Save comparison result
            result_file = output_dir / f"comparison_{prompt_idx:03d}.json"
            with open(result_file, 'w') as f:
                json.dump(comparison, f, indent=2)

    # Aggregate analysis
    if args.mode in ['comparison', 'both'] and all_results:
        print("\n" + "=" * 80)
        print("AGGREGATE ANALYSIS")
        print("=" * 80)

        avg_prompt_concepts = np.mean([r['comparison']['prompt_concepts_count'] for r in all_results])
        avg_gen_concepts = np.mean([r['comparison']['generation_concepts_count'] for r in all_results])
        avg_overlap_pct = np.mean([r['comparison']['overlap_percentage'] for r in all_results])

        print(f"\nAverage concepts per prompt phase: {avg_prompt_concepts:.1f}")
        print(f"Average concepts per generation phase: {avg_gen_concepts:.1f}")
        print(f"Average overlap: {avg_overlap_pct:.1f}%")

        # Check hypothesis: Do most concepts activate during prompt?
        if avg_overlap_pct > 70:
            print("\n✓ HYPOTHESIS CONFIRMED:")
            print("  Core concepts primarily activate during prompt processing")
            print(f"  {avg_overlap_pct:.1f}% of prompt concepts reappear in generation")
        elif avg_overlap_pct > 40:
            print("\n⚠️  HYPOTHESIS PARTIALLY CONFIRMED:")
            print("  Concepts activate in both prompt and generation phases")
            print(f"  {avg_overlap_pct:.1f}% overlap suggests both contribute")
        else:
            print("\n✗ HYPOTHESIS REJECTED:")
            print("  Generation phase activates mostly new concepts")
            print(f"  Only {avg_overlap_pct:.1f}% overlap with prompt concepts")

        # Save aggregate summary
        summary = {
            'test_config': {
                'model': args.model,
                'num_prompts': len(TEST_PROMPTS),
                'mode': args.mode,
                'detection_threshold': args.detection_threshold,
                'max_gen_tokens': args.max_gen_tokens
            },
            'aggregate_stats': {
                'avg_prompt_concepts': float(avg_prompt_concepts),
                'avg_generation_concepts': float(avg_gen_concepts),
                'avg_overlap_percentage': float(avg_overlap_pct),
            },
            'prompts': TEST_PROMPTS
        }

        summary_file = output_dir / "aggregate_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Saved aggregate summary to {summary_file}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
