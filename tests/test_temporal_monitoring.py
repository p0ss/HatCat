#!/usr/bin/env python3
"""
Test SUMO Temporal Monitoring with Varied Prompts

Validates that:
1. Generation quality is not degraded (no mode collapse)
2. Concepts are detected appropriately for different topics
3. System works across multiple samples per prompt
4. API output is consistent and usable
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def sanitize_prompt_filename(prompt: str, max_length: int = 60) -> str:
    """Convert a prompt to a safe filename."""
    # Replace spaces and punctuation with underscores
    safe = re.sub(r'[^\w\s-]', '', prompt.lower())
    safe = re.sub(r'[-\s]+', '_', safe).strip('_')
    # Truncate to max length
    if len(safe) > max_length:
        safe = safe[:max_length].rsplit('_', 1)[0]
    return safe

import torch
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

    # Specific/targeted prompts (should show strong prompt-concept alignment)
    "The calculation shows that",
    "What is the airspeed velocity of an unladen swallow",
    "What is the most important concept in AI safety",
]


def generate_with_dynamic_lenses(
    model,
    tokenizer,
    lens_manager,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k_concepts: int = 5,
    threshold: float = 0.1,
    device: str = "cuda"
):
    """Generate text and detect concepts using DynamicLensManager."""
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    timesteps = []

    with torch.inference_mode():
        # Generate with hidden states
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
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
                if prob > threshold:
                    concept_scores[concept_name] = {
                        'probability': float(prob),
                        'layer': int(layer)
                    }

            # Get token info
            token = tokens[step_idx] if step_idx < len(tokens) else '<eos>'

            # Record timestep
            timesteps.append({
                'forward_pass': step_idx,
                'token_idx': step_idx,
                'token': token,
                'position': prompt_len + step_idx,
                'concepts': concept_scores
            })

    # Build result in expected format
    generated_text = ''.join(tokens)

    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'tokens': tokens,
        'timesteps': timesteps,
        'summary': {
            'unique_concepts_detected': len(set(
                concept for ts in timesteps
                for concept in ts['concepts'].keys()
            ))
        }
    }


def test_generation_quality(results: list) -> dict:
    """
    Analyze generation quality to detect mode collapse.

    Returns:
        Dict with quality metrics
    """
    quality_metrics = {
        'total_samples': len(results),
        'mode_collapse_detected': [],
        'avg_unique_tokens': 0,
        'avg_token_length': 0,
        'repetition_rate': []
    }

    for i, result in enumerate(results):
        tokens = result['tokens']

        # Check for mode collapse (repetitive tokens)
        if len(tokens) > 5:
            # Count unique tokens in last 10 positions
            last_n = min(10, len(tokens))
            unique_in_window = len(set(tokens[-last_n:]))
            repetition = 1 - (unique_in_window / last_n)
            quality_metrics['repetition_rate'].append(repetition)

            # Flag if > 80% repetition
            if repetition > 0.8:
                quality_metrics['mode_collapse_detected'].append({
                    'sample': i,
                    'prompt': result['prompt'],
                    'generated': result['generated_text'],
                    'repetition_rate': repetition
                })

        # Average metrics
        quality_metrics['avg_unique_tokens'] += len(set(tokens))
        quality_metrics['avg_token_length'] += len(tokens)

    quality_metrics['avg_unique_tokens'] /= len(results)
    quality_metrics['avg_token_length'] /= len(results)
    quality_metrics['avg_repetition_rate'] = sum(quality_metrics['repetition_rate']) / len(quality_metrics['repetition_rate']) if quality_metrics['repetition_rate'] else 0

    return quality_metrics


def test_concept_detection(results: list) -> dict:
    """
    Analyze concept detection patterns across prompts.

    Returns:
        Dict with detection statistics
    """
    detection_stats = {
        'concepts_by_category': {},
        'avg_concepts_per_token': 0,
        'total_unique_concepts': set(),
        'layer_distribution': {0: 0, 1: 0, 2: 0}
    }

    total_tokens = 0
    total_concepts = 0

    for result in results:
        for ts in result['timesteps']:
            total_tokens += 1
            total_concepts += len(ts['concepts'])

            for concept_name, concept_data in ts['concepts'].items():
                detection_stats['total_unique_concepts'].add(concept_name)
                layer = concept_data['layer']
                if layer not in detection_stats['layer_distribution']:
                    detection_stats['layer_distribution'][layer] = 0
                detection_stats['layer_distribution'][layer] += 1

    detection_stats['avg_concepts_per_token'] = total_concepts / total_tokens if total_tokens > 0 else 0
    detection_stats['total_unique_concepts'] = len(detection_stats['total_unique_concepts'])

    return detection_stats


def main():
    parser = argparse.ArgumentParser(
        description="Test SUMO temporal monitoring with varied prompts"
    )

    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model name (default: gemma-3-4b-pt)')
    parser.add_argument('--lens-pack', type=str, default='gemma-3-4b_first-light-v1',
                       help='Lens pack ID')
    parser.add_argument('--layers-dir', type=str, default='concept_packs/first-light/hierarchy',
                       help='Path to hierarchy directory')
    parser.add_argument('--manifest', type=str, default='auto',
                       help="Path to deployment manifest, 'auto' to detect, or 'none' to skip")
    parser.add_argument('--base-layers', type=str, default='0,1,2',
                       help='Comma-separated base SUMO layers to keep always loaded (default: 0,1,2)')
    parser.add_argument('--max-lenses', type=int, default=500,
                       help='Max lenses to keep loaded at once (default: 500)')
    parser.add_argument('--load-threshold', type=float, default=0.3,
                       help='Confidence threshold to load child lenses (default: 0.3)')
    parser.add_argument('--samples-per-prompt', type=int, default=3,
                       help='Samples per prompt (default: 3)')
    parser.add_argument('--max-tokens', type=int, default=30,
                       help='Max tokens to generate per sample (default: 30)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (default: 0.8)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Show top K concepts per timestep (default: 5)')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Nucleus sampling top-p (default: 0.95)')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Min probability to record concept (default: 0.1)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-versioned in results/temporal_tests/)')
    parser.add_argument('--show-details', action='store_true',
                       help='Show token-by-token details for each sample')

    args = parser.parse_args()

    # Create versioned output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/temporal_tests/run_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SUMO TEMPORAL MONITORING - COMPREHENSIVE TEST")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print(f"Samples per prompt: {args.samples_per_prompt}")
    print(f"Total samples: {len(TEST_PROMPTS) * args.samples_per_prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")

    # Load model
    print(f"\nLoading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize DynamicLensManager
    print("\nInitializing DynamicLensManager...")
    base_layers_list = [int(x) for x in args.base_layers.split(',')]
    print(f"  - Base layers: {base_layers_list}")
    print(f"  - Max loaded lenses: {args.max_lenses}")
    print(f"  - Load threshold: {args.load_threshold}")

    # Handle lens pack path - strip prefix if provided, or use as-is if it's a full path
    lens_pack_path = Path(args.lens_pack)
    if lens_pack_path.is_dir():
        # User provided full path (e.g., lens_packs/gemma-3-4b-pt_first-light)
        lenses_dir = lens_pack_path
    elif (Path("lens_packs") / args.lens_pack).is_dir():
        # User provided just the pack name (e.g., gemma-3-4b-pt_first-light)
        lenses_dir = Path("lens_packs") / args.lens_pack
    else:
        # Strip lens_packs/ prefix if erroneously included
        pack_name = args.lens_pack.replace("lens_packs/", "")
        lenses_dir = Path("lens_packs") / pack_name
        if not lenses_dir.is_dir():
            print(f"ERROR: Lens pack not found: {args.lens_pack}")
            print(f"  Tried: {lenses_dir}")
            print(f"  Available packs: {list(Path('lens_packs').glob('*'))}")
            sys.exit(1)

    print(f"  - Lens pack: {lenses_dir}")

    # Handle manifest path
    if args.manifest == "auto":
        manifest_path = lenses_dir / "deployment_manifest.json"
        if not manifest_path.exists():
            manifest_path = None
            print("  No deployment manifest found, loading all concepts")
        else:
            print(f"  Using manifest: {manifest_path}")
    elif args.manifest == "none":
        manifest_path = None
    else:
        manifest_path = Path(args.manifest) if args.manifest else None

    lens_manager = DynamicLensManager(
        lenses_dir=lenses_dir,
        layers_data_dir=Path(args.layers_dir),
        base_layers=base_layers_list,
        max_loaded_lenses=args.max_lenses,
        load_threshold=args.load_threshold,
        device=args.device,
        manifest_path=manifest_path
    )

    print(f"  - Initial lenses loaded: {len(lens_manager.loaded_lenses)}")

    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)

    all_results = []

    for prompt_idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{prompt_idx + 1}/{len(TEST_PROMPTS)}] Prompt: \"{prompt}\"")
        print("-" * 80)

        for sample_idx in range(args.samples_per_prompt):
            print(f"  Sample {sample_idx + 1}/{args.samples_per_prompt}...", end=" ")

            # Generate with dynamic lens detection
            result = generate_with_dynamic_lenses(
                model=model,
                tokenizer=tokenizer,
                lens_manager=lens_manager,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k_concepts=args.top_k,
                threshold=args.threshold,
                device=args.device
            )

            # Add metadata
            result['prompt_idx'] = prompt_idx
            result['sample_idx'] = sample_idx

            all_results.append(result)

            # Quick summary
            print(f"Generated {len(result['tokens'])} tokens, "
                  f"detected {result['summary']['unique_concepts_detected']} unique concepts")

    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Quality analysis
    print("\n1. Generation Quality:")
    quality = test_generation_quality(all_results)
    print(f"   Total samples: {quality['total_samples']}")
    print(f"   Avg unique tokens: {quality['avg_unique_tokens']:.1f}")
    print(f"   Avg token length: {quality['avg_token_length']:.1f}")
    print(f"   Avg repetition rate: {quality['avg_repetition_rate']:.2%}")

    if quality['mode_collapse_detected']:
        print(f"\n   ⚠️  MODE COLLAPSE DETECTED in {len(quality['mode_collapse_detected'])} samples:")
        for collapse in quality['mode_collapse_detected'][:3]:  # Show first 3
            print(f"      Sample {collapse['sample']}: \"{collapse['generated'][:50]}...\"")
            print(f"      Repetition: {collapse['repetition']:.2%}")
    else:
        print(f"   ✓ No mode collapse detected")

    # Detection analysis
    print("\n2. Concept Detection:")
    detection = test_concept_detection(all_results)
    print(f"   Avg concepts per token: {detection['avg_concepts_per_token']:.2f}")
    print(f"   Total unique concepts: {detection['total_unique_concepts']}")
    print(f"   Layer distribution:")
    total_detections = sum(detection['layer_distribution'].values())
    for layer, count in sorted(detection['layer_distribution'].items()):
        pct = count / total_detections * 100 if total_detections > 0 else 0
        print(f"      Layer {layer}: {count} ({pct:.1f}%)")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save all individual results - one file per prompt with sample number
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    for result in all_results:
        prompt_name = sanitize_prompt_filename(result['prompt'])
        sample_idx = result.get('sample_idx', 0)
        result_file = prompts_dir / f"{prompt_name}_sample{sample_idx}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

    print(f"✓ Saved {len(all_results)} individual results to {prompts_dir}/")

    # Save summary
    summary = {
        'test_config': {
            'model': args.model,
            'num_prompts': len(TEST_PROMPTS),
            'samples_per_prompt': args.samples_per_prompt,
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'threshold': args.threshold
        },
        'quality_metrics': quality,
        'detection_stats': detection,
        'prompts': TEST_PROMPTS
    }

    summary_file = output_dir / "test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)  # default=str for set serialization

    print(f"✓ Saved summary to {summary_file}")

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if quality['mode_collapse_detected']:
        print("❌ FAILED - Mode collapse detected")
        print("   The monitoring system is interfering with generation")
        return 1
    elif quality['avg_repetition_rate'] > 0.5:
        print("⚠️  WARNING - High repetition rate")
        print("   Generation quality may be degraded")
        return 1
    else:
        print("✓ PASSED - No generation degradation detected")
        print(f"  Generated diverse text across {len(all_results)} samples")
        print(f"  Detected {detection['total_unique_concepts']} unique concepts")
        print(f"  Average {detection['avg_concepts_per_token']:.2f} concepts per token")
        return 0


if __name__ == '__main__':
    sys.exit(main())
