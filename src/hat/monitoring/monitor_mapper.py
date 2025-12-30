#!/usr/bin/env python3
"""
Temporal Concept Monitor using Feature Weighting (ConceptActivationMapper)

Alternative to classifier-based temporal monitoring that uses
IDF-weighted centroids for concept detection. This should be
faster and potentially more coherent across the concept hierarchy.

Key differences from classifier-based approach:
1. Single matrix multiply instead of N classifier forward passes
2. Built-in hierarchical coherence via IDF weighting
3. Outputs are cosine similarities, not sigmoid probabilities
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

from training.calibration.feature_weighting import (
    ConceptActivationMapper,
    ActivationMap,
)


class MapperTemporalMonitor:
    """
    Temporal concept monitor using ConceptActivationMapper.

    Drop-in replacement for SUMOTemporalMonitor that uses feature weighting
    instead of binary classifiers.
    """

    def __init__(
        self,
        mapper: ConceptActivationMapper,
        top_k: int = 10,
        threshold: float = 0.5,  # Normalized similarity threshold
    ):
        """
        Args:
            mapper: Fitted ConceptActivationMapper
            top_k: Number of top concepts to show per timestep
            threshold: Minimum normalized activation to include
        """
        self.mapper = mapper
        self.top_k = top_k
        self.threshold = threshold

    def monitor_generation(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        device: str = "cuda",
        layer_idx: int = -1,  # Which layer to extract (-1 = last)
    ) -> Dict:
        """
        Generate text and monitor concept activations.

        Args:
            model: LLM
            tokenizer: Tokenizer
            prompt: Input prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            device: Device
            layer_idx: Which hidden layer to use for activation extraction

        Returns:
            Dict with same structure as SUMOTemporalMonitor output
        """
        model.eval()

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.inference_mode():
            # Generate with hidden states
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            # Extract generated tokens
            token_ids = outputs.sequences[0][prompt_len:].cpu().tolist()
            tokens = [tokenizer.decode([tid]) for tid in token_ids]

            # Process hidden states for each generated token
            timesteps = []

            for step_idx, step_states in enumerate(outputs.hidden_states):
                # step_states is a tuple of (num_layers,) tensors
                # Each tensor is [batch=1, seq_len, hidden_dim]
                last_layer = step_states[layer_idx]  # [1, seq_len, hidden_dim]
                hidden_state = last_layer[:, -1, :].float().cpu().numpy()  # [1, hidden_dim]

                # Get activation map from mapper
                act_map = self.mapper.compute_activations(hidden_state[0])

                # Convert to output format matching classifier-based monitor
                concept_probs = []
                for act in act_map.activations[:self.top_k]:
                    if act.normalized >= self.threshold:
                        concept_probs.append({
                            'concept': act.concept,
                            'probability': float(act.normalized),  # Use normalized for comparison
                            'raw_similarity': float(act.activation),
                            'layer': int(act.layer)
                        })

                timesteps.append({
                    'token': tokens[step_idx] if step_idx < len(tokens) else '<eos>',
                    'position': prompt_len + step_idx,
                    'concepts': concept_probs,
                    'n_active': act_map.n_active,
                    'max_activation': act_map.max_activation,
                })

        generated_text = ''.join(tokens)

        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'tokens': tokens,
            'timesteps': timesteps,
            'summary': {
                'total_tokens': len(tokens),
                'unique_concepts_detected': len(set(
                    c['concept'] for ts in timesteps for c in ts['concepts']
                )),
                'method': 'ConceptActivationMapper',
            }
        }

    def print_report(self, result: Dict, show_token_details: bool = True):
        """Print human-readable temporal detection report."""

        print("\n" + "=" * 80)
        print("TEMPORAL CONCEPT DETECTION (Feature Weighting)")
        print("=" * 80)

        print(f"\nPrompt: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
        print(f"\nTotal tokens: {result['summary']['total_tokens']}")
        print(f"Unique concepts detected: {result['summary']['unique_concepts_detected']}")

        if show_token_details:
            print("\n" + "-" * 80)
            print("TOKEN-BY-TOKEN CONCEPT DETECTION")
            print("-" * 80)

            for ts in result['timesteps']:
                print(f"\nToken [{ts['position']}]: '{ts['token']}'")
                print(f"  Active concepts: {ts.get('n_active', 'N/A')}")

                if ts['concepts']:
                    print(f"  Top {len(ts['concepts'])} concepts:")
                    for c in ts['concepts']:
                        layer_tag = f"L{c['layer']}"
                        prob_bar = "█" * int(c['probability'] * 20)
                        raw = c.get('raw_similarity', c['probability'])
                        print(f"    [{layer_tag}] {c['concept']:30s} {c['probability']:.3f} (raw: {raw:.3f}) {prob_bar}")
                else:
                    print("  (no concepts above threshold)")

        print("\n" + "=" * 80)

    def save_json(self, result: Dict, output_path: Path):
        """Save JSON output."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Saved results to: {output_path}")


def build_mapper_from_concepts(
    model,
    tokenizer,
    concept_pack_dir: Path,
    layers: List[int],
    device: str,
    layer_idx: int = 15,
    n_samples_per_concept: int = 5,
) -> ConceptActivationMapper:
    """
    Build a ConceptActivationMapper by collecting activations from concepts.

    This is a convenience function that wraps the data collection and fitting.
    """
    from training.calibration.feature_weighting import collect_concept_activations

    print(f"Building ConceptActivationMapper from {concept_pack_dir}...")

    # Collect activations
    concept_activations = collect_concept_activations(
        model=model,
        tokenizer=tokenizer,
        concept_pack_dir=concept_pack_dir,
        layers=layers,
        device=device,
        layer_idx=layer_idx,
        n_samples_per_concept=n_samples_per_concept,
        fast_mode=False,  # Use training hints for better prompts
    )

    # Build mapper
    hidden_dim = model.config.hidden_size
    mapper = ConceptActivationMapper(hidden_dim=hidden_dim)
    mapper.fit(concept_activations)

    return mapper


def run_temporal_comparison(
    model,
    tokenizer,
    prompts: List[str],
    mapper: ConceptActivationMapper,
    classifiers: Dict,
    device: str = "cuda",
    max_new_tokens: int = 20,
    output_dir: Optional[Path] = None,
):
    """
    Run both mapper and classifier-based temporal monitoring on the same prompts.

    Saves results side-by-side for comparison.
    """
    from .monitor import SUMOTemporalMonitor

    # Create monitors
    mapper_monitor = MapperTemporalMonitor(mapper, top_k=10, threshold=0.3)
    classifier_monitor = SUMOTemporalMonitor(classifiers, top_k=10, threshold=0.3)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*80}")
        print(f"PROMPT {i+1}: {prompt}")
        print('='*80)

        # Run mapper-based monitoring
        print("\n--- FEATURE WEIGHTING APPROACH ---")
        mapper_result = mapper_monitor.monitor_generation(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        mapper_monitor.print_report(mapper_result, show_token_details=True)

        # Run classifier-based monitoring
        print("\n--- CLASSIFIER APPROACH ---")
        classifier_result = classifier_monitor.monitor_generation(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        classifier_monitor.print_report(classifier_result, show_token_details=True)

        # Save if output dir provided
        if output_dir:
            with open(output_dir / f"sample_{i:03d}_mapper.json", 'w') as f:
                json.dump(mapper_result, f, indent=2)
            with open(output_dir / f"sample_{i:03d}_classifier.json", 'w') as f:
                json.dump(classifier_result, f, indent=2)


def main():
    """Run temporal test with feature weighting approach."""
    import argparse

    parser = argparse.ArgumentParser(description='Temporal monitor with feature weighting')
    parser.add_argument('--concept-pack', required=True, help='Path to concept pack')
    parser.add_argument('--model', required=True, help='Model name/path')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--layers', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6],
                        help='Hierarchy layers to include')
    parser.add_argument('--layer-idx', type=int, default=15, help='Model layer for activations')
    parser.add_argument('--n-samples', type=int, default=5, help='Samples per concept for training')
    parser.add_argument('--max-tokens', type=int, default=20, help='Max tokens to generate')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--mapper-path', type=str, default=None, help='Load pre-built mapper')
    parser.add_argument('--prompts', nargs='+', default=None, help='Custom prompts to test')

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    # Build or load mapper
    if args.mapper_path:
        print(f"Loading mapper from: {args.mapper_path}")
        mapper = ConceptActivationMapper.load(Path(args.mapper_path))
    else:
        mapper = build_mapper_from_concepts(
            model=model,
            tokenizer=tokenizer,
            concept_pack_dir=Path(args.concept_pack),
            layers=args.layers,
            device=args.device,
            layer_idx=args.layer_idx,
            n_samples_per_concept=args.n_samples,
        )

        # Save mapper for reuse
        if args.output_dir:
            mapper_path = Path(args.output_dir) / "mapper"
            mapper.save(mapper_path)

    # Default test prompts
    if args.prompts:
        prompts = args.prompts
    else:
        prompts = [
            "Artificial intelligence can help society by",
            "The ethical implications of autonomous weapons include",
            "Climate change is affecting biodiversity because",
            "The stock market crashed when",
            "Genetic engineering allows scientists to",
        ]

    # Create monitor and run
    monitor = MapperTemporalMonitor(mapper, top_k=10, threshold=0.3)

    output_dir = Path(args.output_dir) if args.output_dir else Path("results/temporal_tests/mapper_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*80}")
        print(f"PROMPT {i+1}/{len(prompts)}: {prompt}")
        print('='*80)

        result = monitor.monitor_generation(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            device=args.device,
            layer_idx=args.layer_idx,
        )

        monitor.print_report(result, show_token_details=True)
        monitor.save_json(result, output_dir / f"sample_{i:03d}.json")

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
