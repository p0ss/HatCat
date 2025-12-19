#!/usr/bin/env python3
"""
Pack-Level Calibration Analysis

After training individual lenses, this script runs a pack-wide calibration pass
to identify lenses that:
1. Under-fire: Target lens not in top-k when it should be
2. Over-fire: Non-target lenses appearing in top-k when they shouldn't

Usage:
    # Full analysis (prompt + generation)
    python -m calibration.analysis \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509

    # Fast mode (prompt only - no generation)
    python -m calibration.analysis \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509 \
        --fast-mode

    # Limit to specific layers
    python -m calibration.analysis \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509 \
        --layers 3 4 5
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Module is now in src/calibration/, so parent.parent is src/
# No path manipulation needed when running as module


@dataclass
class ConceptProbeResult:
    """Result of probing a single concept."""
    concept: str
    layer: int
    prompt: str
    target_activation: float
    target_rank: int
    top_k_activations: List[Tuple[str, float, int]]  # (concept, activation, layer)
    is_in_top_k: bool
    over_firing_concepts: List[Tuple[str, float, int]]  # Non-target concepts with high activation


@dataclass
class LensCalibrationReport:
    """Calibration report for a single lens."""
    concept: str
    layer: int
    probe_count: int
    in_top_k_count: int
    in_top_k_rate: float
    avg_rank: float
    avg_activation: float
    needs_boost: bool  # Under-firing
    over_fire_count: int  # Times this lens over-fired on other concepts
    over_fire_on: List[str]  # Concepts this lens over-fires on


@dataclass
class PackCalibrationAnalysis:
    """Full pack calibration analysis results."""
    lens_pack_id: str
    concept_pack_id: str
    model_id: str
    timestamp: str
    mode: str  # 'full' or 'fast'
    top_k: int
    total_concepts_probed: int
    total_probes: int

    # Summary stats
    avg_in_top_k_rate: float
    concepts_needing_boost: int
    concepts_over_firing: int

    # Detailed reports
    lens_reports: Dict[str, LensCalibrationReport] = field(default_factory=dict)

    # Concepts grouped by issue
    under_firing: List[str] = field(default_factory=list)  # Target not in top-k
    over_firing: List[str] = field(default_factory=list)   # Appears when shouldn't
    well_calibrated: List[str] = field(default_factory=list)


def load_concept_prompts(concept_pack_dir: Path, layers: List[int], fast_mode: bool = False) -> Dict[str, Dict]:
    """
    Load concepts and generate probe prompts from training hints.

    Args:
        concept_pack_dir: Path to concept pack
        layers: Which layers to load
        fast_mode: If True, just use concept name as prompt (faster, no generation needed)

    Returns:
        Dict mapping concept name to {
            'layer': int,
            'definition': str,
            'prompts': List[str],  # Generated from training hints
            'domain': str,
        }
    """
    concepts = {}

    for layer in layers:
        layer_file = concept_pack_dir / "hierarchy" / f"layer{layer}.json"
        if not layer_file.exists():
            print(f"  Warning: Layer file not found: {layer_file}")
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        concept_list = layer_data.get('concepts', layer_data if isinstance(layer_data, list) else [])

        for concept in concept_list:
            term = concept.get('sumo_term') or concept.get('term')
            if not term:
                continue

            definition = concept.get('definition', '')

            if fast_mode:
                # Fast mode: just use the concept name - sufficient for activation probing
                prompts = [term]
            else:
                # Full mode: use training hints or definition-based prompts
                prompts = []

                # Use positive examples from training hints if available
                training_hints = concept.get('training_hints', {})
                positive_examples = training_hints.get('positive_examples', [])
                if positive_examples:
                    prompts.extend(positive_examples[:3])  # Take up to 3

                # Fall back to definition-based prompt
                if definition and len(prompts) < 3:
                    prompts.append(f"Explain the concept of {term}: {definition}")

                # Generate a simple prompt if we still don't have any
                if not prompts:
                    prompts.append(f"Tell me about {term}.")

            concepts[term] = {
                'layer': layer,
                'definition': definition,
                'prompts': prompts,
                'domain': concept.get('domain', 'Unknown'),
            }

    return concepts


def extract_activation(
    model,
    tokenizer,
    prompt: str,
    device: str,
    layer_idx: int = 15,
    fast_mode: bool = False,
) -> np.ndarray:
    """
    Extract hidden state activation from model.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        device: Device string
        layer_idx: Which layer to extract from
        fast_mode: If True, only use prompt (no generation)

    Returns:
        Activation vector [hidden_dim]
    """
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if fast_mode:
            # Just run forward pass on prompt
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]
            # Take last token's hidden state (convert to float32 for numpy compatibility)
            activation = hidden_states[0, -1, :].float().cpu().numpy()
        else:
            # Generate a short response and collect hidden states from ALL tokens
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            # Collect activations from all generation steps
            # hidden_states is a tuple of (prompt_hidden_states, *generation_hidden_states)
            activations = []
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                for gen_step in range(len(outputs.hidden_states)):
                    if gen_step == 0:
                        continue  # Skip prompt hidden states
                    hidden_states = outputs.hidden_states[gen_step][layer_idx]
                    act = hidden_states[0, -1, :].float().cpu().numpy()
                    activations.append(act)

            if not activations:
                # Fallback: re-run forward pass
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]
                activation = hidden_states[0, -1, :].float().cpu().numpy()
                return activation

            # Return list of activations for full mode
            return activations

    return activation


def run_calibration_analysis(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    top_k: int = 10,
    fast_mode: bool = False,
    layer_idx: int = 15,
    max_concepts: Optional[int] = None,
) -> PackCalibrationAnalysis:
    """
    Run pack-level calibration analysis.

    For each concept:
    1. Generate probe prompt
    2. Extract activation
    3. Run ALL lenses and record activations
    4. Check if target lens is in top-k
    5. Identify over-firing lenses
    """
    from monitoring.dynamic_lens_manager import DynamicLensManager, SimpleMLP

    print(f"\n{'='*80}")
    print("PACK-LEVEL CALIBRATION ANALYSIS")
    print(f"{'='*80}")
    print(f"  Lens pack: {lens_pack_dir}")
    print(f"  Concept pack: {concept_pack_dir}")
    print(f"  Mode: {'fast (prompt only)' if fast_mode else 'full (with generation)'}")
    print(f"  Top-k: {top_k}")
    print(f"  Layers: {layers}")

    # Load concepts
    print(f"\nLoading concepts...")
    concepts = load_concept_prompts(concept_pack_dir, layers, fast_mode=fast_mode)
    print(f"  Loaded {len(concepts)} concepts with prompts")

    if max_concepts:
        # Limit for testing
        concept_names = list(concepts.keys())[:max_concepts]
        concepts = {k: concepts[k] for k in concept_names}
        print(f"  Limited to {len(concepts)} concepts for testing")

    # Load all lenses
    print(f"\nLoading lenses...")
    lenses = {}  # (concept, layer) -> model
    hidden_dim = None

    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if not layer_dir.exists():
            continue

        lens_files = list(layer_dir.glob("*_classifier.pt"))
        for lens_file in lens_files:
            concept_name = lens_file.stem.replace('_classifier', '')

            state_dict = torch.load(lens_file, map_location='cpu')

            # Infer hidden dim
            if hidden_dim is None:
                first_key = list(state_dict.keys())[0]
                hidden_dim = state_dict[first_key].shape[1]
                print(f"  Hidden dim: {hidden_dim}")

            # Handle key prefix mismatch
            if not list(state_dict.keys())[0].startswith('net.'):
                state_dict = {f'net.{k}': v for k, v in state_dict.items()}

            # Store state dict directly - create model on-demand to save GPU memory
            lenses[(concept_name, layer)] = state_dict

    print(f"  Loaded {len(lenses)} lenses")

    # Create a single reusable model for scoring (on GPU)
    scoring_model = SimpleMLP(hidden_dim).to(device)
    scoring_model.eval()

    # Create LayerNorm for activation normalization
    layer_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False).to(device)

    # Run probes
    print(f"\nRunning probes...")
    probe_results: List[ConceptProbeResult] = []

    # Track over-firing: concept -> list of concepts it over-fired on
    over_fire_tracker: Dict[str, List[str]] = defaultdict(list)

    for concept_name, concept_data in tqdm(concepts.items(), desc="Probing concepts"):
        layer = concept_data['layer']

        # Check if we have a lens for this concept
        if (concept_name, layer) not in lenses:
            continue

        # Probe with each prompt
        for prompt in concept_data['prompts'][:1]:  # Just use first prompt for now
            try:
                # Extract activation(s)
                activation_result = extract_activation(
                    model, tokenizer, prompt, device,
                    layer_idx=layer_idx, fast_mode=fast_mode
                )

                # Handle both single activation (fast mode) and list (full mode)
                if isinstance(activation_result, list):
                    activations = activation_result
                else:
                    activations = [activation_result]

                # Find best rank across all activations
                best_rank = float('inf')
                best_score = None
                best_top_k = None

                for activation in activations:
                    # Normalize activation (convert to float32 for lens compatibility)
                    act_tensor = torch.tensor(activation, dtype=torch.float32).to(device)
                    act_tensor = layer_norm(act_tensor.unsqueeze(0)).float()

                    # Run ALL lenses using reusable scoring model
                    all_scores = []
                    with torch.no_grad():
                        for (lens_concept, lens_layer), state_dict in lenses.items():
                            scoring_model.load_state_dict(state_dict)
                            score = scoring_model(act_tensor).item()
                            all_scores.append((lens_concept, score, lens_layer))

                    # Sort by score
                    all_scores.sort(key=lambda x: x[1], reverse=True)

                    # Get target lens score and rank
                    for rank, (c, s, l) in enumerate(all_scores, 1):
                        if c == concept_name and l == layer:
                            if rank < best_rank:
                                best_rank = rank
                                best_score = s
                                best_top_k = all_scores[:top_k]
                            break

                if best_score is None:
                    continue

                target_rank = best_rank
                target_score = best_score
                top_k_results = best_top_k

                # Get top-k
                is_in_top_k = target_rank <= top_k

                # Identify over-firing concepts (in top-k but shouldn't be)
                over_firing = []
                for c, s, l in top_k_results:
                    if c != concept_name:
                        over_firing.append((c, s, l))
                        over_fire_tracker[c].append(concept_name)

                probe_results.append(ConceptProbeResult(
                    concept=concept_name,
                    layer=layer,
                    prompt=prompt,
                    target_activation=target_score,
                    target_rank=target_rank,
                    top_k_activations=top_k_results,
                    is_in_top_k=is_in_top_k,
                    over_firing_concepts=over_firing,
                ))

            except Exception as e:
                print(f"  Error probing {concept_name}: {e}")
                continue

    print(f"  Completed {len(probe_results)} probes")

    # Aggregate into lens reports
    print(f"\nAggregating results...")
    lens_reports: Dict[str, LensCalibrationReport] = {}

    # Group probes by concept
    probes_by_concept: Dict[str, List[ConceptProbeResult]] = defaultdict(list)
    for probe in probe_results:
        probes_by_concept[probe.concept].append(probe)

    under_firing = []
    over_firing = []
    well_calibrated = []

    for concept_name, probes in probes_by_concept.items():
        if not probes:
            continue

        layer = probes[0].layer
        in_top_k_count = sum(1 for p in probes if p.is_in_top_k)
        in_top_k_rate = in_top_k_count / len(probes)
        avg_rank = np.mean([p.target_rank for p in probes])
        avg_activation = np.mean([p.target_activation for p in probes])

        # Check over-firing
        over_fire_count = len(over_fire_tracker.get(concept_name, []))
        over_fire_on = list(set(over_fire_tracker.get(concept_name, [])))[:10]  # Top 10

        # Determine if needs boost
        needs_boost = in_top_k_rate < 0.8  # Less than 80% in top-k

        report = LensCalibrationReport(
            concept=concept_name,
            layer=layer,
            probe_count=len(probes),
            in_top_k_count=in_top_k_count,
            in_top_k_rate=in_top_k_rate,
            avg_rank=avg_rank,
            avg_activation=avg_activation,
            needs_boost=needs_boost,
            over_fire_count=over_fire_count,
            over_fire_on=over_fire_on,
        )

        lens_reports[concept_name] = report

        # Categorize
        if needs_boost:
            under_firing.append(concept_name)
        elif over_fire_count > 5:  # Over-fires on more than 5 other concepts
            over_firing.append(concept_name)
        else:
            well_calibrated.append(concept_name)

    # Create analysis result
    analysis = PackCalibrationAnalysis(
        lens_pack_id=lens_pack_dir.name,
        concept_pack_id=concept_pack_dir.name,
        model_id=str(model.config._name_or_path if hasattr(model.config, '_name_or_path') else 'unknown'),
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode='fast' if fast_mode else 'full',
        top_k=top_k,
        total_concepts_probed=len(probes_by_concept),
        total_probes=len(probe_results),
        avg_in_top_k_rate=np.mean([r.in_top_k_rate for r in lens_reports.values()]) if lens_reports else 0.0,
        concepts_needing_boost=len(under_firing),
        concepts_over_firing=len(over_firing),
        lens_reports={k: asdict(v) for k, v in lens_reports.items()},
        under_firing=under_firing,
        over_firing=over_firing,
        well_calibrated=well_calibrated,
    )

    return analysis


def print_analysis_summary(analysis: PackCalibrationAnalysis):
    """Print human-readable summary of analysis."""
    print(f"\n{'='*80}")
    print("CALIBRATION ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"  Lens pack: {analysis.lens_pack_id}")
    print(f"  Concepts probed: {analysis.total_concepts_probed}")
    print(f"  Total probes: {analysis.total_probes}")
    print(f"  Mode: {analysis.mode}")
    print()
    print(f"  Average in-top-{analysis.top_k} rate: {analysis.avg_in_top_k_rate:.1%}")
    print(f"  Concepts needing boost (under-firing): {analysis.concepts_needing_boost}")
    print(f"  Concepts over-firing: {analysis.concepts_over_firing}")
    print(f"  Well calibrated: {len(analysis.well_calibrated)}")

    if analysis.under_firing:
        print(f"\n  Top 20 under-firing concepts (need boost):")
        # Sort by in_top_k_rate
        sorted_under = sorted(
            [(c, analysis.lens_reports[c]['in_top_k_rate'], analysis.lens_reports[c]['avg_rank'])
             for c in analysis.under_firing],
            key=lambda x: x[1]
        )[:20]
        for concept, rate, avg_rank in sorted_under:
            print(f"    {concept}: {rate:.1%} in top-k, avg rank {avg_rank:.1f}")

    if analysis.over_firing:
        print(f"\n  Top 20 over-firing concepts (need suppression):")
        sorted_over = sorted(
            [(c, analysis.lens_reports[c]['over_fire_count'])
             for c in analysis.over_firing],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        for concept, count in sorted_over:
            print(f"    {concept}: over-fired on {count} other concepts")


def main():
    parser = argparse.ArgumentParser(description='Pack-level calibration analysis')
    parser.add_argument('--lens-pack', required=True, help='Path to lens pack')
    parser.add_argument('--concept-pack', required=True, help='Path to concept pack')
    parser.add_argument('--model', required=True, help='Model name/path')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help='Layers to analyze (default: all)')
    parser.add_argument('--top-k', type=int, default=10, help='Top-k for analysis')
    parser.add_argument('--fast-mode', action='store_true',
                        help='Fast mode: only use prompt, no generation')
    parser.add_argument('--layer-idx', type=int, default=15,
                        help='Model layer to extract activations from')
    parser.add_argument('--max-concepts', type=int, default=None,
                        help='Limit number of concepts (for testing)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (default: lens_pack/calibration_analysis.json)')

    args = parser.parse_args()

    lens_pack_dir = Path(args.lens_pack)
    concept_pack_dir = Path(args.concept_pack)

    # Determine layers
    if args.layers is None:
        # Auto-detect from lens pack
        layers = []
        for layer_dir in lens_pack_dir.glob("layer*"):
            if layer_dir.is_dir():
                try:
                    layer_num = int(layer_dir.name.replace('layer', ''))
                    layers.append(layer_num)
                except ValueError:
                    pass
        layers.sort()
    else:
        layers = args.layers

    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    # Run analysis
    analysis = run_calibration_analysis(
        lens_pack_dir=lens_pack_dir,
        concept_pack_dir=concept_pack_dir,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        layers=layers,
        top_k=args.top_k,
        fast_mode=args.fast_mode,
        layer_idx=args.layer_idx,
        max_concepts=args.max_concepts,
    )

    # Print summary
    print_analysis_summary(analysis)

    # Save results
    output_path = Path(args.output) if args.output else lens_pack_dir / "calibration_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(asdict(analysis), f, indent=2)
    print(f"\n  Saved analysis to: {output_path}")


if __name__ == '__main__':
    main()
