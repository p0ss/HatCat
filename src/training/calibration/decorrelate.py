"""
PCA-based decorrelation for lens packs.

The hypothesis: over-firing happens because lenses share common response patterns
rather than learning truly discriminative features. This module:

1. Loads all lens weights from a pack
2. Uses PCA to identify shared components across all lenses
3. Removes these shared components, leaving only discriminative features
4. Tests the result and applies soft limiting to remaining over-firers

This is a one-shot mathematical transformation, not iterative training.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
from tqdm import tqdm
import json


@dataclass
class DecorrelationReport:
    """Report from decorrelation process."""
    lens_pack_id: str
    total_lenses: int
    variance_explained_by_shared: float  # How much variance was in shared components
    components_removed: int
    lenses_by_layer: Dict[int, int]


def load_lens_weights(lens_pack_dir: Path) -> Tuple[Dict[str, Tuple[Path, int, torch.Tensor]], int]:
    """
    Load output layer weights from all lenses in pack.

    Returns:
        Dict mapping concept_name -> (path, layer, output_weights)
        hidden_dim (inferred from first lens)
    """
    lens_data = {}
    hidden_dim = None

    for layer_dir in sorted(lens_pack_dir.glob('layer*')):
        layer_num = int(layer_dir.name.replace('layer', ''))

        for lens_file in layer_dir.glob('*_classifier.pt'):
            concept_name = lens_file.stem.replace('_classifier', '')
            state_dict = torch.load(lens_file, map_location='cpu')

            # Get output layer weights - handle both key formats
            if 'net.6.weight' in state_dict:
                output_weight = state_dict['net.6.weight']
                output_bias = state_dict['net.6.bias']
            elif '6.weight' in state_dict:
                output_weight = state_dict['6.weight']
                output_bias = state_dict['6.bias']
            else:
                print(f"  Warning: Unknown state dict format for {concept_name}")
                continue

            if hidden_dim is None:
                hidden_dim = output_weight.shape[1]

            # Concatenate weight and bias for full output layer representation
            # Weight is [1, 64], bias is [1], so combined is [65]
            combined = torch.cat([output_weight.flatten(), output_bias.flatten()])

            lens_data[concept_name] = (lens_file, layer_num, combined)

    return lens_data, hidden_dim


def decorrelate_pack(
    lens_pack_dir: Path,
    n_components_to_remove: int = 1,
    variance_threshold: Optional[float] = None,
    dry_run: bool = False,
) -> DecorrelationReport:
    """
    Remove shared components from all lenses using PCA.

    Args:
        lens_pack_dir: Path to lens pack
        n_components_to_remove: Number of top PCs to remove (default 1 = mean direction)
        variance_threshold: If set, remove components explaining this much variance
        dry_run: If True, analyze but don't modify files

    Returns:
        DecorrelationReport with statistics
    """
    lens_pack_dir = Path(lens_pack_dir)
    print(f"Loading lenses from {lens_pack_dir}...")

    lens_data, hidden_dim = load_lens_weights(lens_pack_dir)
    print(f"  Loaded {len(lens_data)} lenses (hidden_dim={hidden_dim})")

    # Count by layer
    lenses_by_layer = {}
    for name, (path, layer, weights) in lens_data.items():
        lenses_by_layer[layer] = lenses_by_layer.get(layer, 0) + 1

    # Stack all output weights into matrix
    concept_names = list(lens_data.keys())
    weight_matrix = torch.stack([lens_data[name][2] for name in concept_names])
    weight_matrix_np = weight_matrix.numpy()

    print(f"  Weight matrix shape: {weight_matrix_np.shape}")

    # Fit PCA
    print(f"\nPerforming PCA...")
    pca = PCA()
    pca.fit(weight_matrix_np)

    # Analyze variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"  Variance explained by top components:")
    for i in range(min(10, len(cumulative_variance))):
        print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}% (cumulative: {cumulative_variance[i]*100:.2f}%)")

    # Determine components to remove
    if variance_threshold is not None:
        n_components_to_remove = np.searchsorted(cumulative_variance, variance_threshold) + 1
        print(f"\n  Removing {n_components_to_remove} components to hit {variance_threshold*100:.1f}% variance threshold")

    variance_removed = cumulative_variance[n_components_to_remove - 1]
    print(f"\n  Removing {n_components_to_remove} component(s) ({variance_removed*100:.2f}% of variance)")

    if dry_run:
        print("\n  DRY RUN - not modifying files")
        return DecorrelationReport(
            lens_pack_id=lens_pack_dir.name,
            total_lenses=len(lens_data),
            variance_explained_by_shared=variance_removed,
            components_removed=n_components_to_remove,
            lenses_by_layer=lenses_by_layer,
        )

    # Project out top components
    # Transform to PC space, zero out top components, transform back
    transformed = pca.transform(weight_matrix_np)
    transformed[:, :n_components_to_remove] = 0
    reconstructed = pca.inverse_transform(transformed)

    # Save modified lenses
    print(f"\nSaving decorrelated lenses...")
    for i, concept_name in enumerate(tqdm(concept_names, desc="Saving")):
        path, layer, original_weights = lens_data[concept_name]

        # Load full state dict
        state_dict = torch.load(path, map_location='cpu')

        # Extract new weights and bias from reconstructed
        new_combined = torch.tensor(reconstructed[i], dtype=torch.float32)
        new_weight = new_combined[:-1].reshape(1, -1)  # [1, 64]
        new_bias = new_combined[-1:].reshape(1)  # [1]

        # Update state dict
        if 'net.6.weight' in state_dict:
            state_dict['net.6.weight'] = new_weight
            state_dict['net.6.bias'] = new_bias
        else:
            state_dict['6.weight'] = new_weight
            state_dict['6.bias'] = new_bias

        torch.save(state_dict, path)

    print(f"\n  Decorrelation complete!")
    print(f"  Removed {variance_removed*100:.2f}% shared variance from {len(lens_data)} lenses")

    return DecorrelationReport(
        lens_pack_id=lens_pack_dir.name,
        total_lenses=len(lens_data),
        variance_explained_by_shared=variance_removed,
        components_removed=n_components_to_remove,
        lenses_by_layer=lenses_by_layer,
    )


def apply_soft_limiter(
    lens_pack_dir: Path,
    over_fire_counts: Dict[str, int],
    total_concepts: int,
) -> int:
    """
    Apply soft limiting to over-firers proportional to their intrusion %.

    Unlike weight modification, this shifts the bias to reduce overall firing
    without destroying learned features.

    Args:
        lens_pack_dir: Path to lens pack
        over_fire_counts: Dict mapping concept_name -> over_fire_count
        total_concepts: Total concepts in pack (for calculating %)

    Returns:
        Number of lenses limited
    """
    lens_pack_dir = Path(lens_pack_dir)
    limited = 0

    for concept_name, over_fire_count in tqdm(over_fire_counts.items(), desc="Applying soft limiter"):
        over_fire_pct = over_fire_count / total_concepts

        # Find lens file
        lens_file = None
        for layer_dir in lens_pack_dir.glob('layer*'):
            potential = layer_dir / f'{concept_name}_classifier.pt'
            if potential.exists():
                lens_file = potential
                break

        if lens_file is None:
            continue

        # Soft limiter: shift bias proportionally to over-firing
        # This reduces firing probability without destroying weight structure
        # More over-firing = more bias reduction
        bias_shift = -3.0 * over_fire_pct  # Max -3.0 at 100%

        state_dict = torch.load(lens_file, map_location='cpu')

        if 'net.6.bias' in state_dict:
            state_dict['net.6.bias'] += bias_shift
        else:
            state_dict['6.bias'] += bias_shift

        torch.save(state_dict, lens_file)
        limited += 1

    return limited


def run_full_calibration(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    model_name: str,
    n_components: int = 1,
    device: str = 'cuda',
) -> Dict:
    """
    Full calibration pipeline:
    1. Decorrelate (remove shared components)
    2. Analyze (find remaining over-firers)
    3. Soft limit (muffle remaining over-firers)

    Args:
        lens_pack_dir: Path to lens pack
        concept_pack_dir: Path to concept pack
        model_name: Model name for analysis
        n_components: Number of PCA components to remove
        device: Compute device

    Returns:
        Dict with calibration results
    """
    from .batched_analysis import run_production_analysis

    lens_pack_dir = Path(lens_pack_dir)
    concept_pack_dir = Path(concept_pack_dir)

    print("=" * 80)
    print("DECORRELATION-BASED CALIBRATION")
    print("=" * 80)

    # Step 1: Decorrelate
    print("\n--- Step 1: PCA Decorrelation ---")
    decorr_report = decorrelate_pack(
        lens_pack_dir,
        n_components_to_remove=n_components,
    )

    # Step 2: Analyze
    print("\n--- Step 2: Post-Decorrelation Analysis ---")
    analysis = run_production_analysis(
        lens_pack_dir=lens_pack_dir,
        concept_pack_dir=concept_pack_dir,
        model_name=model_name,
        device=device,
    )

    over_firing = analysis.over_firing
    print(f"\n  Over-firers remaining: {len(over_firing)}")

    if over_firing:
        # Step 3: Soft limit remaining over-firers
        print("\n--- Step 3: Soft Limiting Remaining Over-firers ---")

        over_fire_counts = {
            name: analysis.lens_reports[name]['over_fire_count']
            for name in over_firing
            if name in analysis.lens_reports
        }

        limited = apply_soft_limiter(
            lens_pack_dir,
            over_fire_counts,
            len(analysis.lens_reports),
        )
        print(f"  Applied soft limiter to {limited} lenses")

        # Calculate "junk" percentage
        total_limited_intrusion = sum(over_fire_counts.values())
        junk_score = total_limited_intrusion / (len(analysis.lens_reports) * len(over_fire_counts)) if over_fire_counts else 0
        print(f"\n  Pack quality assessment:")
        print(f"    Lenses needing limiting: {len(over_firing)} ({len(over_firing)/len(analysis.lens_reports)*100:.1f}%)")
        print(f"    These lenses likely didn't learn discriminative features")
    else:
        print("\n  No over-firers remaining - pack is clean!")
        junk_score = 0

    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)

    return {
        'decorrelation': decorr_report,
        'over_firers_after_decorr': len(over_firing),
        'junk_percentage': len(over_firing) / len(analysis.lens_reports) * 100 if analysis.lens_reports else 0,
    }


def run_soft_limit_only(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    model_name: str,
    device: str = 'cuda',
) -> Dict:
    """
    Simplified calibration: just analyze and soft limit.

    Since PCA shows weights are already decorrelated, the over-firing issue
    is in activation space not weight space. We just need to:
    1. Analyze to find over-firers
    2. Apply soft limiting proportional to intrusion %

    This treats over-firers as "lenses that didn't learn discriminative features"
    and muffles their signal proportionally.
    """
    from .batched_analysis import run_production_analysis
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lens_pack_dir = Path(lens_pack_dir)
    concept_pack_dir = Path(concept_pack_dir)

    print("=" * 80)
    print("SOFT LIMITER CALIBRATION")
    print("=" * 80)

    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # Get layers from pack
    pack_info_path = lens_pack_dir / 'pack_info.json'
    if pack_info_path.exists():
        with open(pack_info_path) as f:
            pack_info = json.load(f)
        layers = pack_info.get('trained_layers', [0, 1, 2, 3, 4, 5, 6])
    else:
        layers = [0, 1, 2, 3, 4, 5, 6]

    # Step 1: Analyze
    print("\n--- Step 1: Analysis ---")
    analysis = run_production_analysis(
        lens_pack_dir=lens_pack_dir,
        concept_pack_dir=concept_pack_dir,
        model=model,
        tokenizer=tokenizer,
        device=device,
        layers=layers,
    )

    over_firing = analysis.over_firing
    total_lenses = len(analysis.lens_reports)
    print(f"\n  Total lenses: {total_lenses}")
    print(f"  Over-firers detected: {len(over_firing)}")

    if over_firing:
        # Step 2: Soft limit over-firers
        print("\n--- Step 2: Soft Limiting ---")

        over_fire_counts = {
            name: analysis.lens_reports[name]['over_fire_count']
            for name in over_firing
            if name in analysis.lens_reports
        }

        # Show top over-firers
        sorted_over = sorted(over_fire_counts.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top 10 over-firers:")
        for name, count in sorted_over:
            pct = count / total_lenses * 100
            bias_shift = -3.0 * (count / total_lenses)
            print(f"    {name}: {count} ({pct:.1f}%) → bias shift {bias_shift:.2f}")

        limited = apply_soft_limiter(
            lens_pack_dir,
            over_fire_counts,
            total_lenses,
        )

        junk_pct = len(over_firing) / total_lenses * 100
        print(f"\n  Pack quality assessment:")
        print(f"    Lenses limited: {limited} ({junk_pct:.1f}% of pack)")
        print(f"    These lenses likely respond to noise rather than discriminative features")
    else:
        print("\n  No over-firers - pack is clean!")
        junk_pct = 0

    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)

    return {
        'over_firers': len(over_firing),
        'total_lenses': total_lenses,
        'junk_percentage': junk_pct,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Lens pack calibration via decorrelation or soft limiting')
    parser.add_argument('--lens-pack', required=True, help='Path to lens pack')
    parser.add_argument('--concept-pack', help='Path to concept pack (for analysis)')
    parser.add_argument('--model', help='Model name (for analysis)')
    parser.add_argument('--n-components', type=int, default=1, help='Number of PCA components to remove')
    parser.add_argument('--variance-threshold', type=float, help='Remove components up to this variance threshold')
    parser.add_argument('--dry-run', action='store_true', help='Analyze only, do not modify')
    parser.add_argument('--decorrelate', action='store_true', help='Run PCA decorrelation')
    parser.add_argument('--soft-limit', action='store_true', help='Run soft limiting only (recommended)')
    parser.add_argument('--full', action='store_true', help='Run full pipeline (decorr + analysis + limit)')

    args = parser.parse_args()

    if args.soft_limit:
        if not args.concept_pack or not args.model:
            parser.error("--soft-limit requires --concept-pack and --model")

        run_soft_limit_only(
            lens_pack_dir=args.lens_pack,
            concept_pack_dir=args.concept_pack,
            model_name=args.model,
        )
    elif args.full:
        if not args.concept_pack or not args.model:
            parser.error("--full requires --concept-pack and --model")

        run_full_calibration(
            lens_pack_dir=args.lens_pack,
            concept_pack_dir=args.concept_pack,
            model_name=args.model,
            n_components=args.n_components,
        )
    elif args.decorrelate or args.dry_run:
        decorrelate_pack(
            lens_pack_dir=args.lens_pack,
            n_components_to_remove=args.n_components,
            variance_threshold=args.variance_threshold,
            dry_run=args.dry_run,
        )
    else:
        parser.print_help()
