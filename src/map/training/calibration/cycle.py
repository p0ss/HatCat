#!/usr/bin/env python3
"""
Run Full Calibration Cycle

Iteratively runs analysis → fine-tune → re-analysis until convergence
or max cycles reached.

Usage:
    # Run calibration cycle (default 3 iterations)
    python -m training.calibration.cycle \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509

    # Run more cycles
    python -m training.calibration.cycle \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509 \
        --max-cycles 5

    # Fast mode (prompt-only analysis, faster but less accurate)
    python -m training.calibration.cycle \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509 \
        --fast-mode
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def run_calibration_cycle(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    max_cycles: int = 3,
    convergence_threshold: float = 0.05,  # Stop if improvement < 5%
    top_k: int = 10,
    fast_mode: bool = False,
    production_mode: bool = False,
    layer_idx: int = 15,
    max_finetune_epochs: int = 20,
    max_concepts: Optional[int] = None,
    run_cross_activation: bool = True,
    cross_activation_samples: int = 5,
    cross_activation_threshold: float = 0.5,
    use_activation_cache: bool = True,
    model_name: str = "",
) -> Dict:
    """
    Run iterative calibration cycles.

    Each cycle:
    1. Run analysis to identify under/over-firing lenses
    2. Fine-tune problematic lenses
    3. Re-run analysis to measure improvement
    4. Repeat until convergence or max_cycles

    After cycles complete:
    5. Run cross-activation calibration to measure per-concept noise floors
       This produces calibration.json with self_mean/cross_mean per concept,
       enabling normalized scores at inference (1.0=signal, 0.5=noise floor, 0.0=floor)

    Returns:
        Dict with cycle history and final metrics
    """
    from dataclasses import asdict

    # Choose analysis/finetune based on mode
    if production_mode:
        from .batched_analysis import run_production_analysis, print_batched_analysis_summary
        from .finetune import run_dual_criteria_finetune, print_finetune_summary
        mode_str = 'production'
    else:
        from .analysis import run_calibration_analysis, print_analysis_summary
        from .finetune import run_calibration_finetune, print_finetune_summary
        mode_str = 'fast' if fast_mode else 'full'

    print(f"\n{'='*80}")
    print("CALIBRATION CYCLE")
    print(f"{'='*80}")
    print(f"  Lens pack: {lens_pack_dir}")
    print(f"  Concept pack: {concept_pack_dir}")
    print(f"  Max cycles: {max_cycles}")
    print(f"  Convergence threshold: {convergence_threshold:.1%}")
    print(f"  Mode: {mode_str}")
    print(f"  Top-k: {top_k}")
    print(f"  Activation cache: {'enabled' if use_activation_cache else 'disabled'}")

    # Build or load activation cache
    activation_cache = None
    if use_activation_cache:
        from .activation_cache import get_or_build_cache

        print(f"\n--- Activation Cache ---")
        activation_cache = get_or_build_cache(
            lens_pack_dir=lens_pack_dir,
            concept_pack_dir=concept_pack_dir,
            model=model,
            tokenizer=tokenizer,
            device=device,
            layers=layers,
            model_layer=layer_idx,
            model_name=model_name,
            n_samples_per_concept=cross_activation_samples,
            max_concepts=max_concepts,
        )

    cycle_history = []
    previous_rate = 0.0

    for cycle in range(max_cycles):
        print(f"\n{'='*80}")
        print(f"CYCLE {cycle + 1}/{max_cycles}")
        print(f"{'='*80}")

        # Step 1: Analysis
        print(f"\n--- Step 1: Analysis ---")

        if production_mode:
            analysis = run_production_analysis(
                lens_pack_dir=lens_pack_dir,
                concept_pack_dir=concept_pack_dir,
                model=model,
                tokenizer=tokenizer,
                device=device,
                layers=layers,
                top_k=top_k,
                max_concepts=max_concepts,
                layer_idx=layer_idx,
            )
            print_batched_analysis_summary(analysis)
        else:
            analysis = run_calibration_analysis(
                lens_pack_dir=lens_pack_dir,
                concept_pack_dir=concept_pack_dir,
                model=model,
                tokenizer=tokenizer,
                device=device,
                layers=layers,
                top_k=top_k,
                fast_mode=fast_mode,
                layer_idx=layer_idx,
                max_concepts=max_concepts,
            )
            print_analysis_summary(analysis)

        current_rate = analysis.avg_in_top_k_rate
        improvement = current_rate - previous_rate
        current_under_firing = len(analysis.under_firing)
        current_over_firing = len(analysis.over_firing)  # Now tracks chronic over-firers

        cycle_record = {
            'cycle': cycle + 1,
            'in_top_k_rate': current_rate,
            'under_firing': current_under_firing,
            'chronic_over_firers': current_over_firing,  # Renamed for clarity
            'improvement': improvement,
        }
        cycle_history.append(cycle_record)

        # Save analysis
        analysis_path = lens_pack_dir / f"calibration_analysis_cycle{cycle+1}.json"
        with open(analysis_path, 'w') as f:
            json.dump(asdict(analysis), f, indent=2)
        print(f"\n  Saved analysis to: {analysis_path}")

        # Check convergence - based on top-k rate only (not over-firing)
        if cycle > 0 and improvement < convergence_threshold:
            print(f"\n  Converged! Improvement ({improvement:.1%}) < threshold ({convergence_threshold:.1%})")
            # Don't break if there are still chronic over-firers to suppress
            if current_over_firing == 0:
                break
            else:
                print(f"  But {current_over_firing} chronic over-firers remain - continuing to suppress")

        # Check if fully calibrated (no under-firing AND no chronic over-firers)
        if not analysis.under_firing and current_over_firing == 0:
            print(f"\n  All lenses well-calibrated!")
            break

        # Step 2: Fine-tuning
        print(f"\n--- Step 2: Fine-tuning ---")
        analysis_dict = asdict(analysis)

        if production_mode:
            finetune_report = run_dual_criteria_finetune(
                lens_pack_dir=lens_pack_dir,
                concept_pack_dir=concept_pack_dir,
                analysis=analysis_dict,
                model=model,
                tokenizer=tokenizer,
                device=device,
                layers=layers,
                max_epochs=max_finetune_epochs,
                layer_idx=layer_idx,
            )
        else:
            finetune_report = run_calibration_finetune(
                lens_pack_dir=lens_pack_dir,
                concept_pack_dir=concept_pack_dir,
                analysis=analysis_dict,
                model=model,
                tokenizer=tokenizer,
                device=device,
                layers=layers,
                max_epochs=max_finetune_epochs,
                fast_mode=fast_mode,
                layer_idx=layer_idx,
            )

        print_finetune_summary(finetune_report)

        # Save finetune report
        finetune_path = lens_pack_dir / f"calibration_finetune_cycle{cycle+1}.json"
        with open(finetune_path, 'w') as f:
            json.dump(asdict(finetune_report), f, indent=2, default=str)
        print(f"\n  Saved finetune report to: {finetune_path}")

        previous_rate = current_rate

    # Final summary
    print(f"\n{'='*80}")
    print("CALIBRATION CYCLE COMPLETE")
    print(f"{'='*80}")
    print(f"\n  Cycles completed: {len(cycle_history)}")
    print(f"\n  Progress:")
    for record in cycle_history:
        over_fire_count = record.get('chronic_over_firers', record.get('over_firing', 0))
        print(f"    Cycle {record['cycle']}: {record['in_top_k_rate']:.1%} in top-k "
              f"(+{record['improvement']:.1%}), "
              f"{record['under_firing']} under-firing, "
              f"{over_fire_count} chronic over-firers")

    if cycle_history:
        total_improvement = cycle_history[-1]['in_top_k_rate'] - cycle_history[0]['in_top_k_rate'] + cycle_history[0]['improvement']
        print(f"\n  Total improvement: {total_improvement:.1%}")

    # Step 3: Cross-activation calibration (final step)
    cross_activation_result = None
    if run_cross_activation:
        print(f"\n{'='*80}")
        print("CROSS-ACTIVATION CALIBRATION")
        print(f"{'='*80}")
        print("  Measuring per-concept noise floors for normalized scoring...")

        from .cross_activation import run_cross_activation_calibration

        cross_activation_result = run_cross_activation_calibration(
            lens_pack_dir=lens_pack_dir,
            concept_pack_dir=concept_pack_dir,
            model=model,
            tokenizer=tokenizer,
            device=device,
            layers=layers,
            n_samples_per_concept=cross_activation_samples,
            firing_threshold=cross_activation_threshold,
            layer_idx=layer_idx,
            max_concepts=max_concepts,
            activation_cache=activation_cache,
        )

        # Save calibration.json
        calibration_path = lens_pack_dir / "calibration.json"
        with open(calibration_path, 'w') as f:
            json.dump(cross_activation_result, f, indent=2)
        print(f"\n  ✓ Saved calibration to: {calibration_path}")
        print(f"  ✓ Calibrated {len(cross_activation_result.get('calibration', {}))} concepts")

        # Show top over-firers
        cal_data = cross_activation_result.get('calibration', {})
        if cal_data:
            by_cross_rate = sorted(cal_data.values(), key=lambda x: x.get('cross_fire_rate', 0), reverse=True)
            print(f"\n  Top over-firers (by cross-fire rate):")
            for i, c in enumerate(by_cross_rate[:10]):
                print(f"    {i+1:2d}. {c['concept']:35s} L{c['layer']} "
                      f"cross_rate={c['cross_fire_rate']:.3f} "
                      f"self={c['self_mean']:.2f} cross={c['cross_mean']:.2f}")

    # Save summary
    summary = {
        'lens_pack_id': lens_pack_dir.name,
        'concept_pack_id': concept_pack_dir.name,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'cycles_completed': len(cycle_history),
        'max_cycles': max_cycles,
        'mode': 'fast' if fast_mode else 'full',
        'cycle_history': cycle_history,
        'final_in_top_k_rate': cycle_history[-1]['in_top_k_rate'] if cycle_history else 0,
        'cross_activation_calibration': {
            'enabled': run_cross_activation,
            'concepts_calibrated': len(cross_activation_result.get('calibration', {})) if cross_activation_result else 0,
        } if run_cross_activation else None,
    }

    summary_path = lens_pack_dir / "calibration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run full calibration cycle')
    parser.add_argument('--lens-pack', required=True, help='Path to lens pack')
    parser.add_argument('--concept-pack', required=True, help='Path to concept pack')
    parser.add_argument('--model', required=True, help='Model name/path')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help='Layers to process')
    parser.add_argument('--max-cycles', type=int, default=20,
                        help='Max calibration cycles')
    parser.add_argument('--convergence-threshold', type=float, default=0.05,
                        help='Stop when improvement < this')
    parser.add_argument('--top-k', type=int, default=10, help='Top-k for analysis')
    parser.add_argument('--fast-mode', action='store_true',
                        help='Fast mode (prompt only)')
    parser.add_argument('--production', action='store_true',
                        help='Production mode (test against full DynamicLensManager population)')
    parser.add_argument('--layer-idx', type=int, default=15,
                        help='Model layer for activations')
    parser.add_argument('--max-finetune-epochs', type=int, default=20,
                        help='Max epochs per lens during fine-tuning')
    parser.add_argument('--max-concepts', type=int, default=None,
                        help='Limit concepts (for testing)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Log directory (defaults to lens pack directory)')
    parser.add_argument('--no-cross-activation', action='store_true',
                        help='Skip cross-activation calibration (runs by default)')
    parser.add_argument('--cross-activation-samples', type=int, default=5,
                        help='Samples per concept for cross-activation')
    parser.add_argument('--cross-activation-threshold', type=float, default=0.5,
                        help='Firing threshold for cross-activation')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable activation caching (regenerate activations each cycle)')

    args = parser.parse_args()

    lens_pack_dir = Path(args.lens_pack)
    concept_pack_dir = Path(args.concept_pack)

    # Set up logging to lens pack log subdirectory
    import sys
    log_dir = Path(args.log_dir) if args.log_dir else lens_pack_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"calibration_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Tee stdout to both console and log file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_handle = open(log_file, 'w')
    sys.stdout = TeeOutput(sys.__stdout__, log_handle)
    sys.stderr = TeeOutput(sys.__stderr__, log_handle)
    print(f"Logging to: {log_file}")

    # Determine layers
    if args.layers is None:
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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    # Run cycle
    run_calibration_cycle(
        lens_pack_dir=lens_pack_dir,
        concept_pack_dir=concept_pack_dir,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        layers=layers,
        max_cycles=args.max_cycles,
        convergence_threshold=args.convergence_threshold,
        top_k=args.top_k,
        fast_mode=args.fast_mode,
        production_mode=args.production,
        layer_idx=args.layer_idx,
        max_finetune_epochs=args.max_finetune_epochs,
        max_concepts=args.max_concepts,
        run_cross_activation=not args.no_cross_activation,
        cross_activation_samples=args.cross_activation_samples,
        cross_activation_threshold=args.cross_activation_threshold,
        use_activation_cache=not args.no_cache,
        model_name=args.model,
    )


if __name__ == '__main__':
    main()
