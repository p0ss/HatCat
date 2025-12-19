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
) -> Dict:
    """
    Run iterative calibration cycles.

    Each cycle:
    1. Run analysis to identify under/over-firing lenses
    2. Fine-tune problematic lenses
    3. Re-run analysis to measure improvement
    4. Repeat until convergence or max_cycles

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
    )


if __name__ == '__main__':
    main()
