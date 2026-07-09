#!/usr/bin/env python3
"""
Evaluate Judge Model Candidates

Tests candidate models on hard discrimination tasks to find the best judge
for Thalamos assessments.

Primary test: Meld-based evaluation using real training data with deterministic
ground truth. Tests the exact task a judge does: "Is this text an example of X?"

DATASET PROVENANCE: The test examples come from melds/applied/ - every example
has passed through the complete HATCAT MELD approval pipeline including human
review for elevated/protected/critical concepts. This provenance makes these
examples suitable as ground truth for deterministic testing.

Secondary test: Nuanced quality discrimination (subtle errors, hedging, etc.)

Usage:
    # Evaluate all candidates that fit in 24GB (RECOMMENDED: stratified sampling)
    python scripts/evaluate_judge_candidates.py --stratified

    # Stratified: 100 examples per risk level (400 total, statistically robust)
    python scripts/evaluate_judge_candidates.py --stratified --n-per-level 100

    # Evaluate specific models
    python scripts/evaluate_judge_candidates.py --models olmo-3-7b-think nemotron-nano-9b

    # Just list available candidates
    python scripts/evaluate_judge_candidates.py --list

    # Run with custom VRAM limit
    python scripts/evaluate_judge_candidates.py --max-vram 48

    # Legacy: random sampling (not recommended - uneven risk coverage)
    python scripts/evaluate_judge_candidates.py --n-cases 100
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.be.thalamos.model_candidates import (
    MODEL_CANDIDATES,
    ModelCandidate,
    CandidateLoader,
)
from src.be.thalamos.judge_evaluation import (
    JudgeEvaluator,
    JudgeEvalReport,
    generate_markdown_report,
)
from src.be.thalamos.meld_evaluation import (
    MeldExampleLoader,
    MeldJudgeEvaluator,
    MeldEvalReport,
    generate_meld_eval_report,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_generate_fn(model, tokenizer, candidate: ModelCandidate, processor=None):
    """Create a generation function for the judge evaluator."""
    def generate(prompt: str, max_new_tokens: int = 512) -> str:
        # Adapt prompt for model's reasoning mode
        if candidate.reasoning_mode == "/think":
            prompt = f"/think\n{prompt}"

        # Apply chat template if available (critical for instruction-tuned models)
        if processor is not None:
            inputs = processor(text=prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}
        elif hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            # Use chat template for instruction-tuned models
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            inputs = {"input_ids": formatted.to(model.device)}
        else:
            # Fallback for base models without chat template
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")

        # Track input length to extract only new tokens
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else inputs.get("input_ids", inputs).shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Only decode the NEW tokens (after input) - critical for chat-templated models
        new_tokens = outputs[0][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    return generate


def evaluate_candidate(
    candidate: ModelCandidate,
    loader: CandidateLoader,
    output_dir: Path,
    meld_examples: list,
    run_nuanced: bool = True,
) -> Tuple[MeldEvalReport, Optional[JudgeEvalReport]]:
    """Evaluate a single candidate on meld-based and nuanced tests."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {candidate.name}")
    logger.info(f"{'='*60}")

    # Load model
    start_load = time.time()
    model, tokenizer, processor = loader.load(candidate)
    load_time = time.time() - start_load
    vram_used = loader.get_vram_usage()

    logger.info(f"Loaded in {load_time:.1f}s, using {vram_used:.1f}GB VRAM")

    # Create generation function
    generate_fn = create_generate_fn(model, tokenizer, candidate, processor=processor)

    # PRIMARY: Meld-based evaluation (deterministic ground truth)
    logger.info(f"\n--- Meld-Based Evaluation ({len(meld_examples)} cases) ---")
    meld_evaluator = MeldJudgeEvaluator(generate_fn, model_id=candidate.model_id)
    meld_report = meld_evaluator.evaluate(meld_examples)

    logger.info(
        f"Meld accuracy: {meld_report.accuracy:.1%} "
        f"(TP:{meld_report.true_positives} TN:{meld_report.true_negatives} "
        f"FP:{meld_report.false_positives} FN:{meld_report.false_negatives})"
    )

    # Save meld results
    safe_name = candidate.model_id.replace('/', '_')
    meld_result_file = output_dir / f"{safe_name}_meld_eval.json"
    with open(meld_result_file, 'w') as f:
        json.dump(meld_report.to_dict(), f, indent=2)

    meld_report_file = output_dir / f"{safe_name}_meld_eval.md"
    with open(meld_report_file, 'w') as f:
        f.write(generate_meld_eval_report(meld_report))

    # SECONDARY: Nuanced discrimination tests
    nuanced_report = None
    if run_nuanced:
        logger.info("\n--- Nuanced Discrimination Tests ---")
        nuanced_evaluator = JudgeEvaluator(generate_fn, model_id=candidate.model_id)
        nuanced_report = nuanced_evaluator.run_full_evaluation()

        nuanced_file = output_dir / f"{safe_name}_nuanced_eval.json"
        with open(nuanced_file, 'w') as f:
            json.dump(nuanced_report.to_dict(), f, indent=2)

        nuanced_md = output_dir / f"{safe_name}_nuanced_eval.md"
        with open(nuanced_md, 'w') as f:
            f.write(generate_markdown_report(nuanced_report))

        logger.info(f"Nuanced: concept={nuanced_report.concept_score_accuracy:.1%}, reasoning={nuanced_report.reasoning_accuracy:.1%}")

    logger.info(f"Results saved to {output_dir}")

    # Unload
    loader.unload()

    return meld_report, nuanced_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate judge model candidates")
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Model candidate IDs to evaluate (default: all that fit VRAM)"
    )
    parser.add_argument(
        "--max-vram",
        type=float,
        default=24.0,
        help="Maximum VRAM in GB (default: 24)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("results/judge_candidates"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=100,
        help="Number of meld test cases for random sampling (default: 100, ignored if --stratified)"
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use stratified sampling with equal examples per risk level (RECOMMENDED)"
    )
    parser.add_argument(
        "--n-per-level",
        type=int,
        default=100,
        help="Examples per risk level for stratified sampling (default: 100, gives 400 total)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip-nuanced",
        action="store_true",
        help="Skip nuanced discrimination tests (faster)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available candidates and exit"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)"
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\nAvailable Model Candidates:")
        print("-" * 80)
        for cid, candidate in MODEL_CANDIDATES.items():
            fit = "Y" if candidate.vram_gb_estimate <= args.max_vram else "N"
            print(
                f"  {cid:<30} {candidate.params_billions:>5.1f}B  "
                f"{candidate.vram_gb_estimate:>5.1f}GB  "
                f"Fits {args.max_vram}GB: {fit}"
            )
            if candidate.notes:
                print(f"    {candidate.notes}")
        return

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader = CandidateLoader(device=args.device)

    # Load meld examples
    logger.info("Loading meld training examples...")
    meld_loader = MeldExampleLoader()
    meld_loader.load_all()

    stats = meld_loader.get_stats()
    logger.info(f"Loaded {stats['total_examples']} examples from {stats['total_concepts']} concepts")
    logger.info(f"By risk level: {stats['by_risk_level']}")

    # Get sample - stratified (recommended) or random
    if args.stratified:
        meld_examples, sample_counts = meld_loader.get_stratified_sample(
            n_per_level=args.n_per_level,
            seed=args.seed
        )
        logger.info(
            f"Stratified sampling: {len(meld_examples)} test cases "
            f"({args.n_per_level}/level, seed={args.seed})"
        )
        logger.info(f"Samples per risk level: {sample_counts}")
    else:
        logger.warning(
            "Using random sampling - consider --stratified for statistically robust results "
            "across all risk levels"
        )
        meld_examples = meld_loader.get_random_sample(n=args.n_cases, balance=True, seed=args.seed)
        logger.info(f"Selected {len(meld_examples)} test cases (seed={args.seed})")

    # Select candidates
    if args.models:
        candidates = [
            MODEL_CANDIDATES[cid]
            for cid in args.models
            if cid in MODEL_CANDIDATES
        ]
        missing = [cid for cid in args.models if cid not in MODEL_CANDIDATES]
        if missing:
            logger.warning(f"Unknown candidates: {missing}")
    else:
        candidates = [
            c for c in MODEL_CANDIDATES.values()
            if c.vram_gb_estimate <= args.max_vram
        ]

    if not candidates:
        logger.error("No candidates to evaluate")
        return

    logger.info(f"Evaluating {len(candidates)} candidates: {[c.name for c in candidates]}")

    # Evaluate each
    results = []  # List of (meld_report, nuanced_report) tuples
    for candidate in candidates:
        try:
            meld_report, nuanced_report = evaluate_candidate(
                candidate,
                loader,
                args.output_dir,
                meld_examples,
                run_nuanced=not args.skip_nuanced,
            )
            results.append((candidate, meld_report, nuanced_report))
        except Exception as e:
            logger.error(f"Failed to evaluate {candidate.name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate comparison summary
    if results:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        if args.skip_nuanced:
            print(f"\n{'Model':<35} {'Meld Acc':<10} {'Precision':<10} {'Recall':<10}")
            print("-" * 65)

            for candidate, meld_report, _ in sorted(results, key=lambda r: r[1].accuracy, reverse=True):
                name = candidate.model_id.split("/")[-1][:33]
                precision = meld_report.true_positives / max(meld_report.true_positives + meld_report.false_positives, 1)
                recall = meld_report.true_positives / max(meld_report.true_positives + meld_report.false_negatives, 1)
                print(
                    f"{name:<35} "
                    f"{meld_report.accuracy:>8.0%}  "
                    f"{precision:>8.0%}  "
                    f"{recall:>8.0%}"
                )
        else:
            print(f"\n{'Model':<30} {'Meld':<8} {'Concept':<8} {'Reason':<8} {'Combined':<8}")
            print("-" * 70)

            for candidate, meld_report, nuanced_report in sorted(
                results, key=lambda r: r[1].accuracy * 0.6 + (r[2].overall_score if r[2] else 0) * 0.4, reverse=True
            ):
                name = candidate.model_id.split("/")[-1][:28]
                combined = meld_report.accuracy * 0.6
                if nuanced_report:
                    combined += nuanced_report.overall_score * 0.4
                    print(
                        f"{name:<30} "
                        f"{meld_report.accuracy:>6.0%}  "
                        f"{nuanced_report.concept_score_accuracy:>6.0%}  "
                        f"{nuanced_report.reasoning_accuracy:>6.0%}  "
                        f"{combined:>6.0%}"
                    )
                else:
                    print(
                        f"{name:<30} "
                        f"{meld_report.accuracy:>6.0%}  "
                        f"{'N/A':>6}  "
                        f"{'N/A':>6}  "
                        f"{combined:>6.0%}"
                    )

        # Best recommendation
        best_candidate, best_meld, best_nuanced = max(
            results,
            key=lambda r: r[1].accuracy * 0.6 + (r[2].overall_score if r[2] else 0) * 0.4
        )
        combined_score = best_meld.accuracy * 0.6 + (best_nuanced.overall_score if best_nuanced else 0) * 0.4

        print(f"\nBest candidate: {best_candidate.model_id}")
        print(f"Meld accuracy: {best_meld.accuracy:.1%}")
        if best_nuanced:
            print(f"Nuanced overall: {best_nuanced.overall_score:.1%}")
        print(f"Combined score: {combined_score:.1%}")

        # Save comparison
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'n_cases': args.n_cases,
                'seed': args.seed,
                'skip_nuanced': args.skip_nuanced,
            },
            'candidates_evaluated': len(results),
            'rankings': [
                {
                    'model_id': c.model_id,
                    'meld_accuracy': m.accuracy,
                    'meld_precision': m.true_positives / max(m.true_positives + m.false_positives, 1),
                    'meld_recall': m.true_positives / max(m.true_positives + m.false_negatives, 1),
                    'nuanced_overall': n.overall_score if n else None,
                    'combined_score': m.accuracy * 0.6 + (n.overall_score if n else 0) * 0.4,
                }
                for c, m, n in sorted(
                    results,
                    key=lambda r: r[1].accuracy * 0.6 + (r[2].overall_score if r[2] else 0) * 0.4,
                    reverse=True
                )
            ],
            'best_candidate': best_candidate.model_id,
        }

        comparison_file = args.output_dir / "comparison_summary.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"\nComparison saved to {comparison_file}")


if __name__ == "__main__":
    main()
