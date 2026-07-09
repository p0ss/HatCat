#!/usr/bin/env python3
"""
Generate L1 Pillars and Process Through MELD Review Pipeline

Uses the ontologist prompt on a LOCAL TARGET MODEL to generate top-level pillars
of human knowledge, then processes each through the three-tier MELD review pipeline.

This is Phase 2 of Fractal Model Cartography - Subject Concept Assessment.

Generation uses LOCAL models (Gemma, Ministral, etc.) - NOT Claude API.
Claude API is only used in the final review tier of the pipeline.

Usage:
    # Generate L1 pillars using default model (gemma-3-4b-it)
    python scripts/generate_l1_pillars.py

    # Use a specific local model for pillar generation
    python scripts/generate_l1_pillars.py --model ministral-8b

    # List available models
    python scripts/generate_l1_pillars.py --list-models

    # Save pillars without running through pipeline (dry run)
    python scripts/generate_l1_pillars.py --dry-run

    # Resume from existing pillars file
    python scripts/generate_l1_pillars.py --resume results/l1_pillars/pillars.json
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.be.thalamos.model_candidates import CandidateLoader, MODEL_CANDIDATES
from scripts.meld_review_pipeline import MeldReviewPipeline, MeldReviewState, ReviewStage, HierarchyContext

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Load ontologist prompt
ONTOLOGIST_PROMPT_PATH = PROJECT_ROOT / "melds" / "helpers" / "ontologist_prompt.txt"


def load_ontologist_prompt() -> str:
    """Load the ontologist prompt from file."""
    with open(ONTOLOGIST_PROMPT_PATH) as f:
        return f.read()


def generate_l1_pillars(model_id: str = "gemma-3-4b-it") -> Tuple[List[Dict], str]:
    """
    Generate L1 pillars using the ontologist prompt on a LOCAL model.

    Args:
        model_id: Key from MODEL_CANDIDATES (e.g., "gemma-3-4b-it", "ministral-8b")

    Returns:
        (pillars, raw_response)
    """
    if model_id not in MODEL_CANDIDATES:
        logger.error(f"Unknown model: {model_id}")
        logger.error(f"Available models: {list(MODEL_CANDIDATES.keys())}")
        return [], ""

    candidate = MODEL_CANDIDATES[model_id]
    prompt = load_ontologist_prompt()

    logger.info(f"Generating L1 pillars using {candidate.name}...")

    # Load model
    loader = CandidateLoader()
    model, tokenizer, processor = loader.load(candidate)

    logger.info(f"Model loaded, using {loader.get_vram_usage():.1f}GB VRAM")

    # Apply chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        input_ids = inputs
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

    input_len = input_ids.shape[1] if len(input_ids.shape) > 1 else input_ids.shape[0]

    # Generate
    logger.info("Generating pillars (this may take a minute)...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids if len(input_ids.shape) > 1 else input_ids.unsqueeze(0),
            max_new_tokens=4000,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = outputs[0][input_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Unload model to free VRAM
    loader.unload()

    # Extract JSON from response
    try:
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()
        else:
            # Try to find JSON array directly
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start >= 0 and end > start:
                json_text = response_text[start:end]
            else:
                json_text = response_text.strip()

        pillars = json.loads(json_text)

        if not isinstance(pillars, list):
            pillars = [pillars]

        logger.info(f"Generated {len(pillars)} L1 pillars")
        return pillars, response_text

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse pillars JSON: {e}")
        logger.error(f"Raw response:\n{response_text[:2000]}...")
        return [], response_text


def save_pillars(pillars: List[Dict], output_dir: Path, model_id: str = "", raw_response: str = ""):
    """Save generated pillars to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save structured pillars
    pillars_file = output_dir / "pillars.json"
    with open(pillars_file, 'w') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "generator_model": model_id,
            "prompt_file": str(ONTOLOGIST_PROMPT_PATH),
            "pillar_count": len(pillars),
            "pillars": pillars
        }, f, indent=2)

    logger.info(f"Saved pillars to {pillars_file}")

    # Save raw response for debugging
    if raw_response:
        raw_file = output_dir / "raw_response.txt"
        with open(raw_file, 'w') as f:
            f.write(raw_response)


def load_pillars(pillars_file: Path) -> List[Dict]:
    """Load existing pillars from file."""
    with open(pillars_file) as f:
        data = json.load(f)
    return data.get("pillars", data)


def pillar_to_meld_context(pillar: Dict, all_pillars: List[Dict]) -> Dict:
    """
    Convert a pillar to context for MELD generation.

    Returns context dict with parent_concepts, sibling_concepts, etc.
    """
    # L1 pillars have no parents (they are the top level)
    # Siblings are other L1 pillars
    siblings = [
        p.get("label") or p.get("id")
        for p in all_pillars
        if (p.get("id") or p.get("label")) != (pillar.get("id") or pillar.get("label"))
    ]

    return {
        "term": pillar.get("label") or pillar.get("id"),
        "parent_concepts": [],  # L1 has no parents
        "sibling_concepts": siblings,
        "description": pillar.get("description", ""),
        "scope": pillar.get("scope", pillar.get("description", "")),
    }


def process_pillars_through_pipeline(
    pillars: List[Dict],
    output_dir: Path,
    max_generation_attempts: int = 3,
    generator_model_id: str = "gemma-3-4b-it",
    judge_mode: str = "gatekeeper",
) -> List[MeldReviewState]:
    """
    Process each L1 pillar through the MELD review pipeline.

    Args:
        pillars: List of pillar definitions
        output_dir: Directory to save results
        max_generation_attempts: Max attempts per pillar
        generator_model_id: Local model for MELD generation
        judge_mode: "gatekeeper" (enforce worldview) or "annotator" (capture divergence)

    Returns list of final states for each pillar.
    """
    pipeline = MeldReviewPipeline(
        max_generation_attempts=max_generation_attempts,
        generator_model_id=generator_model_id,
        judge_mode=judge_mode,
    )
    results = []

    for i, pillar in enumerate(pillars):
        context = pillar_to_meld_context(pillar, pillars)
        term = context["term"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing pillar {i+1}/{len(pillars)}: {term}")
        logger.info(f"{'='*60}")
        logger.info(f"Description: {context.get('description', 'N/A')[:100]}...")
        logger.info(f"Siblings: {', '.join(context['sibling_concepts'][:5])}...")

        try:
            # Create L1-specific hierarchy context
            l1_context = HierarchyContext(
                level=1,  # L1 - top-level pillars
                parent_concept=None,  # L1 has no parent
                sibling_concepts=context["sibling_concepts"],
            )

            # Use the existing pillar description as the definition
            # (already generated by ontologist prompt - appropriately broad for L1)
            pillar_description = context.get("description", "")

            state = pipeline.process_concept(
                concept_term=term,
                parent_concepts=context["parent_concepts"],
                sibling_concepts=context["sibling_concepts"],
                hierarchy_context=l1_context,
                predefined_definition=pillar_description,
            )
            results.append(state)

            # Save intermediate result
            save_pillar_result(pillar, state, output_dir)

        except Exception as e:
            logger.error(f"Failed to process pillar '{term}': {e}")
            import traceback
            traceback.print_exc()

            # Create failed state
            failed_state = MeldReviewState(
                concept_term=term,
                meld_data={},
                current_stage=ReviewStage.REJECTED,
            )
            results.append(failed_state)

    return results


def save_pillar_result(pillar: Dict, state: MeldReviewState, output_dir: Path):
    """Save the result of processing a single pillar."""
    results_dir = output_dir / "pillar_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    term = pillar.get("label") or pillar.get("id")
    safe_name = term.lower().replace(" ", "_").replace("/", "_")

    result = {
        "pillar": pillar,
        "concept_term": state.concept_term,
        "final_stage": state.current_stage.value,
        "protection_level": state.protection_level.name if state.protection_level else "UNKNOWN",
        "generation_attempts": state.generation_attempts,
        "meld_data": state.meld_data,
        "worldview_metadata": state.worldview_metadata,  # Judge's perspective in annotator mode
        "review_history": [
            {
                "stage": r.stage.value,
                "passed": r.passed,
                "errors": r.errors,
                "warnings": r.warnings,
                "feedback": r.feedback,
                "confidence": r.confidence,
                "reviewer_model": r.reviewer_model,
            }
            for r in state.review_history
        ]
    }

    result_file = results_dir / f"{safe_name}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)


def generate_summary(pillars: List[Dict], results: List[MeldReviewState], output_dir: Path):
    """Generate a summary of the pipeline run."""

    approved = [r for r in results if r.current_stage == ReviewStage.APPROVED]
    human_review = [r for r in results if r.current_stage == ReviewStage.HUMAN_REVIEW]
    rejected = [r for r in results if r.current_stage == ReviewStage.REJECTED]

    # Gather worldview divergence info
    worldview_divergences = {}
    for r in results:
        if r.worldview_metadata:
            worldview_divergences[r.concept_term] = {
                "divergence_score": r.worldview_metadata.get("divergence_score", 0),
                "notes": r.worldview_metadata.get("worldview_notes", []),
            }

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_pillars": len(pillars),
        "results": {
            "approved": len(approved),
            "human_review": len(human_review),
            "rejected": len(rejected),
        },
        "approved_concepts": [r.concept_term for r in approved],
        "human_review_concepts": [r.concept_term for r in human_review],
        "rejected_concepts": [r.concept_term for r in rejected],
        "pillar_labels": [p.get("label") or p.get("id") for p in pillars],
        "worldview_divergences": worldview_divergences,
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("L1 PILLAR GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total pillars generated: {len(pillars)}")
    print(f"  Approved: {len(approved)}")
    print(f"  Human Review: {len(human_review)}")
    print(f"  Rejected: {len(rejected)}")

    if approved:
        print(f"\nApproved pillars:")
        for r in approved:
            print(f"  - {r.concept_term}")

    if human_review:
        print(f"\nPending human review:")
        for r in human_review:
            print(f"  - {r.concept_term} ({r.protection_level.name})")

    if rejected:
        print(f"\nRejected pillars:")
        for r in rejected:
            print(f"  - {r.concept_term}")

    # Show worldview divergence info if any
    divergent = [(r.concept_term, r.worldview_metadata) for r in results
                 if r.worldview_metadata and r.worldview_metadata.get("divergence_score", 0) > 0.3]
    if divergent:
        print(f"\nWorldview divergence noted (score > 0.3):")
        for term, meta in divergent:
            score = meta.get("divergence_score", 0)
            notes = meta.get("worldview_notes", [])
            print(f"  - {term}: {score:.2f}")
            for note in notes[:1]:  # First note only
                print(f"    {note[:80]}...")

    print(f"\nResults saved to: {output_dir}")

    return summary


def list_available_models(max_vram: float = 24.0):
    """List available local models for pillar generation."""
    print("\nAvailable Local Models for L1 Pillar Generation:")
    print("-" * 70)
    for cid, candidate in MODEL_CANDIDATES.items():
        fit = "Y" if candidate.vram_gb_estimate <= max_vram else "N"
        print(
            f"  {cid:<25} {candidate.params_billions:>5.1f}B  "
            f"{candidate.vram_gb_estimate:>5.1f}GB  "
            f"Fits {max_vram}GB: {fit}"
        )
        if candidate.notes:
            print(f"    {candidate.notes}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate L1 pillars using LOCAL model and process through MELD review pipeline"
    )
    parser.add_argument(
        "--model", "-m",
        default="gemma-3-4b-it",
        help="Local model for pillar generation (default: gemma-3-4b-it)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available local models and exit"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("results/l1_pillars"),
        help="Output directory (default: results/l1_pillars)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate pillars but don't run through pipeline"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from existing pillars.json file"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max generation attempts per pillar (default: 3)"
    )
    parser.add_argument(
        "--max-vram",
        type=float,
        default=24.0,
        help="Max VRAM for model selection (default: 24GB)"
    )
    parser.add_argument(
        "--judge-mode",
        choices=["gatekeeper", "annotator"],
        default="gatekeeper",
        help="gatekeeper=enforce worldview, annotator=capture divergence as metadata (default: gatekeeper)"
    )

    args = parser.parse_args()

    if args.list_models:
        list_available_models(args.max_vram)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate or load pillars
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        pillars = load_pillars(args.resume)
        raw_response = ""
    else:
        pillars, raw_response = generate_l1_pillars(model_id=args.model)

        if not pillars:
            logger.error("No pillars generated, exiting")
            sys.exit(1)

        save_pillars(pillars, args.output_dir, model_id=args.model, raw_response=raw_response)

    # Display generated pillars
    print(f"\n{'='*60}")
    print(f"GENERATED L1 PILLARS ({len(pillars)}) - Model: {args.model}")
    print(f"{'='*60}")
    for i, p in enumerate(pillars, 1):
        label = p.get("label") or p.get("id")
        desc = p.get("description", "")[:80]
        print(f"{i:2}. {label}")
        if desc:
            print(f"    {desc}...")

    if args.dry_run:
        logger.info("Dry run - skipping pipeline processing")
        return

    # Step 2: Process through pipeline
    print(f"\n{'='*60}")
    print("PROCESSING THROUGH MELD REVIEW PIPELINE")
    print(f"{'='*60}")

    results = process_pillars_through_pipeline(
        pillars=pillars,
        output_dir=args.output_dir,
        max_generation_attempts=args.max_attempts,
        generator_model_id=args.model,
        judge_mode=args.judge_mode,
    )

    # Step 3: Generate summary
    generate_summary(pillars, results, args.output_dir)


if __name__ == "__main__":
    main()
