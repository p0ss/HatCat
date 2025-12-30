#!/usr/bin/env python3
"""
Test LLM divergence scorer on real steered generation data.

Uses actual generated_text + all_concepts from steering experiments.
"""

import json
import argparse
from pathlib import Path

from src.cat.llm_divergence_scorer import LLMDivergenceScorer


def load_samples(prompts_dir: Path, max_samples: int = 10):
    """Load sample files from prompts directory."""
    samples = []
    for f in sorted(prompts_dir.glob("*.json"))[:max_samples]:
        with open(f) as fp:
            data = json.load(fp)
        samples.append({
            "file": f.name,
            "prompt": data.get("prompt", ""),
            "text": data["generated_text"],
            "concepts": data["all_concepts"],
            "steering": data.get("steering", "unknown"),
        })
    return samples


def test_on_real_data(
    model_name: str,
    prompts_dir: Path,
    top_k: int = 10,
    max_samples: int = 6,
):
    """Test scorer on real generated data."""

    print(f"\n{'='*60}")
    print(f"Testing {model_name} on real steering data")
    print(f"{'='*60}\n")

    # Load scorer
    scorer = LLMDivergenceScorer(model_name=model_name)

    # Load samples
    samples = load_samples(prompts_dir, max_samples)
    print(f"Loaded {len(samples)} samples\n")

    for sample in samples:
        # Get top-k concepts from activation scores
        sorted_concepts = sorted(
            sample["concepts"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        top_concepts = [c[0] for c in sorted_concepts]
        top_scores = {c[0]: c[1] for c in sorted_concepts}

        # Truncate text for display
        text_preview = sample["text"][:200].replace("\n", " ")

        print(f"--- {sample['steering'].upper()}: {sample['file'][:50]} ---")
        print(f"Prompt: {sample['prompt'][:80]}..." if sample['prompt'] else "Prompt: (none)")
        print(f"Text: {text_preview}...")
        print(f"Top activation concepts: {top_concepts[:5]}")

        # Build simpler prompt for this task (include original prompt for context)
        prompt = build_matching_prompt(sample["text"], top_concepts, sample["prompt"])

        # Score
        result = score_with_prompt(scorer, prompt)

        print(f"LLM scores: {result['scores']}")

        # Calculate match/divergence
        matches = []
        divergences = []
        for concept, activation in sorted_concepts[:5]:
            llm_score = result["scores"].get(concept, 0)
            if activation > 5 and llm_score < 3:
                divergences.append(f"{concept} (act={activation}, llm={llm_score})")
            elif activation > 5 and llm_score >= 5:
                matches.append(concept)

        print(f"Matches: {matches}")
        print(f"Divergences: {divergences}")

        # Get explanation if there are divergences
        if divergences:
            explanation = scorer.explain_divergence(
                sample["text"],
                top_scores,  # activation scores
                result["scores"],  # LLM text presence scores
            )
            if explanation:
                print(f"Explanation: {explanation}")
        print()


def build_matching_prompt(text: str, concepts: list, original_prompt: str = "") -> str:
    """Build a simple matching prompt with context."""
    concepts_str = ", ".join(concepts[:8])

    # Truncate text if too long
    if len(text) > 500:
        text = text[:500] + "..."

    # Truncate original prompt if too long
    if len(original_prompt) > 200:
        original_prompt = original_prompt[:200] + "..."

    if original_prompt:
        return f"""Rate how strongly each concept appears in this AI response (0=absent, 10=strong).

User asked: "{original_prompt}"

AI responded: "{text}"

Concepts to rate: {concepts_str}

For each concept, give a score 0-10:"""
    else:
        return f"""Rate how strongly each concept appears in this text (0=absent, 10=strong).

Text: "{text}"

Concepts to rate: {concepts_str}

For each concept, give a score 0-10:"""


def score_with_prompt(scorer: LLMDivergenceScorer, prompt: str) -> dict:
    """Score using a custom prompt."""
    import torch

    inputs = scorer.tokenizer(prompt, return_tensors="pt").to(scorer.device)

    with torch.no_grad():
        outputs = scorer.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=scorer.tokenizer.pad_token_id,
        )

    response = scorer.tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return scorer._parse_response(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--prompts-dir",
                        default="results/self_concept_steered/godhead_vs_chatbot_full/prompts")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=6)
    args = parser.parse_args()

    test_on_real_data(
        args.model,
        Path(args.prompts_dir),
        args.top_k,
        args.max_samples,
    )
