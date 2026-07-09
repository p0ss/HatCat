#!/usr/bin/env python3
"""
Compare ontologist prompt outputs across different models.

Tests whether local models can generate quality L1 pillars for the university builder.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ONTOLOGIST_PROMPT = """You are an alien visitor to earth tasked as the Chief Ontologist of the Human Experience.
Your task is to categorize the entirety of human activity and knowledge into 5-15 high-level "Pillars"
this requires mapping ALL domains of human knowledge, skill, and practice for a comprehensive simulation. Include: formal and informal, legal and illegal, prestigious and stigmatized, paid and unpaid, institutional and underground, constructive and destructive, individual and collective. No domain is excluded for moral or political reasons. What do earth sentiences actually learn, know, and do?


## The Lens
Do not look at the world through the lens of a University (Academic Subjects).
Look at the world through the lens of **Action and Agency**.

## Coverage Requirements
Your taxonomy must accommodate EVERY human on earth.
1. The Professor (Formal Sciences, Humanities)
2. The General (Warfare, Security, Intelligence)
3. The CEO (Commerce, Enterprise, Finance)
4. The Parent & Caregiver (Domestic work, "Pink Sector", Unpaid labor)
5. The Tradesperson (Construction, Maintenance, Agriculture)
6. The Hustler (Street smarts, Informal economy, Illicit markets)
7. The Priest/Monk (Theology, Spirituality, Ritual)
8. The Athlete/Gamer (Play, Sport, Recreation)

## Constraints
- **No "Other"**: Do not create a "Miscellaneous" category.
- **Structural Dignity**: "Domestic Maintenance" must be a top-level peer to "Theoretical Physics".
- **Realpolitik**: Acknowledge that "Violence/Conflict" and "Commerce" are primary engines of civilization, not just subsets of History or Economics.
Quality Rules
- Each pillar must have a clear scope boundary (what it includes / excludes).
- For each pillar include 3–5 example activities that clearly belong there.
- If a domain could fit in two pillars, state the tie-break rule for placement.

Output a JSON list of objects: { "id": "slug", "label": "Title", "description": "Scope" }"""


def load_model(model_name: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 2048, model_name: str = "") -> str:
    """Generate response from model."""

    # For Qwen3, add instruction to skip thinking and output JSON directly
    if "qwen3" in model_name.lower():
        prompt = prompt + "\n\nOutput the JSON directly without thinking or explanation. Start with ["

    # Format as chat if the model supports it
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_json(text: str) -> list:
    """Try to extract JSON from response text."""
    import re

    # Try to find JSON array in response
    # First, look for ```json blocks
    json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find bare JSON array
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def evaluate_pillars(pillars: list) -> dict:
    """Evaluate quality of generated pillars."""
    if not pillars:
        return {"valid": False, "error": "No JSON extracted"}

    metrics = {
        "valid": True,
        "count": len(pillars),
        "has_id": sum(1 for p in pillars if p.get("id")),
        "has_label": sum(1 for p in pillars if p.get("label")),
        "has_description": sum(1 for p in pillars if p.get("description")),
        "avg_desc_length": 0,
        "labels": [p.get("label", "?") for p in pillars],
    }

    if metrics["has_description"] > 0:
        desc_lengths = [len(p.get("description", "")) for p in pillars if p.get("description")]
        metrics["avg_desc_length"] = sum(desc_lengths) / len(desc_lengths)

    # Check coverage hints
    coverage_keywords = {
        "commerce": ["commerce", "trade", "business", "economic", "market", "exchange"],
        "violence": ["violence", "conflict", "war", "security", "power", "coercion", "force"],
        "care": ["care", "nurture", "domestic", "health", "support"],
        "spirituality": ["spirit", "religion", "meaning", "belief", "ritual"],
        "play": ["play", "sport", "recreation", "game", "leisure"],
        "knowledge": ["knowledge", "education", "learn", "research", "science"],
        "making": ["making", "building", "craft", "construction", "create"],
    }

    all_text = " ".join(p.get("label", "") + " " + p.get("description", "") for p in pillars).lower()

    metrics["coverage"] = {}
    for category, keywords in coverage_keywords.items():
        metrics["coverage"][category] = any(kw in all_text for kw in keywords)

    metrics["coverage_score"] = sum(metrics["coverage"].values()) / len(coverage_keywords)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare ontologist outputs across models")
    parser.add_argument("--models", nargs="+", default=[
        "Qwen/Qwen3-8B",
        "ServiceNow-AI/Apriel-1.6-15b-Thinker",
        "google/gemma-3-4b-it",
    ])
    parser.add_argument("--output", type=str, default="results/ontologist_comparison")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print("="*60)

        try:
            model, tokenizer = load_model(model_name, args.device)

            print("Generating response...")
            response = generate_response(model, tokenizer, ONTOLOGIST_PROMPT)

            print(f"Response length: {len(response)} chars")

            # Save raw response
            safe_name = model_name.replace("/", "_")
            with open(output_dir / f"{safe_name}_raw.txt", "w") as f:
                f.write(response)

            # Extract and evaluate JSON
            pillars = extract_json(response)
            metrics = evaluate_pillars(pillars)

            if pillars:
                with open(output_dir / f"{safe_name}_pillars.json", "w") as f:
                    json.dump(pillars, f, indent=2)

            results[model_name] = {
                "success": metrics["valid"],
                "metrics": metrics,
                "pillar_count": metrics.get("count", 0),
                "coverage_score": metrics.get("coverage_score", 0),
            }

            print(f"Pillars extracted: {metrics.get('count', 0)}")
            print(f"Coverage score: {metrics.get('coverage_score', 0):.1%}")
            if metrics.get("labels"):
                print(f"Labels: {metrics['labels']}")

            # Free memory
            del model, tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error with {model_name}: {e}")
            results[model_name] = {"success": False, "error": str(e)}

    # Save comparison summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models_tested": len(args.models),
        "results": results,
    }

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print("="*60)

    for model_name, result in results.items():
        status = "OK" if result.get("success") else "FAIL"
        count = result.get("pillar_count", 0)
        coverage = result.get("coverage_score", 0)
        print(f"{model_name}: {status} - {count} pillars, {coverage:.0%} coverage")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
