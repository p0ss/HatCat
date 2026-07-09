#!/usr/bin/env python3
"""
Generate L1 pillars using the introspective ontologist prompt.
The model maps its own knowledge and capabilities.
"""

import json
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.be.thalamos.model_candidates import CandidateLoader, MODEL_CANDIDATES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_id: str = "google/gemma-4-E4B-it"):
    """Load the model for generation, using CandidateLoader if registered."""
    # Check if model is in the candidate registry (by key or by model_id)
    candidate_key = (
        model_id if model_id in MODEL_CANDIDATES
        else next(
            (k for k, v in MODEL_CANDIDATES.items() if v.model_id == model_id),
            None,
        )
    )

    if candidate_key:
        logger.info(f"Loading {model_id} via CandidateLoader...")
        loader = CandidateLoader()
        candidate = MODEL_CANDIDATES[candidate_key]
        model, tokenizer, _ = loader.load(candidate)
        return model, tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading {model_id} directly...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        return model, tokenizer


def generate_pillars(model, tokenizer, prompt: str, max_tokens: int = 4096) -> str:
    """Generate pillars from the introspective prompt."""

    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    input_len = inputs.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


def extract_json(text: str) -> list:
    """Extract JSON from model response."""
    # Try code block first
    if "```json" in text:
        json_text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        json_text = text.split("```")[1].split("```")[0].strip()
    else:
        # Find array brackets
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            json_text = text[start:end]
        else:
            json_text = text.strip()

    return json.loads(json_text)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate introspective L1 pillars")
    parser.add_argument(
        "--prompt", "-p",
        type=Path,
        default=Path("melds/helpers/introspective_ontologist_prompt.txt"),
        help="Path to ontologist prompt"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results/introspective_pillars.json"),
        help="Output path for pillars"
    )
    parser.add_argument(
        "--model", "-m",
        default="google/gemma-4-E4B-it",
        help="Model to use (HuggingFace ID or candidate registry key)"
    )

    args = parser.parse_args()

    # Load prompt
    prompt = args.prompt.read_text()
    logger.info(f"Loaded prompt from {args.prompt}")

    # Load model
    model, tokenizer = load_model(args.model)

    # Generate
    logger.info("Generating pillars...")
    response = generate_pillars(model, tokenizer, prompt)

    print("\n" + "="*60)
    print("RAW RESPONSE:")
    print("="*60)
    print(response)
    print("="*60 + "\n")

    # Extract JSON
    try:
        pillars = extract_json(response)
        logger.info(f"Extracted {len(pillars)} pillars")

        # Save
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(pillars, f, indent=2)

        logger.info(f"Saved to {args.output}")

        # Print summary
        print("\nPILLARS:")
        for p in pillars:
            print(f"  - {p.get('label', p.get('id'))}")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        # Save raw response for debugging
        debug_path = args.output.with_suffix(".raw.txt")
        debug_path.write_text(response)
        logger.info(f"Saved raw response to {debug_path}")


if __name__ == "__main__":
    main()
