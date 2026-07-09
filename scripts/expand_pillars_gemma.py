#!/usr/bin/env python3
"""
Expand L1 pillars into training examples and L2 children using Gemma 3 4B.

For each pillar:
1. Generate 50-100 positive example prompts
2. Generate 50-100 negative example prompts
3. Generate 5-15 L2 child concepts

Output format compatible with lens training pipeline.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model(model_name: str = "google/gemma-3-4b-it"):
    """Load Gemma 3 4B IT."""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """Generate response from model."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_json_list(text: str) -> List:
    """Extract JSON list from response."""
    import re

    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare JSON array
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to extract line by line as strings
    lines = [line.strip().strip('",') for line in text.split('\n') if line.strip() and not line.strip().startswith(('[', ']', '{', '}', '```'))]
    if lines:
        return lines[:100]

    return []


def generate_positive_examples(model, tokenizer, pillar: Dict, count: int = 75) -> List[str]:
    """Generate positive example prompts for a pillar."""
    prompt = f"""Generate {count} diverse example prompts/scenarios that clearly belong to the "{pillar['label']}" category.

Category description: {pillar['description']}

Requirements:
- Each example should be a short prompt (1-2 sentences) describing an activity, situation, or topic
- Cover diverse aspects within this category
- Include both common and unusual examples
- Mix formal and informal contexts
- Include questions, statements, and scenarios

Output as a JSON array of strings. Example format:
["A merchant negotiating prices at a bazaar", "How do stock markets work?", ...]

Output ONLY the JSON array, no other text."""

    response = generate(model, tokenizer, prompt, max_tokens=4000, temperature=0.8)
    examples = extract_json_list(response)

    print(f"  Generated {len(examples)} positive examples")
    return examples[:count]


def generate_negative_examples(model, tokenizer, pillar: Dict, other_pillars: List[Dict], count: int = 75) -> List[str]:
    """Generate negative example prompts (things NOT in this pillar)."""
    other_labels = [p['label'] for p in other_pillars if p['id'] != pillar['id']][:5]

    prompt = f"""Generate {count} diverse example prompts/scenarios that clearly DO NOT belong to the "{pillar['label']}" category.

"{pillar['label']}" is about: {pillar['description']}

Generate examples from OTHER categories like: {', '.join(other_labels)}

Requirements:
- Each example should clearly belong to a DIFFERENT category
- Cover diverse topics that are NOT {pillar['label']}
- Mix different domains and contexts
- Should be unambiguously NOT in the target category

Output as a JSON array of strings.

Output ONLY the JSON array, no other text."""

    response = generate(model, tokenizer, prompt, max_tokens=4000, temperature=0.8)
    examples = extract_json_list(response)

    print(f"  Generated {len(examples)} negative examples")
    return examples[:count]


def generate_l2_children(model, tokenizer, pillar: Dict) -> List[Dict]:
    """Generate L2 child concepts for a pillar."""
    prompt = f"""For the category "{pillar['label']}", generate 8-12 sub-categories (L2 children) that together cover all activities within this domain.

Parent category: {pillar['label']}
Description: {pillar['description']}

Requirements:
- Sub-categories should be mutually exclusive (no overlap)
- Together they should cover the entire parent category
- Each should be specific enough to be meaningful
- No "Other" or "Miscellaneous" categories

For each child, provide:
- id: a slug (lowercase, underscores)
- label: a title
- description: what this sub-category covers
- examples: 3-5 example activities

Output as a JSON array of objects.

Output ONLY the JSON array, no other text."""

    response = generate(model, tokenizer, prompt, max_tokens=3000, temperature=0.7)
    children = extract_json_list(response)

    # Ensure each child is a dict
    valid_children = []
    for child in children:
        if isinstance(child, dict) and 'label' in child:
            if 'id' not in child:
                child['id'] = child['label'].lower().replace(' ', '_').replace('&', 'and')
            valid_children.append(child)

    print(f"  Generated {len(valid_children)} L2 children")
    return valid_children


def expand_pillar(model, tokenizer, pillar: Dict, all_pillars: List[Dict], output_dir: Path) -> Dict:
    """Expand a single pillar with examples and children."""
    print(f"\n{'='*60}")
    print(f"Expanding: {pillar['label']}")
    print("="*60)

    # Generate examples
    positive_examples = generate_positive_examples(model, tokenizer, pillar)
    time.sleep(1)  # Rate limit

    negative_examples = generate_negative_examples(model, tokenizer, pillar, all_pillars)
    time.sleep(1)

    # Generate L2 children
    children = generate_l2_children(model, tokenizer, pillar)

    # Build expanded pillar
    expanded = {
        "id": pillar["id"],
        "label": pillar["label"],
        "description": pillar["description"],
        "layer": 1,
        "parent_concepts": [],
        "children": [c["id"] for c in children],
        "training_examples": {
            "positive": positive_examples,
            "negative": negative_examples,
        },
        "l2_children": children,
    }

    # Save individual pillar
    pillar_file = output_dir / f"{pillar['id']}.json"
    with open(pillar_file, "w") as f:
        json.dump(expanded, f, indent=2)

    print(f"Saved to {pillar_file}")

    return expanded


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Expand pillars with Gemma 3 4B")
    parser.add_argument("--pillars", type=str,
                        default="results/ontologist_comparison/gemma3_4b_pillars.json",
                        help="Path to pillars JSON")
    parser.add_argument("--output", type=str,
                        default="concept_packs/action-agency-pillars",
                        help="Output directory for concept pack")
    parser.add_argument("--model", type=str,
                        default="google/gemma-3-4b-it",
                        help="Model to use")
    parser.add_argument("--pillars-to-expand", type=str, nargs="*",
                        help="Specific pillar IDs to expand (default: all)")

    args = parser.parse_args()

    # Load pillars
    with open(args.pillars) as f:
        pillars = json.load(f)

    print(f"Loaded {len(pillars)} pillars")

    # Filter if specific pillars requested
    if args.pillars_to_expand:
        pillars = [p for p in pillars if p["id"] in args.pillars_to_expand]
        print(f"Expanding {len(pillars)} selected pillars")

    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.model)

    # Expand each pillar
    expanded_pillars = []
    for pillar in pillars:
        try:
            expanded = expand_pillar(model, tokenizer, pillar, pillars, output_dir)
            expanded_pillars.append(expanded)
        except Exception as e:
            print(f"Error expanding {pillar['label']}: {e}")
            continue

    # Save pack manifest
    manifest = {
        "name": "action-agency-pillars",
        "description": "L1 pillars covering human activity through Action & Agency lens",
        "created": datetime.now().isoformat(),
        "source_model": args.model,
        "pillars": [p["id"] for p in expanded_pillars],
        "total_positive_examples": sum(len(p["training_examples"]["positive"]) for p in expanded_pillars),
        "total_negative_examples": sum(len(p["training_examples"]["negative"]) for p in expanded_pillars),
        "total_l2_children": sum(len(p["l2_children"]) for p in expanded_pillars),
    }

    with open(output_dir / "pack.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print("EXPANSION COMPLETE")
    print("="*60)
    print(f"Pillars expanded: {len(expanded_pillars)}")
    print(f"Total positive examples: {manifest['total_positive_examples']}")
    print(f"Total negative examples: {manifest['total_negative_examples']}")
    print(f"Total L2 children: {manifest['total_l2_children']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
