#!/usr/bin/env python3
"""
Expand L2 children into L3 grandchildren using Gemma 3 4B.

For each L2 concept:
1. Generate 5-10 L3 child concepts
2. Generate training examples for each

This gives L2 concepts children, enabling relationship-based training.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

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

    return []


def generate_l3_children(model, tokenizer, l2_concept: Dict, l1_parent: Dict) -> List[Dict]:
    """Generate L3 children for an L2 concept."""
    prompt = f"""For the category "{l2_concept['label']}", which is a sub-category of "{l1_parent['label']}", generate 5-8 more specific sub-categories (L3 children).

Parent (L1): {l1_parent['label']} - {l1_parent['description']}
Category (L2): {l2_concept['label']} - {l2_concept.get('description', '')}

Requirements:
- Sub-categories should be mutually exclusive (minimal overlap)
- Together they should cover the L2 category well
- Each should be specific and actionable
- No "Other" or "Miscellaneous" categories

For each child, provide:
- id: a slug (lowercase, underscores)
- label: a concise title
- description: 1 sentence describing scope
- examples: 2-3 example activities

Output as a JSON array of objects. Output ONLY the JSON array, no other text."""

    response = generate(model, tokenizer, prompt, max_tokens=2000, temperature=0.7)
    children = extract_json_list(response)

    # Validate and clean
    valid_children = []
    for child in children:
        if isinstance(child, dict) and 'label' in child:
            if 'id' not in child:
                child['id'] = child['label'].lower().replace(' ', '_').replace('&', 'and').replace('-', '_')
            # Prefix with parent ID for uniqueness
            child['id'] = f"{l2_concept['id']}_{child['id']}"
            child['parent_id'] = l2_concept['id']
            valid_children.append(child)

    return valid_children[:8]


def generate_l2_examples(model, tokenizer, l2_concept: Dict, l1_parent: Dict, count: int = 30) -> tuple:
    """Generate positive and negative examples for L2 concept."""

    # Positive examples
    pos_prompt = f"""Generate {count} diverse example prompts/scenarios that clearly belong to "{l2_concept['label']}" (a type of {l1_parent['label']}).

Category: {l2_concept['label']}
Description: {l2_concept.get('description', '')}
Parent category: {l1_parent['label']}

Requirements:
- Each example should be a short prompt (1-2 sentences)
- Cover diverse aspects within this specific category
- Be clearly about {l2_concept['label']}, not just general {l1_parent['label']}

Output as a JSON array of strings. Output ONLY the JSON array."""

    pos_response = generate(model, tokenizer, pos_prompt, max_tokens=2000, temperature=0.8)
    positives = extract_json_list(pos_response)

    # Negative examples (siblings and other domains)
    neg_prompt = f"""Generate {count} example prompts/scenarios that are NOT "{l2_concept['label']}" but might be confused with it.

"{l2_concept['label']}" is about: {l2_concept.get('description', '')}
Parent: {l1_parent['label']}

Generate examples from:
1. Sibling categories (other types of {l1_parent['label']})
2. Completely different domains

Output as a JSON array of strings. Output ONLY the JSON array."""

    neg_response = generate(model, tokenizer, neg_prompt, max_tokens=2000, temperature=0.8)
    negatives = extract_json_list(neg_response)

    return positives[:count], negatives[:count]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Expand L2 to L3 with Gemma 3 4B")
    parser.add_argument("--pack-dir", type=str,
                        default="concept_packs/action-agency-pillars",
                        help="Path to concept pack")
    parser.add_argument("--model", type=str,
                        default="google/gemma-3-4b-it",
                        help="Model to use")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start from L2 index (for resuming)")
    parser.add_argument("--max-l2", type=int, default=None,
                        help="Max L2 concepts to process")

    args = parser.parse_args()

    pack_dir = Path(args.pack_dir)
    hierarchy_dir = pack_dir / "hierarchy"

    # Load existing hierarchy
    with open(hierarchy_dir / "layer0.json") as f:
        layer0 = json.load(f)
    with open(hierarchy_dir / "layer1.json") as f:
        layer1 = json.load(f)

    l1_concepts = {c["sumo_term"]: c for c in layer0["concepts"]}
    l2_concepts = layer1["concepts"]

    print(f"Loaded {len(l1_concepts)} L1 concepts")
    print(f"Loaded {len(l2_concepts)} L2 concepts")

    # Load model
    model, tokenizer = load_model(args.model)

    # Process L2 concepts
    l3_concepts = []
    updated_l2 = []

    end_idx = len(l2_concepts) if args.max_l2 is None else min(args.start_from + args.max_l2, len(l2_concepts))

    for i, l2 in enumerate(l2_concepts):
        if i < args.start_from:
            updated_l2.append(l2)
            continue
        if i >= end_idx:
            updated_l2.append(l2)
            continue

        # Find L1 parent
        parent_terms = l2.get("parent_concepts", [])
        l1_parent = None
        for pt in parent_terms:
            if pt in l1_concepts:
                l1_parent = l1_concepts[pt]
                break

        if not l1_parent:
            print(f"[{i+1}/{len(l2_concepts)}] Skipping {l2['sumo_term']} - no L1 parent found")
            updated_l2.append(l2)
            continue

        print(f"\n[{i+1}/{len(l2_concepts)}] Expanding: {l2['sumo_term']} (under {l1_parent['sumo_term']})")

        # Generate L3 children
        l2_dict = {
            "id": l2["sumo_term"],
            "label": l2["sumo_term"],
            "description": l2.get("definition", ""),
        }
        l1_dict = {
            "label": l1_parent["sumo_term"],
            "description": l1_parent.get("definition", ""),
        }

        children = generate_l3_children(model, tokenizer, l2_dict, l1_dict)
        print(f"  Generated {len(children)} L3 children")

        # Generate examples for L2
        positives, negatives = generate_l2_examples(model, tokenizer, l2_dict, l1_dict)
        print(f"  Generated {len(positives)} positive, {len(negatives)} negative examples")

        # Update L2 with children and examples
        l2_updated = l2.copy()
        l2_updated["category_children"] = [c["id"] for c in children]
        l2_updated["child_concepts"] = [c["id"] for c in children]
        l2_updated["positive_examples"] = positives
        l2_updated["negative_examples"] = negatives
        updated_l2.append(l2_updated)

        # Add L3 children
        for child in children:
            l3_concept = {
                "sumo_term": child["id"],
                "layer": 2,
                "domain": "ActionAgency",
                "is_category_lens": False,
                "is_pseudo_sumo": True,
                "definition": child.get("description", ""),
                "parent_concepts": [l2["sumo_term"]],
                "child_concepts": [],
                "category_children": [],
                "positive_examples": [],
                "negative_examples": [],
            }
            # Add examples from the child definition
            if "examples" in child:
                for ex in child.get("examples", []):
                    l3_concept["positive_examples"].append(f"An example of {child['label']}: {ex}")
            l3_concepts.append(l3_concept)

        time.sleep(0.5)  # Rate limit

        # Save progress periodically
        if (i + 1) % 10 == 0:
            print(f"\n  Saving progress at {i+1} concepts...")
            _save_hierarchy(hierarchy_dir, layer0, updated_l2, l3_concepts)

    # Final save
    _save_hierarchy(hierarchy_dir, layer0, updated_l2, l3_concepts)

    # Update pack.json
    with open(pack_dir / "pack.json") as f:
        pack = json.load(f)

    pack["concept_metadata"]["total_concepts"] = len(layer0["concepts"]) + len(updated_l2) + len(l3_concepts)
    pack["concept_metadata"]["layers"] = [0, 1, 2]
    pack["concept_metadata"]["layer_distribution"] = {
        "0": len(layer0["concepts"]),
        "1": len(updated_l2),
        "2": len(l3_concepts),
    }
    pack["version"] = "0.2.0"

    with open(pack_dir / "pack.json", "w") as f:
        json.dump(pack, f, indent=2)

    print(f"\n{'='*60}")
    print("EXPANSION COMPLETE")
    print("="*60)
    print(f"L1 concepts: {len(layer0['concepts'])}")
    print(f"L2 concepts: {len(updated_l2)} (now with children)")
    print(f"L3 concepts: {len(l3_concepts)} (NEW)")
    print(f"Total: {len(layer0['concepts']) + len(updated_l2) + len(l3_concepts)}")


def _save_hierarchy(hierarchy_dir: Path, layer0: dict, l2_concepts: list, l3_concepts: list):
    """Save hierarchy files."""
    # Update layer1
    layer1 = {"layer": 1, "concepts": l2_concepts}
    with open(hierarchy_dir / "layer1.json", "w") as f:
        json.dump(layer1, f, indent=2)

    # Create layer2
    layer2 = {"layer": 2, "concepts": l3_concepts}
    with open(hierarchy_dir / "layer2.json", "w") as f:
        json.dump(layer2, f, indent=2)


if __name__ == "__main__":
    main()
