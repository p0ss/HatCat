#!/usr/bin/env python3
"""
Generate definitions for concepts that are missing them.

Uses Claude API to generate high-quality definitions based on:
- Concept name and layer
- Parent concepts
- Domain
- Opposite concepts (if available)
- Sibling concepts

Usage:
    python scripts/ontology/generate_missing_definitions.py \
        --concept-pack concept_packs/first-light \
        --output results/generated_definitions.json \
        --batch-size 20 \
        --apply  # To apply directly to concept files
"""

import anthropic
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


SYSTEM_PROMPT = """You are an expert ontologist helping to write definitions for a concept hierarchy used in AI interpretability.

## Context: HatCat Concept Pack

HatCat trains "lenses" - linear probes that detect when specific concepts are active in a language model's hidden states. The hierarchy organizes ~8000 concepts across layers (0-6) with parent-child relationships.

Layer meanings:
- Layer 0: Top-level domains (CreatedThings, LivingThings, Information, MindsAndAgents, PhysicalWorld)
- Layer 1: Major categories within domains
- Layer 2: Specific concept types
- Layer 3-6: Increasingly specific concepts

## Your Task

Write a clear, concise definition (30-150 characters) for each concept. The definition should:

1. Start with "A/An [noun]..." or "[Noun] that..." - be definitional, not descriptive
2. Capture the semantic essence - what makes this concept distinct
3. NOT be circular (don't just restate the term or list children)
4. Be useful for training a classifier to distinguish this from related concepts
5. Consider the parent-child relationship and where this concept fits

## Output Format

Return a JSON array with this structure:
[
  {
    "term": "<concept name>",
    "definition": "<clear 30-150 char definition>"
  },
  ...
]
"""


def load_hierarchy(concept_pack_path: Path) -> Dict:
    """Load hierarchy.json for parent/child relationships."""
    hier_path = concept_pack_path / "hierarchy.json"
    if hier_path.exists():
        with open(hier_path) as f:
            return json.load(f)
    return {}


def find_missing_definitions(concept_pack_path: Path) -> List[Dict]:
    """Find all concepts missing definitions."""
    concepts_dir = concept_pack_path / "concepts"
    missing = []

    for layer_dir in sorted(concepts_dir.glob("layer*")):
        layer_num = int(layer_dir.name.replace("layer", ""))
        for concept_file in layer_dir.glob("*.json"):
            with open(concept_file) as f:
                concept = json.load(f)

            definition = concept.get("definition", "")
            if not definition or len(definition.strip()) < 10:
                missing.append({
                    "term": concept.get("term", concept_file.stem),
                    "layer": layer_num,
                    "domain": concept.get("domain", ""),
                    "parent_concepts": concept.get("parent_concepts", []),
                    "opposite": concept.get("relationships", {}).get("opposite", []),
                    "opposite_reasoning": concept.get("opposite_reasoning", ""),
                    "file_path": str(concept_file),
                })

    return missing


def get_sibling_concepts(concept: Dict, hierarchy: Dict) -> List[str]:
    """Get sibling concepts (other children of same parent)."""
    parent_to_children = hierarchy.get("parent_to_children", {})
    siblings = []

    for parent in concept.get("parent_concepts", []):
        # Try different key formats
        for key in [parent, f"{parent}:{concept['layer']-1}"]:
            if key in parent_to_children:
                siblings.extend([c for c in parent_to_children[key] if c != concept["term"]])

    return siblings[:10]  # Limit to 10


def build_batch_prompt(concepts: List[Dict], hierarchy: Dict) -> str:
    """Build prompt for a batch of concepts."""
    concept_entries = []

    for c in concepts:
        siblings = get_sibling_concepts(c, hierarchy)

        entry = f"""
Concept: {c['term']}
- Layer: {c['layer']}
- Domain: {c['domain']}
- Parent(s): {', '.join(c['parent_concepts']) if c['parent_concepts'] else 'none'}
- Opposite: {', '.join(c['opposite']) if c['opposite'] else 'none'}
- Siblings: {', '.join(siblings[:5]) if siblings else 'none'}
"""
        if c.get('opposite_reasoning'):
            entry += f"- Opposite reasoning: {c['opposite_reasoning'][:200]}...\n"

        concept_entries.append(entry)

    return f"""Please write definitions for these {len(concepts)} concepts:

{''.join(concept_entries)}

Return a JSON array with term and definition for each."""


def generate_definitions(
    concepts: List[Dict],
    hierarchy: Dict,
    batch_size: int = 20,
    model: str = "claude-sonnet-4-20250514",
) -> List[Dict]:
    """Generate definitions using Claude API."""
    client = anthropic.Anthropic()
    results = []

    for i in range(0, len(concepts), batch_size):
        batch = concepts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(concepts) + batch_size - 1)//batch_size}...")

        prompt = build_batch_prompt(batch, hierarchy)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            content = response.content[0].text

            # Extract JSON from response
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                definitions = json.loads(content[start:end])
                results.extend(definitions)
                print(f"  Generated {len(definitions)} definitions")
            else:
                print(f"  Warning: Could not parse response")
                print(f"  Response: {content[:500]}")

        except Exception as e:
            print(f"  Error: {e}")

        # Rate limiting
        time.sleep(1)

    return results


def apply_definitions(
    definitions: List[Dict],
    missing_concepts: List[Dict],
    dry_run: bool = False,
) -> int:
    """Apply generated definitions to concept files."""
    # Build lookup from term to definition
    def_lookup = {d["term"]: d["definition"] for d in definitions}

    # Build lookup from term to file path
    path_lookup = {c["term"]: c["file_path"] for c in missing_concepts}

    applied = 0
    for term, definition in def_lookup.items():
        if term not in path_lookup:
            print(f"  Warning: No file found for {term}")
            continue

        file_path = Path(path_lookup[term])
        if not file_path.exists():
            continue

        with open(file_path) as f:
            concept = json.load(f)

        # Only update if still missing
        if concept.get("definition") and len(concept["definition"].strip()) >= 10:
            continue

        concept["definition"] = definition

        if dry_run:
            print(f"  Would update: {term}")
        else:
            with open(file_path, "w") as f:
                json.dump(concept, f, indent=2)

        applied += 1

    return applied


def main():
    parser = argparse.ArgumentParser(description="Generate missing concept definitions")
    parser.add_argument("--concept-pack", required=True, help="Path to concept pack")
    parser.add_argument("--output", default="results/generated_definitions.json",
                        help="Output JSON file")
    parser.add_argument("--batch-size", type=int, default=20, help="Concepts per API call")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model to use")
    parser.add_argument("--apply", action="store_true", help="Apply definitions to concept files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be applied")
    parser.add_argument("--max-concepts", type=int, default=None, help="Limit concepts (for testing)")
    args = parser.parse_args()

    concept_pack_path = Path(args.concept_pack)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Finding concepts missing definitions in {concept_pack_path}...")
    missing = find_missing_definitions(concept_pack_path)
    print(f"  Found {len(missing)} concepts missing definitions")

    if args.max_concepts:
        missing = missing[:args.max_concepts]
        print(f"  Limited to {len(missing)} for testing")

    if not missing:
        print("No concepts need definitions!")
        return

    print(f"\nLoading hierarchy...")
    hierarchy = load_hierarchy(concept_pack_path)
    print(f"  Loaded hierarchy with {len(hierarchy.get('parent_to_children', {}))} parents")

    print(f"\nGenerating definitions with {args.model}...")
    definitions = generate_definitions(
        missing,
        hierarchy,
        batch_size=args.batch_size,
        model=args.model,
    )

    # Save results
    result = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "total_missing": len(missing),
        "definitions_generated": len(definitions),
        "definitions": definitions,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {len(definitions)} definitions to {output_path}")

    # Apply if requested
    if args.apply or args.dry_run:
        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Applying definitions...")
        applied = apply_definitions(definitions, missing, dry_run=args.dry_run)
        print(f"  {'Would apply' if args.dry_run else 'Applied'} {applied} definitions")


if __name__ == "__main__":
    main()
