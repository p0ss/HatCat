#!/usr/bin/env python3
"""
Generate meld operations for concepts missing definitions/synsets.

Identifies concepts with missing data and generates meld descriptions
using Claude API to fill in:
- definition
- positive_examples
- negative_examples
- training_hints (disambiguation, confusable_with)

Usage:
    ANTHROPIC_API_KEY=<key> python scripts/tools/generate_missing_concept_melds.py \
        --pack concept_packs/first-light \
        --output results/missing_concept_melds.json \
        --batch-size 10 \
        --max-concepts 100
"""

import anthropic
import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


SYSTEM_PROMPT = """You are an expert ontologist helping to create training data for concept classifiers.

Your task is to generate high-quality definitions and training examples for SUMO ontology concepts.

For each concept, provide:

1. **definition** (50-200 chars): A clear definitional statement ("X is a...")
   - Should capture the semantic essence
   - First sentence should be definitional
   - Avoid circular definitions

2. **positive_examples** (4-6 sentences): Natural language where this concept activates
   - Diverse scenarios and phrasings
   - Natural, not technical jargon
   - Should genuinely involve the concept

3. **negative_examples** (4-6 sentences): Where this concept should NOT activate
   - Near-misses from the same domain (most important!)
   - Sibling concepts or parent concepts
   - Things commonly confused with this concept

4. **training_hints**:
   - disambiguation: How to distinguish from similar concepts
   - confusable_with: List of concepts this might be confused with
   - key_features: Distinguishing characteristics

Return valid JSON only, no markdown."""


CONCEPT_PROMPT = """Generate training data for this SUMO concept:

**Term**: {term}
**Layer**: {layer} (0=most abstract, 6=most specific)
**Parent concepts**: {parents}
**Sibling concepts**: {siblings}
**Children**: {children}
**Existing definition**: {existing_def}

Return JSON:
{{
  "term": "{term}",
  "definition": "clear definitional statement",
  "positive_examples": ["example1", "example2", "example3", "example4"],
  "negative_examples": ["near-miss1", "near-miss2", "near-miss3", "near-miss4"],
  "training_hints": {{
    "disambiguation": "how to distinguish from similar concepts",
    "confusable_with": ["sibling1", "sibling2"],
    "key_features": ["feature1", "feature2"]
  }}
}}"""


def load_concepts(pack_dir: Path) -> tuple[List[Dict], Dict[str, Dict]]:
    """Load all concepts from pack hierarchy."""
    hierarchy_dir = pack_dir / "hierarchy"
    all_concepts = []

    for layer in range(7):
        layer_file = hierarchy_dir / f"layer{layer}.json"
        if not layer_file.exists():
            continue
        with open(layer_file) as f:
            data = json.load(f)
        for concept in data.get("concepts", []):
            concept["layer"] = layer
            all_concepts.append(concept)

    concept_map = {c["sumo_term"]: c for c in all_concepts}
    return all_concepts, concept_map


def find_missing_concepts(all_concepts: List[Dict]) -> List[Dict]:
    """Find concepts missing definition or training data."""
    missing = []

    for concept in all_concepts:
        needs_data = False

        # Check for missing definition
        if not concept.get("definition") and not concept.get("sumo_definition"):
            needs_data = True

        # Check for missing training hints
        hints = concept.get("training_hints", {})
        if not hints.get("positive_examples") and not hints.get("disambiguation"):
            needs_data = True

        if needs_data:
            missing.append(concept)

    return missing


def get_siblings(concept: Dict, concept_map: Dict[str, Dict]) -> List[str]:
    """Get sibling concepts (same parent)."""
    siblings = set()
    for parent_name in concept.get("parent_concepts", []):
        parent = concept_map.get(parent_name, {})
        for child in parent.get("category_children", []):
            if child != concept["sumo_term"]:
                siblings.add(child)
    return list(siblings)[:10]  # Limit for prompt size


class MeldGenerator:
    """Generate meld descriptions using Claude API."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20241001"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    async def generate_for_concept(
        self,
        concept: Dict,
        concept_map: Dict[str, Dict]
    ) -> Dict:
        """Generate meld data for a single concept."""
        term = concept["sumo_term"]
        siblings = get_siblings(concept, concept_map)

        prompt = CONCEPT_PROMPT.format(
            term=term,
            layer=concept.get("layer", "unknown"),
            parents=", ".join(concept.get("parent_concepts", [])[:5]),
            siblings=", ".join(siblings[:8]),
            children=", ".join(concept.get("category_children", [])[:8]),
            existing_def=concept.get("sumo_definition", concept.get("definition", "None"))[:200]
        )

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text

            # Extract JSON
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_text = response_text.strip()

            result = json.loads(json_text)
            result["sumo_term"] = term
            result["layer"] = concept.get("layer")
            result["status"] = "success"
            return result

        except Exception as e:
            return {
                "sumo_term": term,
                "layer": concept.get("layer"),
                "status": "error",
                "error": str(e)
            }

    async def generate_batch(
        self,
        concepts: List[Dict],
        concept_map: Dict[str, Dict],
        batch_size: int = 10,
        max_concurrent: int = 5
    ) -> List[Dict]:
        """Generate melds for a batch of concepts."""
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(concept):
            async with semaphore:
                return await self.generate_for_concept(concept, concept_map)

        total_batches = (len(concepts) - 1) // batch_size + 1

        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i + batch_size]
            batch_num = i // batch_size + 1

            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} concepts)...")

            batch_results = await asyncio.gather(
                *[generate_with_semaphore(c) for c in batch],
                return_exceptions=True
            )

            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "sumo_term": batch[idx]["sumo_term"],
                        "status": "error",
                        "error": str(result)
                    })
                else:
                    results.append(result)

            # Rate limiting
            if i + batch_size < len(concepts):
                await asyncio.sleep(1)

        return results


def create_meld_operations(results: List[Dict]) -> List[Dict]:
    """Convert generation results to meld operations format."""
    operations = []

    for result in results:
        if result.get("status") != "success":
            continue

        op = {
            "operation": "enhance",
            "target": result["sumo_term"],
            "target_layer": result.get("layer", 0),
            "enhancements": {}
        }

        if result.get("definition"):
            op["enhancements"]["definition"] = result["definition"]

        if result.get("positive_examples"):
            op["enhancements"]["positive_examples"] = result["positive_examples"]

        if result.get("negative_examples"):
            op["enhancements"]["negative_examples"] = result["negative_examples"]

        if result.get("training_hints"):
            op["enhancements"]["training_hints"] = result["training_hints"]

        if op["enhancements"]:
            operations.append(op)

    return operations


async def main():
    parser = argparse.ArgumentParser(
        description="Generate melds for concepts missing definitions/training data"
    )
    parser.add_argument(
        "--pack", type=Path, required=True,
        help="Path to concept pack directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output file for meld operations"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Concepts per batch"
    )
    parser.add_argument(
        "--max-concepts", type=int, default=None,
        help="Maximum concepts to process (for testing)"
    )
    parser.add_argument(
        "--only-missing-def", action="store_true",
        help="Only process concepts missing definitions"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATE MISSING CONCEPT MELDS")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n❌ ANTHROPIC_API_KEY not set")
        return

    # Load concepts
    print(f"\nLoading concepts from: {args.pack}")
    all_concepts, concept_map = load_concepts(args.pack)
    print(f"  Total concepts: {len(all_concepts)}")

    # Find missing
    print("\nFinding concepts with missing data...")
    missing = find_missing_concepts(all_concepts)
    print(f"  Missing definition or training data: {len(missing)}")

    if args.only_missing_def:
        missing = [c for c in missing
                   if not c.get("definition") and not c.get("sumo_definition")]
        print(f"  Missing definition only: {len(missing)}")

    if args.max_concepts:
        missing = missing[:args.max_concepts]
        print(f"  Limited to: {len(missing)}")

    if not missing:
        print("\nNo concepts need processing!")
        return

    # Estimate
    cost = len(missing) * 0.003
    time_mins = len(missing) * 0.5 / 60
    print(f"\nEstimated cost: ~${cost:.2f}")
    print(f"Estimated time: ~{time_mins:.0f} minutes")

    response = input("\nProceed? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Generate
    print("\nGenerating meld descriptions...")
    generator = MeldGenerator(api_key)
    results = await generator.generate_batch(
        missing,
        concept_map,
        batch_size=args.batch_size
    )

    # Convert to meld operations
    operations = create_meld_operations(results)

    # Summary
    success = len([r for r in results if r.get("status") == "success"])
    errors = len([r for r in results if r.get("status") == "error"])

    print(f"\n✓ Generated {success} meld operations")
    print(f"  Errors: {errors}")

    # Save
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pack": str(args.pack),
            "total_processed": len(results),
            "successful": success,
            "errors": errors
        },
        "operations": operations,
        "raw_results": results
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved to: {args.output}")
    print(f"\nNext: Apply melds with scripts/tools/apply_melds.py")


if __name__ == "__main__":
    asyncio.run(main())
