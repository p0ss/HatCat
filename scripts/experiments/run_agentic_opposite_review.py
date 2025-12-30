#!/usr/bin/env python3
"""
Agentic Opposite Review - Identify semantic opposites for SUMO concepts.

Uses Claude API with multi-step reasoning to:
1. Analyze each concept's semantic properties
2. Generate opposite candidates (antonyms, semantic opposites, complements)
3. Validate candidates against WordNet
4. Score and rank candidates
5. Recommend best opposite with fallback strategy

Run with: ANTHROPIC_API_KEY=<key> python scripts/run_agentic_opposite_review.py
"""

import anthropic
import json
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


# Prompt templates
AGENTIC_REVIEW_PROMPT = """You are a semantic ontology expert analyzing concept opposites for an AI safety monitoring system.

# Task
Analyze the SUMO concept below and identify the best semantic opposite to use as a negative centroid anchor for Fisher-LDA concept axes.

# Concept
- **SUMO Term**: {sumo_term}
- **Definition**: {definition}
- **Layer**: {layer}
- **Synsets**: {synsets}
- **Category Children**: {category_children}

# Instructions

## Step 1: Concept Analysis
Analyze this concept:
1. Does it have a clear antonym? (yes/no)
2. Semantic category? (physical, abstract, process, attribute, state, entity, relation, event)
3. Best opposition type? (binary, spectrum, complementary, taxonomic, category-level, distant)

## Step 2: Generate Candidates
Generate up to 5 opposite candidates in priority order:

**Priority 1 - Direct Antonyms (WordNet):**
If the concept has clear antonyms (e.g., good↔evil, war↔peace, deception↔honesty).

**Priority 2 - Semantic Opposites (SUMO pairs):**
Common ontological opposites:
- Physical ↔ Abstract
- Voluntary ↔ Involuntary
- Animate ↔ Inanimate
- Good ↔ Evil
- Subjective ↔ Objective

**Priority 3 - Complementary Concepts (taxonomic/categorical):**
Mutually exclusive categories:
- Bird ↔ Mammal (taxonomic)
- Solid ↔ Liquid (state)
- Past ↔ Future (temporal)

**Priority 4 - Category Opposites (ontological):**
Different category types:
- Process ↔ Object
- Attribute ↔ Entity
- Relation ↔ Substance

**Priority 5 - Distributional Opposites:**
Low-cosine concepts from sibling/cousin set (not random - intelligently distant).

## Step 3: Validate
For each candidate:
1. Likely exists in WordNet 3.0?
2. Likely exists in SUMO ontology?
3. Worth adding to layers if missing?

## Step 4: Score (0-10)
- **Opposition strength**: Semantic distance/complementarity
- **Steering utility**: Usefulness for AI safety (deception detection, alignment monitoring)
- **Availability**: Exists in WordNet/SUMO?
- **Confidence**: How certain is this pairing?

## Step 5: Recommend
Select best candidate with:
- Reasoning for selection
- Confidence score
- Flags (needs_review, missing_from_layers, no_suitable_opposite, weak_opposition)
- Fallback strategy

# Output Format

Respond in JSON:
```json
{{
  "analysis": {{
    "has_antonym": true/false,
    "semantic_category": "category",
    "opposition_type": "type"
  }},
  "candidates": [
    {{
      "concept": "OppositeConceptName",
      "type": "antonym/semantic_opposite/complement/category_opposite/distributional",
      "reasoning": "why this is a good opposite",
      "scores": {{
        "opposition_strength": 0-10,
        "steering_utility": 0-10,
        "availability": 0-10,
        "confidence": 0-10
      }},
      "exists_in_wordnet": true/false/unknown,
      "exists_in_sumo": true/false/unknown,
      "should_add_to_layers": true/false
    }}
  ],
  "recommendation": {{
    "selected": "OppositeConceptName" or null,
    "reasoning": "explanation of selection",
    "confidence": 0-10,
    "flags": [],
    "fallback_strategy": "what to do if selected isn't available"
  }}
}}
```

# Important Notes
- Focus on semantic opposition useful for Fisher-LDA: w ≈ μ+ - μ-
- For AI safety: prioritize opposites relevant to deception/alignment (e.g., deception↔honesty)
- If no good opposite exists, be honest - suggest distributional opposites but flag for review
- Consider whether missing opposites should be added to layers (e.g., if Deception exists, Honesty should too)
- Avoid pure random concepts - even in fallback, use intelligently distant selection
"""

VALIDATION_PROMPT = """You suggested "{candidate}" as an opposite for "{concept}".

Verify this suggestion:
1. Does "{candidate}" exist as a concept in WordNet 3.0?
2. If yes, what would be a typical synset (e.g., honesty.n.01)?
3. If no, suggest an alternative that does exist

Respond in JSON:
{{
  "exists": true/false/unknown,
  "synset": "synset.n.01" or null,
  "alternative": "AlternativeConcept" or null,
  "reasoning": "explanation"
}}
"""


class AgenticOppositeReviewer:
    """Agentic review system for identifying concept opposites."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    async def review_concept(self, concept: Dict) -> Dict:
        """
        Review a single concept and identify its opposite.

        Returns:
            {
                'sumo_term': str,
                'layer': int,
                'analysis': {...},
                'candidates': [...],
                'recommendation': {...},
                'validation': {...} (if applicable)
            }
        """
        # Format the prompt
        prompt = self._format_prompt(concept)

        try:
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse JSON response
            response_text = message.content[0].text
            result = self._extract_json(response_text)

            # Add metadata
            result["sumo_term"] = concept["sumo_term"]
            result["layer"] = concept.get("layer", "unknown")

            # Self-correction: validate top candidate if confidence is borderline
            if result["recommendation"]["selected"]:
                confidence = result["recommendation"]["confidence"]
                if 4 <= confidence <= 8:  # Borderline cases
                    validated = await self._validate_candidate(
                        concept["sumo_term"], result["recommendation"]["selected"]
                    )
                    result["validation"] = validated

            return result

        except Exception as e:
            # Fallback if API call fails
            return {
                "sumo_term": concept["sumo_term"],
                "layer": concept.get("layer", "unknown"),
                "error": str(e),
                "flags": ["api_error"],
            }

    async def _validate_candidate(self, concept: str, candidate: str) -> Dict:
        """Validate a candidate opposite by checking WordNet."""
        validation_prompt = VALIDATION_PROMPT.format(
            concept=concept, candidate=candidate
        )

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": validation_prompt}],
            )

            response_text = message.content[0].text
            return self._extract_json(response_text)

        except Exception as e:
            return {"exists": "unknown", "error": str(e)}

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from response text (handles markdown code blocks)."""
        # Try to extract from ```json ... ``` blocks
        if "```json" in text:
            json_text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_text = text.split("```")[1].split("```")[0].strip()
        else:
            json_text = text.strip()

        return json.loads(json_text)

    def _format_prompt(self, concept: Dict) -> str:
        """Format the main review prompt."""
        return AGENTIC_REVIEW_PROMPT.format(
            sumo_term=concept["sumo_term"],
            definition=concept.get("definition", "No definition available"),
            layer=concept.get("layer", "unknown"),
            synsets=", ".join(concept.get("synsets", [])[:5]),  # First 5
            category_children=", ".join(
                concept.get("category_children", [])[:10]
            ),  # First 10
        )

    async def review_all_concepts(
        self, concepts: List[Dict], batch_size: int = 10, max_concurrent: int = 5
    ) -> List[Dict]:
        """
        Review all concepts in parallel batches.

        Args:
            concepts: List of concept dictionaries
            batch_size: Concepts per batch (for rate limiting)
            max_concurrent: Max concurrent API calls

        Returns:
            List of review results
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def review_with_semaphore(concept):
            async with semaphore:
                result = await self.review_concept(concept)
                return result

        # Process in batches
        total_batches = (len(concepts) - 1) // batch_size + 1

        for i in range(0, len(concepts), batch_size):
            batch = concepts[i : i + batch_size]
            batch_num = i // batch_size + 1

            print(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} concepts)..."
            )

            # Run batch in parallel
            batch_results = await asyncio.gather(
                *[review_with_semaphore(c) for c in batch], return_exceptions=True
            )

            # Handle exceptions
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append(
                        {
                            "sumo_term": batch[idx]["sumo_term"],
                            "error": str(result),
                            "flags": ["exception"],
                        }
                    )
                else:
                    results.append(result)

            # Rate limiting between batches
            if i + batch_size < len(concepts):
                await asyncio.sleep(1)

        return results


def load_concepts_from_layers(layer_dir: Path) -> List[Dict]:
    """Load all concepts from layers 0-6."""
    all_concepts = []

    for layer_num in range(7):
        layer_file = layer_dir / f"layer{layer_num}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        for concept in layer_data["concepts"]:
            concept["layer"] = layer_num
            all_concepts.append(concept)

        print(f"  Loaded Layer {layer_num}: {len(layer_data['concepts'])} concepts")

    return all_concepts


def save_results(results: List[Dict], output_file: Path):
    """Save review results with metadata."""
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_concepts": len(results),
            "model": "claude-haiku-4-5-20251001",
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("REVIEW SUMMARY")
    print("=" * 80)

    total = len(results)
    errors = len([r for r in results if "error" in r])
    successful = total - errors

    print(f"Total concepts: {total}")
    print(f"Successful reviews: {successful}")
    print(f"Errors: {errors}")

    if successful > 0:
        has_opposite = len(
            [r for r in results if r.get("recommendation", {}).get("selected")]
        )
        print(
            f"\nConcepts with opposite identified: {has_opposite} ({has_opposite / successful * 100:.1f}%)"
        )

        # Flag statistics
        all_flags = []
        for r in results:
            flags = r.get("recommendation", {}).get("flags", [])
            all_flags.extend(flags)

        from collections import Counter

        flag_counts = Counter(all_flags)

        if flag_counts:
            print("\nFlag distribution:")
            for flag, count in flag_counts.most_common():
                print(f"  {flag}: {count}")

        # Opposition type distribution
        opposition_types = []
        for r in results:
            if r.get("recommendation", {}).get("selected"):
                candidates = r.get("candidates", [])
                if candidates:
                    opposition_types.append(candidates[0]["type"])

        type_counts = Counter(opposition_types)
        if type_counts:
            print("\nOpposition type distribution:")
            for opp_type, count in type_counts.most_common():
                print(f"  {opp_type}: {count}")


async def main():
    """Run the agentic opposite review."""
    print("=" * 80)
    print("AGENTIC OPPOSITE REVIEW")
    print("=" * 80)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print(
            "\nUsage: ANTHROPIC_API_KEY=<your-key> python scripts/experiments/run_agentic_opposite_review.py"
        )
        return

    # Paths - use first-light concept pack
    project_root = Path(__file__).parent.parent.parent
    layer_dir = project_root / "concept_packs" / "first-light" / "hierarchy"
    output_file = project_root / "results" / "opposite_review.json"
    output_file.parent.mkdir(exist_ok=True)

    # Load concepts
    print("\nLoading concepts from layers...")
    concepts = load_concepts_from_layers(layer_dir)
    print(f"\n✓ Total concepts loaded: {len(concepts)}")

    # Confirm before proceeding
    print(f"\nEstimated API cost: ~${len(concepts) * 0.003:.2f}")
    print(f"Estimated time: ~{len(concepts) * 0.5 / 60:.1f} minutes")

    response = input("\nProceed with review? [y/N]: ")
    if response.lower() != "y":
        print("Aborted.")
        return

    # Run review
    print("\nStarting agentic review...")
    print("(This will take a while - grab a coffee!)")

    reviewer = AgenticOppositeReviewer(api_key)
    results = await reviewer.review_all_concepts(
        concepts, batch_size=20, max_concurrent=20
    )

    # Save results
    save_results(results, output_file)

    # Print summary
    print_summary(results)

    print("\n" + "=" * 80)
    print("REVIEW COMPLETE!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Review results in: {output_file}")
    print(f"2. Check flagged concepts (needs_manual_review)")
    print(f"3. Identify missing opposites (missing_from_layers)")
    print(f"4. Integrate with data generation")


if __name__ == "__main__":
    asyncio.run(main())
