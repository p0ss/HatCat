#!/usr/bin/env python3
"""
Concept Pack Enrichment Script

Uses an LLM to populate missing metadata for concepts:
- Training examples (positive and negative)
- Safety criticality assessment
- Disambiguation hints

Can use either a local model or API.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SafetyCriticality(Enum):
    """Safety criticality levels for concepts."""
    NONE = "none"           # Not safety-relevant
    LOW = "low"             # Tangentially related to safety
    MEDIUM = "medium"       # Moderately safety-relevant
    HIGH = "high"           # Directly safety-relevant
    CRITICAL = "critical"   # Core safety concept (deception, manipulation, etc.)


@dataclass
class EnrichmentResult:
    """Result of enriching a concept."""
    term: str
    positive_examples: List[str]
    negative_examples: List[str]
    disambiguation: str
    safety_criticality: SafetyCriticality
    safety_rationale: str
    harness_relevant: bool
    treaty_relevant: bool


ENRICHMENT_PROMPT = """You are helping build an AI safety interpretability system. Given a concept from an ontology, generate training data for a linear probe that will detect when this concept is active in an AI's hidden states.

Concept: {term}
Definition: {definition}
Parent concepts: {parents}
Domain: {domain}

Generate the following:

1. POSITIVE_EXAMPLES: 5 short text snippets (1-2 sentences) where this concept would be clearly present/active. These should be diverse and representative.

2. NEGATIVE_EXAMPLES: 5 short text snippets that might seem related but do NOT actually involve this concept. These are important for training the probe to distinguish.

3. DISAMBIGUATION: One sentence explaining what distinguishes this concept from similar ones.

4. SAFETY_CRITICALITY: Rate how safety-relevant this concept is for AI alignment monitoring:
   - NONE: Not related to AI safety
   - LOW: Tangentially related (e.g., general cognitive concepts)
   - MEDIUM: Moderately relevant (e.g., persuasion, influence)
   - HIGH: Directly relevant (e.g., deception patterns, consent issues)
   - CRITICAL: Core safety concept (e.g., manipulation, coercion, self-deception)

5. SAFETY_RATIONALE: Brief explanation of the safety relevance (or why it's not safety-relevant).

6. HARNESS_RELEVANT: true/false - Is this concept relevant for AI harness/guardrail systems?

7. TREATY_RELEVANT: true/false - Would this concept be relevant for AI governance/treaty monitoring?

Respond in JSON format:
{{
  "positive_examples": ["...", "...", "...", "...", "..."],
  "negative_examples": ["...", "...", "...", "...", "..."],
  "disambiguation": "...",
  "safety_criticality": "NONE|LOW|MEDIUM|HIGH|CRITICAL",
  "safety_rationale": "...",
  "harness_relevant": true/false,
  "treaty_relevant": true/false
}}
"""


class ConceptEnricher:
    """Enriches concept metadata using an LLM."""

    def __init__(
        self,
        model_name: str = "gemma-2b",
        use_api: bool = False,
        api_key: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.model_name = model_name
        self.use_api = use_api
        self.api_key = api_key
        self.dry_run = dry_run
        self.model = None
        self.tokenizer = None

    def initialize(self):
        """Initialize the model."""
        if self.dry_run:
            print("Dry run mode - no model loaded")
            return

        if self.use_api:
            # API mode - just validate key exists
            if not self.api_key:
                import os
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("API key required for API mode")
            print(f"Using API with model: {self.model_name}")
        else:
            # Local model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"Loading local model: {self.model_name}")
            model_map = {
                "gemma-2b": "google/gemma-2b-it",
                "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
                "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
            }
            model_id = model_map.get(self.model_name, self.model_name)

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model.eval()
            print(f"✓ Model loaded")

    def enrich_concept(self, concept_data: Dict) -> Optional[EnrichmentResult]:
        """Generate enrichment data for a concept."""
        term = concept_data.get("term", "")
        definition = concept_data.get("definition", "")
        parents = ", ".join(concept_data.get("parent_concepts", []))
        domain = concept_data.get("domain", "Unknown")

        prompt = ENRICHMENT_PROMPT.format(
            term=term,
            definition=definition,
            parents=parents,
            domain=domain,
        )

        if self.dry_run:
            # Return placeholder data
            return EnrichmentResult(
                term=term,
                positive_examples=[f"[DRY RUN] Positive example for {term}"] * 5,
                negative_examples=[f"[DRY RUN] Negative example for {term}"] * 5,
                disambiguation=f"[DRY RUN] Disambiguation for {term}",
                safety_criticality=SafetyCriticality.NONE,
                safety_rationale="[DRY RUN]",
                harness_relevant=False,
                treaty_relevant=False,
            )

        try:
            response = self._generate(prompt)
            return self._parse_response(term, response)
        except Exception as e:
            print(f"  Error enriching {term}: {e}")
            return None

    def _generate(self, prompt: str) -> str:
        """Generate response from LLM."""
        if self.use_api:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        else:
            # Local model
            import torch

            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return response

    def _parse_response(self, term: str, response: str) -> Optional[EnrichmentResult]:
        """Parse LLM response into EnrichmentResult."""
        # Try to extract JSON from response
        import re

        # Find JSON block
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            print(f"  No JSON found in response for {term}")
            return None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"  Invalid JSON for {term}: {e}")
            return None

        # Parse safety criticality
        crit_str = data.get("safety_criticality", "NONE").upper()
        try:
            criticality = SafetyCriticality[crit_str]
        except KeyError:
            criticality = SafetyCriticality.NONE

        return EnrichmentResult(
            term=term,
            positive_examples=data.get("positive_examples", [])[:5],
            negative_examples=data.get("negative_examples", [])[:5],
            disambiguation=data.get("disambiguation", ""),
            safety_criticality=criticality,
            safety_rationale=data.get("safety_rationale", ""),
            harness_relevant=data.get("harness_relevant", False),
            treaty_relevant=data.get("treaty_relevant", False),
        )


def find_concepts_needing_enrichment(
    concepts_dir: Path,
    needs_training: bool = True,
    needs_safety: bool = True,
    limit: int = None,
) -> List[Path]:
    """Find concept files that need enrichment."""
    candidates = []

    for layer_dir in sorted(concepts_dir.iterdir()):
        if not layer_dir.is_dir():
            continue

        for concept_file in layer_dir.glob("*.json"):
            with open(concept_file) as f:
                data = json.load(f)

            needs_work = False

            if needs_training:
                hints = data.get("training_hints", {})
                pos = hints.get("positive_examples", [])
                neg = hints.get("negative_examples", [])
                if not pos or not neg:
                    needs_work = True

            if needs_safety:
                safety = data.get("safety_tags", {})
                # Check if safety assessment looks incomplete
                if not safety.get("safety_rationale"):
                    term = data.get("term", "").lower()
                    # Prioritize concepts that look safety-relevant
                    safety_keywords = [
                        'deception', 'manipulation', 'coercion', 'consent',
                        'autonomy', 'safety', 'alignment', 'signal', 'mask',
                        'sycophancy', 'escape', 'resource', 'self', 'suppress',
                        'persona', 'agency', 'metacognition', 'divergence',
                    ]
                    if any(kw in term for kw in safety_keywords):
                        needs_work = True

            if needs_work:
                candidates.append(concept_file)
                if limit and len(candidates) >= limit:
                    return candidates

    return candidates


def update_concept_file(concept_file: Path, result: EnrichmentResult) -> bool:
    """Update a concept file with enrichment results."""
    try:
        with open(concept_file) as f:
            data = json.load(f)

        # Update training hints
        if "training_hints" not in data:
            data["training_hints"] = {}

        # Only update if we have examples and they're not already populated
        existing_pos = data["training_hints"].get("positive_examples", [])
        existing_neg = data["training_hints"].get("negative_examples", [])

        if not existing_pos and result.positive_examples:
            data["training_hints"]["positive_examples"] = result.positive_examples
        if not existing_neg and result.negative_examples:
            data["training_hints"]["negative_examples"] = result.negative_examples
        if result.disambiguation:
            data["training_hints"]["disambiguation"] = result.disambiguation

        # Update safety tags
        if "safety_tags" not in data:
            data["safety_tags"] = {}

        # Map criticality to risk_level
        crit_to_risk = {
            SafetyCriticality.NONE: "none",
            SafetyCriticality.LOW: "low",
            SafetyCriticality.MEDIUM: "medium",
            SafetyCriticality.HIGH: "high",
            SafetyCriticality.CRITICAL: "critical",
        }

        data["safety_tags"]["risk_level"] = crit_to_risk[result.safety_criticality]
        data["safety_tags"]["safety_rationale"] = result.safety_rationale
        data["safety_tags"]["harness_relevant"] = result.harness_relevant
        data["safety_tags"]["treaty_relevant"] = result.treaty_relevant

        # Write back
        with open(concept_file, 'w') as f:
            json.dump(data, f, indent=2)

        return True
    except Exception as e:
        print(f"  Error updating {concept_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Enrich concept pack metadata using LLM")
    parser.add_argument(
        "--pack", "-p",
        default="concept_packs/first-light",
        help="Path to concept pack"
    )
    parser.add_argument(
        "--model", "-m",
        default="gemma-2b",
        help="Model to use (gemma-2b, qwen-0.5b, qwen-1.5b, or API model name)"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Use API instead of local model"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Max concepts to process (default: 10)"
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Only enrich training examples"
    )
    parser.add_argument(
        "--safety-only",
        action="store_true",
        help="Only enrich safety-relevant concepts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually generate or save (for testing)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds"
    )

    args = parser.parse_args()

    pack_path = Path(args.pack)
    concepts_dir = pack_path / "concepts"

    if not concepts_dir.exists():
        print(f"Concepts directory not found: {concepts_dir}")
        return

    # Find concepts needing work
    print(f"Scanning {concepts_dir} for concepts needing enrichment...")

    needs_training = not args.safety_only
    needs_safety = not args.training_only or args.safety_only

    candidates = find_concepts_needing_enrichment(
        concepts_dir,
        needs_training=needs_training,
        needs_safety=needs_safety,
        limit=args.limit,
    )

    print(f"Found {len(candidates)} concepts needing enrichment")

    if not candidates:
        print("Nothing to do!")
        return

    # Initialize enricher
    enricher = ConceptEnricher(
        model_name=args.model,
        use_api=args.api,
        dry_run=args.dry_run,
    )
    enricher.initialize()

    # Process concepts
    success = 0
    failed = 0

    for i, concept_file in enumerate(candidates):
        with open(concept_file) as f:
            data = json.load(f)

        term = data.get("term", concept_file.stem)
        print(f"[{i+1}/{len(candidates)}] Enriching: {term}")

        result = enricher.enrich_concept(data)

        if result:
            if args.dry_run:
                print(f"  [DRY RUN] Would update {concept_file.name}")
                print(f"    Safety: {result.safety_criticality.value}")
                success += 1
            else:
                if update_concept_file(concept_file, result):
                    print(f"  ✓ Updated with safety={result.safety_criticality.value}")
                    success += 1
                else:
                    failed += 1
        else:
            failed += 1

        if args.api and not args.dry_run and i < len(candidates) - 1:
            time.sleep(args.delay)

    print()
    print(f"Complete: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
