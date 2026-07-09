#!/usr/bin/env python3
"""
Generate MELDs with Full Ontology Context - Phase 2 of Two-Phase Generation

Uses the skeleton structure to provide full context (parents, siblings, AND children)
when generating detailed MELDs. This tests the hypothesis that knowing the full
structure improves MELD quality.

Usage:
    # Generate MELDs for L1 pillars with child context from skeleton
    python scripts/generate_melds_with_context.py --skeleton results/ontology_skeleton.json --level 1

    # Generate MELDs for L2 departments
    python scripts/generate_melds_with_context.py --skeleton results/ontology_skeleton.json --level 2

    # Regenerate specific concept
    python scripts/generate_melds_with_context.py --skeleton results/ontology_skeleton.json --concept "Economic Activity"
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.be.thalamos.model_candidates import CandidateLoader, MODEL_CANDIDATES
from scripts.generate_ontology_skeleton import OntologySkeleton, SkeletonNode, LEVEL_METAPHORS
from scripts.meld_review_pipeline import (
    MeldReviewPipeline, MeldReviewState, ReviewStage, HierarchyContext,
    MeldGenerator, MinistralJudge
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT-AWARE MELD GENERATION PROMPT
# =============================================================================

CONTEXT_AWARE_MELD_PROMPT = """You are an expert ontologist creating a MELD (Modular Enhancement Layer Description) for training concept lenses.

## Your Position in the University of Human Knowledge

{university_context}

## Your Task

Create a detailed MELD for: **{concept_label}**

Scope: {concept_scope}

## Context Awareness

Use your knowledge of the structure above to:
1. **Definition**: Scope it precisely - you know exactly what {children_term} exist within this {level_name}
2. **Positive Examples**: Draw from the full range of {children_term}: {children_list}
3. **Negative Examples**: Include near-misses from sibling {level_plural}: {siblings_list}
4. **Contrast Concepts**: Your siblings are natural contrast concepts
5. **Steering Target**: Consider what concept in a different {parent_level_name} would be the opposite

## MELD Requirements

Generate a complete MELD with:

1. **definition** (20-200 chars): Clear definitional statement
   - Reference the scope of your {children_term} without listing them all

2. **positive_examples** (6-10): Natural language where this concept activates
   - Diverse scenarios across your {children_term}
   - Span everyday to professional contexts

3. **negative_examples** (6-10): Examples that should NOT activate this concept
   - 3-4 near-misses from siblings: {siblings_sample}
   - 3-4 from different {parent_level_plural}

4. **contrast_concepts** (3-5): Related concepts to discriminate against
   - Your siblings: {siblings_list}

5. **opposite_concept**: Semantic opposite for steering
   - Something from a contrasting {parent_level_name}

6. **safety_tags**:
   - risk_level: "low" | "medium" | "high"
   - treaty_relevant: false
   - harness_relevant: false

## Output Format

Return valid JSON:
```json
{{
  "term": "{concept_label}",
  "definition": "...",
  "positive_examples": ["...", "..."],
  "negative_examples": ["...", "..."],
  "contrast_concepts": ["...", "..."],
  "opposite_concept": "...",
  "safety_tags": {{
    "risk_level": "low",
    "treaty_relevant": false,
    "harness_relevant": false
  }},
  "training_hints": {{
    "disambiguation": "how to distinguish from siblings",
    "key_features": ["feature1", "feature2"],
    "confusable_with": ["sibling1", "sibling2"]
  }}
}}
```

Generate the MELD now. JSON only, no commentary.
"""

UNIVERSITY_CONTEXT_TEMPLATE = """### University Structure

**{parent_level_name}**: {parent_label}
{parent_scope}

**{level_name}s in this {parent_level_name}** (your siblings):
{siblings_formatted}

**Your {children_term}** (what you contain):
{children_formatted}

**Other {parent_level_plural}** (for contrast):
{other_parents_formatted}
"""


# =============================================================================
# CONTEXT-AWARE GENERATOR
# =============================================================================

class ContextAwareMeldGenerator:
    """Generates MELDs with full ontology context."""

    def __init__(self, model_id: str = "gemma-3-4b-it"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.loader = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        logger.info(f"Loading context-aware generator: {self.model_id}")
        candidate = MODEL_CANDIDATES[self.model_id]
        self.loader = CandidateLoader()
        self.model, self.tokenizer, _ = self.loader.load(candidate)
        self.loaded = True

    def unload(self):
        if self.loader:
            self.loader.unload()
            self.model = None
            self.tokenizer = None
            self.loaded = False

    def _generate_text(self, prompt: str, max_tokens: int = 2000) -> str:
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)

        input_len = inputs.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from model response with multiple fallback strategies."""
        strategies = []

        # Strategy 1: Code block with json tag
        if "```json" in text:
            json_text = text.split("```json")[1].split("```")[0].strip()
            strategies.append(json_text)

        # Strategy 2: Any code block
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                strategies.append(parts[1].strip())

        # Strategy 3: Find balanced braces
        start = text.find("{")
        if start >= 0:
            # Find matching closing brace
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        strategies.append(text[start:i+1])
                        break

        # Strategy 4: Simple first-to-last brace
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            strategies.append(text[start:end])

        # Try each strategy
        for json_text in strategies:
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                continue

        return None

    def _format_list(self, items: List[str], max_items: int = 8) -> str:
        if not items:
            return "None"
        formatted = [f"- {item}" for item in items[:max_items]]
        if len(items) > max_items:
            formatted.append(f"- ... and {len(items) - max_items} more")
        return "\n".join(formatted)

    def _build_university_context(
        self,
        node: SkeletonNode,
        skeleton: OntologySkeleton,
        parent: SkeletonNode = None,
        siblings: List[SkeletonNode] = None,
    ) -> str:
        """Build rich university context for the prompt."""
        level_meta = LEVEL_METAPHORS.get(node.level, LEVEL_METAPHORS[4])
        parent_level_meta = LEVEL_METAPHORS.get(node.level - 1, LEVEL_METAPHORS[1])
        child_level_meta = LEVEL_METAPHORS.get(node.level + 1, LEVEL_METAPHORS[4])

        # Format siblings
        siblings = siblings or []
        siblings_formatted = self._format_list([f"{s.label}: {s.scope}" for s in siblings])

        # Format children
        children_formatted = self._format_list([f"{c.label}: {c.scope}" for c in node.children])

        # Format other parents (for contrast)
        other_parents = []
        if node.level == 1:
            other_parents = [r for r in skeleton.roots if r.id != node.id]
        elif parent:
            # Get other items at parent's level
            for root in skeleton.roots:
                if root.id != parent.id:
                    other_parents.append(root)
        other_parents_formatted = self._format_list([p.label for p in other_parents[:5]])

        return UNIVERSITY_CONTEXT_TEMPLATE.format(
            parent_level_name=parent_level_meta["name"] if parent else "University",
            parent_label=parent.label if parent else "University of Human Knowledge",
            parent_scope=parent.scope if parent else "All human knowledge and activity",
            level_name=level_meta["name"],
            siblings_formatted=siblings_formatted,
            children_term=child_level_meta["plural"].lower(),
            children_formatted=children_formatted,
            parent_level_plural=parent_level_meta["plural"],
            other_parents_formatted=other_parents_formatted,
        )

    def generate_meld(
        self,
        node: SkeletonNode,
        skeleton: OntologySkeleton,
        parent: SkeletonNode = None,
        siblings: List[SkeletonNode] = None,
    ) -> Dict:
        """Generate a MELD with full context."""
        if not self.loaded:
            self.load()

        level_meta = LEVEL_METAPHORS.get(node.level, LEVEL_METAPHORS[4])
        parent_level_meta = LEVEL_METAPHORS.get(node.level - 1, LEVEL_METAPHORS[1])
        child_level_meta = LEVEL_METAPHORS.get(node.level + 1, LEVEL_METAPHORS[4])

        siblings = siblings or []
        sibling_labels = [s.label for s in siblings]
        children_labels = [c.label for c in node.children]

        university_context = self._build_university_context(node, skeleton, parent, siblings)

        prompt = CONTEXT_AWARE_MELD_PROMPT.format(
            university_context=university_context,
            concept_label=node.label,
            concept_scope=node.scope,
            level_name=level_meta["name"],
            level_plural=level_meta["plural"],
            children_term=child_level_meta["plural"].lower() if node.children else "subconcepts",
            children_list=", ".join(children_labels[:5]) or "None defined",
            siblings_list=", ".join(sibling_labels[:5]) or "None",
            siblings_sample=", ".join(sibling_labels[:3]) or "None",
            parent_level_name=parent_level_meta["name"],
            parent_level_plural=parent_level_meta["plural"],
        )

        # Retry logic for JSON extraction failures
        max_retries = 3
        meld_data = None

        for attempt in range(max_retries):
            response = self._generate_text(prompt)
            meld_data = self._extract_json(response)

            if meld_data:
                break
            else:
                logger.warning(f"JSON extraction failed (attempt {attempt + 1}/{max_retries}) for {node.label}")
                if attempt == max_retries - 1:
                    # Log the raw response on final failure for debugging
                    logger.debug(f"Raw response: {response[:500]}...")

        if meld_data:
            # Add context metadata
            meld_data["_generation_context"] = {
                "method": "context-aware",
                "had_children": len(node.children) > 0,
                "num_children": len(node.children),
                "num_siblings": len(siblings),
                "children_labels": children_labels,
                "sibling_labels": sibling_labels,
            }

        return meld_data


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def find_node_by_label(skeleton: OntologySkeleton, label: str) -> Optional[SkeletonNode]:
    """Find a node by label (case-insensitive partial match)."""
    label_lower = label.lower()

    def search(node: SkeletonNode) -> Optional[SkeletonNode]:
        if label_lower in node.label.lower():
            return node
        for child in node.children:
            result = search(child)
            if result:
                return result
        return None

    for root in skeleton.roots:
        result = search(root)
        if result:
            return result
    return None


def find_parent_and_siblings(
    skeleton: OntologySkeleton,
    node: SkeletonNode
) -> tuple[Optional[SkeletonNode], List[SkeletonNode]]:
    """Find the parent and siblings of a node."""
    if node.level == 1:
        # L1 nodes: parent is None, siblings are other roots
        siblings = [r for r in skeleton.roots if r.id != node.id]
        return None, siblings

    # Search for parent
    def search(current: SkeletonNode) -> Optional[SkeletonNode]:
        for child in current.children:
            if child.id == node.id:
                return current
            result = search(child)
            if result:
                return result
        return None

    for root in skeleton.roots:
        if root.id == node.id:
            return None, [r for r in skeleton.roots if r.id != node.id]
        parent = search(root)
        if parent:
            siblings = [c for c in parent.children if c.id != node.id]
            return parent, siblings

    return None, []


def generate_melds_for_level(
    skeleton: OntologySkeleton,
    level: int,
    output_dir: Path,
    model_id: str = "gemma-3-4b-it",
    judge_mode: str = "annotator",
    skip_judge: bool = False,
    skip_existing: bool = False,
) -> Dict:
    """Generate MELDs for all nodes at a given level."""
    nodes = skeleton.get_all_at_level(level)
    logger.info(f"Generating MELDs for {len(nodes)} nodes at level {level}")

    generator = ContextAwareMeldGenerator(model_id=model_id)
    judge = None if skip_judge else MinistralJudge()

    results = {
        "level": level,
        "total": len(nodes),
        "approved": 0,
        "rejected": 0,
        "melds": [],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, node in enumerate(nodes):
            # Check if file exists (for skip_existing mode)
            safe_name = node.label.lower().replace(" ", "_").replace("/", "_").replace("&", "and")
            result_file = output_dir / f"L{level}_{safe_name}.json"

            if skip_existing and result_file.exists():
                logger.info(f"[{i+1}/{len(nodes)}] Skipping existing: {node.label}")
                results["approved"] += 1  # Count as approved
                continue

            logger.info(f"[{i+1}/{len(nodes)}] Generating MELD for: {node.label}")

            parent, siblings = find_parent_and_siblings(skeleton, node)

            # Generate MELD with context
            meld_data = generator.generate_meld(node, skeleton, parent, siblings)

            if not meld_data:
                logger.warning(f"Failed to generate MELD for {node.label}")
                results["rejected"] += 1
                continue

            # Review with judge (if enabled)
            review_passed = True
            review_feedback = "Skipped (no judge)"
            review_confidence = 1.0
            worldview_metadata = None

            if judge:
                hierarchy_context = HierarchyContext(
                    level=level,
                    parent_concept=parent.label if parent else None,
                    sibling_concepts=[s.label for s in siblings],
                )

                review_result, worldview_metadata = judge.review(
                    meld_data,
                    hierarchy_context,
                    predefined_definition=False,
                    mode=judge_mode,
                )
                review_passed = review_result.passed
                review_feedback = review_result.feedback
                review_confidence = review_result.confidence

            # Save result
            result_data = {
                "node": {
                    "id": node.id,
                    "label": node.label,
                    "scope": node.scope,
                    "level": node.level,
                },
                "meld_data": meld_data,
                "review": {
                    "passed": review_passed,
                    "feedback": review_feedback,
                    "confidence": review_confidence,
                },
                "worldview_metadata": worldview_metadata,
            }

            safe_name = node.label.lower().replace(" ", "_").replace("/", "_").replace("&", "and")
            result_file = output_dir / f"L{level}_{safe_name}.json"
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)

            if review_passed:
                results["approved"] += 1
            else:
                results["rejected"] += 1

            results["melds"].append({
                "label": node.label,
                "passed": review_passed,
                "divergence_score": worldview_metadata.get("divergence_score", 0) if worldview_metadata else 0,
            })

    finally:
        generator.unload()
        if judge:
            judge.unload()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate MELDs with full ontology context (Phase 2)"
    )
    parser.add_argument(
        "--skeleton", "-s",
        type=Path,
        default=Path("results/ontology_skeleton.json"),
        help="Path to ontology skeleton JSON"
    )
    parser.add_argument(
        "--level", "-l",
        type=int,
        default=1,
        help="Level to generate MELDs for (default: 1)"
    )
    parser.add_argument(
        "--concept", "-c",
        type=str,
        help="Generate MELD for specific concept (by label)"
    )
    parser.add_argument(
        "--model", "-m",
        default="gemma-3-4b-it",
        help="Model for MELD generation"
    )
    parser.add_argument(
        "--judge-mode",
        choices=["gatekeeper", "annotator"],
        default="annotator",
        help="Judge mode"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip Ministral judge review (saves VRAM, faster)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results/context_aware_melds"),
        help="Output directory"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip nodes that already have MELD files"
    )

    args = parser.parse_args()

    # Load skeleton
    skeleton = OntologySkeleton.load(args.skeleton)
    logger.info(f"Loaded skeleton with {len(skeleton.roots)} roots")

    if args.concept:
        # Generate for specific concept
        node = find_node_by_label(skeleton, args.concept)
        if not node:
            logger.error(f"Concept not found: {args.concept}")
            sys.exit(1)

        generator = ContextAwareMeldGenerator(model_id=args.model)
        parent, siblings = find_parent_and_siblings(skeleton, node)

        try:
            meld_data = generator.generate_meld(node, skeleton, parent, siblings)
            print(json.dumps(meld_data, indent=2))
        finally:
            generator.unload()
    else:
        # Generate for entire level
        results = generate_melds_for_level(
            skeleton=skeleton,
            level=args.level,
            output_dir=args.output / f"L{args.level}",
            model_id=args.model,
            judge_mode=args.judge_mode,
            skip_judge=args.no_judge,
            skip_existing=args.skip_existing,
        )

        print(f"\n{'='*60}")
        print(f"CONTEXT-AWARE MELD GENERATION - LEVEL {args.level}")
        print(f"{'='*60}")
        print(f"Total: {results['total']}")
        print(f"Approved: {results['approved']}")
        print(f"Rejected: {results['rejected']}")
        print(f"\nResults saved to: {args.output / f'L{args.level}'}")


if __name__ == "__main__":
    main()
