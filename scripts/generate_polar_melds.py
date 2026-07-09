#!/usr/bin/env python3
"""
Generate Polar MELDs - capturing both positive and negative poles of each concept.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Load the polar MELD prompt template
POLAR_MELD_PROMPT = Path("melds/prompts/polar_meld_prompt.txt").read_text()

LEVEL_NAMES = {
    1: "Faculty",
    2: "University",
    3: "School",
    4: "Department"
}


class SkeletonNode:
    """Simple skeleton node."""
    def __init__(self, data: dict):
        self.id = data.get("id", "")
        self.label = data.get("label", "")
        self.scope = data.get("scope", "")
        self.level = data.get("level", 1)
        self.children = [SkeletonNode(c) for c in data.get("children", [])]


class PolarMeldGenerator:
    """Generate polar MELDs using a local model."""

    def __init__(self, model_id: str = "google/gemma-3-4b-it"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        logger.info(f"Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.loaded = True
        logger.info("Model loaded")

    def unload(self):
        if self.model:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.loaded = False

    def _generate_text(self, prompt: str, max_tokens: int = 3000) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

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
        """Extract JSON with multiple fallback strategies."""
        import re

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
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        strategies.append(text[start:i+1])
                        break

        # Try each strategy
        for json_text in strategies:
            try:
                # Clean control characters
                json_text = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f]', '', json_text)
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

    def _build_context(
        self,
        node: SkeletonNode,
        parent: Optional[SkeletonNode],
        siblings: List[SkeletonNode],
    ) -> str:
        """Build university context for the prompt."""
        parent_label = parent.label if parent else "Root (Model Self-Map)"
        parent_scope = parent.scope if parent else "The complete map of model knowledge and capabilities"

        siblings_formatted = self._format_list([f"{s.label}: {s.scope}" for s in siblings])
        children_formatted = self._format_list([f"{c.label}: {c.scope}" for c in node.children])

        return f"""## Hierarchy Position
Parent {LEVEL_NAMES.get(node.level - 1, 'Concept')}: {parent_label}
  Scope: {parent_scope}

Sibling {LEVEL_NAMES.get(node.level, 'Concept')}s:
{siblings_formatted}

Child {LEVEL_NAMES.get(node.level + 1, 'Concept')}s of this node:
{children_formatted}
"""

    def generate_polar_meld(
        self,
        node: SkeletonNode,
        parent: Optional[SkeletonNode],
        siblings: List[SkeletonNode],
    ) -> Optional[Dict]:
        """Generate a polar MELD for a concept."""
        if not self.loaded:
            self.load()

        context = self._build_context(node, parent, siblings)
        sibling_labels = [s.label for s in siblings]
        children_labels = [c.label for c in node.children]

        prompt = POLAR_MELD_PROMPT.format(
            university_context=context,
            concept_label=node.label,
            concept_scope=node.scope,
            level_name=LEVEL_NAMES.get(node.level, "Concept"),
            parent_label=parent.label if parent else "Root",
            children_list=", ".join(children_labels[:5]) or "None",
            siblings_list=", ".join(sibling_labels[:5]) or "None",
        )

        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            response = self._generate_text(prompt)
            meld_data = self._extract_json(response)

            if meld_data:
                # Add generation context
                meld_data["_generation_context"] = {
                    "method": "polar-meld-v1",
                    "level": node.level,
                    "parent": parent.label if parent else None,
                    "siblings": sibling_labels,
                    "children": children_labels,
                }
                return meld_data
            else:
                logger.warning(f"JSON extraction failed (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    logger.debug(f"Raw response: {response[:500]}...")

        return None


def load_skeleton(path: Path) -> tuple[List[SkeletonNode], Dict]:
    """Load skeleton and return roots + lookup."""
    data = json.loads(path.read_text())
    roots = [SkeletonNode(r) for r in data.get("roots", [])]
    return roots, data


def get_nodes_at_level(roots: List[SkeletonNode], level: int) -> List[tuple[SkeletonNode, Optional[SkeletonNode], List[SkeletonNode]]]:
    """Get all nodes at a level with their parent and siblings."""
    results = []

    def traverse(node: SkeletonNode, parent: Optional[SkeletonNode], siblings: List[SkeletonNode]):
        if node.level == level:
            results.append((node, parent, siblings))
        for child in node.children:
            child_siblings = [c for c in node.children if c.id != child.id]
            traverse(child, node, child_siblings)

    # For L1, siblings are other roots
    for root in roots:
        root_siblings = [r for r in roots if r.id != root.id]
        traverse(root, None, root_siblings)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate Polar MELDs")
    parser.add_argument(
        "--skeleton", "-s",
        type=Path,
        default=Path("results/introspective_skeleton.json"),
        help="Path to skeleton JSON"
    )
    parser.add_argument(
        "--level", "-l",
        type=int,
        default=1,
        help="Level to generate MELDs for"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results/polar_melds"),
        help="Output directory"
    )
    parser.add_argument(
        "--model", "-m",
        default="google/gemma-3-4b-it",
        help="Model for generation"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip nodes that already have MELD files"
    )

    args = parser.parse_args()

    # Load skeleton
    roots, _ = load_skeleton(args.skeleton)
    logger.info(f"Loaded skeleton with {len(roots)} roots")

    # Get nodes at target level
    nodes = get_nodes_at_level(roots, args.level)
    logger.info(f"Found {len(nodes)} nodes at level {args.level}")

    # Setup output
    output_dir = args.output / f"L{args.level}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate
    generator = PolarMeldGenerator(model_id=args.model)

    results = {"total": len(nodes), "success": 0, "failed": 0}

    try:
        for i, (node, parent, siblings) in enumerate(nodes):
            safe_name = node.label.lower().replace(" ", "_").replace("/", "_").replace("&", "and")
            output_file = output_dir / f"L{args.level}_{safe_name}.json"

            if args.skip_existing and output_file.exists():
                logger.info(f"[{i+1}/{len(nodes)}] Skipping existing: {node.label}")
                results["success"] += 1
                continue

            logger.info(f"[{i+1}/{len(nodes)}] Generating polar MELD for: {node.label}")

            meld_data = generator.generate_polar_meld(node, parent, siblings)

            if meld_data:
                # Save
                result = {
                    "node": {
                        "id": node.id,
                        "label": node.label,
                        "scope": node.scope,
                        "level": node.level,
                    },
                    "polar_meld": meld_data,
                }
                output_file.write_text(json.dumps(result, indent=2))
                results["success"] += 1
                logger.info(f"  Saved: {output_file.name}")
            else:
                results["failed"] += 1
                logger.warning(f"  Failed to generate MELD for {node.label}")

    finally:
        generator.unload()

    print(f"\n{'='*60}")
    print(f"POLAR MELD GENERATION - LEVEL {args.level}")
    print(f"{'='*60}")
    print(f"Total: {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
