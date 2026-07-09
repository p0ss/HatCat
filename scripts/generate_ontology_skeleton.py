#!/usr/bin/env python3
"""
Generate Ontology Skeleton - Phase 1 of Two-Phase MELD Generation

Uses a hierarchical structure to recursively build the full ontology skeleton
BEFORE populating detailed MELDs. This gives each concept full context about its
siblings, parents, and children when we later generate MELDs.

Structure (deep hierarchy for mapping model internals):
- L1: Fields (13 from ontologist) - e.g., "Violence & Conflict (Strategic & Reactive)"
- L2: Universities (~12 per field) - e.g., "University of Strategic Warfare"
- L3: Schools (~12 per university) - e.g., "School of Military Intelligence"
- L4: Departments (~12 per school) - e.g., "Department of Signals Intelligence"
- L5: Courses (optional) - e.g., "SIGINT Analysis 301"

MELD Training Scope:
- L1-L3: ~2,000 MELDs to train (~13 + ~156 + ~1,872)
- L4+: Skeleton only (provides child context for L3 MELD generation)

Usage:
    # Generate L2-L4 skeleton from existing L1 pillars (fields)
    python scripts/generate_ontology_skeleton.py --from-l1 results/l1_pillars/pillars.json

    # Generate full skeleton L1->L4
    python scripts/generate_ontology_skeleton.py --full --max-depth 4

    # Resume from existing skeleton and expand a specific level
    python scripts/generate_ontology_skeleton.py --resume results/skeleton.json --expand-level 2
"""

import json
import re
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.be.thalamos.model_candidates import CandidateLoader, MODEL_CANDIDATES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# UNIVERSITY METAPHOR DEFINITIONS
# =============================================================================

LEVEL_METAPHORS = {
    1: {
        "name": "Field",
        "plural": "Fields",
        "description": "Top-level domain of human knowledge and activity (from ontologist)",
        "example": "Violence & Conflict (Strategic & Reactive)",
        "child_prompt": "universities",
        "expected_children": "10-14",
    },
    2: {
        "name": "University",
        "plural": "Universities",
        "description": "A focused area of study within a field, like a specialized university",
        "example": "University of Strategic Warfare",
        "child_prompt": "schools",
        "expected_children": "10-14",
    },
    3: {
        "name": "School",
        "plural": "Schools",
        "description": "A faculty/school within the university covering a major subdomain",
        "example": "School of Military Intelligence",
        "child_prompt": "departments",
        "expected_children": "10-14",
    },
    4: {
        "name": "Department",
        "plural": "Departments",
        "description": "Academic department within a school",
        "example": "Department of Signals Intelligence",
        "child_prompt": "courses",
        "expected_children": "8-12",
    },
    5: {
        "name": "Course",
        "plural": "Courses",
        "description": "Specific course/class within a department",
        "example": "SIGINT Analysis 301",
        "child_prompt": "topics",
        "expected_children": "5-10",
    },
}


# =============================================================================
# SKELETON DATA STRUCTURES
# =============================================================================

@dataclass
class SkeletonNode:
    """A node in the ontology skeleton."""
    id: str
    label: str
    scope: str  # Brief description of what this covers
    level: int
    parent_id: Optional[str] = None
    children: List['SkeletonNode'] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "label": self.label,
            "scope": self.scope,
            "level": self.level,
            "parent_id": self.parent_id,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SkeletonNode':
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            id=data["id"],
            label=data["label"],
            scope=data.get("scope", ""),
            level=data["level"],
            parent_id=data.get("parent_id"),
            children=children,
            metadata=data.get("metadata", {}),
        )

    def get_sibling_labels(self, parent: 'SkeletonNode' = None) -> List[str]:
        """Get labels of siblings (other children of same parent)."""
        if parent is None:
            return []
        return [c.label for c in parent.children if c.id != self.id]

    def get_child_labels(self) -> List[str]:
        """Get labels of direct children."""
        return [c.label for c in self.children]


@dataclass
class OntologySkeleton:
    """The complete ontology skeleton structure."""
    roots: List[SkeletonNode] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generator_model: str = ""
    max_depth: int = 5

    def to_dict(self) -> Dict:
        return {
            "generated_at": self.generated_at,
            "generator_model": self.generator_model,
            "max_depth": self.max_depth,
            "roots": [r.to_dict() for r in self.roots],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'OntologySkeleton':
        roots = [SkeletonNode.from_dict(r) for r in data.get("roots", [])]
        return cls(
            roots=roots,
            generated_at=data.get("generated_at", ""),
            generator_model=data.get("generator_model", ""),
            max_depth=data.get("max_depth", 4),
        )

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved skeleton to {path}")

    @classmethod
    def load(cls, path: Path) -> 'OntologySkeleton':
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def count_nodes(self) -> Dict[int, int]:
        """Count nodes at each level."""
        counts = {}
        def count_recursive(node: SkeletonNode):
            counts[node.level] = counts.get(node.level, 0) + 1
            for child in node.children:
                count_recursive(child)
        for root in self.roots:
            count_recursive(root)
        return counts

    def get_all_at_level(self, level: int) -> List[SkeletonNode]:
        """Get all nodes at a specific level."""
        nodes = []
        def collect_recursive(node: SkeletonNode):
            if node.level == level:
                nodes.append(node)
            for child in node.children:
                collect_recursive(child)
        for root in self.roots:
            collect_recursive(root)
        return nodes


# =============================================================================
# SKELETON GENERATOR
# =============================================================================

SKELETON_PROMPT = """You are designing the curriculum for the **University of Human Knowledge**.

{context}

## Your Task

For the {level_name} "{parent_label}", list the {child_prompt} that should exist within it.

Requirements:
- List {expected_children} {child_plural}
- Each should be distinct and non-overlapping (MECE - Mutually Exclusive, Collectively Exhaustive)
- Together they should cover the full scope of "{parent_label}"
- Provide a brief scope statement (1-2 sentences) for each

## Output Format

Return JSON array:
```json
[
  {{"id": "kebab-case-id", "label": "Human Readable Label", "scope": "Brief description of what this covers"}},
  ...
]
```

Generate the {child_plural} now. JSON only, no commentary.
"""

CONTEXT_TEMPLATE = """## University Structure So Far

{structure_summary}

## Current Position

You are now populating the {level_name} level.
- Parent: {parent_label} ({parent_level_name})
- Siblings of parent: {parent_siblings}
"""


class SkeletonGenerator:
    """Generates the ontology skeleton using a local model."""

    def __init__(self, model_id: str = "gemma-3-4b-it"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.loader = None
        self.loaded = False

    def load(self):
        """Load the local model."""
        if self.loaded:
            return

        if self.model_id not in MODEL_CANDIDATES:
            raise ValueError(f"Unknown model: {self.model_id}")

        logger.info(f"Loading skeleton generator: {self.model_id}")
        candidate = MODEL_CANDIDATES[self.model_id]
        self.loader = CandidateLoader()
        self.model, self.tokenizer, _ = self.loader.load(candidate)
        self.loaded = True
        logger.info(f"Generator loaded, using {self.loader.get_vram_usage():.1f}GB VRAM")

    def unload(self):
        """Unload model to free VRAM."""
        if self.loader is not None:
            self.loader.unload()
            self.model = None
            self.tokenizer = None
            self.loaded = False

    def _generate_text(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text from a prompt."""
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
            input_ids = inputs
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_ids = inputs["input_ids"]

        input_len = input_ids.shape[1] if len(input_ids.shape) > 1 else input_ids.shape[0]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids if len(input_ids.shape) > 1 else input_ids.unsqueeze(0),
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _extract_json(self, text: str) -> Optional[List[Dict]]:
        """Extract JSON array from response text."""
        try:
            if "```json" in text:
                json_text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_text = text.split("```")[1].split("```")[0].strip()
            else:
                start = text.find("[")
                end = text.rfind("]") + 1
                if start >= 0 and end > start:
                    json_text = text[start:end]
                else:
                    json_text = text.strip()
            return json.loads(json_text)
        except json.JSONDecodeError:
            return None

    def _build_structure_summary(self, skeleton: OntologySkeleton, current_level: int) -> str:
        """Build a summary of the structure so far for context."""
        lines = []

        for root in skeleton.roots:
            lines.append(f"**Field**: {root.label}")
            if current_level > 2 and root.children:
                for uni in root.children[:4]:
                    lines.append(f"  - University: {uni.label}")
                    if current_level > 3 and uni.children:
                        for school in uni.children[:3]:
                            lines.append(f"    - School: {school.label}")
                            if current_level > 4 and school.children:
                                for dept in school.children[:2]:
                                    lines.append(f"      - Department: {dept.label}")
                if len(root.children) > 4:
                    lines.append(f"  - ... and {len(root.children) - 4} more universities")

        return "\n".join(lines) if lines else "Starting fresh - no structure yet."

    def generate_children(
        self,
        parent: SkeletonNode,
        skeleton: OntologySkeleton,
        parent_siblings: List[str] = None,
    ) -> List[SkeletonNode]:
        """Generate children for a parent node."""
        if not self.loaded:
            self.load()

        child_level = parent.level + 1
        if child_level > 5:
            return []  # Don't go beyond L5

        level_meta = LEVEL_METAPHORS.get(child_level, LEVEL_METAPHORS[4])
        parent_level_meta = LEVEL_METAPHORS.get(parent.level, LEVEL_METAPHORS[1])

        # Build context
        context = CONTEXT_TEMPLATE.format(
            structure_summary=self._build_structure_summary(skeleton, child_level),
            level_name=level_meta["name"],
            parent_label=parent.label,
            parent_level_name=parent_level_meta["name"],
            parent_siblings=", ".join(parent_siblings[:5]) if parent_siblings else "None",
        )

        prompt = SKELETON_PROMPT.format(
            context=context,
            level_name=parent_level_meta["name"],
            parent_label=parent.label,
            child_prompt=level_meta["plural"].lower(),
            expected_children=level_meta["expected_children"],
            child_plural=level_meta["plural"],
        )

        response = self._generate_text(prompt)
        children_data = self._extract_json(response)

        if not children_data:
            logger.warning(f"Failed to parse children for {parent.label}")
            return []

        children = []
        for item in children_data:
            # Generate base ID from model output or derive from label
            base_id = item.get("id", item.get("label", "unknown").lower().replace(" ", "-"))
            # Strip any numeric suffix the model may have added (e.g., "-9")
            base_id = re.sub(r'-\d+$', '', base_id)
            # Create qualified ID: parent--child to ensure uniqueness across branches
            qualified_id = f"{parent.id}--{base_id}"

            child = SkeletonNode(
                id=qualified_id,
                label=item.get("label", "Unknown"),
                scope=item.get("scope", ""),
                level=child_level,
                parent_id=parent.id,
            )
            children.append(child)

        logger.info(f"Generated {len(children)} {level_meta['plural'].lower()} for {parent.label}")
        return children

    def expand_level(
        self,
        skeleton: OntologySkeleton,
        level: int,
    ) -> int:
        """Expand all nodes at a given level by generating their children.

        Returns the number of children generated.
        """
        nodes_at_level = skeleton.get_all_at_level(level)
        total_children = 0

        for i, node in enumerate(nodes_at_level):
            if node.children:
                logger.info(f"Skipping {node.label} - already has children")
                continue

            logger.info(f"[{i+1}/{len(nodes_at_level)}] Expanding: {node.label}")

            # Find parent to get siblings
            parent_siblings = []
            if node.parent_id:
                for root in skeleton.roots:
                    if root.id == node.parent_id:
                        parent_siblings = [c.label for c in root.children if c.id != node.id]
                        break
                    for child in root.children:
                        if child.id == node.parent_id:
                            parent_siblings = [c.label for c in child.children if c.id != node.id]
                            break

            children = self.generate_children(node, skeleton, parent_siblings)
            node.children = children
            total_children += len(children)

        return total_children


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_l1_pillars(pillars_path: Path) -> List[SkeletonNode]:
    """Load L1 pillars from existing pillars.json and convert to skeleton nodes."""
    with open(pillars_path) as f:
        data = json.load(f)

    pillars = data.get("pillars", data)
    nodes = []

    for pillar in pillars:
        node = SkeletonNode(
            id=pillar.get("id", pillar.get("label", "unknown").lower().replace(" ", "-")),
            label=pillar.get("label", pillar.get("id", "Unknown")),
            scope=pillar.get("description", ""),
            level=1,
        )
        nodes.append(node)

    return nodes


def generate_skeleton(
    model_id: str = "gemma-3-4b-it",
    l1_path: Path = None,
    max_depth: int = 4,
    output_path: Path = None,
) -> OntologySkeleton:
    """Generate the full ontology skeleton."""

    # Initialize skeleton
    if l1_path:
        roots = load_l1_pillars(l1_path)
        logger.info(f"Loaded {len(roots)} L1 pillars from {l1_path}")
    else:
        raise ValueError("Must provide L1 pillars path for now")

    skeleton = OntologySkeleton(
        roots=roots,
        generator_model=model_id,
        max_depth=max_depth,
    )

    # Generate children for each level
    generator = SkeletonGenerator(model_id=model_id)

    try:
        for level in range(1, max_depth):
            logger.info(f"\n{'='*60}")
            logger.info(f"Expanding Level {level} -> Level {level + 1}")
            logger.info(f"{'='*60}")

            children_count = generator.expand_level(skeleton, level)
            logger.info(f"Generated {children_count} nodes at level {level + 1}")

            # Save intermediate progress
            if output_path:
                skeleton.save(output_path)

            # Print summary
            counts = skeleton.count_nodes()
            logger.info(f"Current totals: {counts}")

    finally:
        generator.unload()

    return skeleton


def main():
    parser = argparse.ArgumentParser(
        description="Generate ontology skeleton (Phase 1 of two-phase MELD generation)"
    )
    parser.add_argument(
        "--from-l1", "-l",
        type=Path,
        default=Path("results/l1_pillars/pillars.json"),
        help="Path to L1 pillars JSON file"
    )
    parser.add_argument(
        "--model", "-m",
        default="gemma-3-4b-it",
        help="Local model for skeleton generation"
    )
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=4,
        help="Maximum depth to generate (default: 4 = Fields->Universities->Schools->Departments)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results/ontology_skeleton.json"),
        help="Output path for skeleton JSON"
    )
    parser.add_argument(
        "--resume", "-r",
        type=Path,
        help="Resume from existing skeleton file"
    )
    parser.add_argument(
        "--expand-level",
        type=int,
        help="Only expand a specific level (use with --resume)"
    )

    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        skeleton = OntologySkeleton.load(args.resume)

        if args.expand_level:
            generator = SkeletonGenerator(model_id=args.model)
            try:
                children_count = generator.expand_level(skeleton, args.expand_level)
                logger.info(f"Generated {children_count} children at level {args.expand_level + 1}")
            finally:
                generator.unload()

        skeleton.save(args.output)
    else:
        skeleton = generate_skeleton(
            model_id=args.model,
            l1_path=args.from_l1,
            max_depth=args.max_depth,
            output_path=args.output,
        )

    # Print final summary
    counts = skeleton.count_nodes()
    print(f"\n{'='*60}")
    print("ONTOLOGY SKELETON SUMMARY")
    print(f"{'='*60}")
    for level, count in sorted(counts.items()):
        meta = LEVEL_METAPHORS.get(level, {"plural": f"L{level}"})
        print(f"  L{level} ({meta['plural']}): {count}")
    print(f"\nSkeleton saved to: {args.output}")


if __name__ == "__main__":
    main()
