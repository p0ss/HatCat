#!/usr/bin/env python3
"""
University Builder: The "Chain of Command" Taxonomy Generator

This script builds a dense concept hierarchy by simulating a university structure.
It moves recursively from the Provost (L2) down to the Teaching Assistants (L6),
passing context and constraints down the chain to prevent overlap.

Levels & Personas:
L1 -> L2 (Schools):     University Provost
L2 -> L3 (Courses):     School Dean
L3 -> L4 (Units):       Professor
L4 -> L5 (Lessons):     Senior Lecturer
L5 -> L6 (Concepts):    Assistant Lecturer / TA

Usage:
    python university_builder.py --subject "Computer Science" --output cs_university.json
"""

import anthropic
import json
import argparse
import time
import re
import os
from typing import Dict, List, Any, Tuple, Optional

# --- Personas & Prompts ---

PERSONAS = {
    1: {
        "role": "University Provost",
        "task": "Establish the 5-15 major Schools (L2) within this field.",
        "context": "These are distinct administrative and research divisions (e.g., 'School of Theory', 'School of Systems'). They will each have a Dean.",
        "constraint": "Ensure Schools are mutually exclusive coverage of the entire field."
    },
    2: {
        "role": "Dean of the School",
        "task": "Design the 5-15 core Courses (L3) offered by your School.",
        "context": "These are standard university catalog courses (e.g., 'Compilers 101', 'Advanced Graph Algorithms'). Each will be run by a Professor.",
        "constraint": "Do not overlap with courses in other Schools. Focus strictly on your School's domain."
    },
    3: {
        "role": "Professor",
        "task": "Design the 5-15 Modules/Units (L4) that make up your Course.",
        "context": "These are the thematic blocks of the semester (e.g., 'Syntax Analysis', 'Optimization phases').",
        "constraint": "Stay within the scope of this specific course. Do not cover prerequisites or downstream advanced topics."
    },
    4: {
        "role": "Senior Lecturer",
        "task": "Plan the 5-15 specific Lessons (L5) for this Module.",
        "context": "These are individual lecture topics (e.g., 'LR(1) Parsing Tables', 'Loop Invariant Code Motion').",
        "constraint": "Concrete, teachable topics. Not general headings."
    },
    5: {
        "role": "Assistant Lecturer / TA",
        "task": "List the 5-15 atomic Concepts (L6) covered in this Lesson.",
        "context": "These are the specific definitions, theorems, or mechanisms students must memorize (e.g., 'The closure property', 'Shift-Reduce conflict').",
        "constraint": "Atomic, precise, testable concepts. No fluff."
    }
}

SYSTEM_PROMPT = """You are an expert academic {role}. Stay disciplined within your remit.

University Field: {root_subject}
Curricular Path: {path}

Lane Boundaries:
- Parent Scope: {parent_scope}
- Sibling Topics (avoid overlap): {siblings}
- Adjacent Domains (stay out): {adjacent_domains}

Non-Negotiable Rules:
1. **Lane Discipline** – Cover only content that legitimately belongs under this node. Never jump sideways into siblings or upstream/downstream domains.
2. **MECE Coverage** – Children must be Mutually Exclusive and Collectively Exhaustive, fully covering the parent scope without redundancy.
3. **Specificity** – No generic placeholders ("General", "Introduction", "Overview"). Use precise, concrete technical terms in CamelCase or PascalCase.
4. **Count & Structure** – Produce between {min_children} and {max_children} children. Each child must be a JSON object with `slug` and `label`.
5. **No Cross-School Appeals** – Do not describe “how this applies to other Schools/Courses”. Stay strictly in-lane.

Role Goal: {task}
Additional Context: {context}
Constraint: {constraint}
"""

USER_PROMPT = """Provide the list of children for this node as a JSON array.
Each item must be `{{"slug": "unique-slug", "label": "CamelCaseTitle"}}`.
No commentary, no Markdown fences, JSON only.
"""

PLACEHOLDER_TERMS = {"general", "introduction", "overview", "basics", "misc", "other", "foundations", "fundamentals"}
MIN_CHILDREN = 5
MAX_CHILDREN = 15

class UniversityBuilder:
    def __init__(self, subject: str, output_file: str, model: str):
        self.client = anthropic.Anthropic()
        self.subject = subject
        self.output_file = output_file
        self.model = model
        self.tree = self._load_or_init_tree()

    def _load_or_init_tree(self) -> Dict:
        if os.path.exists(self.output_file):
            print(f"Resuming from {self.output_file}...")
            with open(self.output_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Starting fresh university for: {self.subject}")
            return {
                "id": "root",
                "label": self.subject,
                "level": 1,
                "children": []
            }

    def save(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.tree, f, indent=2)

    def slugify(self, text: str) -> str:
        slug = re.sub(r'[^a-z0-9]+', '_', text.lower().strip()).strip('_')
        return slug or 'item'

    def camelize(self, text: str) -> str:
        tokens = re.split(r'[^A-Za-z0-9]+', text)
        tokens = [t for t in tokens if t]
        if not tokens:
            return ""
        normalized = []
        for token in tokens:
            if token.isupper():
                normalized.append(token)
            else:
                normalized.append(token[:1].upper() + token[1:])
        return "".join(normalized)

    def summarize_adjacent_domains(self, ancestors: List[Dict]) -> str:
        if not ancestors:
            return "(None defined yet)"

        collected = []
        parent = ancestors[-1]

        # Include other branches under the same grandparent (parent's siblings)
        if len(ancestors) >= 2:
            grandparent = ancestors[-2]
            for branch in grandparent.get('children', []):
                if branch['label'] != parent['label']:
                    collected.append(branch['label'])
        else:
            # parent is the root; include other root-level schools
            for branch in self.tree.get('children', []):
                if branch['label'] != parent['label']:
                    collected.append(branch['label'])

        if not collected:
            return "(None defined yet)"
        return ", ".join(sorted(set(collected))[:20])

    def build_prompt(self, node: Dict, ancestors: List[Dict]) -> Tuple[str, str]:
        level = node['level']
        persona = PERSONAS[level]

        # Construct the path string (e.g. Computer Science > Systems > OS)
        path_names = [a['label'] for a in ancestors] + [node['label']]
        path_str = " > ".join(path_names)

        # Get siblings (children of the parent) to avoid overlap
        # If I am the root, I have no siblings.
        siblings_list = []
        if ancestors:
            parent = ancestors[-1]
            # Exclude self from siblings
            siblings_list = [c['label'] for c in parent.get('children', []) if c['label'] != node['label']]

        siblings_str = ", ".join(siblings_list[:20])  # Limit to 20 to save tokens
        if not siblings_str:
            siblings_str = "(None - you are the first/only entry so far)"

        parent_scope = ancestors[-1]['label'] if ancestors else node['label']
        adjacent_domains = self.summarize_adjacent_domains(ancestors)

        system_prompt = SYSTEM_PROMPT.format(
            role=persona['role'],
            root_subject=self.subject,
            path=path_str,
            task=persona['task'],
            context=persona['context'],
            constraint=persona['constraint'],
            siblings=siblings_str,
            parent_scope=parent_scope,
            adjacent_domains=adjacent_domains,
            min_children=MIN_CHILDREN,
            max_children=MAX_CHILDREN
        )

        return system_prompt, USER_PROMPT

    def extract_json(self, content: str) -> str:
        if "```" in content:
            parts = content.split("```")
            # Prefer the first JSON block after ```
            for part in parts:
                stripped = part.strip()
                if stripped.lower().startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith('[') or stripped.startswith('{'):
                    return stripped
        return content

    def validate_children(self, data: Any) -> Optional[List[Dict[str, str]]]:
        if not isinstance(data, list):
            print("  !!! Model response was not a list")
            return None

        if not (MIN_CHILDREN <= len(data) <= MAX_CHILDREN):
            print(f"  !!! Expected {MIN_CHILDREN}-{MAX_CHILDREN} items, got {len(data)}")
            return None

        cleaned = []
        seen_labels = set()
        seen_slugs = set()

        for idx, child in enumerate(data, start=1):
            if not isinstance(child, dict):
                print(f"  !!! Child #{idx} is not an object")
                return None

            raw_label = (child.get('label') or '').strip()
            raw_slug = (child.get('slug') or '').strip()

            if not raw_label:
                print(f"  !!! Child #{idx} missing label")
                return None

            lower_label = raw_label.lower()
            tokens = [tok for tok in re.split(r'[^a-z0-9]+', lower_label) if tok]
            if any(term in tokens for term in PLACEHOLDER_TERMS):
                print(f"  !!! Child '{raw_label}' contains placeholder wording")
                return None

            label = self.camelize(raw_label)
            if not label:
                print(f"  !!! Child #{idx} label invalid after camelizing")
                return None

            if label in seen_labels:
                print(f"  !!! Duplicate label '{label}' in response")
                return None
            seen_labels.add(label)

            slug = raw_slug or self.slugify(label)
            while slug in seen_slugs:
                slug = f"{slug}_{len(seen_slugs)}"
            seen_slugs.add(slug)

            cleaned.append({"label": label, "slug": slug})

        return cleaned

    def expand_node(self, node: Dict, ancestors: List[Dict]):
        current_level = node.get('level', 1)

        # Stop if we are at the concept leaves (Level 6)
        if current_level >= 6:
            return

        # Check if node needs expansion (has no children)
        if 'children' not in node or not node['children']:
            print(f"\n[L{current_level}->L{current_level+1}] {node['label']} ({PERSONAS[current_level]['role']})")
            system_prompt, user_prompt = self.build_prompt(node, ancestors)

            attempts = 0
            children_payload: Optional[List[Dict[str, str]]] = None
            while attempts < 3 and children_payload is None:
                attempts += 1
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=2048,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )

                    content = response.content[0].text.strip()
                    content = self.extract_json(content)
                    children_data = json.loads(content)
                    children_payload = self.validate_children(children_data)
                    if children_payload is None:
                        print(f"  ...retrying ({attempts}/3)")
                        time.sleep(1.0)

                except Exception as e:
                    print(f"  !!! Error expanding {node['label']} (attempt {attempts}): {e}")
                    time.sleep(1.0)

            if not children_payload:
                print(f"  !!! Giving up on {node['label']} after 3 attempts")
                return

            new_children = []
            for child in children_payload:
                new_child = {
                    "id": f"{node['id']}.{child['slug']}",
                    "label": child['label'],
                    "level": current_level + 1
                }
                new_children.append(new_child)

            node['children'] = new_children
            self.save()  # Checkpoint

            print(f"  -> Generated {len(new_children)} items.")
            time.sleep(0.5)  # Rate limit politeness

        # Recursion: Drill down
        # We process depth-first so we finish a School before starting the next one
        if 'children' in node:
            next_ancestors = ancestors + [node]
            for child in node['children']:
                self.expand_node(child, next_ancestors)

    def run(self):
        # Start the recursion from the root
        self.expand_node(self.tree, [])
        print("\nUniversity Curriculum generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a University Curriculum')
    parser.add_argument('--subject', type=str, required=True, help='Root subject (e.g. "Computer Science")')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet-20241022', help='Anthropic model')
    
    args = parser.parse_args()
    
    builder = UniversityBuilder(args.subject, args.output, args.model)
    builder.run()
