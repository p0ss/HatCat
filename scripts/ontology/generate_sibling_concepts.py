#!/usr/bin/env python3
"""
Generate sibling concepts for single-child parents using Anthropic API.

For each parent node that has only one child, this script asks Claude to suggest
5-6 additional sibling concepts that would be appropriate children of that parent.

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    python scripts/ontology/generate_sibling_concepts.py --dry-run  # Preview
    python scripts/ontology/generate_sibling_concepts.py             # Generate
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


def load_keep_both_nodes(filepath: Path) -> List[Dict]:
    """Load the nodes that need additional children."""
    with open(filepath) as f:
        return json.load(f)


def camel_to_readable(name: str) -> str:
    """Convert CamelCase to readable string."""
    words = re.findall(r'[A-Z][a-z]*|[A-Z]+(?=[A-Z]|$)', name)
    return ' '.join(words) if words else name


def generate_siblings_via_api(
    parent: str,
    existing_child: str,
    grandparent: str,
    num_siblings: int = 5
) -> List[Dict]:
    """Generate sibling concepts using Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Install with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Get your API key from https://console.anthropic.com/"
        )

    client = anthropic.Anthropic(api_key=api_key)

    parent_readable = camel_to_readable(parent)
    child_readable = camel_to_readable(existing_child)
    grandparent_readable = camel_to_readable(grandparent)

    prompt = f"""You are helping to expand a SUMO-style ontology hierarchy.

Current structure:
- Grandparent: {grandparent} ({grandparent_readable})
  - Parent: {parent} ({parent_readable})
    - Child: {existing_child} ({child_readable})

The parent "{parent}" currently has only one child "{existing_child}".
We need {num_siblings} additional sibling concepts that would also be appropriate children of "{parent}".

Requirements:
1. Each sibling should be at the same level of specificity as "{existing_child}"
2. Siblings should be distinct from each other (no overlapping concepts)
3. Use CamelCase naming convention (e.g., BufferOverflowAttack, not buffer_overflow_attack)
4. Names should be concise but descriptive (typically 2-4 words in CamelCase)
5. Focus on concepts that are:
   - Well-established and commonly recognized
   - Clearly distinct types/instances of the parent category
   - At a similar level of abstraction to the existing child

Return ONLY a valid JSON array with {num_siblings} objects, each containing:
- "name": CamelCase concept name
- "description": Brief 1-sentence description

Example format:
```json
[
  {{"name": "ExampleConcept", "description": "A brief description of this concept."}},
  {{"name": "AnotherConcept", "description": "A brief description of this concept."}}
]
```

Generate {num_siblings} sibling concepts for "{parent}":"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract JSON from response
        content = response.content[0].text

        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            siblings = json.loads(json_match.group())
            return siblings
        else:
            print(f"  Warning: Could not parse JSON from response for {parent}")
            return []

    except Exception as e:
        print(f"  Error generating siblings for {parent}: {e}")
        return []


def apply_siblings_to_tree(tree: dict, parent: str, siblings: List[Dict]) -> bool:
    """Add sibling concepts to the tree under the parent node."""

    def find_and_add(node: dict, target: str, new_children: List[Dict]) -> bool:
        if not isinstance(node, dict):
            return False

        for k, v in node.items():
            # Strip markers for comparison
            clean_k = k
            for prefix in ['NEW:', 'MOVED:', 'ELEVATED:', 'ABSORBED:', 'RENAMED:']:
                clean_k = clean_k.replace(prefix, '')

            if clean_k == target:
                if isinstance(v, dict):
                    # Add new children
                    for sibling in new_children:
                        name = sibling['name']
                        # Mark as NEW and set depth placeholder (will be fixed later)
                        node[k][f"NEW:{name}"] = 0  # Leaf node
                    return True
                return False

            if isinstance(v, dict):
                if find_and_add(v, target, new_children):
                    return True

        return False

    return find_and_add(tree, parent, siblings)


def fix_depths(node, depth=0):
    """Set leaf values to their actual depth."""
    if not isinstance(node, dict):
        return node

    new_node = {}
    for k, v in node.items():
        if isinstance(v, dict):
            new_node[k] = fix_depths(v, depth + 1)
        elif isinstance(v, int):
            new_node[k] = depth
        else:
            new_node[k] = v
    return new_node


def clean_new_prefix(node):
    """Remove NEW: prefix from keys."""
    if not isinstance(node, dict):
        return node

    new_node = {}
    for k, v in node.items():
        clean_k = k.replace('NEW:', '')
        new_node[clean_k] = clean_new_prefix(v)

    return new_node


def main():
    parser = argparse.ArgumentParser(description='Generate sibling concepts for single-child parents')
    parser.add_argument('--input', default='scripts/ontology/keep_both_nodes.json',
                        help='Input file with nodes needing children')
    parser.add_argument('--tree', default='concept_packs/first-light/hierarchy/hierarchy_tree_v11.json',
                        help='Hierarchy tree to update')
    parser.add_argument('--output', default='concept_packs/first-light/hierarchy/hierarchy_tree_v12.json',
                        help='Output tree file')
    parser.add_argument('--siblings-output', default='scripts/ontology/generated_siblings.json',
                        help='Output file for generated siblings data')
    parser.add_argument('--num-siblings', type=int, default=5,
                        help='Number of siblings to generate per parent')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be generated without calling API')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between API calls in seconds')
    args = parser.parse_args()

    # Load nodes needing children
    nodes = load_keep_both_nodes(Path(args.input))
    print(f"Loaded {len(nodes)} parents needing additional children")

    if args.dry_run:
        print("\n[DRY RUN] Would generate siblings for:")
        for node in nodes:
            print(f"  {node['grandparent']} -> {node['parent']} -> {node['child']}")
            print(f"    Will add {args.num_siblings} siblings to {node['parent']}")
        return

    # Load tree
    with open(args.tree) as f:
        tree = json.load(f)
    print(f"Loaded tree from {args.tree}")

    # Generate siblings for each parent
    all_generated = []
    failed = []

    print(f"\nGenerating {args.num_siblings} siblings for each of {len(nodes)} parents...")

    for node in tqdm(nodes, desc="Generating"):
        parent = node['parent']
        child = node['child']
        grandparent = node['grandparent']

        siblings = generate_siblings_via_api(
            parent=parent,
            existing_child=child,
            grandparent=grandparent,
            num_siblings=args.num_siblings
        )

        if siblings:
            # Apply to tree
            success = apply_siblings_to_tree(tree, parent, siblings)

            all_generated.append({
                'parent': parent,
                'existing_child': child,
                'grandparent': grandparent,
                'generated_siblings': siblings,
                'applied_to_tree': success
            })

            if not success:
                print(f"  Warning: Could not find {parent} in tree")
        else:
            failed.append(node)

        # Rate limiting
        time.sleep(args.delay)

    # Clean up tree
    tree = clean_new_prefix(tree)
    tree = fix_depths(tree)

    # Save results
    with open(args.siblings_output, 'w') as f:
        json.dump(all_generated, f, indent=2)
    print(f"\n✓ Saved generated siblings to {args.siblings_output}")

    with open(args.output, 'w') as f:
        json.dump(tree, f, indent=2)
    print(f"✓ Saved updated tree to {args.output}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Parents processed: {len(nodes)}")
    print(f"  Successful: {len(all_generated)}")
    print(f"  Failed: {len(failed)}")

    total_new = sum(len(g['generated_siblings']) for g in all_generated)
    print(f"  Total new concepts: {total_new}")

    if failed:
        print(f"\n  Failed parents:")
        for f in failed:
            print(f"    - {f['parent']}")


if __name__ == '__main__':
    main()
