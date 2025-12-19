#!/usr/bin/env python3
"""
Taxonomy Expander

Recursively expands a JSON concept tree by generating children for leaf nodes
using an LLM (Claude/Anthropic), ensuring high density (5-15 children per node).

Usage:
    python expand_taxonomy.py \
        --input seed_tree.json \
        --output full_taxonomy.json \
        --max-depth 6 \
        --branch-density "5-15"
"""

import anthropic
import json
import argparse
import time
import re
from pathlib import Path
from typing import Dict, List, Optional

# --- Configuration ---

SYSTEM_PROMPT = """You are an expert academic ontologist and curriculum designer.
Your goal is to flesh out a high-density taxonomy for a specific domain.

## Rules for Expansion
1. **Density**: Generate between 5 and 15 sub-concepts for the given parent.
2. **MECE**: Children should be Mutually Exclusive and Collectively Exhaustive where possible.
3. **Specificity**: Avoid generic placeholders like "General X" or "Introduction to X". Use precise technical terms.
4. **Depth**: You are generating Level {next_level} concepts.
   - Level 2: Major Disciplines (e.g., Theory, Systems)
   - Level 3: Standard University Courses (e.g., Operating Systems, Graph Theory)
   - Level 4: Course Units/Chapters (e.g., Paging, Shortest Path Algs)
   - Level 5: Lessons/Topics (e.g., LRU Replacement, Dijkstra's Algorithm)
   - Level 6: Specific Atomic Concepts (e.g., The Relaxation Step, Belady's Anomaly)

## Output Format
Return ONLY a valid JSON list of objects. Do not include the parent ID prefix, just the slug.
Example:
[
  {"slug": "syntax", "label": "Syntax & Structure"},
  {"slug": "semantics", "label": "Semantics & Valuation"}
]
"""

def slugify(text: str) -> str:
    """Create a clean ID slug from a label."""
    slug = text.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    slug = slug.strip('_')
    return slug

def build_expansion_prompt(node: Dict, ancestors: List[Dict], next_level: int) -> str:
    """Construct the prompt providing context of where we are in the tree."""
    
    path_str = " > ".join([a['label'] for a in ancestors] + [node['label']])
    
    # Contextual hints based on siblings (if we had them, omitted here for brevity)
    
    prompt = f"""
Current Context Path: **{path_str}**
Current Node ID: `{node['id']}`
Current Level: {node.get('level', 1)}

**TASK**: Generate the **Level {next_level}** sub-concepts for "{node['label']}".
Target count: 5-15 distinct children.

Respond with JSON only.
"""
    return prompt

def generate_children(
    client: anthropic.Anthropic, 
    node: Dict, 
    ancestors: List[Dict], 
    model: str
) -> List[Dict]:
    """Call API to generate children for a node."""
    
    next_level = node.get('level', 1) + 1
    prompt = build_expansion_prompt(node, ancestors, next_level)
    
    # Inject level into system prompt
    sys_prompt = SYSTEM_PROMPT.format(next_level=next_level)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text.strip()
        
        # Clean markdown
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        children_data = json.loads(content)
        
        # Hydrate children with IDs and Levels
        processed_children = []
        for child in children_data:
            # Fallback if LLM creates a weird slug
            slug = child.get('slug', slugify(child['label']))
            
            new_child = {
                "id": f"{node['id']}.{slug}",
                "label": child['label'],
                "level": next_level
            }
            processed_children.append(new_child)
            
        return processed_children

    except Exception as e:
        print(f"Error generating children for {node['label']}: {e}")
        return []

def get_node_count(node: Dict) -> int:
    count = 1
    if 'children' in node:
        for child in node['children']:
            count += get_node_count(child)
    return count

def expand_tree_recursive(
    client: anthropic.Anthropic, 
    node: Dict, 
    ancestors: List[Dict], 
    max_depth: int, 
    model: str
) -> bool:
    """
    Recursively expand the tree. 
    Returns True if changes were made (for saving).
    """
    current_level = node.get('level', 1)
    
    # Base case: Max depth reached
    if current_level >= max_depth:
        return False

    changes_made = False

    # Check if we need to generate children for THIS node
    # If it has no children, or very few (and isn't L6), let's expand it.
    # Note: You might want to remove the 'len < 2' check if you want to force re-generation
    # of existing sparse nodes, but usually we just look for empty leaves.
    if 'children' not in node or len(node['children']) == 0:
        print(f"Expanding: {node['id']} ({node['label']}) -> Level {current_level + 1}")
        
        new_children = generate_children(client, node, ancestors, model)
        
        if new_children:
            node['children'] = new_children
            changes_made = True
            time.sleep(1.0) # Rate limit politeness
        else:
            print(f"  No children generated for {node['label']}")

    # Recursive step: process children
    if 'children' in node:
        # Create a copy of ancestors for the children
        new_ancestors = ancestors + [node]
        
        for child in node['children']:
            # Recurse
            if expand_tree_recursive(client, child, new_ancestors, max_depth, model):
                changes_made = True

    return changes_made

def main():
    parser = argparse.ArgumentParser(description='Recursively expand concept taxonomy')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--max-depth', type=int, default=6, help='Target depth (e.g. 6)')
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet-20241022', help='Model ID')
    
    args = parser.parse_args()

    # Load Seed
    try:
        with open(args.input, 'r') as f:
            tree = json.load(f)
        print(f"Loaded tree. Root: {tree['label']}")
    except FileNotFoundError:
        print("Input file not found.")
        return

    client = anthropic.Anthropic()
    
    # We loop until no more expansions occur (Breadth-first-ish via recursion)
    # But since the recursive function does a full pass, calling it once covers the frontier.
    # If you want to fill L3, then L4, then L5, you can run the script multiple times 
    # increasing --max-depth by 1 each time.
    
    print(f"Starting expansion to max_depth={args.max_depth}...")
    
    # The tree root might be a list or an object. Assuming Object based on your seed.
    # If your file is a list of roots, loop over them.
    root_node = tree 
    
    expand_tree_recursive(client, root_node, [], args.max_depth, args.model)
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(tree, f, indent=2)
    
    total_nodes = get_node_count(tree)
    print(f"Done. Saved to {args.output}")
    print(f"Total nodes in tree: {total_nodes}")

if __name__ == '__main__':
    main()