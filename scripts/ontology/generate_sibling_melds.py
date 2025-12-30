#!/usr/bin/env python3
"""
Generate meld request from generated_siblings.json.

Creates a proper meld request with attachment points and candidate concepts
following the MAP_MELDING.md specification.

Usage:
    python scripts/ontology/generate_sibling_melds.py --dry-run  # Preview
    python scripts/ontology/generate_sibling_melds.py            # Generate
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


def load_generated_siblings(filepath: Path) -> List[Dict]:
    """Load the generated siblings data."""
    with open(filepath) as f:
        return json.load(f)


def load_layer_files(hierarchy_dir: Path) -> Dict[str, Dict]:
    """Load all layer files to get existing concept metadata."""
    concepts = {}
    for i in range(7):  # layers 0-6
        layer_file = hierarchy_dir / f"layer{i}.json"
        if layer_file.exists():
            with open(layer_file) as f:
                layer_data = json.load(f)
                # Handle both formats: list or dict with 'concepts' key
                if isinstance(layer_data, dict) and 'concepts' in layer_data:
                    concept_list = layer_data['concepts']
                elif isinstance(layer_data, list):
                    concept_list = layer_data
                else:
                    continue
                for concept in concept_list:
                    # Use sumo_term or term as key
                    term = concept.get('sumo_term') or concept.get('term')
                    if term:
                        concepts[term] = concept
    return concepts


def camel_to_readable(name: str) -> str:
    """Convert CamelCase to readable string."""
    words = re.findall(r'[A-Z][a-z]*|[A-Z]+(?=[A-Z]|$)', name)
    return ' '.join(words).lower() if words else name.lower()


def infer_domain(parent: str, existing_concepts: Dict[str, Dict]) -> str:
    """Infer domain from parent concept."""
    if parent in existing_concepts:
        return existing_concepts[parent].get('domain', 'Entity')
    # Default mappings based on common patterns
    domain_hints = {
        'Attack': 'Information',
        'Process': 'MindsAndAgents',
        'Substance': 'PhysicalWorld',
        'Vehicle': 'CreatedThings',
        'Building': 'CreatedThings',
        'Weapon': 'CreatedThings',
        'Machine': 'CreatedThings',
        'Animal': 'LivingThings',
        'Game': 'MindsAndAgents',
        'Model': 'Information',
        'Network': 'Information',
        'Operation': 'MindsAndAgents',
        'War': 'MindsAndAgents',
    }
    for hint, domain in domain_hints.items():
        if hint in parent:
            return domain
    return 'Entity'


def get_parent_layer(parent: str, existing_concepts: Dict[str, Dict]) -> int:
    """Get layer of parent concept."""
    if parent in existing_concepts:
        return existing_concepts[parent].get('layer', 3)
    return 3  # Default assumption


def generate_training_hints(name: str, description: str, parent: str, existing_child: str) -> Dict:
    """Generate training hints for a new concept."""
    readable_name = camel_to_readable(name)
    readable_parent = camel_to_readable(parent)
    readable_sibling = camel_to_readable(existing_child)

    # Generate positive examples based on the description
    positive_examples = [
        f"This is an example of {readable_name}.",
        f"The {readable_name} demonstrated clear characteristics.",
        f"We observed {readable_name} in this context.",
        f"A typical case of {readable_name} would involve these features.",
        f"The concept of {readable_name} applies here.",
        f"{description}"
    ]

    # Generate negative examples using sibling and parent
    negative_examples = [
        f"This is an example of {readable_sibling}, not {readable_name}.",
        f"This represents {readable_parent} in general.",
        f"The weather is pleasant today.",
        f"Water boils at 100 degrees Celsius.",
        f"The document was filed in the cabinet.",
        f"She walked to the store to buy groceries."
    ]

    return {
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "disambiguation": f"Not to be confused with: {existing_child}, or other types of {parent}"
    }


def generate_meld_request(
    siblings_data: List[Dict],
    existing_concepts: Dict[str, Dict],
    pack_spec_id: str = "org.hatcat/first-light@1.0.0"
) -> Dict:
    """Generate a complete meld request from siblings data."""

    timestamp = datetime.now(timezone.utc).isoformat()

    meld_request = {
        "meld_request_id": f"org.hatcat/sibling-expansion@0.1.0",
        "target_pack_spec_id": pack_spec_id,
        "metadata": {
            "name": "Sibling Concept Expansion",
            "description": "Additional sibling concepts generated for single-child parents",
            "source": "manual",
            "author": "hatcat-meld-generator",
            "created": timestamp,
            "generator_model": "claude-sonnet-4-20250514"
        },
        "attachment_points": [],
        "candidates": [],
        "validation": {
            "status": "pending",
            "errors": [],
            "warnings": []
        }
    }

    # Process each parent's siblings
    for entry in siblings_data:
        parent = entry['parent']
        existing_child = entry['existing_child']
        grandparent = entry['grandparent']
        siblings = entry['generated_siblings']

        parent_layer = get_parent_layer(parent, existing_concepts)
        child_layer = parent_layer + 1
        domain = infer_domain(parent, existing_concepts)

        for sibling in siblings:
            name = sibling['name']
            description = sibling['description']

            # Add attachment point
            meld_request["attachment_points"].append({
                "target_concept_id": f"{pack_spec_id}::concept/{parent}",
                "relationship": "parent_of",
                "candidate_concept": name
            })

            # Add candidate concept
            candidate = {
                "term": name,
                "role": "concept",
                "parent_concepts": [parent],
                "layer_hint": child_layer,
                "definition": description,
                "domain": domain,
                "training_hints": generate_training_hints(
                    name, description, parent, existing_child
                ),
                "relationships": {
                    "related": [existing_child],
                    "antonyms": [],
                    "part_of": [],
                    "has_part": []
                },
                "safety_tags": {
                    "risk_level": "low",
                    "impacts": [],
                    "treaty_relevant": False,
                    "harness_relevant": False
                },
                "children": []
            }

            meld_request["candidates"].append(candidate)

    return meld_request


def main():
    parser = argparse.ArgumentParser(description='Generate meld request from sibling concepts')
    parser.add_argument('--input', default='scripts/ontology/generated_siblings.json',
                        help='Input file with generated siblings')
    parser.add_argument('--hierarchy-dir', default='concept_packs/first-light/hierarchy',
                        help='Directory with layer files')
    parser.add_argument('--output', default='melds/pending/sibling-expansion.json',
                        help='Output meld request file')
    parser.add_argument('--pack-spec-id', default='org.hatcat/first-light@1.0.0',
                        help='Target pack spec ID')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be generated without writing')
    args = parser.parse_args()

    # Load data
    print(f"Loading generated siblings from {args.input}...")
    siblings_data = load_generated_siblings(Path(args.input))
    print(f"  Found {len(siblings_data)} parents with siblings")

    total_siblings = sum(len(s['generated_siblings']) for s in siblings_data)
    print(f"  Total new concepts: {total_siblings}")

    print(f"\nLoading existing concepts from {args.hierarchy_dir}...")
    existing_concepts = load_layer_files(Path(args.hierarchy_dir))
    print(f"  Found {len(existing_concepts)} existing concepts")

    # Generate meld request
    print("\nGenerating meld request...")
    meld_request = generate_meld_request(
        siblings_data,
        existing_concepts,
        pack_spec_id=args.pack_spec_id
    )

    print(f"\nMeld request summary:")
    print(f"  Attachment points: {len(meld_request['attachment_points'])}")
    print(f"  Candidates: {len(meld_request['candidates'])}")

    # Show sample
    if meld_request['candidates']:
        print(f"\nSample candidate:")
        sample = meld_request['candidates'][0]
        print(f"  Term: {sample['term']}")
        print(f"  Parent: {sample['parent_concepts']}")
        print(f"  Layer: {sample['layer_hint']}")
        print(f"  Domain: {sample['domain']}")
        print(f"  Definition: {sample['definition'][:80]}...")

    if args.dry_run:
        print(f"\n[DRY RUN] Would write to: {args.output}")
    else:
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(meld_request, f, indent=2)
        print(f"\n✓ Wrote meld request to {args.output}")


if __name__ == '__main__':
    main()
