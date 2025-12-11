#!/usr/bin/env python3
"""
Convert generated meld descriptions to MAP_MELD_PROTOCOL format.

Takes the flat generated_melds.json output and converts it to proper
meld request format, split by parent concept for granular review.

Usage:
    python scripts/convert_generated_melds.py \
        --input data/meld_descriptions/generated_melds.json \
        --output-dir melds/pending/1.\ validation/ \
        --target-pack org.hatcat/sumo-wordnet-v4@5.0.0
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any


def slugify(name: str) -> str:
    """Convert concept name to filename-safe slug."""
    return name.lower().replace(" ", "-").replace("_", "-")


def get_domain_for_parent(parent: str, hierarchy_tree: Dict) -> str:
    """Determine domain based on parent location in hierarchy."""
    # Walk the tree to find the parent and determine its domain
    domain_map = {
        "MindsAndAgents": ["Agent", "Cognitive", "Mental", "Psychological", "Social",
                          "Communication", "Intention", "Process", "Action", "Reasoning"],
        "CreatedThings": ["Artifact", "Device", "Tool", "Container", "Machine", "Vehicle",
                         "Furniture", "Weapon", "Instrument", "Building"],
        "PhysicalWorld": ["Physical", "Natural", "Geographic", "Weather", "Material",
                         "Chemical", "Motion", "Force", "Energy"],
        "LivingThings": ["Organism", "Animal", "Plant", "Body", "Biological", "Anatomical"],
        "Information": ["Information", "Content", "Text", "Data", "Document", "Record",
                       "Proposition", "Representation"]
    }

    # Simple heuristic based on parent name
    parent_lower = parent.lower()
    for domain, keywords in domain_map.items():
        for kw in keywords:
            if kw.lower() in parent_lower:
                return domain

    # Default fallback
    return "CreatedThings"


def convert_meld_to_candidate(term: str, meld: Dict, meld_ops: Dict) -> Dict:
    """Convert a single generated meld to MAP candidate schema."""

    # Find matching meld operation for additional context
    meld_op = None
    for op in meld_ops.get("operations", {}).get("create_meld", []):
        if op.get("target") == term:
            meld_op = op
            break

    parent = meld.get("parent", "")
    layer = meld.get("layer", 3)
    children = meld.get("children", [])

    candidate = {
        "term": term,
        "role": "concept",
        "parent_concepts": [parent] if parent else [],
        "layer_hint": layer,
        "definition": meld.get("definition", ""),
        "domain": get_domain_for_parent(parent, {}),

        # Training hints in MAP schema format
        "training_hints": {
            "positive_examples": meld.get("positive_examples", []),
            "negative_examples": meld.get("negative_examples", []),
            "disambiguation": f"Not to be confused with: {', '.join(meld.get('contrast_concepts', []))}"
        },

        # Relationships
        "relationships": {
            "related": meld.get("contrast_concepts", []),
            "has_part": children
        },

        # Default safety tags (standard level for most)
        "safety_tags": {
            "risk_level": "low",
            "impacts": [],
            "treaty_relevant": False,
            "harness_relevant": False
        }
    }

    # Add children reference
    if children:
        candidate["children"] = children

    return candidate


def create_meld_request(
    parent: str,
    candidates: List[Dict],
    target_pack: str,
    version: str = "0.1.0"
) -> Dict:
    """Create a full meld request for candidates under a single parent."""

    slug = slugify(parent) + "-melds"
    meld_id = f"org.hatcat/{slug}@{version}"

    # Determine overall domain from candidates
    domains = [c.get("domain", "CreatedThings") for c in candidates]
    primary_domain = max(set(domains), key=domains.count)

    request = {
        "meld_request_id": meld_id,
        "target_pack_spec_id": target_pack,

        "metadata": {
            "name": f"{parent} Melds",
            "description": f"New intermediate concepts under {parent}",
            "source": "manual",
            "author": "hatcat-meld-generator",
            "created": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator_model": "claude-sonnet-4-20250514"
        },

        # Attachment points - each candidate attaches to the parent
        "attachment_points": [
            {
                "target_concept_id": f"{target_pack}::concept/{parent}",
                "relationship": "parent_of",
                "candidate_concept": c["term"]
            }
            for c in candidates
        ],

        "candidates": candidates,

        "validation": {
            "status": "pending",
            "errors": [],
            "warnings": [],
            "validated_at": None
        }
    }

    return request


def main():
    parser = argparse.ArgumentParser(description='Convert generated melds to MAP format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input generated_melds.json file')
    parser.add_argument('--meld-ops', type=str,
                        default='lens_packs/apertus-8b_first-light-v1/meld_operations.json',
                        help='Meld operations file for additional context')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for meld request files')
    parser.add_argument('--target-pack', type=str,
                        default='org.hatcat/sumo-wordnet-v4@5.0.0',
                        help='Target pack spec ID')
    parser.add_argument('--version', type=str, default='0.1.0',
                        help='Meld request version')
    parser.add_argument('--min-per-file', type=int, default=1,
                        help='Minimum candidates per file (smaller groups get merged)')

    args = parser.parse_args()

    # Load generated melds
    print(f"Loading generated melds from: {args.input}")
    with open(args.input) as f:
        data = json.load(f)

    melds = data.get("melds", {})
    print(f"Found {len(melds)} generated melds")

    # Load meld operations for context
    meld_ops = {}
    if Path(args.meld_ops).exists():
        with open(args.meld_ops) as f:
            meld_ops = json.load(f)
        print(f"Loaded meld operations for context")

    # Group by parent
    by_parent = defaultdict(list)
    for term, meld in melds.items():
        parent = meld.get("parent", "Unknown")
        candidate = convert_meld_to_candidate(term, meld, meld_ops)
        by_parent[parent].append(candidate)

    print(f"\nGrouped into {len(by_parent)} parent categories:")
    for parent, candidates in sorted(by_parent.items(), key=lambda x: -len(x[1])):
        print(f"  {parent}: {len(candidates)} melds")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate meld request files
    print(f"\nWriting meld requests to: {output_dir}")

    files_written = 0
    total_candidates = 0

    # Handle small groups - merge into "misc" if below threshold
    misc_candidates = []
    misc_parents = []

    for parent, candidates in sorted(by_parent.items()):
        if len(candidates) < args.min_per_file:
            misc_candidates.extend(candidates)
            misc_parents.append(parent)
            continue

        request = create_meld_request(parent, candidates, args.target_pack, args.version)

        filename = f"{slugify(parent)}-melds.json"
        output_path = output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(request, f, indent=2)

        files_written += 1
        total_candidates += len(candidates)
        print(f"  {filename}: {len(candidates)} candidates")

    # Write misc file if there are orphans
    if misc_candidates:
        # Group misc by their actual parents for attachment points
        request = {
            "meld_request_id": f"org.hatcat/misc-melds@{args.version}",
            "target_pack_spec_id": args.target_pack,
            "metadata": {
                "name": "Miscellaneous Melds",
                "description": f"Small meld groups from: {', '.join(misc_parents)}",
                "source": "manual",
                "author": "hatcat-meld-generator",
                "created": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            },
            "attachment_points": [
                {
                    "target_concept_id": f"{args.target_pack}::concept/{c['parent_concepts'][0]}",
                    "relationship": "parent_of",
                    "candidate_concept": c["term"]
                }
                for c in misc_candidates if c.get("parent_concepts")
            ],
            "candidates": misc_candidates,
            "validation": {"status": "pending", "errors": [], "warnings": [], "validated_at": None}
        }

        output_path = output_dir / "misc-melds.json"
        with open(output_path, 'w') as f:
            json.dump(request, f, indent=2)

        files_written += 1
        total_candidates += len(misc_candidates)
        print(f"  misc-melds.json: {len(misc_candidates)} candidates (from {len(misc_parents)} small groups)")

    # Summary
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Files written: {files_written}")
    print(f"Total candidates: {total_candidates}")
    print(f"Output directory: {output_dir}")
    print(f"\nNext step: validate with")
    print(f"  python -m src.data.validate_meld '{output_dir}/*.json' -v --summary")


if __name__ == '__main__':
    main()
