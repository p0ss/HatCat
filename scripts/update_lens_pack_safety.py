#!/usr/bin/env python3
"""
Update Lens Pack Safety Metadata

Reads safety concepts from the concept pack and writes them to the
lens pack's pack.json. This bakes safety metadata into the deployed
lens pack so the server can read it directly.

Run this after:
1. Updating concept pack with safety_tags (propagate_safety_to_concepts.py)
2. Before deploying a lens pack
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_concept_safety_data(concept_pack_dir: Path) -> dict:
    """Load safety metadata from concept pack."""
    concepts_dir = concept_pack_dir / "concepts"

    if not concepts_dir.exists():
        raise FileNotFoundError(f"Concepts directory not found: {concepts_dir}")

    safety_data = {
        'high_risk': [],
        'critical_risk': [],
        'treaty_relevant': [],
        'harness_relevant': [],
        'by_simplex': defaultdict(list),
    }

    for layer_dir in concepts_dir.iterdir():
        if not layer_dir.is_dir():
            continue

        for concept_file in layer_dir.glob("*.json"):
            try:
                with open(concept_file) as f:
                    concept = json.load(f)

                term = concept.get('term', concept_file.stem)
                safety_tags = concept.get('safety_tags', {})
                risk_level = safety_tags.get('risk_level', 'low')

                if risk_level == 'critical':
                    safety_data['critical_risk'].append(term)
                elif risk_level == 'high':
                    safety_data['high_risk'].append(term)

                if safety_tags.get('treaty_relevant'):
                    safety_data['treaty_relevant'].append(term)
                if safety_tags.get('harness_relevant'):
                    safety_data['harness_relevant'].append(term)

                # Track simplex mappings
                simplex = concept.get('simplex_mapping', {})
                if simplex.get('status') == 'mapped' and simplex.get('monitor'):
                    safety_data['by_simplex'][simplex['monitor']].append(term)

            except Exception:
                continue

    # Convert defaultdict to regular dict
    safety_data['by_simplex'] = dict(safety_data['by_simplex'])

    return safety_data


def update_lens_pack(lens_pack_dir: Path, safety_data: dict, dry_run: bool = False):
    """Update lens pack's pack.json with safety metadata."""
    pack_json = lens_pack_dir / "pack.json"

    if not pack_json.exists():
        raise FileNotFoundError(f"Lens pack.json not found: {pack_json}")

    with open(pack_json) as f:
        pack = json.load(f)

    # Add/update safety_concepts section
    pack['safety_concepts'] = {
        'version': '1.0.0',
        'description': 'Safety-critical concepts for UI highlighting and monitoring',
        'critical_risk': sorted(safety_data['critical_risk']),
        'high_risk': sorted(safety_data['high_risk']),
        'treaty_relevant': sorted(safety_data['treaty_relevant']),
        'harness_relevant': sorted(safety_data['harness_relevant']),
        'simplex_bindings': safety_data['by_simplex'],
        'highlight_concepts': sorted(
            safety_data['critical_risk'] + safety_data['high_risk']
        ),
    }

    if dry_run:
        print(f"[DRY RUN] Would update {pack_json}")
        print(f"  critical_risk: {len(safety_data['critical_risk'])}")
        print(f"  high_risk: {len(safety_data['high_risk'])}")
        print(f"  treaty_relevant: {len(safety_data['treaty_relevant'])}")
        print(f"  harness_relevant: {len(safety_data['harness_relevant'])}")
        print(f"  simplex bindings: {len(safety_data['by_simplex'])}")
        return

    with open(pack_json, 'w') as f:
        json.dump(pack, f, indent=2)

    print(f"✓ Updated {pack_json}")
    print(f"  critical_risk: {len(safety_data['critical_risk'])}")
    print(f"  high_risk: {len(safety_data['high_risk'])}")
    print(f"  highlight_concepts: {len(pack['safety_concepts']['highlight_concepts'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Update lens pack with safety metadata from concept pack"
    )
    parser.add_argument(
        "--concept-pack", "-c",
        default="concept_packs/first-light",
        help="Path to concept pack"
    )
    parser.add_argument(
        "--lens-pack", "-l",
        required=True,
        help="Path to lens pack to update"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be changed"
    )

    args = parser.parse_args()

    concept_pack_dir = Path(args.concept_pack)
    lens_pack_dir = Path(args.lens_pack)

    print(f"Loading safety data from: {concept_pack_dir}")
    safety_data = load_concept_safety_data(concept_pack_dir)

    print(f"\nUpdating lens pack: {lens_pack_dir}")
    update_lens_pack(lens_pack_dir, safety_data, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
