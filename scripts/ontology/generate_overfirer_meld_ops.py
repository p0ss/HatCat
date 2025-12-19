#!/usr/bin/env python3
"""
Generate meld operations for over-firing concepts from calibration analysis.

These are existing concepts that need proper training data (definitions,
positive/negative examples) because they were imported from an older ontology
without full meld descriptions.

The over_fire_on list from calibration analysis tells us what concepts the
lens is incorrectly activating on - these inform negative example generation.

Usage:
    python scripts/ontology/generate_overfirer_meld_ops.py \
        --analysis lens_packs/apertus-8b_first-light_calibration-test/calibration_analysis_cycle3.json \
        --concept-pack concept_packs/first-light \
        --output lens_packs/apertus-8b_first-light_calibration-test/overfirer_meld_operations.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set


def load_concept_hierarchy(concept_pack_dir: Path) -> Dict:
    """Load concept hierarchy to get parent relationships."""
    hierarchy_file = concept_pack_dir / "hierarchy.json"
    if hierarchy_file.exists():
        with open(hierarchy_file) as f:
            return json.load(f)
    return {}


def load_concept_metadata(concept_pack_dir: Path) -> Dict[str, Dict]:
    """Load concept metadata from layer files."""
    concepts = {}
    hierarchy_dir = concept_pack_dir / "hierarchy"

    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        layer = int(layer_file.stem.replace("layer", ""))
        with open(layer_file) as f:
            data = json.load(f)

        for concept in data.get("concepts", []):
            term = concept["sumo_term"]
            concepts[term] = {
                "layer": layer,
                "definition": concept.get("definition", ""),
                "parent_concepts": concept.get("parent_concepts", []),
                "category_children": concept.get("category_children", []),
                "training_hints": concept.get("training_hints", {}),
            }

    return concepts


def main():
    parser = argparse.ArgumentParser(
        description="Generate meld operations for over-firing concepts"
    )
    parser.add_argument(
        "--analysis", type=str, required=True,
        help="Path to calibration analysis JSON"
    )
    parser.add_argument(
        "--concept-pack", type=str, required=True,
        help="Path to concept pack directory"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output meld operations JSON file"
    )
    parser.add_argument(
        "--min-over-fire", type=int, default=50,
        help="Minimum over-fire count to include (default: 50)"
    )
    parser.add_argument(
        "--max-intruders", type=int, default=20,
        help="Max intruder concepts to include per over-firer (default: 20)"
    )

    args = parser.parse_args()

    # Load calibration analysis
    print(f"Loading calibration analysis: {args.analysis}")
    with open(args.analysis) as f:
        analysis = json.load(f)

    over_firing = analysis.get("over_firing", [])
    lens_reports = analysis.get("lens_reports", {})

    print(f"  Over-firing concepts: {len(over_firing)}")

    # Load concept metadata
    concept_pack_dir = Path(args.concept_pack)
    print(f"Loading concept metadata: {concept_pack_dir}")
    concepts = load_concept_metadata(concept_pack_dir)
    hierarchy = load_concept_hierarchy(concept_pack_dir)
    child_to_parent = hierarchy.get("child_to_parent", {})

    print(f"  Loaded {len(concepts)} concepts")

    # Build meld operations
    meld_ops = []
    skipped_low_count = 0
    skipped_not_found = 0

    for over_firer in over_firing:
        report = lens_reports.get(over_firer, {})
        over_fire_count = report.get("over_fire_count", 0)
        over_fire_on = report.get("over_fire_on", [])

        # Skip if below threshold
        if over_fire_count < args.min_over_fire:
            skipped_low_count += 1
            continue

        # Get concept metadata
        if over_firer not in concepts:
            skipped_not_found += 1
            continue

        concept_meta = concepts[over_firer]
        layer = concept_meta["layer"]

        # Find parent from hierarchy
        parent = None
        concept_key = f"{over_firer}:{layer}"
        if concept_key in child_to_parent:
            parent_key = child_to_parent[concept_key]
            parent = parent_key.rsplit(":", 1)[0]
        elif concept_meta["parent_concepts"]:
            parent = concept_meta["parent_concepts"][0]

        # Sample intruders for negative example generation
        # Prioritize diverse intruders (from different parts of hierarchy)
        sampled_intruders = over_fire_on[:args.max_intruders]

        meld_op = {
            "target": over_firer,
            "target_layer": layer,
            "new_parent": parent or "Unknown",
            "source_concepts": concept_meta.get("category_children", []),
            "existing_definition": concept_meta.get("definition", ""),
            "existing_training_hints": concept_meta.get("training_hints", {}),
            # Extra context for over-firer meld generation
            "over_fire_count": over_fire_count,
            "over_fire_on_sample": sampled_intruders,
            "operation_type": "enhance_existing",  # Not creating new, enhancing existing
        }

        meld_ops.append(meld_op)

    # Sort by over-fire severity
    meld_ops.sort(key=lambda x: x["over_fire_count"], reverse=True)

    # Create output
    output = {
        "version": "1.0",
        "source_analysis": args.analysis,
        "concept_pack": str(concept_pack_dir),
        "description": "Meld operations for over-firing concepts needing enhanced training data",
        "operations": {
            "create_meld": meld_ops  # Using same key for compatibility with generate_meld_descriptions.py
        },
        "summary": {
            "total_over_firers": len(over_firing),
            "included": len(meld_ops),
            "skipped_low_count": skipped_low_count,
            "skipped_not_found": skipped_not_found,
        }
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nMeld operations written to: {output_path}")
    print(f"  Included: {len(meld_ops)}")
    print(f"  Skipped (low count): {skipped_low_count}")
    print(f"  Skipped (not found): {skipped_not_found}")

    # Show top offenders
    print(f"\nTop 10 over-firers:")
    for op in meld_ops[:10]:
        print(f"  {op['over_fire_count']:5d} - {op['target']} (layer {op['target_layer']})")


if __name__ == "__main__":
    main()
