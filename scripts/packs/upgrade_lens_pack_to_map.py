#!/usr/bin/env python3
"""
Upgrade a lens pack to MAP (MINDMELD Architectural Protocol) compliance.

This script:
1. Builds lens_index from classifier files in hierarchy/
2. Adds MAP fields to lens_pack.json:
   - lens_pack_id
   - concept_pack_spec_id
   - lens_output_schema
   - lens_index

Usage:
    python scripts/upgrade_lens_pack_to_map.py \\
        --lens-pack lens_packs/gemma-3-4b-pt_sumo-wordnet-v3 \\
        --concept-pack concept_packs/sumo-wordnet-v4 \\
        --model-name google/gemma-3-4b-pt

"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def load_concept_pack_metadata(concept_pack_dir):
    """Load concept pack metadata to get spec_id and layer information."""
    pack_json_path = concept_pack_dir / "pack.json"

    if not pack_json_path.exists():
        raise ValueError(f"Concept pack not found at {concept_pack_dir}")

    with open(pack_json_path) as f:
        pack_metadata = json.load(f)

    # Extract spec_id (or build it if missing)
    spec_id = pack_metadata.get("spec_id")
    if not spec_id:
        # Build spec_id from pack_id and version
        pack_id = pack_metadata["pack_id"]
        version = pack_metadata["version"]
        spec_id = f"org.hatcat/{pack_id}@{version}"
        print(f"WARNING: spec_id not found in concept pack, using: {spec_id}")

    return pack_metadata, spec_id


def load_layer_metadata(concept_pack_dir):
    """Load layer metadata from hierarchy/ to get concept info."""
    hierarchy_dir = concept_pack_dir / "hierarchy"

    if not hierarchy_dir.exists():
        raise ValueError(f"Hierarchy directory not found: {hierarchy_dir}")

    concept_map = {}  # Map concept name to layer info

    # Load all layer files
    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        with open(layer_file) as f:
            layer_data = json.load(f)

        layer_num = layer_data.get("layer", 0)

        for concept in layer_data.get("concepts", []):
            sumo_term = concept.get("sumo_term")
            if sumo_term:
                concept_map[sumo_term] = {
                    "layer": layer_num,
                    "domain": concept.get("domain", "Unknown"),
                    "definition": concept.get("definition", "")
                }

    return concept_map


def build_lens_index(lens_pack_dir, concept_map, lens_pack_id, concept_pack_spec_id):
    """Build lens_index by scanning hierarchy/ for classifier files."""
    hierarchy_dir = lens_pack_dir / "hierarchy"

    if not hierarchy_dir.exists():
        raise ValueError(f"Hierarchy directory not found in lens pack: {hierarchy_dir}")

    lens_index = {}
    classifier_files = list(hierarchy_dir.glob("*_classifier.pt"))

    print(f"Found {len(classifier_files)} classifier files in {hierarchy_dir}")

    for classifier_file in classifier_files:
        # Extract concept name from filename (remove _classifier.pt suffix)
        concept_name = classifier_file.stem.replace("_classifier", "")

        # Get layer info from concept_map
        concept_info = concept_map.get(concept_name, {})
        layer = concept_info.get("layer", 0)

        # Build lens_id and concept_id
        lens_id = f"{lens_pack_id}::lens/{concept_name}"
        concept_id = f"{concept_pack_spec_id}::concept/{concept_name}"

        # Relative path from lens pack root
        relative_file = f"hierarchy/{classifier_file.name}"

        lens_index[concept_name] = {
            "lens_id": lens_id,
            "concept_id": concept_id,
            "layer": layer,
            "file": relative_file,
            "output_schema": {
                "$ref": "#/lens_output_schema"
            }
        }

    return lens_index


def upgrade_lens_pack(lens_pack_dir, concept_pack_dir, model_name):
    """Upgrade lens pack to MAP compliance."""

    print("=" * 80)
    print("UPGRADING LENS PACK TO MAP COMPLIANCE")
    print("=" * 80)
    print()

    # Load concept pack metadata
    print(f"Loading concept pack metadata from {concept_pack_dir}...")
    concept_pack_metadata, concept_pack_spec_id = load_concept_pack_metadata(concept_pack_dir)
    print(f"  Concept Pack Spec ID: {concept_pack_spec_id}")
    print()

    # Load layer metadata
    print("Loading layer metadata...")
    concept_map = load_layer_metadata(concept_pack_dir)
    print(f"  Loaded {len(concept_map)} concepts from layer files")
    print()

    # Load existing lens_pack.json
    lens_pack_json_path = lens_pack_dir / "lens_pack.json"

    if not lens_pack_json_path.exists():
        raise ValueError(f"lens_pack.json not found at {lens_pack_json_path}")

    print(f"Loading existing lens_pack.json from {lens_pack_json_path}...")
    with open(lens_pack_json_path) as f:
        lens_pack_data = json.load(f)
    print()

    # Extract model short name for lens_pack_id
    model_short = model_name.replace("google/", "").replace("swiss-ai/", "")
    pack_version = lens_pack_data.get("version", "1.0.0")

    # Build lens_pack_id
    concept_pack_name = concept_pack_metadata["pack_id"]
    concept_pack_version = concept_pack_metadata["version"]
    lens_pack_id = f"org.hatcat/{model_short}__{concept_pack_name}@{concept_pack_version}__v{pack_version.split('.')[0]}"

    print(f"Building lens pack ID: {lens_pack_id}")
    print()

    # Build lens_index
    print("Building lens_index from classifier files...")
    lens_index = build_lens_index(lens_pack_dir, concept_map, lens_pack_id, concept_pack_spec_id)
    print(f"  Built index for {len(lens_index)} lenses")
    print()

    # Add MAP fields to lens_pack_data
    print("Adding MAP fields to lens_pack.json...")

    # Set lens_pack_id
    lens_pack_data["lens_pack_id"] = lens_pack_id

    # Set concept_pack_spec_id
    lens_pack_data["concept_pack_spec_id"] = concept_pack_spec_id

    # Add lens_output_schema
    lens_pack_data["lens_output_schema"] = {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "description": "Concept activation score"
            },
            "null_pole": {
                "type": "number",
                "description": "Null/absence pole score"
            },
            "entropy": {
                "type": "number",
                "description": "Prediction entropy"
            }
        },
        "required": ["score"]
    }

    # Add lens_index
    lens_pack_data["lens_index"] = lens_index

    # Update timestamp
    lens_pack_data["updated"] = datetime.now().isoformat()

    # Write updated lens_pack.json
    print(f"Writing upgraded lens_pack.json to {lens_pack_json_path}...")
    with open(lens_pack_json_path, 'w') as f:
        json.dump(lens_pack_data, f, indent=2)

    print()
    print("=" * 80)
    print("UPGRADE COMPLETE")
    print("=" * 80)
    print()
    print(f"Lens Pack ID: {lens_pack_id}")
    print(f"Concept Pack Spec ID: {concept_pack_spec_id}")
    print(f"Total Lenses in Index: {len(lens_index)}")
    print()
    print("The lens pack is now MAP-compliant!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Upgrade lens pack to MAP compliance"
    )

    parser.add_argument('--lens-pack', required=True,
                        help='Path to lens pack directory (e.g., lens_packs/gemma-3-4b-pt_sumo-wordnet-v3)')
    parser.add_argument('--concept-pack', required=True,
                        help='Path to concept pack directory (e.g., concept_packs/sumo-wordnet-v4)')
    parser.add_argument('--model-name', required=True,
                        help='Full model name (e.g., google/gemma-3-4b-pt)')

    args = parser.parse_args()

    # Convert to Path objects
    lens_pack_dir = PROJECT_ROOT / args.lens_pack
    concept_pack_dir = PROJECT_ROOT / args.concept_pack

    # Validate paths
    if not lens_pack_dir.exists():
        print(f"Error: Lens pack directory not found: {lens_pack_dir}")
        sys.exit(1)

    if not concept_pack_dir.exists():
        print(f"Error: Concept pack directory not found: {concept_pack_dir}")
        sys.exit(1)

    # Run upgrade
    try:
        upgrade_lens_pack(lens_pack_dir, concept_pack_dir, args.model_name)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
