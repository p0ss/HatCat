#!/usr/bin/env python3
"""
Convert expanded pillars into concept pack format for lens training.

Takes the output of expand_pillars_gemma.py and creates:
1. pack.json with proper metadata
2. hierarchy/layer0.json with L1 pillars
3. hierarchy/layer1.json with L2 children
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def convert_pillar_to_concept(pillar: dict, layer: int = 0) -> dict:
    """Convert a pillar dict to the concept format expected by the training pipeline."""
    # Use title case for sumo_term, keeping it simple
    sumo_term = pillar["id"].replace("_", " ").title().replace(" ", "")

    concept = {
        "sumo_term": sumo_term,
        "layer": layer,
        "domain": "ActionAgency",  # All pillars share this domain
        "is_category_lens": True,
        "is_pseudo_sumo": True,  # Not from SUMO ontology
        "definition": pillar["description"],
        "parent_concepts": pillar.get("parent_concepts", []),
        "child_concepts": [],
        "category_children": [],
        # Training examples from the pillar expansion
        "positive_examples": pillar.get("training_examples", {}).get("positive", []),
        "negative_examples": pillar.get("training_examples", {}).get("negative", []),
    }

    # Add L2 children references if present
    if "l2_children" in pillar:
        for child in pillar["l2_children"]:
            child_term = child["id"].replace("-", "_").replace("_", " ").title().replace(" ", "")
            concept["category_children"].append(child_term)
            concept["child_concepts"].append(child_term)

    return concept


def convert_l2_child_to_concept(child: dict, parent_term: str, layer: int = 1) -> dict:
    """Convert an L2 child dict to concept format."""
    sumo_term = child["id"].replace("-", "_").replace("_", " ").title().replace(" ", "")

    concept = {
        "sumo_term": sumo_term,
        "layer": layer,
        "domain": "ActionAgency",
        "is_category_lens": False,  # L2 children are not category lenses
        "is_pseudo_sumo": True,
        "definition": child.get("description", ""),
        "parent_concepts": [parent_term],
        "child_concepts": [],
        "category_children": [],
        # Generate positive examples from the child's example list
        "positive_examples": [],
        "negative_examples": [],
    }

    # Convert examples to prompts
    if "examples" in child:
        for example in child.get("examples", []):
            concept["positive_examples"].append(f"An example of {child['label']}: {example}")
            concept["positive_examples"].append(f"Tell me about {example}")

    return concept


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Format pillars as concept pack")
    parser.add_argument("--input", type=str,
                        default="concept_packs/action-agency-pillars",
                        help="Input directory with expanded pillars")
    parser.add_argument("--output", type=str,
                        default="concept_packs/action-agency-pillars",
                        help="Output directory for formatted pack (can be same as input)")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Load pack manifest
    pack_path = input_dir / "pack.json"
    if not pack_path.exists():
        print(f"Error: pack.json not found at {pack_path}")
        sys.exit(1)

    with open(pack_path) as f:
        pack_manifest = json.load(f)

    print(f"Loading pillars from {input_dir}")
    print(f"Found {len(pack_manifest['pillars'])} pillars")

    # Load all pillar files
    pillars = []
    for pillar_id in pack_manifest["pillars"]:
        pillar_file = input_dir / f"{pillar_id}.json"
        if pillar_file.exists():
            with open(pillar_file) as f:
                pillars.append(json.load(f))
        else:
            print(f"Warning: Pillar file not found: {pillar_file}")

    # Convert to concept format
    layer0_concepts = []
    layer1_concepts = []

    for pillar in pillars:
        # Convert L1 pillar
        l1_concept = convert_pillar_to_concept(pillar, layer=0)
        layer0_concepts.append(l1_concept)

        # Convert L2 children
        parent_term = l1_concept["sumo_term"]
        for child in pillar.get("l2_children", []):
            l2_concept = convert_l2_child_to_concept(child, parent_term, layer=1)
            layer1_concepts.append(l2_concept)

    print(f"Converted {len(layer0_concepts)} L1 concepts")
    print(f"Converted {len(layer1_concepts)} L2 concepts")

    # Create hierarchy directory
    hierarchy_dir = output_dir / "hierarchy"
    hierarchy_dir.mkdir(parents=True, exist_ok=True)

    # Write layer files
    layer0_data = {
        "layer": 0,
        "concepts": layer0_concepts
    }
    with open(hierarchy_dir / "layer0.json", "w") as f:
        json.dump(layer0_data, f, indent=2)

    layer1_data = {
        "layer": 1,
        "concepts": layer1_concepts
    }
    with open(hierarchy_dir / "layer1.json", "w") as f:
        json.dump(layer1_data, f, indent=2)

    # Create proper pack.json
    new_pack = {
        "pack_id": "action-agency-pillars",
        "spec_id": "org.hatcat/action-agency-pillars@0.1.0",
        "version": "0.1.0",
        "created": datetime.now().isoformat() + "Z",
        "description": "L1 pillars covering human activity through Action & Agency lens, generated by Gemma 3 4B as self-description of conceptual space",
        "ontology_stack": {
            "base_ontology": {
                "name": "Generated",
                "version": "0.1.0",
                "source": "google/gemma-3-4b-it",
                "note": "Self-generated taxonomy - model describing its own conceptual space"
            },
            "hierarchy_builder": {
                "script": "scripts/expand_pillars_gemma.py",
                "version": "0.1.0",
                "method": "Action & Agency lens ontologist prompt"
            }
        },
        "concept_metadata": {
            "total_concepts": len(layer0_concepts) + len(layer1_concepts),
            "layers": [0, 1],
            "layer_distribution": {
                "0": len(layer0_concepts),
                "1": len(layer1_concepts)
            },
            "domain_distribution": {
                "ActionAgency": len(layer0_concepts) + len(layer1_concepts)
            },
            "hierarchy_file": "hierarchy/",
            "with_wordnet_mappings": False
        },
        "source_model": pack_manifest.get("source_model", "google/gemma-3-4b-it"),
        "training_examples_summary": {
            "total_positive": pack_manifest.get("total_positive_examples", 0),
            "total_negative": pack_manifest.get("total_negative_examples", 0),
            "total_l2_children": pack_manifest.get("total_l2_children", 0)
        },
        "compatibility": {
            "hatcat_version": ">=0.1.0"
        },
        "distribution": {
            "license": "MIT",
            "authors": ["HatCat Team"]
        }
    }

    with open(output_dir / "pack.json", "w") as f:
        json.dump(new_pack, f, indent=2)

    print(f"\nConcept pack formatted successfully!")
    print(f"  pack.json: {output_dir / 'pack.json'}")
    print(f"  hierarchy/layer0.json: {len(layer0_concepts)} L1 concepts")
    print(f"  hierarchy/layer1.json: {len(layer1_concepts)} L2 concepts")
    print(f"\nTo train lenses:")
    print(f"  python -m src.map.training.train_concept_pack_lenses \\")
    print(f"    --concept-pack action-agency-pillars \\")
    print(f"    --model google/gemma-3-4b-it \\")
    print(f"    --output-dir lens_packs/gemma3-4b_action-agency \\")
    print(f"    --multi-layer")


if __name__ == "__main__":
    main()
