#!/usr/bin/env python3
"""
Convert Polar MELDs to hierarchy format for use with existing training pipeline.

Creates layerN.json files that can be used with train_concept_pack_lenses.py.

For each polar MELD concept, creates TWO hierarchy entries:
- {term}_positive: Trained on positive pole examples vs confusables
- {term}_negative: Trained on negative pole examples vs confusables

Usage:
    python scripts/tools/convert_polar_melds_to_hierarchy.py \
        --meld-dir results/polar_melds \
        --output concept_packs/polar-introspective/hierarchy \
        --level 1
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "map"))

from training.polar_meld_adapter import polar_meld_to_concept


def load_polar_melds(meld_dir: Path, level: int) -> List[Dict]:
    """Load all polar MELDs for a given level."""
    level_dir = meld_dir / f"L{level}"
    melds = []

    for meld_file in sorted(level_dir.glob("*.json")):
        try:
            data = json.loads(meld_file.read_text())
            melds.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {meld_file}: {e}")

    return melds


def create_hierarchy_layer(melds: List[Dict], level: int) -> Dict:
    """
    Convert polar MELDs to hierarchy layer format.

    Creates two concepts per MELD: one for positive pole, one for negative pole.
    """
    concepts = []

    for meld_data in melds:
        # Create positive pole concept
        pos_concept = polar_meld_to_concept(meld_data, "positive")
        concepts.append(pos_concept)

        # Create negative pole concept
        neg_concept = polar_meld_to_concept(meld_data, "negative")
        concepts.append(neg_concept)

    return {
        "layer": level,
        "concepts": concepts,
        "metadata": {
            "source": "polar_melds",
            "generated_at": datetime.now().isoformat(),
            "n_source_melds": len(melds),
            "n_concepts": len(concepts),
        }
    }


def create_pack_json(output_dir: Path, levels: List[int], total_concepts: int) -> Dict:
    """Create pack.json metadata file."""
    return {
        "pack_id": "polar-introspective",
        "spec_id": "org.hatcat/polar-introspective@1.0.0",
        "version": "1.0.0",
        "created": datetime.now().isoformat() + "Z",
        "description": "Introspective ontology with polar MELDs - positive and negative pole lenses for each concept",
        "concept_metadata": {
            "total_concepts": total_concepts,
            "layers": levels,
            "hierarchy_file": "hierarchy/",
            "source": "polar_melds",
            "note": "Each source concept has two entries: {term}_positive and {term}_negative"
        },
        "compatibility": {
            "hatcat_version": ">=0.1.0"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Convert polar MELDs to hierarchy format")
    parser.add_argument("--meld-dir", type=Path, default=Path("results/polar_melds"))
    parser.add_argument("--output", type=Path, default=Path("concept_packs/polar-introspective/hierarchy"))
    parser.add_argument("--level", "-l", type=int, nargs="+", default=[1],
                       help="Level(s) to convert (default: 1)")

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    total_concepts = 0
    converted_levels = []

    for level in args.level:
        print(f"\nConverting level {level}...")

        # Load polar MELDs
        melds = load_polar_melds(args.meld_dir, level)
        print(f"  Loaded {len(melds)} polar MELDs")

        if not melds:
            print(f"  No MELDs found for level {level}, skipping")
            continue

        # Convert to hierarchy format
        layer_data = create_hierarchy_layer(melds, level)
        n_concepts = len(layer_data["concepts"])
        total_concepts += n_concepts
        converted_levels.append(level)

        # Save layer file
        layer_file = args.output / f"layer{level}.json"
        layer_file.write_text(json.dumps(layer_data, indent=2))
        print(f"  Created {layer_file} with {n_concepts} concepts")

    # Create pack.json
    pack_dir = args.output.parent
    pack_json = create_pack_json(pack_dir, converted_levels, total_concepts)
    pack_file = pack_dir / "pack.json"
    pack_file.write_text(json.dumps(pack_json, indent=2))
    print(f"\nCreated {pack_file}")

    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Levels converted: {converted_levels}")
    print(f"Total concepts: {total_concepts}")
    print(f"Output: {args.output}")
    print(f"\nTo train, run:")
    print(f"  python src/map/training/train_concept_pack_lenses.py \\")
    print(f"    --concept-pack polar-introspective \\")
    print(f"    --model google/gemma-3-4b-it \\")
    print(f"    --output lens_packs/gemma-3-4b_polar-introspective \\")
    print(f"    --layers {' '.join(str(l) for l in converted_levels)}")


if __name__ == "__main__":
    main()
