#!/usr/bin/env python3
"""
Apply enhancement melds to existing concepts in a concept pack.

Takes the output from generate_missing_concept_melds.py and updates
existing concepts with new definitions, examples, and training hints.

Usage:
    python scripts/tools/apply_enhancement_melds.py \
        --melds results/missing_concept_melds.json \
        --pack concept_packs/first-light \
        --dry-run

This is a simpler alternative to apply_meld.py which only handles
new concept additions. This script updates existing concepts in place.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple


def load_melds(meld_file: Path) -> Dict:
    """Load the enhancement meld file."""
    with open(meld_file) as f:
        return json.load(f)


def load_all_layers(pack_dir: Path) -> Tuple[Dict[int, Dict], Dict[str, Tuple[int, int]]]:
    """
    Load all layer files and build concept index.

    Returns:
        (layer_data, concept_index)
        - layer_data: {layer_num: layer_json}
        - concept_index: {sumo_term: (layer_num, index_in_concepts_list)}
    """
    hierarchy_dir = pack_dir / "hierarchy"
    layer_data = {}
    concept_index = {}

    for layer_num in range(7):
        layer_file = hierarchy_dir / f"layer{layer_num}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            data = json.load(f)

        layer_data[layer_num] = data

        for idx, concept in enumerate(data.get("concepts", [])):
            term = concept.get("sumo_term", "")
            if term:
                concept_index[term] = (layer_num, idx)

    return layer_data, concept_index


def apply_enhancement(
    concept: Dict,
    enhancement: Dict,
    meld_id: str
) -> bool:
    """
    Apply enhancements to a concept.

    Returns True if any changes were made.
    """
    changed = False

    # Update definition
    if "definition" in enhancement:
        if concept.get("definition") != enhancement["definition"]:
            concept["definition"] = enhancement["definition"]
            # Also update sumo_definition if it's the same or empty
            if not concept.get("sumo_definition"):
                concept["sumo_definition"] = enhancement["definition"]
            changed = True

    # Update positive_examples
    if "positive_examples" in enhancement:
        hints = concept.setdefault("training_hints", {})
        existing = set(hints.get("positive_examples", []))
        new_examples = enhancement["positive_examples"]
        if not existing or existing != set(new_examples):
            hints["positive_examples"] = new_examples
            changed = True

    # Update negative_examples
    if "negative_examples" in enhancement:
        hints = concept.setdefault("training_hints", {})
        existing = set(hints.get("negative_examples", []))
        new_examples = enhancement["negative_examples"]
        if not existing or existing != set(new_examples):
            hints["negative_examples"] = new_examples
            changed = True

    # Update training_hints fields
    if "training_hints" in enhancement:
        hints = concept.setdefault("training_hints", {})
        new_hints = enhancement["training_hints"]

        if "disambiguation" in new_hints:
            if hints.get("disambiguation") != new_hints["disambiguation"]:
                hints["disambiguation"] = new_hints["disambiguation"]
                changed = True

        if "confusable_with" in new_hints:
            if hints.get("confusable_with") != new_hints["confusable_with"]:
                hints["confusable_with"] = new_hints["confusable_with"]
                changed = True

        if "key_features" in new_hints:
            if hints.get("key_features") != new_hints["key_features"]:
                hints["key_features"] = new_hints["key_features"]
                changed = True

    # Add provenance if changed
    if changed:
        concept["enhancement_source"] = {
            "meld_id": meld_id,
            "applied_at": datetime.now().isoformat() + "Z"
        }

    return changed


def save_layers(pack_dir: Path, layer_data: Dict[int, Dict], modified_layers: Set[int]):
    """Save modified layer files."""
    hierarchy_dir = pack_dir / "hierarchy"

    for layer_num in modified_layers:
        if layer_num not in layer_data:
            continue

        data = layer_data[layer_num]

        # Update summary
        data["summary"] = {
            "total_concepts": len(data.get("concepts", [])),
            "last_modified": datetime.now().isoformat() + "Z"
        }

        layer_file = hierarchy_dir / f"layer{layer_num}.json"
        with open(layer_file, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Apply enhancement melds to existing concepts"
    )
    parser.add_argument(
        "--melds", type=Path, required=True,
        help="Path to meld file (from generate_missing_concept_melds.py)"
    )
    parser.add_argument(
        "--pack", type=Path, required=True,
        help="Path to concept pack directory"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--only-missing", action="store_true",
        help="Only apply to concepts that currently have no definition"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("APPLY ENHANCEMENT MELDS")
    print("=" * 60)

    # Load meld file
    print(f"\nLoading melds from: {args.melds}")
    meld_data = load_melds(args.melds)

    metadata = meld_data.get("metadata", {})
    operations = meld_data.get("operations", [])

    print(f"  Total operations: {len(operations)}")
    print(f"  Source: {metadata.get('pack', 'unknown')}")
    print(f"  Generated: {metadata.get('timestamp', 'unknown')}")

    # Load concept pack
    print(f"\nLoading concept pack: {args.pack}")
    layer_data, concept_index = load_all_layers(args.pack)
    print(f"  Loaded {len(concept_index)} concepts across {len(layer_data)} layers")

    # Apply operations
    print("\nApplying enhancements...")

    stats = {
        "total": len(operations),
        "applied": 0,
        "skipped_not_found": 0,
        "skipped_has_data": 0,
        "unchanged": 0,
        "by_layer": {}
    }

    modified_layers: Set[int] = set()
    meld_id = f"enhance-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    for op in operations:
        if op.get("operation") != "enhance":
            continue

        target = op.get("target", "")
        enhancements = op.get("enhancements", {})

        # Find concept
        if target not in concept_index:
            stats["skipped_not_found"] += 1
            if args.verbose:
                print(f"  SKIP {target}: not found in pack")
            continue

        layer_num, idx = concept_index[target]
        concept = layer_data[layer_num]["concepts"][idx]

        # Check if we should skip (already has data)
        if args.only_missing:
            if concept.get("definition") or concept.get("sumo_definition"):
                stats["skipped_has_data"] += 1
                if args.verbose:
                    print(f"  SKIP {target}: already has definition")
                continue

        # Apply enhancement
        if not args.dry_run:
            changed = apply_enhancement(concept, enhancements, meld_id)
        else:
            # Simulate change check for dry run
            changed = bool(enhancements)

        if changed:
            stats["applied"] += 1
            modified_layers.add(layer_num)
            stats["by_layer"][layer_num] = stats["by_layer"].get(layer_num, 0) + 1

            if args.verbose:
                print(f"  OK {target} (layer {layer_num})")
        else:
            stats["unchanged"] += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total operations: {stats['total']}")
    print(f"Applied: {stats['applied']}")
    print(f"Skipped (not found): {stats['skipped_not_found']}")
    if args.only_missing:
        print(f"Skipped (has data): {stats['skipped_has_data']}")
    print(f"Unchanged: {stats['unchanged']}")

    if stats["by_layer"]:
        print("\nBy layer:")
        for layer, count in sorted(stats["by_layer"].items()):
            print(f"  Layer {layer}: {count}")

    # Save
    if args.dry_run:
        print("\n[DRY RUN] No changes made")
    else:
        save_layers(args.pack, layer_data, modified_layers)
        print(f"\n✓ Updated {len(modified_layers)} layer files")
        print(f"  Enhanced {stats['applied']} concepts")


if __name__ == "__main__":
    main()
