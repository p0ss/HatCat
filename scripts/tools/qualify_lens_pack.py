#!/usr/bin/env python3
"""
Rename lens pack probes to use qualified node IDs.

Updates:
- .pt file names: {old_id}_positive.pt -> {new_id}_positive.pt
- results.json: node_id and file references

Usage:
    python scripts/tools/qualify_lens_pack.py \
        --lens-pack lens_packs/gemma-3-4b_polar-introspective-v2 \
        --melds results/polar_melds_v3 \
        --dry-run
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple


def build_id_mapping(melds_dir: Path) -> Dict[str, str]:
    """Build mapping from old node_id to new qualified node_id from MELDs."""
    mapping = {}

    for level_dir in sorted(melds_dir.iterdir()):
        if not level_dir.is_dir() or not level_dir.name.startswith("L"):
            continue

        for meld_file in level_dir.glob("*.json"):
            try:
                with open(meld_file) as f:
                    meld = json.load(f)

                new_id = meld.get("node", {}).get("id", "")

                # Try to find old_id from the generation context or infer from new_id
                # The new_id format is "parent--child", old_id is "child" or "child-N"
                if "--" in new_id:
                    # Extract the child part (after last --)
                    child_part = new_id.split("--")[-1]
                    # Also need to map with numeric suffixes
                    # Check _generation_context for clues
                    gen_ctx = meld.get("polar_meld", {}).get("_generation_context", {})

                    # We need to find what the original skeleton had
                    # For now, map the base child to the new full id
                    mapping[child_part] = new_id

            except (json.JSONDecodeError, IOError):
                continue

    return mapping


def build_id_mapping_from_melds(original_melds: Path, qualified_melds: Path) -> Dict[str, str]:
    """Build mapping by comparing original and qualified MELDs."""
    mapping = {}

    for level_dir in sorted(original_melds.iterdir()):
        if not level_dir.is_dir() or not level_dir.name.startswith("L"):
            continue

        qualified_level = qualified_melds / level_dir.name
        if not qualified_level.exists():
            continue

        for meld_file in level_dir.glob("*.json"):
            qualified_file = qualified_level / meld_file.name
            if not qualified_file.exists():
                continue

            try:
                with open(meld_file) as f:
                    old_meld = json.load(f)
                with open(qualified_file) as f:
                    new_meld = json.load(f)

                old_id = old_meld.get("node", {}).get("id", "")
                new_id = new_meld.get("node", {}).get("id", "")

                if old_id and new_id and old_id != new_id:
                    mapping[old_id] = new_id

            except (json.JSONDecodeError, IOError):
                continue

    return mapping


def rename_lens_pack(
    lens_pack: Path,
    id_mapping: Dict[str, str],
    dry_run: bool = False
) -> Dict[str, int]:
    """Rename probes in lens pack to use qualified IDs."""
    stats = {"files_renamed": 0, "results_updated": 0, "concepts_updated": 0}

    for level_dir in sorted(lens_pack.iterdir()):
        if not level_dir.is_dir() or not level_dir.name.startswith("L"):
            continue

        for layer_dir in level_dir.iterdir():
            if not layer_dir.is_dir() or not layer_dir.name.startswith("layer"):
                continue

            results_file = layer_dir / "results.json"
            if not results_file.exists():
                continue

            # Load results
            with open(results_file) as f:
                results = json.load(f)

            updated = False

            for concept in results.get("concepts", []):
                old_id = concept.get("node_id", "")

                if old_id not in id_mapping:
                    continue

                new_id = id_mapping[old_id]

                # Update concept metadata
                concept["node_id"] = new_id
                stats["concepts_updated"] += 1
                updated = True

                # Rename positive and negative probe files
                for pole in ["positive", "negative"]:
                    pole_data = concept.get(pole)
                    if not pole_data or not isinstance(pole_data, dict):
                        continue

                    old_file = pole_data.get("file", "")
                    if not old_file:
                        continue

                    # Construct new filename
                    new_file = old_file.replace(old_id, new_id)
                    concept[pole]["file"] = new_file

                    # Rename the actual file
                    old_path = layer_dir / old_file
                    new_path = layer_dir / new_file

                    if old_path.exists():
                        if not dry_run:
                            os.rename(old_path, new_path)
                        stats["files_renamed"] += 1
                        print(f"  {old_file} -> {new_file}")

            # Save updated results
            if updated:
                if not dry_run:
                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=2)
                stats["results_updated"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Rename lens pack probes to qualified IDs")
    parser.add_argument(
        "--lens-pack", "-l",
        type=Path,
        default=Path("lens_packs/gemma-3-4b_polar-introspective-v2"),
        help="Lens pack directory"
    )
    parser.add_argument(
        "--original-melds", "-o",
        type=Path,
        default=Path("results/polar_melds"),
        help="Original MELDs directory (with old IDs)"
    )
    parser.add_argument(
        "--qualified-melds", "-q",
        type=Path,
        default=Path("results/polar_melds_v3"),
        help="Qualified MELDs directory (with new IDs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("QUALIFYING LENS PACK IDs")
    print("=" * 60)
    print(f"Lens pack: {args.lens_pack}")
    print(f"Original MELDs: {args.original_melds}")
    print(f"Qualified MELDs: {args.qualified_melds}")
    if args.dry_run:
        print("[DRY RUN]")
    print()

    # Build ID mapping
    print("Building ID mapping...")
    id_mapping = build_id_mapping_from_melds(args.original_melds, args.qualified_melds)
    print(f"  Found {len(id_mapping)} ID mappings")

    # Show samples
    if id_mapping:
        print("  Sample mappings:")
        for old_id, new_id in list(id_mapping.items())[:5]:
            print(f"    {old_id} -> {new_id}")
        if len(id_mapping) > 5:
            print(f"    ... and {len(id_mapping) - 5} more")
    print()

    # Rename probes
    print("Renaming probes...")
    stats = rename_lens_pack(args.lens_pack, id_mapping, dry_run=args.dry_run)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files renamed: {stats['files_renamed']}")
    print(f"Results files updated: {stats['results_updated']}")
    print(f"Concepts updated: {stats['concepts_updated']}")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified")


if __name__ == "__main__":
    main()
