#!/usr/bin/env python3
"""
Qualify skeleton node IDs with parent context.

Transforms IDs like "bias-detection-9" to "explainable-internal-state--bias-detection"
to ensure uniqueness across branches.

Also updates polar MELDs to use the new qualified IDs.

Usage:
    python scripts/tools/qualify_skeleton_ids.py \
        --skeleton results/introspective_skeleton.json \
        --melds results/polar_melds_v2 \
        --output-skeleton results/introspective_skeleton_qualified.json \
        --output-melds results/polar_melds_v3
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def strip_numeric_suffix(id_str: str) -> str:
    """Remove trailing numeric suffix like -9 from an ID."""
    return re.sub(r'-\d+$', '', id_str)


def slugify(text: str) -> str:
    """Convert text to kebab-case slug."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')


def qualify_node_ids(
    node: Dict,
    parent_base_id: Optional[str] = None,
    parent_full_id: Optional[str] = None,
    id_mapping: Dict[str, str] = None,
    seen_ids: set = None
) -> Tuple[Dict, Dict[str, str]]:
    """
    Recursively qualify node IDs with immediate parent context.

    Uses format: {parent_base_id}--{child_base_id}
    This keeps IDs reasonably short while still disambiguating across branches.

    For collision cases (same parent-child name), adds level prefix: L{n}-{id}

    Returns:
        Tuple of (updated_node, id_mapping) where id_mapping tracks old->new ID changes
    """
    if id_mapping is None:
        id_mapping = {}
    if seen_ids is None:
        seen_ids = set()

    old_id = node.get("id", "")
    level = node.get("level", 1)
    base_id = strip_numeric_suffix(old_id)

    # For root nodes (L1), keep the base ID
    # For children, qualify with immediate parent's base ID only
    if parent_base_id:
        new_id = f"{parent_base_id}--{base_id}"
    else:
        new_id = base_id

    # Handle collisions by adding level prefix and counter if needed
    if new_id in seen_ids:
        # Try with level prefix first
        candidate = f"{parent_base_id}--L{level}-{base_id}" if parent_base_id else f"L{level}-{base_id}"
        if candidate in seen_ids:
            # Add counter for multiple collisions
            counter = 2
            while f"{candidate}-{counter}" in seen_ids:
                counter += 1
            new_id = f"{candidate}-{counter}"
        else:
            new_id = candidate

    seen_ids.add(new_id)

    # Track the mapping (old_id -> new_id)
    if old_id != new_id:
        id_mapping[old_id] = new_id

    # Update node
    node["id"] = new_id
    if parent_full_id:
        node["parent_id"] = parent_full_id

    # Recursively process children
    # Pass our BASE id as their parent_base_id (for constructing their qualified ID)
    # Pass our FULL new_id as their parent_full_id (for parent_id field reference)
    if "children" in node:
        for child in node["children"]:
            qualify_node_ids(child, base_id, new_id, id_mapping, seen_ids)

    return node, id_mapping


def update_skeleton(skeleton_path: Path) -> Tuple[Dict, Dict[str, str]]:
    """Load and update skeleton with qualified IDs."""
    with open(skeleton_path) as f:
        skeleton = json.load(f)

    all_mappings = {}

    # Process each root
    for root in skeleton.get("roots", []):
        _, mappings = qualify_node_ids(root, parent_base_id=None, parent_full_id=None)
        all_mappings.update(mappings)

    return skeleton, all_mappings


def update_meld(meld_path: Path, id_mapping: Dict[str, str]) -> Optional[Dict]:
    """Update a polar MELD with qualified node ID."""
    try:
        with open(meld_path) as f:
            meld = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: Failed to load {meld_path}: {e}")
        return None

    # Get old node ID
    old_id = meld.get("node", {}).get("id", "")

    # Get parent from generation context
    parent_label = meld.get("polar_meld", {}).get("_generation_context", {}).get("parent")

    if not old_id:
        return meld

    # Always construct qualified ID from parent label if available
    # (id_mapping has collisions when multiple nodes share the same old_id)
    if parent_label:
        parent_slug = slugify(parent_label)
        base_id = strip_numeric_suffix(old_id)
        new_id = f"{parent_slug}--{base_id}"
    elif old_id in id_mapping:
        # Fallback to mapping for root-level concepts without parent
        new_id = id_mapping[old_id]
    else:
        # No parent, just strip numeric suffix
        new_id = strip_numeric_suffix(old_id)

    # Update the node ID
    meld["node"]["id"] = new_id

    return meld


def process_melds(melds_dir: Path, output_dir: Path, id_mapping: Dict[str, str]) -> Dict[str, int]:
    """Process all MELDs in directory structure."""
    stats = {"processed": 0, "updated": 0, "failed": 0}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each level directory
    for level_dir in sorted(melds_dir.iterdir()):
        if not level_dir.is_dir() or not level_dir.name.startswith("L"):
            continue

        output_level_dir = output_dir / level_dir.name
        output_level_dir.mkdir(exist_ok=True)

        for meld_file in sorted(level_dir.glob("*.json")):
            stats["processed"] += 1

            updated_meld = update_meld(meld_file, id_mapping)

            if updated_meld is None:
                stats["failed"] += 1
                continue

            # Check if ID changed
            old_id = None
            try:
                with open(meld_file) as f:
                    old_data = json.load(f)
                    old_id = old_data.get("node", {}).get("id")
            except:
                pass

            new_id = updated_meld.get("node", {}).get("id")
            if old_id != new_id:
                stats["updated"] += 1

            # Write updated MELD
            output_path = output_level_dir / meld_file.name
            with open(output_path, "w") as f:
                json.dump(updated_meld, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Qualify skeleton and MELD IDs with parent context")
    parser.add_argument(
        "--skeleton", "-s",
        type=Path,
        default=Path("results/introspective_skeleton.json"),
        help="Input skeleton file"
    )
    parser.add_argument(
        "--melds", "-m",
        type=Path,
        default=Path("results/polar_melds_v2"),
        help="Input MELDs directory"
    )
    parser.add_argument(
        "--output-skeleton", "-os",
        type=Path,
        default=None,
        help="Output skeleton file (default: overwrite input)"
    )
    parser.add_argument(
        "--output-melds", "-om",
        type=Path,
        default=None,
        help="Output MELDs directory (default: overwrite input)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing"
    )

    args = parser.parse_args()

    # Default outputs to inputs (in-place update)
    output_skeleton = args.output_skeleton or args.skeleton
    output_melds = args.output_melds or args.melds

    print("=" * 60)
    print("QUALIFYING SKELETON AND MELD IDs")
    print("=" * 60)
    print(f"Skeleton: {args.skeleton}")
    print(f"MELDs: {args.melds}")
    print(f"Output skeleton: {output_skeleton}")
    print(f"Output MELDs: {output_melds}")
    print()

    # Process skeleton
    print("Processing skeleton...")
    skeleton, id_mapping = update_skeleton(args.skeleton)

    print(f"  Found {len(id_mapping)} IDs to update")

    # Show sample mappings
    if id_mapping:
        print("  Sample mappings:")
        for old_id, new_id in list(id_mapping.items())[:5]:
            print(f"    {old_id} -> {new_id}")
        if len(id_mapping) > 5:
            print(f"    ... and {len(id_mapping) - 5} more")

    # Process MELDs
    print("\nProcessing MELDs...")
    if args.melds.exists():
        meld_stats = process_melds(args.melds, output_melds, id_mapping)
        print(f"  Processed: {meld_stats['processed']}")
        print(f"  Updated: {meld_stats['updated']}")
        print(f"  Failed: {meld_stats['failed']}")
    else:
        print(f"  MELDs directory not found: {args.melds}")
        meld_stats = {"processed": 0, "updated": 0, "failed": 0}

    # Write outputs
    if not args.dry_run:
        print("\nWriting outputs...")

        # Write skeleton
        with open(output_skeleton, "w") as f:
            json.dump(skeleton, f, indent=2)
        print(f"  Wrote skeleton: {output_skeleton}")

        print("\nDone!")
    else:
        print("\n[DRY RUN] No files written")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Skeleton IDs updated: {len(id_mapping)}")
    print(f"MELDs processed: {meld_stats['processed']}")
    print(f"MELDs updated: {meld_stats['updated']}")


if __name__ == "__main__":
    main()
