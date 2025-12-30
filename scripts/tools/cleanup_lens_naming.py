#!/usr/bin/env python3
"""
Cleanup lens pack naming: standardize on {concept}.pt without _classifier suffix.

This script:
1. Finds all lens files with _classifier suffix
2. If a duplicate exists without suffix, keeps the newer one
3. Renames remaining _classifier files to drop the suffix
4. Reports statistics

Usage:
    # Dry run (default):
    python scripts/tools/cleanup_lens_naming.py --input lens_packs/apertus-8b_first-light-bf16

    # Actually perform cleanup:
    python scripts/tools/cleanup_lens_naming.py --input lens_packs/apertus-8b_first-light-bf16 --execute
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict


def cleanup_lens_naming(input_dir: Path, execute: bool = False):
    """Clean up lens naming to standardize on {concept}.pt format."""

    input_dir = Path(input_dir)

    print("=" * 70)
    print("LENS NAMING CLEANUP")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Mode: {'EXECUTE' if execute else 'DRY RUN'}")
    print("=" * 70)

    # Find all .pt files
    all_files = list(input_dir.glob("layer*/*.pt"))
    print(f"\nFound {len(all_files)} lens files")

    # Group by (layer, base_concept_name)
    # base_concept_name = stem without _classifier suffix
    groups = defaultdict(list)  # (layer, concept) -> [paths]

    for f in all_files:
        layer = f.parent.name  # "layer0", "layer1", etc.
        stem = f.stem

        if stem.endswith("_classifier"):
            base_name = stem[:-11]  # Remove "_classifier"
            has_suffix = True
        else:
            base_name = stem
            has_suffix = False

        groups[(layer, base_name)].append({
            "path": f,
            "has_suffix": has_suffix,
            "mtime": f.stat().st_mtime,
            "size": f.stat().st_size,
        })

    # Analyze groups
    unique_concepts = len(groups)
    with_suffix_only = 0
    without_suffix_only = 0
    duplicates = 0

    to_delete = []
    to_rename = []

    for (layer, concept), files in groups.items():
        has_suffixed = [f for f in files if f["has_suffix"]]
        has_clean = [f for f in files if not f["has_suffix"]]

        if has_suffixed and has_clean:
            # Duplicate! Keep the newer one, prefer clean name
            duplicates += 1

            # Sort by mtime, newest first
            all_sorted = sorted(files, key=lambda x: x["mtime"], reverse=True)

            # Keep the newest clean one, or newest suffixed if no clean
            if has_clean:
                # Prefer clean name, keep newest clean
                clean_sorted = sorted(has_clean, key=lambda x: x["mtime"], reverse=True)
                keep = clean_sorted[0]
                delete = [f for f in files if f["path"] != keep["path"]]
            else:
                # No clean, keep newest suffixed and rename
                keep = all_sorted[0]
                delete = all_sorted[1:]

            for f in delete:
                to_delete.append(f["path"])

            # If we kept a suffixed file, rename it
            if keep["has_suffix"]:
                new_name = keep["path"].parent / f"{concept}.pt"
                to_rename.append((keep["path"], new_name))

        elif has_suffixed:
            # Only suffixed version exists - rename it
            with_suffix_only += 1
            for f in has_suffixed:
                new_name = f["path"].parent / f"{concept}.pt"
                to_rename.append((f["path"], new_name))

        else:
            # Only clean version exists - good!
            without_suffix_only += 1

    print(f"\nAnalysis:")
    print(f"  Unique concepts: {unique_concepts}")
    print(f"  Already clean (no suffix): {without_suffix_only}")
    print(f"  Need renaming (suffix only): {with_suffix_only}")
    print(f"  Duplicates (both versions): {duplicates}")
    print(f"\nActions needed:")
    print(f"  Files to delete: {len(to_delete)}")
    print(f"  Files to rename: {len(to_rename)}")

    if to_delete:
        print(f"\nSample deletions:")
        for p in to_delete[:5]:
            print(f"  DELETE: {p.relative_to(input_dir)}")
        if len(to_delete) > 5:
            print(f"  ... and {len(to_delete) - 5} more")

    if to_rename:
        print(f"\nSample renames:")
        for old, new in to_rename[:5]:
            print(f"  {old.name} -> {new.name}")
        if len(to_rename) > 5:
            print(f"  ... and {len(to_rename) - 5} more")

    if execute:
        print(f"\n{'=' * 70}")
        print("EXECUTING CLEANUP")
        print("=" * 70)

        # Delete duplicates
        deleted = 0
        for p in to_delete:
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                print(f"  Error deleting {p}: {e}")
        print(f"  Deleted {deleted} duplicate files")

        # Rename suffixed files
        renamed = 0
        for old, new in to_rename:
            try:
                if new.exists():
                    # Target exists (shouldn't happen after deletions, but be safe)
                    print(f"  Warning: {new} already exists, skipping rename of {old}")
                    continue
                old.rename(new)
                renamed += 1
            except Exception as e:
                print(f"  Error renaming {old} -> {new}: {e}")
        print(f"  Renamed {renamed} files")

        # Final count
        final_files = list(input_dir.glob("layer*/*.pt"))
        print(f"\nFinal file count: {len(final_files)}")

        # Verify no _classifier files remain
        remaining_suffixed = [f for f in final_files if f.stem.endswith("_classifier")]
        if remaining_suffixed:
            print(f"  Warning: {len(remaining_suffixed)} files still have _classifier suffix")
        else:
            print("  All files now use clean naming")
    else:
        print(f"\n{'=' * 70}")
        print("DRY RUN - No changes made. Use --execute to apply.")
        print("=" * 70)

    return {
        "unique_concepts": unique_concepts,
        "deleted": len(to_delete) if execute else 0,
        "renamed": len(to_rename) if execute else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Cleanup lens file naming")
    parser.add_argument("--input", "-i", type=str, required=True, help="Lens pack directory")
    parser.add_argument("--execute", action="store_true", help="Actually perform cleanup (default: dry run)")

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)

    cleanup_lens_naming(input_dir, execute=args.execute)


if __name__ == "__main__":
    main()
