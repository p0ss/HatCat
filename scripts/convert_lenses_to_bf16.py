#!/usr/bin/env python3
"""
Convert lens pack from fp32 to bfloat16.

Reduces lens pack size by ~50%, speeding up loading from disk/RAM to GPU.
Precision loss is negligible for inference.

Usage:
    python scripts/convert_lenses_to_bf16.py \
        --input lens_packs/apertus-8b_first-light_calibration-test-2 \
        --output lens_packs/apertus-8b_first-light_calibration-test-2-bf16

    # In-place conversion (backs up originals to .fp32.bak):
    python scripts/convert_lenses_to_bf16.py \
        --input lens_packs/apertus-8b_first-light_calibration-test-2 \
        --in-place
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch


def convert_lens_file(src_path: Path, dst_path: Path, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Convert a single lens file to target dtype."""
    try:
        # Load state dict
        state_dict = torch.load(src_path, map_location='cpu')

        # Check if already in target dtype
        first_tensor = next(iter(state_dict.values()))
        if first_tensor.dtype == dtype:
            # Already converted, just copy
            if src_path != dst_path:
                shutil.copy2(src_path, dst_path)
            return {"status": "skipped", "reason": "already_bf16"}

        # Convert to target dtype
        converted = {k: v.to(dtype) if v.dtype.is_floating_point else v
                     for k, v in state_dict.items()}

        # Save
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(converted, dst_path)

        # Calculate size reduction
        src_size = src_path.stat().st_size
        dst_size = dst_path.stat().st_size
        reduction = (1 - dst_size / src_size) * 100

        return {
            "status": "converted",
            "src_size": src_size,
            "dst_size": dst_size,
            "reduction_pct": reduction,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def convert_lens_pack(
    input_dir: Path,
    output_dir: Path,
    dtype: torch.dtype = torch.bfloat16,
    in_place: bool = False,
    workers: int = 8,
):
    """Convert all lenses in a pack to target dtype."""

    input_dir = Path(input_dir)

    if in_place:
        output_dir = input_dir
        backup_suffix = ".fp32.bak"
    else:
        output_dir = Path(output_dir)
        backup_suffix = None

    print(f"{'=' * 60}")
    print(f"LENS PACK CONVERSION: fp32 → bfloat16")
    print(f"{'=' * 60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode:   {'In-place (with backup)' if in_place else 'Copy to new location'}")
    print(f"Workers: {workers}")
    print(f"{'=' * 60}")

    # Find all .pt files
    pt_files = list(input_dir.glob("**/*.pt"))
    print(f"\nFound {len(pt_files)} lens files")

    if not pt_files:
        print("No .pt files found!")
        return

    # Copy non-.pt files first (json, logs, etc.)
    if not in_place:
        print("\nCopying metadata files...")
        for src in input_dir.glob("**/*"):
            if src.is_file() and not src.suffix == ".pt":
                rel_path = src.relative_to(input_dir)
                dst = output_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    # Convert lens files in parallel
    print(f"\nConverting {len(pt_files)} lens files...")

    results = {"converted": 0, "skipped": 0, "errors": 0}
    total_src_size = 0
    total_dst_size = 0

    def process_file(src_path):
        rel_path = src_path.relative_to(input_dir)
        dst_path = output_dir / rel_path

        if in_place:
            # Backup original
            backup_path = src_path.with_suffix(src_path.suffix + backup_suffix)
            if not backup_path.exists():
                shutil.copy2(src_path, backup_path)

        return src_path, convert_lens_file(src_path, dst_path, dtype)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_file, f) for f in pt_files]

        for i, future in enumerate(as_completed(futures)):
            src_path, result = future.result()

            if result["status"] == "converted":
                results["converted"] += 1
                total_src_size += result["src_size"]
                total_dst_size += result["dst_size"]
            elif result["status"] == "skipped":
                results["skipped"] += 1
            else:
                results["errors"] += 1
                print(f"\n  Error: {src_path.name}: {result.get('error', 'unknown')}")

            # Progress
            if (i + 1) % 500 == 0 or (i + 1) == len(pt_files):
                print(f"  Progress: {i + 1}/{len(pt_files)} files processed")

    # Summary
    print(f"\n{'=' * 60}")
    print("CONVERSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Converted: {results['converted']}")
    print(f"Skipped:   {results['skipped']} (already bf16)")
    print(f"Errors:    {results['errors']}")

    if total_src_size > 0:
        reduction = (1 - total_dst_size / total_src_size) * 100
        print(f"\nSize reduction:")
        print(f"  Before: {total_src_size / 1024 / 1024 / 1024:.2f} GB")
        print(f"  After:  {total_dst_size / 1024 / 1024 / 1024:.2f} GB")
        print(f"  Saved:  {(total_src_size - total_dst_size) / 1024 / 1024 / 1024:.2f} GB ({reduction:.1f}%)")

    # Update pack_info.json to note dtype
    pack_info_path = output_dir / "pack_info.json"
    if pack_info_path.exists():
        pack_info = json.load(open(pack_info_path))
        pack_info["dtype"] = "bfloat16"
        pack_info["converted_from_fp32"] = True
        with open(pack_info_path, "w") as f:
            json.dump(pack_info, f, indent=2)
        print(f"\nUpdated pack_info.json with dtype=bfloat16")

    print(f"\n{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert lens pack to bfloat16")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input lens pack directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory (required unless --in-place)")
    parser.add_argument("--in-place", action="store_true", help="Convert in place (backs up originals)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")

    args = parser.parse_args()

    if not args.in_place and not args.output:
        parser.error("--output is required unless --in-place is specified")

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_dir

    convert_lens_pack(
        input_dir=input_dir,
        output_dir=output_dir,
        in_place=args.in_place,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
