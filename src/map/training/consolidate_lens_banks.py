#!/usr/bin/env python3
"""
Consolidate individual lens .pt files into per-parent bank files.

This dramatically reduces the number of torch.load() calls during
dynamic hierarchy expansion:
- Before: 6-30 individual torch.load() per parent activation
- After: 1 torch.load() per parent activation

Usage:
    python scripts/consolidate_lens_banks.py \
        --input lens_packs/apertus-8b_first-light-bf16 \
        --output lens_packs/apertus-8b_first-light-bf16-banked

The output structure:
    lens_packs/apertus-8b_first-light-bf16-banked/
        pack_info.json
        banks/
            layer0/
                Entity.bank.pt     # Contains all children of Entity
                Physical.bank.pt   # Contains all children of Physical
                ...
            layer1/
                ...
        individual/    # Concepts without children (root/orphan lenses)
            layer0/
                Entity.pt
            ...
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import torch


def load_hierarchy(concept_pack_path: Path) -> Dict:
    """Load hierarchy from concept pack."""
    hierarchy_path = concept_pack_path / "hierarchy.json"
    if hierarchy_path.exists():
        return json.load(open(hierarchy_path))
    return {}


def find_lens_file(lenses_dir: Path, concept_name: str, layer: int) -> Path | None:
    """Find lens file for a concept."""
    # Try layerN/concept_name.pt format
    layer_dir = lenses_dir / f"layer{layer}"
    if layer_dir.exists():
        lens_path = layer_dir / f"{concept_name}.pt"
        if lens_path.exists():
            return lens_path
    return None


def consolidate_lens_pack(
    input_dir: Path,
    output_dir: Path,
    concept_pack_path: Path | None = None,
    workers: int = 8,
    min_children_for_bank: int = 2,  # Only create bank if parent has >= N children
):
    """Consolidate lens pack into per-parent banks."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print("=" * 70)
    print("LENS BANK CONSOLIDATION")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Min children for bank: {min_children_for_bank}")
    print("=" * 70)

    # Load pack info
    pack_info_path = input_dir / "pack_info.json"
    if not pack_info_path.exists():
        print("Error: No pack_info.json found")
        return

    pack_info = json.load(open(pack_info_path))
    source_pack = pack_info.get("source_pack", "first-light")

    # Find concept pack
    if concept_pack_path is None:
        concept_pack_path = Path("concept_packs") / source_pack

    if not concept_pack_path.exists():
        print(f"Error: Concept pack not found: {concept_pack_path}")
        return

    # Load hierarchy
    print("\nLoading hierarchy...")
    hierarchy = load_hierarchy(concept_pack_path)
    parent_to_children = hierarchy.get("parent_to_children", {})
    child_to_parent = hierarchy.get("child_to_parent", {})

    print(f"  Parents: {len(parent_to_children)}")
    print(f"  Total parent-child links: {sum(len(v) for v in parent_to_children.values())}")

    # Find all lens files
    print("\nScanning lens files...")
    all_lenses = list(input_dir.glob("layer*/*.pt"))
    print(f"  Found {len(all_lenses)} lens files")

    # Parse lens files into (concept_name, layer) tuples
    concept_lenses: Dict[Tuple[str, int], Path] = {}
    for lens_path in all_lenses:
        layer = int(lens_path.parent.name.replace("layer", ""))
        concept_name = lens_path.stem
        concept_lenses[(concept_name, layer)] = lens_path

    # Identify which concepts should go into banks vs individual files
    # A concept goes into a bank if its parent has >= min_children_for_bank children
    concepts_in_banks: Set[Tuple[str, int]] = set()
    concepts_individual: Set[Tuple[str, int]] = set()

    for (concept_name, layer), lens_path in concept_lenses.items():
        # Parse parent from concept key format "Concept:layer"
        concept_key = f"{concept_name}:{layer}"
        parent_key = child_to_parent.get(concept_key)

        if parent_key:
            # Extract parent name
            parent_name = parent_key.split(":")[0]
            parent_layer = int(parent_key.split(":")[1]) if ":" in parent_key else layer - 1

            # Check how many siblings this concept has
            siblings = parent_to_children.get(parent_key, [])

            if len(siblings) >= min_children_for_bank:
                concepts_in_banks.add((concept_name, layer))
            else:
                concepts_individual.add((concept_name, layer))
        else:
            # No parent - individual file
            concepts_individual.add((concept_name, layer))

    print(f"\n  Concepts going into banks: {len(concepts_in_banks)}")
    print(f"  Individual concepts: {len(concepts_individual)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    banks_dir = output_dir / "banks"
    individual_dir = output_dir / "individual"
    banks_dir.mkdir(exist_ok=True)
    individual_dir.mkdir(exist_ok=True)

    # Copy pack_info.json and metadata
    print("\nCopying metadata...")
    for src in input_dir.glob("*.json"):
        shutil.copy2(src, output_dir / src.name)

    # Update pack_info to indicate banked format
    pack_info["banked"] = True
    pack_info["bank_min_children"] = min_children_for_bank
    with open(output_dir / "pack_info.json", "w") as f:
        json.dump(pack_info, f, indent=2)

    # Copy individual lenses
    print(f"\nCopying {len(concepts_individual)} individual lenses...")
    for concept_name, layer in concepts_individual:
        src_path = concept_lenses[(concept_name, layer)]
        dst_dir = individual_dir / f"layer{layer}"
        dst_dir.mkdir(exist_ok=True)
        shutil.copy2(src_path, dst_dir / src_path.name)

    # Create banks per parent
    print(f"\nCreating banks...")

    # Group concepts by parent
    parent_to_concept_paths: Dict[str, List[Tuple[str, int, Path]]] = defaultdict(list)

    for concept_name, layer in concepts_in_banks:
        concept_key = f"{concept_name}:{layer}"
        parent_key = child_to_parent.get(concept_key)
        if parent_key:
            lens_path = concept_lenses[(concept_name, layer)]
            parent_to_concept_paths[parent_key].append((concept_name, layer, lens_path))

    # Filter parents with enough children
    valid_banks = {
        parent: children
        for parent, children in parent_to_concept_paths.items()
        if len(children) >= min_children_for_bank
    }

    print(f"  Creating {len(valid_banks)} bank files...")

    # Create banks
    banks_created = 0
    concepts_banked = 0
    total_src_size = 0
    total_dst_size = 0

    bank_index = {}  # parent_key -> {concept_name -> offset in bank}

    for parent_key, children in valid_banks.items():
        parent_name = parent_key.split(":")[0]
        parent_layer = int(parent_key.split(":")[1]) if ":" in parent_key else 0

        # Child layer is parent layer + 1
        child_layer = parent_layer + 1

        # Load all child state dicts
        bank_data = {}
        for concept_name, layer, lens_path in children:
            state_dict = torch.load(lens_path, map_location="cpu")
            bank_data[concept_name] = state_dict
            total_src_size += lens_path.stat().st_size

        # Save bank
        bank_dir = banks_dir / f"layer{child_layer}"
        bank_dir.mkdir(exist_ok=True)
        bank_path = bank_dir / f"{parent_name}.bank.pt"
        torch.save(bank_data, bank_path)

        total_dst_size += bank_path.stat().st_size
        banks_created += 1
        concepts_banked += len(children)

        # Index for lookup
        bank_index[parent_key] = {
            "bank_path": str(bank_path.relative_to(output_dir)),
            "concepts": [c[0] for c in children],
        }

        if banks_created % 100 == 0:
            print(f"    Created {banks_created} banks ({concepts_banked} concepts)...")

    # Save bank index
    with open(output_dir / "bank_index.json", "w") as f:
        json.dump(bank_index, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("CONSOLIDATION COMPLETE")
    print("=" * 70)
    print(f"Banks created: {banks_created}")
    print(f"Concepts in banks: {concepts_banked}")
    print(f"Individual lenses: {len(concepts_individual)}")
    print(f"Total concepts: {concepts_banked + len(concepts_individual)}")

    if total_src_size > 0:
        # Note: banks might be slightly larger due to dict overhead
        print(f"\nBank size overhead:")
        print(f"  Original (banked only): {total_src_size / 1024 / 1024:.1f} MB")
        print(f"  Banked: {total_dst_size / 1024 / 1024:.1f} MB")
        overhead = ((total_dst_size / total_src_size) - 1) * 100
        print(f"  Overhead: {overhead:+.1f}%")

    print(f"\nAvg concepts per bank: {concepts_banked / banks_created:.1f}")
    print(f"torch.load() reduction: {concepts_banked} -> {banks_created} ({100 * (1 - banks_created/concepts_banked):.0f}% reduction)")

    print(f"\nOutput: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Consolidate lens pack into per-parent banks")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input lens pack directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--concept-pack", type=str, help="Concept pack path (auto-detected from pack_info)")
    parser.add_argument("--min-children", type=int, default=2, help="Min children to create bank (default: 2)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    concept_pack_path = Path(args.concept_pack) if args.concept_pack else None

    consolidate_lens_pack(
        input_dir=input_dir,
        output_dir=Path(args.output),
        concept_pack_path=concept_pack_path,
        workers=args.workers,
        min_children_for_bank=args.min_children,
    )


if __name__ == "__main__":
    main()
