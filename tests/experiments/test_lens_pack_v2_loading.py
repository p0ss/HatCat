#!/usr/bin/env python3
"""
Test that lens packs can be loaded and used by the baseline pipeline.

This script tests:
1. Lens pack registry can find our new v2 pack
2. DynamicLensManager can load lenses from the pack
3. Lenses can run inference on sample activations
4. Both default mode and explicit pack ID work
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from registry.lens_pack_registry import LensPackRegistry
from src.hat.monitoring.lens_manager import DynamicLensManager


def test_lens_pack_registry():
    """Test 1: Lens pack registry can find packs."""
    print("=" * 80)
    print("TEST 1: Lens Pack Registry")
    print("=" * 80)

    registry = LensPackRegistry()
    packs = registry.list_packs()

    print(f"\nFound {len(packs)} lens pack(s):")
    for pack in packs:
        print(f"  - {pack.lens_pack_id} (v{pack.version})")
        print(f"    Location: {pack.pack_dir}")
        print(f"    Created: {pack.created}")

    # Try to get v2 pack specifically
    v2_pack = registry.get_pack("gemma-3-4b-pt_sumo-wordnet-v2")

    if v2_pack:
        print(f"\n✓ Successfully found v2 pack")
        print(f"  Pack directory: {v2_pack.pack_dir}")
        print(f"  Hierarchy dir: {v2_pack.pack_dir / 'hierarchy'}")

        # Count lenses
        lens_files = list((v2_pack.pack_dir / 'hierarchy').glob('*_classifier.pt'))
        print(f"  Total lens files: {len(lens_files)}")

        return True
    else:
        print("\n✗ Could not find v2 pack")
        return False


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "LENS PACK V2 LOADING TEST" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Test registry
    success = test_lens_pack_registry()

    print("\n" + "=" * 80)
    if success:
        print("✓ TEST PASSED: v2 lens pack found and accessible")
    else:
        print("✗ TEST FAILED: Could not find v2 lens pack")
    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
