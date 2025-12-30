#!/usr/bin/env python3
"""
Test MAP compliance for HatCat lens pack discovery and loading.

Tests:
1. Discovery of concept packs
2. Discovery of lens packs (both legacy and MAP)
3. Version resolution
4. Legacy pack loading with deprecation warning
5. Manager initialization with auto-detection
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.monitoring.lens_manager import DynamicLensManager


def test_concept_pack_discovery():
    """Test 1: Discover concept packs."""
    print("\n" + "=" * 80)
    print("TEST 1: Concept Pack Discovery")
    print("=" * 80)

    packs = DynamicLensManager.discover_concept_packs()

    print(f"\n✓ Found {len(packs)} concept pack(s):")
    for pack_id, path in packs.items():
        print(f"  - {pack_id}")
        print(f"    Path: {path}")

    return len(packs) > 0, packs


def test_lens_pack_discovery():
    """Test 2: Discover lens packs (legacy and MAP)."""
    print("\n" + "=" * 80)
    print("TEST 2: Lens Pack Discovery")
    print("=" * 80)

    packs = DynamicLensManager.discover_lens_packs()

    print(f"\n✓ Found {len(packs)} lens pack(s):")

    legacy_count = 0
    map_count = 0

    for pack_id, info in sorted(packs.items()):
        pack_type = info['type']
        if pack_type == 'legacy':
            legacy_count += 1
            print(f"  - {pack_id} [LEGACY]")
        else:
            map_count += 1
            print(f"  - {pack_id} [MAP]")
            print(f"    Concept Pack: {info.get('concept_pack_id')}")
            print(f"    Substrate: {info.get('substrate_id')}")
        print(f"    Path: {info['path']}")

    print(f"\n  Summary: {legacy_count} legacy, {map_count} MAP-compliant")

    return len(packs) > 0, packs


def test_version_resolution():
    """Test 3: Version resolution."""
    print("\n" + "=" * 80)
    print("TEST 3: Version Resolution")
    print("=" * 80)

    # Try to find latest version of a concept pack
    test_names = ["sumo-wordnet", "ai-safety"]

    for name in test_names:
        latest = DynamicLensManager.get_latest_version(name)
        if latest:
            print(f"\n✓ Latest version of '{name}': {latest}")
        else:
            print(f"\n  No versions found for '{name}'")

    return True, None


def test_legacy_loading_with_deprecation():
    """Test 4: Legacy pack loading with deprecation warning."""
    print("\n" + "=" * 80)
    print("TEST 4: Legacy Pack Loading (with deprecation warning)")
    print("=" * 80)

    # Get first legacy pack
    packs = DynamicLensManager.discover_lens_packs()
    legacy_packs = {k: v for k, v in packs.items() if v['type'] == 'legacy'}

    if not legacy_packs:
        print("\n⚠ No legacy packs found to test")
        return True, None

    pack_id = list(legacy_packs.keys())[0]
    print(f"\nAttempting to load legacy pack: {pack_id}")
    print("Expected: Deprecation warning should appear\n")

    try:
        manager = DynamicLensManager(
            lens_pack_id=pack_id,
            base_layers=[0],
            device='cpu'
        )

        print(f"\n✓ Manager initialized successfully")
        print(f"  Loaded {len(manager.loaded_activation_lenses)} activation lenses")
        print(f"  Loaded {len(manager.loaded_text_lenses)} text lenses")

        return True, manager

    except Exception as e:
        print(f"\n✗ Error loading legacy pack: {e}")
        return False, None


def test_auto_detection():
    """Test 5: Auto-detection when no pack specified."""
    print("\n" + "=" * 80)
    print("TEST 5: Auto-Detection")
    print("=" * 80)

    print("\nAttempting to initialize manager with auto-detection")
    print("(Using nonexistent lenses_dir to trigger auto-detection)\n")

    try:
        manager = DynamicLensManager(
            lenses_dir=Path("nonexistent_dir"),
            base_layers=[0],
            device='cpu'
        )

        print(f"\n✓ Manager auto-detected and loaded pack")
        print(f"  Using pack: {manager.lenses_dir.name}")
        print(f"  Loaded {len(manager.loaded_activation_lenses)} activation lenses")
        print(f"  Loaded {len(manager.loaded_text_lenses)} text lenses")

        return True, manager

    except Exception as e:
        print(f"\n✗ Error with auto-detection: {e}")
        return False, None


def main():
    """Run all MAP compliance tests."""
    print("\n" + "=" * 80)
    print("HatCat MAP Compliance Test Suite")
    print("=" * 80)

    results = {}

    # Run tests
    results['concept_discovery'] = test_concept_pack_discovery()
    results['lens_discovery'] = test_lens_pack_discovery()
    results['version_resolution'] = test_version_resolution()
    results['legacy_loading'] = test_legacy_loading_with_deprecation()
    results['auto_detection'] = test_auto_detection()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = 0
    failed = 0

    for test_name, (success, _) in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:25s} {status}")
        if success:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
