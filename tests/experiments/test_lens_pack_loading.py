#!/usr/bin/env python3
"""
Test loading lenses from lens pack structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.monitoring.lens_manager import DynamicLensManager

def test_lens_pack_loading():
    print("=" * 80)
    print("TESTING LENS PACK LOADING")
    print("=" * 80)
    print()

    # Test with lens pack
    print("Loading with lens pack ID: gemma-3-4b-pt_sumo-wordnet-v1")
    print()

    manager = DynamicLensManager(
        lens_pack_id="gemma-3-4b-pt_sumo-wordnet-v1",
        base_layers=[0],
        use_activation_lenses=True,
        use_text_lenses=True,
        keep_top_k=10,
        device="cpu"  # Use CPU to avoid OOM with server running
    )

    print()
    print("=" * 80)
    print("LOADING TEST RESULTS")
    print("=" * 80)
    print()
    print(f"Total concepts in metadata: {len(manager.concept_metadata)}")
    print(f"Loaded activation lenses: {len(manager.loaded_activation_lenses)}")
    print(f"Loaded text lenses: {len(manager.loaded_text_lenses)}")
    print()

    # Show some examples
    print("Example loaded lenses:")
    for i, (concept_key, lens) in enumerate(list(manager.loaded_activation_lenses.items())[:5]):
        concept, layer = concept_key
        print(f"  {concept} (layer {layer})")

    print()
    print("✅ Lens pack loading successful!")

if __name__ == "__main__":
    test_lens_pack_loading()
