#!/usr/bin/env python3
"""
Test loading probes from probe pack structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.dynamic_probe_manager import DynamicProbeManager

def test_probe_pack_loading():
    print("=" * 80)
    print("TESTING PROBE PACK LOADING")
    print("=" * 80)
    print()

    # Test with probe pack
    print("Loading with probe pack ID: gemma-3-4b-pt_sumo-wordnet-v1")
    print()

    manager = DynamicProbeManager(
        probe_pack_id="gemma-3-4b-pt_sumo-wordnet-v1",
        base_layers=[0],
        use_activation_probes=True,
        use_text_probes=True,
        keep_top_k=10,
        device="cpu"  # Use CPU to avoid OOM with server running
    )

    print()
    print("=" * 80)
    print("LOADING TEST RESULTS")
    print("=" * 80)
    print()
    print(f"Total concepts in metadata: {len(manager.concept_metadata)}")
    print(f"Loaded activation probes: {len(manager.loaded_activation_probes)}")
    print(f"Loaded text probes: {len(manager.loaded_text_probes)}")
    print()

    # Show some examples
    print("Example loaded probes:")
    for i, (concept_key, probe) in enumerate(list(manager.loaded_activation_probes.items())[:5]):
        concept, layer = concept_key
        print(f"  {concept} (layer {layer})")

    print()
    print("✅ Probe pack loading successful!")

if __name__ == "__main__":
    test_probe_pack_loading()
