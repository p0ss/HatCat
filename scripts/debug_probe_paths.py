#!/usr/bin/env python3
"""Debug probe path loading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.dynamic_probe_manager import DynamicProbeManager

manager = DynamicProbeManager(device="cpu", base_layers=[])

print("Layer 0 concepts:")
for name, meta in manager.concept_metadata.items():
    if meta.layer == 0:
        probe_exists = "✓" if meta.activation_probe_path else "✗"
        print(f"  {probe_exists} {name:30s} path={meta.activation_probe_path}")
