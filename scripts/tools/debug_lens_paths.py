#!/usr/bin/env python3
"""Debug lens path loading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hat.monitoring.lens_manager import DynamicLensManager

manager = DynamicLensManager(device="cpu", base_layers=[])

print("Layer 0 concepts:")
for name, meta in manager.concept_metadata.items():
    if meta.layer == 0:
        lens_exists = "✓" if meta.activation_lens_path else "✗"
        print(f"  {lens_exists} {name:30s} path={meta.activation_lens_path}")
