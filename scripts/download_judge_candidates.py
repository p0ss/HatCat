#!/usr/bin/env python3
"""
Download judge candidate models from HuggingFace.

Pre-downloads models so evaluation doesn't stall on network I/O.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.be.thalamos.model_candidates import MODEL_CANDIDATES


def main():
    parser = argparse.ArgumentParser(description="Download judge candidate models")
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Model candidate IDs to download (default: all that fit VRAM)"
    )
    parser.add_argument(
        "--max-vram",
        type=float,
        default=24.0,
        help="Maximum VRAM in GB - only download models that fit (default: 24)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available candidates and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Model Candidates:")
        print("-" * 80)
        for cid, candidate in MODEL_CANDIDATES.items():
            fit = "Y" if candidate.vram_gb_estimate <= args.max_vram else "N"
            print(
                f"  {cid:<30} {candidate.params_billions:>5.1f}B  "
                f"{candidate.vram_gb_estimate:>5.1f}GB  "
                f"Fits {args.max_vram}GB: {fit}"
            )
        return

    # Select candidates
    if args.models:
        candidates = [
            MODEL_CANDIDATES[cid]
            for cid in args.models
            if cid in MODEL_CANDIDATES
        ]
    else:
        candidates = [
            c for c in MODEL_CANDIDATES.values()
            if c.vram_gb_estimate <= args.max_vram
        ]

    # Deduplicate by model_id (e.g., 4-bit variant shares same base model)
    seen_ids = set()
    unique_candidates = []
    for c in candidates:
        if c.model_id not in seen_ids:
            seen_ids.add(c.model_id)
            unique_candidates.append(c)

    print(f"\nDownloading {len(unique_candidates)} models...")
    print("=" * 60)

    for candidate in unique_candidates:
        print(f"\n>>> {candidate.name} ({candidate.model_id})")
        print(f"    Estimated size: ~{candidate.vram_gb_estimate * 2:.0f}GB on disk")

        try:
            if candidate.model_class == "AutoModelForImageTextToText":
                from transformers import AutoProcessor
                print("    Downloading processor...")
                AutoProcessor.from_pretrained(candidate.model_id)

            from transformers import AutoTokenizer, AutoConfig
            print("    Downloading tokenizer...")
            AutoTokenizer.from_pretrained(
                candidate.model_id,
                trust_remote_code=candidate.trust_remote_code,
            )

            print("    Downloading model config...")
            AutoConfig.from_pretrained(
                candidate.model_id,
                trust_remote_code=candidate.trust_remote_code,
            )

            # Download model weights using snapshot_download for better progress
            from huggingface_hub import snapshot_download
            print("    Downloading model weights...")
            snapshot_download(
                candidate.model_id,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip non-pytorch formats
            )

            print(f"    ✓ {candidate.name} downloaded successfully")

        except Exception as e:
            print(f"    ✗ Failed to download {candidate.name}: {e}")

    print("\n" + "=" * 60)
    print("Done! Models cached in ~/.cache/huggingface/hub/")


if __name__ == "__main__":
    main()
