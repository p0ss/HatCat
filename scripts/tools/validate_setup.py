"""
Quick validation script to check if the setup is working.
This is a minimal test before running the full concept capture.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.hat.utils.model_loader import ModelLoader


def main():
    """Quick validation of setup."""
    print("=" * 80)
    print("HATCAT SETUP VALIDATION")
    print("=" * 80)

    # Check PyTorch
    print("\n1. PyTorch Setup:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check if we can load model metadata (without downloading)
    print("\n2. Model Access:")
    print("   Checking Hugging Face access...")

    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("google/gemma-3-270m", trust_remote_code=True)
        print(f"   ✓ Can access Gemma-3 270M config")
        print(f"   - Model type: {config.model_type}")
        print(f"   - Hidden size: {config.hidden_size}")
        print(f"   - Num layers: {config.num_hidden_layers}")
    except Exception as e:
        print(f"   ✗ Cannot access model: {e}")
        print("   Note: You may need to accept the license at https://huggingface.co/google/gemma-3-270m")
        return 1

    # Check dependencies
    print("\n3. Dependencies:")
    deps = {
        'torch': torch,
        'transformers': None,
        'h5py': None,
        'numpy': None,
        'tqdm': None
    }

    try:
        import transformers
        deps['transformers'] = transformers
        import h5py
        deps['h5py'] = h5py
        import numpy
        deps['numpy'] = numpy
        import tqdm
        deps['tqdm'] = tqdm

        for name, module in deps.items():
            if module is not None:
                version = getattr(module, '__version__', 'unknown')
                print(f"   ✓ {name}: {version}")

    except ImportError as e:
        print(f"   ✗ Missing dependency: {e}")
        return 1

    # Check directory structure
    print("\n4. Directory Structure:")
    required_dirs = [
        "src/activation_capture",
        "src/utils",
        "tests",
        "scripts",
        "data/raw",
        "data/processed",
        "models"
    ]

    for dir_path in required_dirs:
        full_path = Path(__file__).parent.parent / dir_path
        status = "✓" if full_path.exists() else "✗"
        print(f"   {status} {dir_path}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: python tests/test_activation_capture.py")
    print("2. Run: python scripts/capture_concepts.py")
    print("\nNote: First run will download Gemma-3 270M (~540MB)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
