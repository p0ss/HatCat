"""GPU memory cleanup utilities.

This module provides utilities to properly cleanup GPU memory after experiments,
preventing OOM errors when running multiple scripts sequentially.
"""

import gc
import torch
from typing import Any


def cleanup_model(*objects: Any) -> None:
    """Cleanup GPU memory by deleting model/tokenizer objects and clearing cache.

    Args:
        *objects: Variable number of objects to delete (model, tokenizer, etc.)

    Example:
        >>> cleanup_model(model, tokenizer)
    """
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage information.

    Returns:
        dict with keys: allocated_gb, reserved_gb, free_gb, total_gb
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - allocated

    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
        "total_gb": round(total, 2)
    }


def print_gpu_memory() -> None:
    """Print current GPU memory usage."""
    info = get_gpu_memory_info()
    if "error" in info:
        print(f"GPU Memory: {info['error']}")
    else:
        print(f"GPU Memory: {info['allocated_gb']:.2f}GB allocated, "
              f"{info['free_gb']:.2f}GB free / {info['total_gb']:.2f}GB total")
