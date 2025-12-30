"""
Model loading utilities for activation capture.
Supports Gemma-3 270M and other transformer models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional


class ModelLoader:
    """Handles loading and setup of models for activation capture."""

    @staticmethod
    def load_gemma_270m(
        model_name: str = "google/gemma-3-270m",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load Gemma-3 270M model and tokenizer.

        Args:
            model_name: Hugging Face model identifier
            device: Device to load model on (None = auto-detect)
            load_in_8bit: Whether to use 8-bit quantization
            trust_remote_code: Whether to trust remote code

        Returns:
            Tuple of (model, tokenizer)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {model_name} on {device}...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "dtype": torch.float16 if device == "cuda" else torch.float32,
        }

        if load_in_8bit and device == "cuda":
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = device

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        model.eval()  # Set to evaluation mode

        print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, tokenizer

    @staticmethod
    def get_layer_info(model: torch.nn.Module) -> dict:
        """
        Get information about model layers.

        Returns:
            Dictionary with layer names, types, and shapes
        """
        layer_info = {}

        for name, module in model.named_modules():
            layer_info[name] = {
                "type": type(module).__name__,
                "parameters": sum(p.numel() for p in module.parameters()),
            }

        return layer_info

    @staticmethod
    def print_layer_structure(model: torch.nn.Module, filter_str: Optional[str] = None):
        """
        Print model layer structure.

        Args:
            model: PyTorch model
            filter_str: Optional string to filter layer names
        """
        print("\nModel Layer Structure:")
        print("-" * 80)

        for name, module in model.named_modules():
            if filter_str is None or filter_str.lower() in name.lower():
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    print(f"{name:60s} | {type(module).__name__:25s} | {param_count:,}")

        print("-" * 80)
