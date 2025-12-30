#!/usr/bin/env python3
"""
HatCat Doctor - Model Compatibility Self-Test

Verifies that a model is compatible with HatCat by checking:
1. Model loads successfully
2. Transformer blocks can be enumerated
3. Hidden states can be captured from each layer
4. Generation with hidden state capture works
5. A micro-calibration using HatCat's actual functions works

Usage:
    python scripts/tools/hatcat_doctor.py --model google/gemma-3-4b-pt
    python scripts/tools/hatcat_doctor.py --model meta-llama/Llama-2-7b-hf
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for importing HatCat modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import HatCat modules
try:
    from src.hat.utils.model_loader import ModelLoader
    from src.map.training.activations import get_mean_activation
    HATCAT_IMPORTS_OK = True
except ImportError as e:
    HATCAT_IMPORTS_OK = False
    HATCAT_IMPORT_ERROR = str(e)


class HatCatDoctor:
    """Diagnose model compatibility with HatCat."""

    # Common optional dependencies and their install commands
    OPTIONAL_DEPS = {
        "sentencepiece": "pip install sentencepiece",
        "tiktoken": "pip install tiktoken",
        "protobuf": "pip install protobuf",
        "safetensors": "pip install safetensors",
        "flash_attn": "pip install flash-attn --no-build-isolation",
        "xformers": "pip install xformers",
        "bitsandbytes": "pip install bitsandbytes",
        "einops": "pip install einops",
        "rotary_embedding_torch": "pip install rotary-embedding-torch",
        "triton": "pip install triton",
        "mamba_ssm": "pip install mamba-ssm",
        "causal_conv1d": "pip install causal-conv1d",
    }

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.results = {}
        self.missing_deps = []

    def print_header(self, text: str):
        """Print a section header."""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print('='*60)

    def print_check(self, name: str, passed: bool, details: str = ""):
        """Print a check result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset}  {name}")
        if details:
            print(f"         {details}")

    def check_missing_dep(self, error_msg: str) -> str | None:
        """Check if error is a missing dependency and return install hint."""
        error_lower = error_msg.lower()

        # Check for "No module named 'X'" pattern
        match = re.search(r"no module named ['\"](\w+)['\"]", error_lower)
        if match:
            module = match.group(1)
            if module in self.OPTIONAL_DEPS:
                self.missing_deps.append(module)
                return self.OPTIONAL_DEPS[module]

        # Check for specific package mentions in error
        for dep, install_cmd in self.OPTIONAL_DEPS.items():
            if dep in error_lower or dep.replace('_', '-') in error_lower:
                self.missing_deps.append(dep)
                return install_cmd

        return None

    def run_all_checks(self) -> bool:
        """Run all compatibility checks. Returns True if all pass."""
        self.print_header(f"HatCat Doctor - Model Compatibility Test")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")

        checks = [
            ("HatCat Imports", self.check_hatcat_imports),
            ("Load Model", self.check_model_load),
            ("Enumerate Blocks", self.check_enumerate_blocks),
            ("Capture Hidden States", self.check_hidden_states),
            ("Multi-Layer Capture", self.check_multi_layer),
            ("Generation with Hidden States", self.check_generation),
            ("HatCat Activation Extraction", self.check_hatcat_activation),
            ("Micro-Calibration", self.check_micro_calibration),
        ]

        all_passed = True
        for name, check_fn in checks:
            self.print_header(name)
            try:
                passed = check_fn()
                self.results[name] = passed
                if not passed:
                    all_passed = False
            except Exception as e:
                self.print_check(name, False, f"Exception: {e}")
                self.results[name] = False
                all_passed = False

        self.print_summary(all_passed)
        return all_passed

    def check_hatcat_imports(self) -> bool:
        """Check 0: Can we import HatCat modules?"""
        if HATCAT_IMPORTS_OK:
            self.print_check("HatCat modules imported", True,
                           "ModelLoader, get_mean_activation available")
            return True
        else:
            self.print_check("HatCat imports", False, HATCAT_IMPORT_ERROR)
            return False

    def check_model_load(self) -> bool:
        """Check 1: Can we load the model and tokenizer?"""
        try:
            start = time.time()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.print_check("Tokenizer loaded", True)

            # Load model
            model_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                dtype=model_dtype,
                device_map=self.device
            )
            self.model.eval()

            elapsed = time.time() - start
            param_count = sum(p.numel() for p in self.model.parameters())
            self.print_check("Model loaded", True,
                           f"{param_count:,} parameters in {elapsed:.1f}s")

            return True

        except Exception as e:
            error_msg = str(e)
            install_hint = self.check_missing_dep(error_msg)
            if install_hint:
                self.print_check("Model load", False, f"Missing dependency")
                print(f"         \033[93mInstall with: {install_hint}\033[0m")
            else:
                self.print_check("Model load", False, error_msg[:200])
            return False

    def check_enumerate_blocks(self) -> bool:
        """Check 2: Can we find and enumerate transformer blocks?"""
        if self.model is None:
            self.print_check("Enumerate blocks", False, "Model not loaded")
            return False

        # Try common patterns for finding transformer layers
        # Order matters - prefer main decoder layers over vision/encoder layers
        layer_patterns = [
            "model.layers",           # Llama, Gemma, Mistral (main decoder)
            "transformer.h",          # GPT-2, GPT-J
            "transformer.encoder.layers",  # ChatGLM
            "gpt_neox.layers",        # GPT-NeoX
            "model.decoder.layers",   # BART, T5
            "language_model.model.layers",  # Some VLMs
        ]

        layers = None
        pattern_found = None

        for pattern in layer_patterns:
            try:
                obj = self.model
                for attr in pattern.split('.'):
                    obj = getattr(obj, attr)
                if hasattr(obj, '__len__') and len(obj) > 0:
                    layers = obj
                    pattern_found = pattern
                    break
            except AttributeError:
                continue

        if layers is None:
            # Try generic search - but avoid vision/encoder layers
            skip_patterns = ['vision', 'encoder', 'embed']
            for name, module in self.model.named_modules():
                if any(skip in name.lower() for skip in skip_patterns):
                    continue
                if 'layers' in name.lower() and hasattr(module, '__len__'):
                    try:
                        if len(module) > 0:
                            layers = module
                            pattern_found = name
                            break
                    except:
                        continue

        if layers is not None:
            num_layers = len(layers)
            self.print_check("Found transformer blocks", True,
                           f"{num_layers} layers via '{pattern_found}'")
            self.results['num_layers'] = num_layers
            self.results['layer_pattern'] = pattern_found

            # Check layer structure
            first_layer = layers[0]
            layer_type = type(first_layer).__name__
            self.print_check("Layer type identified", True, layer_type)

            return True
        else:
            self.print_check("Find transformer blocks", False,
                           "Could not locate layer array")
            return False

    def check_hidden_states(self) -> bool:
        """Check 3: Can we capture hidden states from a forward pass?"""
        if self.model is None or self.tokenizer is None:
            self.print_check("Hidden states", False, "Model not loaded")
            return False

        try:
            test_prompt = "The quick brown fox"
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

            if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                self.print_check("Hidden states available", False,
                               "Model did not return hidden_states")
                return False

            hidden_states = outputs.hidden_states
            num_layers = len(hidden_states)
            hidden_dim = hidden_states[0].shape[-1]

            self.print_check("Hidden states captured", True,
                           f"{num_layers} layers, dim={hidden_dim}")
            self.results['hidden_dim'] = hidden_dim

            # Check that hidden states are not all zeros or mostly NaN
            # Try multiple layers in case some have issues
            valid_layer_found = False
            for layer_idx in [-1, -2, len(hidden_states)//2, 0]:
                if layer_idx >= len(hidden_states):
                    continue
                sample_state = hidden_states[layer_idx]
                nan_ratio = torch.isnan(sample_state).float().mean().item()

                if nan_ratio < 0.1:  # Less than 10% NaN is acceptable
                    valid_layer_found = True
                    if nan_ratio > 0:
                        self.print_check("Hidden state validity", True,
                                       f"Layer {layer_idx}: {nan_ratio*100:.1f}% NaN (acceptable)")
                    else:
                        # Compute stats on non-NaN values
                        valid_vals = sample_state[~torch.isnan(sample_state)]
                        if (valid_vals == 0).all():
                            continue  # Try another layer
                        self.print_check("Hidden state validity", True,
                                       f"mean={valid_vals.mean().item():.4f}, "
                                       f"std={valid_vals.std().item():.4f}")
                    break

            if not valid_layer_found:
                self.print_check("Hidden state validity", False,
                               "All layers have >10% NaN values")
                return False

            return True

        except Exception as e:
            self.print_check("Hidden states", False, str(e))
            return False

    def check_multi_layer(self) -> bool:
        """Check 4: Can we capture from multiple specific layers?"""
        if self.model is None:
            self.print_check("Multi-layer capture", False, "Model not loaded")
            return False

        try:
            num_layers = self.results.get('num_layers', 0)
            if num_layers == 0:
                self.print_check("Multi-layer capture", False, "Layer count unknown")
                return False

            # Test layers at different depths
            test_layers = [0, num_layers // 4, num_layers // 2,
                          3 * num_layers // 4, num_layers - 1]
            test_layers = list(set(test_layers))  # Remove duplicates

            test_prompt = "Testing activation capture"
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            captured_layers = []
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

                for layer_idx in test_layers:
                    if layer_idx < len(outputs.hidden_states):
                        state = outputs.hidden_states[layer_idx]
                        if not torch.isnan(state).any():
                            captured_layers.append(layer_idx)

            self.print_check("Multi-layer capture", True,
                           f"Successfully captured layers: {captured_layers}")
            return True

        except Exception as e:
            self.print_check("Multi-layer capture", False, str(e))
            return False

    def check_generation(self) -> bool:
        """Check 5: Can we generate text while capturing hidden states?"""
        if self.model is None or self.tokenizer is None:
            self.print_check("Generation", False, "Model not loaded")
            return False

        try:
            test_prompt = "The meaning of life is"
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            # Remove token_type_ids - some models (Llama) don't accept it
            inputs = {k: v.to(self.device) for k, v in inputs.items()
                     if k != 'token_type_ids'}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Check we got generated tokens
            if not hasattr(outputs, 'sequences'):
                self.print_check("Generation output", False, "No sequences returned")
                return False

            generated_tokens = outputs.sequences.shape[1] - inputs['input_ids'].shape[1]
            self.print_check("Token generation", True,
                           f"Generated {generated_tokens} tokens")

            # Check we got hidden states during generation
            if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                self.print_check("Hidden states during generation", False,
                               "No hidden_states in generate output")
                return False

            num_steps = len(outputs.hidden_states)
            self.print_check("Hidden states during generation", True,
                           f"Captured {num_steps} generation steps")

            # Check that hidden states are valid (at least some layer in some step)
            valid_found = False
            for step_idx, step_states in enumerate(outputs.hidden_states):
                for layer_idx, layer_state in enumerate(step_states):
                    if not torch.isnan(layer_state).all():
                        valid_found = True
                        break
                if valid_found:
                    break

            if not valid_found:
                self.print_check("Hidden state validity (generation)", False,
                               "All hidden states are NaN during generation")
                return False

            self.print_check("Hidden state validity (generation)", True,
                           "Valid activations captured")

            # Decode and show generated text
            generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            preview = generated_text[:60] + "..." if len(generated_text) > 60 else generated_text
            self.print_check("Generated text", True, f'"{preview}"')

            return True

        except Exception as e:
            self.print_check("Generation", False, str(e))
            return False

    def check_hatcat_activation(self) -> bool:
        """Check 6: Test HatCat's actual get_mean_activation function."""
        if not HATCAT_IMPORTS_OK:
            self.print_check("HatCat activation extraction", False,
                           "HatCat modules not available")
            return False

        if self.model is None or self.tokenizer is None:
            self.print_check("HatCat activation extraction", False, "Model not loaded")
            return False

        try:
            test_prompt = "The purpose of HatCat is"

            # Call the actual HatCat function
            activation = get_mean_activation(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=test_prompt,
                device=self.device,
                layer_idx=-1
            )

            # Verify the result
            if activation is None:
                self.print_check("get_mean_activation returned", False, "None")
                return False

            if not isinstance(activation, np.ndarray):
                self.print_check("get_mean_activation type", False,
                               f"Expected ndarray, got {type(activation)}")
                return False

            # Check shape matches expected hidden dim
            expected_dim = self.results.get('hidden_dim')
            if expected_dim and activation.shape[0] != expected_dim:
                self.print_check("Activation shape", False,
                               f"Expected dim {expected_dim}, got {activation.shape}")
                return False

            # Check for NaN values
            nan_ratio = np.isnan(activation).mean()
            if nan_ratio > 0.1:
                self.print_check("Activation validity", False,
                               f"{nan_ratio*100:.1f}% NaN values")
                return False

            # Compute stats
            valid_vals = activation[~np.isnan(activation)]
            self.print_check("get_mean_activation", True,
                           f"shape={activation.shape}, mean={valid_vals.mean():.4f}, "
                           f"std={valid_vals.std():.4f}")

            return True

        except Exception as e:
            self.print_check("HatCat activation extraction", False, str(e))
            return False

    def check_micro_calibration(self) -> bool:
        """Check 5: Run a 10-prompt micro-calibration."""
        if self.model is None:
            self.print_check("Micro-calibration", False, "Model not loaded")
            return False

        calibration_prompts = [
            "What is a person?",
            "Describe a happy moment.",
            "The weather today is",
            "Mathematics involves",
            "In the kitchen, one can find",
            "A computer program is",
            "The color blue reminds me of",
            "When traveling, it's important to",
            "Music can make people feel",
            "The purpose of education is",
        ]

        try:
            start = time.time()
            activations = []

            for i, prompt in enumerate(calibration_prompts):
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True
                    )

                    # Get last layer, last token activation
                    # Try to find a valid layer (some VLMs have NaN in certain layers)
                    activation = None
                    for layer_idx in [-1, -2, len(outputs.hidden_states)//2, 0, 1]:
                        if layer_idx >= len(outputs.hidden_states):
                            continue
                        hidden = outputs.hidden_states[layer_idx]
                        act = hidden[0, -1, :].cpu()
                        if not torch.isnan(act).any():  # No NaN at all
                            activation = act.numpy()
                            break

                    if activation is None:
                        # Fall back to first layer with <50% NaN
                        for layer_idx in range(len(outputs.hidden_states)):
                            hidden = outputs.hidden_states[layer_idx]
                            act = hidden[0, -1, :].cpu()
                            nan_ratio = torch.isnan(act).float().mean().item()
                            if nan_ratio < 0.5:
                                activation = act.numpy()
                                break

                    if activation is None:
                        activation = outputs.hidden_states[0][0, -1, :].cpu().numpy()

                    activations.append(activation)

                print(f"    Prompt {i+1}/10 processed", end='\r')

            elapsed = time.time() - start
            print(" " * 40, end='\r')  # Clear line

            # Compute basic statistics
            activations = np.stack(activations)

            # Handle NaN values
            if np.isnan(activations).all():
                self.print_check("Micro-calibration", False,
                               "All activations are NaN")
                return False

            # Compute stats on valid values
            valid_mask = ~np.isnan(activations)
            if valid_mask.sum() == 0:
                self.print_check("Micro-calibration", False,
                               "No valid activation values")
                return False

            mean_activation = np.nanmean(activations, axis=0)
            std_activation = np.nanstd(activations, axis=0)

            # Check for reasonable variance (not all same)
            mean_std = np.nanmean(std_activation)
            if np.isnan(mean_std) or mean_std < 1e-6:
                self.print_check("Micro-calibration", False,
                               "No variance in activations")
                return False

            self.print_check("Micro-calibration", True,
                           f"10 prompts in {elapsed:.2f}s, "
                           f"activation std={mean_std:.4f}")

            # Report throughput
            prompts_per_sec = len(calibration_prompts) / elapsed
            self.print_check("Throughput", True, f"{prompts_per_sec:.1f} prompts/sec")

            return True

        except Exception as e:
            self.print_check("Micro-calibration", False, str(e))
            return False

    def print_summary(self, all_passed: bool):
        """Print final summary."""
        self.print_header("Summary")

        if all_passed:
            print(f"  \033[92m✓ Model '{self.model_name}' is COMPATIBLE with HatCat\033[0m")
            print()
            print("  Detected configuration:")
            print(f"    - Layers: {self.results.get('num_layers', 'unknown')}")
            print(f"    - Hidden dim: {self.results.get('hidden_dim', 'unknown')}")
            print(f"    - Layer pattern: {self.results.get('layer_pattern', 'unknown')}")
        else:
            print(f"  \033[91m✗ Model '{self.model_name}' has COMPATIBILITY ISSUES\033[0m")
            print()
            print("  Failed checks:")
            for name, passed in self.results.items():
                if passed is False:
                    print(f"    - {name}")

            # Show missing dependencies with install commands
            if self.missing_deps:
                print()
                print("  \033[93mMissing optional dependencies:\033[0m")
                for dep in set(self.missing_deps):
                    if dep in self.OPTIONAL_DEPS:
                        print(f"    {self.OPTIONAL_DEPS[dep]}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="HatCat Doctor - Test model compatibility"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )

    args = parser.parse_args()

    doctor = HatCatDoctor(args.model, args.device)
    success = doctor.run_all_checks()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
