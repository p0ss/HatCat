"""
Multi-Resolution Fuzzing

Extends the base fuzzer to support different levels of dimensional reduction:
- 1-bit: Binary (current default) - responds / doesn't respond
- 2-bit: Ternary - none / weak / strong
- Continuous: Full float magnitude

This allows testing whether discovered clusters are artifacts of the
1-bit thresholding or persist across resolution levels.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from enum import Enum
import logging

from .fuzzer import TopologyFuzzer, FuzzConfig, LayerFuzzResult

logger = logging.getLogger(__name__)


class Resolution(Enum):
    """Resolution levels for connectivity measurement."""
    ONE_BIT = "1-bit"      # Binary: 0 or 1
    TWO_BIT = "2-bit"      # Ternary: 0, 0.5, 1 (none/weak/strong)
    CONTINUOUS = "continuous"  # Full float values


@dataclass
class MultiResConfig(FuzzConfig):
    """Extended config for multi-resolution fuzzing."""
    resolution: Resolution = Resolution.ONE_BIT
    # Thresholds for 2-bit mode
    weak_threshold: float = 0.05   # Below this = none
    strong_threshold: float = 0.2  # Above this = strong, between = weak


class MultiResolutionFuzzer(TopologyFuzzer):
    """
    Fuzzer that supports multiple resolution levels.

    Usage:
        config = MultiResConfig(resolution=Resolution.TWO_BIT)
        fuzzer = MultiResolutionFuzzer(model, tokenizer, config)
        results = fuzzer.fuzz_all_layers()
    """

    def __init__(self, model, tokenizer, config: Optional[MultiResConfig] = None):
        self.multi_config = config or MultiResConfig()
        super().__init__(model, tokenizer, self.multi_config)

        logger.info(f"MultiResolutionFuzzer: resolution={self.multi_config.resolution.value}")

    def _quantize_connectivity(
        self,
        connectivity: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Quantize connectivity matrices based on resolution setting.

        Args:
            connectivity: Raw connectivity matrices per target layer

        Returns:
            Quantized connectivity matrices
        """
        resolution = self.multi_config.resolution

        if resolution == Resolution.CONTINUOUS:
            # No quantization - return as-is (already normalized)
            return connectivity

        elif resolution == Resolution.ONE_BIT:
            # Binary: above threshold = 1, else = 0
            quantized = {}
            for layer_idx, conn in connectivity.items():
                quantized[layer_idx] = (conn > self.config.response_threshold).astype(np.float32)
            return quantized

        elif resolution == Resolution.TWO_BIT:
            # Ternary: none (0) / weak (0.5) / strong (1.0)
            quantized = {}
            weak_t = self.multi_config.weak_threshold
            strong_t = self.multi_config.strong_threshold

            for layer_idx, conn in connectivity.items():
                result = np.zeros_like(conn)
                # Weak: between weak and strong thresholds
                weak_mask = (conn > weak_t) & (conn <= strong_t)
                result[weak_mask] = 0.5
                # Strong: above strong threshold
                strong_mask = conn > strong_t
                result[strong_mask] = 1.0
                quantized[layer_idx] = result

            return quantized

        else:
            raise ValueError(f"Unknown resolution: {resolution}")

    def fuzz_layer_batched(
        self,
        source_layer: int,
        target_layers: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> LayerFuzzResult:
        """
        Fuzz layer with resolution-appropriate quantization.
        """
        # Get raw results from parent
        result = super().fuzz_layer_batched(source_layer, target_layers, show_progress)

        # Quantize based on resolution
        result.connectivity = self._quantize_connectivity(result.connectivity)

        return result

    def fuzz_layer(
        self,
        source_layer: int,
        target_layers: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> LayerFuzzResult:
        """
        Fuzz layer (sequential) with resolution-appropriate quantization.
        """
        result = super().fuzz_layer(source_layer, target_layers, show_progress)
        result.connectivity = self._quantize_connectivity(result.connectivity)
        return result


def compare_resolutions(
    results_1bit: Dict[int, LayerFuzzResult],
    results_2bit: Dict[int, LayerFuzzResult],
    results_continuous: Dict[int, LayerFuzzResult],
) -> Dict:
    """
    Compare fuzzing results across resolution levels.

    Returns metrics on how similar the connectivity patterns are.
    """
    comparisons = {}

    for layer in results_1bit.keys():
        if layer not in results_2bit or layer not in results_continuous:
            continue

        layer_comparison = {}

        for target_layer in results_1bit[layer].connectivity.keys():
            conn_1bit = results_1bit[layer].connectivity.get(target_layer)
            conn_2bit = results_2bit[layer].connectivity.get(target_layer)
            conn_cont = results_continuous[layer].connectivity.get(target_layer)

            if conn_1bit is None or conn_2bit is None or conn_cont is None:
                continue

            # Correlation between resolutions
            flat_1bit = conn_1bit.flatten()
            flat_2bit = conn_2bit.flatten()
            flat_cont = conn_cont.flatten()

            # Handle zero variance
            def safe_corr(a, b):
                if a.var() == 0 or b.var() == 0:
                    return 0.0
                return np.corrcoef(a, b)[0, 1]

            layer_comparison[target_layer] = {
                "corr_1bit_2bit": safe_corr(flat_1bit, flat_2bit),
                "corr_1bit_cont": safe_corr(flat_1bit, flat_cont),
                "corr_2bit_cont": safe_corr(flat_2bit, flat_cont),
                # Density (fraction of non-zero entries)
                "density_1bit": (flat_1bit > 0).mean(),
                "density_2bit": (flat_2bit > 0).mean(),
                "density_cont": (flat_cont > 0).mean(),
            }

        comparisons[layer] = layer_comparison

    # Aggregate statistics
    all_corr_1bit_2bit = []
    all_corr_1bit_cont = []
    all_corr_2bit_cont = []

    for layer_comp in comparisons.values():
        for target_comp in layer_comp.values():
            all_corr_1bit_2bit.append(target_comp["corr_1bit_2bit"])
            all_corr_1bit_cont.append(target_comp["corr_1bit_cont"])
            all_corr_2bit_cont.append(target_comp["corr_2bit_cont"])

    return {
        "per_layer": comparisons,
        "aggregate": {
            "mean_corr_1bit_2bit": np.mean(all_corr_1bit_2bit),
            "mean_corr_1bit_cont": np.mean(all_corr_1bit_cont),
            "mean_corr_2bit_cont": np.mean(all_corr_2bit_cont),
            "std_corr_1bit_2bit": np.std(all_corr_1bit_2bit),
            "std_corr_1bit_cont": np.std(all_corr_1bit_cont),
            "std_corr_2bit_cont": np.std(all_corr_2bit_cont),
        }
    }
