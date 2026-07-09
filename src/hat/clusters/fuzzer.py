"""
1-Bit Forward Fuzzer

Probes inter-layer connectivity by perturbing individual neurons and measuring
downstream responses. "1-bit" refers to the binary on/off perturbation pattern.

Two modes:
1. Batched fuzzing (default): Perturb different neurons in each batch element.
   Reduces forward passes from O(N) to O(N/batch_size).
2. Sequential fuzzing (legacy): One forward pass per neuron. Clean but slow.

Strategy:
1. Run baseline forward pass to get typical activation magnitudes
2. For each neuron batch, inject perturbations (one neuron per batch element)
3. Measure which downstream neurons respond (above noise threshold)
4. Build forward connectivity map: source -> downstream responders

This captures ~85% of effective connectivity, missing higher-order interactions.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
import logging
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


@dataclass
class FuzzConfig:
    """Configuration for topology fuzzing."""
    perturbation_scale: float = 3.0  # Multiplier on typical activation magnitude
    response_threshold: float = 0.1  # Min response to count as connected
    batch_size: int = 64  # Neurons to fuzz in parallel
    layers_to_fuzz: Optional[List[int]] = None  # None = all layers
    layers_to_measure: Optional[List[int]] = None  # None = all downstream
    max_neurons_per_layer: Optional[int] = None  # None = all neurons
    use_rank_scoring: bool = True  # Convert to ranks before aggregation
    random_seed: int = 42


@dataclass
class FuzzResult:
    """Result of fuzzing a single source neuron."""
    source_layer: int
    source_neuron: int
    responses: Dict[int, np.ndarray]  # layer -> response magnitudes per neuron
    top_responders: Dict[int, List[int]]  # layer -> top responding neuron indices


@dataclass
class LayerFuzzResult:
    """Aggregated fuzzing results for a whole layer."""
    source_layer: int
    n_neurons: int
    # Connectivity matrix: [n_source_neurons, n_target_neurons] per target layer
    connectivity: Dict[int, np.ndarray]
    # Top-k connections per source neuron per target layer
    top_k_connections: Dict[int, List[List[int]]]  # layer -> [neuron -> [top targets]]


class TopologyFuzzer:
    """
    Forward fuzzer for discovering inter-layer connectivity.

    Usage:
        fuzzer = TopologyFuzzer(model, tokenizer, config)

        # Batched fuzzing (fast, recommended)
        result = fuzzer.fuzz_layer_batched(source_layer=5)

        # Sequential fuzzing (slow, for comparison)
        result = fuzzer.fuzz_layer(source_layer=5)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[FuzzConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or FuzzConfig()

        # Get model structure
        self.layers = self._get_layers()
        self.n_layers = len(self.layers)
        self.hidden_dim = self._get_hidden_dim()
        self.device = next(self.model.parameters()).device

        # Storage for captured activations
        self._captured_activations: Dict[int, torch.Tensor] = {}
        self._hooks: List = []

        # Baseline activations (computed once)
        self._baseline: Optional[Dict[int, torch.Tensor]] = None
        self._baseline_magnitudes: Optional[Dict[int, float]] = None

        # For batched fuzzing
        self._batch_perturbation_mask: Optional[torch.Tensor] = None

        logger.info(f"TopologyFuzzer initialized: {self.n_layers} layers, hidden_dim={self.hidden_dim}")

    def _get_layers(self) -> nn.ModuleList:
        """Get transformer layers from model."""
        if hasattr(self.model, 'model'):
            inner = self.model.model
            if hasattr(inner, 'layers'):
                return inner.layers
            if hasattr(inner, 'language_model') and hasattr(inner.language_model, 'layers'):
                return inner.language_model.layers
        raise AttributeError(f"Cannot find layers in model: {type(self.model)}")

    def _get_hidden_dim(self) -> int:
        """Get hidden dimension from model config or first layer."""
        if hasattr(self.model, 'config'):
            config = self.model.config
            # Direct attribute
            if hasattr(config, 'hidden_size'):
                return config.hidden_size
            # Nested config (e.g., Gemma 3 uses text_config)
            if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
                return config.text_config.hidden_size
        # Fallback: inspect first layer
        for name, param in self.layers[0].named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                return param.shape[-1]
        raise ValueError("Cannot determine hidden dimension")

    def _capture_hook(self, layer_idx: int) -> Callable:
        """Create hook to capture activations at a layer."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Store last token activation [batch, hidden_dim]
            self._captured_activations[layer_idx] = hidden[:, -1, :].detach()
        return hook

    def _batched_inject_hook(self, layer_idx: int, neuron_indices: torch.Tensor, perturbation: float) -> Callable:
        """
        Create hook for batched perturbation injection.

        Each batch element i gets neuron neuron_indices[i] perturbed.
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            batch_size = hidden.shape[0]
            n_indices = len(neuron_indices)

            # Each batch element gets its own neuron perturbed
            for b in range(min(batch_size, n_indices)):
                neuron_idx = neuron_indices[b].item()
                hidden[b, -1, neuron_idx] = hidden[b, -1, neuron_idx] + perturbation

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    def _inject_hook(self, layer_idx: int, neuron_idx: int, perturbation: float) -> Callable:
        """Create hook to inject perturbation at specific neuron (sequential mode)."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                hidden[:, -1, neuron_idx] = hidden[:, -1, neuron_idx] + perturbation
                return (hidden,) + output[1:]
            else:
                output[:, -1, neuron_idx] = output[:, -1, neuron_idx] + perturbation
                return output
        return hook

    def _register_capture_hooks(self, layers_to_capture: List[int]):
        """Register hooks to capture activations at specified layers."""
        self._remove_hooks()
        for layer_idx in layers_to_capture:
            hook = self.layers[layer_idx].register_forward_hook(self._capture_hook(layer_idx))
            self._hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._captured_activations = {}

    def _get_baseline_context(self) -> Dict[str, torch.Tensor]:
        """Get a baseline input for probing."""
        text = "The quick brown fox jumps over the lazy dog."
        inputs = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _get_batched_context(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get batched input for parallel probing."""
        text = "The quick brown fox jumps over the lazy dog."
        inputs = self.tokenizer(text, return_tensors="pt")
        # Repeat for batch
        batched = {
            k: v.repeat(batch_size, 1).to(self.device)
            for k, v in inputs.items()
        }
        return batched

    def compute_baseline(self):
        """Compute baseline activations and magnitudes."""
        logger.info("Computing baseline activations...")

        inputs = self._get_baseline_context()

        # Capture all layers
        self._register_capture_hooks(list(range(self.n_layers)))

        with torch.no_grad():
            self.model(**inputs)

        self._baseline = {k: v.clone() for k, v in self._captured_activations.items()}

        # Compute typical magnitudes per layer
        self._baseline_magnitudes = {}
        for layer_idx, acts in self._baseline.items():
            self._baseline_magnitudes[layer_idx] = float(acts.abs().mean())

        self._remove_hooks()

        logger.info(f"Baseline computed. Mean magnitudes: {list(self._baseline_magnitudes.values())[:5]}...")

    def fuzz_layer_batched(
        self,
        source_layer: int,
        target_layers: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> LayerFuzzResult:
        """
        Fuzz all neurons in a layer using batched forward passes.

        Much faster than sequential: O(N/batch_size) instead of O(N) forward passes.

        Args:
            source_layer: Layer to fuzz all neurons in
            target_layers: Layers to measure response at
            show_progress: Whether to show progress bar

        Returns:
            LayerFuzzResult with connectivity matrices
        """
        if self._baseline is None:
            self.compute_baseline()

        if target_layers is None:
            target_layers = list(range(source_layer + 1, self.n_layers))

        n_neurons = self.hidden_dim
        if self.config.max_neurons_per_layer:
            n_neurons = min(n_neurons, self.config.max_neurons_per_layer)

        batch_size = min(self.config.batch_size, n_neurons)
        n_batches = (n_neurons + batch_size - 1) // batch_size

        # Initialize connectivity matrices
        connectivity = {
            layer: np.zeros((n_neurons, self.hidden_dim), dtype=np.float32)
            for layer in target_layers
        }
        top_k_connections = {layer: [[] for _ in range(n_neurons)] for layer in target_layers}

        # Perturbation magnitude scaled by typical activation
        perturbation = self.config.perturbation_scale * self._baseline_magnitudes[source_layer]

        # Process in batches
        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Fuzzing layer {source_layer} (batched)")

        for batch_idx in iterator:
            start_neuron = batch_idx * batch_size
            end_neuron = min(start_neuron + batch_size, n_neurons)
            current_batch_size = end_neuron - start_neuron

            # Neuron indices for this batch
            neuron_indices = torch.arange(start_neuron, end_neuron, device=self.device)

            # Get batched input
            inputs = self._get_batched_context(current_batch_size)

            # Register capture hooks for target layers
            self._register_capture_hooks(target_layers)

            # Add batched injection hook at source layer
            inject_hook = self.layers[source_layer].register_forward_hook(
                self._batched_inject_hook(source_layer, neuron_indices, perturbation)
            )
            self._hooks.append(inject_hook)

            # Run batched forward pass
            with torch.no_grad():
                self.model(**inputs)

            # Process responses for each neuron in batch
            for b, neuron_idx in enumerate(range(start_neuron, end_neuron)):
                for layer_idx in target_layers:
                    if layer_idx in self._captured_activations and layer_idx in self._baseline:
                        # Get this batch element's activation
                        perturbed = self._captured_activations[layer_idx][b]  # [hidden_dim]
                        baseline = self._baseline[layer_idx].squeeze()  # [hidden_dim]

                        # Response is the change from baseline
                        response = (perturbed - baseline).abs().cpu().numpy()

                        # Normalize by baseline magnitude
                        response = response / (self._baseline_magnitudes[layer_idx] + 1e-8)

                        # Store in connectivity matrix
                        connectivity[layer_idx][neuron_idx, :] = response

                        # Find top responders above threshold (early exit for zeros)
                        above_threshold = np.where(response > self.config.response_threshold)[0]
                        if len(above_threshold) > 0:
                            top_k_connections[layer_idx][neuron_idx] = above_threshold.tolist()

            self._remove_hooks()

        # Apply rank-based scoring if configured
        if self.config.use_rank_scoring:
            connectivity = self._apply_rank_scoring(connectivity)

        return LayerFuzzResult(
            source_layer=source_layer,
            n_neurons=n_neurons,
            connectivity=connectivity,
            top_k_connections=top_k_connections,
        )

    def _apply_rank_scoring(
        self,
        connectivity: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Convert raw scores to ranks within each layer.

        This normalizes across layers with different magnitude scales.
        """
        ranked = {}
        for layer_idx, conn in connectivity.items():
            # Rank each row (source neuron's connections)
            ranked_conn = np.zeros_like(conn)
            for i in range(conn.shape[0]):
                row = conn[i, :]
                # Only rank non-zero entries
                nonzero_mask = row > 0
                if nonzero_mask.any():
                    # Rank and normalize to [0, 1]
                    ranks = rankdata(row[nonzero_mask], method='average')
                    ranks = ranks / (len(ranks) + 1)  # Normalize
                    ranked_conn[i, nonzero_mask] = ranks
            ranked[layer_idx] = ranked_conn
        return ranked

    def fuzz_neuron(
        self,
        source_layer: int,
        source_neuron: int,
        target_layers: Optional[List[int]] = None,
    ) -> FuzzResult:
        """
        Fuzz a single neuron and measure downstream response (sequential mode).

        Args:
            source_layer: Layer containing neuron to perturb
            source_neuron: Index of neuron to perturb
            target_layers: Layers to measure response at (default: all downstream)

        Returns:
            FuzzResult with response magnitudes at each target layer
        """
        if self._baseline is None:
            self.compute_baseline()

        if target_layers is None:
            target_layers = list(range(source_layer + 1, self.n_layers))

        inputs = self._get_baseline_context()

        # Perturbation magnitude scaled by typical activation
        perturbation = self.config.perturbation_scale * self._baseline_magnitudes[source_layer]

        # Register capture hooks for target layers
        self._register_capture_hooks(target_layers)

        # Add injection hook at source layer
        inject_hook = self.layers[source_layer].register_forward_hook(
            self._inject_hook(source_layer, source_neuron, perturbation)
        )
        self._hooks.append(inject_hook)

        # Run perturbed forward pass
        with torch.no_grad():
            self.model(**inputs)

        # Compute response: difference from baseline, normalized
        responses = {}
        top_responders = {}

        for layer_idx in target_layers:
            if layer_idx in self._captured_activations and layer_idx in self._baseline:
                perturbed = self._captured_activations[layer_idx]
                baseline = self._baseline[layer_idx]

                # Response is the change from baseline
                response = (perturbed - baseline).abs().squeeze().cpu().numpy()

                # Normalize by baseline magnitude
                response = response / (self._baseline_magnitudes[layer_idx] + 1e-8)

                responses[layer_idx] = response

                # Find top responders above threshold (skip zeros early)
                above_threshold = np.where(response > self.config.response_threshold)[0]
                if len(above_threshold) > 0:
                    top_responders[layer_idx] = above_threshold.tolist()

        self._remove_hooks()

        return FuzzResult(
            source_layer=source_layer,
            source_neuron=source_neuron,
            responses=responses,
            top_responders=top_responders,
        )

    def fuzz_layer(
        self,
        source_layer: int,
        target_layers: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> LayerFuzzResult:
        """
        Fuzz all neurons in a layer sequentially (legacy mode).

        Use fuzz_layer_batched() for faster results.

        Args:
            source_layer: Layer to fuzz all neurons in
            target_layers: Layers to measure response at
            show_progress: Whether to show progress bar

        Returns:
            LayerFuzzResult with connectivity matrices
        """
        if self._baseline is None:
            self.compute_baseline()

        if target_layers is None:
            target_layers = list(range(source_layer + 1, self.n_layers))

        n_neurons = self.hidden_dim
        if self.config.max_neurons_per_layer:
            n_neurons = min(n_neurons, self.config.max_neurons_per_layer)

        # Initialize connectivity matrices
        connectivity = {
            layer: np.zeros((n_neurons, self.hidden_dim), dtype=np.float32)
            for layer in target_layers
        }
        top_k_connections = {layer: [] for layer in target_layers}

        # Fuzz each neuron
        iterator = range(n_neurons)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Fuzzing layer {source_layer} (sequential)")

        for neuron_idx in iterator:
            result = self.fuzz_neuron(source_layer, neuron_idx, target_layers)

            for layer_idx in target_layers:
                if layer_idx in result.responses:
                    connectivity[layer_idx][neuron_idx, :] = result.responses[layer_idx]
                    top_k_connections[layer_idx].append(result.top_responders.get(layer_idx, []))

        # Apply rank-based scoring if configured
        if self.config.use_rank_scoring:
            connectivity = self._apply_rank_scoring(connectivity)

        return LayerFuzzResult(
            source_layer=source_layer,
            n_neurons=n_neurons,
            connectivity=connectivity,
            top_k_connections=top_k_connections,
        )

    def fuzz_all_layers(
        self,
        show_progress: bool = True,
        use_batched: bool = True,
    ) -> Dict[int, LayerFuzzResult]:
        """
        Fuzz all layers (except last) and build complete forward connectivity map.

        Args:
            show_progress: Whether to show progress bar
            use_batched: Use batched fuzzing (faster) vs sequential

        Returns:
            Dict mapping source_layer -> LayerFuzzResult
        """
        layers_to_fuzz = self.config.layers_to_fuzz
        if layers_to_fuzz is None:
            layers_to_fuzz = list(range(self.n_layers - 1))  # All except last

        results = {}

        fuzz_fn = self.fuzz_layer_batched if use_batched else self.fuzz_layer

        for source_layer in layers_to_fuzz:
            logger.info(f"Fuzzing layer {source_layer}/{self.n_layers - 1}")
            results[source_layer] = fuzz_fn(source_layer, show_progress=show_progress)

        return results

    def compare_batched_vs_sequential(
        self,
        source_layer: int,
        n_neurons: int = 64,
    ) -> Dict:
        """
        Compare batched vs sequential fuzzing for validation.

        Returns correlation and timing comparison.
        """
        import time

        original_max = self.config.max_neurons_per_layer
        original_rank = self.config.use_rank_scoring
        self.config.max_neurons_per_layer = n_neurons
        self.config.use_rank_scoring = False  # Compare raw values

        # Reset baseline to ensure both use same baseline
        self._baseline = None

        # Sequential
        start = time.time()
        seq_result = self.fuzz_layer(source_layer, show_progress=False)
        seq_time = time.time() - start

        # Batched (use same baseline)
        start = time.time()
        batch_result = self.fuzz_layer_batched(source_layer, show_progress=False)
        batch_time = time.time() - start

        self.config.max_neurons_per_layer = original_max
        self.config.use_rank_scoring = original_rank

        # Compare connectivity matrices
        correlations = {}
        debug_info = {}
        for layer_idx in seq_result.connectivity:
            seq_conn = seq_result.connectivity[layer_idx].flatten()
            batch_conn = batch_result.connectivity[layer_idx].flatten()

            # Debug: check for variance
            seq_var = seq_conn.var()
            batch_var = batch_conn.var()
            seq_nonzero = (seq_conn > 0).sum()
            batch_nonzero = (batch_conn > 0).sum()

            debug_info[layer_idx] = {
                "seq_var": float(seq_var),
                "batch_var": float(batch_var),
                "seq_nonzero": int(seq_nonzero),
                "batch_nonzero": int(batch_nonzero),
            }

            # Only compute correlation if both have variance
            if seq_var > 0 and batch_var > 0:
                corr = np.corrcoef(seq_conn, batch_conn)[0, 1]
                correlations[layer_idx] = corr
            else:
                correlations[layer_idx] = float('nan')

        valid_corrs = [c for c in correlations.values() if not np.isnan(c)]
        mean_corr = np.mean(valid_corrs) if valid_corrs else float('nan')

        return {
            "sequential_time": seq_time,
            "batched_time": batch_time,
            "speedup": seq_time / batch_time,
            "correlations": correlations,
            "mean_correlation": mean_corr,
            "debug": debug_info,
        }

    def save_results(self, results: Dict[int, LayerFuzzResult], path: str):
        """Save fuzzing results to disk."""
        import json
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save connectivity matrices as npz
        for source_layer, result in results.items():
            for target_layer, conn_matrix in result.connectivity.items():
                np.save(
                    path / f"connectivity_L{source_layer}_to_L{target_layer}.npy",
                    conn_matrix
                )

        # Save metadata
        metadata = {
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "config": {
                "perturbation_scale": self.config.perturbation_scale,
                "response_threshold": self.config.response_threshold,
                "use_rank_scoring": self.config.use_rank_scoring,
                "batch_size": self.config.batch_size,
            },
            "layers_fuzzed": list(results.keys()),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Results saved to {path}")

    @classmethod
    def load_results(cls, path: str) -> Tuple[Dict, Dict[int, Dict[int, np.ndarray]]]:
        """Load fuzzing results from disk."""
        from pathlib import Path
        import json

        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        connectivity = {}
        for source_layer in metadata["layers_fuzzed"]:
            connectivity[source_layer] = {}
            for target_layer in range(source_layer + 1, metadata["n_layers"]):
                conn_path = path / f"connectivity_L{source_layer}_to_L{target_layer}.npy"
                if conn_path.exists():
                    connectivity[source_layer][target_layer] = np.load(conn_path)

        return metadata, connectivity
