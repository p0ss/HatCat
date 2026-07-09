"""
Reverse Tracer

Traces upstream sources for downstream activations using:
1. Attention path tracing: Q·K scores show which tokens were attended
2. MLP path tracing: Weight alignment with output direction

For attention: score_i = softmax(Q_j · K_i) · ||V_i||
For MLP: alignment of upstream with projection weights

Aggregate across many random contexts to get stable backward connectivity.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


@dataclass
class TraceConfig:
    """Configuration for reverse tracing."""
    n_contexts: int = 100  # Number of random contexts to aggregate
    top_k_sources: int = 50  # Top sources to keep per target
    min_attention_score: float = 0.01  # Min attention to count
    random_seed: int = 42
    context_length: int = 32  # Length of random contexts


@dataclass
class TraceResult:
    """Result of tracing a single target neuron."""
    target_layer: int
    target_neuron: int
    # Attention-mediated sources: layer -> (neuron_indices, scores)
    attention_sources: Dict[int, Tuple[np.ndarray, np.ndarray]]
    # MLP-mediated sources: layer -> (neuron_indices, scores)
    mlp_sources: Dict[int, Tuple[np.ndarray, np.ndarray]]


@dataclass
class LayerTraceResult:
    """Aggregated trace results for neurons in a layer."""
    target_layer: int
    n_neurons_traced: int
    # Backward connectivity: source_layer -> [n_target, n_source] scores
    attention_connectivity: Dict[int, np.ndarray]
    mlp_connectivity: Dict[int, np.ndarray]


class ReverseTracer:
    """
    Reverse tracer for discovering upstream connectivity.

    Usage:
        tracer = ReverseTracer(model, tokenizer, config)
        result = tracer.trace_layer(target_layer=10, n_contexts=100)
        # result.attention_connectivity[5] gives backward attention influence
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[TraceConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TraceConfig()

        # Get model structure
        self.layers = self._get_layers()
        self.n_layers = len(self.layers)
        self.hidden_dim = self._get_hidden_dim()
        self.n_heads = self._get_n_heads()
        self.head_dim = self.hidden_dim // self.n_heads

        # Storage for captured attention weights
        self._attention_weights: Dict[int, torch.Tensor] = {}
        self._value_norms: Dict[int, torch.Tensor] = {}
        self._hooks: List = []

        # Random contexts for aggregation
        self._random_contexts: List[Dict] = []

        logger.info(
            f"ReverseTracer initialized: {self.n_layers} layers, "
            f"hidden_dim={self.hidden_dim}, n_heads={self.n_heads}"
        )

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
        """Get hidden dimension from model config."""
        if hasattr(self.model, 'config'):
            config = self.model.config
            if hasattr(config, 'hidden_size'):
                return config.hidden_size
            # Nested config (e.g., Gemma 3 uses text_config)
            if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
                return config.text_config.hidden_size
        raise ValueError("Cannot determine hidden dimension")

    def _get_n_heads(self) -> int:
        """Get number of attention heads from model config."""
        if hasattr(self.model, 'config'):
            config = self.model.config
            if hasattr(config, 'num_attention_heads'):
                return config.num_attention_heads
            # Nested config (e.g., Gemma 3 uses text_config)
            if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_attention_heads'):
                return config.text_config.num_attention_heads
        raise ValueError("Cannot determine number of attention heads")

    def _get_attention_module(self, layer_idx: int):
        """Get the attention module from a layer."""
        layer = self.layers[layer_idx]
        # Try common attribute names
        for attr in ['self_attn', 'attention', 'attn']:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise AttributeError(f"Cannot find attention module in layer {layer_idx}")

    def _attention_hook(self, layer_idx: int) -> Callable:
        """Create hook to capture attention weights and value norms."""
        def hook(module, input, output):
            # output is typically (hidden_states, attention_weights, ...)
            # or just hidden_states depending on model
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]  # [batch, heads, seq, seq]
                if attn_weights is not None:
                    self._attention_weights[layer_idx] = attn_weights.detach()
        return hook

    def _qkv_hook(self, layer_idx: int) -> Callable:
        """
        Create hook to capture attention weights AND value norms for QKV backward tracing.

        Captures both the attention pattern (softmax(Q·K)) and the value magnitudes (||V||)
        to compute: score_i = attn_weight[j,i] * ||V_i||
        """
        def hook(module, input, output):
            # Capture attention weights from output
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]  # [batch, heads, seq, seq]
                if attn_weights is not None:
                    self._attention_weights[layer_idx] = attn_weights.detach()

        return hook

    def _layer_pre_hook(self, layer_idx: int) -> Callable:
        """
        Create pre-hook to capture hidden states entering a layer (for value norm computation).
        Pre-hook receives (module, input) before forward pass.
        """
        def hook(module, input):
            # Get hidden states from layer input
            hidden_states = None
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0]

            if hidden_states is None:
                return

            # Find v_proj weight in the attention module
            try:
                attn_module = self._get_attention_module(layer_idx)
                v_proj = None
                for name, param in attn_module.named_parameters():
                    if 'v_proj' in name and 'weight' in name:
                        v_proj = param
                        break

                if v_proj is not None:
                    with torch.no_grad():
                        # V = hidden_states @ W_v.T  (if W_v is [out, in])
                        # hidden_states: [batch, seq, hidden_dim]
                        # v_proj: [n_heads * head_dim, hidden_dim] typically

                        # Project to get V
                        V = torch.matmul(hidden_states.float(), v_proj.T.float())

                        # Compute norm per position (across the value dimension)
                        value_norms = V.norm(dim=-1)  # [batch, seq]
                        self._value_norms[layer_idx] = value_norms.detach()
            except Exception as e:
                logger.debug(f"Could not compute value norms for layer {layer_idx}: {e}")

        return hook

    def _register_qkv_hooks(self, layers: List[int]):
        """Register hooks to capture attention weights and value norms."""
        self._remove_hooks()

        for layer_idx in layers:
            try:
                # Pre-hook the layer to capture hidden states for value norm computation
                layer_pre_hook = self.layers[layer_idx].register_forward_pre_hook(
                    self._layer_pre_hook(layer_idx)
                )
                self._hooks.append(layer_pre_hook)

                # Hook the attention module for attention weights
                attn_module = self._get_attention_module(layer_idx)
                attn_hook = attn_module.register_forward_hook(self._qkv_hook(layer_idx))
                self._hooks.append(attn_hook)
            except AttributeError as e:
                logger.warning(f"Could not hook QKV at layer {layer_idx}: {e}")

    def _generate_random_contexts(self, n_contexts: int):
        """Generate random token sequences as contexts."""
        logger.info(f"Generating {n_contexts} random contexts...")

        torch.manual_seed(self.config.random_seed)
        device = next(self.model.parameters()).device

        # Get vocab size
        vocab_size = self.tokenizer.vocab_size

        self._random_contexts = []
        for _ in range(n_contexts):
            # Generate random token ids (avoiding special tokens)
            random_ids = torch.randint(
                1000, vocab_size - 1000,  # Avoid special tokens at edges
                (1, self.config.context_length),
                device=device
            )
            attention_mask = torch.ones_like(random_ids)
            self._random_contexts.append({
                'input_ids': random_ids,
                'attention_mask': attention_mask,
            })

    def _register_attention_hooks(self, layers: List[int]):
        """Register hooks to capture attention at specified layers."""
        self._remove_hooks()

        for layer_idx in layers:
            try:
                attn_module = self._get_attention_module(layer_idx)
                # Need to capture with output_attentions=True
                hook = attn_module.register_forward_hook(self._attention_hook(layer_idx))
                self._hooks.append(hook)
            except AttributeError as e:
                logger.warning(f"Could not hook attention at layer {layer_idx}: {e}")

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._attention_weights = {}
        self._value_norms = {}

    def trace_attention_path(
        self,
        target_layer: int,
        source_layers: Optional[List[int]] = None,
        n_contexts: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Trace attention-mediated connectivity from source layers to target layer.

        For each context, captures attention weights and aggregates to find
        which upstream positions consistently influence the target.

        Args:
            target_layer: Layer to trace back from
            source_layers: Layers to trace to (default: all previous)
            n_contexts: Number of contexts to aggregate over

        Returns:
            Dict mapping source_layer -> [n_heads, seq, seq] average attention
        """
        if source_layers is None:
            source_layers = list(range(target_layer))

        if n_contexts is None:
            n_contexts = self.config.n_contexts

        if len(self._random_contexts) < n_contexts:
            self._generate_random_contexts(n_contexts)

        # We need to trace through the transformer's residual stream
        # The direct attention influence from layer L to layer L+1 is via attention at L+1
        # For non-adjacent layers, influence is indirect

        # For now, capture attention at target layer across contexts
        aggregated_attention = None

        iterator = range(n_contexts)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Tracing to layer {target_layer}")

        for ctx_idx in iterator:
            context = self._random_contexts[ctx_idx]

            self._register_attention_hooks([target_layer])

            with torch.no_grad():
                # Enable attention output
                try:
                    self.model(**context, output_attentions=True)
                except TypeError:
                    # Some models don't support output_attentions in forward
                    self.model(**context)

            if target_layer in self._attention_weights:
                attn = self._attention_weights[target_layer]  # [1, heads, seq, seq]
                attn_np = attn.squeeze(0).cpu().numpy()  # [heads, seq, seq]

                if aggregated_attention is None:
                    aggregated_attention = attn_np
                else:
                    aggregated_attention += attn_np

        self._remove_hooks()

        if aggregated_attention is not None:
            aggregated_attention /= n_contexts

        # Return attention pattern at target layer
        # This shows which positions (tokens) attend to which
        # For neuron-level analysis, we'd need to go deeper into the computation
        return {target_layer: aggregated_attention}

    def trace_qkv_backward(
        self,
        target_layer: int,
        n_contexts: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Trace attention-mediated backward connectivity using QKV inspection.

        Implements the formula from MODEL_TOPOLOGY.md:
            score_i = softmax(Q_j · K_i) · ||V_i||

        This captures how much each upstream position i contributed to
        downstream position j via attention, weighted by value magnitude.

        Args:
            target_layer: Layer to trace backward from
            n_contexts: Number of random contexts to aggregate over

        Returns:
            Dict with:
                - 'attention_scores': [n_heads, seq, seq] - raw attention patterns
                - 'value_weighted_scores': [seq, seq] - attention weighted by ||V||
                - 'per_head_weighted': [n_heads, seq, seq] - per-head weighted scores
        """
        if n_contexts is None:
            n_contexts = self.config.n_contexts

        if len(self._random_contexts) < n_contexts:
            self._generate_random_contexts(n_contexts)

        # Aggregators
        aggregated_attention = None  # [heads, seq, seq]
        aggregated_weighted = None   # [heads, seq, seq] - attention * ||V||

        iterator = range(n_contexts)
        if show_progress:
            iterator = tqdm(iterator, desc=f"QKV backward trace layer {target_layer}")

        for ctx_idx in iterator:
            context = self._random_contexts[ctx_idx]

            # Register QKV hooks to capture both attention and value norms
            self._register_qkv_hooks([target_layer])

            with torch.no_grad():
                try:
                    self.model(**context, output_attentions=True)
                except TypeError:
                    self.model(**context)

            if target_layer in self._attention_weights:
                # attention: [batch=1, heads, seq, seq]
                attn = self._attention_weights[target_layer].squeeze(0)  # [heads, seq, seq]
                attn_np = attn.cpu().numpy()

                if aggregated_attention is None:
                    aggregated_attention = attn_np
                else:
                    aggregated_attention += attn_np

                # Weight by value norms if available
                if target_layer in self._value_norms:
                    # value_norms: [batch=1, seq] -> [seq]
                    v_norms = self._value_norms[target_layer].squeeze(0).cpu().numpy()

                    # score_i = attn[j, i] * ||V_i||
                    # Broadcast: attn is [heads, seq_j, seq_i], v_norms is [seq_i]
                    weighted = attn_np * v_norms[np.newaxis, np.newaxis, :]  # [heads, seq, seq]

                    if aggregated_weighted is None:
                        aggregated_weighted = weighted
                    else:
                        aggregated_weighted += weighted

            self._remove_hooks()

        # Average across contexts
        if aggregated_attention is not None:
            aggregated_attention /= n_contexts

        if aggregated_weighted is not None:
            aggregated_weighted /= n_contexts
            # Also compute head-averaged version
            value_weighted_avg = aggregated_weighted.mean(axis=0)  # [seq, seq]
        else:
            value_weighted_avg = None

        return {
            'attention_scores': aggregated_attention,
            'value_weighted_scores': value_weighted_avg,
            'per_head_weighted': aggregated_weighted,
        }

    def compute_qkv_connectivity(
        self,
        target_layers: Optional[List[int]] = None,
        n_contexts: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Compute QKV-based backward connectivity for multiple layers.

        Returns position-level connectivity matrices that can be used
        alongside MLP connectivity for full backward tracing.

        Args:
            target_layers: Layers to trace (default: all except first)
            n_contexts: Contexts for aggregation

        Returns:
            Dict mapping layer -> [seq, seq] value-weighted attention scores
        """
        if target_layers is None:
            target_layers = list(range(1, self.n_layers))

        connectivity = {}

        for layer_idx in target_layers:
            if show_progress:
                logger.info(f"Computing QKV connectivity for layer {layer_idx}")

            result = self.trace_qkv_backward(
                layer_idx,
                n_contexts=n_contexts,
                show_progress=show_progress,
            )

            if result['value_weighted_scores'] is not None:
                connectivity[layer_idx] = result['value_weighted_scores']
            elif result['attention_scores'] is not None:
                # Fallback to unweighted if value norms not captured
                connectivity[layer_idx] = result['attention_scores'].mean(axis=0)

        return connectivity

    def compute_mlp_influence(
        self,
        target_layer: int,
        target_neuron: int,
    ) -> Dict[int, np.ndarray]:
        """
        Compute MLP-mediated influence on a target neuron.

        Uses weight matrix alignment: neurons that strongly project to
        target neuron's position in the residual stream.

        This is a static analysis based on weights, not activations.

        Args:
            target_layer: Layer containing target neuron
            target_neuron: Index of target neuron

        Returns:
            Dict mapping source_layer -> influence scores per source neuron
        """
        influences = {}

        # The residual stream means each layer's output adds to previous
        # MLP at layer L projects: hidden -> 4*hidden -> hidden
        # We want to find which MLP neurons strongly influence target position

        for layer_idx in range(target_layer):
            layer = self.layers[layer_idx]

            # Find MLP weights
            mlp = None
            for attr in ['mlp', 'feed_forward', 'ffn']:
                if hasattr(layer, attr):
                    mlp = getattr(layer, attr)
                    break

            if mlp is None:
                continue

            # Get output projection weights
            # This maps from intermediate dim back to hidden dim
            out_proj = None
            for name, param in mlp.named_parameters():
                if 'down_proj' in name or 'out_proj' in name or 'c_proj' in name:
                    if len(param.shape) == 2:
                        out_proj = param
                        break
                # Some models use fc2
                if 'fc2' in name and 'weight' in name:
                    out_proj = param
                    break

            if out_proj is None:
                continue

            # out_proj: [hidden_dim, intermediate_dim] or [intermediate_dim, hidden_dim]
            # We want: how much does each intermediate neuron contribute to target_neuron

            # The contribution to output neuron j is: sum_i(out_proj[j, i] * intermediate[i])
            # So out_proj[target_neuron, :] gives the influence of each intermediate neuron

            with torch.no_grad():
                if out_proj.shape[0] == self.hidden_dim:
                    # [hidden, intermediate]
                    influence = out_proj[target_neuron, :].abs().cpu().numpy()
                else:
                    # [intermediate, hidden]
                    influence = out_proj[:, target_neuron].abs().cpu().numpy()

            influences[layer_idx] = influence

        return influences

    def compute_mlp_connectivity_vectorized(
        self,
        target_layer: int,
        source_layers: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Compute MLP connectivity for all neurons at once (vectorized).

        Returns full connectivity matrices without per-neuron loops.
        Much faster than calling compute_mlp_influence per neuron.

        Args:
            target_layer: Layer to trace back from
            source_layers: Layers to trace to (default: all previous)

        Returns:
            Dict mapping source_layer -> [hidden_dim, hidden_dim] connectivity
        """
        if source_layers is None:
            source_layers = list(range(target_layer))

        connectivity = {}

        for layer_idx in source_layers:
            layer = self.layers[layer_idx]

            # Find MLP module
            mlp = None
            for attr in ['mlp', 'feed_forward', 'ffn']:
                if hasattr(layer, attr):
                    mlp = getattr(layer, attr)
                    break

            if mlp is None:
                continue

            # Get output projection weights
            out_proj = None
            for name, param in mlp.named_parameters():
                if 'down_proj' in name or 'out_proj' in name or 'c_proj' in name:
                    if len(param.shape) == 2:
                        out_proj = param
                        break
                if 'fc2' in name and 'weight' in name:
                    out_proj = param
                    break

            if out_proj is None:
                continue

            with torch.no_grad():
                # out_proj: [hidden_dim, intermediate_dim] or [intermediate_dim, hidden_dim]
                weight_abs = out_proj.abs().cpu().numpy()

                if out_proj.shape[0] == self.hidden_dim:
                    # [hidden, intermediate] - rows are target neurons
                    # Need to map intermediate -> hidden
                    if out_proj.shape[1] == self.hidden_dim:
                        connectivity[layer_idx] = weight_abs
                    else:
                        # Average pool intermediate dim to hidden dim
                        intermediate_dim = out_proj.shape[1]
                        ratio = intermediate_dim / self.hidden_dim
                        pooled = np.zeros((self.hidden_dim, self.hidden_dim), dtype=np.float32)
                        for i in range(self.hidden_dim):
                            start = int(i * ratio)
                            end = int((i + 1) * ratio)
                            pooled[:, i] = weight_abs[:, start:end].mean(axis=1)
                        connectivity[layer_idx] = pooled
                else:
                    # [intermediate, hidden] - columns are target neurons, transpose
                    weight_t = weight_abs.T  # Now [hidden, intermediate]
                    if weight_t.shape[1] == self.hidden_dim:
                        connectivity[layer_idx] = weight_t
                    else:
                        intermediate_dim = weight_t.shape[1]
                        ratio = intermediate_dim / self.hidden_dim
                        pooled = np.zeros((self.hidden_dim, self.hidden_dim), dtype=np.float32)
                        for i in range(self.hidden_dim):
                            start = int(i * ratio)
                            end = int((i + 1) * ratio)
                            pooled[:, i] = weight_t[:, start:end].mean(axis=1)
                        connectivity[layer_idx] = pooled

        return connectivity

    def trace_layer_backward(
        self,
        target_layer: int,
        source_layers: Optional[List[int]] = None,
        n_contexts: int = 100,
        n_neurons: Optional[int] = None,
        show_progress: bool = True,
        use_qkv_tracing: bool = True,
    ) -> LayerTraceResult:
        """
        Trace backward connectivity for neurons in a layer.

        Combines attention tracing (dynamic, context-dependent) with
        MLP influence analysis (static, weight-based).

        Args:
            target_layer: Layer to trace back from
            source_layers: Layers to trace to
            n_contexts: Contexts for attention aggregation
            n_neurons: Number of neurons to trace (None = all)
            use_qkv_tracing: Use QKV backward tracing (value-weighted attention)

        Returns:
            LayerTraceResult with backward connectivity matrices
        """
        if source_layers is None:
            source_layers = list(range(target_layer))

        if n_neurons is None:
            n_neurons = self.hidden_dim

        # Attention-based backward connectivity
        if use_qkv_tracing:
            # QKV tracing: score_i = softmax(Q_j · K_i) · ||V_i||
            # This weights attention by value magnitude for better backward attribution
            if show_progress:
                logger.info(f"Computing QKV backward connectivity for layer {target_layer}")
            qkv_result = self.trace_qkv_backward(
                target_layer, n_contexts=n_contexts, show_progress=False
            )
            # Use value-weighted scores, falling back to raw attention
            attn_connectivity = qkv_result.get('value_weighted_scores')
            if attn_connectivity is None:
                attn_connectivity = qkv_result.get('attention_scores')
                if attn_connectivity is not None:
                    attn_connectivity = attn_connectivity.mean(axis=0)  # Average heads
        else:
            # Legacy: raw attention capture (token-level, not value-weighted)
            attn_result = self.trace_attention_path(
                target_layer, source_layers, n_contexts, show_progress=False
            )
            attn_connectivity = attn_result.get(target_layer)

        # MLP-based backward connectivity (vectorized - no per-neuron loop)
        if show_progress:
            logger.info(f"Computing MLP connectivity for layer {target_layer} (vectorized)")
        mlp_connectivity = self.compute_mlp_connectivity_vectorized(target_layer, source_layers)

        # Ensure all source layers have entries (even if empty)
        for layer in source_layers:
            if layer not in mlp_connectivity:
                mlp_connectivity[layer] = np.zeros((self.hidden_dim, self.hidden_dim))

        return LayerTraceResult(
            target_layer=target_layer,
            n_neurons_traced=n_neurons,
            attention_connectivity={target_layer: attn_connectivity},
            mlp_connectivity=mlp_connectivity,
        )

    def trace_all_layers(
        self,
        n_contexts: int = 100,
        show_progress: bool = True,
    ) -> Dict[int, LayerTraceResult]:
        """
        Trace backward connectivity for all layers (except first).

        Returns:
            Dict mapping target_layer -> LayerTraceResult
        """
        results = {}

        for target_layer in range(1, self.n_layers):
            logger.info(f"Tracing layer {target_layer}/{self.n_layers - 1}")
            results[target_layer] = self.trace_layer_backward(
                target_layer,
                n_contexts=n_contexts,
                show_progress=show_progress,
            )

        return results

    def save_results(self, results: Dict[int, LayerTraceResult], path: str):
        """Save trace results to disk."""
        import json
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for target_layer, result in results.items():
            # Save MLP connectivity
            for source_layer, conn in result.mlp_connectivity.items():
                np.save(
                    path / f"mlp_connectivity_L{source_layer}_to_L{target_layer}.npy",
                    conn
                )

        # Save metadata
        metadata = {
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "config": {
                "n_contexts": self.config.n_contexts,
            },
            "layers_traced": list(results.keys()),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Results saved to {path}")
