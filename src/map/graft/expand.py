"""
Expand mode for true dimension expansion in scions.

This module handles the architecture-dependent logic for actually adding
a new neuron (expanding hidden_dim by 1) rather than just applying deltas.

When we expand hidden_dim from D to D+1, we must modify:
1. Every weight matrix that has D as an input or output dimension
2. Embeddings (input and output)
3. Layer norms (if they have per-dimension parameters)
4. Position embeddings (if present)

Architecture Taxonomy:
- **Llama-family**: Llama 2/3, Mistral, Apertus, Qwen2
- **Gemma-family**: Gemma 1/2
- **MPT/Falcon**: Different MLP structure
- **MoE**: Mixtral, Qwen2-MoE (expert routing adds complexity)

Key differences:
- MLP structure: some have gate_proj (GLU variants), some don't
- Attention: GQA (grouped query) vs MHA (multi-head) vs MQA (multi-query)
- Normalization: pre-norm vs post-norm, RMSNorm vs LayerNorm
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MLPType(Enum):
    """MLP architecture types."""
    STANDARD = "standard"      # up_proj, down_proj only (GPT-2 style)
    GLU = "glu"               # up_proj, gate_proj, down_proj (Llama/SwiGLU)
    PARALLEL = "parallel"      # Parallel attention + MLP (some GPT variants)


class AttentionType(Enum):
    """Attention architecture types."""
    MHA = "mha"               # Multi-head attention (all heads independent)
    GQA = "gqa"               # Grouped query attention (K/V heads < Q heads)
    MQA = "mqa"               # Multi-query attention (1 K/V head)


class NormType(Enum):
    """Normalization types."""
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"


@dataclass
class ArchitectureSpec:
    """
    Specification for a transformer architecture.

    This defines exactly which weight matrices need to be expanded
    when adding a new neuron (expanding hidden_dim).
    """
    name: str
    family: str  # llama, gemma, gpt2, moe, etc.

    # Core dimensions
    hidden_size: int
    intermediate_size: int  # MLP hidden dim (often 4x hidden_size or similar)
    num_attention_heads: int
    num_key_value_heads: int  # For GQA; equals num_attention_heads for MHA
    head_dim: int  # hidden_size // num_attention_heads typically

    # Architecture types
    mlp_type: MLPType = MLPType.GLU
    attention_type: AttentionType = AttentionType.GQA
    norm_type: NormType = NormType.RMS_NORM

    # Component names (architecture-specific)
    component_names: Dict[str, str] = field(default_factory=dict)

    # Whether this is a MoE model
    is_moe: bool = False
    num_experts: int = 1

    # Layer access pattern
    layers_attr: str = "model.layers"  # How to access layers from model

    def __post_init__(self):
        # Set defaults for component names if not provided
        if not self.component_names:
            self.component_names = self._default_component_names()

    def _default_component_names(self) -> Dict[str, str]:
        """Default Llama-style component names."""
        return {
            # Embeddings
            "embed_tokens": "model.embed_tokens",
            "lm_head": "lm_head",

            # Per-layer attention
            "q_proj": "self_attn.q_proj",
            "k_proj": "self_attn.k_proj",
            "v_proj": "self_attn.v_proj",
            "o_proj": "self_attn.o_proj",

            # Per-layer MLP
            "up_proj": "mlp.up_proj",
            "gate_proj": "mlp.gate_proj",  # Only for GLU
            "down_proj": "mlp.down_proj",

            # Per-layer norms
            "input_norm": "input_layernorm",
            "post_attn_norm": "post_attention_layernorm",
        }


@dataclass
class ExpansionTarget:
    """
    A single weight matrix that needs expansion.

    Specifies which dimension to expand and how.
    """
    component_path: str  # e.g., "model.layers.0.mlp.up_proj"
    expand_dim: int      # 0 for output (rows), 1 for input (cols)
    current_size: int    # Current size of that dimension
    new_size: int        # Size after expansion (current + 1)

    # For biased initialization
    init_value: float = 0.0
    init_from_bias: Optional[torch.Tensor] = None

    # For tied weights (embedding/lm_head)
    tied_to: Optional[str] = None


@dataclass
class ExpansionPlan:
    """
    Complete plan for expanding hidden_dim by 1.

    Lists all weight matrices that need modification and how.
    """
    architecture: ArchitectureSpec
    old_hidden_dim: int
    new_hidden_dim: int

    # All targets that need expansion
    targets: List[ExpansionTarget] = field(default_factory=list)

    # Embeddings (special handling)
    embedding_targets: List[ExpansionTarget] = field(default_factory=list)

    # Norms (special handling - may need new parameter)
    norm_targets: List[ExpansionTarget] = field(default_factory=list)

    # MoE expert handling
    expert_targets: Dict[int, List[ExpansionTarget]] = field(default_factory=dict)

    def total_new_parameters(self) -> int:
        """Count how many new parameters will be added."""
        total = 0
        for target in self.targets + self.embedding_targets + self.norm_targets:
            # Each expansion adds a row or column
            if target.expand_dim == 0:
                # Adding a row: new_params = num_cols
                total += target.current_size  # The other dimension
            else:
                # Adding a column: new_params = num_rows
                total += target.current_size
        return total


# =============================================================================
# Architecture Detection
# =============================================================================

def detect_architecture(model: nn.Module) -> ArchitectureSpec:
    """
    Detect the architecture of a model from its config and structure.

    Returns an ArchitectureSpec with all the information needed for expansion.
    """
    config = model.config

    # Get basic dimensions
    hidden_size = config.hidden_size
    intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
    num_attention_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
    head_dim = hidden_size // num_attention_heads

    # Detect architecture family
    arch_type = getattr(config, 'model_type', 'unknown')

    # Determine attention type
    if num_kv_heads == 1:
        attn_type = AttentionType.MQA
    elif num_kv_heads < num_attention_heads:
        attn_type = AttentionType.GQA
    else:
        attn_type = AttentionType.MHA

    # Detect MLP type by checking for gate_proj
    mlp_type = MLPType.STANDARD
    if hasattr(config, 'hidden_act'):
        act = config.hidden_act.lower() if isinstance(config.hidden_act, str) else ""
        if 'silu' in act or 'swiglu' in act or 'gelu' in act:
            mlp_type = MLPType.GLU

    # Try to detect from model structure
    layers = _get_model_layers(model)
    if layers and len(layers) > 0:
        layer = layers[0]
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            if hasattr(mlp, 'gate_proj') or hasattr(mlp, 'w3'):
                mlp_type = MLPType.GLU

    # Detect norm type
    norm_type = NormType.RMS_NORM  # Default for modern models
    if hasattr(config, 'layer_norm_eps'):
        # Could be either, check actual module type
        if layers and len(layers) > 0:
            layer = layers[0]
            norm = getattr(layer, 'input_layernorm', None) or getattr(layer, 'ln_1', None)
            if norm is not None and 'LayerNorm' in type(norm).__name__:
                norm_type = NormType.LAYER_NORM

    # Check for MoE
    is_moe = False
    num_experts = 1
    if hasattr(config, 'num_local_experts'):
        is_moe = True
        num_experts = config.num_local_experts
    elif hasattr(config, 'num_experts'):
        is_moe = True
        num_experts = config.num_experts

    # Build component names based on architecture
    component_names = _get_component_names(model, arch_type)

    # Detect layers attribute
    layers_attr = _detect_layers_attr(model)

    return ArchitectureSpec(
        name=arch_type,
        family=_get_family(arch_type),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        mlp_type=mlp_type,
        attention_type=attn_type,
        norm_type=norm_type,
        component_names=component_names,
        is_moe=is_moe,
        num_experts=num_experts,
        layers_attr=layers_attr
    )


def _get_family(arch_type: str) -> str:
    """Map architecture type to family."""
    families = {
        # Llama family
        "llama": "llama",
        "mistral": "llama",
        "mixtral": "llama",  # MoE but same structure
        "qwen2": "llama",
        "qwen2_moe": "llama",

        # Gemma family
        "gemma": "gemma",
        "gemma2": "gemma",

        # GPT-2 family
        "gpt2": "gpt2",
        "gpt_neox": "gpt_neox",

        # Falcon family
        "falcon": "falcon",

        # MPT family
        "mpt": "mpt",

        # Phi family
        "phi": "phi",
        "phi3": "phi",
    }
    return families.get(arch_type.lower(), "unknown")


def _get_component_names(model: nn.Module, arch_type: str) -> Dict[str, str]:
    """Get architecture-specific component names."""

    # Llama/Mistral/Qwen2 style (most common)
    llama_style = {
        "embed_tokens": "model.embed_tokens",
        "lm_head": "lm_head",
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "up_proj": "mlp.up_proj",
        "gate_proj": "mlp.gate_proj",
        "down_proj": "mlp.down_proj",
        "input_norm": "input_layernorm",
        "post_attn_norm": "post_attention_layernorm",
    }

    # Gemma style (slightly different names)
    gemma_style = {
        "embed_tokens": "model.embed_tokens",
        "lm_head": "lm_head",  # Often tied to embed_tokens
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "up_proj": "mlp.up_proj",
        "gate_proj": "mlp.gate_proj",
        "down_proj": "mlp.down_proj",
        "input_norm": "input_layernorm",
        "post_attn_norm": "post_attention_layernorm",
        "pre_feedforward_norm": "pre_feedforward_layernorm",  # Gemma 2 specific
        "post_feedforward_norm": "post_feedforward_layernorm",  # Gemma 2 specific
    }

    # GPT-2 style
    gpt2_style = {
        "embed_tokens": "transformer.wte",
        "lm_head": "lm_head",
        "qkv_proj": "attn.c_attn",  # Fused QKV
        "o_proj": "attn.c_proj",
        "up_proj": "mlp.c_fc",
        "down_proj": "mlp.c_proj",
        "input_norm": "ln_1",
        "post_attn_norm": "ln_2",
    }

    # GPT-NeoX style
    neox_style = {
        "embed_tokens": "gpt_neox.embed_in",
        "lm_head": "embed_out",
        "qkv_proj": "attention.query_key_value",  # Fused QKV
        "o_proj": "attention.dense",
        "up_proj": "mlp.dense_h_to_4h",
        "down_proj": "mlp.dense_4h_to_h",
        "input_norm": "input_layernorm",
        "post_attn_norm": "post_attention_layernorm",
    }

    arch_lower = arch_type.lower()

    if arch_lower in ['gemma', 'gemma2']:
        return gemma_style
    elif arch_lower in ['gpt2']:
        return gpt2_style
    elif arch_lower in ['gpt_neox', 'pythia']:
        return neox_style
    else:
        # Default to Llama style (covers Mistral, Qwen2, etc.)
        return llama_style


def _detect_layers_attr(model: nn.Module) -> str:
    """Detect how to access transformer layers."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return "model.layers"
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return "transformer.h"
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return "gpt_neox.layers"
    elif hasattr(model, 'layers'):
        return "layers"
    else:
        return "model.layers"  # Default


def _get_model_layers(model: nn.Module) -> List[nn.Module]:
    """Get the list of transformer layers."""
    if hasattr(model, 'model'):
        m = model.model
    else:
        m = model

    if hasattr(m, 'layers'):
        return list(m.layers)
    elif hasattr(m, 'h'):
        return list(m.h)

    return []


# =============================================================================
# Expansion Planning
# =============================================================================

def plan_expansion(
    model: nn.Module,
    scion: Any,  # Scion from scion.py
    target_layers: Optional[List[int]] = None
) -> ExpansionPlan:
    """
    Create a complete plan for expanding hidden_dim by 1.

    This analyzes the model architecture and creates a list of all
    weight matrices that need to be modified.

    Args:
        model: The model to expand
        scion: The scion being applied (contains biases for initialization)
        target_layers: If specified, only expand these layers (for efficiency)

    Returns:
        ExpansionPlan with all targets listed
    """
    arch = detect_architecture(model)
    old_dim = arch.hidden_size
    new_dim = old_dim + 1

    plan = ExpansionPlan(
        architecture=arch,
        old_hidden_dim=old_dim,
        new_hidden_dim=new_dim
    )

    # Get layers to process
    layers = _get_model_layers(model)
    num_layers = len(layers)

    if target_layers is None:
        target_layers = list(range(num_layers))

    # Plan embeddings expansion
    plan.embedding_targets = _plan_embedding_expansion(model, arch, old_dim, new_dim)

    # Plan per-layer expansion
    for layer_idx in target_layers:
        if layer_idx >= num_layers:
            continue

        layer_targets = _plan_layer_expansion(
            model, arch, layer_idx, old_dim, new_dim, scion
        )
        plan.targets.extend(layer_targets)

        # Plan norm expansion
        norm_targets = _plan_norm_expansion(
            model, arch, layer_idx, old_dim, new_dim
        )
        plan.norm_targets.extend(norm_targets)

        # MoE expert expansion
        if arch.is_moe:
            expert_targets = _plan_moe_expansion(
                model, arch, layer_idx, old_dim, new_dim, scion
            )
            plan.expert_targets[layer_idx] = expert_targets

    return plan


def _plan_embedding_expansion(
    model: nn.Module,
    arch: ArchitectureSpec,
    old_dim: int,
    new_dim: int
) -> List[ExpansionTarget]:
    """Plan expansion of embedding layers."""
    targets = []

    # Input embeddings: shape (vocab_size, hidden_dim)
    # Need to expand dim 1 (columns)
    embed_path = arch.component_names.get("embed_tokens", "model.embed_tokens")
    embed = _get_nested_attr(model, embed_path)
    if embed is not None:
        vocab_size = embed.weight.shape[0]
        targets.append(ExpansionTarget(
            component_path=embed_path + ".weight",
            expand_dim=1,  # Expand columns
            current_size=old_dim,
            new_size=new_dim,
            init_value=0.0
        ))

    # Output head: shape (vocab_size, hidden_dim)
    # Need to expand dim 1 (columns)
    lm_head_path = arch.component_names.get("lm_head", "lm_head")
    lm_head = _get_nested_attr(model, lm_head_path)

    # Check if tied to embeddings
    tied = False
    if embed is not None and lm_head is not None:
        if embed.weight is lm_head.weight:
            tied = True

    if lm_head is not None and not tied:
        targets.append(ExpansionTarget(
            component_path=lm_head_path + ".weight",
            expand_dim=1,  # Expand columns (input dim for output projection)
            current_size=old_dim,
            new_size=new_dim,
            init_value=0.0,
            tied_to=embed_path + ".weight" if tied else None
        ))

    return targets


def _plan_layer_expansion(
    model: nn.Module,
    arch: ArchitectureSpec,
    layer_idx: int,
    old_dim: int,
    new_dim: int,
    scion: Any
) -> List[ExpansionTarget]:
    """Plan expansion for a single transformer layer."""
    targets = []

    # Get layer path
    layers_attr = arch.layers_attr
    layer_path = f"{layers_attr}.{layer_idx}"

    # === Attention projections ===
    # Q, K, V projections: (num_heads * head_dim, hidden_dim)
    # Input is hidden_dim (expand cols)

    # Q projection
    q_path = f"{layer_path}.{arch.component_names['q_proj']}"
    targets.append(ExpansionTarget(
        component_path=q_path + ".weight",
        expand_dim=1,  # Input dimension
        current_size=old_dim,
        new_size=new_dim,
        init_value=0.0
    ))

    # K projection - may be smaller for GQA
    k_path = f"{layer_path}.{arch.component_names['k_proj']}"
    targets.append(ExpansionTarget(
        component_path=k_path + ".weight",
        expand_dim=1,
        current_size=old_dim,
        new_size=new_dim,
        init_value=0.0
    ))

    # V projection - may be smaller for GQA
    v_path = f"{layer_path}.{arch.component_names['v_proj']}"
    targets.append(ExpansionTarget(
        component_path=v_path + ".weight",
        expand_dim=1,
        current_size=old_dim,
        new_size=new_dim,
        init_value=0.0
    ))

    # O projection: (hidden_dim, num_heads * head_dim)
    # Output is hidden_dim (expand rows)
    o_path = f"{layer_path}.{arch.component_names['o_proj']}"
    targets.append(ExpansionTarget(
        component_path=o_path + ".weight",
        expand_dim=0,  # Output dimension
        current_size=old_dim,
        new_size=new_dim,
        init_value=0.0
    ))

    # === MLP projections ===

    # up_proj: (intermediate_size, hidden_dim)
    # Input is hidden_dim (expand cols)
    up_path = f"{layer_path}.{arch.component_names['up_proj']}"
    up_bias = _get_scion_bias(scion, layer_idx, 'up_proj', 'col')
    targets.append(ExpansionTarget(
        component_path=up_path + ".weight",
        expand_dim=1,
        current_size=old_dim,
        new_size=new_dim,
        init_from_bias=up_bias
    ))

    # gate_proj (if GLU): (intermediate_size, hidden_dim)
    if arch.mlp_type == MLPType.GLU:
        gate_name = arch.component_names.get('gate_proj')
        if gate_name:
            gate_path = f"{layer_path}.{gate_name}"
            targets.append(ExpansionTarget(
                component_path=gate_path + ".weight",
                expand_dim=1,
                current_size=old_dim,
                new_size=new_dim,
                init_value=0.0
            ))

    # down_proj: (hidden_dim, intermediate_size)
    # Output is hidden_dim (expand rows)
    down_path = f"{layer_path}.{arch.component_names['down_proj']}"
    down_bias = _get_scion_bias(scion, layer_idx, 'down_proj', 'row')
    targets.append(ExpansionTarget(
        component_path=down_path + ".weight",
        expand_dim=0,
        current_size=old_dim,
        new_size=new_dim,
        init_from_bias=down_bias
    ))

    return targets


def _plan_norm_expansion(
    model: nn.Module,
    arch: ArchitectureSpec,
    layer_idx: int,
    old_dim: int,
    new_dim: int
) -> List[ExpansionTarget]:
    """Plan expansion for layer norms."""
    targets = []

    layers_attr = arch.layers_attr
    layer_path = f"{layers_attr}.{layer_idx}"

    # Input layer norm: weight shape (hidden_dim,)
    input_norm_name = arch.component_names.get('input_norm', 'input_layernorm')
    input_norm_path = f"{layer_path}.{input_norm_name}"

    targets.append(ExpansionTarget(
        component_path=input_norm_path + ".weight",
        expand_dim=0,  # 1D tensor, just append
        current_size=old_dim,
        new_size=new_dim,
        init_value=1.0  # Norm weights typically initialized to 1
    ))

    # Post-attention norm
    post_attn_name = arch.component_names.get('post_attn_norm', 'post_attention_layernorm')
    post_attn_path = f"{layer_path}.{post_attn_name}"

    targets.append(ExpansionTarget(
        component_path=post_attn_path + ".weight",
        expand_dim=0,
        current_size=old_dim,
        new_size=new_dim,
        init_value=1.0
    ))

    # Gemma 2 has additional norms
    if 'pre_feedforward_norm' in arch.component_names:
        pre_ff_path = f"{layer_path}.{arch.component_names['pre_feedforward_norm']}"
        targets.append(ExpansionTarget(
            component_path=pre_ff_path + ".weight",
            expand_dim=0,
            current_size=old_dim,
            new_size=new_dim,
            init_value=1.0
        ))

    if 'post_feedforward_norm' in arch.component_names:
        post_ff_path = f"{layer_path}.{arch.component_names['post_feedforward_norm']}"
        targets.append(ExpansionTarget(
            component_path=post_ff_path + ".weight",
            expand_dim=0,
            current_size=old_dim,
            new_size=new_dim,
            init_value=1.0
        ))

    return targets


def _plan_moe_expansion(
    model: nn.Module,
    arch: ArchitectureSpec,
    layer_idx: int,
    old_dim: int,
    new_dim: int,
    scion: Any
) -> List[ExpansionTarget]:
    """Plan expansion for MoE expert weights."""
    targets = []

    # MoE models have multiple experts per layer
    # Each expert has its own up/gate/down projections
    # Router also needs to be updated

    layers_attr = arch.layers_attr
    layer_path = f"{layers_attr}.{layer_idx}"

    for expert_idx in range(arch.num_experts):
        expert_path = f"{layer_path}.block_sparse_moe.experts.{expert_idx}"

        # Each expert typically has w1 (up), w2 (down), w3 (gate)
        for proj_name, expand_dim in [('w1', 1), ('w2', 0), ('w3', 1)]:
            proj_path = f"{expert_path}.{proj_name}"
            targets.append(ExpansionTarget(
                component_path=proj_path + ".weight",
                expand_dim=expand_dim,
                current_size=old_dim if expand_dim == 1 else old_dim,
                new_size=new_dim,
                init_value=0.0
            ))

    # Router gate: (num_experts, hidden_dim) - expands in hidden_dim
    router_path = f"{layer_path}.block_sparse_moe.gate"
    targets.append(ExpansionTarget(
        component_path=router_path + ".weight",
        expand_dim=1,
        current_size=old_dim,
        new_size=new_dim,
        init_value=0.0
    ))

    return targets


def _get_scion_bias(
    scion: Any,
    layer_idx: int,
    component: str,
    bias_type: str  # 'row' or 'col'
) -> Optional[torch.Tensor]:
    """Get the appropriate bias from scion for initialization."""
    if scion is None or not hasattr(scion, 'neuron_biases'):
        return None

    key = f"layer{layer_idx}_mlp.{component}_{bias_type}"
    return scion.neuron_biases.get(key)


def _get_nested_attr(obj: Any, path: str) -> Any:
    """Get a nested attribute by dot-separated path."""
    parts = path.split('.')
    current = obj
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif part.isdigit() and hasattr(current, '__getitem__'):
            current = current[int(part)]
        else:
            return None
    return current


# =============================================================================
# Expansion Execution
# =============================================================================

def execute_expansion(
    model: nn.Module,
    plan: ExpansionPlan,
    device: str = "cuda"
) -> nn.Module:
    """
    Execute an expansion plan, modifying the model in place.

    This is the core function that actually expands hidden_dim.

    WARNING: This modifies the model irreversibly. Make sure to
    save the original weights first if you need to revert.

    Args:
        model: Model to expand
        plan: ExpansionPlan from plan_expansion()
        device: Device to use for new parameters

    Returns:
        The modified model
    """
    logger.info(f"Executing expansion: {plan.old_hidden_dim} -> {plan.new_hidden_dim}")
    logger.info(f"Total new parameters: {plan.total_new_parameters():,}")

    # Update model config
    model.config.hidden_size = plan.new_hidden_dim

    # Expand embeddings
    for target in plan.embedding_targets:
        _expand_weight_matrix(model, target, device)

    # Expand layer weights
    for target in plan.targets:
        _expand_weight_matrix(model, target, device)

    # Expand norms
    for target in plan.norm_targets:
        _expand_norm(model, target, device)

    # Expand MoE experts
    for layer_idx, expert_targets in plan.expert_targets.items():
        for target in expert_targets:
            _expand_weight_matrix(model, target, device)

    logger.info("Expansion complete")
    return model


def _expand_weight_matrix(
    model: nn.Module,
    target: ExpansionTarget,
    device: str
):
    """Expand a single weight matrix."""
    # Parse path to get parent module and weight name
    parts = target.component_path.rsplit('.', 1)
    if len(parts) != 2:
        logger.warning(f"Invalid path: {target.component_path}")
        return

    parent_path, weight_name = parts
    parent = _get_nested_attr(model, parent_path)

    if parent is None:
        logger.warning(f"Could not find parent: {parent_path}")
        return

    if not hasattr(parent, weight_name):
        logger.warning(f"Parent has no {weight_name}")
        return

    old_weight = getattr(parent, weight_name)
    if not isinstance(old_weight, (torch.Tensor, nn.Parameter)):
        logger.warning(f"{target.component_path} is not a tensor")
        return

    # Create new expanded weight
    old_shape = list(old_weight.shape)
    new_shape = old_shape.copy()
    new_shape[target.expand_dim] = target.new_size

    # Determine initialization value
    if target.init_from_bias is not None:
        init_val = target.init_from_bias[-1].item() if len(target.init_from_bias) > 0 else 0.0
    else:
        init_val = target.init_value

    # Create new weight
    new_weight = torch.zeros(new_shape, dtype=old_weight.dtype, device=device)

    # Copy old weights
    if target.expand_dim == 0:
        # Expanding rows
        new_weight[:target.current_size, ...] = old_weight.to(device)
        new_weight[target.current_size:, ...] = init_val
    else:
        # Expanding columns
        new_weight[..., :target.current_size] = old_weight.to(device)
        new_weight[..., target.current_size:] = init_val

    # Replace weight
    if isinstance(old_weight, nn.Parameter):
        setattr(parent, weight_name, nn.Parameter(new_weight))
    else:
        setattr(parent, weight_name, new_weight)

    logger.debug(f"Expanded {target.component_path}: {old_shape} -> {new_shape}")


def _expand_norm(
    model: nn.Module,
    target: ExpansionTarget,
    device: str
):
    """Expand a normalization layer weight (1D tensor)."""
    parts = target.component_path.rsplit('.', 1)
    if len(parts) != 2:
        return

    parent_path, weight_name = parts
    parent = _get_nested_attr(model, parent_path)

    if parent is None:
        return

    if not hasattr(parent, weight_name):
        return

    old_weight = getattr(parent, weight_name)

    # Create new weight
    new_weight = torch.full(
        (target.new_size,),
        target.init_value,
        dtype=old_weight.dtype,
        device=device
    )
    new_weight[:target.current_size] = old_weight.to(device)

    if isinstance(old_weight, nn.Parameter):
        setattr(parent, weight_name, nn.Parameter(new_weight))
    else:
        setattr(parent, weight_name, new_weight)

    # Also handle bias if present
    if hasattr(parent, 'bias') and parent.bias is not None:
        old_bias = parent.bias
        new_bias = torch.zeros(target.new_size, dtype=old_bias.dtype, device=device)
        new_bias[:target.current_size] = old_bias.to(device)
        parent.bias = nn.Parameter(new_bias) if isinstance(old_bias, nn.Parameter) else new_bias


# =============================================================================
# Scion Metadata Requirements
# =============================================================================

@dataclass
class ScionExpandMetadata:
    """
    Metadata that a scion needs to support expand mode.

    This captures everything needed to initialize the new neuron's
    connections in the expanded model.
    """
    # Which layers were trained (only these need biased initialization)
    trained_layers: List[int]

    # Per-layer biases for each weight matrix
    # key: "layer{idx}.{component}" -> {"row": tensor, "col": tensor}
    layer_biases: Dict[str, Dict[str, torch.Tensor]]

    # Architecture info at training time (to verify compatibility)
    original_hidden_dim: int
    architecture_family: str

    # Feature importance (from cleft analysis)
    # Which features in the original model were most affected
    top_features: Dict[int, List[Tuple[int, float]]]  # layer -> [(idx, importance), ...]

    def validate_compatibility(self, arch: ArchitectureSpec) -> bool:
        """Check if this metadata is compatible with an architecture."""
        return (
            self.original_hidden_dim == arch.hidden_size and
            self.architecture_family == arch.family
        )


def extract_expand_metadata(scion: Any) -> ScionExpandMetadata:
    """
    Extract the metadata needed for expand mode from a trained scion.

    Args:
        scion: A trained Scion object

    Returns:
        ScionExpandMetadata ready for expansion
    """
    # Get trained layers from config
    trained_layers = scion.training_config.injection_layers

    # Convert neuron_biases to layer_biases format
    layer_biases = {}
    for key, bias in scion.neuron_biases.items():
        # Parse key like "layer2_mlp.up_proj_col"
        parts = key.split('_')
        layer_str = parts[0]  # "layer2"
        component = '_'.join(parts[1:-1])  # "mlp.up_proj"
        bias_type = parts[-1]  # "row" or "col"

        full_key = f"{layer_str}.{component}"
        if full_key not in layer_biases:
            layer_biases[full_key] = {}
        layer_biases[full_key][bias_type] = bias

    # Get top features from weight deltas
    top_features = {}
    for delta in scion.weight_deltas:
        layer_idx = delta.layer_index
        if layer_idx not in top_features:
            top_features[layer_idx] = []

        # Get top changed features
        if "down_proj" in delta.component:
            # Row indices represent output (hidden_dim) features
            row_mags = torch.norm(delta.delta, dim=1)
            top_k = min(100, len(row_mags))
            top_vals, top_idxs = torch.topk(row_mags, top_k)
            top_features[layer_idx].extend([
                (idx.item(), val.item())
                for idx, val in zip(top_idxs, top_vals)
            ])

    return ScionExpandMetadata(
        trained_layers=trained_layers,
        layer_biases=layer_biases,
        original_hidden_dim=scion.neuron_index,  # neuron_index was set to hidden_dim
        architecture_family="llama",  # TODO: get from model config
        top_features=top_features
    )


# =============================================================================
# Summary: Architecture Requirements for Expand Mode
# =============================================================================

"""
ARCHITECTURE REQUIREMENTS SUMMARY
=================================

For each architecture family, here's what needs to be expanded when adding
a new neuron (increasing hidden_dim by 1):

## Llama Family (Llama 2/3, Mistral, Apertus, Qwen2)
----------------------------------------------------
Per Layer:
- self_attn.q_proj.weight: (n_heads * head_dim, hidden_dim) -> expand col
- self_attn.k_proj.weight: (n_kv_heads * head_dim, hidden_dim) -> expand col
- self_attn.v_proj.weight: (n_kv_heads * head_dim, hidden_dim) -> expand col
- self_attn.o_proj.weight: (hidden_dim, n_heads * head_dim) -> expand row
- mlp.up_proj.weight: (intermediate_size, hidden_dim) -> expand col
- mlp.gate_proj.weight: (intermediate_size, hidden_dim) -> expand col
- mlp.down_proj.weight: (hidden_dim, intermediate_size) -> expand row
- input_layernorm.weight: (hidden_dim,) -> expand
- post_attention_layernorm.weight: (hidden_dim,) -> expand

Global:
- model.embed_tokens.weight: (vocab_size, hidden_dim) -> expand col
- lm_head.weight: (vocab_size, hidden_dim) -> expand col (often tied)


## Gemma Family
---------------
Same as Llama but:
- Gemma 2 has additional pre_feedforward_layernorm and post_feedforward_layernorm
- lm_head is always tied to embed_tokens


## MoE (Mixtral, Qwen2-MoE)
--------------------------
Same as Llama plus per expert:
- block_sparse_moe.experts.{i}.w1.weight -> expand col
- block_sparse_moe.experts.{i}.w2.weight -> expand row
- block_sparse_moe.experts.{i}.w3.weight -> expand col
- block_sparse_moe.gate.weight: (num_experts, hidden_dim) -> expand col


## GPT-2 Family
---------------
Different structure:
- transformer.wte.weight: (vocab_size, hidden_dim) -> expand col
- Per layer:
  - attn.c_attn.weight: (3 * hidden_dim, hidden_dim) -> expand col AND row
    (Fused QKV - needs special handling)
  - attn.c_proj.weight: (hidden_dim, hidden_dim) -> expand both
  - mlp.c_fc.weight: (4 * hidden_dim, hidden_dim) -> expand col
  - mlp.c_proj.weight: (hidden_dim, 4 * hidden_dim) -> expand row
  - ln_1.weight, ln_2.weight: (hidden_dim,) -> expand


## SCION METADATA REQUIRED
--------------------------
For expand mode to work, the scion needs:

1. trained_layers: List[int]
   - Which layers were actually trained (only these get biased init)

2. layer_biases: Dict[str, Dict[str, Tensor]]
   - Per-layer, per-component row and column biases
   - Derived from training weight deltas

3. original_hidden_dim: int
   - To verify model compatibility

4. architecture_family: str
   - To select correct component paths

5. top_features: Dict[int, List[Tuple[int, float]]]
   - Most affected features per layer
   - For smarter initialization of new connections
"""
