"""
Detached Jacobian approach for concept extraction.

Based on "LLMs are Locally Linear" (https://openreview.net/forum?id=oDWbJsIuEp)
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

logger = logging.getLogger(__name__)


def compute_jacobian(
    model,
    tokenizer,
    text: str,
    device: str = "cuda",
    layer_idx: Optional[int] = None,
    compute_dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Jacobian matrix for input text using detached gradient approach.

    The Jacobian J[i,j,k] represents how output embedding dimension k changes
    with respect to input embedding dimension j of token i.

    Uses "islands of precision" approach: model can be in BF16/FP16, but
    embeddings are upcast to FP32 for gradient computation.

    Args:
        model: Language model (can be in any dtype)
        tokenizer: Tokenizer
        text: Input text
        device: Device
        layer_idx: Optional specific layer to compute Jacobian for (default: final layer)
        compute_dtype: Dtype for Jacobian computation (default: float32)

    Returns:
        jacobian: (seq_len, hidden_dim, hidden_dim) Jacobian matrix
        embeds: (seq_len, hidden_dim) Input embeddings
    """
    # Get embeddings with gradient
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Access embed_tokens for different architectures
    if hasattr(model, 'language_model'):
        # Gemma-3: model.language_model is Gemma3TextModel
        embed_tokens = model.language_model.embed_tokens
    else:
        embed_tokens = model.model.embed_tokens

    # Get embeddings in model's native dtype (for mixed precision)
    # Keep in model dtype to avoid dtype mismatches in forward pass
    model_dtype = next(model.parameters()).dtype
    embeds = embed_tokens(inputs['input_ids']).to(model_dtype)
    embeds.requires_grad_(True)

    # Define forward function for Jacobian computation
    def forward_fn(embeds_input):
        """Forward pass through model to get output embedding"""
        # Get batch size and sequence length
        batch_size, seq_length = embeds_input.shape[:2]

        # Access the right model structure
        if hasattr(model, 'language_model'):
            model_layers = model.language_model  # Gemma3TextModel
        else:
            model_layers = model.model

        # Create position IDs
        position_ids = torch.arange(seq_length, device=embeds_input.device).unsqueeze(0).expand(batch_size, -1)

        # Generate position embeddings
        position_embeddings_global = model_layers.rotary_emb(embeds_input, position_ids)
        position_embeddings_local = model_layers.rotary_emb_local(embeds_input, position_ids)

        # Cache position
        cache_position = torch.arange(seq_length, device=embeds_input.device)

        # Process through layers
        hidden_states = embeds_input
        num_layers = len(model_layers.layers)
        end_layer = layer_idx if layer_idx is not None else num_layers

        for li in range(end_layer):
            layer = model_layers.layers[li]

            # Store residual
            residual = hidden_states

            # Input layer norm
            hidden_states = layer.input_layernorm(hidden_states)

            # Determine position embeddings based on layer type
            if layer.self_attn.is_sliding:
                position_embeddings = position_embeddings_local
            else:
                position_embeddings = position_embeddings_global

            # Create causal attention mask using modern API
            mask_converter = AttentionMaskConverter(is_causal=True)
            causal_mask = mask_converter.to_causal_4d(
                batch_size=batch_size,
                query_length=seq_length,
                key_value_length=seq_length,
                dtype=embeds_input.dtype,
                device=embeds_input.device
            )

            # Attention
            hidden_states, _ = layer.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                cache_position=cache_position
            )

            # Post attention layer norm
            hidden_states = layer.post_attention_layernorm(hidden_states)

            # Add residual
            hidden_states = residual + hidden_states

            # Store residual for MLP
            residual = hidden_states

            # Pre-feedforward layer norm
            hidden_states = layer.pre_feedforward_layernorm(hidden_states)

            # MLP
            hidden_states = layer.mlp(hidden_states)

            # Post-feedforward layer norm
            hidden_states = layer.post_feedforward_layernorm(hidden_states)

            # Add residual
            hidden_states = residual + hidden_states

        # Final normalization
        hidden_states = model_layers.norm(hidden_states)

        # Return last token's embedding
        return hidden_states[0, -1]

    # Compute Jacobian using PyTorch autograd
    # Note: Will compute in model's dtype (BF16), which is sufficient for finding directions
    model.eval()
    with torch.enable_grad():
        jacobian = torch.autograd.functional.jacobian(
            forward_fn,
            embeds,
            vectorize=True,
            strategy="reverse-mode"
        ).squeeze()

    # Upcast result to compute_dtype if requested (for numerical stability in SVD)
    if compute_dtype != model_dtype:
        jacobian = jacobian.to(compute_dtype)

    return jacobian, embeds


def extract_concept_from_jacobian(
    jacobian: torch.Tensor,
    embeds: torch.Tensor,
    n_components: int = 1
) -> np.ndarray:
    """
    Extract concept vector from Jacobian via SVD.

    The top singular vectors of the Jacobian represent the directions
    in activation space that most influence the output.

    Args:
        jacobian: (seq_len, hidden_dim, hidden_dim) Jacobian matrix
        embeds: (seq_len, hidden_dim) Input embeddings
        n_components: Number of components to extract

    Returns:
        concept_vector: (hidden_dim,) concept direction
    """
    # Ensure same dtype for matmul
    embeds = embeds.to(jacobian.dtype)

    # Sum weighted by input embeddings: sum_i(J_i @ embed_i)
    # This gives us the effective transformation
    jacobian_weighted = torch.stack([
        torch.matmul(jacobian[:, i, :], embeds[0, i, :])
        for i in range(jacobian.shape[1])
    ], dim=0)

    # Sum across tokens to get overall effect
    jacobian_output = torch.sum(jacobian_weighted, dim=0)

    # Convert to numpy for SVD
    J_np = jacobian_output.cpu().detach().float().numpy()

    # Normalize
    J_np = J_np / (np.linalg.norm(J_np) + 1e-8)

    return J_np


def compute_jacobian_svd(
    jacobian: torch.Tensor,
    n_components: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SVD of Jacobian to find low-dimensional semantic subspace.

    Args:
        jacobian: (seq_len, hidden_dim, hidden_dim) Jacobian matrix
        n_components: Number of singular vectors to return

    Returns:
        U: Left singular vectors (semantic directions in output space)
        S: Singular values
        Vh: Right singular vectors (semantic directions in input space)
    """
    # Average across tokens
    J_avg = jacobian.mean(dim=0)

    # Convert to numpy
    J_np = J_avg.cpu().detach().float().numpy()

    # SVD
    U, S, Vh = np.linalg.svd(J_np, full_matrices=False)

    # Return top components
    return U[:, :n_components], S[:n_components], Vh[:n_components, :]


def extract_concept_vector_jacobian(
    model,
    tokenizer,
    concept: str,
    device: str = "cuda",
    layer_idx: Optional[int] = None,
    prompt_template: str = "The concept of {concept} means"
) -> np.ndarray:
    """
    Extract concept vector using detached Jacobian approach.

    Args:
        model: Language model
        tokenizer: Tokenizer
        concept: Concept name
        device: Device
        layer_idx: Optional layer to extract from (default: final layer)
        prompt_template: Template for concept prompt

    Returns:
        concept_vector: (hidden_dim,) normalized concept direction
    """
    text = prompt_template.format(concept=concept)

    logger.info(f"Computing Jacobian for: '{text}'")
    jacobian, embeds = compute_jacobian(model, tokenizer, text, device, layer_idx)

    logger.info(f"Extracting concept from Jacobian (shape: {jacobian.shape})")
    concept_vector = extract_concept_from_jacobian(jacobian, embeds)

    return concept_vector
