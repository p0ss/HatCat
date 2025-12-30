#!/usr/bin/env python3
"""
Batched Lens Inference

Efficient batched inference for running N lenses in a single forward pass.
Stacks lens weights into batched tensors for ~10x speedup over sequential.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn


class BatchedLensBank(nn.Module):
    """
    Batched lens inference for running N lenses in a single forward pass.

    Stacks lens weights into batched tensors for efficient GPU utilization.
    Reduces N separate kernel launches to 3 batched matmuls.

    Expected speedup: ~10x for 20+ lenses (based on kernel launch overhead ~10µs).
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.concept_keys: List[str] = []
        self.is_compiled = False
        self.has_layer_norm = False

        # Batched weight tensors (registered as buffers, not parameters)
        self.register_buffer('LN_w', None)  # [N, input_dim] - LayerNorm weights (optional)
        self.register_buffer('LN_b', None)  # [N, input_dim] - LayerNorm bias (optional)
        self.register_buffer('W1', None)  # [N, hidden1, input_dim]
        self.register_buffer('b1', None)  # [N, hidden1]
        self.register_buffer('W2', None)  # [N, hidden2, hidden1]
        self.register_buffer('b2', None)  # [N, hidden2]
        self.register_buffer('W3', None)  # [N, 1, hidden2]
        self.register_buffer('b3', None)  # [N, 1]

    def add_lenses(self, lenses: Dict[str, nn.Module]):
        """
        Add lenses to the bank and recompile batched weights.

        Args:
            lenses: Dict of concept_key → SimpleMLP lens
        """
        if not lenses:
            return

        # Extract weights from each lens
        W1_list, b1_list = [], []
        W2_list, b2_list = [], []
        W3_list, b3_list = [], []
        LN_w_list, LN_b_list = [], []  # LayerNorm weights (optional)
        has_layer_norm = None

        for concept_key, lens in lenses.items():
            # Detect structure based on first layer type
            # With LayerNorm: [LN(0), Linear(1), ReLU(2), Drop(3), Linear(4), ReLU(5), Drop(6), Linear(7)]
            # Without: [Linear(0), ReLU(1), Drop(2), Linear(3), ReLU(4), Drop(5), Linear(6)]
            first_is_ln = hasattr(lens, 'has_layer_norm') and lens.has_layer_norm

            if has_layer_norm is None:
                has_layer_norm = first_is_ln
            elif has_layer_norm != first_is_ln:
                # Mixed architectures - can't batch, fall back to sequential
                # This happens with packs that have some old lenses (no LN) and some new (with LN)
                self.is_compiled = False
                return

            if first_is_ln:
                # With LayerNorm
                LN_w_list.append(lens.net[0].weight.data)
                LN_b_list.append(lens.net[0].bias.data)
                W1_list.append(lens.net[1].weight.data)
                b1_list.append(lens.net[1].bias.data)
                W2_list.append(lens.net[4].weight.data)
                b2_list.append(lens.net[4].bias.data)
                W3_list.append(lens.net[7].weight.data)
                b3_list.append(lens.net[7].bias.data)
            else:
                # Without LayerNorm
                W1_list.append(lens.net[0].weight.data)
                b1_list.append(lens.net[0].bias.data)
                W2_list.append(lens.net[3].weight.data)
                b2_list.append(lens.net[3].bias.data)
                W3_list.append(lens.net[6].weight.data)
                b3_list.append(lens.net[6].bias.data)

            self.concept_keys.append(concept_key)

        # Store LayerNorm flag and weights
        self.has_layer_norm = has_layer_norm or False
        if self.has_layer_norm:
            self.register_buffer('LN_w', torch.stack(LN_w_list).to(self.device))
            self.register_buffer('LN_b', torch.stack(LN_b_list).to(self.device))

        # Stack into batched tensors
        self.W1 = torch.stack(W1_list).to(self.device)  # [N, 128, input_dim]
        self.b1 = torch.stack(b1_list).to(self.device)  # [N, 128]
        self.W2 = torch.stack(W2_list).to(self.device)  # [N, 64, 128]
        self.b2 = torch.stack(b2_list).to(self.device)  # [N, 64]
        self.W3 = torch.stack(W3_list).to(self.device)  # [N, 1, 64]
        self.b3 = torch.stack(b3_list).to(self.device)  # [N, 1]

        self.is_compiled = True

    def clear(self):
        """Clear all lenses from bank."""
        self.concept_keys = []
        self.has_layer_norm = False
        self.LN_w = None
        self.LN_b = None
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W3 = None
        self.b3 = None
        self.is_compiled = False

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float]]]:
        """
        Batched forward pass for all lenses.

        Args:
            x: Input hidden state [1, input_dim] or [input_dim]
            return_logits: If True, return (probs_dict, logits_dict)

        Returns:
            Dict of concept_key → probability (and optionally logits)
        """
        if not self.is_compiled or self.W1 is None:
            if return_logits:
                return {}, {}
            return {}

        # Ensure proper shape: [1, input_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Match dtype to weights
        if x.dtype != self.W1.dtype:
            x = x.to(dtype=self.W1.dtype)

        N = len(self.concept_keys)

        # Apply LayerNorm if present (batched element-wise)
        if self.has_layer_norm and self.LN_w is not None:
            # Normalize input: (x - mean) / sqrt(var + eps)
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + 1e-5)
            # Apply per-lens affine: x_norm * weight + bias
            # x_norm: [1, input_dim], LN_w: [N, input_dim] -> [N, input_dim]
            x_expanded = x_norm * self.LN_w + self.LN_b  # [N, input_dim]
            x_expanded = x_expanded.unsqueeze(1)  # [N, 1, input_dim]
        else:
            # Layer 1: [N, 128] = [1, input_dim] @ [N, input_dim, 128] + [N, 128]
            # Expand input for bmm: [N, 1, input_dim]
            x_expanded = x.expand(N, -1, -1)  # [N, 1, input_dim]
        h1 = torch.bmm(x_expanded, self.W1.transpose(1, 2))  # [N, 1, 128]
        h1 = h1.squeeze(1) + self.b1  # [N, 128]
        h1 = torch.relu(h1)

        # Layer 2: [N, 64]
        h2 = torch.bmm(h1.unsqueeze(1), self.W2.transpose(1, 2))  # [N, 1, 64]
        h2 = h2.squeeze(1) + self.b2  # [N, 64]
        h2 = torch.relu(h2)

        # Layer 3: [N, 1]
        logits = torch.bmm(h2.unsqueeze(1), self.W3.transpose(1, 2))  # [N, 1, 1]
        logits = logits.squeeze(-1).squeeze(-1) + self.b3.squeeze(-1)  # [N]
        probs = torch.sigmoid(logits)

        # Convert to dicts (float() handles any dtype including bfloat16)
        probs_dict = {key: float(probs[i].item()) for i, key in enumerate(self.concept_keys)}

        if return_logits:
            logits_dict = {key: float(logits[i].item()) for i, key in enumerate(self.concept_keys)}
            return probs_dict, logits_dict

        return probs_dict

    def __len__(self):
        return len(self.concept_keys)


__all__ = ["BatchedLensBank"]
