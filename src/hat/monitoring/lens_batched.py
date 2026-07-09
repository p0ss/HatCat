#!/usr/bin/env python3
"""
Batched Lens Inference

Efficient batched inference for running N lenses in a single forward pass.
Stacks lens weights into batched tensors for ~10x speedup over sequential.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn


def wilson_score_interval(positive: float, negative: float, z: float = 1.645, scale: float = 10.0) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for polar probe outputs.

    Scales probe probabilities to vote counts before computing Wilson bounds.
    A probability of 1.0 is treated as `scale` votes, making the statistical
    calculation more meaningful for the 0-1 probability range.

    Example: pos=0.8, neg=0.4 with scale=10 becomes 8 votes positive, 4 negative.
    This gives n=12 votes total, proportion p=0.667 toward positive.

    The formula rewards:
    - Higher total activation (more evidence = narrower confidence interval)
    - Clearer polarity (proportion further from 0.5)
    - Combined via multiplicative boost for total activation

    When both poles are highly activated (e.g., 0.8/0.4), the concept is more
    "present" than when only one pole is strong (e.g., 0.8/0.2), even though
    the direction is less certain. The evidence_boost factor captures this.

    Args:
        positive: Positive pole probability (0-1)
        negative: Negative pole probability (0-1)
        z: Z-score for confidence level (1.645 = 90%, 1.96 = 95%)
        scale: Number of votes a probability of 1.0 represents (default 10)

    Returns:
        (confidence, polarity) where:
        - confidence: Evidence-weighted Wilson score (0-1), suitable for ranking
        - polarity: Direction as continuous value (-1 to +1)
    """
    # Scale probabilities to vote counts
    pos_votes = positive * scale
    neg_votes = negative * scale
    n = pos_votes + neg_votes  # Total votes

    if n < 0.01:  # Effectively no signal
        return 0.0, 0.0

    # Proportion favoring positive pole
    p = pos_votes / n

    # Wilson score lower bound formula
    # This gives a conservative estimate that penalizes low sample sizes
    denominator = 1 + (z * z) / n
    centre = p + (z * z) / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)

    wilson_lower = (centre - spread) / denominator
    wilson_upper = (centre + spread) / denominator

    # Confidence is the lower bound of whichever direction is stronger
    # If positive dominates (p > 0.5), use wilson_lower directly
    # If negative dominates (p < 0.5), use 1 - wilson_upper
    if p >= 0.5:
        direction_confidence = wilson_lower
    else:
        direction_confidence = 1 - wilson_upper

    # Evidence boost: value higher total activation
    # A concept with both poles strongly activated is more "present"
    # even if direction is uncertain
    # Normalized so max activation (pos=1, neg=1) gives boost of 1.0
    total_activation = positive + negative  # 0 to 2 range
    evidence_boost = total_activation / 2.0  # 0 to 1 range

    # Combined confidence: blend direction confidence with evidence boost
    # Weight evidence_boost more heavily to match intuition that
    # 0.8/0.4 (more present) > 0.8/0.2 (clearer direction)
    confidence = direction_confidence * 0.4 + evidence_boost * 0.6

    # Polarity: continuous -1 to +1 based on proportion
    # p=1.0 -> +1, p=0.5 -> 0, p=0.0 -> -1
    polarity = 2 * p - 1

    return confidence, polarity


class BatchedLensBank(nn.Module):
    """
    Batched lens inference for running N lenses in a single forward pass.

    Stacks lens weights into batched tensors for efficient GPU utilization.
    Reduces N separate kernel launches to 3 batched matmuls.

    Expected speedup: ~10x for 20+ lenses (based on kernel launch overhead ~10µs).

    Supports polar lenses (positive/negative probe pairs) with steering score output.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.concept_keys: List[str] = []
        self.is_compiled = False
        self.has_layer_norm = False

        # Track polar concepts (those with positive/negative pairs)
        # polar_concept_indices: set of original concept indices that are polar
        # polar_index_order: list preserving the ORDER they were added (for weight tensor indexing)
        self.polar_concept_indices: Set[int] = set()
        self.polar_index_order: List[int] = []

        # Batched weight tensors (registered as buffers, not parameters)
        self.register_buffer('LN_w', None)  # [N, input_dim] - LayerNorm weights (optional)
        self.register_buffer('LN_b', None)  # [N, input_dim] - LayerNorm bias (optional)
        self.register_buffer('W1', None)  # [N, hidden1, input_dim]
        self.register_buffer('b1', None)  # [N, hidden1]
        self.register_buffer('W2', None)  # [N, hidden2, hidden1]
        self.register_buffer('b2', None)  # [N, hidden2]
        self.register_buffer('W3', None)  # [N, 1, hidden2]
        self.register_buffer('b3', None)  # [N, 1]

        # Negative pole weights for polar concepts (parallel to main weights)
        self.register_buffer('neg_LN_w', None)
        self.register_buffer('neg_LN_b', None)
        self.register_buffer('neg_W1', None)
        self.register_buffer('neg_b1', None)
        self.register_buffer('neg_W2', None)
        self.register_buffer('neg_b2', None)
        self.register_buffer('neg_W3', None)
        self.register_buffer('neg_b3', None)
        self.has_polar_lenses = False

    def add_lenses(
        self,
        lenses: Dict[str, nn.Module],
        negative_lenses: Optional[Dict[str, nn.Module]] = None
    ):
        """
        Add lenses to the bank and recompile batched weights.

        Args:
            lenses: Dict of concept_key → SimpleMLP lens (positive pole for polar lenses)
            negative_lenses: Optional dict of concept_key → negative pole lens for polar concepts
        """
        if not lenses:
            return

        negative_lenses = negative_lenses or {}

        # Extract weights from each lens
        W1_list, b1_list = [], []
        W2_list, b2_list = [], []
        W3_list, b3_list = [], []
        LN_w_list, LN_b_list = [], []  # LayerNorm weights (optional)

        # Negative pole weights (parallel lists, only for polar concepts)
        neg_W1_list, neg_b1_list = [], []
        neg_W2_list, neg_b2_list = [], []
        neg_W3_list, neg_b3_list = [], []
        neg_LN_w_list, neg_LN_b_list = [], []

        has_layer_norm = None
        polar_indices = []

        for idx, (concept_key, lens) in enumerate(lenses.items()):
            # Detect structure based on first layer type
            # With LayerNorm: [LN(0), Linear(1), ReLU(2), Drop(3), Linear(4), ReLU(5), Drop(6), Linear(7)]
            # Without: [Linear(0), ReLU(1), Drop(2), Linear(3), ReLU(4), Drop(5), Linear(6)]
            first_is_ln = hasattr(lens, 'has_layer_norm') and lens.has_layer_norm

            if has_layer_norm is None:
                has_layer_norm = first_is_ln
            elif has_layer_norm != first_is_ln:
                # Mixed architectures - can't batch, fall back to sequential
                self.is_compiled = False
                return

            # Extract positive/main weights
            if first_is_ln:
                LN_w_list.append(lens.net[0].weight.data)
                LN_b_list.append(lens.net[0].bias.data)
                W1_list.append(lens.net[1].weight.data)
                b1_list.append(lens.net[1].bias.data)
                W2_list.append(lens.net[4].weight.data)
                b2_list.append(lens.net[4].bias.data)
                W3_list.append(lens.net[7].weight.data)
                b3_list.append(lens.net[7].bias.data)
            else:
                W1_list.append(lens.net[0].weight.data)
                b1_list.append(lens.net[0].bias.data)
                W2_list.append(lens.net[3].weight.data)
                b2_list.append(lens.net[3].bias.data)
                W3_list.append(lens.net[6].weight.data)
                b3_list.append(lens.net[6].bias.data)

            # Check if this concept has a negative pole
            neg_lens = negative_lenses.get(concept_key)
            if neg_lens is not None:
                polar_indices.append(idx)
                neg_is_ln = hasattr(neg_lens, 'has_layer_norm') and neg_lens.has_layer_norm

                if neg_is_ln:
                    neg_LN_w_list.append(neg_lens.net[0].weight.data)
                    neg_LN_b_list.append(neg_lens.net[0].bias.data)
                    neg_W1_list.append(neg_lens.net[1].weight.data)
                    neg_b1_list.append(neg_lens.net[1].bias.data)
                    neg_W2_list.append(neg_lens.net[4].weight.data)
                    neg_b2_list.append(neg_lens.net[4].bias.data)
                    neg_W3_list.append(neg_lens.net[7].weight.data)
                    neg_b3_list.append(neg_lens.net[7].bias.data)
                else:
                    neg_W1_list.append(neg_lens.net[0].weight.data)
                    neg_b1_list.append(neg_lens.net[0].bias.data)
                    neg_W2_list.append(neg_lens.net[3].weight.data)
                    neg_b2_list.append(neg_lens.net[3].bias.data)
                    neg_W3_list.append(neg_lens.net[6].weight.data)
                    neg_b3_list.append(neg_lens.net[6].bias.data)

            self.concept_keys.append(concept_key)

        # Store LayerNorm flag and weights
        self.has_layer_norm = has_layer_norm or False
        if self.has_layer_norm:
            self.register_buffer('LN_w', torch.stack(LN_w_list).to(self.device))
            self.register_buffer('LN_b', torch.stack(LN_b_list).to(self.device))

        # Stack into batched tensors
        self.W1 = torch.stack(W1_list).to(self.device)
        self.b1 = torch.stack(b1_list).to(self.device)
        self.W2 = torch.stack(W2_list).to(self.device)
        self.b2 = torch.stack(b2_list).to(self.device)
        self.W3 = torch.stack(W3_list).to(self.device)
        self.b3 = torch.stack(b3_list).to(self.device)

        # Store polar concept indices
        # polar_index_order preserves the iteration order for correct weight tensor indexing
        self.polar_concept_indices = set(polar_indices)
        self.polar_index_order = polar_indices  # Preserve order!
        self.has_polar_lenses = len(polar_indices) > 0

        # Stack negative pole weights if we have polar lenses
        if self.has_polar_lenses and neg_W1_list:
            if self.has_layer_norm and neg_LN_w_list:
                self.register_buffer('neg_LN_w', torch.stack(neg_LN_w_list).to(self.device))
                self.register_buffer('neg_LN_b', torch.stack(neg_LN_b_list).to(self.device))
            self.neg_W1 = torch.stack(neg_W1_list).to(self.device)
            self.neg_b1 = torch.stack(neg_b1_list).to(self.device)
            self.neg_W2 = torch.stack(neg_W2_list).to(self.device)
            self.neg_b2 = torch.stack(neg_b2_list).to(self.device)
            self.neg_W3 = torch.stack(neg_W3_list).to(self.device)
            self.neg_b3 = torch.stack(neg_b3_list).to(self.device)

        self.is_compiled = True

    def clear(self):
        """Clear all lenses from bank."""
        self.concept_keys = []
        self.has_layer_norm = False
        self.polar_concept_indices = set()
        self.polar_index_order = []
        self.has_polar_lenses = False

        # Clear main weights
        self.LN_w = None
        self.LN_b = None
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W3 = None
        self.b3 = None

        # Clear negative pole weights
        self.neg_LN_w = None
        self.neg_LN_b = None
        self.neg_W1 = None
        self.neg_b1 = None
        self.neg_W2 = None
        self.neg_b2 = None
        self.neg_W3 = None
        self.neg_b3 = None

        self.is_compiled = False

    def _run_forward(
        self,
        x: torch.Tensor,
        W1: torch.Tensor, b1: torch.Tensor,
        W2: torch.Tensor, b2: torch.Tensor,
        W3: torch.Tensor, b3: torch.Tensor,
        LN_w: Optional[torch.Tensor] = None,
        LN_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run batched forward pass with given weights, returning probs and logits."""
        N = W1.shape[0]

        if self.has_layer_norm and LN_w is not None:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + 1e-5)
            x_expanded = x_norm * LN_w + LN_b
            x_expanded = x_expanded.unsqueeze(1)
        else:
            x_expanded = x.expand(N, -1, -1)

        h1 = torch.bmm(x_expanded, W1.transpose(1, 2))
        h1 = h1.squeeze(1) + b1
        h1 = torch.relu(h1)

        h2 = torch.bmm(h1.unsqueeze(1), W2.transpose(1, 2))
        h2 = h2.squeeze(1) + b2
        h2 = torch.relu(h2)

        logits = torch.bmm(h2.unsqueeze(1), W3.transpose(1, 2))
        logits = logits.squeeze(-1).squeeze(-1) + b3.squeeze(-1)
        probs = torch.sigmoid(logits)

        return probs, logits

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
        return_polar_details: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float]], Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict]]]:
        """
        Batched forward pass for all lenses.

        For polar concepts, returns Wilson score confidence as the main score.
        Wilson score provides a statistically conservative estimate of confidence
        that accounts for the strength of both positive and negative signals.

        Args:
            x: Input hidden state [1, input_dim] or [input_dim]
            return_logits: If True, return (probs_dict, logits_dict)
            return_polar_details: If True, include polar details (pos_prob, neg_prob, confidence, polarity)

        Returns:
            Dict of concept_key → score (and optionally logits and polar details)
            For polar concepts: score is Wilson confidence (0-1) with sign indicating polarity
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

        # Run main (positive) forward pass
        probs, logits = self._run_forward(
            x, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3,
            self.LN_w, self.LN_b
        )

        # Run negative pole forward pass if we have polar lenses
        neg_probs = None
        if self.has_polar_lenses and self.neg_W1 is not None:
            neg_probs, _ = self._run_forward(
                x, self.neg_W1, self.neg_b1, self.neg_W2, self.neg_b2, self.neg_W3, self.neg_b3,
                self.neg_LN_w, self.neg_LN_b
            )

        # Build output dicts
        probs_dict = {}
        logits_dict = {}
        polar_details = {}

        # Map from concept index to polar index (for negative probs lookup)
        # CRITICAL: Use polar_index_order (iteration order) NOT sorted order!
        # The negative weight tensors were stacked in iteration order during add_lenses.
        polar_idx_map = {orig_idx: polar_idx for polar_idx, orig_idx in enumerate(self.polar_index_order)}

        for i, key in enumerate(self.concept_keys):
            pos_prob = float(probs[i].item())

            if i in self.polar_concept_indices and neg_probs is not None:
                # Polar concept: compute Wilson score confidence
                polar_idx = polar_idx_map[i]
                neg_prob = float(neg_probs[polar_idx].item())

                # Wilson score gives us confidence (0-1) and polarity (-1 to +1)
                confidence, polarity = wilson_score_interval(pos_prob, neg_prob)

                # Main score is signed confidence (positive = toward positive pole)
                # This preserves ranking by confidence while indicating direction
                signed_score = confidence * (1 if polarity >= 0 else -1)
                probs_dict[key] = signed_score

                if return_polar_details:
                    polar_details[key] = {
                        'positive_prob': pos_prob,
                        'negative_prob': neg_prob,
                        'confidence': confidence,
                        'polarity': polarity,
                        'is_polar': True
                    }
            else:
                # Non-polar concept: use probability directly
                probs_dict[key] = pos_prob

                if return_polar_details:
                    polar_details[key] = {
                        'positive_prob': pos_prob,
                        'negative_prob': None,
                        'confidence': None,
                        'polarity': None,
                        'is_polar': False
                    }

            if return_logits:
                logits_dict[key] = float(logits[i].item())

        if return_polar_details:
            if return_logits:
                return probs_dict, logits_dict, polar_details
            return probs_dict, polar_details

        if return_logits:
            return probs_dict, logits_dict

        return probs_dict

    def __len__(self):
        return len(self.concept_keys)


__all__ = ["BatchedLensBank", "wilson_score_interval"]
