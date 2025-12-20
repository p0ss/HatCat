"""
Semantic interpreter: Activation patterns → Concept predictions
Minimal transformer-based architecture for proof of concept.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class SemanticInterpreter(nn.Module):
    """
    Minimal interpreter: Activation → Concept predictions

    Input: [batch, hidden_dim] activation vectors
    Output: [batch, num_concepts] concept logits + confidence scores
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_concepts: int = 1000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize interpreter model.

        Args:
            hidden_dim: Input activation dimension
            num_concepts: Number of concepts in encyclopedia
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts
        self.d_model = d_model

        # Project activations to model dimension
        self.input_proj = nn.Linear(hidden_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Positional embedding (single position since we have one vector)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projections
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_concepts)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Xavier uniform for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        activations: torch.Tensor,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            activations: [batch, hidden_dim] activation vectors
            return_hidden: Whether to return hidden states

        Returns:
            logits: [batch, num_concepts] concept logits
            confidence: [batch] confidence scores (1 - normalized_entropy)
        """
        batch_size = activations.shape[0]

        # Project to transformer dimension
        x = self.input_proj(activations)  # [B, d_model]
        x = self.input_norm(x)

        # Add positional embedding and expand to sequence
        x = x.unsqueeze(1)  # [B, 1, d_model]
        x = x + self.pos_embedding  # [B, 1, d_model]

        # Transform
        hidden = self.transformer(x)  # [B, 1, d_model]
        hidden = hidden.squeeze(1)  # [B, d_model]

        # Output projection
        hidden = self.output_norm(hidden)
        logits = self.output_proj(hidden)  # [B, num_concepts]

        # Compute confidence from prediction entropy
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        max_entropy = np.log(self.num_concepts)
        confidence = 1 - (entropy / max_entropy)  # [B]

        if return_hidden:
            return logits, confidence, hidden
        return logits, confidence

    def predict(
        self,
        activations: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict top-k concepts for given activations.

        Args:
            activations: [batch, hidden_dim] activation vectors
            top_k: Number of top predictions to return

        Returns:
            top_indices: [batch, top_k] concept indices
            top_probs: [batch, top_k] probabilities
            confidence: [batch] overall confidence
        """
        with torch.no_grad():
            logits, confidence = self.forward(activations)
            probs = torch.softmax(logits, dim=-1)

            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        return top_indices, top_probs, confidence


class InterpreterWithHierarchy(nn.Module):
    """
    Extended interpreter with hierarchical concept prediction.
    First predicts category, then specific concept within category.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_concepts: int = 1000,
        num_categories: int = 10,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4
    ):
        """
        Initialize hierarchical interpreter.

        Args:
            hidden_dim: Input activation dimension
            num_concepts: Number of concepts
            num_categories: Number of concept categories
            d_model: Transformer dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts
        self.num_categories = num_categories

        # Shared encoder
        self.encoder = SemanticInterpreter(
            hidden_dim=hidden_dim,
            num_concepts=num_concepts,  # Shared for now
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

        # Category prediction head
        self.category_head = nn.Linear(d_model, num_categories)

        # Concept prediction head (existing)
        # Uses encoder's output_proj

    def forward(self, activations: torch.Tensor):
        """
        Forward pass with hierarchical prediction.

        Args:
            activations: [batch, hidden_dim]

        Returns:
            concept_logits: [batch, num_concepts]
            category_logits: [batch, num_categories]
            confidence: [batch]
        """
        # Get hidden states from encoder
        concept_logits, confidence, hidden = self.encoder(activations, return_hidden=True)

        # Predict categories
        category_logits = self.category_head(hidden)

        return concept_logits, category_logits, confidence


class MultiTaskInterpreter(nn.Module):
    """
    Multi-task interpreter with concept classification + similarity prediction.
    More advanced training objectives for better generalization.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_concepts: int = 1000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4
    ):
        super().__init__()

        # Base interpreter
        self.base = SemanticInterpreter(
            hidden_dim=hidden_dim,
            num_concepts=num_concepts,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

        # Additional heads for multi-task learning
        self.similarity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4)
        )

    def forward(self, activations: torch.Tensor):
        """
        Multi-task forward pass.

        Args:
            activations: [batch, hidden_dim]

        Returns:
            concept_logits: [batch, num_concepts]
            embedding: [batch, d_model//4] for similarity
            confidence: [batch]
        """
        concept_logits, confidence, hidden = self.base(activations, return_hidden=True)
        embedding = self.similarity_head(hidden)

        return concept_logits, embedding, confidence
