"""
Graft training procedure.

Trains a graft that:
1. Adds a new labelled dimension to the substrate for the concept
2. Learns biases to existing weights that encode relational structure
3. Trains a lens that reads from the new dimension plus auxiliary dims

Per MAP_GRAFTING.md Section 5.

For simplified initial testing, we implement a "soft graft" approach:
- Instead of actually expanding the model's hidden dimension (which requires
  modifying the model architecture), we use a projection-based approach
- The concept dimension is represented as a learned vector that projects
  into/out of the existing hidden space
- This allows testing the graft concept without major architectural changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime
import logging
import json

from .data_structures import (
    Graft, GraftConfig, ConceptRegion, SubstrateBias, InjectionPoint
)

logger = logging.getLogger(__name__)


class DimensionProjection(nn.Module):
    """
    Learned projection for a concept dimension.

    In the full spec, this would expand the hidden dimension.
    For simplified testing, we learn a projection vector that:
    - Projects hidden states onto a concept "dimension" (scalar per position)
    - Can be used to bias the hidden states
    """

    def __init__(self, hidden_dim: int, init_method: str = "learned"):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Projection vector: maps hidden_dim -> 1 (concept activation)
        self.projection = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        # Bias vector: maps concept activation -> hidden_dim (adds to residual)
        self.bias_direction = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        if init_method == "zero":
            nn.init.zeros_(self.projection)
            nn.init.zeros_(self.bias_direction)
        elif init_method == "random":
            nn.init.normal_(self.projection, std=0.1)
            nn.init.normal_(self.bias_direction, std=0.1)
        # "learned" uses the small random initialization above

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute concept activation from hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            activations: (batch, seq_len, 1) concept activation values
        """
        # Normalize projection for stability
        proj_normalized = self.projection / (torch.norm(self.projection) + 1e-8)
        activations = torch.matmul(hidden_states, proj_normalized)
        return activations.unsqueeze(-1)

    def get_bias(self, concept_activation: torch.Tensor) -> torch.Tensor:
        """
        Get bias to add to hidden states based on concept activation.

        Args:
            concept_activation: (batch, seq_len, 1) or (batch, seq_len)

        Returns:
            bias: (batch, seq_len, hidden_dim) to add to residual stream
        """
        if concept_activation.dim() == 3:
            concept_activation = concept_activation.squeeze(-1)

        # Normalize bias direction
        bias_normalized = self.bias_direction / (torch.norm(self.bias_direction) + 1e-8)
        return concept_activation.unsqueeze(-1) * bias_normalized


class SparseBiasAccumulator(nn.Module):
    """
    Accumulates sparse biases during graft training.

    Learns which dimensions should be biased and by how much.
    Uses L1 regularization to encourage sparsity.
    """

    def __init__(
        self,
        hidden_dim: int,
        target_sparsity: float = 0.95
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_sparsity = target_sparsity

        # Learnable bias vector
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add bias to hidden states."""
        return hidden_states + self.bias

    def l1_norm(self) -> torch.Tensor:
        """L1 norm for sparsity regularization."""
        return torch.abs(self.bias).sum()

    def l2_norm(self) -> torch.Tensor:
        """L2 norm for magnitude regularization."""
        return torch.norm(self.bias)

    def threshold_small_values(self, threshold: float):
        """Zero out bias values below threshold."""
        with torch.no_grad():
            mask = torch.abs(self.bias) < threshold
            self.bias[mask] = 0.0

    def get_sparsity(self) -> float:
        """Current sparsity level (fraction of zeros)."""
        with torch.no_grad():
            return float((self.bias == 0).sum() / len(self.bias))

    def to_sparse_delta(self) -> Dict[str, Any]:
        """Export bias as sparse representation."""
        with torch.no_grad():
            nonzero_mask = self.bias != 0
            indices = torch.where(nonzero_mask)[0].cpu().numpy().tolist()
            values = self.bias[nonzero_mask].cpu().numpy().tolist()

        return {
            "format": "sparse_coo",
            "indices": indices,
            "values": values,
            "nnz": len(indices),
            "shape": [self.hidden_dim]
        }


class ConceptLens(nn.Module):
    """
    Lens that reads from primary dimension plus auxiliary dimensions.

    In the full spec, this would read from a new labelled dimension.
    For simplified testing, it reads from the DimensionProjection output
    plus selected auxiliary dimensions from the original hidden space.
    """

    def __init__(
        self,
        hidden_dim: int,
        auxiliary_dims: List[int],
        intermediate_dim: int = 64
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.auxiliary_dims = auxiliary_dims

        # Weight for primary dimension (concept projection)
        self.primary_weight = nn.Parameter(torch.ones(1))

        # Weights for auxiliary dimensions
        n_aux = len(auxiliary_dims)
        if n_aux > 0:
            self.aux_weights = nn.Linear(n_aux, intermediate_dim, bias=False)
            self.aux_to_out = nn.Linear(intermediate_dim, 1, bias=False)
        else:
            self.aux_weights = None
            self.aux_to_out = None

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        primary_activation: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute lens score.

        Args:
            primary_activation: (batch, seq_len) from DimensionProjection
            hidden_states: (batch, seq_len, hidden_dim) original hidden states

        Returns:
            scores: (batch, seq_len) lens output (pre-sigmoid)
        """
        # Primary contribution
        score = self.primary_weight * primary_activation

        # Auxiliary contribution
        if self.aux_weights is not None and len(self.auxiliary_dims) > 0:
            aux_features = hidden_states[:, :, self.auxiliary_dims]  # (batch, seq, n_aux)
            aux_hidden = self.aux_weights(aux_features)  # (batch, seq, intermediate)
            aux_contrib = self.aux_to_out(F.relu(aux_hidden)).squeeze(-1)  # (batch, seq)
            score = score + aux_contrib

        return score + self.bias


def train_graft(
    model: nn.Module,
    tokenizer: Any,
    dataset: Dict[str, List[str]],  # {"positive": [...], "negative": [...]}
    region: ConceptRegion,
    concept_id: str,
    config: Optional[GraftConfig] = None,
    output_dir: Optional[Path] = None,
    device: str = "cuda",
    verbose: bool = True
) -> Graft:
    """
    Train a graft that adds a concept dimension and biases to the substrate.

    For simplified initial testing, this trains:
    1. A DimensionProjection that computes concept activation
    2. A SparseBiasAccumulator that learns which dimensions to bias
    3. A ConceptLens that reads from the projection + auxiliary dims

    Args:
        model: The substrate model (frozen during graft training)
        tokenizer: Model tokenizer
        dataset: Dict with "positive" and "negative" text examples
        region: ConceptRegion identifying important dimensions
        concept_id: Identifier for the concept
        config: Training configuration
        output_dir: Directory to save graft artifacts
        device: Device to train on
        verbose: Print training progress

    Returns:
        Trained Graft object
    """
    if config is None:
        config = GraftConfig()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get model hidden dimension
    hidden_dim = model.config.hidden_size
    injection_layer = config.injection_layers[0] if config.injection_layers else 18

    # Initialize graft components
    dim_projection = DimensionProjection(hidden_dim, config.dimension_init).to(device)
    bias_accum = SparseBiasAccumulator(hidden_dim, config.bias_sparsity_target).to(device)

    auxiliary_dims = region.get_all_indices(injection_layer)
    lens = ConceptLens(hidden_dim, auxiliary_dims).to(device)

    # Optimizer for all graft components
    optimizer = torch.optim.AdamW(
        list(dim_projection.parameters()) +
        list(bias_accum.parameters()) +
        list(lens.parameters()),
        lr=config.learning_rate
    )

    # Prepare training data
    positive_texts = dataset.get("positive", [])
    negative_texts = dataset.get("negative", [])

    if not positive_texts or not negative_texts:
        raise ValueError("Dataset must contain both positive and negative examples")

    # Training loop
    model.eval()
    training_metrics = []

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # Shuffle and pair positive/negative examples
        np.random.shuffle(positive_texts)
        np.random.shuffle(negative_texts)

        batch_texts = []
        batch_labels = []

        for i in range(min(len(positive_texts), len(negative_texts))):
            batch_texts.extend([positive_texts[i], negative_texts[i]])
            batch_labels.extend([1.0, 0.0])

            if len(batch_texts) >= config.batch_size or i == min(len(positive_texts), len(negative_texts)) - 1:
                # Process batch
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)

                labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)

                optimizer.zero_grad()

                # Get hidden states from model (frozen)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    # Get hidden states at injection layer
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        hidden_states = outputs.hidden_states[injection_layer + 1]  # +1 for embedding layer
                    else:
                        # Fallback: use last hidden state
                        hidden_states = outputs.last_hidden_state

                # Apply graft components
                # 1. Compute concept activation
                concept_act = dim_projection(hidden_states).squeeze(-1)  # (batch, seq)

                # 2. Apply bias (for training signal, not actually modifying model)
                biased_hidden = bias_accum(hidden_states)

                # 3. Compute lens score (use mean over sequence)
                scores = lens(concept_act, biased_hidden)  # (batch, seq)
                # Take mean over sequence positions (excluding padding)
                attention_mask = inputs.get('attention_mask', torch.ones_like(scores))
                masked_scores = scores * attention_mask
                mean_scores = masked_scores.sum(dim=1) / attention_mask.sum(dim=1)

                # Compute loss
                bce_loss = F.binary_cross_entropy_with_logits(mean_scores, labels)

                # Regularization losses
                sparsity_loss = config.bias_magnitude_penalty * bias_accum.l1_norm()
                magnitude_loss = config.bias_magnitude_penalty * 0.1 * bias_accum.l2_norm()

                total_loss = bce_loss + sparsity_loss + magnitude_loss
                total_loss.backward()
                optimizer.step()

                # Enforce sparsity via thresholding
                bias_accum.threshold_small_values(config.sparsity_threshold)

                # Track metrics
                with torch.no_grad():
                    predictions = (torch.sigmoid(mean_scores) > 0.5).float()
                    correct = (predictions == labels).sum().item()
                    epoch_correct += correct
                    epoch_total += len(labels)
                    epoch_loss += bce_loss.item()

                # Clear batch
                batch_texts = []
                batch_labels = []

        # Epoch summary
        epoch_acc = epoch_correct / max(epoch_total, 1)
        avg_loss = epoch_loss / max(epoch_total // config.batch_size, 1)

        training_metrics.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": epoch_acc,
            "sparsity": bias_accum.get_sparsity()
        })

        if verbose:
            print(f"  Epoch {epoch + 1}/{config.epochs}: "
                  f"loss={avg_loss:.4f}, acc={epoch_acc:.3f}, "
                  f"sparsity={bias_accum.get_sparsity():.3f}")

    # Save graft artifacts
    graft_id = f"graft-{concept_id.replace('/', '-')}-v1"
    training_run_id = f"trainrun-{concept_id.replace('/', '-')}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    if output_dir:
        # Save projection
        proj_path = output_dir / f"{graft_id}_projection.pt"
        torch.save(dim_projection.state_dict(), proj_path)

        # Save lens
        lens_path = output_dir / f"{graft_id}_lens.pt"
        torch.save(lens.state_dict(), lens_path)

        # Save bias as sparse tensor
        bias_path = output_dir / f"{graft_id}_bias.json"
        with open(bias_path, 'w') as f:
            json.dump(bias_accum.to_sparse_delta(), f, indent=2)
    else:
        proj_path = None
        lens_path = None
        bias_path = None

    # Construct injection points
    injection_points = [
        InjectionPoint(
            layer=injection_layer,
            component="residual",
            projection_path=str(proj_path) if proj_path else None
        )
    ]

    # Construct substrate biases
    bias_delta = bias_accum.to_sparse_delta()
    substrate_biases = [
        SubstrateBias(
            layer=injection_layer,
            component="residual",
            bias_delta_path=str(bias_path) if bias_path else "",
            nnz=bias_delta["nnz"],
            shape=bias_delta["shape"],
            magnitude_stats={
                "mean": float(np.mean(np.abs(bias_delta["values"]))) if bias_delta["values"] else 0.0,
                "max": float(np.max(np.abs(bias_delta["values"]))) if bias_delta["values"] else 0.0,
                "l2_norm": float(np.linalg.norm(bias_delta["values"])) if bias_delta["values"] else 0.0
            }
        )
    ]

    # Create Graft object
    graft = Graft(
        graft_id=graft_id,
        concept_id=concept_id,
        concept_version="1.0.0",
        dimension_index=hidden_dim,  # Would be new index in full implementation
        dimension_label=f"concept/{concept_id}",
        injection_points=injection_points,
        substrate_biases=substrate_biases,
        lens_path=str(lens_path) if lens_path else "",
        primary_dimension=hidden_dim,
        auxiliary_dimensions=auxiliary_dims,
        substrate_id=model.config.name_or_path if hasattr(model.config, 'name_or_path') else "unknown",
        pre_graft_dim=hidden_dim,
        post_graft_dim=hidden_dim + 1,
        training_run_id=training_run_id,
        source_region_id=region.region_id,
        config=config,
        metrics={
            "final_accuracy": training_metrics[-1]["accuracy"] if training_metrics else 0.0,
            "final_loss": training_metrics[-1]["loss"] if training_metrics else 0.0,
            "final_sparsity": training_metrics[-1]["sparsity"] if training_metrics else 0.0,
            "epochs_trained": config.epochs
        }
    )

    if output_dir:
        graft.save(output_dir / f"{graft_id}.json")

    return graft
