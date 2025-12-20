"""
Scion training and application.

A Scion is a permanent graft that:
1. Trains only the cleft regions (weights associated with tagged concepts)
2. Captures the delta (how much each feature changed during training)
3. Adds one new neuron with biases proportional to the training deltas
4. Permanently modifies the substrate

The training flow:
1. Load experience data with concept tags
2. Build union cleft from all tagged concepts' clefts
3. Snapshot weights in the cleft before training
4. Train on experience data with cleft-aware freezing
5. Compute delta = trained_weights - snapshot
6. Create scion: new neuron with biases = delta magnitudes

Terminology:
- Bud: soft/temporary graft using hooks (for testing)
- Scion: hard/permanent graft that modifies weights and adds neuron
- Cleft: the region of weights being modified (from lens analysis)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import copy

from .cleft import Cleft, UnionCleft, CleftAwareFreezer, merge_clefts, _get_layer, _get_component, _get_model_layers

logger = logging.getLogger(__name__)


@dataclass
class ScionConfig:
    """Configuration for scion training."""
    # Training
    learning_rate: float = 1e-4
    epochs: int = 3
    batch_size: int = 8

    # Regularization
    weight_decay: float = 0.01

    # Delta thresholding (for sparse biases)
    delta_threshold: float = 1e-5  # Below this, bias is zero

    # Layers to inject the new neuron
    injection_layers: List[int] = field(default_factory=lambda: [18, 20, 22])


@dataclass
class WeightDelta:
    """
    Captures the change in a weight matrix during scion training.
    """
    layer_index: int
    component: str
    delta: torch.Tensor  # The actual weight change
    cleft_mask: torch.Tensor  # Which elements were trainable

    @property
    def magnitude(self) -> float:
        """L2 norm of the delta."""
        return float(torch.norm(self.delta).item())

    @property
    def sparsity(self) -> float:
        """Fraction of elements that are effectively zero."""
        return float((torch.abs(self.delta) < 1e-6).sum() / self.delta.numel())

    def to_sparse(self, threshold: float = 1e-5) -> Dict[str, Any]:
        """Convert to sparse representation."""
        mask = torch.abs(self.delta) >= threshold
        indices = torch.nonzero(mask, as_tuple=False).tolist()
        values = self.delta[mask].tolist()

        return {
            "layer_index": self.layer_index,
            "component": self.component,
            "shape": list(self.delta.shape),
            "indices": indices,
            "values": values,
            "nnz": len(values),
            "magnitude": self.magnitude,
            "sparsity": self.sparsity
        }


@dataclass
class Scion:
    """
    A permanent graft that adds a new concept neuron to the substrate.

    Contains:
    - The weight deltas from training (what changed in the cleft)
    - The new neuron's biases (derived from delta magnitudes)
    - Provenance information
    """
    scion_id: str
    concept_id: str

    # The weight changes that define this concept's "meaning"
    weight_deltas: List[WeightDelta]

    # New neuron specification
    neuron_index: int  # Index in the expanded hidden_dim
    neuron_biases: Dict[str, torch.Tensor]  # layer.component -> bias vector

    # Source information
    source_cleft_concepts: List[str]  # Concepts whose clefts were trained
    training_config: ScionConfig

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Lifecycle
    applied: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_total_delta_magnitude(self) -> float:
        """Total L2 norm of all weight deltas."""
        return sum(wd.magnitude for wd in self.weight_deltas)

    def save(self, output_dir: Path):
        """Save scion to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta = {
            "scion_id": self.scion_id,
            "concept_id": self.concept_id,
            "neuron_index": self.neuron_index,
            "source_cleft_concepts": self.source_cleft_concepts,
            "training_config": {
                "learning_rate": self.training_config.learning_rate,
                "epochs": self.training_config.epochs,
                "batch_size": self.training_config.batch_size,
                "injection_layers": self.training_config.injection_layers
            },
            "metrics": self.metrics,
            "applied": self.applied,
            "created_at": self.created_at,
            "weight_deltas": [wd.to_sparse() for wd in self.weight_deltas]
        }

        with open(output_dir / f"{self.scion_id}.json", 'w') as f:
            json.dump(meta, f, indent=2)

        # Save neuron biases as tensors
        torch.save(self.neuron_biases, output_dir / f"{self.scion_id}_biases.pt")

        # Save full deltas for potential retraining
        deltas_dict = {
            f"layer{wd.layer_index}_{wd.component}": wd.delta
            for wd in self.weight_deltas
        }
        torch.save(deltas_dict, output_dir / f"{self.scion_id}_deltas.pt")


class ScionTrainer:
    """
    Trains a scion by:
    1. Snapshotting cleft weights before training
    2. Training with cleft-aware freezing
    3. Computing deltas after training
    4. Creating the new neuron with bias magnitudes
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        union_cleft: UnionCleft,
        config: Optional[ScionConfig] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.union_cleft = union_cleft
        self.config = config or ScionConfig()
        self.device = device

        # Will be populated during training
        self._weight_snapshots: Dict[str, torch.Tensor] = {}
        self._freezer: Optional[CleftAwareFreezer] = None

    def _snapshot_cleft_weights(self):
        """Take a snapshot of all weights in the cleft before training."""
        self._weight_snapshots = {}

        for layer_idx in self.union_cleft.get_all_layers():
            layer = _get_layer(self.model, layer_idx)
            if layer is None:
                continue

            for component_name in self.union_cleft.get_components_for_layer(layer_idx):
                component = _get_component(layer, component_name)
                if component is None or not hasattr(component, 'weight'):
                    continue

                key = f"layer{layer_idx}_{component_name}"
                self._weight_snapshots[key] = component.weight.data.clone()

    def _compute_deltas(self) -> List[WeightDelta]:
        """Compute weight deltas after training."""
        deltas = []

        for layer_idx in self.union_cleft.get_all_layers():
            layer = _get_layer(self.model, layer_idx)
            if layer is None:
                continue

            for component_name in self.union_cleft.get_components_for_layer(layer_idx):
                component = _get_component(layer, component_name)
                if component is None or not hasattr(component, 'weight'):
                    continue

                key = f"layer{layer_idx}_{component_name}"
                if key not in self._weight_snapshots:
                    continue

                # Compute delta
                delta = component.weight.data - self._weight_snapshots[key]

                # Get the cleft mask
                mask = self.union_cleft.get_trainable_mask(
                    layer_idx,
                    component_name,
                    component.weight.shape
                ).to(self.device)

                # Zero out deltas outside the cleft (should be zero anyway, but ensure)
                delta = delta * mask.float()

                deltas.append(WeightDelta(
                    layer_index=layer_idx,
                    component=component_name,
                    delta=delta.cpu(),
                    cleft_mask=mask.cpu()
                ))

        return deltas

    def _create_neuron_biases(self, deltas: List[WeightDelta]) -> Dict[str, torch.Tensor]:
        """
        Create bias vectors for the new neuron based on training deltas.

        The bias for each feature is proportional to how much that feature
        changed during training - encoding "how much does this concept
        relate to this feature".
        """
        biases = {}

        for delta in deltas:
            key = f"layer{delta.layer_index}_{delta.component}"

            # For each weight matrix, compute per-row and per-column magnitudes
            # These represent how much each input/output feature was affected

            # Row magnitudes: how much each output feature changed
            row_magnitudes = torch.norm(delta.delta, dim=1)

            # Col magnitudes: how much each input feature contributed
            col_magnitudes = torch.norm(delta.delta, dim=0)

            # Threshold small values
            row_magnitudes[row_magnitudes < self.config.delta_threshold] = 0
            col_magnitudes[col_magnitudes < self.config.delta_threshold] = 0

            biases[f"{key}_row"] = row_magnitudes
            biases[f"{key}_col"] = col_magnitudes

        return biases

    def train(
        self,
        dataset: Dict[str, List[str]],
        concept_id: str,
        verbose: bool = True
    ) -> Scion:
        """
        Train a scion on the given dataset.

        Args:
            dataset: Dict with "positive" and "negative" examples
            concept_id: ID for the new concept
            verbose: Print training progress

        Returns:
            Trained Scion ready for application
        """
        if verbose:
            print(f"Training scion for concept: {concept_id}")
            print(f"  Cleft concepts: {self.union_cleft.concept_ids}")

        # Step 1: Snapshot weights before training
        if verbose:
            print("  Snapshotting cleft weights...")
        self._snapshot_cleft_weights()

        # Step 2: Set up cleft-aware freezing
        self._freezer = CleftAwareFreezer(self.model, self.union_cleft)
        self._freezer.freeze()

        if verbose:
            print(f"  Trainable params: {self._freezer.get_trainable_param_count():,}")
            print(f"  Frozen params: {self._freezer.get_frozen_param_count():,}")

        # Step 3: Training loop
        positive_texts = dataset.get("positive", [])
        negative_texts = dataset.get("negative", [])

        if not positive_texts or not negative_texts:
            raise ValueError("Dataset must contain positive and negative examples")

        # Simple contrastive training: maximize activation for positive, minimize for negative
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.model.train()
        training_losses = []

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Process in batches
            for i in range(0, min(len(positive_texts), len(negative_texts)), self.config.batch_size):
                pos_batch = positive_texts[i:i + self.config.batch_size]
                neg_batch = negative_texts[i:i + self.config.batch_size]

                # Encode
                pos_inputs = self.tokenizer(
                    pos_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)

                neg_inputs = self.tokenizer(
                    neg_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)

                optimizer.zero_grad()

                # Forward pass
                pos_outputs = self.model(**pos_inputs, output_hidden_states=True)
                neg_outputs = self.model(**neg_inputs, output_hidden_states=True)

                # Get hidden states from a middle layer
                layer_idx = self.config.injection_layers[0] if self.config.injection_layers else 18
                layer_idx = min(layer_idx, len(pos_outputs.hidden_states) - 1)

                pos_hidden = pos_outputs.hidden_states[layer_idx]
                neg_hidden = neg_outputs.hidden_states[layer_idx]

                # Mean pool over sequence
                pos_mask = pos_inputs.attention_mask.unsqueeze(-1).float()
                neg_mask = neg_inputs.attention_mask.unsqueeze(-1).float()

                pos_pooled = (pos_hidden * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
                neg_pooled = (neg_hidden * neg_mask).sum(dim=1) / neg_mask.sum(dim=1)

                # Contrastive loss: push positive and negative apart
                # Using margin loss on the difference of norms
                pos_norm = pos_pooled.norm(dim=-1)
                neg_norm = neg_pooled.norm(dim=-1)

                # Also use cosine similarity between pos/neg pairs
                if len(pos_pooled) == len(neg_pooled):
                    cos_sim = F.cosine_similarity(pos_pooled, neg_pooled, dim=-1)
                    loss = cos_sim.mean() + 0.1  # Want similarity to be negative
                else:
                    loss = torch.tensor(0.0, device=self.device)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            training_losses.append(avg_loss)

            if verbose:
                print(f"  Epoch {epoch + 1}/{self.config.epochs}: loss={avg_loss:.4f}")

        self.model.eval()

        # Step 4: Remove freezing
        self._freezer.unfreeze()

        # Step 5: Compute deltas
        if verbose:
            print("  Computing weight deltas...")
        deltas = self._compute_deltas()

        total_delta_mag = sum(d.magnitude for d in deltas)
        if verbose:
            print(f"  Total delta magnitude: {total_delta_mag:.4f}")
            for d in deltas:
                print(f"    {d.layer_index}.{d.component}: mag={d.magnitude:.4f}, sparsity={d.sparsity:.2%}")

        # Step 6: Create neuron biases
        if verbose:
            print("  Creating neuron biases...")
        neuron_biases = self._create_neuron_biases(deltas)

        # Step 7: Create Scion
        hidden_dim = self.model.config.hidden_size
        scion = Scion(
            scion_id=f"scion-{concept_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            concept_id=concept_id,
            weight_deltas=deltas,
            neuron_index=hidden_dim,  # Will be the next dimension
            neuron_biases=neuron_biases,
            source_cleft_concepts=self.union_cleft.concept_ids,
            training_config=self.config,
            metrics={
                "final_loss": training_losses[-1] if training_losses else 0.0,
                "total_delta_magnitude": total_delta_mag,
                "trainable_params": self._freezer.get_trainable_param_count() if self._freezer else 0,
                "epochs": self.config.epochs
            }
        )

        if verbose:
            print(f"  Scion created: {scion.scion_id}")

        return scion


def apply_scion(
    model: nn.Module,
    scion: Scion,
    mode: str = "delta"
) -> nn.Module:
    """
    Apply a scion to a model, permanently modifying it.

    Modes:
    - "delta": Add the training deltas back to the weights
    - "expand": Actually expand hidden_dim and add new neuron (complex)

    For initial testing, we use "delta" mode which re-applies the
    weight changes from training. This is equivalent to keeping the
    model in its post-training state.

    Args:
        model: The model to modify
        scion: The scion to apply
        mode: Application mode

    Returns:
        The modified model
    """
    if mode == "delta":
        # Simply add the deltas back to the weights
        for delta in scion.weight_deltas:
            layer = _get_layer(model, delta.layer_index)
            if layer is None:
                continue

            component = _get_component(layer, delta.component)
            if component is None or not hasattr(component, 'weight'):
                continue

            with torch.no_grad():
                component.weight.data += delta.delta.to(component.weight.device)

        scion.applied = True
        logger.info(f"Applied scion {scion.scion_id} in delta mode")

    elif mode == "expand":
        # Full dimension expansion
        from .expand import plan_expansion, execute_expansion

        # Create expansion plan
        plan = plan_expansion(model, scion, target_layers=scion.training_config.injection_layers)

        # Execute the expansion
        execute_expansion(model, plan, device=str(next(model.parameters()).device))

        scion.applied = True
        logger.info(f"Applied scion {scion.scion_id} in expand mode (hidden_dim +1)")

    else:
        raise ValueError(f"Unknown apply mode: {mode}")

    return model


def revert_scion(model: nn.Module, scion: Scion) -> nn.Module:
    """
    Revert a scion by subtracting its deltas.

    Only works for scions applied in "delta" mode.
    """
    if not scion.applied:
        logger.warning(f"Scion {scion.scion_id} was not applied, nothing to revert")
        return model

    for delta in scion.weight_deltas:
        layer = _get_layer(model, delta.layer_index)
        if layer is None:
            continue

        component = _get_component(layer, delta.component)
        if component is None or not hasattr(component, 'weight'):
            continue

        with torch.no_grad():
            component.weight.data -= delta.delta.to(component.weight.device)

    scion.applied = False
    logger.info(f"Reverted scion {scion.scion_id}")

    return model
