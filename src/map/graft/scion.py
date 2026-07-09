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
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import copy

from .cleft import Cleft, UnionCleft, CleftAwareFreezer, merge_clefts, _get_layer, _get_component, _get_model_layers

# Import classifier utilities for bound lens training
from ...hat.classifiers.classifier import MLPClassifier, save_classifier

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
    mode: str = "delta",
    # New parameters for expand mode with lens training:
    tokenizer: Any = None,
    training_data: Optional[Dict[str, List[str]]] = None,
    auxiliary_dimensions: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    device: str = "cuda",
) -> Union[nn.Module, Tuple[nn.Module, Optional[Path]]]:
    """
    Apply a scion to a model, permanently modifying it.

    Modes:
    - "delta": Add the training deltas back to the weights
    - "expand": Actually expand hidden_dim and add new neuron

    For expand mode, if training_data is provided, also trains a bound
    lens that monitors the new dimension.

    Args:
        model: The model to modify
        scion: The scion to apply
        mode: Application mode ("delta" or "expand")
        tokenizer: Tokenizer (required for expand mode with lens training)
        training_data: Dict with "positive" and "negative" examples for lens training
        auxiliary_dimensions: Additional dimensions for the bound lens to read
        output_dir: Directory to save the bound lens
        device: Device for lens training

    Returns:
        For backward compatibility:
        - If training_data is provided: Tuple[nn.Module, Optional[Path]] with
          (modified_model, lens_path)
        - Otherwise: nn.Module (just the modified model)
    """
    lens_path = None

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

        # Create expansion plan for ALL layers (required for model consistency)
        # The scion's injection_layers only affect biased initialization, not which
        # layers get expanded - all layers must expand for the model to be valid
        plan = plan_expansion(model, scion, target_layers=None)

        # Execute the expansion
        execute_expansion(model, plan, device=str(next(model.parameters()).device))

        scion.applied = True
        logger.info(f"Applied scion {scion.scion_id} in expand mode (hidden_dim +1)")

        # Train bound lens if training data provided
        if training_data is not None and tokenizer is not None:
            logger.info("Training bound lens for expanded dimension...")
            lens_path = train_bound_lens(
                model=model,
                tokenizer=tokenizer,
                scion=scion,
                dataset=training_data,
                auxiliary_dimensions=auxiliary_dimensions,
                output_dir=output_dir,
                device=device,
            )
        elif training_data is not None:
            logger.warning("training_data provided but tokenizer is None; skipping lens training")

    else:
        raise ValueError(f"Unknown apply mode: {mode}")

    # Return tuple only when training_data is provided (new behavior)
    # Otherwise return just the model for backward compatibility
    if training_data is not None:
        return model, lens_path
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


def derive_auxiliary_dimensions(scion: Scion, top_k: int = 10) -> List[int]:
    """
    Automatically derive auxiliary dimensions from a scion's training data.

    This function analyzes the scion's neuron_biases and/or weight_deltas
    to identify which dimensions in the model's hidden state are most
    related to the concept this scion represents.

    Strategy:
    1. First, examine neuron_biases to find dimensions with highest magnitude
       changes (these indicate features strongly affected during training)
    2. If neuron_biases are insufficient, fall back to extract_expand_metadata()
       to get top_features from weight deltas

    Args:
        scion: The trained Scion object
        top_k: Number of top dimension indices to return

    Returns:
        List of dimension indices (up to top_k) most related to the concept
    """
    from .expand import extract_expand_metadata

    dimension_scores: Dict[int, float] = {}

    # Strategy 1: Extract from neuron_biases
    # neuron_biases has keys like "layer{idx}_{component}_row" or "_col"
    # Row biases indicate output feature importance
    for key, bias_tensor in scion.neuron_biases.items():
        # We care about row biases which represent hidden_dim features
        if not key.endswith("_row"):
            continue

        # bias_tensor has shape (hidden_dim,) or (output_dim,)
        # For down_proj, this represents hidden_dim output features
        if "down_proj" in key or "o_proj" in key:
            # These output to hidden_dim, so indices correspond to hidden dimensions
            for idx in range(len(bias_tensor)):
                magnitude = abs(float(bias_tensor[idx]))
                if magnitude > 0:
                    # Accumulate scores across layers/components
                    dimension_scores[idx] = dimension_scores.get(idx, 0.0) + magnitude

    # Strategy 2: If we didn't get enough from neuron_biases, use weight_deltas
    if len(dimension_scores) < top_k and scion.weight_deltas:
        try:
            expand_meta = extract_expand_metadata(scion)
            # top_features: Dict[int, List[Tuple[int, float]]]
            # Maps layer -> list of (dimension_idx, importance)
            for layer_idx, features in expand_meta.top_features.items():
                for dim_idx, importance in features:
                    dimension_scores[dim_idx] = dimension_scores.get(dim_idx, 0.0) + importance
        except Exception as e:
            logger.debug(f"Could not extract expand metadata: {e}")

    # Sort by score descending and take top_k
    if not dimension_scores:
        logger.warning("No dimension scores found; returning empty list")
        return []

    sorted_dims = sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
    top_dims = [dim_idx for dim_idx, score in sorted_dims[:top_k]]

    logger.info(f"Derived {len(top_dims)} auxiliary dimensions from scion training data")
    logger.debug(f"Top auxiliary dimensions: {top_dims}")

    return top_dims


def train_bound_lens(
    model: nn.Module,
    tokenizer: Any,
    scion: Scion,
    dataset: Dict[str, List[str]],
    auxiliary_dimensions: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    device: str = "cuda",
    epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
) -> Path:
    """
    Train a lens bound to the scion's new dimension.

    The lens reads from:
    - Primary: scion.neuron_index (the new dimension)
    - Auxiliary: top dimensions from cleft analysis (auto-derived if not provided)

    This creates a specialized classifier that monitors the new concept
    dimension along with auxiliary context dimensions. The classifier
    is saved in a format compatible with the HAT lens loading infrastructure.

    Output files:
    - {concept_id}_bound_lens.pt: MLPClassifier state dict (compatible with load_classifier)
    - {concept_id}_bound_lens_metadata.json: Bound lens metadata (primary/aux dimensions)

    Args:
        model: The model (already expanded)
        tokenizer: The tokenizer
        scion: The scion that was applied
        dataset: Dict with "positive" and "negative" examples
        auxiliary_dimensions: Additional dimension indices to read from.
            If None, automatically derived from the scion's training data
            using derive_auxiliary_dimensions().
        output_dir: Directory to save the lens (default: temp directory)
        device: Device for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size

    Returns:
        Path to saved lens weights (.pt file)
    """
    import tempfile

    positive_texts = dataset.get("positive", [])
    negative_texts = dataset.get("negative", [])

    if not positive_texts or not negative_texts:
        raise ValueError("Dataset must contain positive and negative examples")

    # Determine input dimensions for the lens
    # Primary dimension is the new neuron
    primary_dim = scion.neuron_index

    # Automatically derive auxiliary dimensions if not provided
    if auxiliary_dimensions is None:
        aux_dims = derive_auxiliary_dimensions(scion)
        logger.info(f"  Auto-derived {len(aux_dims)} auxiliary dimensions from training data")
    else:
        aux_dims = auxiliary_dimensions

    # Total input size for the lens classifier
    input_size = 1 + len(aux_dims)

    # Get the injection layer (first layer from config)
    injection_layer = scion.training_config.injection_layers[0] if scion.training_config.injection_layers else 18

    logger.info(f"Training bound lens for {scion.concept_id}")
    logger.info(f"  Primary dimension: {primary_dim}")
    logger.info(f"  Auxiliary dimensions: {aux_dims}")
    logger.info(f"  Injection layer: {injection_layer}")

    # Collect training features
    def extract_features(texts: List[str]) -> torch.Tensor:
        """Extract features from texts at the injection layer."""
        all_features = []

        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(device)

                outputs = model(**inputs, output_hidden_states=True)

                # Get hidden states at injection layer
                hidden = outputs.hidden_states[min(injection_layer, len(outputs.hidden_states) - 1)]

                # Mean pool over sequence
                mask = inputs.attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

                # Extract primary and auxiliary dimensions
                primary_feat = pooled[:, primary_dim:primary_dim + 1]
                if aux_dims:
                    aux_feat = pooled[:, aux_dims]
                    features = torch.cat([primary_feat, aux_feat], dim=-1)
                else:
                    features = primary_feat

                all_features.append(features.cpu())

        return torch.cat(all_features, dim=0)

    # Extract features
    logger.info("  Extracting features from positive examples...")
    pos_features = extract_features(positive_texts)

    logger.info("  Extracting features from negative examples...")
    neg_features = extract_features(negative_texts)

    # Create labels
    pos_labels = torch.ones(len(pos_features), 1)
    neg_labels = torch.zeros(len(neg_features), 1)

    # Combine into training set
    all_features = torch.cat([pos_features, neg_features], dim=0)
    all_labels = torch.cat([pos_labels, neg_labels], dim=0)

    # Shuffle
    perm = torch.randperm(len(all_features))
    all_features = all_features[perm]
    all_labels = all_labels[perm]

    # Create classifier using HAT MLPClassifier
    # For bound lenses with small input sizes, we use a smaller hidden_dim
    # to keep the architecture appropriate for the reduced input dimensionality
    hidden_dim = min(32, max(16, input_size * 4))  # Scale hidden size with input
    classifier = MLPClassifier(
        input_dim=input_size,
        hidden_dim=hidden_dim,
        dropout=0.1,
        layer_norm=True,
    ).to(device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    logger.info(f"  Training classifier for {epochs} epochs...")
    classifier.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(all_features), batch_size):
            batch_features = all_features[i:i + batch_size].to(device)
            batch_labels = all_labels[i:i + batch_size].to(device)

            optimizer.zero_grad()
            logits = classifier(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"    Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        all_preds = classifier.predict_proba(all_features.to(device))
        accuracy = ((all_preds > 0.5).float() == all_labels.to(device).squeeze(-1)).float().mean()
        logger.info(f"  Training accuracy: {accuracy:.4f}")

    # Save the classifier
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="bound_lens_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Use HAT's save_classifier for the weights (.pt file)
    lens_path = output_dir / f"{scion.concept_id}_bound_lens.pt"
    save_classifier(classifier, lens_path)

    # Save bound lens metadata as separate JSON (compatible with LensManager discovery)
    # This allows LensManager to load the lens and understand its special requirements
    metadata_path = output_dir / f"{scion.concept_id}_bound_lens_metadata.json"
    metadata = {
        "lens_type": "bound",
        "scion_id": scion.scion_id,
        "concept_id": scion.concept_id,
        "primary_dimension": primary_dim,
        "auxiliary_dimensions": aux_dims,
        "injection_layer": injection_layer,
        "input_size": input_size,
        "hidden_dim": hidden_dim,
        "technique": "mlp",
        "metrics": {
            "accuracy": float(accuracy.item()),
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "created_at": datetime.now().isoformat(),
        "source_cleft_concepts": scion.source_cleft_concepts,
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Bound lens saved to: {lens_path}")
    logger.info(f"  Metadata saved to: {metadata_path}")

    return lens_path


def load_bound_lens(
    lens_path: Path,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a bound lens and its metadata.

    This is a convenience function for loading bound lenses that were
    saved using train_bound_lens(). It loads both the classifier weights
    and the associated metadata.

    Args:
        lens_path: Path to the .pt file
        device: Device to load onto

    Returns:
        Tuple of (classifier, metadata) where metadata contains:
        - primary_dimension: The new neuron index
        - auxiliary_dimensions: Additional dimension indices
        - injection_layer: Model layer to extract activations from
        - input_size: Expected input dimension
        - Other training metadata
    """
    lens_path = Path(lens_path)

    # Load metadata first to get classifier architecture
    metadata_path = lens_path.parent / f"{lens_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        # Fallback: try to infer from state_dict
        logger.warning(f"Metadata file not found: {metadata_path}")
        metadata = None

    # Load the state dict
    state_dict = torch.load(lens_path, map_location=device, weights_only=True)

    # Create classifier with correct architecture from metadata
    if metadata and "input_size" in metadata:
        input_size = metadata["input_size"]
        hidden_dim = metadata.get("hidden_dim", min(32, max(16, input_size * 4)))
        classifier = MLPClassifier(
            input_dim=input_size,
            hidden_dim=hidden_dim,
            dropout=0.1,
            layer_norm=True,
        )
        classifier.load_state_dict(state_dict)
        classifier.to(device)
        classifier.eval()
    else:
        # Fallback to load_classifier for standard HAT format
        from ...hat.classifiers.classifier import load_classifier
        classifier = load_classifier(lens_path, device=device, classifier_type="mlp")
        metadata = {
            "lens_type": "bound",
            "concept_id": lens_path.stem.replace("_bound_lens", ""),
            "input_size": classifier.input_dim,
            "primary_dimension": None,
            "auxiliary_dimensions": [],
            "injection_layer": None,
        }

    return classifier, metadata
