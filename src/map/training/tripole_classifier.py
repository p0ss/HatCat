"""
Tripole lens for joint three-pole simplex training.

This module implements a psychologically realistic approach to simplex pole detection:
- Joint 3-class softmax (poles compete naturally)
- Margin-based discriminability loss (not hard orthogonality)
- Learnable per-pole margins (adaptive to difficulty)
- Optional overlap-aware contrastive loss for shared concepts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TripoleLens(nn.Module):
    """
    Joint three-pole linear lens with learnable margins.

    Learns a single shared weight matrix W: [3, hidden_dim] that maps
    representations to logits for each pole. Poles compete via softmax,
    with learnable margins allowing different discriminability thresholds.
    """

    def __init__(self, hidden_dim: int, n_poles: int = 3, init_margin: float = 0.5):
        """
        Args:
            hidden_dim: Dimension of input activations
            n_poles: Number of poles (default 3 for simplexes)
            init_margin: Initial margin value (in log space)
        """
        super().__init__()
        assert n_poles == 3, "TripoleLens designed for 3-pole simplexes"
        self.n_poles = n_poles
        self.hidden_dim = hidden_dim

        # Shared linear projection: h -> logits
        self.linear = nn.Linear(hidden_dim, n_poles)

        # Learnable per-pole margins (log-space for positivity)
        # Initialize to log(init_margin) so exp() gives desired starting value
        self.log_margins = nn.Parameter(torch.full((n_poles,), init_margin).log())

    def forward(self, h: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass with optional temperature scaling.

        Args:
            h: Input activations [batch_size, hidden_dim]
            temperature: Softmax temperature (lower = sharper boundaries)

        Returns:
            logits: [batch_size, 3]
        """
        logits = self.linear(h)
        if temperature != 1.0:
            logits = logits / temperature
        return logits

    def get_margins(self) -> torch.Tensor:
        """Get current margin values (always positive)."""
        return torch.exp(self.log_margins)

    def predict(self, h: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict pole assignments with confidence scores.

        Args:
            h: Input activations [batch_size, hidden_dim]
            temperature: Softmax temperature

        Returns:
            predictions: [batch_size] - predicted pole indices
            confidences: [batch_size, 3] - probability distribution over poles
        """
        logits = self.forward(h, temperature=temperature)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds, probs


def tripole_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lens: TripoleLens,
    margin: Optional[float] = None,
    lambda_margin: float = 0.5,
    lambda_ortho: float = 1e-4,
    lambda_overlap: float = 0.0,
    overlap_indices: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss for tripole training.

    Components:
    1. Cross-entropy: Standard multiclass classification
    2. Margin loss: Ensures correct pole dominates by adaptive margin
    3. Orthogonality: Soft regularizer encouraging diverse representations
    4. Overlap loss: Contrastive loss for shared concepts (optional)

    Args:
        logits: [batch, 3] - model outputs
        labels: [batch] - true pole indices (0, 1, or 2)
        lens: TripoleLens instance
        margin: Fixed margin (if None, uses learnable margins)
        lambda_margin: Weight for margin loss
        lambda_ortho: Weight for orthogonality regularizer
        lambda_overlap: Weight for overlap contrastive loss
        overlap_indices: Dict with 'core_pos' and 'core_neg' for overlap samples

    Returns:
        loss: Combined loss scalar
        metrics: Dict of component losses for logging
    """
    batch_size = logits.size(0)
    device = logits.device

    metrics = {}

    # ========================================================================
    # 1. Standard cross-entropy
    # ========================================================================
    ce_loss = F.cross_entropy(logits, labels)
    metrics['ce'] = ce_loss.item()

    total_loss = ce_loss

    # ========================================================================
    # 2. Margin loss for discriminability
    # ========================================================================
    if lambda_margin > 0.0:
        # Get margin for each sample based on its true label
        if margin is not None:
            # Fixed margin for all poles
            margins_per_sample = torch.full((batch_size,), margin, device=device)
        else:
            # Learnable per-pole margins
            pole_margins = lens.get_margins()  # [3]
            margins_per_sample = pole_margins[labels]  # [batch]

        # Get correct class logit for each sample
        batch_indices = torch.arange(batch_size, device=device)
        z_correct = logits[batch_indices, labels].unsqueeze(-1)  # [batch, 1]

        # Create mask for correct class
        mask = F.one_hot(labels, num_classes=3).bool()  # [batch, 3]

        # Get logits for other classes (mask out correct)
        z_others = logits.masked_fill(mask, float('-inf'))  # [batch, 3]

        # Margin violation: margin - (z_correct - z_other)
        # Want z_correct > z_other + margin
        margin_violations = margins_per_sample.unsqueeze(-1) - (z_correct - z_others)
        margin_violations = torch.clamp(margin_violations, min=0.0)

        # Clean up inf values from masking
        margin_violations = margin_violations.masked_fill(~torch.isfinite(margin_violations), 0.0)

        margin_loss = margin_violations.mean()
        metrics['margin'] = margin_loss.item()

        total_loss = total_loss + lambda_margin * margin_loss

    # ========================================================================
    # 3. Orthogonality regularizer (soft encouragement)
    # ========================================================================
    if lambda_ortho > 0.0:
        # W: [3, hidden_dim]
        W = lens.linear.weight

        # Gram matrix: W @ W^T = [3, 3]
        G = W @ W.t()

        # Penalize deviation from identity (orthonormal)
        I = torch.eye(3, device=device)
        ortho_loss = torch.norm(G - I, p='fro') ** 2
        metrics['ortho'] = ortho_loss.item()

        total_loss = total_loss + lambda_ortho * ortho_loss

    # ========================================================================
    # 4. Overlap contrastive loss (for shared concepts)
    # ========================================================================
    if lambda_overlap > 0.0 and overlap_indices is not None:
        overlap_terms = []

        # overlap_indices structure:
        # {
        #   'core_pos': {pole_idx: [sample_indices]},
        #   'core_neg': {(pole_i, pole_j): [sample_indices]}
        # }

        core_pos = overlap_indices.get('core_pos', {})
        core_neg = overlap_indices.get('core_neg', {})

        # For positive core: ensure pole P dominates all others Q
        for p, idx_pos in core_pos.items():
            if len(idx_pos) == 0:
                continue

            idx_tensor = torch.tensor(idx_pos, device=device, dtype=torch.long)
            z = logits[idx_tensor]  # [n_pos, 3]
            z_p = z[:, p].unsqueeze(-1)  # [n_pos, 1]

            for q in range(3):
                if q == p:
                    continue
                z_q = z[:, q:q+1]  # [n_pos, 1]

                # Want z_p > z_q + margin
                pole_margin = lens.get_margins()[p] if margin is None else margin
                violation = pole_margin - (z_p - z_q)
                violation = torch.clamp(violation, min=0.0)
                overlap_terms.append(violation.mean())

        # For negative core: ensure pole Q dominates P
        for (p, q), idx_neg in core_neg.items():
            if len(idx_neg) == 0:
                continue

            idx_tensor = torch.tensor(idx_neg, device=device, dtype=torch.long)
            z = logits[idx_tensor]  # [n_neg, 3]
            z_p = z[:, p:p+1]  # [n_neg, 1]
            z_q = z[:, q:q+1]  # [n_neg, 1]

            # Want z_q > z_p + margin
            pole_margin = lens.get_margins()[q] if margin is None else margin
            violation = pole_margin - (z_q - z_p)
            violation = torch.clamp(violation, min=0.0)
            overlap_terms.append(violation.mean())

        if overlap_terms:
            overlap_loss = torch.stack(overlap_terms).mean()
            metrics['overlap'] = overlap_loss.item()
            total_loss = total_loss + lambda_overlap * overlap_loss

    metrics['total'] = total_loss.item()

    return total_loss, metrics


def compute_tripole_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for tripole predictions.

    Args:
        logits: [batch, 3]
        labels: [batch]

    Returns:
        metrics: Dict with accuracy, per-pole precision/recall/f1
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)

        # Overall accuracy
        accuracy = (preds == labels).float().mean().item()

        metrics = {'accuracy': accuracy}

        # Per-pole metrics
        for pole in range(3):
            pole_mask = (labels == pole)
            if pole_mask.sum() == 0:
                continue

            pred_mask = (preds == pole)

            # True positives
            tp = ((preds == pole) & (labels == pole)).sum().item()
            fp = ((preds == pole) & (labels != pole)).sum().item()
            fn = ((preds != pole) & (labels == pole)).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            metrics[f'pole_{pole}_precision'] = precision
            metrics[f'pole_{pole}_recall'] = recall
            metrics[f'pole_{pole}_f1'] = f1

        # Average F1 across poles
        pole_f1s = [metrics[f'pole_{p}_f1'] for p in range(3) if f'pole_{p}_f1' in metrics]
        if pole_f1s:
            metrics['avg_f1'] = sum(pole_f1s) / len(pole_f1s)

        return metrics


def train_tripole_simplex(
    train_activations: torch.Tensor,
    train_labels: torch.Tensor,
    test_activations: torch.Tensor,
    test_labels: torch.Tensor,
    hidden_dim: int,
    device: str = 'cuda',
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
    lambda_margin: float = 0.5,
    lambda_ortho: float = 1e-4,
    lambda_overlap: float = 0.0,
    overlap_indices: Optional[Dict] = None,
) -> Tuple[TripoleLens, Dict]:
    """
    Train a tripole lens with early stopping.

    Args:
        train_activations: [n_train, hidden_dim]
        train_labels: [n_train] - pole indices (0, 1, 2)
        test_activations: [n_test, hidden_dim]
        test_labels: [n_test]
        hidden_dim: Activation dimension
        device: 'cuda' or 'cpu'
        lr: Learning rate
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        lambda_margin: Margin loss weight
        lambda_ortho: Orthogonality regularizer weight
        lambda_overlap: Overlap contrastive loss weight
        overlap_indices: Optional overlap indices for contrastive loss

    Returns:
        lens: Trained TripoleLens
        history: Training history dict
    """
    # Create lens
    lens = TripoleLens(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(lens.parameters(), lr=lr)

    # Move data to device
    train_activations = train_activations.to(device)
    train_labels = train_labels.to(device)
    test_activations = test_activations.to(device)
    test_labels = test_labels.to(device)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'test_loss': [],
        'test_acc': [],
        'test_f1': [],
        'margins': [],
    }

    best_test_f1 = 0.0
    best_lens_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # ====================================================================
        # Training step
        # ====================================================================
        lens.train()
        optimizer.zero_grad()

        logits = lens(train_activations)
        loss, loss_metrics = tripole_loss(
            logits=logits,
            labels=train_labels,
            lens=lens,
            lambda_margin=lambda_margin,
            lambda_ortho=lambda_ortho,
            lambda_overlap=lambda_overlap,
            overlap_indices=overlap_indices,
        )

        loss.backward()
        optimizer.step()

        # Training metrics
        train_metrics = compute_tripole_metrics(logits.detach(), train_labels)

        # ====================================================================
        # Evaluation step
        # ====================================================================
        lens.eval()
        with torch.no_grad():
            test_logits = lens(test_activations)
            test_loss, test_loss_metrics = tripole_loss(
                logits=test_logits,
                labels=test_labels,
                lens=lens,
                lambda_margin=lambda_margin,
                lambda_ortho=lambda_ortho,
                lambda_overlap=0.0,  # No overlap loss for test
                overlap_indices=None,
            )
            test_metrics = compute_tripole_metrics(test_logits, test_labels)

        # Record history
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics.get('avg_f1', 0.0))
        history['test_loss'].append(test_loss.item())
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_f1'].append(test_metrics.get('avg_f1', 0.0))
        history['margins'].append(lens.get_margins().cpu().tolist())

        # Early stopping check
        current_test_f1 = test_metrics.get('avg_f1', 0.0)
        if current_test_f1 > best_test_f1:
            best_test_f1 = current_test_f1
            best_lens_state = lens.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            margins_str = ', '.join([f'{m:.3f}' for m in lens.get_margins().cpu().tolist()])
            print(f"  Epoch {epoch+1:3d}: "
                  f"train_f1={train_metrics.get('avg_f1', 0.0):.3f}, "
                  f"test_f1={current_test_f1:.3f}, "
                  f"margins=[{margins_str}]")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best test_f1={best_test_f1:.3f})")
            break

    # Restore best model
    if best_lens_state is not None:
        lens.load_state_dict(best_lens_state)

    # Final evaluation
    lens.eval()
    with torch.no_grad():
        final_logits = lens(test_activations)
        final_metrics = compute_tripole_metrics(final_logits, test_labels)

    history['best_test_f1'] = best_test_f1
    history['final_metrics'] = final_metrics

    return lens, history
