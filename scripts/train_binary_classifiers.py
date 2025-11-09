"""
Train 50K Binary Classifiers for Temporal Concept Detection

Each classifier learns: "Is concept X active in this activation window?"
- Input: Activation sequence window [window_size, hidden_dim]
- Output: Probability that concept is active

Production usage:
- Sliding window over generation activations
- Run all 50K classifiers on each window
- Keep concepts above threshold (e.g., 0.5)
- Naturally handles polysemy (multiple concepts per window)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import h5py
import numpy as np
from pathlib import Path
import argparse
from typing import List


class SequenceDataset(Dataset):
    """Dataset for binary classification on activation sequences."""

    def __init__(self, h5_path: Path, concept_idx: int, layer: int = -1):
        """
        Load positive and negative sequences for a single concept.

        Args:
            h5_path: Path to Stage 1.5 temporal HDF5
            concept_idx: Which concept to load
            layer: Layer index
        """
        self.h5_path = h5_path
        self.concept_idx = concept_idx

        with h5py.File(h5_path, 'r') as f:
            layer_key = f'layer_{layer}'
            concept_grp = f[f'{layer_key}/concept_{concept_idx}']

            # Load sequences and lengths
            self.pos_seqs = concept_grp['positive_sequences'][:].astype(np.float32)
            self.neg_seqs = concept_grp['negative_sequences'][:].astype(np.float32)
            self.pos_lens = concept_grp['positive_lengths'][:]
            self.neg_lens = concept_grp['negative_lengths'][:]

        # Create labels: 1 for positive, 0 for negative
        self.sequences = np.concatenate([self.pos_seqs, self.neg_seqs], axis=0)
        self.lengths = np.concatenate([self.pos_lens, self.neg_lens], axis=0)
        self.labels = np.concatenate([
            np.ones(len(self.pos_seqs)),
            np.zeros(len(self.neg_seqs))
        ])

        # Shuffle
        indices = np.random.permutation(len(self.labels))
        self.sequences = self.sequences[indices]
        self.lengths = self.lengths[indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])
        length = self.lengths[idx]
        label = self.labels[idx]

        return seq, length, label


class BinaryConceptClassifier(nn.Module):
    """
    Binary classifier for single concept detection.

    Uses LSTM to process variable-length sequences.
    """

    def __init__(self, hidden_dim: int, lstm_dim: int = 256):
        super().__init__()

        self.lstm = nn.LSTM(
            hidden_dim,
            lstm_dim,
            batch_first=True,
            bidirectional=True
        )

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, sequences, lengths):
        """
        Args:
            sequences: [batch, max_seq_len, hidden_dim]
            lengths: [batch] - actual lengths before padding

        Returns:
            probs: [batch] - probability concept is active
        """
        # Pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        _, (hidden, _) = self.lstm(packed)

        # Concatenate forward and backward final states
        # hidden: [2, batch, lstm_dim]
        final_state = torch.cat([hidden[0], hidden[1]], dim=-1)  # [batch, lstm_dim*2]

        # Classify
        logits = self.classifier(final_state)  # [batch, 1]
        probs = logits.squeeze(-1)  # [batch]

        return probs


class BinaryClassifierTrainer(pl.LightningModule):
    """PyTorch Lightning module for training binary classifier."""

    def __init__(self, hidden_dim: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = BinaryConceptClassifier(hidden_dim)
        self.loss_fn = nn.BCELoss()

    def forward(self, sequences, lengths):
        return self.model(sequences, lengths)

    def training_step(self, batch, batch_idx):
        sequences, lengths, labels = batch
        probs = self.model(sequences, lengths)
        loss = self.loss_fn(probs, labels.float())

        # Accuracy
        preds = (probs > 0.5).float()
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        sequences, lengths, labels = batch
        probs = self.model(sequences, lengths)
        loss = self.loss_fn(probs, labels.float())

        preds = (probs > 0.5).float()
        acc = (preds == labels).float().mean()

        self.log('val_loss', loss)
        self.log('val_acc', acc)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def train_concept_classifier(
    h5_path: Path,
    concept_idx: int,
    output_path: Path,
    hidden_dim: int,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3
):
    """Train a single binary classifier for one concept."""

    # Create dataset
    dataset = SequenceDataset(h5_path, concept_idx)

    # Split train/val (80/20)
    n_train = int(len(dataset) * 0.8)
    train_dataset = torch.utils.data.Subset(dataset, range(n_train))
    val_dataset = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = BinaryClassifierTrainer(hidden_dim, lr)

    # Callbacks
    checkpoint = ModelCheckpoint(
        dirpath=output_path / f'concept_{concept_idx}',
        filename='best',
        monitor='val_acc',
        mode='max'
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint],
        logger=False,
        enable_progress_bar=False
    )

    trainer.fit(model, train_loader, val_loader)

    return checkpoint.best_model_score


def main():
    parser = argparse.ArgumentParser(description="Train binary concept classifiers")

    parser.add_argument('--data', type=str, required=True,
                       help='Path to Stage 1.5 temporal HDF5')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for classifiers')
    parser.add_argument('--concept-ids', type=str, default=None,
                       help='Comma-separated concept IDs to train (default: all)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs per classifier (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')

    args = parser.parse_args()

    h5_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with h5py.File(h5_path, 'r') as f:
        n_concepts = f.attrs['n_concepts']

        # Get actual hidden_dim from data (not attrs, which may be wrong)
        concept_grp = f['layer_-1/concept_0']
        actual_shape = concept_grp['positive_sequences'].shape
        hidden_dim = actual_shape[2]  # [n_samples, seq_len, hidden_dim]

        print(f"Detected hidden_dim from data: {hidden_dim}")

    # Determine which concepts to train
    if args.concept_ids:
        concept_ids = [int(x) for x in args.concept_ids.split(',')]
    else:
        concept_ids = list(range(n_concepts))

    print(f"Training {len(concept_ids)} binary classifiers...")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print()

    # Train each classifier
    results = {}
    for i, concept_idx in enumerate(concept_ids):
        print(f"[{i+1}/{len(concept_ids)}] Training concept {concept_idx}...", end=' ')

        acc = train_concept_classifier(
            h5_path, concept_idx, output_path, hidden_dim,
            args.epochs, args.batch_size, args.lr
        )

        results[concept_idx] = acc
        print(f"Val acc: {acc:.3f}")

    # Summary
    print()
    print("=" * 70)
    print("✓ TRAINING COMPLETE")
    print("=" * 70)
    print(f"Trained {len(results)} classifiers")
    print(f"Mean val accuracy: {np.mean(list(results.values())):.3f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
