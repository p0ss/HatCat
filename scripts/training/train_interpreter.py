"""
Train semantic interpreter on Stage 0 encyclopedia.
PyTorch Lightning-based training pipeline.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import h5py
import numpy as np
from pathlib import Path
import argparse
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hat.interpreter.model import SemanticInterpreter


class EncyclopediaDataset(Dataset):
    """Dataset for loading encyclopedia activations."""

    def __init__(
        self,
        h5_path: Path,
        layer: int = -1,
        split: str = 'train',
        train_ratio: float = 0.8,
        normalize: bool = True
    ):
        """
        Initialize dataset.

        Args:
            h5_path: Path to HDF5 encyclopedia
            layer: Layer index to use
            split: 'train', 'val', or 'test'
            train_ratio: Ratio of data for training
            normalize: Whether to normalize activations
        """
        self.h5_path = h5_path
        self.layer = layer
        self.split = split
        self.normalize = normalize

        with h5py.File(h5_path, 'r') as f:
            self.n_concepts = f.attrs['n_concepts']
            self.concepts = f['concepts'][:].astype(str)

            # Load all activations into memory (for speed)
            layer_key = f'layer_{layer}'
            acts = f[f'{layer_key}/activations'][:].astype(np.float32)

            # Check if this is Stage 1+ (multiple samples) or Stage 0 (single sample)
            if f'{layer_key}/labels' in f:
                # Stage 1+: Has multiple samples per concept with labels
                labels = f[f'{layer_key}/labels'][:].astype(np.int64)
                samples_per_concept = f.attrs.get('samples_per_concept', 1)

                print(f"  Stage 1+ dataset: {len(acts)} samples, {self.n_concepts} concepts, ~{samples_per_concept} samples/concept")

                # Split SAMPLES within each concept (not concepts themselves)
                # This tests if the model can recognize a concept from different contexts
                indices = np.arange(len(acts))
                train_indices = []
                val_indices = []
                test_indices = []

                for concept_id in range(self.n_concepts):
                    concept_mask = labels == concept_id
                    concept_indices = indices[concept_mask]

                    # Split this concept's samples
                    n_samples = len(concept_indices)
                    n_train = int(n_samples * train_ratio)
                    n_val = (n_samples - n_train) // 2

                    train_indices.extend(concept_indices[:n_train])
                    val_indices.extend(concept_indices[n_train:n_train+n_val])
                    test_indices.extend(concept_indices[n_train+n_val:])

                if split == 'train':
                    mask = np.isin(indices, train_indices)
                elif split == 'val':
                    mask = np.isin(indices, val_indices)
                else:  # test
                    mask = np.isin(indices, test_indices)

                self.activations = acts[mask]
                self.labels = labels[mask]

                print(f"  Split: Train has {(self.labels[:, None] == np.arange(self.n_concepts)).any(axis=0).sum()}/{self.n_concepts} concepts with samples")

            else:
                # Stage 0: Single sample per concept
                print(f"  Stage 0 dataset: {len(acts)} samples (1 per concept)")

                # Split data
                # For Stage 0: 1 sample per concept, so we split concepts into train/val/test
                # The model should learn ALL concepts (full num_concepts output)
                # But we only train on some and validate on held-out concepts
                n_train = int(self.n_concepts * train_ratio)
                n_val = (self.n_concepts - n_train) // 2

                if split == 'train':
                    self.activations = acts[:n_train]
                    self.labels = np.arange(n_train)  # Labels 0 to n_train-1
                elif split == 'val':
                    self.activations = acts[n_train:n_train+n_val]
                    self.labels = np.arange(n_train, n_train+n_val)  # Labels n_train to n_train+n_val-1
                else:  # test
                    self.activations = acts[n_train+n_val:]
                    self.labels = np.arange(n_train+n_val, self.n_concepts)  # Labels n_train+n_val to n_concepts-1

                # IMPORTANT: Model must have num_concepts outputs (not just n_train)
                # This means val/test concepts are "zero-shot" - never seen during training
                # This tests generalization!

            # Normalize if requested
            if self.normalize:
                mean = self.activations.mean(axis=0, keepdims=True)
                std = self.activations.std(axis=0, keepdims=True) + 1e-8
                self.activations = (self.activations - mean) / std
                self.mean = mean
                self.std = std

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        activation = torch.from_numpy(self.activations[idx])
        label = self.labels[idx]
        return activation, label


class InterpreterTrainer(pl.LightningModule):
    """PyTorch Lightning module for training interpreter."""

    def __init__(
        self,
        hidden_dim: int,
        num_concepts: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500
    ):
        """
        Initialize trainer.

        Args:
            hidden_dim: Input activation dimension
            num_concepts: Number of concepts in encyclopedia
            d_model: Transformer dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Learning rate warmup steps
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = SemanticInterpreter(
            hidden_dim=hidden_dim,
            num_concepts=num_concepts,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # For tracking best metrics
        self.best_val_acc = 0.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        activations, labels = batch
        logits, confidence = self.model(activations)

        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_confidence', confidence.mean(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        activations, labels = batch
        logits, confidence = self.model(activations)

        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        # Top-5 accuracy
        top5_preds = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices
        top5_acc = (top5_preds == labels.unsqueeze(-1)).any(dim=-1).float().mean()

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_top5_acc', top5_acc, on_step=False, on_epoch=True)
        self.log('val_confidence', confidence.mean(), on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        activations, labels = batch
        logits, confidence = self.model(activations)

        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_confidence', confidence.mean())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos'
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Train semantic interpreter")

    # Data args
    parser.add_argument('--data', type=str, required=True,
                       help='Path to encyclopedia HDF5 file')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Layer index to use (default: -1)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Train/val/test split ratio (default: 0.8)')

    # Model args
    parser.add_argument('--d-model', type=int, default=512,
                       help='Transformer model dimension (default: 512)')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of transformer layers (default: 4)')

    # Training args
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')

    # System args
    parser.add_argument('--accelerator', type=str, default='auto',
                       help='Accelerator (auto/gpu/cpu, default: auto)')
    parser.add_argument('--devices', type=int, default=1,
                       help='Number of devices (default: 1)')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory (default: models)')

    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING SEMANTIC INTERPRETER")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Layer: {args.layer}")
    print(f"Model: d_model={args.d_model}, nhead={args.nhead}, layers={args.num_layers}")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print()

    # Load dataset info
    with h5py.File(args.data, 'r') as f:
        n_concepts = f.attrs['n_concepts']
        hidden_dim = f.attrs['activation_dim']
        print(f"Encyclopedia: {n_concepts} concepts, {hidden_dim}-dim activations")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = EncyclopediaDataset(
        args.data,
        layer=args.layer,
        split='train',
        train_ratio=args.train_ratio
    )
    val_dataset = EncyclopediaDataset(
        args.data,
        layer=args.layer,
        split='val',
        train_ratio=args.train_ratio
    )
    test_dataset = EncyclopediaDataset(
        args.data,
        layer=args.layer,
        split='test',
        train_ratio=args.train_ratio
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    print("\nInitializing model...")
    model = InterpreterTrainer(
        hidden_dim=hidden_dim,
        num_concepts=n_concepts,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Callbacks
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='interpreter-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=5,
        mode='max',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Logger (optional - fallback to CSV if TensorBoard not available)
    try:
        logger = TensorBoardLogger('logs', name='interpreter')
    except (ModuleNotFoundError, ImportError):
        print("Warning: TensorBoard not available, using CSV logger")
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger('logs', name='interpreter')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
        deterministic=False,
        precision='16-mixed' if args.accelerator == 'gpu' else '32'
    )

    # Train
    print("\nTraining...")
    print("=" * 70)
    trainer.fit(model, train_loader, val_loader)

    # Test
    print("\n" + "=" * 70)
    print("Testing on held-out set...")
    trainer.test(model, test_loader, ckpt_path='best')

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val accuracy: {checkpoint_callback.best_model_score:.4f}")

    # Stage 0 explanation
    if checkpoint_callback.best_model_score < 0.01:
        print("\n" + "=" * 70)
        print("NOTE: Stage 0 Validation")
        print("=" * 70)
        print("Val accuracy is 0% - this is EXPECTED for Stage 0!")
        print("Reason: Only 1 sample per concept, so val concepts are completely unseen.")
        print("The model learns train concepts but can't generalize to new ones.")
        print("\nThis validates the need for Stage 1 (multi-sample refinement):")
        print("  - Stage 1: Add 5-10 samples per concept → enables generalization")
        print("  - With multiple samples, model learns concept patterns, not memorization")
        print(f"\nTrain accuracy reached: {model.trainer.callback_metrics.get('train_acc_epoch', 0):.1%}")
        print("This shows the model CAN learn when it has training data.")


if __name__ == "__main__":
    main()
