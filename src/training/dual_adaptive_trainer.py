#!/usr/bin/env python3
"""
Dual Adaptive Trainer - Independent graduation for activation and text probes.

Each probe type trains until it reaches target accuracy, with separate:
- Baseline sample counts
- Increment sizes
- Maximum sample limits
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import time
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from .text_probes import BinaryTextProbe


class DualAdaptiveTrainer:
    """
    Train both activation and text probes with independent adaptive cycles.

    Each probe graduates independently when reaching target accuracy.
    """

    def __init__(
        self,
        # Shared config
        target_accuracy: float = 0.95,
        max_iterations: int = 50,

        # Activation probe config
        activation_baseline: int = 10,
        activation_increment: int = 1,
        activation_max_samples: int = 100,

        # Text probe config
        text_baseline: int = 10,
        text_increment: int = 5,  # Larger (text learns faster)
        text_max_samples: int = 200,  # Can handle more data

        # Flags
        train_activation: bool = True,
        train_text: bool = True,
    ):
        """
        Args:
            target_accuracy: Graduation threshold (both probe types)
            max_iterations: Safety limit on adaptive cycles
            activation_baseline: Starting samples for activation probe
            activation_increment: Samples added per cycle for activation
            activation_max_samples: Max samples for activation
            text_baseline: Starting samples for text probe
            text_increment: Samples added per cycle for text
            text_max_samples: Max samples for text
            train_activation: Whether to train activation probe
            train_text: Whether to train text probe
        """
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations

        self.activation_baseline = activation_baseline
        self.activation_increment = activation_increment
        self.activation_max_samples = activation_max_samples

        self.text_baseline = text_baseline
        self.text_increment = text_increment
        self.text_max_samples = text_max_samples

        self.train_activation = train_activation
        self.train_text = train_text

    def train_concept(
        self,
        concept_name: str,
        # Activation probe inputs
        train_activations: Optional[np.ndarray],
        train_labels: Optional[np.ndarray],
        test_activations: Optional[np.ndarray],
        test_labels: Optional[np.ndarray],
        # Text probe inputs
        train_texts: Optional[List[str]],
        test_texts: Optional[List[str]],
    ) -> Dict:
        """
        Train both probe types until graduation.

        Args:
            concept_name: Name of concept
            train_activations: Training activation vectors [n_samples, hidden_dim]
            train_labels: Training labels [n_samples]
            test_activations: Test activation vectors [n_test, hidden_dim]
            test_labels: Test labels [n_test]
            train_texts: Training text prompts
            test_texts: Test text prompts

        Returns:
            Dict with training results:
            {
                'activation': {...} or None,
                'text': {...} or None,
                'total_iterations': int,
                'total_time': float,
            }
        """
        start_time = time.time()

        # State
        activation_level = self.activation_baseline
        text_level = self.text_baseline
        activation_graduated = not self.train_activation  # Skip if disabled
        text_graduated = not self.train_text

        activation_classifier = None
        text_probe = None
        activation_results = None
        text_results = None

        iteration = 0

        print(f"\n  Dual Adaptive Training: {concept_name}")
        print(f"  {'─' * 60}")

        while not (activation_graduated and text_graduated) and iteration < self.max_iterations:
            iteration += 1

            # === ACTIVATION PROBE ===
            if not activation_graduated and train_activations is not None:
                # Get balanced samples (data is [pos1..posN, neg1..negN])
                n_samples_per_class = activation_level // 2

                # Find where negatives start
                neg_start = np.where(train_labels == 0)[0][0] if 0 in train_labels else len(train_labels)

                # Get balanced positives and negatives
                n_pos = min(n_samples_per_class, neg_start)
                n_neg = min(n_samples_per_class, len(train_labels) - neg_start)

                # Sample activations and labels
                X_train = np.vstack([
                    train_activations[:n_pos],
                    train_activations[neg_start:neg_start + n_neg]
                ])
                y_train = np.concatenate([
                    train_labels[:n_pos],
                    train_labels[neg_start:neg_start + n_neg]
                ])

                # Import here to avoid circular dependency
                from .sumo_classifiers import train_simple_classifier

                activation_classifier, metrics = train_simple_classifier(
                    X_train, y_train,
                    test_activations, test_labels,
                    hidden_dim=128,
                    epochs=50,
                    lr=0.001,
                )

                test_acc = metrics['test_f1']  # Use F1 as graduation metric

                n_samples = n_pos + n_neg

                # Check graduation
                if test_acc >= self.target_accuracy:
                    activation_graduated = True
                    activation_results = {
                        'samples': n_samples,
                        'iterations': iteration,
                        'test_f1': test_acc,
                        'test_precision': metrics['test_precision'],
                        'test_recall': metrics['test_recall'],
                    }
                    print(f"    [Iter {iteration:2d}] Activation: {n_samples:3d} samples ({n_pos}+{n_neg}), "
                          f"F1={test_acc:.3f} ✓ GRADUATED")
                else:
                    print(f"    [Iter {iteration:2d}] Activation: {n_samples:3d} samples ({n_pos}+{n_neg}), "
                          f"F1={test_acc:.3f}")

                # Increment for next iteration
                activation_level += self.activation_increment
                activation_level = min(activation_level, self.activation_max_samples)

            # === TEXT PROBE ===
            if not text_graduated and train_texts is not None:
                # Get balanced samples (data is [pos1..posN, neg1..negN])
                # Need to ensure we get both classes
                n_samples_per_class = text_level // 2

                # Find where negatives start (labels switch from 1 to 0)
                neg_start = np.where(train_labels == 0)[0][0] if 0 in train_labels else len(train_labels)

                # Get balanced positives and negatives
                n_pos = min(n_samples_per_class, neg_start)
                n_neg = min(n_samples_per_class, len(train_labels) - neg_start)

                # Sample texts and labels
                pos_texts = train_texts[:n_pos]
                neg_texts = train_texts[neg_start:neg_start + n_neg]
                train_texts_sample = pos_texts + neg_texts
                train_labels_sample = [1] * n_pos + [0] * n_neg

                # Train text probe
                text_probe = BinaryTextProbe(concept_name)
                metrics = text_probe.train(
                    train_texts=train_texts_sample,
                    train_labels=train_labels_sample,
                    test_texts=test_texts,
                    test_labels=test_labels,
                )

                test_acc = metrics['test_f1']
                n_samples = len(train_texts_sample)

                # Check graduation
                if test_acc >= self.target_accuracy:
                    text_graduated = True
                    text_results = {
                        'samples': n_samples,
                        'iterations': iteration,
                        'test_f1': test_acc,
                        'test_precision': metrics['test_precision'],
                        'test_recall': metrics['test_recall'],
                    }
                    print(f"    [Iter {iteration:2d}] Text:       {n_samples:3d} samples ({n_pos}+{n_neg}), "
                          f"F1={test_acc:.3f} ✓ GRADUATED")
                else:
                    print(f"    [Iter {iteration:2d}] Text:       {n_samples:3d} samples ({n_pos}+{n_neg}), "
                          f"F1={test_acc:.3f}")

                # Increment for next iteration
                text_level += self.text_increment
                text_level = min(text_level, self.text_max_samples)

            # Early stop if both graduated
            if activation_graduated and text_graduated:
                break

        # Warnings for non-graduation
        if not activation_graduated and self.train_activation:
            print(f"    ⚠️  Activation probe did not graduate after {iteration} iterations")
            if activation_classifier is not None:
                # Use final results
                activation_results = {
                    'samples': activation_level,
                    'iterations': iteration,
                    'graduated': False,
                }

        if not text_graduated and self.train_text:
            print(f"    ⚠️  Text probe did not graduate after {iteration} iterations")
            if text_probe is not None:
                text_results = {
                    'samples': text_level,
                    'iterations': iteration,
                    'graduated': False,
                }

        total_time = time.time() - start_time

        return {
            'concept': concept_name,
            'activation': activation_results,
            'text': text_results,
            'activation_classifier': activation_classifier,
            'text_probe': text_probe,
            'total_iterations': iteration,
            'total_time': total_time,
        }


__all__ = [
    'DualAdaptiveTrainer',
]
