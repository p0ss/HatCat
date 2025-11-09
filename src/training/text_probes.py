"""Train binary text probes for fast token→concept classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score, precision_score, recall_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class BinaryTextProbe:
    """
    Binary TF-IDF + LogisticRegression probe for single concept.

    Same architecture as activation probes, but operates on text.
    Fast inference (~50-100μs per token).
    """

    def __init__(
        self,
        concept_name: str,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 1.0,
        max_iter: int = 100,
    ):
        """
        Initialize binary text probe for a single concept.

        Args:
            concept_name: SUMO concept name
            ngram_range: N-gram range for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            max_iter: Max iterations for LogisticRegression
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required: pip install scikit-learn")

        self.concept_name = concept_name

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                strip_accents="unicode",
                lowercase=True,
                token_pattern=r'\b\w+\b',
            )),
            ("clf", LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                solver='lbfgs',
            ))
        ])

        self.is_fitted = False

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        test_texts: List[str],
        test_labels: List[int],
    ) -> Dict[str, float]:
        """
        Train binary probe.

        Args:
            train_texts: Training text samples
            train_labels: Binary labels (1=positive, 0=negative)
            test_texts: Test text samples
            test_labels: Test binary labels

        Returns:
            Metrics dict (train_f1, test_f1, etc.)
        """
        # Fit pipeline
        self.pipeline.fit(train_texts, train_labels)
        self.is_fitted = True

        # Evaluate
        train_preds = self.pipeline.predict(train_texts)
        test_preds = self.pipeline.predict(test_texts)

        metrics = {
            "train_f1": float(f1_score(train_labels, train_preds, zero_division=0)),
            "test_f1": float(f1_score(test_labels, test_preds, zero_division=0)),
            "train_precision": float(precision_score(train_labels, train_preds, zero_division=0)),
            "test_precision": float(precision_score(test_labels, test_preds, zero_division=0)),
            "train_recall": float(recall_score(train_labels, train_preds, zero_division=0)),
            "test_recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        }

        return metrics

    def predict(self, text: str) -> float:
        """
        Predict probability that text expresses this concept.

        Args:
            text: Input text (token or short context)

        Returns:
            Probability [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError("Probe not fitted")

        # Get probability for positive class
        prob = self.pipeline.predict_proba([text])[0, 1]
        return float(prob)

    def save(self, path: Path):
        """Save trained probe."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted probe")

        joblib.dump({
            'concept_name': self.concept_name,
            'pipeline': self.pipeline,
        }, path)

    @classmethod
    def load(cls, path: Path) -> 'BinaryTextProbe':
        """Load trained probe."""
        data = joblib.load(path)

        probe = cls(concept_name=data['concept_name'])
        probe.pipeline = data['pipeline']
        probe.is_fitted = True

        return probe


def train_text_probe_for_concept(
    concept_name: str,
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    output_path: Path,
) -> Dict[str, float]:
    """
    Train and save a single text probe.

    Args:
        concept_name: SUMO concept name
        train_texts: Training texts
        train_labels: Training labels (1=positive, 0=negative)
        test_texts: Test texts
        test_labels: Test labels
        output_path: Where to save the trained probe

    Returns:
        Training metrics
    """
    probe = BinaryTextProbe(concept_name)

    metrics = probe.train(
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
    )

    # Save
    probe.save(output_path)

    return metrics


def train_text_probes_for_layer(
    layer: int,
    text_samples_dir: Path,
    output_dir: Path,
) -> Dict:
    """
    Train text probes for all concepts in a layer.

    Mirrors the activation probe training process.

    Args:
        layer: Layer number
        text_samples_dir: Directory with text sample JSONs
        output_dir: Where to save text probes

    Returns:
        Summary dict with training statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all text sample files
    sample_files = list(text_samples_dir.glob("*.json"))

    print(f"\nTraining text probes for {len(sample_files)} concepts...")

    results = []
    failed = []

    for i, sample_file in enumerate(sample_files, 1):
        with open(sample_file) as f:
            data = json.load(f)

        concept_name = data['concept']
        print(f"[{i}/{len(sample_files)}] {concept_name}...", end=" ")

        try:
            metrics = train_text_probe_for_concept(
                concept_name=concept_name,
                train_texts=data['train_prompts'],
                train_labels=data['train_labels'],
                test_texts=data['test_prompts'],
                test_labels=data['test_labels'],
                output_path=output_dir / f"{concept_name}_text_probe.joblib",
            )

            print(f"✓ F1: {metrics['test_f1']:.3f}")

            results.append({
                'concept': concept_name,
                **metrics,
            })

        except Exception as e:
            print(f"✗ {e}")
            failed.append({
                'concept': concept_name,
                'error': str(e),
            })

    # Save summary
    summary = {
        'layer': layer,
        'n_concepts': len(sample_files),
        'n_successful': len(results),
        'n_failed': len(failed),
        'results': results,
        'failed': failed,
    }

    summary_path = output_dir / "text_probe_results.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved summary to {summary_path}")
    print(f"  Successful: {len(results)}/{len(sample_files)}")
    if results:
        avg_f1 = sum(r['test_f1'] for r in results) / len(results)
        print(f"  Average test F1: {avg_f1:.3f}")

    return summary


__all__ = [
    "BinaryTextProbe",
    "train_text_probe_for_concept",
    "train_text_probes_for_layer",
]
