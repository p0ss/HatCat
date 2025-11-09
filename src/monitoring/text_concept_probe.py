"""
Text-based concept probes for fast token→SUMO mapping.

Uses text generated during classifier training to learn direct mappings
from surface tokens to SUMO concepts, bypassing WordNet/graph lookups.
"""

from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TfidfConceptProbe:
    """
    Fast TF-IDF + LogisticRegression probe for token→concept mapping.

    Trained on text samples generated during classifier training.
    Much faster than WordNet/graph lookup (~50-100μs vs 1.7ms).
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        max_iter: int = 200,
    ):
        """
        Initialize TF-IDF concept probe.

        Args:
            ngram_range: N-gram range for TF-IDF (default: unigrams + bigrams)
            min_df: Minimum document frequency
            max_df: Maximum document frequency (remove common words)
            max_iter: Max iterations for LogisticRegression
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required: pip install scikit-learn")

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                strip_accents="unicode",
                lowercase=True,
                token_pattern=r'\b\w+\b',  # Word boundaries
            )),
            ("clf", LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                solver='lbfgs',
                multi_class='multinomial',
            ))
        ])

        self.concept_names: List[str] = []
        self.is_fitted = False

    def train(
        self,
        texts: List[str],
        concept_ids: List[str],
        concept_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Train probe on text samples.

        Args:
            texts: List of text strings (tokens or short contexts)
            concept_ids: Corresponding SUMO concept IDs
            concept_names: Optional human-readable names

        Returns:
            Training metrics dict
        """
        if concept_names is None:
            concept_names = concept_ids

        # Fit pipeline
        self.pipeline.fit(texts, concept_ids)
        self.concept_names = list(self.pipeline.classes_)
        self.is_fitted = True

        # Get training accuracy
        y_pred = self.pipeline.predict(texts)
        accuracy = (np.array(concept_ids) == y_pred).mean()

        return {
            'num_samples': len(texts),
            'num_concepts': len(self.concept_names),
            'train_accuracy': float(accuracy),
        }

    def predict(
        self,
        text: str,
        top_k: int = 5,
        return_probs: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Predict SUMO concepts for text.

        Args:
            text: Input text (token or context)
            top_k: Return top K concepts
            return_probs: Include probabilities

        Returns:
            List of (concept_name, probability) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Probe not fitted. Call train() first.")

        # Get probabilities
        probs = self.pipeline.predict_proba([text])[0]

        # Get top K
        top_indices = np.argsort(probs)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            concept = self.concept_names[idx]
            prob = float(probs[idx])
            results.append((concept, prob))

        return results

    def save(self, path: Path):
        """Save trained probe to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted probe")

        joblib.dump({
            'pipeline': self.pipeline,
            'concept_names': self.concept_names,
        }, path)

    @classmethod
    def load(cls, path: Path) -> 'TfidfConceptProbe':
        """Load trained probe from disk."""
        data = joblib.load(path)

        probe = cls()
        probe.pipeline = data['pipeline']
        probe.concept_names = data['concept_names']
        probe.is_fitted = True

        return probe


class MultiHeadTextProbe(nn.Module):
    """
    Multi-label text→concept probe using sentence transformers.

    More powerful than TF-IDF but slower (~2-5ms vs ~100μs).
    Good for distilling knowledge from large models.
    """

    def __init__(
        self,
        num_concepts: int,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dropout: float = 0.1,
    ):
        """
        Initialize multi-head probe.

        Args:
            num_concepts: Number of SUMO concepts
            base_model: Base sentence transformer model
            dropout: Dropout rate
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required: pip install transformers")

        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)

        # Freeze encoder (optional - can fine-tune for better accuracy)
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        # Multi-label classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_concepts)
        )

        self.num_concepts = num_concepts

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids: Tokenized input [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Logits [batch, num_concepts]
        """
        # Get [CLS] token representation
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = out.last_hidden_state[:, 0]  # [batch, hidden_size]

        # Multi-label classification
        logits = self.head(cls_embedding)  # [batch, num_concepts]

        return logits

    def predict(
        self,
        text: str,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        device: str = "cpu",
    ) -> List[Tuple[int, float]]:
        """
        Predict concepts for text.

        Args:
            text: Input text
            threshold: Probability threshold
            top_k: Return only top K (overrides threshold)
            device: Device for inference

        Returns:
            List of (concept_idx, probability) tuples
        """
        self.eval()
        self.to(device)

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        # Predict
        with torch.no_grad():
            logits = self.forward(inputs.input_ids, inputs.attention_mask)
            probs = torch.sigmoid(logits)[0]  # [num_concepts]

        # Get predictions
        if top_k is not None:
            top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
            results = [(int(idx), float(prob)) for idx, prob in zip(top_indices, top_probs)]
        else:
            results = [
                (idx, float(prob))
                for idx, prob in enumerate(probs)
                if prob > threshold
            ]
            results.sort(key=lambda x: x[1], reverse=True)

        return results


def train_text_probes_from_classifier_data(
    classifier_results_dir: Path = Path("results/sumo_classifiers"),
    output_dir: Path = Path("results/text_probes"),
    layers: List[int] = [0, 1, 2],
    probe_type: str = "tfidf",
) -> Dict:
    """
    Train text probes using data from classifier training.

    During classifier training, we generated text samples for each concept.
    This function uses that text to train fast token→concept probes.

    Args:
        classifier_results_dir: Directory with classifier training results
        output_dir: Where to save text probes
        layers: Which layers to process
        probe_type: "tfidf" or "transformer"

    Returns:
        Training statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_texts = []
    all_concepts = []
    concept_to_texts = {}

    # Load training data from classifier results
    for layer in layers:
        layer_dir = classifier_results_dir / f"layer{layer}"
        results_file = layer_dir / "results.json"

        if not results_file.exists():
            print(f"Warning: No results for layer {layer}")
            continue

        with open(results_file) as f:
            results = json.load(f)

        # Extract text samples for each concept
        for concept_name, concept_data in results.items():
            if 'generated_samples' in concept_data:
                texts = concept_data['generated_samples']

                # Store for training
                all_texts.extend(texts)
                all_concepts.extend([concept_name] * len(texts))

                if concept_name not in concept_to_texts:
                    concept_to_texts[concept_name] = []
                concept_to_texts[concept_name].extend(texts)

    print(f"Loaded {len(all_texts)} text samples for {len(concept_to_texts)} concepts")

    if probe_type == "tfidf":
        # Train TF-IDF probe
        probe = TfidfConceptProbe()
        metrics = probe.train(all_texts, all_concepts)

        # Save probe
        probe_path = output_dir / "tfidf_probe.joblib"
        probe.save(probe_path)

        print(f"\n✓ Trained TF-IDF probe:")
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  Concepts: {metrics['num_concepts']}")
        print(f"  Accuracy: {metrics['train_accuracy']:.3f}")
        print(f"  Saved to: {probe_path}")

        return metrics

    else:
        raise NotImplementedError(f"Probe type '{probe_type}' not implemented yet")


__all__ = [
    "TfidfConceptProbe",
    "MultiHeadTextProbe",
    "train_text_probes_from_classifier_data",
]
