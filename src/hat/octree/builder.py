"""
Octree Builder

Constructs an activation octree from collected activation samples.

Process:
1. Collect activations from diverse inputs
2. Optionally reduce dimensionality via PCA
3. Recursively partition space using variance-based splits
4. Stop subdivision at min_samples or max_depth
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
import numpy as np
import torch
import logging
from pathlib import Path

from .tree import ActivationOctree, OctreeNode, CellAddress

logger = logging.getLogger(__name__)


@dataclass
class OctreeConfig:
    """Configuration for octree construction."""
    max_depth: int = 10
    min_samples: int = 10
    pca_dimensions: Optional[int] = 64  # None = no PCA
    layers: List[int] = None  # Which model layers to include

    def __post_init__(self):
        if self.layers is None:
            self.layers = [-1]  # Default: last layer only


class OctreeBuilder:
    """
    Builds an activation octree from model activations.

    Usage:
        builder = OctreeBuilder(config)
        builder.collect_from_texts(model, tokenizer, texts)
        octree = builder.build()
    """

    def __init__(self, config: Optional[OctreeConfig] = None):
        self.config = config or OctreeConfig()
        self.activations: List[np.ndarray] = []
        self.pca_components: Optional[np.ndarray] = None
        self.pca_mean: Optional[np.ndarray] = None

    def collect_from_texts(
        self,
        model,
        tokenizer,
        texts: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
    ):
        """
        Collect activations by running model on texts.

        Args:
            model: HuggingFace model with output_hidden_states support
            tokenizer: Corresponding tokenizer
            texts: Input texts to process
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
        """
        from tqdm import tqdm

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Collecting activations")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_activations = self._extract_batch(model, tokenizer, batch_texts)
            self.activations.extend(batch_activations)

        logger.info(f"Collected {len(self.activations)} activation vectors")

    def _extract_batch(
        self,
        model,
        tokenizer,
        texts: List[str],
    ) -> List[np.ndarray]:
        """Extract activations for a batch of texts."""
        # Tokenize
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)

        # Extract from specified layers
        batch_activations = []
        for batch_idx in range(len(texts)):
            # Get last non-padding token position
            attention_mask = inputs.attention_mask[batch_idx]
            last_pos = attention_mask.sum().item() - 1

            # Concatenate activations from specified layers
            layer_acts = []
            for layer_idx in self.config.layers:
                act = hidden_states[layer_idx][batch_idx, last_pos, :].cpu().numpy()
                layer_acts.append(act)

            combined = np.concatenate(layer_acts)
            batch_activations.append(combined)

        return batch_activations

    def collect_from_arrays(self, activations: np.ndarray):
        """
        Directly add activation arrays.

        Args:
            activations: Array of shape (n_samples, hidden_dim)
        """
        for i in range(len(activations)):
            self.activations.append(activations[i])
        logger.info(f"Added {len(activations)} activation vectors")

    def _fit_pca(self, activations: np.ndarray) -> np.ndarray:
        """Fit PCA and transform activations."""
        from sklearn.decomposition import PCA

        n_components = min(self.config.pca_dimensions, activations.shape[1], len(activations))

        logger.info(f"Fitting PCA: {activations.shape[1]} -> {n_components} dimensions")

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(activations)

        # Store for later use
        self.pca_components = pca.components_
        self.pca_mean = pca.mean_

        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

        return reduced

    def _apply_pca(self, activations: np.ndarray) -> np.ndarray:
        """Apply fitted PCA to new activations."""
        if self.pca_components is None:
            return activations
        centered = activations - self.pca_mean
        return centered @ self.pca_components.T

    def build(self) -> ActivationOctree:
        """
        Build the octree from collected activations.

        Returns:
            Constructed ActivationOctree
        """
        if not self.activations:
            raise ValueError("No activations collected. Call collect_* methods first.")

        # Stack activations
        activations = np.stack(self.activations)
        logger.info(f"Building octree from {len(activations)} samples, dim={activations.shape[1]}")

        # Apply PCA if configured
        if self.config.pca_dimensions is not None:
            activations = self._fit_pca(activations)

        n_dimensions = activations.shape[1]

        # Build tree recursively
        indices = list(range(len(activations)))
        root = self._build_node(
            activations=activations,
            indices=indices,
            address=CellAddress(""),
            depth=0,
        )

        octree = ActivationOctree(
            root=root,
            n_dimensions=n_dimensions,
            max_depth=self.config.max_depth,
            min_samples=self.config.min_samples,
        )

        logger.info(f"Built octree: {octree}")

        return octree

    def _build_node(
        self,
        activations: np.ndarray,
        indices: List[int],
        address: CellAddress,
        depth: int,
    ) -> OctreeNode:
        """Recursively build a node and its children."""
        n_samples = len(indices)
        subset = activations[indices]
        centroid = subset.mean(axis=0)

        # Check stopping conditions
        if depth >= self.config.max_depth or n_samples < self.config.min_samples * 2:
            # Create leaf node
            return OctreeNode(
                address=address,
                sample_indices=indices,
                centroid=centroid,
                n_samples=n_samples,
            )

        # Find best split dimension (highest variance)
        variances = subset.var(axis=0)
        split_dim = int(np.argmax(variances))

        # Split at median for balanced tree
        values = subset[:, split_dim]
        split_threshold = float(np.median(values))

        # Partition indices
        left_mask = values < split_threshold
        left_indices = [indices[i] for i in range(n_samples) if left_mask[i]]
        right_indices = [indices[i] for i in range(n_samples) if not left_mask[i]]

        # Handle edge case: all points on one side
        if len(left_indices) == 0 or len(right_indices) == 0:
            return OctreeNode(
                address=address,
                sample_indices=indices,
                centroid=centroid,
                n_samples=n_samples,
            )

        # Recursively build children
        left_child = self._build_node(
            activations, left_indices,
            address=CellAddress(address.bits + "0"),
            depth=depth + 1,
        )
        right_child = self._build_node(
            activations, right_indices,
            address=CellAddress(address.bits + "1"),
            depth=depth + 1,
        )

        return OctreeNode(
            address=address,
            split_dim=split_dim,
            split_threshold=split_threshold,
            left=left_child,
            right=right_child,
            centroid=centroid,
            n_samples=n_samples,
        )

    def save_pca(self, path: Path):
        """Save PCA parameters for later use."""
        if self.pca_components is None:
            raise ValueError("PCA not fitted yet")

        path = Path(path)
        np.savez(
            path,
            components=self.pca_components,
            mean=self.pca_mean,
        )

    def load_pca(self, path: Path):
        """Load PCA parameters."""
        data = np.load(path)
        self.pca_components = data["components"]
        self.pca_mean = data["mean"]


def collect_diverse_activations(
    model,
    tokenizer,
    n_samples: int = 10000,
    sources: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate diverse text samples for activation collection.

    Args:
        model: Not used, but kept for API consistency
        tokenizer: Not used, but kept for API consistency
        n_samples: Target number of samples
        sources: Optional list of source types to include

    Returns:
        List of diverse text samples
    """
    # This is a placeholder - in practice you'd load from:
    # - Wikipedia
    # - Code repositories
    # - Dialogue datasets
    # - Technical documentation
    # - Creative writing
    # etc.

    samples = []

    # For now, generate some synthetic diversity
    prompts = [
        "Explain the concept of",
        "What is the relationship between",
        "Describe how",
        "The main difference between",
        "In the context of",
        "Consider the following scenario:",
        "The key insight is that",
        "One important aspect of",
    ]

    topics = [
        "machine learning", "democracy", "photosynthesis", "economics",
        "philosophy", "music theory", "software engineering", "biology",
        "history", "mathematics", "psychology", "physics", "literature",
        "ethics", "chemistry", "sociology", "linguistics", "art",
    ]

    import random
    for _ in range(n_samples):
        prompt = random.choice(prompts)
        topic = random.choice(topics)
        samples.append(f"{prompt} {topic}")

    return samples[:n_samples]
