"""
Binary classifier for concept detection in neural activations.

Phase 4 methodology: Simple MLP classifier trained on mean activations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional


class BinaryClassifier(nn.Module):
    """
    Simple MLP for binary concept classification.

    Architecture:
    - Input: hidden_dim (e.g., 2560 for Gemma-3-4b)
    - Hidden: intermediate_dim (default 128)
    - Output: 1 (binary classification)
    """

    def __init__(self, input_dim: int, intermediate_dim: int = 128, dtype: torch.dtype = None):
        """
        Args:
            input_dim: Input feature dimension
            intermediate_dim: Hidden layer dimension
            dtype: Parameter dtype. If None, uses default (float32).
                   Use torch.bfloat16 for memory-efficient inference.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, 1, dtype=dtype),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_binary_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    intermediate_dim: int = 128,
    lr: float = 0.001,
    epochs: int = 100,
    device: str = "cuda",
    verbose: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> BinaryClassifier:
    """
    Train a binary classifier on concept activations.

    Args:
        X_train: Training activations (n_samples, hidden_dim)
        y_train: Labels (n_samples,) with 0/1 values
        input_dim: Hidden dimension size
        intermediate_dim: Intermediate layer size
        lr: Learning rate
        epochs: Number of training epochs
        device: Device to train on
        verbose: Print training progress
        dtype: Parameter dtype. Default bfloat16 for memory efficiency.

    Returns:
        Trained classifier

    Example:
        >>> X = np.random.randn(20, 2560)  # 20 samples
        >>> y = np.array([1]*10 + [0]*10)  # 10 positive, 10 negative
        >>> classifier = train_binary_classifier(X, y, input_dim=2560, epochs=50)
        >>> # Use classifier for prediction
        >>> with torch.no_grad():
        ...     pred = classifier(torch.tensor(X[0], dtype=torch.bfloat16).to("cuda"))
        ...     pred.item() > 0.5  # Check if classified as positive
        True
    """
    # Convert to tensors with matching dtype
    X = torch.from_numpy(X_train).to(device=device, dtype=dtype)
    y = torch.from_numpy(y_train).unsqueeze(1).to(device=device, dtype=dtype)

    # Create model
    model = BinaryClassifier(input_dim, intermediate_dim, dtype=dtype).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    return model
