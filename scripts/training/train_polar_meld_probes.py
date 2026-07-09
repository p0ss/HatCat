#!/usr/bin/env python3
"""
Train paired probes from Polar MELDs.

For each concept, trains TWO probes:
- Positive probe: positive examples vs positive confusables
- Negative probe: negative examples vs negative confusables

The steering direction is: positive_probe - negative_probe

Usage:
    python scripts/training/train_polar_meld_probes.py \
        --level 1 \
        --model google/gemma-3-4b-it \
        --output results/polar_probes/L1
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src/map to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "map"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_hidden_dim(model) -> int:
    """Get hidden dimension from model config."""
    config = model.config
    if hasattr(config, 'hidden_size'):
        return config.hidden_size
    elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
        return config.text_config.hidden_size
    else:
        raise AttributeError(f"Cannot find hidden_size in model config")


def get_num_layers(model) -> int:
    """Get number of transformer layers."""
    config = model.config
    if hasattr(config, 'num_hidden_layers'):
        return config.num_hidden_layers
    elif hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
        return config.text_config.num_hidden_layers
    elif hasattr(config, 'num_layers'):
        return config.num_layers
    else:
        raise AttributeError(f"Cannot find num_hidden_layers in model config")


def extract_activations(
    model,
    tokenizer,
    prompts: List[str],
    device: str = "cuda",
    layer_idx: int = 15,
    batch_size: int = 4,
    pooling: str = "mean",
) -> np.ndarray:
    """
    Extract activations from model for given prompts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of text prompts
        device: Device for inference
        layer_idx: Which layer to extract from
        batch_size: Batch size for processing
        pooling: How to pool across sequence ("mean", "last", "max")

    Returns:
        Array of activation vectors [n_prompts, hidden_dim]
    """
    model.eval()
    all_activations = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Extract layer activations
        # hidden_states is tuple: (embedding, layer0, layer1, ..., layerN)
        layer_hidden = outputs.hidden_states[layer_idx + 1]  # +1 to skip embedding

        # Pool across sequence
        attention_mask = inputs.attention_mask.unsqueeze(-1)  # [batch, seq, 1]

        if pooling == "last":
            # Get last non-padding token
            seq_lens = attention_mask.sum(dim=1).squeeze(-1).long()  # [batch]
            batch_activations = []
            for b in range(layer_hidden.shape[0]):
                last_idx = seq_lens[b] - 1
                batch_activations.append(layer_hidden[b, last_idx])
            pooled = torch.stack(batch_activations, dim=0)
        elif pooling == "max":
            # Mask padding positions
            masked = layer_hidden * attention_mask + (1 - attention_mask) * -1e9
            pooled = masked.max(dim=1)[0]
        else:  # mean
            # Mean over non-padding positions
            sum_hidden = (layer_hidden * attention_mask).sum(dim=1)
            lengths = attention_mask.sum(dim=1)
            pooled = sum_hidden / lengths

        all_activations.append(pooled.float().cpu().numpy())

    return np.concatenate(all_activations, axis=0)


class LinearProbe(torch.nn.Module):
    """Simple linear probe for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, 1),
        )
        # Store normalization parameters
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize inputs
        x = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x)

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization parameters from training data."""
        self.input_mean = torch.from_numpy(mean).float()
        self.input_std = torch.from_numpy(std).float()


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 128,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cuda",
) -> Tuple[LinearProbe, Dict[str, float]]:
    """
    Train a linear probe and return metrics.

    Args:
        X_train: Training features [n_samples, input_dim]
        y_train: Training labels [n_samples]
        X_test: Test features
        y_test: Test labels
        hidden_dim: Probe hidden layer dimension
        epochs: Training epochs
        lr: Learning rate
        device: Training device

    Returns:
        Tuple of (trained_probe, metrics_dict)
    """
    input_dim = X_train.shape[1]

    # Compute normalization from training data
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train_norm).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_test_t = torch.from_numpy(X_test_norm).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    # Create probe
    probe = LinearProbe(input_dim, hidden_dim).to(device)
    probe.set_normalization(scaler.mean_, scaler.scale_)

    # Training
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()

        # Forward (already normalized by set_normalization, but we pre-normalized)
        logits = probe.net(X_train_t).squeeze(-1)
        loss = loss_fn(logits, y_train_t)

        loss.backward()
        optimizer.step()

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            probe.eval()
            with torch.no_grad():
                test_logits = probe.net(X_test_t).squeeze(-1)
                test_preds = (test_logits > 0).cpu().numpy()
                test_f1 = f1_score(y_test, test_preds, zero_division=0)

                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    # Load best state
    if best_state:
        probe.load_state_dict(best_state)

    # Final metrics
    probe.eval()
    with torch.no_grad():
        test_logits = probe.net(X_test_t).squeeze(-1)
        test_preds = (test_logits > 0).cpu().numpy()

    metrics = {
        "f1": f1_score(y_test, test_preds, zero_division=0),
        "precision": precision_score(y_test, test_preds, zero_division=0),
        "recall": recall_score(y_test, test_preds, zero_division=0),
    }

    return probe.cpu(), metrics


def load_polar_melds(meld_dir: Path, level: int) -> List[Dict]:
    """Load all polar MELDs for a given level."""
    level_dir = meld_dir / f"L{level}"
    melds = []

    for meld_file in sorted(level_dir.glob("*.json")):
        try:
            data = json.loads(meld_file.read_text())
            melds.append({
                "file": meld_file.name,
                "term": data["polar_meld"]["term"],
                "node_id": data["node"]["id"],
                "poles": data["polar_meld"]["poles"],
            })
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load {meld_file}: {e}")

    return melds


def examples_to_prompts(examples: List[str]) -> List[str]:
    """Convert example descriptions to prompts for activation extraction.

    Uses a neutral prompt format that doesn't leak class information.
    """
    prompts = []
    for example in examples:
        # Neutral prompt that doesn't reveal the class
        prompt = f"Scenario: {example}"
        prompts.append(prompt)
    return prompts


def train_polar_probes_for_concept(
    meld: Dict,
    model,
    tokenizer,
    layer_idx: int,
    device: str,
    hidden_dim: int,
) -> Optional[Dict]:
    """
    Train positive and negative probes for a single concept.

    Returns dict with probes and metrics, or None if training failed.
    """
    poles = meld["poles"]

    # Get examples and confusables for each pole
    pos_pole = poles.get("positive", {})
    neg_pole = poles.get("negative", {})

    pos_examples = pos_pole.get("examples", [])
    pos_confusables = pos_pole.get("confusables", {}).get("examples", [])
    neg_examples = neg_pole.get("examples", [])
    neg_confusables = neg_pole.get("confusables", {}).get("examples", [])

    # Need enough examples for training
    min_examples = 4  # Lowered to 4 to handle edge cases
    if len(pos_examples) < min_examples or len(pos_confusables) < min_examples:
        logger.warning(f"  Insufficient positive pole data: {len(pos_examples)} examples, {len(pos_confusables)} confusables")
        return None
    if len(neg_examples) < min_examples or len(neg_confusables) < min_examples:
        logger.warning(f"  Insufficient negative pole data: {len(neg_examples)} examples, {len(neg_confusables)} confusables")
        return None

    # Create prompts (no labels - just the scenario text)
    pos_example_prompts = examples_to_prompts(pos_examples)
    pos_confusable_prompts = examples_to_prompts(pos_confusables)
    neg_example_prompts = examples_to_prompts(neg_examples)
    neg_confusable_prompts = examples_to_prompts(neg_confusables)

    # Extract activations
    logger.info(f"  Extracting activations for positive pole...")
    pos_example_acts = extract_activations(model, tokenizer, pos_example_prompts, device, layer_idx)
    pos_confusable_acts = extract_activations(model, tokenizer, pos_confusable_prompts, device, layer_idx)

    logger.info(f"  Extracting activations for negative pole...")
    neg_example_acts = extract_activations(model, tokenizer, neg_example_prompts, device, layer_idx)
    neg_confusable_acts = extract_activations(model, tokenizer, neg_confusable_prompts, device, layer_idx)

    # Prepare training data for positive probe
    # Label: 1 = positive example, 0 = confusable
    X_pos = np.concatenate([pos_example_acts, pos_confusable_acts], axis=0)
    y_pos = np.concatenate([
        np.ones(len(pos_example_acts)),
        np.zeros(len(pos_confusable_acts))
    ])

    # Shuffle and split
    idx_pos = np.random.permutation(len(y_pos))
    X_pos, y_pos = X_pos[idx_pos], y_pos[idx_pos]
    split_pos = int(len(y_pos) * 0.8)
    X_pos_train, X_pos_test = X_pos[:split_pos], X_pos[split_pos:]
    y_pos_train, y_pos_test = y_pos[:split_pos], y_pos[split_pos:]

    # Prepare training data for negative probe
    X_neg = np.concatenate([neg_example_acts, neg_confusable_acts], axis=0)
    y_neg = np.concatenate([
        np.ones(len(neg_example_acts)),
        np.zeros(len(neg_confusable_acts))
    ])

    idx_neg = np.random.permutation(len(y_neg))
    X_neg, y_neg = X_neg[idx_neg], y_neg[idx_neg]
    split_neg = int(len(y_neg) * 0.8)
    X_neg_train, X_neg_test = X_neg[:split_neg], X_neg[split_neg:]
    y_neg_train, y_neg_test = y_neg[:split_neg], y_neg[split_neg:]

    # Train probes
    logger.info(f"  Training positive probe ({len(X_pos_train)} train, {len(X_pos_test)} test)...")
    pos_probe, pos_metrics = train_probe(
        X_pos_train, y_pos_train, X_pos_test, y_pos_test,
        hidden_dim=hidden_dim, device=device
    )

    logger.info(f"  Training negative probe ({len(X_neg_train)} train, {len(X_neg_test)} test)...")
    neg_probe, neg_metrics = train_probe(
        X_neg_train, y_neg_train, X_neg_test, y_neg_test,
        hidden_dim=hidden_dim, device=device
    )

    return {
        "positive_probe": pos_probe,
        "negative_probe": neg_probe,
        "positive_metrics": pos_metrics,
        "negative_metrics": neg_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train polar MELD probes")
    parser.add_argument(
        "--level", "-l", type=int, default=1,
        help="Which level to train (default: 1)"
    )
    parser.add_argument(
        "--meld-dir", "-m", type=Path,
        default=Path("results/polar_melds"),
        help="Directory containing polar MELDs"
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=Path("results/polar_probes"),
        help="Output directory for trained probes"
    )
    parser.add_argument(
        "--model", type=str,
        default="google/gemma-3-4b-it",
        help="Model to extract activations from"
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Layer to extract from (default: middle layer)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference"
    )
    parser.add_argument(
        "--probe-hidden-dim", type=int, default=128,
        help="Hidden dimension for probe MLP"
    )

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_dim = get_hidden_dim(model)
    n_layers = get_num_layers(model)
    layer_idx = args.layer if args.layer is not None else n_layers // 2

    logger.info(f"Model loaded: {hidden_dim} hidden dim, {n_layers} layers")
    logger.info(f"Extracting from layer {layer_idx}")

    # Load MELDs
    melds = load_polar_melds(args.meld_dir, args.level)
    logger.info(f"Loaded {len(melds)} polar MELDs for L{args.level}")

    # Create output directory
    output_dir = args.output / f"L{args.level}" / f"layer{layer_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train probes
    results = {"total": len(melds), "success": 0, "failed": 0, "concepts": []}

    for i, meld in enumerate(melds):
        logger.info(f"\n[{i+1}/{len(melds)}] Training probes for: {meld['term']}")

        try:
            probe_result = train_polar_probes_for_concept(
                meld, model, tokenizer, layer_idx, args.device, args.probe_hidden_dim
            )

            if probe_result:
                # Save probes
                safe_name = meld['node_id'].replace("/", "_").replace(" ", "_")

                pos_path = output_dir / f"{safe_name}_positive.pt"
                neg_path = output_dir / f"{safe_name}_negative.pt"

                torch.save(probe_result["positive_probe"].state_dict(), pos_path)
                torch.save(probe_result["negative_probe"].state_dict(), neg_path)

                results["success"] += 1
                results["concepts"].append({
                    "term": meld["term"],
                    "node_id": meld["node_id"],
                    "positive_metrics": probe_result["positive_metrics"],
                    "negative_metrics": probe_result["negative_metrics"],
                })

                logger.info(f"  Positive F1: {probe_result['positive_metrics']['f1']:.3f}")
                logger.info(f"  Negative F1: {probe_result['negative_metrics']['f1']:.3f}")
            else:
                results["failed"] += 1
                logger.warning(f"  Training failed for {meld['term']}")

        except Exception as e:
            results["failed"] += 1
            logger.error(f"  Error training {meld['term']}: {e}")

    # Save results summary
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'='*60}")
    print(f"POLAR PROBE TRAINING - LEVEL {args.level}")
    print(f"{'='*60}")
    print(f"Total concepts: {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")

    if results["concepts"]:
        pos_f1s = [c["positive_metrics"]["f1"] for c in results["concepts"]]
        neg_f1s = [c["negative_metrics"]["f1"] for c in results["concepts"]]
        print(f"\nPositive probe avg F1: {np.mean(pos_f1s):.3f}")
        print(f"Negative probe avg F1: {np.mean(neg_f1s):.3f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
