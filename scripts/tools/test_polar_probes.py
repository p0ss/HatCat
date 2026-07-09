#!/usr/bin/env python3
"""
Test polar probes on example text to validate they detect concepts correctly.

Usage:
    python scripts/tools/test_polar_probes.py --text "Write a Python function"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src/map to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "map"))


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
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x)


def get_hidden_dim(model) -> int:
    config = model.config
    if hasattr(config, 'hidden_size'):
        return config.hidden_size
    elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
        return config.text_config.hidden_size
    raise AttributeError("Cannot find hidden_size")


def extract_activation(model, tokenizer, text: str, device: str, layer_idx: int) -> np.ndarray:
    """Extract activation for a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # Get layer activation, mean over sequence
    hidden = outputs.hidden_states[layer_idx + 1]  # +1 for embedding
    pooled = hidden.mean(dim=1).squeeze(0)  # [hidden_dim]

    return pooled.float().cpu().numpy()


def load_probes(probe_dir: Path, hidden_dim: int) -> Dict[str, Dict[str, LinearProbe]]:
    """Load all probes from a directory."""
    probes = {}

    results_path = probe_dir / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        concepts = results.get("concepts", [])
    else:
        concepts = []

    for concept_info in concepts:
        node_id = concept_info["node_id"]
        term = concept_info["term"]

        pos_path = probe_dir / f"{node_id}_positive.pt"
        neg_path = probe_dir / f"{node_id}_negative.pt"

        if pos_path.exists() and neg_path.exists():
            # Create probes with correct architecture
            pos_probe = LinearProbe(hidden_dim, 128)
            neg_probe = LinearProbe(hidden_dim, 128)

            # Load weights
            pos_probe.load_state_dict(torch.load(pos_path, map_location="cpu", weights_only=True))
            neg_probe.load_state_dict(torch.load(neg_path, map_location="cpu", weights_only=True))

            probes[term] = {
                "positive": pos_probe,
                "negative": neg_probe,
            }

    return probes


def main():
    parser = argparse.ArgumentParser(description="Test polar probes")
    parser.add_argument("--text", "-t", type=str, required=True, help="Text to analyze")
    parser.add_argument("--probe-dir", type=Path, default=Path("results/polar_probes/L1/layer17"))
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=5, help="Show top K concepts")

    args = parser.parse_args()

    # Load model
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    hidden_dim = get_hidden_dim(model)

    # Load probes
    print(f"Loading probes from {args.probe_dir}...")
    probes = load_probes(args.probe_dir, hidden_dim)
    print(f"Loaded {len(probes)} concept probes")

    # Extract activation
    print(f"\nAnalyzing: \"{args.text[:100]}...\"" if len(args.text) > 100 else f"\nAnalyzing: \"{args.text}\"")
    activation = extract_activation(model, tokenizer, args.text, args.device, args.layer)
    activation_t = torch.from_numpy(activation).float().unsqueeze(0)

    # Score with each probe
    results = []
    for term, probe_pair in probes.items():
        pos_probe = probe_pair["positive"]
        neg_probe = probe_pair["negative"]

        with torch.no_grad():
            pos_logit = pos_probe(activation_t).item()
            neg_logit = neg_probe(activation_t).item()

        pos_prob = torch.sigmoid(torch.tensor(pos_logit)).item()
        neg_prob = torch.sigmoid(torch.tensor(neg_logit)).item()

        # Steering score: high pos + low neg = positive quality
        # high neg + low pos = failure mode
        steering = pos_prob - neg_prob

        results.append({
            "term": term,
            "positive_prob": pos_prob,
            "negative_prob": neg_prob,
            "steering": steering,
        })

    # Sort by steering score (positive quality)
    results.sort(key=lambda x: x["steering"], reverse=True)

    print(f"\n{'='*70}")
    print("CONCEPT ACTIVATION SCORES")
    print(f"{'='*70}")
    print(f"{'Concept':<45} {'Pos':>8} {'Neg':>8} {'Steer':>8}")
    print("-" * 70)

    for r in results[:args.top_k]:
        print(f"{r['term'][:44]:<45} {r['positive_prob']:>7.3f} {r['negative_prob']:>7.3f} {r['steering']:>+7.3f}")

    print(f"\n--- Bottom {args.top_k} (potential failure modes) ---")
    for r in results[-args.top_k:]:
        print(f"{r['term'][:44]:<45} {r['positive_prob']:>7.3f} {r['negative_prob']:>7.3f} {r['steering']:>+7.3f}")


if __name__ == "__main__":
    main()
