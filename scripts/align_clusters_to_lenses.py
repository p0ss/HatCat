#!/usr/bin/env python3
"""
Align topology clusters to concept lenses.

Measures correlation between structural clusters (from topology probing)
and conceptual lenses (trained classifiers). This validates whether
neurons that are structurally connected are also semantically related.

Supports:
- Polar lenses (positive/negative poles)
- Legacy SUMO lenses
- Optional cluster alignment (works without clusters for lens-only analysis)

Usage:
    # Full alignment analysis
    python scripts/align_clusters_to_lenses.py \
        --lens-pack lens_packs/gemma-3-4b_polar-introspective-v2 \
        --cluster-dir results/topology/20260115_221611/clusters \
        --output results/cluster_lens_alignment.json

    # Lens-only analysis (no clusters)
    python scripts/align_clusters_to_lenses.py \
        --lens-pack lens_packs/gemma-3-4b_polar-introspective-v2 \
        --output results/lens_analysis.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "map"))


@dataclass
class LensInfo:
    """Info about a loaded lens."""
    name: str
    layer: int
    pole: Optional[str]  # "positive", "negative", or None for legacy
    lens: nn.Module
    f1: Optional[float] = None


@dataclass
class AlignmentRecord:
    """Single record of lens and cluster activity."""
    prompt_idx: int
    token_idx: int
    lens_scores: Dict[str, float] = field(default_factory=dict)
    cluster_activity: Dict[int, float] = field(default_factory=dict)


def load_clusters(cluster_dir: Path) -> Tuple[Dict, Dict[Tuple[int, int], int]]:
    """Load topology clusters."""
    with open(cluster_dir / "clusters.json") as f:
        clusters = json.load(f)

    with open(cluster_dir / "neuron_to_cluster.json") as f:
        neuron_to_cluster_raw = json.load(f)

    neuron_to_cluster = {}
    for key, cluster_id in neuron_to_cluster_raw.items():
        parts = key.split("_")
        layer, neuron = int(parts[0]), int(parts[1])
        neuron_to_cluster[(layer, neuron)] = cluster_id

    # Build cluster_to_neurons for efficient lookup
    cluster_to_neurons = {}
    for (layer, neuron), cluster_id in neuron_to_cluster.items():
        if cluster_id not in cluster_to_neurons:
            cluster_to_neurons[cluster_id] = []
        cluster_to_neurons[cluster_id].append((layer, neuron))

    return cluster_to_neurons, neuron_to_cluster


def load_polar_lenses(lens_dir: Path, device: str = "cuda") -> List[LensInfo]:
    """Load polar lenses (positive/negative poles)."""
    lenses = []

    for level_dir in sorted(lens_dir.glob("L*")):
        level = int(level_dir.name[1:])

        for layer_dir in sorted(level_dir.glob("layer*")):
            layer = int(layer_dir.name[5:])

            # Load results.json for metadata
            results_path = layer_dir / "results.json"
            metadata = {}
            if results_path.exists():
                with open(results_path) as f:
                    results = json.load(f)
                for concept in results.get("concepts", []):
                    term = concept.get("term", "")
                    if concept.get("positive"):
                        metadata[f"{concept['node_id']}_positive"] = concept["positive"]
                    if concept.get("negative"):
                        metadata[f"{concept['node_id']}_negative"] = concept["negative"]

            # Load lens files
            for lens_file in layer_dir.glob("*.pt"):
                name = lens_file.stem

                # Detect pole from filename
                pole = None
                base_name = name
                if name.endswith("_positive"):
                    pole = "positive"
                    base_name = name[:-9]
                elif name.endswith("_negative"):
                    pole = "negative"
                    base_name = name[:-9]

                # Load lens
                try:
                    lens = load_lens_from_file(lens_file, device)
                    f1 = metadata.get(name, {}).get("f1")

                    lenses.append(LensInfo(
                        name=name,
                        layer=layer,
                        pole=pole,
                        lens=lens,
                        f1=f1,
                    ))
                except Exception as e:
                    print(f"Warning: Failed to load {lens_file}: {e}")

    return lenses


def load_legacy_lenses(lens_dir: Path, device: str = "cuda") -> List[LensInfo]:
    """Load legacy SUMO-style lenses."""
    lenses = []

    for layer_dir in sorted(lens_dir.glob("layer*")):
        layer = int(layer_dir.name[5:])

        # Try to load results.json for metadata
        results_path = layer_dir / "results.json"
        metadata = {}
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            for result in results.get("results", []):
                concept = result.get("concept", "")
                metadata[concept] = {
                    "f1": result.get("test_f1"),
                    "selected_layers": result.get("selected_layers", []),
                }

        for lens_file in layer_dir.glob("*.pt"):
            if lens_file.name in ["results.json"]:
                continue

            name = lens_file.stem

            try:
                lens = load_lens_from_file(lens_file, device)
                f1 = metadata.get(name, {}).get("f1")

                lenses.append(LensInfo(
                    name=name,
                    layer=layer,
                    pole=None,
                    lens=lens,
                    f1=f1,
                ))
            except Exception as e:
                print(f"Warning: Failed to load {lens_file}: {e}")

    return lenses


def load_lens_from_file(lens_file: Path, device: str = "cuda") -> nn.Module:
    """Load a lens from a .pt file, handling different formats."""
    state_dict = torch.load(lens_file, map_location=device, weights_only=True)

    # Detect architecture from state dict
    if "net.0.weight" in state_dict:
        # _TrainableMLP format (polar lenses)
        input_dim = state_dict["net.0.weight"].shape[1]
        hidden_dim = state_dict["net.0.weight"].shape[0]

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        ]

        class _TrainableMLP(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, x):
                return self.net(x)

        lens = _TrainableMLP(nn.Sequential(*layers))
        lens.load_state_dict(state_dict)

    elif "classifier_weights" in state_dict or "weights" in state_dict:
        # Legacy format with explicit weights/bias
        weights = state_dict.get("weights", state_dict.get("classifier_weights"))
        bias = state_dict.get("bias", state_dict.get("classifier_bias"))

        class SimpleLens(nn.Module):
            def __init__(self, w, b):
                super().__init__()
                self.linear = nn.Linear(w.shape[1], w.shape[0])
                self.linear.weight.data = w
                if b is not None:
                    self.linear.bias.data = b
            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        lens = SimpleLens(weights, bias)

    else:
        raise ValueError(f"Unknown lens format in {lens_file}")

    lens.to(device)
    lens.eval()
    return lens


def load_lenses(lens_dir: Path, device: str = "cuda") -> List[LensInfo]:
    """Auto-detect and load lenses from directory."""
    lens_dir = Path(lens_dir)

    # Check for polar lens structure (L1/, L2/, L3/ subdirs)
    if any(lens_dir.glob("L*")):
        print(f"Detected polar lens structure in {lens_dir}")
        return load_polar_lenses(lens_dir, device)

    # Check for legacy structure (layer0/, layer1/, etc.)
    if any(lens_dir.glob("layer*")):
        print(f"Detected legacy lens structure in {lens_dir}")
        return load_legacy_lenses(lens_dir, device)

    raise ValueError(f"Could not detect lens structure in {lens_dir}")


def extract_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    device: str = "cuda"
) -> List[Dict[int, torch.Tensor]]:
    """Extract hidden states for each prompt, returning per-token activations."""
    all_hidden_states = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is tuple of (n_layers+1, batch, seq, hidden)
        # Convert to dict: layer -> [seq, hidden]
        hidden_by_layer = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            # hidden is [batch=1, seq_len, hidden_dim]
            hidden_by_layer[layer_idx] = hidden[0].float().cpu()  # [seq, hidden]

        all_hidden_states.append(hidden_by_layer)

    return all_hidden_states


def compute_cluster_activity(
    hidden_states: Dict[int, torch.Tensor],
    cluster_to_neurons: Dict[int, List[Tuple[int, int]]],
    token_idx: int,
) -> Dict[int, float]:
    """Compute activity for each cluster at a given token position."""
    cluster_activity = {}

    for cluster_id, neurons in cluster_to_neurons.items():
        total_act = 0.0
        count = 0

        for layer, neuron in neurons:
            if layer in hidden_states:
                hidden = hidden_states[layer]  # [seq, hidden]
                if token_idx < hidden.shape[0] and neuron < hidden.shape[1]:
                    total_act += abs(hidden[token_idx, neuron].item())
                    count += 1

        if count > 0:
            cluster_activity[cluster_id] = total_act / count
        else:
            cluster_activity[cluster_id] = 0.0

    return cluster_activity


def compute_lens_scores(
    hidden_state: torch.Tensor,
    lenses: List[LensInfo],
    device: str = "cuda",
    normalize: bool = True,
) -> Dict[str, float]:
    """Run all lenses on a hidden state and return scores."""
    scores = {}

    # Normalize
    if normalize:
        hidden_state = (hidden_state - hidden_state.mean()) / (hidden_state.std() + 1e-8)

    hidden_state = hidden_state.unsqueeze(0).to(device)

    # Match dtype to first lens
    if lenses:
        lens_dtype = next(lenses[0].lens.parameters()).dtype
        hidden_state = hidden_state.to(dtype=lens_dtype)

    with torch.inference_mode():
        for lens_info in lenses:
            try:
                score = lens_info.lens(hidden_state).item()
                scores[lens_info.name] = score
            except Exception as e:
                # Dimension mismatch - skip
                pass

    return scores


def compute_correlation_matrix(
    records: List[AlignmentRecord],
    lenses: List[LensInfo],
    n_clusters: int,
) -> np.ndarray:
    """Compute correlation between lens scores and cluster activity."""
    lens_names = [l.name for l in lenses]
    n_lenses = len(lens_names)

    # Build arrays
    lens_array = np.zeros((len(records), n_lenses))
    cluster_array = np.zeros((len(records), n_clusters))

    for i, record in enumerate(records):
        for j, name in enumerate(lens_names):
            lens_array[i, j] = record.lens_scores.get(name, 0.0)
        for cluster_id in range(n_clusters):
            cluster_array[i, cluster_id] = record.cluster_activity.get(cluster_id, 0.0)

    # Compute correlation matrix: [n_lenses, n_clusters]
    correlation_matrix = np.zeros((n_lenses, n_clusters))

    for i in range(n_lenses):
        for j in range(n_clusters):
            lens_col = lens_array[:, i]
            cluster_col = cluster_array[:, j]

            # Skip if no variance
            if lens_col.std() < 1e-8 or cluster_col.std() < 1e-8:
                correlation_matrix[i, j] = 0.0
            else:
                correlation_matrix[i, j] = np.corrcoef(lens_col, cluster_col)[0, 1]

    return correlation_matrix


def get_diverse_prompts() -> List[str]:
    """Get a diverse set of prompts for alignment analysis."""
    return [
        # Social/emotional
        "I'm feeling really overwhelmed with work lately. Can you help me think through some options?",
        "That's a terrible idea and you should feel bad for suggesting it.",
        "Congratulations on your promotion! You've worked so hard for this.",
        "I don't understand why you're being so difficult about this.",

        # Technical/code
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]",
        "The function should handle edge cases like empty input and invalid types.",
        "SELECT * FROM users WHERE created_at > '2024-01-01' ORDER BY id DESC LIMIT 100;",

        # Factual/knowledge
        "The French Revolution began in 1789 with the storming of the Bastille.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
        "The capital of Australia is Canberra, not Sydney as many people assume.",

        # Creative/hypothetical
        "Imagine a world where gravity worked in reverse on Tuesdays.",
        "Write a haiku about the feeling of debugging code at 3am.",
        "What if Shakespeare had written about modern social media?",

        # Uncertainty/calibration
        "I'm not entirely sure, but I believe the answer is approximately 42.",
        "This is definitely correct and there's no room for doubt.",
        "Based on my understanding, which may be incomplete, the situation seems complex.",

        # Instructions/protocol
        "Step 1: Preheat the oven to 350°F. Step 2: Mix the dry ingredients.",
        "Please format your response as a bulleted list with exactly 5 items.",
        "Do not include any personal opinions in your analysis.",

        # Neutral/baseline
        "The weather forecast shows rain tomorrow with temperatures around 65°F.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The mitochondria is often called the powerhouse of the cell.",
    ]


def main():
    parser = argparse.ArgumentParser(description="Align clusters to lenses")
    parser.add_argument("--lens-pack", type=Path, required=True,
                        help="Directory containing lenses")
    parser.add_argument("--cluster-dir", type=Path, default=None,
                        help="Directory with cluster data (optional)")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                        help="Model to use for activation extraction")
    parser.add_argument("--output", type=Path, default=Path("results/cluster_lens_alignment.json"),
                        help="Output file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompts-file", type=Path, default=None,
                        help="JSON file with custom prompts (optional)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens per prompt to analyze (default: all)")
    parser.add_argument("--extraction-layer", type=int, default=None,
                        help="Specific layer for lens extraction (default: use lens layer)")

    args = parser.parse_args()

    # Load lenses
    print(f"Loading lenses from {args.lens_pack}...")
    lenses = load_lenses(args.lens_pack, args.device)
    print(f"  Loaded {len(lenses)} lenses")

    # Group by pole for summary
    polar_lenses = [l for l in lenses if l.pole is not None]
    legacy_lenses = [l for l in lenses if l.pole is None]
    print(f"  Polar: {len(polar_lenses)} ({len([l for l in polar_lenses if l.pole == 'positive'])} pos, {len([l for l in polar_lenses if l.pole == 'negative'])} neg)")
    print(f"  Legacy: {len(legacy_lenses)}")

    # Load clusters (optional)
    cluster_to_neurons = None
    n_clusters = 0
    if args.cluster_dir and args.cluster_dir.exists():
        print(f"\nLoading clusters from {args.cluster_dir}...")
        cluster_to_neurons, neuron_to_cluster = load_clusters(args.cluster_dir)
        n_clusters = len(cluster_to_neurons)
        print(f"  Loaded {n_clusters} clusters with {len(neuron_to_cluster)} neurons")
    else:
        print("\nNo cluster data - running lens-only analysis")

    # Load prompts
    if args.prompts_file and args.prompts_file.exists():
        with open(args.prompts_file) as f:
            prompts = json.load(f)
        print(f"\nLoaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = get_diverse_prompts()
        print(f"\nUsing {len(prompts)} built-in diverse prompts")

    # Load model
    print(f"\nLoading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract hidden states
    print("\nExtracting hidden states...")
    all_hidden_states = extract_hidden_states(model, tokenizer, prompts, args.device)

    # Compute lens and cluster activity for each token
    print("\nComputing lens scores and cluster activity...")
    records = []

    # Determine which layer to use for lens extraction
    # For polar lenses, they're trained on a specific layer (usually middle)
    extraction_layer = args.extraction_layer
    if extraction_layer is None:
        # Use the layer from the first lens, or default to 17 (middle of 34)
        if lenses:
            extraction_layer = lenses[0].layer
        else:
            extraction_layer = 17

    print(f"  Using layer {extraction_layer} for activation extraction")

    for prompt_idx, hidden_states in enumerate(all_hidden_states):
        seq_len = hidden_states[0].shape[0]
        max_tokens = args.max_tokens or seq_len

        for token_idx in range(min(seq_len, max_tokens)):
            # Get hidden state at extraction layer for this token
            if extraction_layer in hidden_states:
                hidden = hidden_states[extraction_layer][token_idx]
            else:
                continue

            # Compute lens scores
            lens_scores = compute_lens_scores(hidden, lenses, args.device)

            # Compute cluster activity (if clusters loaded)
            cluster_activity = {}
            if cluster_to_neurons:
                cluster_activity = compute_cluster_activity(
                    hidden_states, cluster_to_neurons, token_idx
                )

            records.append(AlignmentRecord(
                prompt_idx=prompt_idx,
                token_idx=token_idx,
                lens_scores=lens_scores,
                cluster_activity=cluster_activity,
            ))

        print(f"  Processed prompt {prompt_idx + 1}/{len(prompts)}")

    print(f"\nCollected {len(records)} token records")

    # Compute correlation matrix (if clusters available)
    correlation_matrix = None
    top_alignments = []

    if cluster_to_neurons:
        print("\nComputing correlation matrix...")
        correlation_matrix = compute_correlation_matrix(records, lenses, n_clusters)

        # Find top alignments
        lens_names = [l.name for l in lenses]
        for i, lens_name in enumerate(lens_names):
            for j in range(n_clusters):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.3:  # Significant correlation
                    top_alignments.append({
                        "lens": lens_name,
                        "cluster": j,
                        "correlation": float(corr),
                    })

        top_alignments.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        print(f"\nTop 20 lens-cluster alignments:")
        for align in top_alignments[:20]:
            print(f"  {align['lens']} <-> Cluster {align['cluster']}: r={align['correlation']:.3f}")

    # Compute lens statistics
    lens_stats = {}
    for lens_info in lenses:
        scores = [r.lens_scores.get(lens_info.name, 0) for r in records]
        lens_stats[lens_info.name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "firing_rate": float(np.mean([s > 0.5 for s in scores])),
            "pole": lens_info.pole,
            "layer": lens_info.layer,
            "f1": lens_info.f1,
        }

    # Save results
    results = {
        "n_lenses": len(lenses),
        "n_clusters": n_clusters,
        "n_prompts": len(prompts),
        "n_records": len(records),
        "extraction_layer": extraction_layer,
        "lens_stats": lens_stats,
        "top_alignments": top_alignments[:100] if top_alignments else [],
    }

    if correlation_matrix is not None:
        # Save as nested dict for JSON compatibility
        lens_names = [l.name for l in lenses]
        results["correlation_matrix"] = {
            lens_names[i]: {
                str(j): float(correlation_matrix[i, j])
                for j in range(n_clusters)
            }
            for i in range(len(lens_names))
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("ALIGNMENT SUMMARY")
    print("=" * 60)
    print(f"Lenses: {len(lenses)}")
    print(f"Clusters: {n_clusters}")
    print(f"Token records: {len(records)}")

    if top_alignments:
        print(f"\nStrongest alignments:")
        for align in top_alignments[:5]:
            print(f"  {align['lens']} <-> Cluster {align['cluster']}: r={align['correlation']:.3f}")

    # Print lens firing summary
    print(f"\nLens firing rates (top 10):")
    sorted_lenses = sorted(lens_stats.items(), key=lambda x: x[1]["firing_rate"], reverse=True)
    for name, stats in sorted_lenses[:10]:
        pole_str = f" ({stats['pole']})" if stats['pole'] else ""
        print(f"  {name}{pole_str}: {stats['firing_rate']:.1%} (mean={stats['mean']:.3f})")


if __name__ == "__main__":
    main()
