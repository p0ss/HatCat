#!/usr/bin/env python3
"""
Calibration Matrix Builder

Builds the N×N calibration matrix described in CALIBRATION.md:
- Rows: Prompted concepts (what we ask the model about)
- Columns: Lens concepts (which lenses fire)
- Cells: Rank of the column concept when the row concept is prompted

Supports both full and sparse (production) modes.
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import torch
from tqdm import tqdm


@dataclass
class CalibrationMatrixConfig:
    """Configuration for calibration matrix building."""
    # Expected frequency by layer (probability of appearing in top-k)
    expected_frequency_by_layer: Dict[int, float] = field(default_factory=lambda: {
        0: 0.20,  # Domain-level: ~20% of prompts
        1: 0.10,  # Major categories: ~10%
        2: 0.05,  # Mid-level
        3: 0.05,
        4: 0.01,  # Specific concepts: ~1%
        5: 0.01,
        6: 0.01,
    })

    # Z-score thresholds for outlier detection
    over_firing_z_threshold: float = 2.0
    under_firing_z_threshold: float = -2.0

    # Top-k for activation tracking
    top_k: int = 10

    # Minimum probes needed for reliable statistics
    min_probes_for_stats: int = 5


@dataclass
class LensStatistics:
    """Per-lens statistics computed from calibration matrix."""
    concept: str
    layer: int

    # Rank statistics (computed from column data - when this lens fires)
    mean_rank: float
    std_rank: float
    median_rank: float
    min_rank: int
    max_rank: int

    # Frequency statistics
    times_in_top_k: int
    total_probes: int  # Total rows where this lens was evaluated
    observed_frequency: float  # times_in_top_k / total_probes
    expected_frequency: float  # Based on layer

    # Z-score for outlier detection
    z_score: float
    status: str  # "normal", "over_firing", "under_firing"

    # Diagonal rank (rank when own concept is prompted)
    diagonal_rank: Optional[int] = None
    diagonal_activation: Optional[float] = None

    # Concepts this lens fires on inappropriately
    over_fires_on: List[str] = field(default_factory=list)


@dataclass
class CalibrationMatrix:
    """
    N×N Calibration Matrix.

    Sparse representation where matrix[row_concept][col_concept] = rank
    """
    lens_pack_id: str
    concept_pack_id: str
    model_id: str
    timestamp: str
    config: CalibrationMatrixConfig

    # The matrix itself (sparse: row_concept -> col_concept -> rank)
    matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Activation values (sparse: row_concept -> col_concept -> activation)
    activations: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-lens statistics
    lens_stats: Dict[str, LensStatistics] = field(default_factory=dict)

    # Summary statistics
    total_rows: int = 0
    total_cells_populated: int = 0
    avg_diagonal_rank: float = 0.0
    diagonal_in_top_k_rate: float = 0.0

    # Distribution analysis
    rank_histogram: List[int] = field(default_factory=list)
    rank_mean: float = 0.0
    rank_std: float = 0.0

    # Outlier lists
    over_firing_concepts: List[str] = field(default_factory=list)
    under_firing_concepts: List[str] = field(default_factory=list)

    # Hierarchical consistency
    hierarchical_violations: List[Dict] = field(default_factory=list)
    hierarchical_consistency_rate: float = 0.0


class CalibrationMatrixBuilder:
    """
    Builds calibration matrix with optimized batch scoring.
    """

    def __init__(
        self,
        lens_pack_dir: Path,
        concept_pack_dir: Path,
        model,
        tokenizer,
        device: str,
        config: Optional[CalibrationMatrixConfig] = None,
        layer_idx: int = 15,
    ):
        self.lens_pack_dir = Path(lens_pack_dir)
        self.concept_pack_dir = Path(concept_pack_dir)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or CalibrationMatrixConfig()
        self.layer_idx = layer_idx

        # Will be populated by load_lenses()
        self.lenses: Dict[Tuple[str, int], torch.Tensor] = {}  # (concept, layer) -> weight matrix
        self.lens_biases: Dict[Tuple[str, int], torch.Tensor] = {}
        self.hidden_dim: Optional[int] = None
        self.concepts: Dict[str, Dict] = {}  # concept -> metadata
        self.hierarchy: Dict[str, str] = {}  # concept -> parent concept

    def load_lenses(self, layers: List[int]):
        """
        Load all lens weights into memory for batch scoring.

        Instead of loading state_dict per evaluation, we stack all weights
        into matrices for efficient batch matrix multiplication.
        """
        from monitoring.dynamic_lens_manager import SimpleMLP

        print("Loading lenses for batch scoring...")

        lens_weights = []  # Will become [n_lenses, hidden_dim]
        lens_biases = []   # Will become [n_lenses]
        lens_index = []    # [(concept, layer), ...]

        for layer in layers:
            layer_dir = self.lens_pack_dir / f"layer{layer}"
            if not layer_dir.exists():
                continue

            for lens_file in layer_dir.glob("*_classifier.pt"):
                concept_name = lens_file.stem.replace('_classifier', '')

                try:
                    state_dict = torch.load(lens_file, map_location='cpu', weights_only=True)
                except Exception as e:
                    print(f"  Warning: Failed to load {lens_file}: {e}")
                    continue

                # Handle key prefix variations
                if 'net.0.weight' in state_dict:
                    w = state_dict['net.0.weight']  # [1, hidden_dim]
                    b = state_dict['net.0.bias']    # [1]
                elif '0.weight' in state_dict:
                    w = state_dict['0.weight']
                    b = state_dict['0.bias']
                else:
                    # Try to find weight key
                    weight_key = [k for k in state_dict.keys() if 'weight' in k]
                    bias_key = [k for k in state_dict.keys() if 'bias' in k]
                    if weight_key and bias_key:
                        w = state_dict[weight_key[0]]
                        b = state_dict[bias_key[0]]
                    else:
                        print(f"  Warning: Unknown state_dict format for {lens_file}")
                        continue

                if self.hidden_dim is None:
                    self.hidden_dim = w.shape[1]

                lens_weights.append(w.squeeze(0))  # [hidden_dim]
                lens_biases.append(b.squeeze(0))   # scalar
                lens_index.append((concept_name, layer))

        # Stack into batch tensors
        if lens_weights:
            self.weight_matrix = torch.stack(lens_weights).to(self.device)  # [n_lenses, hidden_dim]
            self.bias_vector = torch.stack(lens_biases).to(self.device)     # [n_lenses]
            self.lens_index = lens_index
            print(f"  Loaded {len(lens_index)} lenses, hidden_dim={self.hidden_dim}")
        else:
            raise ValueError("No lenses loaded!")

        # Create layer norm
        self.layer_norm = torch.nn.LayerNorm(self.hidden_dim, elementwise_affine=False).to(self.device)

    def load_concepts(self, layers: List[int], fast_mode: bool = True):
        """Load concepts and their prompts."""
        print("Loading concepts...")

        for layer in layers:
            layer_file = self.concept_pack_dir / "hierarchy" / f"layer{layer}.json"
            if not layer_file.exists():
                continue

            with open(layer_file) as f:
                layer_data = json.load(f)

            concept_list = layer_data.get('concepts', layer_data if isinstance(layer_data, list) else [])

            for concept in concept_list:
                term = concept.get('sumo_term') or concept.get('term')
                if not term:
                    continue

                # Build prompt
                if fast_mode:
                    prompt = term
                else:
                    hints = concept.get('training_hints', {})
                    pos_examples = hints.get('positive_examples', [])
                    if pos_examples:
                        prompt = pos_examples[0]
                    else:
                        prompt = f"Tell me about {term}."

                self.concepts[term] = {
                    'layer': layer,
                    'prompt': prompt,
                    'parent': concept.get('parent'),
                    'domain': concept.get('domain', 'Unknown'),
                }

                # Track hierarchy
                if concept.get('parent'):
                    self.hierarchy[term] = concept['parent']

        print(f"  Loaded {len(self.concepts)} concepts")

    def score_all_lenses(self, activation: torch.Tensor) -> List[Tuple[str, float, int]]:
        """
        Score all lenses against an activation using batch matrix multiply.

        Args:
            activation: [hidden_dim] tensor

        Returns:
            List of (concept, score, layer) sorted by score descending
        """
        # Normalize
        act_normed = self.layer_norm(activation.unsqueeze(0)).float()  # [1, hidden_dim]

        # Batch score: [n_lenses] = [n_lenses, hidden_dim] @ [hidden_dim, 1] + [n_lenses]
        with torch.no_grad():
            scores = (self.weight_matrix @ act_normed.squeeze().unsqueeze(1)).squeeze() + self.bias_vector

        # Convert to list with metadata
        scores_cpu = scores.cpu().numpy()
        results = [
            (self.lens_index[i][0], float(scores_cpu[i]), self.lens_index[i][1])
            for i in range(len(self.lens_index))
        ]

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def extract_activation(self, prompt: str) -> torch.Tensor:
        """Extract activation from model for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer_idx]
            activation = hidden_states[0, -1, :].float()  # [hidden_dim]

        return activation

    def build_matrix(
        self,
        layers: List[int],
        fast_mode: bool = True,
        sample_rate: float = 1.0,
        max_concepts: Optional[int] = None,
    ) -> CalibrationMatrix:
        """
        Build the full calibration matrix.

        Args:
            layers: Which layers to include
            fast_mode: Use concept name as prompt (faster)
            sample_rate: Fraction of concepts to sample (for faster testing)
            max_concepts: Maximum concepts to probe
        """
        # Load everything
        self.load_lenses(layers)
        self.load_concepts(layers, fast_mode)

        # Initialize matrix
        matrix = CalibrationMatrix(
            lens_pack_id=self.lens_pack_dir.name,
            concept_pack_id=self.concept_pack_dir.name,
            model_id=str(self.model.config._name_or_path if hasattr(self.model.config, '_name_or_path') else 'unknown'),
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self.config,
        )

        # Filter concepts to probe
        concepts_to_probe = list(self.concepts.keys())

        if sample_rate < 1.0:
            import random
            n_sample = int(len(concepts_to_probe) * sample_rate)
            concepts_to_probe = random.sample(concepts_to_probe, n_sample)

        if max_concepts:
            concepts_to_probe = concepts_to_probe[:max_concepts]

        print(f"\nBuilding calibration matrix...")
        print(f"  Concepts to probe: {len(concepts_to_probe)}")
        print(f"  Lenses to score: {len(self.lens_index)}")

        # Track per-lens statistics
        lens_ranks: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        lens_in_top_k: Dict[Tuple[str, int], int] = defaultdict(int)
        lens_total_probes: Dict[Tuple[str, int], int] = defaultdict(int)
        over_fires: Dict[Tuple[str, int], List[str]] = defaultdict(list)

        diagonal_ranks = []

        # Build matrix row by row
        for row_concept in tqdm(concepts_to_probe, desc="Building matrix"):
            concept_data = self.concepts[row_concept]
            prompt = concept_data['prompt']
            row_layer = concept_data['layer']

            try:
                # Extract activation
                activation = self.extract_activation(prompt)

                # Score all lenses
                scores = self.score_all_lenses(activation)

                # Store row in matrix
                matrix.matrix[row_concept] = {}
                matrix.activations[row_concept] = {}

                for rank, (col_concept, score, col_layer) in enumerate(scores, 1):
                    # Store rank
                    matrix.matrix[row_concept][col_concept] = rank
                    matrix.activations[row_concept][col_concept] = score

                    # Update per-lens stats
                    lens_key = (col_concept, col_layer)
                    lens_ranks[lens_key].append(rank)
                    lens_total_probes[lens_key] += 1

                    if rank <= self.config.top_k:
                        lens_in_top_k[lens_key] += 1

                        # Track over-firing (in top-k for wrong concept)
                        if col_concept != row_concept:
                            over_fires[lens_key].append(row_concept)

                    # Track diagonal
                    if col_concept == row_concept:
                        diagonal_ranks.append(rank)

                matrix.total_cells_populated += len(scores)

            except Exception as e:
                print(f"  Error processing {row_concept}: {e}")
                continue

        matrix.total_rows = len(matrix.matrix)

        # Compute per-lens statistics
        print("\nComputing per-lens statistics...")
        all_ranks = []

        for lens_key in self.lens_index:
            concept, layer = lens_key
            ranks = lens_ranks.get(lens_key, [])

            if len(ranks) < self.config.min_probes_for_stats:
                continue

            all_ranks.extend(ranks)

            # Compute statistics
            mean_rank = float(np.mean(ranks))
            std_rank = float(np.std(ranks)) if len(ranks) > 1 else 0.0
            median_rank = float(np.median(ranks))

            total_probes = lens_total_probes.get(lens_key, 0)
            in_top_k = lens_in_top_k.get(lens_key, 0)
            observed_freq = in_top_k / total_probes if total_probes > 0 else 0.0
            expected_freq = self.config.expected_frequency_by_layer.get(layer, 0.01)

            # Compute z-score
            # Using binomial std dev approximation: sqrt(p * (1-p) / n)
            if total_probes > 0:
                std_freq = np.sqrt(expected_freq * (1 - expected_freq) / total_probes)
                if std_freq > 0:
                    z_score = (observed_freq - expected_freq) / std_freq
                else:
                    z_score = 0.0
            else:
                z_score = 0.0

            # Determine status
            if z_score > self.config.over_firing_z_threshold:
                status = "over_firing"
                matrix.over_firing_concepts.append(concept)
            elif z_score < self.config.under_firing_z_threshold:
                status = "under_firing"
                matrix.under_firing_concepts.append(concept)
            else:
                status = "normal"

            # Get diagonal rank
            diagonal_rank = None
            diagonal_activation = None
            if concept in matrix.matrix and concept in matrix.matrix[concept]:
                diagonal_rank = matrix.matrix[concept][concept]
                diagonal_activation = matrix.activations[concept].get(concept)

            stats = LensStatistics(
                concept=concept,
                layer=layer,
                mean_rank=mean_rank,
                std_rank=std_rank,
                median_rank=median_rank,
                min_rank=int(min(ranks)),
                max_rank=int(max(ranks)),
                times_in_top_k=in_top_k,
                total_probes=total_probes,
                observed_frequency=observed_freq,
                expected_frequency=expected_freq,
                z_score=z_score,
                status=status,
                diagonal_rank=diagonal_rank,
                diagonal_activation=diagonal_activation,
                over_fires_on=over_fires.get(lens_key, [])[:20],  # Top 20
            )

            matrix.lens_stats[concept] = stats

        # Compute summary statistics
        if diagonal_ranks:
            matrix.avg_diagonal_rank = float(np.mean(diagonal_ranks))
            matrix.diagonal_in_top_k_rate = sum(1 for r in diagonal_ranks if r <= self.config.top_k) / len(diagonal_ranks)

        if all_ranks:
            matrix.rank_mean = float(np.mean(all_ranks))
            matrix.rank_std = float(np.std(all_ranks))

            # Build histogram (100 bins)
            hist, _ = np.histogram(all_ranks, bins=100, range=(1, len(self.lens_index)))
            matrix.rank_histogram = hist.tolist()

        # Check hierarchical consistency
        print("\nChecking hierarchical consistency...")
        violations = []
        total_checks = 0

        for row_concept in matrix.matrix:
            if row_concept not in self.hierarchy:
                continue

            parent = self.hierarchy[row_concept]
            if parent not in matrix.matrix[row_concept]:
                continue

            child_rank = matrix.matrix[row_concept].get(row_concept)
            parent_rank = matrix.matrix[row_concept].get(parent)

            if child_rank is None or parent_rank is None:
                continue

            total_checks += 1

            # Parent should rank at least as high as child
            if parent_rank > child_rank:
                violations.append({
                    'row_concept': row_concept,
                    'child': row_concept,
                    'child_rank': child_rank,
                    'parent': parent,
                    'parent_rank': parent_rank,
                })

        matrix.hierarchical_violations = violations[:100]  # Cap at 100
        if total_checks > 0:
            matrix.hierarchical_consistency_rate = 1.0 - (len(violations) / total_checks)

        print(f"  Hierarchical consistency: {matrix.hierarchical_consistency_rate:.1%}")
        print(f"  Violations: {len(violations)}")

        return matrix

    def save_matrix(self, matrix: CalibrationMatrix, output_path: Optional[Path] = None):
        """Save matrix to JSON file."""
        if output_path is None:
            output_path = self.lens_pack_dir / "calibration_matrix.json"

        # Convert to serializable format
        data = {
            'lens_pack_id': matrix.lens_pack_id,
            'concept_pack_id': matrix.concept_pack_id,
            'model_id': matrix.model_id,
            'timestamp': matrix.timestamp,
            'config': asdict(matrix.config),
            'summary': {
                'total_rows': matrix.total_rows,
                'total_cells_populated': matrix.total_cells_populated,
                'avg_diagonal_rank': matrix.avg_diagonal_rank,
                'diagonal_in_top_k_rate': matrix.diagonal_in_top_k_rate,
                'rank_mean': matrix.rank_mean,
                'rank_std': matrix.rank_std,
                'hierarchical_consistency_rate': matrix.hierarchical_consistency_rate,
            },
            'distribution': {
                'histogram': matrix.rank_histogram,
                'mean': matrix.rank_mean,
                'std': matrix.rank_std,
            },
            'outliers': {
                'over_firing': matrix.over_firing_concepts,
                'under_firing': matrix.under_firing_concepts,
            },
            'per_lens_stats': {
                k: asdict(v) for k, v in matrix.lens_stats.items()
            },
            'hierarchical_violations': matrix.hierarchical_violations[:50],
            # Note: Full matrix is large, save separately or compress
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved calibration results to: {output_path}")

        # Optionally save full matrix (can be very large)
        matrix_path = output_path.parent / "calibration_matrix_full.json"
        full_data = {
            'matrix_sparse': matrix.matrix,
            'activations_sparse': matrix.activations,
        }
        with open(matrix_path, 'w') as f:
            json.dump(full_data, f)

        print(f"Saved full matrix to: {matrix_path}")


def print_matrix_summary(matrix: CalibrationMatrix):
    """Print human-readable summary of calibration matrix."""
    print(f"\n{'='*80}")
    print("CALIBRATION MATRIX SUMMARY")
    print(f"{'='*80}")
    print(f"  Lens pack: {matrix.lens_pack_id}")
    print(f"  Rows (concepts probed): {matrix.total_rows}")
    print(f"  Cells populated: {matrix.total_cells_populated:,}")
    print()
    print(f"  Average diagonal rank: {matrix.avg_diagonal_rank:.1f}")
    print(f"  Diagonal in top-{matrix.config.top_k} rate: {matrix.diagonal_in_top_k_rate:.1%}")
    print()
    print(f"  Rank distribution: mean={matrix.rank_mean:.1f}, std={matrix.rank_std:.1f}")
    print()
    print(f"  Hierarchical consistency: {matrix.hierarchical_consistency_rate:.1%}")
    print(f"  Over-firing concepts: {len(matrix.over_firing_concepts)}")
    print(f"  Under-firing concepts: {len(matrix.under_firing_concepts)}")

    if matrix.over_firing_concepts:
        print(f"\n  Top over-firing (z > {matrix.config.over_firing_z_threshold}):")
        sorted_over = sorted(
            [(c, matrix.lens_stats[c].z_score, matrix.lens_stats[c].observed_frequency)
             for c in matrix.over_firing_concepts if c in matrix.lens_stats],
            key=lambda x: x[1], reverse=True
        )[:10]
        for concept, z, freq in sorted_over:
            print(f"    {concept}: z={z:.2f}, freq={freq:.1%}")

    if matrix.under_firing_concepts:
        print(f"\n  Top under-firing (z < {matrix.config.under_firing_z_threshold}):")
        sorted_under = sorted(
            [(c, matrix.lens_stats[c].z_score, matrix.lens_stats[c].diagonal_rank)
             for c in matrix.under_firing_concepts if c in matrix.lens_stats],
            key=lambda x: x[1]
        )[:10]
        for concept, z, diag in sorted_under:
            diag_str = f"diag_rank={diag}" if diag else "no_diag"
            print(f"    {concept}: z={z:.2f}, {diag_str}")

    if matrix.hierarchical_violations:
        print(f"\n  Sample hierarchical violations:")
        for v in matrix.hierarchical_violations[:5]:
            print(f"    {v['child']} (rank {v['child_rank']}) > {v['parent']} (rank {v['parent_rank']})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build calibration matrix')
    parser.add_argument('--lens-pack', required=True, help='Path to lens pack')
    parser.add_argument('--concept-pack', required=True, help='Path to concept pack')
    parser.add_argument('--model', required=True, help='Model name/path')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help='Layers to analyze')
    parser.add_argument('--fast-mode', action='store_true',
                        help='Use concept name as prompt')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                        help='Fraction of concepts to sample')
    parser.add_argument('--max-concepts', type=int, default=None,
                        help='Maximum concepts to probe')
    parser.add_argument('--layer-idx', type=int, default=15,
                        help='Model layer for activations')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path')

    args = parser.parse_args()

    lens_pack_dir = Path(args.lens_pack)
    concept_pack_dir = Path(args.concept_pack)

    # Auto-detect layers
    if args.layers is None:
        layers = []
        for layer_dir in lens_pack_dir.glob("layer*"):
            if layer_dir.is_dir():
                try:
                    layer_num = int(layer_dir.name.replace('layer', ''))
                    layers.append(layer_num)
                except ValueError:
                    pass
        layers.sort()
    else:
        layers = args.layers

    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    # Build matrix
    builder = CalibrationMatrixBuilder(
        lens_pack_dir=lens_pack_dir,
        concept_pack_dir=concept_pack_dir,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        layer_idx=args.layer_idx,
    )

    matrix = builder.build_matrix(
        layers=layers,
        fast_mode=args.fast_mode,
        sample_rate=args.sample_rate,
        max_concepts=args.max_concepts,
    )

    # Print summary
    print_matrix_summary(matrix)

    # Save
    output_path = Path(args.output) if args.output else None
    builder.save_matrix(matrix, output_path)


if __name__ == '__main__':
    main()
