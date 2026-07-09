#!/usr/bin/env python3
"""
Topology Probing Verification Matrix

Runs comprehensive verification tests to validate that discovered clusters
are real structural features, not artifacts of methodology.

Tests:
1. Resolution invariance: Do clusters persist across 1-bit/2-bit/continuous?
2. Algorithm invariance: Do clusters persist across k-means/DBSCAN/spectral?
3. Seed stability: Do clusters persist across random seeds?
4. Conceptual alignment: Do clusters align with concept probes?

Usage:
    # Quick verification (subset of tests)
    python scripts/run_verification_matrix.py --quick

    # Full verification matrix
    python scripts/run_verification_matrix.py --full

    # Specific test
    python scripts/run_verification_matrix.py --test resolution
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hat.clusters import TopologyFuzzer, ReverseTracer, ConnectivityAggregator, ClusterBuilder
from src.hat.clusters.fuzzer import FuzzConfig
from src.hat.clusters.tracer import TraceConfig
from src.hat.clusters.multi_resolution import MultiResolutionFuzzer, MultiResConfig, Resolution
from src.hat.clusters.alternative_clustering import (
    AlternativeClusterBuilder,
    compare_seed_stability,
)
from src.hat.clusters.builder import quick_cluster_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result from a single verification run."""
    run_id: str
    test_type: str
    params: Dict
    structural_metrics: Dict
    timing: Dict
    comparison_metrics: Optional[Dict] = None


class VerificationRunner:
    """
    Runs verification tests and collects comparable data.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        output_dir: Optional[Path] = None,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = project_root / "results" / "verification" / timestamp
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.results: List[VerificationResult] = []

    def load_model(self):
        """Load model if not already loaded."""
        if self.model is not None:
            return

        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager",  # For attention capture
        )
        self.model.eval()
        logger.info("Model loaded")

    def _get_model_dims(self):
        """Get model dimensions."""
        config = self.model.config
        if hasattr(config, 'text_config'):
            return config.text_config.num_hidden_layers, config.text_config.hidden_size
        return config.num_hidden_layers, config.hidden_size

    def run_fuzzing(
        self,
        resolution: Resolution = Resolution.ONE_BIT,
        n_layers: Optional[int] = None,
        batch_size: int = 64,
    ) -> Dict:
        """Run forward fuzzing at specified resolution."""
        self.load_model()

        config = MultiResConfig(
            resolution=resolution,
            batch_size=batch_size,
            max_neurons_per_layer=None,  # Full
        )

        if n_layers is not None:
            config.layers_to_fuzz = list(range(min(n_layers, self._get_model_dims()[0] - 1)))

        fuzzer = MultiResolutionFuzzer(self.model, self.tokenizer, config)

        import time
        start = time.time()
        results = fuzzer.fuzz_all_layers(show_progress=True)
        fuzz_time = time.time() - start

        return {
            "fuzz_results": results,
            "fuzz_time": fuzz_time,
            "n_layers": fuzzer.n_layers,
            "hidden_dim": fuzzer.hidden_dim,
        }

    def run_tracing(self, n_contexts: int = 100) -> Dict:
        """Run backward tracing."""
        self.load_model()

        config = TraceConfig(n_contexts=n_contexts)
        tracer = ReverseTracer(self.model, self.tokenizer, config)

        import time
        start = time.time()
        results = tracer.trace_all_layers(n_contexts=n_contexts, show_progress=True)
        trace_time = time.time() - start

        return {
            "trace_results": results,
            "trace_time": trace_time,
        }

    def build_connectivity(self, fuzz_results, trace_results, n_layers, hidden_dim):
        """Build connectivity map from fuzz and trace results."""
        aggregator = ConnectivityAggregator(n_layers, hidden_dim)
        aggregator.add_forward_results(fuzz_results)
        aggregator.add_backward_results(trace_results)
        return aggregator.build()

    def test_resolution_invariance(
        self,
        n_clusters: int = 50,
        n_layers: int = 10,  # Subset for speed
        n_contexts: int = 50,
    ) -> Dict:
        """
        Test if clusters persist across resolution levels.

        Runs fuzzing at 1-bit, 2-bit, and continuous, then compares clusters.
        """
        logger.info("=" * 60)
        logger.info("TEST: Resolution Invariance")
        logger.info("=" * 60)

        results_by_res = {}

        for resolution in [Resolution.ONE_BIT, Resolution.TWO_BIT, Resolution.CONTINUOUS]:
            logger.info(f"\nRunning {resolution.value} resolution...")

            # Fuzz
            fuzz_data = self.run_fuzzing(resolution=resolution, n_layers=n_layers)

            # Trace (same for all resolutions)
            if "trace_results" not in results_by_res:
                trace_data = self.run_tracing(n_contexts=n_contexts)
                results_by_res["trace_results"] = trace_data["trace_results"]

            # Build connectivity
            conn_map = self.build_connectivity(
                fuzz_data["fuzz_results"],
                results_by_res["trace_results"],
                fuzz_data["n_layers"],
                fuzz_data["hidden_dim"],
            )

            # Cluster
            builder = ClusterBuilder(conn_map)
            clusters = builder.build(n_clusters=n_clusters)
            stats = quick_cluster_stats(clusters)

            results_by_res[resolution.value] = {
                "stats": stats,
                "clusters": clusters,
                "fuzz_time": fuzz_data["fuzz_time"],
            }

        # Compare clusters across resolutions
        from sklearn.metrics import adjusted_rand_score

        comparisons = {}
        resolutions = ["1-bit", "2-bit", "continuous"]
        for i, res1 in enumerate(resolutions):
            for res2 in resolutions[i + 1:]:
                clusters1 = results_by_res[res1]["clusters"]
                clusters2 = results_by_res[res2]["clusters"]

                # Get labels in same order
                labels1 = []
                labels2 = []
                for key in clusters1.neuron_to_cluster:
                    if key in clusters2.neuron_to_cluster:
                        labels1.append(clusters1.neuron_to_cluster[key])
                        labels2.append(clusters2.neuron_to_cluster[key])

                ari = adjusted_rand_score(labels1, labels2)
                comparisons[f"{res1}_vs_{res2}"] = ari

        result = VerificationResult(
            run_id=f"resolution_invariance_{datetime.now().strftime('%H%M%S')}",
            test_type="resolution_invariance",
            params={
                "n_clusters": n_clusters,
                "n_layers": n_layers,
                "n_contexts": n_contexts,
            },
            structural_metrics={
                res: results_by_res[res]["stats"]
                for res in resolutions
            },
            timing={
                res: results_by_res[res]["fuzz_time"]
                for res in resolutions
            },
            comparison_metrics=comparisons,
        )

        self.results.append(result)
        self._save_result(result)

        logger.info(f"\nResolution comparison (ARI):")
        for key, ari in comparisons.items():
            logger.info(f"  {key}: {ari:.4f}")

        return asdict(result)

    def test_algorithm_invariance(
        self,
        n_clusters: int = 50,
        n_layers: int = 10,
        n_contexts: int = 50,
    ) -> Dict:
        """
        Test if clusters persist across clustering algorithms.
        """
        logger.info("=" * 60)
        logger.info("TEST: Algorithm Invariance")
        logger.info("=" * 60)

        # Run fuzzing and tracing once
        fuzz_data = self.run_fuzzing(resolution=Resolution.ONE_BIT, n_layers=n_layers)
        trace_data = self.run_tracing(n_contexts=n_contexts)

        conn_map = self.build_connectivity(
            fuzz_data["fuzz_results"],
            trace_data["trace_results"],
            fuzz_data["n_layers"],
            fuzz_data["hidden_dim"],
        )

        # Run different algorithms
        alt_builder = AlternativeClusterBuilder(conn_map)
        algo_results = alt_builder.compare_algorithms(
            algorithms=["kmeans", "dbscan", "spectral"],
            n_clusters=n_clusters,
        )

        # Compute agreement
        agreement = alt_builder.compute_algorithm_agreement(algo_results)

        structural_metrics = {}
        for algo, res in algo_results.items():
            structural_metrics[algo] = {
                "n_clusters": res.n_clusters,
                "silhouette": res.silhouette,
                **res.extra_metrics,
            }

        result = VerificationResult(
            run_id=f"algorithm_invariance_{datetime.now().strftime('%H%M%S')}",
            test_type="algorithm_invariance",
            params={
                "n_clusters": n_clusters,
                "n_layers": n_layers,
                "algorithms": list(algo_results.keys()),
            },
            structural_metrics=structural_metrics,
            timing={
                "fuzz_time": fuzz_data["fuzz_time"],
                "trace_time": trace_data["trace_time"],
            },
            comparison_metrics=agreement,
        )

        self.results.append(result)
        self._save_result(result)

        logger.info(f"\nAlgorithm comparison (ARI):")
        for key, ari in agreement.items():
            logger.info(f"  {key}: {ari:.4f}")

        return asdict(result)

    def test_seed_stability(
        self,
        n_clusters: int = 50,
        n_layers: int = 10,
        n_contexts: int = 50,
        seeds: List[int] = [42, 123, 456, 789, 1337],
    ) -> Dict:
        """
        Test if clusters are stable across random seeds.
        """
        logger.info("=" * 60)
        logger.info("TEST: Seed Stability")
        logger.info("=" * 60)

        # Run fuzzing and tracing once
        fuzz_data = self.run_fuzzing(resolution=Resolution.ONE_BIT, n_layers=n_layers)
        trace_data = self.run_tracing(n_contexts=n_contexts)

        conn_map = self.build_connectivity(
            fuzz_data["fuzz_results"],
            trace_data["trace_results"],
            fuzz_data["n_layers"],
            fuzz_data["hidden_dim"],
        )

        # Test stability
        stability = compare_seed_stability(
            conn_map,
            algorithm="kmeans",
            n_clusters=n_clusters,
            seeds=seeds,
        )

        result = VerificationResult(
            run_id=f"seed_stability_{datetime.now().strftime('%H%M%S')}",
            test_type="seed_stability",
            params={
                "n_clusters": n_clusters,
                "n_layers": n_layers,
                "seeds": seeds,
            },
            structural_metrics={
                "mean_ari": stability["mean_ari"],
                "std_ari": stability["std_ari"],
                "min_ari": stability["min_ari"],
            },
            timing={
                "fuzz_time": fuzz_data["fuzz_time"],
                "trace_time": trace_data["trace_time"],
            },
            comparison_metrics={
                k: v for k, v in stability.items()
                if k.startswith("seed_")
            },
        )

        self.results.append(result)
        self._save_result(result)

        logger.info(f"\nSeed stability:")
        logger.info(f"  Mean ARI: {stability['mean_ari']:.4f}")
        logger.info(f"  Std ARI: {stability['std_ari']:.4f}")
        logger.info(f"  Min ARI: {stability['min_ari']:.4f}")

        return asdict(result)

    def _save_result(self, result: VerificationResult):
        """Save individual result to disk."""
        result_path = self.output_dir / f"{result.run_id}.json"

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(result_path, "w") as f:
            json.dump(convert(asdict(result)), f, indent=2)

        logger.info(f"Saved result to {result_path}")

    def save_summary(self):
        """Save summary of all results."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "n_tests": len(self.results),
            "results": [asdict(r) for r in self.results],
        }

        summary_path = self.output_dir / "summary.json"

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(summary_path, "w") as f:
            json.dump(convert(summary), f, indent=2)

        logger.info(f"Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run topology verification tests")
    parser.add_argument("--test", choices=["resolution", "algorithm", "seed", "all"],
                        default="all", help="Which test to run")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer layers")
    parser.add_argument("--full", action="store_true", help="Full mode with all layers")
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--n-contexts", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    n_layers = 5 if args.quick else (None if args.full else 10)
    n_contexts = 20 if args.quick else args.n_contexts

    output_dir = Path(args.output) if args.output else None
    runner = VerificationRunner(output_dir=output_dir)

    try:
        if args.test in ["resolution", "all"]:
            runner.test_resolution_invariance(
                n_clusters=args.n_clusters,
                n_layers=n_layers,
                n_contexts=n_contexts,
            )

        if args.test in ["algorithm", "all"]:
            runner.test_algorithm_invariance(
                n_clusters=args.n_clusters,
                n_layers=n_layers,
                n_contexts=n_contexts,
            )

        if args.test in ["seed", "all"]:
            runner.test_seed_stability(
                n_clusters=args.n_clusters,
                n_layers=n_layers,
                n_contexts=n_contexts,
            )

        runner.save_summary()

        logger.info("=" * 60)
        logger.info("VERIFICATION COMPLETE")
        logger.info(f"Results saved to: {runner.output_dir}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nInterrupted - saving partial results")
        runner.save_summary()


if __name__ == "__main__":
    main()
