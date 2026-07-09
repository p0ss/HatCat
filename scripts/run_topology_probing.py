#!/usr/bin/env python3
"""
Run Topology Probing

Bidirectional probing to discover effective inter-layer connectivity.

Usage:
    # Quick test on small subset
    python scripts/run_topology_probing.py --quick

    # Full probing on all layers
    python scripts/run_topology_probing.py --model google/gemma-3-4b-it

    # Resume from saved fuzzing results
    python scripts/run_topology_probing.py --resume results/topology/fuzz_results
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hat.clusters import (
    TopologyFuzzer,
    ReverseTracer,
    ConnectivityAggregator,
    ClusterBuilder,
)
from src.hat.clusters.fuzzer import FuzzConfig
from src.hat.clusters.tracer import TraceConfig
from src.hat.clusters.builder import quick_cluster_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_name: str, device: str = "cuda"):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {param_count:,} parameters")

    return model, tokenizer


def run_forward_fuzzing(
    model,
    tokenizer,
    output_dir: Path,
    config: FuzzConfig,
    layers_to_fuzz: list = None,
    use_batched: bool = True,
):
    """Run forward fuzzing and save results."""
    logger.info("=" * 60)
    logger.info(f"PHASE 1: Forward Fuzzing (1-bit) - {'Batched' if use_batched else 'Sequential'}")
    logger.info("=" * 60)

    if layers_to_fuzz:
        config.layers_to_fuzz = layers_to_fuzz

    fuzzer = TopologyFuzzer(model, tokenizer, config)

    logger.info(f"Fuzzing {fuzzer.n_layers} layers, hidden_dim={fuzzer.hidden_dim}")
    logger.info(f"Batch size: {config.batch_size}, Rank scoring: {config.use_rank_scoring}")
    if config.max_neurons_per_layer:
        logger.info(f"Limiting to {config.max_neurons_per_layer} neurons per layer")

    results = fuzzer.fuzz_all_layers(show_progress=True, use_batched=use_batched)

    fuzz_dir = output_dir / "fuzz_results"
    fuzzer.save_results(results, str(fuzz_dir))

    logger.info(f"Forward fuzzing complete. Results saved to {fuzz_dir}")

    return results, fuzzer


def run_batched_comparison(
    model,
    tokenizer,
    output_dir: Path,
    source_layer: int = 0,
    n_neurons: int = 64,
):
    """Compare batched vs sequential fuzzing."""
    logger.info("=" * 60)
    logger.info("COMPARISON: Batched vs Sequential Fuzzing")
    logger.info("=" * 60)

    # Disable rank scoring for comparison (compare raw values)
    config = FuzzConfig(use_rank_scoring=False)
    fuzzer = TopologyFuzzer(model, tokenizer, config)

    comparison = fuzzer.compare_batched_vs_sequential(
        source_layer=source_layer,
        n_neurons=n_neurons,
    )

    logger.info(f"Sequential time: {comparison['sequential_time']:.2f}s")
    logger.info(f"Batched time: {comparison['batched_time']:.2f}s")
    logger.info(f"Speedup: {comparison['speedup']:.1f}x")

    mean_corr = comparison['mean_correlation']
    if np.isnan(mean_corr):
        logger.warning("Mean correlation is NaN - likely all-zero responses")
    else:
        logger.info(f"Mean correlation: {mean_corr:.4f}")

    # Show debug info
    if "debug" in comparison:
        debug = comparison["debug"]
        first_layer = list(debug.keys())[0] if debug else None
        if first_layer:
            info = debug[first_layer]
            logger.info(f"Layer {first_layer} nonzero: seq={info['seq_nonzero']}, batch={info['batch_nonzero']}")

    # Save comparison
    import json
    with open(output_dir / "batched_comparison.json", "w") as f:
        # Convert numpy floats to python floats, handle NaN
        def convert(v):
            if isinstance(v, (float, np.floating)):
                return float(v) if not np.isnan(v) else "NaN"
            return v

        comparison_serializable = {k: convert(v) for k, v in comparison.items()}
        comparison_serializable["correlations"] = {
            str(k): convert(v) for k, v in comparison["correlations"].items()
        }
        json.dump(comparison_serializable, f, indent=2)

    return comparison


def run_backward_tracing(
    model,
    tokenizer,
    output_dir: Path,
    config: TraceConfig,
    layers_to_trace: list = None,
):
    """Run backward tracing and save results."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Backward Tracing (QKV + MLP)")
    logger.info("=" * 60)

    tracer = ReverseTracer(model, tokenizer, config)

    logger.info(f"Tracing {tracer.n_layers} layers with {config.n_contexts} contexts")

    if layers_to_trace:
        results = {}
        for layer in layers_to_trace:
            results[layer] = tracer.trace_layer_backward(
                layer,
                n_contexts=config.n_contexts,
                show_progress=True,
            )
    else:
        results = tracer.trace_all_layers(
            n_contexts=config.n_contexts,
            show_progress=True,
        )

    trace_dir = output_dir / "trace_results"
    tracer.save_results(results, str(trace_dir))

    logger.info(f"Backward tracing complete. Results saved to {trace_dir}")

    return results


def aggregate_and_cluster(
    output_dir: Path,
    fuzz_results,
    trace_results,
    n_layers: int,
    hidden_dim: int,
    n_clusters: int = 50,
    auto_clusters: bool = False,
):
    """Aggregate connectivity and build clusters."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Aggregation and Clustering")
    logger.info("=" * 60)

    aggregator = ConnectivityAggregator(n_layers, hidden_dim)
    aggregator.add_forward_results(fuzz_results)
    aggregator.add_backward_results(trace_results)

    conn_map = aggregator.build()

    conn_dir = output_dir / "connectivity"
    aggregator.save(conn_map, str(conn_dir))

    logger.info(f"Connectivity map saved to {conn_dir}")

    # Identify dominant pathways
    dominant = aggregator.identify_dominant_pathways(conn_map, top_k=20)
    logger.info(f"Top 5 dominant pathways:")
    for edge in dominant[:5]:
        logger.info(
            f"  L{edge.source_layer}:N{edge.source_neuron} -> "
            f"L{edge.target_layer}:N{edge.target_neuron} "
            f"(score={edge.combined_score:.3f})"
        )

    # Identify dynamic routing zones
    dynamic_zones = aggregator.identify_dynamic_zones(conn_map)
    logger.info(f"Dynamic routing zones (attention-mediated): {dynamic_zones}")

    # Build clusters
    builder = ClusterBuilder(conn_map)

    # Find optimal k if requested
    if auto_clusters:
        logger.info("Finding optimal cluster count (elbow + silhouette)...")
        k_analysis = builder.find_optimal_clusters(
            k_range=[10, 25, 50, 100, 150, 200, 300, 500],
            sample_size=10000,  # Subsample for faster silhouette
        )
        n_clusters = k_analysis['optimal_k']

        # Save cluster analysis
        import json
        with open(output_dir / "cluster_analysis.json", "w") as f:
            json.dump({
                'elbow_k': int(k_analysis['elbow_k']),
                'silhouette_k': int(k_analysis['silhouette_k']),
                'optimal_k': int(k_analysis['optimal_k']),
                'k_values': [int(k) for k in k_analysis['k_values']],
                'inertias': [float(x) if not np.isnan(x) else None for x in k_analysis['inertias']],
                'silhouettes': [float(s) for s in k_analysis['silhouettes']],
            }, f, indent=2)
        logger.info(f"Cluster analysis saved to {output_dir / 'cluster_analysis.json'}")

    logger.info(f"Building {n_clusters} clusters...")
    clusters = builder.build(n_clusters=n_clusters)

    cluster_dir = output_dir / "clusters"
    builder.save(clusters, str(cluster_dir))

    stats = quick_cluster_stats(clusters)
    logger.info(f"Cluster statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"Clusters saved to {cluster_dir}")

    return conn_map, clusters


def main():
    parser = argparse.ArgumentParser(description="Run topology probing")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="Model to probe",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: results/topology/<timestamp>)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer layers, fewer neurons, fewer contexts",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Specific layers to probe (comma-separated, e.g., '0,5,10,15')",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=50,
        help="Number of clusters to create",
    )
    parser.add_argument(
        "--n-contexts",
        type=int,
        default=100,
        help="Number of random contexts for backward tracing",
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=None,
        help="Maximum neurons per layer to fuzz (for quick testing)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from saved results directory",
    )
    parser.add_argument(
        "--skip-fuzz",
        action="store_true",
        help="Skip forward fuzzing (use with --resume)",
    )
    parser.add_argument(
        "--skip-trace",
        action="store_true",
        help="Skip backward tracing (use with --resume)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run batched vs sequential comparison",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential fuzzing instead of batched (slower)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for batched fuzzing",
    )
    parser.add_argument(
        "--auto-clusters",
        action="store_true",
        help="Automatically find optimal cluster count using elbow + silhouette",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "results" / "topology" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Quick mode settings
    if args.quick:
        logger.info("QUICK MODE: Using reduced settings for testing")
        args.max_neurons = args.max_neurons or 64
        args.n_contexts = min(args.n_contexts, 10)
        if args.layers is None:
            args.layers = "0,2,4"  # Just a few layers

    # Parse layers
    layers_to_probe = None
    if args.layers:
        layers_to_probe = [int(x) for x in args.layers.split(",")]
        logger.info(f"Probing specific layers: {layers_to_probe}")

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Get model dimensions (handle nested configs like Gemma 3)
    if hasattr(model, 'config'):
        config = model.config
        # Try direct attributes first
        if hasattr(config, 'num_hidden_layers'):
            n_layers = config.num_hidden_layers
            hidden_dim = config.hidden_size
        # Nested config (e.g., Gemma 3 uses text_config)
        elif hasattr(config, 'text_config'):
            n_layers = config.text_config.num_hidden_layers
            hidden_dim = config.text_config.hidden_size
        else:
            raise ValueError("Cannot determine model dimensions from config")
    else:
        raise ValueError("Cannot determine model dimensions")

    logger.info(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Configs
    fuzz_config = FuzzConfig(
        max_neurons_per_layer=args.max_neurons,
        layers_to_fuzz=layers_to_probe,
        batch_size=args.batch_size,
        use_rank_scoring=True,
    )

    trace_config = TraceConfig(
        n_contexts=args.n_contexts,
    )

    # Run comparison if requested
    if args.compare:
        run_batched_comparison(
            model, tokenizer, output_dir,
            source_layer=layers_to_probe[0] if layers_to_probe else 0,
            n_neurons=args.max_neurons or 64,
        )
        return

    # Run phases
    fuzz_results = None
    trace_results = None

    if args.resume:
        resume_dir = Path(args.resume)
        if (resume_dir / "fuzz_results").exists() and not args.skip_fuzz:
            logger.info(f"Loading fuzzing results from {resume_dir / 'fuzz_results'}")
            _, fuzz_results_raw = TopologyFuzzer.load_results(str(resume_dir / "fuzz_results"))
            # Convert to expected format (simplified - would need full LayerFuzzResult)
            logger.warning("Resume from fuzz not fully implemented - re-running")
            fuzz_results = None

    if fuzz_results is None and not args.skip_fuzz:
        fuzz_results, fuzzer = run_forward_fuzzing(
            model, tokenizer, output_dir, fuzz_config, layers_to_probe,
            use_batched=not args.sequential,
        )

    if trace_results is None and not args.skip_trace:
        trace_layers = None
        if layers_to_probe:
            # Trace from layers after the first probed layer
            trace_layers = [l for l in layers_to_probe if l > min(layers_to_probe)]
        trace_results = run_backward_tracing(
            model, tokenizer, output_dir, trace_config, trace_layers
        )

    # Aggregate and cluster
    if fuzz_results and trace_results:
        conn_map, clusters = aggregate_and_cluster(
            output_dir,
            fuzz_results,
            trace_results,
            n_layers,
            hidden_dim,
            n_clusters=args.n_clusters,
            auto_clusters=args.auto_clusters,
        )

    logger.info("=" * 60)
    logger.info("TOPOLOGY PROBING COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
