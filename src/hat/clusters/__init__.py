"""
Topology Probing Module

Bidirectional probing to discover effective inter-layer connectivity in transformers.

Components:
- fuzzer: 1-bit forward probing (set neuron to 1, measure downstream response)
- tracer: QKV reverse inspection (trace upstream sources from downstream activation)
- aggregator: Combine forward + backward into connectivity map
- builder: Cluster neurons by connectivity patterns
- multi_resolution: 2-bit and continuous fuzzing for verification
- alternative_clustering: DBSCAN and spectral for algorithm comparison

The goal is to discover the "effective weights" between layers - the connectivity
that emerges from the combination of attention, MLPs, and residual stream.
"""

from .fuzzer import TopologyFuzzer, FuzzResult
from .tracer import ReverseTracer, TraceResult
from .aggregator import ConnectivityAggregator, ConnectivityMap
from .builder import ClusterBuilder, TopologyClusters
from .multi_resolution import MultiResolutionFuzzer, MultiResConfig, Resolution
from .alternative_clustering import AlternativeClusterBuilder, compare_seed_stability

__all__ = [
    "TopologyFuzzer",
    "FuzzResult",
    "ReverseTracer",
    "TraceResult",
    "ConnectivityAggregator",
    "ConnectivityMap",
    "ClusterBuilder",
    "TopologyClusters",
    # Verification tools
    "MultiResolutionFuzzer",
    "MultiResConfig",
    "Resolution",
    "AlternativeClusterBuilder",
    "compare_seed_stability",
]
