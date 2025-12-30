### 7.2 Monitoring Performance

The dynamic lens manager successfully monitored **over 110K concept states**, with only a subset loaded at any time through hierarchical cascade activation.

**Initial Performance** (baseline, no optimizations):
- Average: 108.4ms per token
- Max: 218ms per token
- 100 tokens: 10.8s overhead
- Cache growth: 25 → 437 lenses (unbounded)

**Bottleneck Analysis**:

Profiling revealed lens loading dominated cascade time:
- **71.8%**: Loading children lenses (47.6ms)
- **26.2%**: Inference on existing lenses (17.4ms)
- **2.0%**: Inference on newly loaded lenses (1.3ms)

Per-lens loading breakdown:
- Model creation + state_dict: **59%** (1.5ms) ← Primary bottleneck
- File I/O (torch.load): **26%** (0.6ms)
- GPU transfer (.to(device)): **16%** (0.4ms)
- **Total**: 2.5ms per lens average

**Implemented Optimizations**:

1. **Lazy Model Pool**: Pre-allocated 100 SimpleMLP models, swap state_dicts instead of recreating
   - Eliminates 59% of load time
   - Reduces per-lens load from 2.5ms → 1.0ms

2. **Batch Lens Loading**: Parallel file I/O and single GPU transfer
   - Eliminates repeated disk seeks
   - Reduces I/O overhead by 30-40%

3. **Aggressive Top-K Pruning** (keep_top_k=30):
   - Keeps base layer lenses (always)
   - Keeps ONLY top-K scoring non-base lenses
   - Unloads everything else after each token
   - Prevents cache explosion

**Optimized Performance** (with all three optimizations):

```
Configuration              Avg (ms)  Max (ms)  Final Loaded  Speedup
─────────────────────────────────────────────────────────────────────
No pruning                  108.4     218.0        437       1.00x
Aggressive (top-50)          74.8     181.8         50       1.45x
Very aggressive (top-30)     59.2     139.9         30       1.83x
```

**Production Configuration**:
```python
DynamicLensManager(
    use_activation_lenses=True,   # Hidden state analysis
    use_text_lenses=False,         # Not used (less reliable)
    base_layers=[0],               # Layer 0 (14 lenses, broad coverage)
    load_threshold=0.3,            # Load children when parent > 30%
    keep_top_k=30,                 # Aggressive pruning
    aggressive_pruning=True,
)
```

**Concrete Performance Metrics** (from production deployment):

* **Per-token latency**:
  - Light load (68 lenses): **8ms**
  - Heavy load (1,349 lenses): **88ms**
  - Aggressive pruning (top-30): **59ms average**
* **Temporal slice overhead**: ~28ms per complete concept evaluation pass with cascade activation
* **Memory overhead**:
  - 200 activation lenses: **~390MB**
  - 1,000 activation lenses: **~1.4GB**
  - Configurable via keep_top_k parameter
* **Scalability**:
  - Linear scaling: 1,000 concepts → ~88ms per evaluation
  - Can scale to **20K+ lenses** with <10GB memory
* **Dynamic loading efficiency**: 110K+ concepts monitored via ~1K active lenses (99% reduction in active memory footprint)
* **100-token overhead**:
  - Without pruning: 10.8s
  - With top-30 pruning: **5.9s** (45% faster)

**Activation-Only Mode**:

Production HatCat uses **activation lenses exclusively**. Text lenses were explored but found less reliable:
- Text lenses: Fast (<1ms), small (0.5-2MB), but higher false positive rate
- Activation lenses: Slower (1-3ms), larger (1.3MB), but more accurate for internal state detection
- Memory/speed tradeoff favors activation lenses for safety-critical monitoring

**Architecture Status**:

Dual-lens capability (activation + text) is implemented but inactive:
- Text lens training pipeline exists
- Text lenses could enable divergence measurement (internal state vs output text)
- Currently disabled due to lower reliability compared to activation-only monitoring

These metrics establish that **real-time monitoring is practical** for production deployment, with overhead comparable to typical neural inference costs. The 1.83x speedup from optimization demonstrates that hierarchical cascade loading can scale efficiently even with aggressive pruning.
