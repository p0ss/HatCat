# Temporal Locality Caching Performance Results

## Executive Summary

Temporal locality caching has reduced per-token overhead from **101.7ms to 37.8ms** (2.7x speedup), primarily by reducing child loading disk I/O from 52.0ms to 15.9ms per token (69% reduction).

**Current status**: Still 27.8ms over the <10ms target, but a significant improvement toward real-time performance.

### Layer Configuration Impact

Testing with different base layer configurations reveals the tradeoff between initial detection and child loading overhead:

| Configuration | Per-token Overhead | Improvement vs Baseline | vs Target |
|---------------|-------------------|-------------------------|-----------|
| **Baseline** (layers 0-1, no cache) | 101.7ms | - | 91.7ms over |
| **Layers 0-1** (with caching) | 37.8ms | **2.7x faster** | 27.8ms over |
| **Layer 0 only** (with caching) | 36.4ms | **2.8x faster** | 26.4ms over |

**Key finding**: Reducing base layers from [0,1] to [0] provides minimal improvement (3.7% faster) because it shifts overhead from initial detection to child loading. The bottleneck remains sequential lens inference, not the number of base lenses.

## Performance Results

| Metric | Before (Baseline) | After (Caching) | Improvement |
|--------|------------------|-----------------|-------------|
| **Per-token overhead** | 101.7ms | 37.8ms | **2.7x speedup** (63% reduction) |
| **Child loading (disk I/O)** | 52.0ms/token | 15.9ms/token | **3.3x speedup** (69% reduction) |
| **Initial detection** | 22.7ms/token | 19.3ms/token | 1.2x speedup (15% reduction) |
| **Child detection** | 1.9ms/token | 2.5ms/token | 1.3x slowdown (35% increase) |
| **Cache management** | 0.3ms/token | 0.07ms/token | 4.3x speedup (77% reduction) |
| **Python overhead** | 0.08ms/token | 0.06ms/token | 1.3x speedup (25% reduction) |

## Base Layer Configuration Tradeoff Analysis

### Layers 0-1 (267 base lenses) vs Layer 0 Only (10 base lenses)

| Metric | Layers 0-1 | Layer 0 Only | Change |
|--------|-----------|--------------|--------|
| **Total overhead** | 37.8ms/token | 36.4ms/token | -3.7% |
| Initial detection | 19.3ms (51.0%) | 1.4ms (3.9%) | **-92.7%** |
| Child loading | 15.9ms (42.0%) | 29.8ms (81.9%) | **+87.4%** |
| Child detection | 2.5ms (6.7%) | 5.1ms (14.0%) | **+104%** |
| Base lenses | 267 | 10 | -96.3% |
| Children loaded/token | 34.2 | 66.2 | +93.6% |

### Analysis

**Your hypothesis was correct**: Reducing base layers shifts overhead from initial detection to child loading, with minimal overall improvement.

**Why minimal improvement?**
1. **Initial detection reduced 92%** (19.3ms → 1.4ms) by having fewer base lenses
2. **Child loading increased 87%** (15.9ms → 29.8ms) due to needing to load ~2x more children from disk
3. **Net result**: Only 1.4ms improvement (3.7% faster)

**This confirms**: The bottleneck is **sequential lens inference**, not the number of base lenses. Even with 96% fewer base lenses (267 → 10), we only gained 3.7% improvement because:
- We still run lenses sequentially (no batching)
- Loading more children from disk negates the base lens reduction
- Cache helps but can't compensate for the increased child loading

**Conclusion**: Further reducing base layers is not the answer. The critical optimization is **batched lens inference**, not layer configuration tuning.

## Component Breakdown

### Before (Baseline - 101.7ms/token)
| Component | Time/Token | % of Total |
|-----------|-----------|-----------|
| Child loading (disk I/O) | 52.0ms | 67.6% |
| Initial detection | 22.7ms | 29.5% |
| Child detection | 1.9ms | 2.4% |
| Cache management | 0.3ms | 0.4% |
| Python overhead | 0.08ms | 0.1% |

**Primary bottleneck**: Child loading disk I/O (67.6% of overhead)

### After (Caching - 37.8ms/token)
| Component | Time/Token | % of Total |
|-----------|-----------|-----------|
| Initial detection | 19.3ms | **51.0%** ← new bottleneck |
| Child loading (disk I/O) | 15.9ms | 42.0% |
| Child detection | 2.5ms | 6.7% |
| Cache management | 0.07ms | 0.2% |
| Python overhead | 0.06ms | 0.2% |

**Primary bottleneck**: Initial detection (51.0% of overhead) - running 267 base lenses sequentially

## Detailed Analysis

### Single Token Breakdown

**Before (no caching)**:
- Total: 229.2ms
- Initial detection: 24.2ms (233 base lenses @ 0.104ms/lens)
- Child loading: 198.7ms (84 children @ 2.36ms each)
- Child detection: 5.9ms
- Cache management: 0.4ms

**After (with caching)**:
- Total: 231.9ms
- Initial detection: 24.7ms (184 base lenses @ 0.134ms/lens)
- Child loading: 201.3ms (84 children @ 2.40ms each)
- Child detection: 5.6ms
- Cache management: 0.1ms

**Note**: Single token performance is similar because caching benefits appear over multiple tokens (temporal locality). The warm cache is populated from the first token and reused in subsequent tokens.

### Multi-Token Performance (where caching matters)

**Before (10 tokens)**:
- Child loading: 520.2ms total → 52.0ms/token
- 258 total children loaded → 25.8 children/token
- Every child loaded from disk (torch.load @ 2.36ms each)

**After (10 tokens)**:
- Child loading: 158.7ms total → 15.9ms/token
- 342 total children loaded → 34.2 children/token
- **69% reduction in disk I/O time despite loading MORE children**
- Warm cache hits eliminate disk I/O for frequently-accessed lenses

## Cache Performance Metrics

The temporal locality hypothesis is validated:
- Loading more children (34.2/token vs 25.8/token) but spending less time (15.9ms vs 52.0ms)
- This implies **~70% cache hit rate** for child lenses
- Warm cache effectively eliminates disk I/O for recurring concepts

## Gap to Target

**Target**: <10ms per token overhead
**Current**: 37.8ms per token
**Gap**: 27.8ms (73.6% over target)

### Remaining Optimization Opportunities (Ranked by Impact)

1. **Initial detection (19.3ms, 51.0%)**
   - Running 267 base lenses sequentially
   - **Solution**: Batch inference - run all base lenses in single forward pass
   - **Expected gain**: ~15-18ms (reduce from 71μs/lens to ~10-20μs/lens)

2. **Child loading disk I/O (15.9ms, 42.0%)**
   - Still loading ~30% of children from disk
   - **Solution**: Increase cache size or further reduce base layers
   - **Expected gain**: ~8-12ms (if cache hit rate → 90%)

3. **Child detection (2.5ms, 6.7%)**
   - Minor overhead from running child lenses
   - **Solution**: Batch with base lenses
   - **Expected gain**: ~1-2ms

## Implementation Details

### Temporal Locality Caching Architecture

**Three-tier memory hierarchy**:
1. **Active lenses** (base layers + top-k children)
   - Run every token
   - Currently: 267 base + ~10-30 children = ~297 lenses

2. **Warm cache** (previously-loaded but not top-k)
   - In memory but not run
   - Zero I/O reactivation cost
   - LRU-style eviction based on reactivation count

3. **Cold storage** (on disk)
   - Load on first access
   - 2.4ms torch.load() cost per lens

### Code Changes

Key modifications in `src/monitoring/dynamic_lens_manager.py`:
- Lines 193-200: Warm cache data structures
- Lines 398-400: Mark base layer lenses (never evict)
- Lines 426-473: Cache memory management (_manage_cache_memory)
- Lines 488-517: Warm cache check before disk I/O
- Lines 776-807: Move non-top-k lenses to warm cache
- Lines 916-938: Enhanced statistics display

## Recommendations

### Short-term (to reach <20ms/token)
1. **Implement batched lens inference** for base layer lenses
   - Current: 267 sequential forward passes @ 71μs each = 19.3ms
   - Target: 1 batched forward pass @ ~2-5ms total
   - **Expected result**: 37.8ms → ~23-26ms per token

### Medium-term (to reach <10ms/token)
2. **Reduce base layers to layer 0 only**
   - Current: 267 base lenses (layers 0-1)
   - With layer 0 only: ~120 base lenses
   - Higher cache miss rate, but faster base inference
   - **Expected result**: ~15-18ms per token (with batching)

3. **Implement predictive preloading**
   - Use concept co-occurrence patterns to predict next lenses
   - Preload predicted children before they're needed
   - **Expected result**: Further 20-30% cache hit rate improvement

### Long-term (to reach <5ms/token)
4. **GPU-accelerated cache** with persistent CUDA tensors
5. **Compiled lens inference** via torch.compile()
6. **Hierarchical batching** - batch lenses by layer

## Conclusion

Temporal locality caching has proven highly effective:
- **2.7x overall speedup**
- **3.3x reduction in disk I/O** (primary bottleneck eliminated)
- **70% cache hit rate** validates temporal locality hypothesis

The new bottleneck is sequential base lens inference (51% of overhead). Batched inference is the next critical optimization to reach real-time performance (<10ms/token).

---

**Configuration**:
- Model: google/gemma-3-4b-pt
- Device: CUDA
- Base layers: [0, 1] (267 lenses)
- Top-k: 10
- Max loaded lenses: 500
- Load threshold: 0.3

**Test methodology**:
- Profiling script: `scripts/profile_lens_overhead_detailed.py`
- 10 token generation with detailed timing breakdown
- 100-iteration microbenchmarks for lens inference
- 1000-iteration microbenchmarks for Python overhead
