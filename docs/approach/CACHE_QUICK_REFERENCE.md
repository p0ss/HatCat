# Temporal Locality Cache - Quick Reference

## At a Glance

```
PROBLEM: 52ms/token disk I/O loading 25.8 lenses × 2.36ms each
SOLUTION: Warm cache keeps recently-used lenses in memory
RESULT:  ~10ms/token disk I/O (80% reduction)
```

## Three-Tier Memory

```
┌─────────────────────────────────────┐
│ ACTIVE (loaded_activation_lenses)   │  ← Run every token
│ • Base layers (always)              │
│ • Top-k scoring lenses              │
├─────────────────────────────────────┤
│ WARM CACHE (warm_cache)             │  ← Zero I/O reactivation
│ • Previously loaded, not top-k      │
│ • (lens, reactivation_count)       │
├─────────────────────────────────────┤
│ COLD STORAGE (disk)                 │  ← Load on demand
│ • All other lenses                  │
└─────────────────────────────────────┘
```

## Key Invariants

1. **Base layers always active** (never in warm cache)
2. **Top-k always active** (non-top-k → warm cache)
3. **Budget enforced**: len(active) + len(warm) ≤ max_loaded_lenses
4. **Evict from warm only** (sorted by reactivation count)

## Flow Per Token

```
1. Run active lenses on hidden state
2. Load children of top-k parents
   ├─ In warm cache? → Move to active (0ms) ✓
   └─ Not in cache? → Load from disk (2.36ms)
3. Identify new top-k from all scores
4. Move non-top-k → warm cache (keep base layers active)
5. Evict from warm cache if over budget
```

## New Metrics (timing_info)

```python
timing = {
    'cache_hits': 20,           # Warm cache hits this token
    'cache_misses': 5,          # Disk loads this token
    'warm_cache_size': 150,     # Current warm cache size
    ...
}
```

## Statistics

```python
manager.print_stats()
```

Shows:
- Warm cache size
- Total in memory (active + warm)
- Top reactivated concepts
- Cache hit rate

## Protected Lenses

```python
self.base_layer_lenses  # Never evicted, always in active set
```

Set during `_load_base_layers()`:
```python
for concept_key in self.loaded_activation_lenses.keys():
    self.base_layer_lenses.add(concept_key)
```

## Eviction Policy

```python
# When total_in_memory > max_loaded_lenses:
# 1. Sort warm_cache by reactivation_count (ascending)
# 2. Evict least-reactivated lenses
# 3. Never evict base_layer_lenses
```

## Code Locations

| Component | Method | Lines |
|-----------|--------|-------|
| Base layer marking | `_load_base_layers()` | ~398-400 |
| Warm cache check | `_load_concepts()` | ~447-461 |
| Cache population | `detect_and_expand()` | ~851-873 |
| Cache eviction | `_manage_cache_memory()` | ~430-476 |
| Timing metrics | `detect_and_expand()` | ~909-911 |
| Stats display | `print_stats()` | ~1042-1052 |

## Example Usage

```python
manager = DynamicLensManager(
    device='cuda',
    max_loaded_lenses=500,  # Combined budget (active + warm)
    keep_top_k=50           # Size of active set (+ base layers)
)

# First token (cold)
concepts, timing = manager.detect_and_expand(h1, return_timing=True)
# cache_hits: 0, cache_misses: 25

# Second token (warm)
concepts, timing = manager.detect_and_expand(h2, return_timing=True)
# cache_hits: 20, cache_misses: 5  ← 80% hit rate!
```

## Performance Expectations

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Token 1 | 200ms | 200ms | - (cold start) |
| Token 2+ | 200ms | ~100ms | 2x faster |
| Disk I/O | 52ms | ~10ms | 5x reduction |
| Cache hit rate | 0% | ~80% | - |

## Debugging

Check warm cache state:
```python
print(f"Active: {len(manager.loaded_activation_lenses)}")
print(f"Warm: {len(manager.warm_cache)}")
print(f"Base: {len(manager.base_layer_lenses)}")

# Top reactivated
for key, count in sorted(
    manager.cache_reactivation_count.items(),
    key=lambda x: x[1], reverse=True
)[:5]:
    print(f"{key}: {count} reactivations")
```

## Common Scenarios

### High cache hit rate (good)
```
cache_hits: 20, cache_misses: 5
→ Temporal locality working well
→ Concepts stable across tokens
```

### Low cache hit rate (expected in some cases)
```
cache_hits: 2, cache_misses: 23
→ Topic shift (e.g., "ocean" → "philosophy")
→ Will warm up in subsequent tokens
```

### Eviction happening frequently
```
warm_cache_size oscillating
→ May need larger max_loaded_lenses
→ Or more aggressive keep_top_k
```

## Backward Compatibility

✓ All existing code works unchanged
✓ `loaded_lenses` alias maintained
✓ Same API surface
✓ Falls back gracefully if warm cache empty
