# Temporal Locality Cache Flow

## Three-Tier Memory Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         MEMORY HIERARCHY                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  ACTIVE LENSS (loaded_activation_lenses)                   │  │
│  │  - Base layers (always present)                             │  │
│  │  - Top-k scoring lenses from current token                  │  │
│  │  - Run every token                                          │  │
│  │                                                              │  │
│  │  [Entity] [Process] [Physical] [Attribute] [TopKLens1...]  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ↕                                      │
│                    (move based on top-k)                           │
│                              ↕                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  WARM CACHE (warm_cache)                                    │  │
│  │  - Previously loaded, not in current top-k                  │  │
│  │  - In memory but not run                                    │  │
│  │  - Zero disk I/O when reactivated                           │  │
│  │                                                              │  │
│  │  [(Lens, reactivation_count), ...]                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ↕                                      │
│                    (evict least-reactivated)                       │
│                              ↕                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  COLD STORAGE (disk)                                        │  │
│  │  - All other lenses                                         │  │
│  │  - Load on demand                                           │  │
│  │                                                              │  │
│  │  [110K+ lens files on disk]                                │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Token Processing Flow

### Token N (First occurrence)

```
1. Detect concepts with active lenses
   ↓
2. Find top-k parents → load children from DISK (52ms overhead)
   ↓
3. Run newly loaded children lenses
   ↓
4. Identify new top-k from all scores
   ↓
5. Move non-top-k (but keep base layers) → WARM CACHE
   ↓
6. Manage cache memory (evict from warm cache if needed)

Result: Active = base + top-k, Warm = previously loaded non-top-k
```

### Token N+1 (Subsequent tokens - temporal locality)

```
1. Detect concepts with active lenses
   ↓
2. Find top-k parents → load children
   ├─ IF in WARM CACHE: move to active (0ms - zero I/O!) ✓
   └─ IF NOT in cache: load from disk (2.36ms per lens)
   ↓
3. Run newly loaded/reactivated lenses
   ↓
4. Identify new top-k from all scores
   ↓
5. Move non-top-k → WARM CACHE (increment reactivation count)
   ↓
6. Manage cache memory (evict least-reactivated if over budget)

Result: Most children reused from warm cache → ~80% fewer disk reads
```

## Cache Decision Tree

```
Need to load concept lens X?
│
├─ Is X in loaded_activation_lenses?
│  └─ YES → Already active, use it (cache hit)
│
├─ Is X in warm_cache?
│  └─ YES → Move to active, increment reactivation count (cache hit, 0ms!)
│
└─ Is X on disk?
   └─ YES → Load from disk, add to active (cache miss, 2.36ms)
```

## Eviction Policy

```
When total_in_memory > max_loaded_lenses:

1. Sort warm_cache by reactivation_count (ascending)
   ↓
2. Evict lenses with lowest reactivation count
   ↓
3. Never evict:
   - Base layer lenses (in loaded_lenses, not warm_cache)
   - Current top-k lenses (in loaded_lenses, not warm_cache)
   ↓
4. Free memory by returning lens to model pool
```

## Example: Processing "The ocean contains many fish"

### Token 1: "The"

```
Active:  [Entity, Process, Physical, ...] (base layers)
Warm:    []
Action:  Load children of high-scoring base concepts
         → Load [Substance, Organism, ...] from DISK
Result:  Active = base + [Substance, Organism, ...]
         Warm = []
```

### Token 2: "ocean"

```
Active:  [Entity, Physical, Substance, ...] (base + prev top-k)
Warm:    [Process, Organism, ...] (moved from active)
Action:  Load children of "Physical", "Substance"
         → [LiquidSubstance, WaterBody, ...] from WARM CACHE ✓
Result:  Active = base + [Physical, Substance, WaterBody, ...]
         Warm = [Process, Organism, ...]
```

### Token 3: "contains"

```
Active:  [Entity, WaterBody, Physical, ...]
Warm:    [Substance, LiquidSubstance, ...] (moved from active)
Action:  Load children of top concepts
         → [Process, Relation, ...] from WARM CACHE ✓
Result:  Active = base + [Process, Relation, ...]
         Warm = [WaterBody, Substance, ...]
         (reactivation_count[Process] += 1)
```

## Performance Metrics

```
Token N:   52ms disk I/O (loading 25.8 children × 2.36ms)
Token N+1: ~10ms disk I/O (loading ~4 children × 2.36ms, 21 from cache)

Cache hit rate: ~80%
Overhead reduction: ~80% (from 52ms to ~10ms)
```

## Key Invariants

1. **Base layers always in active set**
   - Never moved to warm cache
   - Never evicted

2. **Top-k always in active set**
   - Determined per-token
   - Non-top-k moved to warm cache

3. **Total budget enforced**
   - len(active) + len(warm) ≤ max_loaded_lenses
   - Evict from warm cache only

4. **Reactivation tracking**
   - Incremented each time lens moves warm → active
   - Used for eviction priority (LRU-style)
