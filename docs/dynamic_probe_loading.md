# Dynamic Hierarchical Probe Loading

## Overview

Dynamic on-demand loading of SUMO concept probes based on parent-child confidence cascading. Enables running 110K+ WordNet concepts with minimal memory footprint.

**Key Innovation**: Instead of loading all probes at once, load only:
1. Base layers (0-1) for broad coverage
2. Children of high-scoring parents (dynamic expansion)
3. Unload cold branches to maintain memory limits

## Architecture

### DynamicProbeManager

Located in: `src/monitoring/dynamic_probe_manager.py`

**Core Components**:

```python
class DynamicProbeManager:
    # Metadata for ALL concepts (lightweight, always in memory)
    concept_metadata: Dict[Tuple[str, int], ConceptMetadata]
    parent_to_children: Dict[Tuple[str, int], List[Tuple[str, int]]]
    child_to_parent: Dict[Tuple[str, int], Tuple[str, int]]

    # Loaded probes (heavy, managed dynamically)
    loaded_probes: Dict[Tuple[str, int], nn.Module]
    probe_scores: Dict[Tuple[str, int], float]

    # Configuration
    base_layers: List[int] = [0, 1]  # Always loaded
    load_threshold: float = 0.5       # Load children when parent > threshold
    unload_threshold: float = 0.1     # Unload when < threshold
    max_loaded_probes: int = 500      # Memory limit
```

**Key Design Decisions**:

1. **Tuple Keys**: Use `(concept_name, layer)` as dict keys to handle concepts that appear in multiple layers
2. **Metadata Always Loaded**: Lightweight metadata (~10KB per concept) for all 121K concepts stays in memory
3. **Probes Dynamically Loaded**: Heavy probe models (~1.3MB each) loaded/unloaded on demand
4. **Parent-Child Mapping**: Pre-computed during initialization for fast lookup

## Performance

### Test Results

**Prompt**: "The cat sat on the mat"

**Cascade Performance**:
- **Step 1**: 14 probes → detect → load 19 children → 33 probes (60ms)
- **Step 2**: 33 probes → detect → load 33 children → 66 probes (79ms)
- **Step 3**: 66 probes → detect → load 5 children → 71 probes (15ms)

**Memory Efficiency**:
- Total concepts available: 10,804 (only layers 0-2 trained)
- Loaded in memory: 71 probes
- **Memory footprint: 0.66%**
- Estimated memory: ~92MB (vs ~14GB if all loaded)

**Timing Breakdown** (Step 1):
- Initial detection: 15.4ms (run 14 probes)
- Child loading: 44.7ms (load 19 probe models)
- Total: 60.2ms

**Detection Quality**:
```
Top concepts detected:
1. [L2] Seat          0.913  (Layer 2 - loaded via Furniture → Seat cascade)
2. [L0] Proposition   0.931  (Layer 0 - base layer)
3. [L1] Shelf         0.824  (Layer 1 - loaded via Artifact children)
4. [L1] PictureFrame  0.748  (Layer 1 - loaded via Artifact children)
```

Notice how "Seat" (Layer 2) was discovered through hierarchical expansion:
```
Entity → Physical → Object → Artifact → Furniture → Seat
```

## Usage

### Basic Usage

```python
from src.monitoring.dynamic_probe_manager import DynamicProbeManager

# Initialize with base layers
manager = DynamicProbeManager(
    device="cuda",
    base_layers=[0],          # Start with only Layer 0
    load_threshold=0.3,       # Load children when parent > 0.3
    max_loaded_probes=1000,   # Memory limit
)

# Extract hidden state from model
hidden_state = model(**inputs, output_hidden_states=True).hidden_states[-1].mean(dim=1)

# Run hierarchical detection (automatically loads children)
results, timing = manager.detect_and_expand(
    hidden_state,
    top_k=10,
    return_timing=True,
)

# Results: [(concept_name, probability, layer), ...]
for concept, prob, layer in results:
    path = manager.get_concept_path(concept, layer)
    print(f"[L{layer}] {concept}: {prob:.3f}")
    print(f"  Path: {' → '.join(path)}")
```

### Multiple Cascade Levels

For best results, run `detect_and_expand()` multiple times to allow deep hierarchical exploration:

```python
# Level 1: Detect with base layers, load first children
results_1, _ = manager.detect_and_expand(hidden_state)

# Level 2: Detect with expanded probes, load grandchildren
results_2, _ = manager.detect_and_expand(hidden_state)

# Level 3: Convergence (usually no new children loaded)
results_3, _ = manager.detect_and_expand(hidden_state)
```

### Integration with Temporal Monitoring

```python
from src.monitoring.temporal_monitor import SUMOTemporalMonitor
from src.monitoring.dynamic_probe_manager import DynamicProbeManager

# Replace static classifier loading with dynamic manager
manager = DynamicProbeManager(device="cuda", base_layers=[0])

# Use in temporal monitoring
monitor = SUMOTemporalMonitor(
    classifiers=manager.loaded_probes,  # Start with base layers
    top_k=10,
    threshold=0.3,
)

# During generation, expand dynamically
result = monitor.monitor_generation(model, tokenizer, prompt)

for timestep in result['timesteps']:
    hidden_state = extract_hidden_state_for_token(timestep)

    # Dynamic expansion
    concepts, _ = manager.detect_and_expand(hidden_state)

    # Update monitor with expanded probes
    monitor.classifiers = manager.loaded_probes
```

## Scaling to 110K Concepts

With dynamic loading, you can theoretically run ALL 110K WordNet concepts:

**Assumptions**:
- 110,000 concepts total
- ~1.3MB per activation probe
- Load threshold: 0.3 (load children when parent > 30%)
- Max loaded probes: 5,000 (memory limit)

**Expected Performance**:
- Memory: 5,000 × 1.3MB = ~6.5GB (vs 143GB if all loaded)
- **Memory reduction: 95.5%**
- Detection time: ~20-30ms per cascade level (5K probes)
- Total cascade (3 levels): ~60-90ms

**Cascade Depth**:
```
Level 0 (Entity, Physical, Abstract)
  ↓ load children (P > 0.3)
Level 1 (Object, Process, Attribute, ...)
  ↓ load children (P > 0.3)
Level 2 (Artifact, Animal, Device, ...)
  ↓ load children (P > 0.3)
Level 3 (Furniture, Vehicle, Weapon, ...)
  ↓ load children (P > 0.3)
Level 4+ (chair.n.01, table.n.02, ...)
```

**Convergence**: Typically converges at 3-5 cascade levels, as only the most relevant branches are explored.

## Configuration

### Base Layers

**Recommendation**: Use Layer 0 only for maximum memory efficiency

```python
# Option 1: Layer 0 only (14 probes, ~18MB)
base_layers=[0]

# Option 2: Layers 0-1 (290 probes, ~377MB)
base_layers=[0, 1]

# Option 3: Layers 0-2 (1349 probes, ~1.75GB)
base_layers=[0, 1, 2]
```

### Load Threshold

Controls when to expand children:

```python
# Aggressive expansion (more concepts, slower)
load_threshold=0.2

# Balanced (recommended)
load_threshold=0.3

# Conservative (fewer concepts, faster)
load_threshold=0.5
```

### Max Loaded Probes

Memory vs accuracy trade-off:

```python
# Low memory (embedding-friendly)
max_loaded_probes=500  # ~650MB

# Balanced
max_loaded_probes=1000  # ~1.3GB

# High accuracy (if RAM available)
max_loaded_probes=5000  # ~6.5GB
```

## Advantages

✅ **Memory Efficient**: 0.66% memory footprint (71/10K concepts)

✅ **Scalable**: Can handle 110K+ concepts with reasonable memory

✅ **Hierarchical**: Exploits SUMO's ontology structure for intelligent exploration

✅ **Fast**: ~15-80ms per cascade level (depending on children loaded)

✅ **Adaptive**: Only explores relevant branches of the ontology

✅ **Stateful**: Caches frequently accessed probes

## Limitations

⚠️ **Cold Start**: First cascade level requires loading children (60ms overhead)

⚠️ **Convergence Required**: Need 2-3 cascade iterations for deep concepts

⚠️ **Threshold Sensitive**: Load threshold affects both memory and accuracy

⚠️ **No Backtracking**: Once a branch is explored, siblings aren't reconsidered (unless they score high independently)

## Future Enhancements

1. **Text Probe Integration**: Support dual probe system (activation + text)
2. **Smart Caching**: Use access patterns to predict which probes to keep loaded
3. **Parallel Loading**: Load children in background while detecting
4. **Confidence Propagation**: Boost child scores based on parent confidence
5. **Branch Pruning**: Unload entire branches when parent scores drop
6. **Adaptive Thresholds**: Adjust load threshold based on available memory

## Files

**Core Implementation**:
- `src/monitoring/dynamic_probe_manager.py` - DynamicProbeManager class

**Tests**:
- `scripts/test_cascade_simple.py` - Simple Layer 0 → 1 → 2 cascade
- `scripts/test_dynamic_probe_cascade.py` - Full benchmark with multi-prompt test

**Documentation**:
- `docs/dynamic_probe_loading.md` - This file
- `docs/dual_probe_training.md` - Text probe system (future integration)

## Example Output

```
================================================================================
STEP 1: Detect with Layer 0 only
================================================================================
Detection time: 15.44ms
Children loaded: 19
Total loaded probes: 33
Total time: 60.17ms

Top 5 concepts:
  1. [L0] Proposition   0.931  (0 children)
  2. [L0] Process       0.550  (0 children)
  3. [L1] Artifact      0.458  (0 children)
  4. [L0] Object        0.391  (0 children)
  5. [L1] Astronomical  0.389  (0 children)

================================================================================
STEP 2: Second detection (should load Layer 1 children)
================================================================================
Detection time: 2.76ms
Children loaded: 33
Total loaded probes: 66
Total time: 78.80ms

Top 10 concepts:
  1. [L0] Proposition   0.931
       Path: Entity → Abstract → Proposition
  2. [L1] Shelf         0.824
       Path: Entity → Physical → Object → Artifact → Shelf
  3. [L1] PictureFrame  0.748
       Path: Entity → Physical → Object → Artifact → PictureFrame
  ...

================================================================================
STEP 3: Third detection (should load Layer 2 children)
================================================================================
Detection time: 4.29ms
Children loaded: 5
Total loaded probes: 71
Total time: 15.48ms

Top 10 concepts:
  1. [L0] Proposition   0.931
       Path: Entity → Abstract → Proposition
  2. [L2] Seat          0.913  ← NEW! Loaded via Furniture parent
       Path: Entity → Physical → Object → Artifact → Furniture → Seat
  3. [L1] Shelf         0.824
       Path: Entity → Physical → Object → Artifact → Shelf
  ...

================================================================================
FINAL STATISTICS
================================================================================
Total concepts in metadata: 10,804
Loaded in memory: 71
Memory footprint: 0.66%
Estimated memory: ~92MB (activation probes)

Loaded probes by layer:
  Layer 0:   14 probes
  Layer 1:   52 probes
  Layer 2:    5 probes
================================================================================
```

Notice how "Seat" (Layer 2) only appears after "Furniture" (Layer 1) scores high enough to trigger loading its children!

## Conclusion

Dynamic hierarchical probe loading enables running massive concept ontologies (110K+ concepts) with minimal memory by:
1. Only loading base layers initially
2. Expanding children when parents fire
3. Caching frequently accessed probes
4. Unloading cold branches

This approach reduces memory usage by **95%+** while maintaining detection quality through intelligent hierarchical exploration. 🚀
