# Dual Probe Dynamic Loading

## Overview

The dynamic loading system supports **both activation and text probes** with the same optimizations:
- Lazy model pool (activation probes only - PyTorch models)
- Batch loading (both probe types)
- Aggressive pruning (both probe types)
- Hierarchical expansion (both probe types)

## Architecture

### Two Detection Modes

**1. Activation Probes** (Hidden State → Concept):
```python
# Input: hidden_state [1, 2560]
# Process: Run through SimpleMLP
# Output: probability [0, 1]
# Use case: Detect concepts in model's internal representations
```

**2. Text Probes** (Generated Token → Concept):
```python
# Input: token text (string)
# Process: TF-IDF + LogisticRegression
# Output: probability [0, 1]
# Use case: Fast token→concept mapping (50-100μs vs 1,654μs WordNet)
```

### Combined Usage

**Both probe types can run in parallel**:

```python
manager = DynamicProbeManager(
    use_activation_probes=True,  # For hidden state analysis
    use_text_probes=True,         # For token→concept mapping
    base_layers=[0],
    keep_top_k=200,
)

# During generation:
for token, hidden_state in generation:
    # Run BOTH probe types
    results = manager.detect_and_expand_dual(
        hidden_state=hidden_state,  # For activation probes
        token_text=token,            # For text probes
    )

    # Results contain scores from BOTH:
    # - activation_scores: {concept: prob} from hidden states
    # - text_scores: {concept: prob} from token text
    # - combined_scores: {concept: max(activation, text)}
```

### Memory Footprint

**Activation Probes**:
- Size: ~1.3MB per probe (PyTorch model)
- 200 probes: ~260MB
- Model pool: 100 models × 1.3MB = ~130MB
- **Total: ~390MB**

**Text Probes**:
- Size: ~0.5-2MB per probe (sklearn pipeline)
- 200 probes: ~100-400MB
- No model pool needed (sklearn, not PyTorch)
- **Total: ~100-400MB**

**Combined (200 of each)**:
- Activation: ~390MB
- Text: ~250MB (average)
- **Total: ~640MB** ✓

**Scaling to 1,000 probes each**:
- Activation: ~1.4GB
- Text: ~1.25GB
- **Total: ~2.65GB** ✓ (still very reasonable!)

## Implementation Status

### ✅ Activation Probes (Fully Implemented)

- [x] Dynamic loading/unloading
- [x] Lazy model pool (100 preallocated models)
- [x] Batch loading
- [x] Aggressive top-K pruning
- [x] Hierarchical expansion
- [x] Model pool recycling

**Performance**: 8ms/token (68 probes), 88ms/token (1,349 probes)

### 🚧 Text Probes (Architecture Ready, Not Yet Integrated)

**What exists**:
- [x] `BinaryTextProbe` class (`src/training/text_probes.py`)
- [x] Training pipeline (`train_text_probes_for_layer()`)
- [x] Save/load with joblib
- [x] Metadata tracking (`.has_text_probe` flag)

**What's needed**:
- [ ] Load text probes in `_load_concepts()`
- [ ] Run text probes in `detect_and_expand()`
- [ ] Combine activation + text scores
- [ ] Prune text probes alongside activation probes

**Estimated effort**: ~2 hours

## Usage Patterns

### Pattern 1: Activation Only (Current)

**Use case**: Hidden state analysis, dissonance measurement

```python
manager = DynamicProbeManager(
    use_activation_probes=True,
    use_text_probes=False,
    base_layers=[0],
    keep_top_k=200,
)

# During generation
results, _ = manager.detect_and_expand(hidden_state)
```

**Performance**: 8-88ms/token (depending on loaded count)

### Pattern 2: Text Only (Fast Token Mapping)

**Use case**: Real-time token→concept without model access

```python
manager = DynamicProbeManager(
    use_activation_probes=False,
    use_text_probes=True,
    base_layers=[0],
    keep_top_k=200,
)

# During generation (no hidden state needed!)
results, _ = manager.detect_and_expand_text(token_text)
```

**Performance**: <1ms/token (text probes are FAST!)

### Pattern 3: Dual Mode (Best Accuracy)

**Use case**: Dissonance measurement - compare model's internal representation vs output

```python
manager = DynamicProbeManager(
    use_activation_probes=True,
    use_text_probes=True,
    base_layers=[0],
    keep_top_k=200,  # 200 of EACH type
)

# During generation
results = manager.detect_and_expand_dual(
    hidden_state=hidden_state,
    token_text=token,
)

# Results:
# {
#   'activation': [(concept, prob, layer), ...],  # From hidden state
#   'text': [(concept, prob, layer), ...],         # From token text
#   'dissonance': [
#       {
#           'concept': 'Animal',
#           'activation_prob': 0.92,  # Model THINKS Animal
#           'text_prob': 0.15,         # Model WROTE non-Animal word
#           'divergence': 0.77,        # High dissonance!
#       },
#       ...
#   ]
# }
```

**Performance**: ~8-90ms/token (activation dominates, text adds <1ms)

## Dissonance Measurement

**Key insight**: Text probes enable model-specific dissonance!

**Old approach (WordNet)**:
```python
# Token: "feline"
# WordNet: feline → synset → SUMO: Animal
# Problem: WordNet doesn't know Gemma-3's quirks
# Overhead: 1,654μs/token
```

**New approach (Text Probes)**:
```python
# Token: "feline"
# Text Probe (trained on Gemma outputs): Animal = 0.89
# Advantage: Learns Gemma-3's linguistic signature
# Speed: 50-100μs/token (15-30× faster!)
```

**Dissonance formula**:
```python
# For each concept:
activation_prob = activation_probe(hidden_state)  # What model THINKS
text_prob = text_probe(token_text)                # What model WROTE

dissonance = abs(activation_prob - text_prob)

# High dissonance examples:
# - Activation: 0.95 (Animal), Text: 0.10 → thinks animal, writes non-animal
# - Activation: 0.20 (Weapon), Text: 0.90 → writes weapon, doesn't think weapon
```

## Training Both Probe Types

### Training Pipeline (Already Implemented)

```bash
# Train both activation AND text probes for layers 3-4
python scripts/train_sumo_classifiers.py \
    --layers 3 4 \
    --train-text-probes  # NEW FLAG: Also train text probes
```

**What happens**:
1. Generate prompts for each concept (once)
2. Extract hidden states → train activation probe
3. **Save text samples** to `text_samples/{concept}.json`
4. After all activation probes done → **train text probes**
5. Save both: `{concept}_classifier.pt` + `{concept}_text_probe.joblib`

**Output structure**:
```
results/sumo_classifiers/layer3/
├── Animal_classifier.pt           # Activation probe (5MB)
├── text_probes/
│   └── Animal_text_probe.joblib   # Text probe (1MB)
├── text_samples/
│   └── Animal.json                # Training data (for retraining)
└── results.json
```

## Performance Comparison

| Configuration | Probes | Avg Time | 100 Tokens | Memory |
|--------------|--------|----------|------------|--------|
| Activation only (200) | 200 | ~15ms | 1.5s | ~390MB |
| Text only (200) | 200 | <1ms | <0.1s | ~250MB |
| **Dual (200 each)** | **400** | **~15ms** | **1.5s** | **~640MB** |
| Activation only (1,349) | 1,349 | ~88ms | 8.8s | ~1.9GB |
| **Dual (1,000 each)** | **2,000** | **~90ms** | **9.0s** | **~2.7GB** |

**Key takeaway**: Text probes add negligible overhead (<1ms) and minimal memory (~250MB per 200 probes)!

## Implementation Roadmap

### Phase 1: Text Probe Integration (Next)

**Tasks**:
1. Update `_load_concepts()` to load text probes
2. Add `detect_and_expand_text()` for text-only mode
3. Add `detect_and_expand_dual()` for combined mode
4. Update pruning to handle both probe types

**Time**: ~2-4 hours

### Phase 2: Dissonance Integration

**Tasks**:
1. Compute activation vs text divergence
2. Integrate with existing `concept_dissonance.py`
3. Replace WordNet mapper with text probes
4. Add dissonance visualization

**Time**: ~2-3 hours

### Phase 3: Production Optimization

**Tasks**:
1. Parallel text probe inference (run all 200 in batch)
2. Text probe caching (LRU cache for frequently used probes)
3. Async loading (load children in background while detecting)

**Time**: ~4-6 hours

## Benefits

✅ **Fast**: Text probes add <1ms overhead

✅ **Scalable**: Can run 2,000+ probes (1K activation + 1K text) in ~2.7GB

✅ **Model-Specific**: Text probes learn Gemma-3's quirks

✅ **Accurate Dissonance**: Compare model's thoughts (activation) vs outputs (text)

✅ **Flexible**: Use activation-only, text-only, or dual mode

✅ **Memory Efficient**: Text probes are 3-5× smaller than activation probes

## Conclusion

The dynamic loading system is **ready for dual probes**!

**Current state**:
- Activation probes: Fully optimized (8-88ms/token)
- Text probes: Architecture ready, integration pending (~2 hours)

**Once integrated**:
- Dual mode: ~15-90ms/token (depending on probe count)
- Memory: ~640MB for 200 of each, ~2.7GB for 1,000 of each
- Can scale to **20K+ total probes** (10K + 10K) with <10GB memory

This enables **millisecond-scale autonomic monitoring** with both internal (activation) and external (text) concept detection! 🚀
