# Temporal Activation Sequence Approach

## Key Insight

Instead of pooled single activations, we need **temporal sequences** to interpret actual model outputs. This enables:
1. **Polysemantic detection** - Multiple concepts active simultaneously
2. **Temporal dynamics** - Concept evolution over generation
3. **Compositional semantics** - Co-occurring concepts

## Architecture Change

### Old Approach (Multi-class on pooled vectors)
```python
# Extract single pooled activation
activation = pool(hidden_states)  # [hidden_dim]

# Train 1 multi-class classifier
logits = classifier(activation)  # [num_concepts]
predicted_concept = argmax(logits)  # Single concept

# Problem: Can't detect multiple concepts, no temporal info
```

### New Approach (Binary per concept on sequences)
```python
# Extract activation SEQUENCE during generation
sequence = hidden_states  # [seq_len, hidden_dim]

# Train 50K BINARY classifiers (one per concept)
for concept in concepts:
    prob = binary_classifier[concept](sequence)
    if prob > threshold:
        active_concepts.add(concept)

# Benefits: Multi-hot detection, temporal awareness
```

## Data Collection (Stage 1.5 Temporal)

### For each concept:
1. **Positive samples** (n=50):
   - Prompt: "What is {concept}?"
   - Generate 20 tokens with temp=1.0
   - Extract activation at each generation step
   - Result: 50 sequences of shape `[~20, hidden_dim]`

2. **Negative samples** (n=50):
   - Prompt: "What is NOT {concept}?"
   - Same generation process
   - Result: 50 negative sequences

### Storage format:
```
HDF5 structure:
/layer_-1/
    /concept_0/
        positive_sequences: [50, 20, hidden_dim]
        negative_sequences: [50, 20, hidden_dim]
        positive_lengths: [50]  # Actual lengths (may be < 20)
        negative_lengths: [50]
    /concept_1/
        ...
```

## Training (Binary Classifiers)

### For each of 50K concepts:
```python
# Load pos/neg sequences for this concept
dataset = SequenceDataset(h5_path, concept_id)

# Model: LSTM + binary head
model = LSTM(hidden_dim, lstm_dim=256, bidirectional=True)
classifier = Linear(lstm_dim*2, 1) + Sigmoid()

# Train on binary classification
for seq, label in dataset:
    prob = model(seq)
    loss = BCE(prob, label)  # 1 if pos, 0 if neg

# Result: 50K independent binary classifiers
```

### Expected accuracy:
- Per-concept binary: 85-95% (easier than 50K-way multi-class)
- Better than 55.7% multi-class accuracy
- Each concept gets focused training

## Production Usage (Sliding Window)

### Interpret a generation:
```python
# Model generates response
response = model.generate("Explain democracy")

# Extract activation sequence
activations = extract_activations(response)  # [gen_len, hidden_dim]

# Sliding window with stride
timeline = []
for window in sliding_windows(activations, size=20, stride=5):
    # Run ALL 50K classifiers on this window
    active_concepts = {}
    for concept_id, classifier in classifiers.items():
        prob = classifier(window)
        if prob > threshold:  # e.g., 0.5
            active_concepts[concept] = prob

    timeline.append({
        'window_position': window_start,
        'concepts': active_concepts
    })
```

### Example output:
```python
timeline = [
    {
        'window_position': 0,
        'concepts': {
            'democracy': 0.89,
            'governance': 0.76,
            'voting': 0.62
        }
    },
    {
        'window_position': 5,
        'concepts': {
            'democracy': 0.82,
            'representation': 0.73,
            'citizens': 0.69,
            'rights': 0.58
        }
    },
    # ... temporal evolution
]
```

## Advantages

1. **Polysemy handled naturally**
   - Window can activate multiple concepts
   - No forced single-concept decision

2. **Temporal narratives**
   - See how concepts evolve during generation
   - "Started with democracy, shifted to voting systems, then election security"

3. **Compositional semantics**
   - Detect concept co-occurrence patterns
   - "democracy + military = military dictatorship concerns"

4. **Better training signal**
   - Binary classification easier than 50K-way
   - Each concept gets full dataset
   - Contrastive learning (pos vs neg)

5. **Interpretable confidence**
   - Probability scores per concept
   - Threshold tunable per use case

## Implementation Status

- [x] Stage 1.5 temporal sequence extraction
- [x] Binary classifier training script
- [ ] Download Gemma-3-4b-pt (in progress)
- [ ] Test on 10 concepts
- [ ] Scale to 1K concepts
- [ ] Production sliding window inference
- [ ] Visualization/narrative reconstruction

## File Sizes (estimated)

- **10 concepts × 100 samples × 20 tokens**: ~12 MB
- **1K concepts × 100 samples × 20 tokens**: ~1.2 GB
- **50K concepts × 100 samples × 20 tokens**: ~60 GB

## Training Time (estimated)

- **Per concept**: ~30 sec (100 samples, 10 epochs)
- **1K concepts**: ~8 hours (sequential)
- **50K concepts**: ~17 days (sequential) or ~17 hours (1000x parallel)

**Optimization**: Train high-priority concepts first, parallelize across GPUs.
