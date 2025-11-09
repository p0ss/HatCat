# Per-Token Training & PCA Subspace Analysis

## Critical Issue: Current Training Uses Mean Pooling

### What We're Doing Now (WRONG!)

```python
# src/training/sumo_classifiers.py:52
hidden_states = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_dim]
pooled = hidden_states.mean(dim=1)  # [1, hidden_dim] ← WRONG!
```

**Problem**:
- Prompt: "The cat sat on the mat" (7 tokens)
- We average ALL 7 token activations into one vector
- Concept "Feline" might only activate on token "cat"
- Signal gets **diluted 7x** by averaging with non-feline tokens

**Example**:
```python
Token activations for "Feline" concept:
  "The"  : [-0.1, 0.2, -0.3, ...]  # Not feline
  "cat"  : [0.9, 0.8, 0.95, ...]   # STRONG feline signal!
  "sat"  : [-0.2, 0.1, -0.1, ...]  # Not feline
  "on"   : [0.0, -0.1, 0.1, ...]   # Not feline
  "the"  : [-0.1, 0.2, -0.2, ...]  # Not feline
  "mat"  : [-0.3, 0.1, 0.0, ...]   # Not feline

Mean pooled: [0.03, 0.2, 0.08, ...]  # Weak signal (diluted!)

Result: Classifier can't learn - signal is too weak!
```

### What We SHOULD Do: Per-Token Training

```python
# Extract activations for EACH token separately
for prompt in prompts:
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_dim]

    # Train on EACH token individually
    for token_idx in range(seq_len):
        token_activation = hidden_states[0, token_idx, :]  # [hidden_dim]
        token_text = tokenizer.decode(inputs.input_ids[0, token_idx])

        # Label: Is this specific token related to concept?
        label = is_concept_token(token_text, concept)

        # Train classifier on this token
        activations.append(token_activation)
        labels.append(label)
```

**Benefits**:
- Strong signal (no dilution)
- Can detect concept appearing/disappearing during generation
- Temporal resolution for dissonance measurement
- Matches how we'll USE the probes (per-token during generation)

## Two Critical Questions

### Question 1: What Tokens Get Positive Labels?

**Option A: Strict - Only exact concept tokens**
```python
Prompt: "The cat sat on the mat"
Concept: "Feline"

Labels:
  "The"  : 0  (not feline)
  "cat"  : 1  (is feline!)
  "sat"  : 0  (not feline)
  ...
```

**Pros**: Clear signal, no ambiguity
**Cons**: Very sparse labels (most tokens = 0), hard to learn context

**Option B: Contextual - Tokens in relevant context**
```python
Prompt: "The cat sat on the mat"
Concept: "Feline"

Labels:
  "The"  : 0  (not in feline context)
  "cat"  : 1  (feline token)
  "sat"  : 1  (feline is doing the sitting - attribute)
  "on"   : 0.5  (spatial relation to feline)
  "the"  : 0  (not feline)
  "mat"  : 0  (object, not feline)
```

**Pros**: Richer signal, learns conceptual relationships
**Cons**: Fuzzy labels, hard to define "context"

**Option C: Window - Tokens ±2 around concept**
```python
Prompt: "The cat sat on the mat"
Concept: "Feline"

Labels:
  "The"  : 1  (window before "cat")
  "cat"  : 1  (concept token)
  "sat"  : 1  (window after "cat")
  "on"   : 0  (outside window)
  "the"  : 0
  "mat"  : 0
```

**Pros**: Simple rule, captures immediate context
**Cons**: Arbitrary window size, includes irrelevant tokens

### Question 2: Do We Need PCA Subspace Analysis?

**The PCA Trick** (from interpretability research):

Instead of training on full 2560-dim hidden state, project onto concept-specific subspace:

```python
# Step 1: Find concept-relevant subspace
# Collect activations for concept and non-concept examples
concept_activations = []  # [n_pos, hidden_dim]
non_concept_activations = []  # [n_neg, hidden_dim]

# Compute difference in means
mean_diff = concept_activations.mean(0) - non_concept_activations.mean(0)

# PCA on concept examples to find top-k directions
pca = PCA(n_components=50)  # Reduce 2560 → 50 dims
pca.fit(concept_activations)

# Step 2: Project activations onto subspace
concept_subspace = pca.components_  # [50, 2560]
projected = activations @ concept_subspace.T  # [n_samples, 50]

# Step 3: Train classifier on 50-dim instead of 2560-dim
classifier.train(projected, labels)
```

**Benefits**:
- **Faster training**: 50-dim vs 2560-dim (50x fewer parameters)
- **Less overfitting**: Lower dimensional space
- **Better interpretability**: Can visualize 50-dim subspace
- **Clearer signal**: Removes irrelevant dimensions

**Costs**:
- Additional preprocessing (PCA fitting)
- Need to store subspace per concept (~10KB extra per probe)
- Slightly more complex inference (project → classify)

**Is it worth it?**

**YES, if**:
- We're training on many tokens (>1000 per concept)
- Overfitting is a problem (test acc << train acc)
- We want faster training (50-dim trains ~10x faster)

**NO, if**:
- Training is already fast enough (<5s per concept)
- We have limited data (<100 samples per concept)
- Current approach achieves target accuracy

## Recommended Approach

### Phase 1: Per-Token Training (Critical - Must Do!)

```python
def extract_per_token_activations(
    model,
    tokenizer,
    prompts: List[str],
    concept: str,
    device: str = "cuda",
    layer_idx: int = -1,
    labeling_strategy: str = "strict",  # or "window" or "contextual"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract per-token activations with concept labels."""

    all_activations = []
    all_labels = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_dim]

        # Get token texts
        token_ids = inputs.input_ids[0]
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Label each token
        for token_idx, token_text in enumerate(tokens):
            activation = hidden_states[0, token_idx, :].cpu().numpy()

            # Determine label based on strategy
            if labeling_strategy == "strict":
                label = is_exact_concept_match(token_text, concept)
            elif labeling_strategy == "window":
                label = is_in_concept_window(token_idx, tokens, concept, window=2)
            elif labeling_strategy == "contextual":
                label = get_contextual_label(token_idx, tokens, concept)

            all_activations.append(activation)
            all_labels.append(label)

    return np.array(all_activations), np.array(all_labels)
```

**Implementation Priority**: **HIGH** - This is critical for proper training!

### Phase 2: PCA Subspace (Optional - Profile First!)

```python
def train_with_pca_subspace(
    activations: np.ndarray,  # [n_samples, 2560]
    labels: np.ndarray,
    n_components: int = 50,
) -> Tuple[nn.Module, PCA]:
    """Train classifier on PCA-projected subspace."""

    # Separate positive and negative examples
    pos_activations = activations[labels == 1]
    neg_activations = activations[labels == 0]

    # Fit PCA on positive examples (concept-specific subspace)
    pca = PCA(n_components=n_components)
    pca.fit(pos_activations)

    # Project all activations onto subspace
    projected = pca.transform(activations)  # [n_samples, 50]

    # Train classifier on projected space
    classifier = SimpleMLP(input_dim=50)  # 50 instead of 2560!
    classifier.train(projected, labels)

    return classifier, pca


def inference_with_pca(hidden_state, classifier, pca):
    """Inference with PCA projection."""
    projected = pca.transform(hidden_state)  # [1, 2560] → [1, 50]
    prob = classifier(projected)
    return prob
```

**Implementation Priority**: **MEDIUM** - Test per-token training first, add PCA if needed

## Expected Impact

### Per-Token Training

**Before (mean pooling)**:
```python
Prompt: "The cat sat on the mat" (7 tokens)
Training samples: 1 (mean of 7 tokens)
Signal strength: Weak (diluted 7x)
Temporal resolution: None
Expected accuracy: 0.70-0.80 (poor)
```

**After (per-token)**:
```python
Prompt: "The cat sat on the mat" (7 tokens)
Training samples: 7 (one per token)
Signal strength: Strong (undiluted)
Temporal resolution: Full
Expected accuracy: 0.90-0.95 (good!)
```

**Impact on training data**:
- 10 prompts × 30 tokens/prompt = **300 training samples** (vs 10 before)
- 30x more data from same prompts!
- Better signal → faster graduation

### PCA Subspace (if added)

**Before (full 2560-dim)**:
```python
Model: SimpleMLP(2560 → 128 → 64 → 1)
Parameters: ~330K parameters
Training time: ~2-5s per concept
Memory: ~1.3MB per probe
```

**After (PCA 50-dim)**:
```python
Model: SimpleMLP(50 → 128 → 64 → 1) + PCA projection
Parameters: ~9K parameters (36x fewer!)
Training time: ~0.5-1s per concept (3-5x faster)
Memory: ~350KB per probe + ~10KB for PCA (4x smaller)
```

## Implementation Plan

### Step 1: Implement Per-Token Extraction (~2 hours)

```python
# New function in src/training/sumo_classifiers.py
def extract_per_token_activations(...):
    # As described above
```

### Step 2: Update Training Loop (~1 hour)

```python
# In train_layer()
X_train, y_train = extract_per_token_activations(
    model, tokenizer, train_prompts, concept
)
X_test, y_test = extract_per_token_activations(
    model, tokenizer, test_prompts, concept
)

# Now X_train has many more samples!
print(f"  Generated {len(y_train)} per-token samples from {len(train_prompts)} prompts")
```

### Step 3: Test on Small Set (~1 hour)

```bash
# Train 10 concepts with per-token approach
python scripts/train_sumo_classifiers.py \
    --layers 0 \
    --concepts 10 \
    --per-token-training  # NEW FLAG

# Compare accuracy vs mean-pooling baseline
```

### Step 4: Profile & Decide on PCA (~2 hours)

```python
# Measure:
# 1. Training time with/without PCA
# 2. Accuracy with/without PCA
# 3. Overfitting degree (train vs test acc gap)

# If PCA helps, implement it. Otherwise, skip.
```

**Total**: ~6 hours for per-token training + optional PCA

## Labeling Strategy Recommendation

**Start with "strict"** (exact concept match):
- Simple, unambiguous
- Works well for nouns (cat, dog, chair)
- Can extend to "window" later if needed

```python
def is_exact_concept_match(token_text, concept):
    """Check if token is directly related to concept."""

    # Get WordNet synsets for token
    synsets = get_synsets(token_text.strip().lower())

    # Get SUMO mapping
    for synset in synsets:
        sumo_concept = get_sumo_from_synset(synset)
        if sumo_concept == concept or is_child_of(sumo_concept, concept):
            return 1

    return 0
```

For text probes, labeling is even simpler:
```python
# Text probe trains on prompt TEXT, not tokens
# So no per-token labeling needed!
text_probe.train(prompts, labels)  # One label per prompt (easier)
```

## Critical Realization: Text vs Activation Probes Differ!

**Activation Probes**:
- Train per-token (need temporal resolution)
- Use hidden states (continuous vectors)
- PCA might help (high-dim → low-dim)

**Text Probes**:
- Train per-prompt (already discrete)
- Use token text (sparse TF-IDF)
- PCA not applicable (already low-dim after TF-IDF)

**This means**:
- Per-token training: **Activation probes only**
- Text probes: Keep current approach (per-prompt is correct!)

## Summary

**Must Do** (Priority 1):
- [x] Implement per-token activation extraction
- [x] Update training loop to use per-token
- [x] Test on small concept set

**Should Profile** (Priority 2):
- [ ] Measure impact on accuracy
- [ ] Measure 30x data multiplier effect
- [ ] Check if it helps graduation speed

**Optional** (Priority 3):
- [ ] Implement PCA subspace if needed
- [ ] Test contextual/window labeling strategies

**Expected Result**: Better accuracy, faster graduation, proper temporal resolution for real-time monitoring!
