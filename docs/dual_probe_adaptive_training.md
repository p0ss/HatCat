# Dual Probe Adaptive Training

## Problem Statement

**You're absolutely right**: Activation and text probes have different learning curves and should use **separate adaptive training regimes**.

Pinning both to the same number of cycles will lead to:
- **Underfitting** one probe type (needs more data)
- **Overfitting** the other probe type (wasting compute)

## Current Adaptive Training (Activation Probes)

From `scripts/phase_2_adaptive_scaling.py`:

```python
class AdaptiveScaler:
    def __init__(self, target_accuracy=0.95, baseline_defs=10, baseline_rels=10):
        # Start: 10 definitions + 10 relationships
        # Increment: +1 def, +1 rel per cycle
        # Graduate: accuracy >= 95% on fixed test set
```

**Strategy**:
1. Start all concepts at 10×10 baseline (definitions × relationships)
2. Test accuracy on fixed test set
3. Concepts above 95% accuracy **graduate** - stop training
4. Add 1×1 more samples to remaining concepts
5. Repeat until all graduate

**Why it works for activation probes**:
- Deep neural network (SimpleMLP with 2 hidden layers)
- Needs ~10-50 samples to learn complex decision boundaries
- Vulnerable to overfitting with too much data
- 50 epochs per cycle

## Expected Differences for Text Probes

### Text Probes Are Simpler

**Architecture**: TF-IDF + LogisticRegression
- Linear model (no hidden layers)
- Much simpler decision boundaries
- Less prone to overfitting
- No epoch-based training (single fit)

**Implications**:
- Will likely graduate **faster** (fewer samples needed)
- Can handle **more data** without overfitting
- Training is **instant** (<0.5s vs 2-5s for activation)

### Predicted Learning Curves

```
Activation Probe:
────────────────────────────────
Samples    Accuracy    Graduate?
10×10      0.70        ❌
20×20      0.85        ❌
30×30      0.92        ❌
40×40      0.96        ✓ (graduated!)

Text Probe:
────────────────────────────────
Samples    Accuracy    Graduate?
10×10      0.88        ❌
15×15      0.94        ❌
20×20      0.97        ✓ (graduated!)

Observation: Text probes graduate ~2x faster!
```

## Proposed Dual Adaptive Training Strategy

### Independent Adaptive Cycles

Train both probe types **in parallel** with **separate graduation criteria**:

```python
class DualAdaptiveTrainer:
    def __init__(
        self,
        # Activation probe config
        activation_target_acc: float = 0.95,
        activation_baseline: int = 10,
        activation_increment: int = 1,
        activation_max_samples: int = 100,

        # Text probe config
        text_target_acc: float = 0.95,
        text_baseline: int = 10,
        text_increment: int = 5,  # Larger increments (faster learning)
        text_max_samples: int = 200,  # Can handle more data
    ):
        self.activation_scaler = AdaptiveScaler(...)
        self.text_scaler = AdaptiveScaler(...)
```

**Key differences**:

| Parameter | Activation Probes | Text Probes | Rationale |
|-----------|------------------|-------------|-----------|
| **Baseline** | 10×10 | 10×10 | Same starting point |
| **Increment** | +1×1 | +5×5 | Text learns faster, larger steps |
| **Target Acc** | 0.95 | 0.95 | Same quality bar |
| **Max Samples** | 100 | 200 | Text can handle more data |
| **Epochs** | 50 | N/A | Text is single-fit |

### Training Loop

```python
# Generate prompts ONCE (shared between both probe types)
prompts_pos, prompts_neg = generate_training_prompts(
    concept,
    n_positives=max(activation_max, text_max),
    n_negatives=max(activation_max, text_max),
)

# Adaptive training loop
iteration = 0
while not (activation_graduated and text_graduated):
    iteration += 1

    # === ACTIVATION PROBE ===
    if not activation_graduated:
        # Extract activations for current sample count
        n_act = activation_baseline + (iteration * activation_increment)
        X_train_act = extract_activations(prompts_pos[:n_act] + prompts_neg[:n_act])

        # Train activation probe
        activation_probe.train(X_train_act, y_train, epochs=50)

        # Test on fixed test set
        acc_act = evaluate(activation_probe, test_set_activations)
        if acc_act >= 0.95:
            activation_graduated = True
            print(f"✓ Activation probe graduated at {n_act} samples, acc={acc_act:.3f}")

    # === TEXT PROBE ===
    if not text_graduated:
        # Use same prompts, different sample count
        n_text = text_baseline + (iteration * text_increment)

        # Train text probe (instant, no epochs)
        text_probe.train(prompts_pos[:n_text], prompts_neg[:n_text])

        # Test on fixed test set
        acc_text = evaluate(text_probe, test_set_text)
        if acc_text >= 0.95:
            text_graduated = True
            print(f"✓ Text probe graduated at {n_text} samples, acc={acc_text:.3f}")

    # Early stop if both graduated
    if activation_graduated and text_graduated:
        break
```

### Expected Behavior

**Scenario 1: Text graduates first (most common)**:
```
Iteration 1:
  Activation (10 samples): 0.72 ❌
  Text (10 samples):       0.88 ❌

Iteration 2:
  Activation (11 samples): 0.75 ❌
  Text (15 samples):       0.94 ❌

Iteration 3:
  Activation (12 samples): 0.78 ❌
  Text (20 samples):       0.97 ✓ GRADUATED!

Iteration 4-30:
  Activation (13-40 samples): 0.80-0.96
  Text: (already done, not trained)

Iteration 30:
  Activation (40 samples): 0.96 ✓ GRADUATED!

Result: Text saved 27 cycles of training!
```

**Scenario 2: Activation graduates first (rare)**:
```
Concept with very simple decision boundary
(e.g., "Physical" - everything has mass)

Iteration 1-5:
  Activation graduates at 15 samples (0.97)
  Text still at 0.92

Iteration 6-10:
  Text graduates at 35 samples (0.96)

Result: Activation saved compute, text needed more data
```

## Implementation Plan

### Phase 1: Measure Learning Curves (First!)

**Before implementing dual adaptive**, we need to **profile** text probe learning:

```python
# Train text probes with varying sample counts
for n_samples in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
    text_probe = BinaryTextProbe(concept)
    text_probe.train(prompts[:n_samples])

    acc = evaluate(text_probe, test_set)
    print(f"{n_samples} samples → {acc:.3f} accuracy")

# Example output:
# 5 samples  → 0.65
# 10 samples → 0.82
# 15 samples → 0.91
# 20 samples → 0.96  ← Graduated!
# ...
```

**Metrics to measure**:
- Samples to graduation (avg, median, std)
- Gradient of learning curve
- Overfitting point (if any)

### Phase 2: Tune Hyperparameters

Based on profiling, set optimal values:

```python
# If text probes graduate at ~20 samples avg:
TEXT_BASELINE = 10      # Start point
TEXT_INCREMENT = 5      # Larger steps (faster learning)
TEXT_MAX_SAMPLES = 50   # Early stop (rarely needs more)

# If activation probes graduate at ~40 samples avg:
ACTIVATION_BASELINE = 10
ACTIVATION_INCREMENT = 1  # Smaller steps (slower learning)
ACTIVATION_MAX_SAMPLES = 100
```

### Phase 3: Implement Dual Trainer

```python
class DualAdaptiveTrainer:
    """Train activation + text probes with independent adaptive cycles."""

    def train_concept(self, concept, prompts_pos, prompts_neg):
        """Train both probe types until both graduate."""

        # Fixed test sets (same for both)
        test_activations = extract_activations(test_prompts)
        test_texts = test_prompts

        # Independent state
        activation_level = ACTIVATION_BASELINE
        text_level = TEXT_BASELINE
        activation_graduated = False
        text_graduated = False

        iteration = 0
        while not (activation_graduated and text_graduated):
            iteration += 1

            # Train activation (if not graduated)
            if not activation_graduated:
                activation_level += ACTIVATION_INCREMENT
                # ... train and test

            # Train text (if not graduated)
            if not text_graduated:
                text_level += TEXT_INCREMENT
                # ... train and test

        return {
            'activation_samples': activation_level,
            'text_samples': text_level,
            'activation_iterations': iteration if activation_graduated else None,
            'text_iterations': iteration if text_graduated else None,
        }
```

### Phase 4: Integration with Current Pipeline

Update `train_sumo_classifiers()`:

```python
def train_layer(
    layer,
    model,
    tokenizer,
    use_adaptive_training=True,  # NEW
    train_text_probes=True,
):
    for concept in concepts:
        # Generate prompts (once, shared)
        prompts_pos, prompts_neg = generate_prompts(concept, max_samples=200)

        if use_adaptive_training:
            # Dual adaptive training
            trainer = DualAdaptiveTrainer()
            results = trainer.train_concept(concept, prompts_pos, prompts_neg)

            print(f"  Activation: {results['activation_samples']} samples")
            print(f"  Text:       {results['text_samples']} samples")
        else:
            # Fixed training (old way)
            train_activation_probe(prompts_pos[:10], prompts_neg[:10])
            if train_text_probes:
                train_text_probe(prompts_pos[:10], prompts_neg[:10])
```

## Benefits

✅ **Efficient**: Don't waste compute training already-graduated probes

✅ **Optimal**: Each probe type gets exactly the data it needs

✅ **Flexible**: Can tune learning rates independently

✅ **Data Efficient**: Shared prompts (generate once, use twice)

✅ **Insightful**: Learn empirical differences between probe types

## Expected Resource Savings

**Baseline (no adaptive, 100 samples each)**:
- Generate 100 prompts
- Extract 100 activations (slow)
- Train activation probe on 100 samples
- Train text probe on 100 samples
- **Total time per concept**: ~10-15s

**With dual adaptive**:
- Generate 100 prompts (once)
- Activation graduates at ~40 samples → extract only 40
- Text graduates at ~20 samples → train only on 20
- **Total time per concept**: ~5-7s
- **Speedup**: 2x

**At scale (10,000 concepts)**:
- Baseline: 10,000 × 15s = 41.7 hours
- Adaptive: 10,000 × 7s = 19.4 hours
- **Savings**: 22.3 hours (53% reduction!)

## Open Questions (Need Profiling!)

1. **What's the typical graduation point for text probes?**
   - Hypothesis: ~15-25 samples (vs ~30-50 for activation)
   - Need empirical data

2. **Do text probes overfit?**
   - Hypothesis: No (linear model, high capacity)
   - If yes, need regularization tuning

3. **What's the optimal increment size?**
   - Hypothesis: +5 for text, +1 for activation
   - Need learning curve analysis

4. **Should target accuracy differ?**
   - Hypothesis: Same 0.95 for both
   - Could go higher for text (0.97) if it's easier

5. **Do some concepts need more data for text probes?**
   - Hypothesis: Abstract concepts harder for text (less linguistic signal)
   - Physical concepts easier for text (clear word associations)

## Recommendation

**Step 1** (1-2 hours): Profile text probe learning curves
```bash
python scripts/profile_text_probe_learning.py \
    --concepts 100 \
    --sample-range 5,10,15,20,25,30,40,50,75,100 \
    --output results/text_probe_learning_curves.json
```

**Step 2** (1 hour): Analyze curves, set hyperparameters

**Step 3** (2-3 hours): Implement `DualAdaptiveTrainer`

**Step 4** (2-4 hours): Integrate with current training pipeline

**Total**: ~6-10 hours for full implementation

**ROI**: 2x training speedup + optimal data efficiency for 10K+ concepts!
