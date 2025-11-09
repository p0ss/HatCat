# HatCat Quick Start Guide - Manifold Steering

Complete guide to training binary classifiers and using dual-subspace manifold steering for concept manipulation.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: A100, 3090, or 4090)
- 16GB+ GPU memory for full experiments
- 50GB+ disk space for models and data

## Installation

```bash
# Clone repository
git clone <repo-url>
cd HatCat

# Setup with Poetry (recommended)
poetry install --no-root

# Activate Poetry environment
poetry shell
```

**Note**: This project uses Poetry for dependency management. See `docs/POETRY_SETUP.md` for details.

## Quick Start: 5-Minute Demo

### 1. Train Binary Classifiers (Phase 2)

Train minimal 1×1 classifiers for 5 concepts:

```bash
poetry run python scripts/train_binary_classifiers.py \
    --concept-graph data/concept_graph/wordnet_v2_top10.json \
    --model google/gemma-3-4b-pt \
    --output-dir results/classifiers_demo \
    --n-concepts 5 \
    --n-definitions 1 \
    --n-negatives 1

# Expected: 100% validation accuracy in ~3-5 minutes
```

**What this does:**
- Loads 5 concepts from WordNet (person, change, animal, object, action)
- Generates 1 positive + 1 negative definition per concept
- Trains binary classifiers (MLP) to distinguish positive from negative
- Extracts steering vectors from classifier weights

### 2. Test Basic Steering (Phase 2.5)

Evaluate steering effectiveness:

```bash
poetry run python scripts/phase_2_5_steering_eval.py \
    --classifiers results/classifiers_demo \
    --concepts person change animal object action \
    --device cuda

# Expected: ~5 minutes
# Output: Steering suppression highly effective (0.93 → 0.05 mentions)
```

### 3. Advanced Manifold Steering (Phase 6.6)

Test dual-subspace manifold steering with contamination removal:

```bash
poetry run python scripts/phase_6_6_dual_subspace.py \
    --device cuda

# Expected: ~10-15 minutes
# Output: results/phase_6_6_dual_subspace/
```

## Full Workflow: Production Pipeline

### Day 1: Scale Test (Phase 2)

Validate 1×1 minimal training scales to 100 concepts:

```bash
# Train 100 classifiers with minimal data
poetry run python scripts/phase_2_scale_test.py \
    --concept-graph data/concept_graph/wordnet_v2_top100.json \
    --model google/gemma-3-4b-pt \
    --output-dir results/phase_2_scale_100 \
    --n-definitions 1 \
    --n-negatives 1

# Expected: ~30-40 minutes
# Expected accuracy: 96-100% @ 100 concepts
```

**Key Metrics:**
- 100% test accuracy per classifier
- ~20-30 seconds per concept
- ~5 KB storage per concept

### Day 2: Steering Quality (Phase 2.5-5)

Comprehensive steering evaluation:

```bash
# Test steering at multiple strengths [-1.0, -0.5, ..., 1.0]
poetry run python scripts/phase_2_5_steering_eval.py \
    --classifiers results/phase_2_scale_100 \
    --device cuda

# Expected: ~1-2 hours for 100 concepts
```

**Expected Results:**
- Detection confidence: 94.5% mean on OOD prompts
- Negative suppression: 0.93 → 0.05 mentions (-94%)
- Positive amplification: Variable (0 to +2.00 mentions)

### Day 3: Contamination Subspace Removal (Phase 6)

Remove shared task-agnostic patterns:

```bash
# Estimate contamination subspace via PCA
poetry run python scripts/phase_6_subspace_removal.py \
    --device cuda \
    --concepts person change animal object action

# Expected: ~15 minutes
# Output: Top-5 principal components representing contamination
```

**What this does:**
1. Stacks all concept vectors into matrix
2. Runs PCA to find top-5 principal components
3. These components represent shared task-agnostic patterns
4. Later stages project these out before steering

### Day 4: Task Manifold Estimation (Phase 6.5)

Estimate concept-specific curved semantic surfaces:

```bash
# Collect activations from low-strength steered generations
poetry run python scripts/phase_6_5_manifold_steering.py \
    --device cuda

# Expected: ~20-30 minutes
# Output: Task manifold vectors for each concept
```

**What this does:**
1. Generate text with low-strength steering (0.05-0.15)
2. Capture activations during steered generation
3. Compute concept-specific task manifold vector via PCA
4. This represents the curved semantic surface for that concept

### Day 5: Dual-Subspace Manifold Steering (Phase 6.6)

Combine contamination removal + task manifold projection:

```bash
# Full dual-subspace steering pipeline
poetry run python scripts/phase_6_6_dual_subspace.py \
    --device cuda

# Expected: ~30-45 minutes
# Output: Comprehensive steering evaluation with both subspaces
```

**Architecture:**
```python
# Step 1: Remove contamination
for basis_vector in contamination_subspace:
    projection = (hidden @ basis_vector) * basis_vector
    hidden = hidden - projection

# Step 2: Steer along task manifold
task_vector = task_manifolds[concept]
projection = (hidden @ task_vector.unsqueeze(-1)) * task_vector
steered = hidden - strength * projection
```

### Day 6: Scaling Validation (Phase 7)

Find optimal training scale via logarithmic sampling:

```bash
# Test scales [2, 4, 8, 16, 32, 64] samples/concept
poetry run python scripts/phase_7_stress_test.py \
    --device cuda

# Expected: ~30-60 minutes
# Output: SE metric curve, cost curve, knee point detection
```

**SE Metric:**
```
SE = 0.5 × (ρ_Δ,s + r_Δ,human) × coherence_rate

Where:
  ρ_Δ,s = Spearman correlation (Δ vs strength)
  r_Δ,human = Pearson correlation (Δ vs LLM judge scores)
  coherence_rate = % outputs with perplexity ≤ 1.5 × baseline
  Δ = concept_score - neg_concept_score (LLM-judged)
```

**Expected Results:**
- Knee point: 8-16 samples per concept
- Diminishing returns: ΔSE < 0.02 beyond knee point
- Optimal scale: ~16 samples for production deployment

## Common Operations

### Check Classifier Training Progress

```bash
# View training logs
tail -f results/classifiers_demo/train.log

# Check validation accuracy
grep "Validation accuracy" results/classifiers_demo/train.log
```

### Visualize Steering Effects

```bash
# Generate steering scatter plots (Phase 6.6)
python -c "
import matplotlib.pyplot as plt
import json
from pathlib import Path

results_dir = Path('results/phase_6_6_dual_subspace')
with open(results_dir / 'results.json') as f:
    data = json.load(f)

# Plot steering effectiveness vs strength
for concept in data['concepts']:
    strengths = concept['strengths']
    mentions = concept['semantic_mentions']
    plt.plot(strengths, mentions, marker='o', label=concept['name'])

plt.xlabel('Steering Strength')
plt.ylabel('Semantic Mentions')
plt.legend()
plt.savefig('steering_effectiveness.png')
print('✓ Saved steering_effectiveness.png')
"
```

### Export Steering Vectors for Inference

```bash
python -c "
import torch
from pathlib import Path

# Load trained classifiers
classifier_dir = Path('results/classifiers_demo')
steering_vectors = {}

for concept_file in classifier_dir.glob('*_classifier.pt'):
    concept = concept_file.stem.replace('_classifier', '')
    checkpoint = torch.load(concept_file)
    steering_vectors[concept] = checkpoint['steering_vector']

# Export for inference
torch.save(steering_vectors, 'steering_vectors_inference.pt')
print(f'✓ Exported {len(steering_vectors)} steering vectors')
"
```

## Troubleshooting

### "CUDA out of memory"

**Solution 1**: Reduce batch size
```bash
# In training scripts, add --batch-size flag
poetry run python scripts/train_binary_classifiers.py --batch-size 16
```

**Solution 2**: Use smaller model
```bash
# Switch to gemma-3-270m instead of 4b
--model google/gemma-3-270m
```

**Solution 3**: Use CPU (slower)
```bash
poetry run python scripts/train_binary_classifiers.py --device cpu
```

### "CUDA assertion error" during Phase 7

**Cause**: GPU state corruption from previous failed runs

**Solution**: Reset GPU state
```bash
# Option 1: Kill all CUDA processes
pkill -9 python

# Option 2: Restart system (if persistent)
sudo reboot
```

### "Classifier accuracy stuck at ~50%"

**Causes:**
1. Insufficient semantic distance between positive/negative
2. Data corruption
3. Learning rate too high

**Solutions:**
```bash
# Increase semantic distance for negatives
--min-semantic-distance 10  # default is 5

# Check data integrity
python -c "
import h5py
with h5py.File('data/processed/concept_data.h5', 'r') as f:
    print(f'Concepts: {len(f[\"concepts\"])}')
    print(f'Activation shape: {f[\"activations\"].shape}')
"

# Reduce learning rate
--lr 1e-5  # default is 1e-4
```

### "Task manifold estimation fails: Insufficient activations"

**Cause**: Dtype mismatch between model (float16) and steering tensors (float32)

**Solution**: Ensure dynamic dtype matching in hooks
```python
# In src/steering/hooks.py and manifold.py
v_matched = v_tensor.to(dtype=hidden.dtype)  # Match model precision
```

## Project Structure Reference

```
HatCat/
├── src/
│   ├── steering/
│   │   ├── manifold.py               # Phase 6.6 dual-subspace manifold steering
│   │   └── hooks.py                  # Forward hooks for steering
│   ├── interpreter/
│   │   ├── model.py                  # Binary classifier (MLP)
│   │   └── steering.py               # Steering vector extraction
│   └── encyclopedia/
│       └── wordnet_graph_v2.py       # WordNet graph builder
├── scripts/
│   ├── train_binary_classifiers.py   # Phase 2: Train classifiers
│   ├── phase_2_scale_test.py         # Phase 2: Scale validation
│   ├── phase_2_5_steering_eval.py    # Phase 2.5: Steering evaluation
│   ├── phase_6_subspace_removal.py   # Phase 6: Contamination removal
│   ├── phase_6_5_manifold_steering.py # Phase 6.5: Task manifold
│   ├── phase_6_6_dual_subspace.py    # Phase 6.6: Dual-subspace steering
│   └── phase_7_stress_test.py        # Phase 7: Scaling validation
├── data/
│   └── concept_graph/
│       ├── wordnet_v2_top10.json     # 10 concepts
│       ├── wordnet_v2_top100.json    # 100 concepts
│       └── wordnet_v2_top1000.json   # 1K concepts
└── results/                          # Experiment outputs
```

## Key Concepts Explained

### Binary Classifiers
- One classifier per concept (polysemy-native)
- Trained to distinguish positive from negative definitions
- Steering vector extracted from classifier weights

### Contamination Subspace
- Shared task-agnostic patterns across all concepts
- Estimated via PCA on all concept vectors
- Top-5 principal components removed before steering

### Task Manifold
- Concept-specific curved semantic surface
- Estimated from activations during low-strength steering
- Represents the natural trajectory of that concept in activation space

### Dual-Subspace Steering
- Step 1: Remove contamination (orthogonal projection)
- Step 2: Steer along task manifold
- Result: High-precision concept manipulation

## Next Steps

After completing this quick start:

1. **Scale to 1000 concepts**: Run Phase 2 at scale
   ```bash
   poetry run python scripts/phase_2_scale_test.py \
       --concept-graph data/concept_graph/wordnet_v2_top1000.json
   ```

2. **Production deployment**: Identify optimal scale via Phase 7
   ```bash
   poetry run python scripts/phase_7_stress_test.py
   ```

3. **10K concept encyclopedia**: Scale to full WordNet coverage
   ```bash
   # Generate 10K concept graph
   poetry run python scripts/generate_concept_graph.py \
       --n-concepts 10000 \
       --output data/concept_graph/wordnet_v2_top10000.json
   ```

4. **Real-time inference**: Implement sliding window monitoring
   - See `docs/INFERENCE_GUIDE.md` (TBD)

## Questions?

- See `README.md` for project overview
- See `docs/PHASE7_BLOCKING_ISSUES.md` for known issues
- See `TEST_DATA_REGISTER.md` for experiment history
