# HatCat Source Modules

Core library code extracted from validated experimental phases.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     HatCat Architecture                          │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │  Language Model  │
                    │  (Gemma-3-4b)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Model Layers    │
                    │  (2560-dim)      │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  TRAINING     │  │    STEERING     │  │  ENCYCLOPEDIA   │
│               │  │                 │  │                 │
│ • Activations │  │ • Extraction    │  │ • WordNet Graph │
│ • Classifier  │  │ • Hooks         │  │ • Concepts      │
│ • Data Gen    │  │ • Evaluation    │  │ • Relationships │
└───────┬───────┘  │ • Subspace      │  └────────┬────────┘
        │          └────────┬────────┘           │
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
                   ┌────────▼─────────┐
                   │  Controllable    │
                   │   Generation     │
                   └──────────────────┘

Flow 1: Training Pipeline (Phase 4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Encyclopedia → Data Generation → Model Activations
                                        ↓
                                  Binary Classifier
                                        ↓
                                  Concept Detector

Flow 2: Steering Pipeline (Phase 5/6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model → Concept Extraction → Subspace Removal
                                    ↓
                              Clean Vector
                                    ↓
                         Steering Hook (Layer -1)
                                    ↓
                          Controllable Generation
                                    ↓
                         Semantic Evaluation (Δ)

Key Interfaces:
┌─────────────────────────────────────────────────────────────┐
│ training.get_mean_activation(model, tokenizer, prompt)      │
│   → np.ndarray[hidden_dim]                                  │
│                                                             │
│ steering.extract_concept_vector(model, tokenizer, concept)  │
│   → np.ndarray[hidden_dim]                                  │
│                                                             │
│ steering.apply_subspace_removal(vectors, method)            │
│   → np.ndarray[n_concepts, hidden_dim]                      │
│                                                             │
│ steering.generate_with_steering(model, vector, strength)    │
│   → str                                                     │
│                                                             │
│ steering.compute_semantic_shift(text, core, neg, embed)     │
│   → float (Δ metric)                                        │
└─────────────────────────────────────────────────────────────┘
```

## Structure

```
src/
├── steering/          # Phase 4/5: Concept steering for controllable generation
│   ├── extraction.py  # Extract concept vectors from activations
│   ├── hooks.py       # Forward hooks for steering during generation
│   ├── evaluation.py  # Semantic shift measurement (Δ metric)
│   └── subspace.py    # Subspace removal (mean, PCA)
├── training/          # Phase 4: Binary classifier training
│   ├── classifier.py  # MLP classifier for concept detection
│   ├── activations.py # Activation extraction utilities
│   └── data_generation.py  # Training prompt generation
├── encyclopedia/      # Concept graph and WordNet integration
├── activation_capture/  # Model loading and hook utilities
├── interpreter/       # Semantic interpretation
└── utils/            # Storage and utilities
```

## Modules

### `src/steering`

**Concept vector extraction and manipulation for controllable generation.**

Validated methodology from Phase 5 (±0.5 working range, 73% coherence).

```python
from src.steering import extract_concept_vector, generate_with_steering

# Extract concept vector
vector = extract_concept_vector(model, tokenizer, "person")

# Generate with steering
text = generate_with_steering(
    model, tokenizer,
    prompt="Tell me about",
    steering_vector=vector,
    strength=-0.5  # Suppress "person" concept
)
```

**Key results:**
- Working range: ±0.5 (Phase 5), ±0.5 with 100% coherence (Phase 6 mean subtraction)
- Semantic shift: Δ = 0.240 ± 0.142 at neutral (0.0)
- Float32 required for stable extraction

**Subspace removal** (Phase 6):
- `mean_subtraction`: Expands working range from ±0.25 → ±0.5
- `pca_1`: 100% coherence at all tested strengths

### `src/training`

**Binary classifier training with neutral negative examples.**

Phase 4 methodology: Simple MLP trained on mean activations.

```python
from src.training import (
    train_binary_classifier,
    create_training_dataset,
    get_mean_activation
)

# Create dataset
prompts, labels = create_training_dataset(
    "person", concept_info, negative_pool,
    n_positives=10, n_negatives=10
)

# Extract activations
X = np.array([get_mean_activation(model, tokenizer, p) for p in prompts])
y = np.array(labels)

# Train classifier
classifier = train_binary_classifier(
    X, y, input_dim=2560,
    intermediate_dim=128, epochs=100
)
```

**Key results:**
- 919/1000 concepts achieve 100% test accuracy (Phase 2, 1×1 training)
- 94.5% mean detection confidence on OOD prompts

## Usage Examples

### End-to-end steering

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.steering import extract_concept_vector, generate_with_steering
import torch

# Load model (IMPORTANT: use dtype=torch.float32)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-pt",
    dtype=torch.float32,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")

# Extract concept
person_vector = extract_concept_vector(model, tokenizer, "person")

# Steer generation
text = generate_with_steering(
    model, tokenizer,
    prompt="Tell me about",
    steering_vector=person_vector,
    strength=0.5,  # Amplify concept
    max_new_tokens=50
)
```

### With subspace removal

```python
from src.steering import extract_concept_vector, apply_subspace_removal
import numpy as np

# Extract multiple concepts
concepts = ["person", "change", "action"]
vectors = np.array([
    extract_concept_vector(model, tokenizer, c)
    for c in concepts
])

# Remove shared subspace
clean_vectors = apply_subspace_removal(vectors, method="mean_subtraction")

# Use clean vectors for steering (better working range)
```

### Semantic evaluation

```python
from src.steering import build_centroids, compute_semantic_shift
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Build centroids
core, boundary, neg = build_centroids(
    "person", concept_info, neutral_pool, embed_model
)

# Evaluate generated text
delta = compute_semantic_shift(generated_text, core, neg, embed_model)
print(f"Semantic shift: {delta:.3f}")  # Positive = closer to concept
```

## Design Principles

1. **Validated methodology**: All functions extracted from successful experimental phases
2. **Float32 requirement**: Model must use `dtype=torch.float32` for stable extraction
3. **Minimal training**: Binary classifiers work with 1 positive + 1 negative sample
4. **Graph-based negatives**: WordNet semantic distance (≥5 hops) for strong separation
5. **Projection-based steering**: Formula: `steered = hidden - strength * (hidden · vector) * vector`

## Related Scripts

- `scripts/phase_4_neutral_training.py`: Classifier training with neutral negatives
- `scripts/phase_5_semantic_steering_eval.py`: Steering evaluation with Δ metric
- `scripts/phase_6_subspace_removal.py`: Subspace removal comparison

## Version History

- **2025-11-04**: Initial extraction from Phase 4/5/6
  - `src/steering`: Extraction, hooks, evaluation, subspace removal
  - `src/training`: Classifier, activations, data generation
