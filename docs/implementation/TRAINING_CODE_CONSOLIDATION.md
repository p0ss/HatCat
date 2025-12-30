# Training Code Consolidation Plan

## Current State Analysis

### Files in src/training/

1. **activations.py** (1.7K)
   - Function: Extract mean activation from model
   - Used by: `dual_adaptive_trainer.py`, `sumo_classifiers.py`
   - Status: ✅ ACTIVE - core utility

2. **classifier.py** (2.8K)
   - Function: Simple MLP binary classifier (BinaryClassifier)
   - Used by: `dual_adaptive_trainer.py`
   - Status: ✅ ACTIVE - core model architecture

3. **data_generation.py** (4.0K) ⚠️ LEGACY
   - Function: Simple "What is X?" prompt generation
   - Used by: Legacy phase_4/phase_5 scripts via `__init__.py` (now removed)
   - Status: ❌ DEPRECATED - superseded by sumo_data_generation.py

4. **sumo_data_generation.py** (13K)
   - Function: SUMO + WordNet hierarchy-aware prompt generation
   - Features: Category relationships, WordNet relationships, camelCase splitting
   - Used by: `sumo_classifiers.py` (main training)
   - Status: ✅ ACTIVE - current approach

5. **dual_adaptive_trainer.py** (13K)
   - Function: Adaptive training that increases sample size until 95% accuracy
   - Used by: `sumo_classifiers.py` when `use_adaptive_training=True`
   - Status: ✅ ACTIVE - but complex

6. **sumo_classifiers.py** (21K)
   - Function: Main training orchestration script
   - Handles: Activation lenses, text lenses, adaptive vs fixed training
   - Status: ✅ ACTIVE - main entry point

7. **text_lenses.py** (8.3K) ⚠️ UNUSED
   - Function: TF-IDF based text classifiers (sklearn)
   - Features: Train LogisticRegression on TF-IDF features
   - Used by: NOT USED ANYMORE (replaced by embedding centroids)
   - Status: ❌ DEPRECATED - kept for backward compatibility only

8. **embedding_centroids.py** (5.0K)
   - Function: Compute embedding centroids for text→concept mapping
   - Features: Cosine similarity instead of TF-IDF
   - Used by: `sumo_classifiers.py` when `train_text_lenses=True`
   - Status: ✅ ACTIVE - but optional (text lenses vs activation lenses)

### Current Training Flow

```
sumo_classifiers.py (main entry point)
  │
  ├─> sumo_data_generation.py (generate training prompts)
  │     └─> split_camel_case() (for better prompts)
  │
  ├─> activations.py (extract hidden states from prompts)
  │
  ├─> [IF use_adaptive_training=True]
  │     └─> dual_adaptive_trainer.py
  │           └─> classifier.py (SimpleMLP model)
  │           └─> Train until 95% accuracy
  │
  ├─> [IF use_adaptive_training=False]
  │     └─> train_simple_classifier() (inline)
  │           └─> classifier.py (SimpleMLP model)
  │
  └─> [IF train_text_lenses=True]
        └─> embedding_centroids.py (compute concept centroids)
```

## Issues Identified

### 1. Too Many Code Paths
- **Adaptive vs Fixed training** - two completely different training loops
- **Text lenses vs No text lenses** - optional feature that complicates everything
- **Legacy vs SUMO data generation** - duplicate functionality

### 2. Unclear Responsibilities
- `sumo_classifiers.py` does EVERYTHING (orchestration, fixed training, file I/O)
- `dual_adaptive_trainer.py` duplicates training logic
- Text lens code exists but we use centroids instead

### 3. Dead Code
- `data_generation.py` - only used by legacy scripts
- `text_lenses.py` - TF-IDF approach replaced by centroids
- Both still in `__init__.py` exports (just cleaned up)

### 4. Missing Validation
- No code to validate lenses post-training
- No detection of "universal firing" lenses like PostalPlace
- 95% accuracy on train/test but fails on real data

## Proposed Consolidation

### Phase 1: Remove Dead Code

**Move to archive/**
```
src/training/data_generation.py          -> archive/training/data_generation.py
src/training/text_lenses.py              -> archive/training/text_lenses.py
```

**Update imports** - Already done, but verify nothing breaks

### Phase 2: Simplify Training Architecture

**Option A: Keep Adaptive Training (Recommended)**
- Remove fixed training path from `sumo_classifiers.py`
- Always use `dual_adaptive_trainer.py`
- Simplify to single code path

**Option B: Remove Adaptive Training**
- Remove `dual_adaptive_trainer.py`
- Use only fixed training with larger sample sizes
- Simpler but less flexible

**Recommendation: Option A** - Adaptive training gives better results, just remove the branching

### Phase 3: Unified Training Interface

Create `src/training/trainer.py`:
```python
class ConceptTrainer:
    """Single unified interface for training concept lenses."""

    def train_activation_lens(
        self,
        concept: Dict,
        all_concepts: List[Dict],
        model, tokenizer, device,
        target_accuracy: float = 0.95,
        max_samples: int = 100,
    ) -> Dict:
        """Train activation lens with adaptive sampling."""
        # Generate training data
        # Extract activations
        # Train until target accuracy
        # Return classifier + metrics

    def compute_text_centroid(
        self,
        concept: Dict,
        model, tokenizer, device,
    ) -> np.ndarray:
        """Compute embedding centroid for text→concept mapping."""
        # Generate positive prompts
        # Extract embeddings
        # Compute centroid
        # Return centroid array
```

Refactor `sumo_classifiers.py` to use this interface:
```python
trainer = ConceptTrainer(target_accuracy=0.95)

for concept in concepts:
    # Train activation lens
    result = trainer.train_activation_lens(
        concept, all_concepts, model, tokenizer, device
    )

    # Compute text centroid (optional)
    if train_text_lenses:
        centroid = trainer.compute_text_centroid(
            concept, model, tokenizer, device
        )
```

### Phase 4: Add Lens Validation

Create `src/training/lens_validation.py`:
```python
def validate_lens_calibration(
    lens: BinaryClassifier,
    concept_name: str,
    test_prompts: Dict[str, str],  # domain -> prompt
    model, tokenizer, device,
    expected_domain: str,
) -> Dict[str, float]:
    """
    Validate lens fires specifically on expected domain.

    Returns calibration metrics:
    - specificity: avg rank on non-target domains
    - sensitivity: rank on target domain
    - calibration_score: specificity / (1 + sensitivity)
    """
```

Add validation step to training:
```python
if validate_lenses:
    calibration = validate_lens_calibration(
        classifier, concept_name,
        TEST_PROMPTS, model, tokenizer, device,
        expected_domain=infer_domain(concept)
    )

    if calibration['calibration_score'] < 0.5:
        print(f"  ⚠️  Lens failed calibration - fires too broadly")
        # Option: skip saving, or flag for review
```

## File Structure After Consolidation

```
src/training/
  __init__.py                    # Clean exports
  trainer.py                     # NEW: Unified training interface
  sumo_data_generation.py        # Prompt generation (keep)
  lens_validation.py            # NEW: Post-training validation

  # Core utilities (keep)
  activations.py                 # Activation extraction
  classifier.py                  # SimpleMLP model
  embedding_centroids.py         # Centroid computation

  # Main script (simplified)
  sumo_classifiers.py            # Orchestration only

archive/training/
  data_generation.py             # Legacy prompt generation
  text_lenses.py                 # Legacy TF-IDF approach
  dual_adaptive_trainer.py       # Absorbed into trainer.py
```

## Migration Plan

### Step 1: Create new files (non-breaking)
- [ ] Create `src/training/trainer.py` with ConceptTrainer
- [ ] Create `src/training/lens_validation.py`
- [ ] Add tests for new interfaces

### Step 2: Refactor sumo_classifiers.py (breaking)
- [ ] Update to use ConceptTrainer interface
- [ ] Remove dual code paths (adaptive vs fixed)
- [ ] Add validation step
- [ ] Test on small dataset

### Step 3: Archive legacy code
- [ ] Move `data_generation.py` to archive/
- [ ] Move `text_lenses.py` to archive/
- [ ] Move `dual_adaptive_trainer.py` to archive/ (after absorbing into trainer.py)
- [ ] Update any scripts that import these

### Step 4: Update documentation
- [ ] Update SUMO_AWARE_TRAINING.md
- [ ] Document new training interface
- [ ] Add calibration validation docs

## Negative Example Strategy

### Current Issues
1. Using other SUMO concepts as negatives (too similar semantically)
2. Lenses learn to distinguish "What is PostalPlace?" from "What is Animal?"
3. But don't learn actual semantic concept boundaries

### Proposed Improvements

**1. Domain-Diverse Negatives**
```python
NEGATIVE_DOMAINS = {
    'location': ['biology', 'technology', 'physics', 'abstract'],
    'process': ['object', 'relation', 'attribute', 'agent'],
    'physical': ['abstract', 'social', 'cognitive', 'linguistic'],
}

def generate_domain_diverse_negatives(
    concept: Dict,
    all_concepts: List[Dict],
    n_negatives: int = 20
) -> List[str]:
    """Generate negatives from semantically distant domains."""
    concept_domain = infer_domain(concept)
    target_domains = NEGATIVE_DOMAINS.get(concept_domain, [])

    # Sample concepts from distant domains
    candidates = [
        c for c in all_concepts
        if infer_domain(c) in target_domains
    ]
    return random.sample(candidates, min(n_negatives, len(candidates)))
```

**2. Natural Text Negatives**
```python
# Use actual text from different domains, not "What is X?" prompts
NATURAL_TEXT_SAMPLES = {
    'biology': [
        "The cell membrane regulates what enters and exits the cell",
        "DNA replication occurs during the S phase of the cell cycle",
    ],
    'technology': [
        "Machine learning algorithms improve with more training data",
        "Neural networks consist of interconnected layers of nodes",
    ],
    # ... more domains
}
```

**3. Hard Negatives (Related Concepts)**
```python
def generate_hard_negatives(concept: Dict, all_concepts: List[Dict]) -> List[str]:
    """Generate negatives from related but distinct concepts."""
    # For PostalPlace, use: Building, Address, Location (parent), but NOT Residence (child)
    siblings = get_sibling_concepts(concept, all_concepts)
    parents = get_parent_concepts(concept, all_concepts)

    # Mix of siblings (hard) and distant concepts (easy)
    return siblings[:10] + sample_distant_concepts(concept, all_concepts, 10)
```

## Testing Plan

1. **Unit Tests**
   - Test each component in isolation
   - Mock dependencies

2. **Integration Tests**
   - Train small set of concepts (5-10)
   - Validate calibration
   - Check file I/O

3. **Regression Tests**
   - Compare results to current training
   - Ensure no accuracy degradation

4. **Calibration Tests**
   - Run `diagnose_lens_calibration.py` on trained lenses
   - Flag any lenses with calibration_score < 0.5

## Timeline

- **Week 1**: Create new files (trainer.py, lens_validation.py)
- **Week 2**: Refactor sumo_classifiers.py, test on small dataset
- **Week 3**: Archive legacy code, update documentation
- **Week 4**: Full retrain with validation, deploy new lenses
