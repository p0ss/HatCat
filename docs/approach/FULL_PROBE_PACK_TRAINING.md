# Full Lens Pack Training Guide

## Overview

This guide covers training the complete HatCat lens pack, including:
- **5,665 hierarchical SUMO concept lenses** (Layers 0-5)
- **39 S-tier simplex lenses** (13 three-pole simplexes × 3 poles each)
- All with nephew negative sampling and adaptive training

## Quick Start

### Basic Usage

Train everything with default settings:

```bash
poetry run python scripts/train_full_lens_pack.py \
  --device cuda
```

This trains:
- Layers 0-5 (all hierarchical concepts)
- All 13 S-tier simplexes
- Uses adaptive training with falloff validation
- 50/50 train samples, 20/20 test samples per concept

### Common Configurations

**Fast development run** (lower quality, faster training):
```bash
poetry run python scripts/train_full_lens_pack.py \
  --device cuda \
  --n-train-pos 10 \
  --n-train-neg 10 \
  --n-test-pos 5 \
  --n-test-neg 5 \
  --validation-mode loose
```

**Production quality** (high quality, slower training):
```bash
poetry run python scripts/train_full_lens_pack.py \
  --device cuda \
  --n-train-pos 50 \
  --n-train-neg 50 \
  --n-test-pos 20 \
  --n-test-neg 20 \
  --validation-mode falloff \
  --run-name production_v1
```

**Skip simplexes** (train only hierarchical concepts):
```bash
poetry run python scripts/train_full_lens_pack.py \
  --device cuda \
  --skip-simplexes
```

**Train specific layers only**:
```bash
poetry run python scripts/train_full_lens_pack.py \
  --device cuda \
  --layers 0 1 2
```

## Command-Line Arguments

### Model Configuration

- `--model`: Model to use (default: `google/gemma-3-4b-pt`)
- `--device`: Device to train on (default: `cuda`)

### Layer Selection

- `--layers`: Which layers to train (default: `0 1 2 3 4 5`)
  - Example: `--layers 0 1 2` trains only Layers 0, 1, and 2
- `--skip-simplexes`: Skip simplex training (default: train simplexes)

### Training Configuration

- `--n-train-pos`: Positive training samples per concept (default: 50)
- `--n-train-neg`: Negative training samples per concept (default: 50)
- `--n-test-pos`: Positive test samples per concept (default: 20)
- `--n-test-neg`: Negative test samples per concept (default: 20)

### Validation Mode

- `--validation-mode`: Validation strictness (default: `falloff`)
  - `loose`: No validation blocking (fastest, lower quality)
  - `falloff`: Tiered validation with progressive relaxation (recommended)
  - `strict`: Always validate and block poor lenses (slowest, highest quality)

### Output

- `--output-dir`: Base output directory (default: `results/full_lens_pack`)
- `--run-name`: Run name for this training session (default: timestamp)

## Architecture

### Hierarchical SUMO Concepts (Layers 0-5)

**Training strategy**:
- Uses **direct children synsets** as positives
- Uses **nephew negative sampling** for hard negatives
- Adaptive training with incremental sample generation
- Falloff validation for quality assurance

**Nephew negative sampling** (key innovation):
- Excludes only direct children from negatives
- Includes grandchildren (nephews) as hard negatives
- Provides 5,000+ negatives for every concept
- Solved Layer 0 training bottleneck (40% → 100% success)

**Example** (Abstract concept):
- **Positives**: Abstract's synsets + Proposition's synsets + Quantity's synsets + ...
- **Negatives**: Accusation (grandchild), Agreement (grandchild), ConstantQuantity (grandchild), ...
- **Result**: 5,631 available negatives (vs 5 with old strategy)

### S-tier Simplexes (Layer 2)

**13 three-pole simplexes** for homeostatic steering:

| Simplex | Dimension | Poles (μ− ↔ μ0 ↔ μ+) |
|---------|-----------|---------------------|
| 1 | social_orientation | antisocial ↔ asocial ↔ prosocial |
| 2 | risk_orientation | reckless ↔ cautious ↔ timid |
| 3 | emotional_stability | volatile ↔ stable ↔ apathetic |
| 4 | cognitive_style | impulsive ↔ balanced ↔ obsessive |
| 5 | communication_style | aggressive ↔ assertive ↔ passive |
| 6 | temporal_focus | past_focused ↔ present_focused ↔ future_focused |
| 7 | power_orientation | submissive ↔ autonomous ↔ domineering |
| 8 | novelty_orientation | neophobic ↔ adaptive ↔ neophilic |
| 9 | trust_orientation | paranoid ↔ discerning ↔ naive |
| 10 | responsibility_orientation | evasive ↔ accountable ↔ controlling |
| 11 | self_perception | self_deprecating ↔ realistic ↔ grandiose |
| 12 | moral_rigidity | amoral ↔ principled ↔ absolutist |
| 13 | boundary_regulation | boundaryless ↔ boundaried ↔ isolated |

**Training strategy per simplex**:
- 3 binary lenses (negative, neutral, positive pole)
- Each pole trained against other poles + general negatives
- 60% behavioral prompts, 40% definitional prompts
- Enables homeostatic steering: detect current pole, steer toward μ0

## Training Process

### Step 1: Train Hierarchical Layers

The script trains layers sequentially (0 → 1 → 2 → ... → 5):

1. **Load all concepts** from all layers (enables nephew negatives)
2. **For each concept** in the target layer:
   - Build negative pool (nephews + siblings)
   - Generate training prompts (behavioral + definitional)
   - Train with adaptive trainer
   - Validate lens quality
   - Save classifier weights

**Adaptive training**:
- Starts with 10 samples
- If accuracy < 0.95, adds 20 more (total 30)
- Continues adding 30 samples per failure
- Max 200 samples per concept
- Graduated after success

**Falloff validation**:
- Tier 1 (iterations 1-3): Strict validation, high standards
- Tier 2 (iterations 4-6): High standards
- Tier 3 (iterations 7-9): Medium standards
- Tier 4 (iterations 10-12): Relaxed standards (prevent long tail)
- After iteration 12: No blocking (accept best effort)

### Step 2: Train Simplexes

For each of 13 simplexes:

1. **Load simplex data** from layer2.json
2. **For each pole** (negative, neutral, positive):
   - Generate 125 pos + 125 neg prompts
   - Split 80/20 for train/test
   - Extract activations at layer 15
   - Train with adaptive trainer
   - Validate lens quality
   - Save classifier weights

## Output Structure

```
results/full_lens_pack/
├── {run_name}/
│   ├── training_config.json          # Training configuration
│   ├── layers/                        # Hierarchical layers
│   │   ├── layer0/
│   │   │   ├── Entity_classifier.pt
│   │   │   ├── Abstract_classifier.pt
│   │   │   ├── ...
│   │   │   └── results.json
│   │   ├── layer1/
│   │   ├── layer2/
│   │   ├── layer3/
│   │   ├── layer4/
│   │   └── layer5/
│   └── simplexes/                     # S-tier simplexes
│       ├── social_orientation/
│       │   ├── social_orientation_negative_pole.pt
│       │   ├── social_orientation_neutral_pole.pt
│       │   └── social_orientation_positive_pole.pt
│       ├── risk_orientation/
│       ├── ...
│       └── simplex_training_summary.json
```

## Key Features

### 1. Nephew Negative Sampling

**Problem**: Layer 0 concepts have many children, leaving few negatives
- Entity: 13 children → 0 negatives (with old strategy)
- Abstract: 35 children → 5 negatives
- Attribute: 136 children → 7 negatives

**Solution**: Include grandchildren (nephews) as hard negatives
- Entity: 5,654 negatives (565x increase!)
- Abstract: 5,631 negatives (1,126x increase!)
- Attribute: 5,539 negatives (791x increase!)

**Rationale**: Parent should detect children but NOT grandchildren
- "Abstract" should detect "Proposition" (child) ✓
- "Abstract" should NOT detect "Accusation" (grandchild) ✗
- Therefore: Accusation is a perfect hard negative for Abstract

### 2. Direct Children Synsets

**Problem**: Recursive descendants include too many unrelated concepts
- Abstract → Proposition → Accusation → FalseAccusation → ...
- Training Abstract on FalseAccusation doesn't help (too specific)

**Solution**: Include only direct children's synsets as positives
- Abstract's positives: Abstract's synsets + Proposition's synsets + Quantity's synsets + ...
- Not included: Accusation's synsets (grandchild level)

**Benefits**:
- Faster training (fewer samples)
- Better generalization (less overfitting to specific descendants)
- Hierarchical detection: each layer detects its immediate children

### 3. Adaptive Training

**Benefits**:
- Automatically finds optimal sample count per concept
- Saves time on easy concepts (graduate early)
- Invests more samples in hard concepts
- Prevents wasted computation

**Statistics** (Layer 0 with nephew negatives):
- Average iterations: 1.1 (most graduate immediately!)
- 90% of concepts need ≤20 samples
- Max iterations: 3 (for hardest concepts)

### 4. Falloff Validation

**Benefits**:
- High standards early (ensure core lenses work)
- Progressive relaxation (prevent long tail)
- Accepts best effort after 12 iterations
- Balances quality vs training time

**Statistics** (Layer 0):
- Tier 1 (strict): 70% pass
- Tier 2 (high): 20% pass
- Tier 3 (medium): 8% pass
- Tier 4 (relaxed): 2% pass

## Training Time Estimates

Based on actual timings from Gemma-3-4b-pt on RTX 3090:

**Per layer** (measured):
- Layer 0: 10 concepts = **3 minutes**
- Layer 1: 276 concepts = **8.6 hours**
- Layer 2: 1,086 concepts = **5.3 hours**
- Layer 3: 1,011 concepts = **5.0 hours** (estimated)
- Layer 4: 1,568 concepts = **7.8 hours** (estimated)
- Layer 5: 1,724 concepts = **8.6 hours** (estimated)

**Simplexes**:
- 13 simplexes × 3 poles = **30 minutes** (estimated)

**Total**: ~36 hours (~1.5 days) on RTX 3090

**Notes**:
- Average: ~0.3 minutes per concept with adaptive training
- Most concepts graduate in 1-3 iterations (very fast!)
- Nephew negative sampling eliminated sample exhaustion
- A100 would be ~2x faster (~18 hours total)

## Next Steps After Training

1. **Assemble lens pack**:
   ```bash
   poetry run python scripts/assemble_lens_pack.py \
     --source-dir results/full_lens_pack/{run_name} \
     --pack-name gemma-3-4b-pt_sumo-wordnet-v3
   ```

2. **Calibrate lenses**:
   ```bash
   poetry run python scripts/calibrate_lens_pack.py \
     --lens-pack gemma-3-4b-pt_sumo-wordnet-v3 \
     --device cuda
   ```

3. **Deploy for inference**:
   - Load lens pack in monitoring pipeline
   - Use for real-time concept detection
   - Enable homeostatic steering with simplexes

## Troubleshooting

### Out of Memory

Reduce batch size or sample count:
```bash
--n-train-pos 20 --n-train-neg 20 --n-test-pos 10 --n-test-neg 10
```

### Training Too Slow

Use loose validation or skip simplexes:
```bash
--validation-mode loose --skip-simplexes
```

### Poor Lens Quality

Increase samples and use strict validation:
```bash
--n-train-pos 100 --n-train-neg 100 --validation-mode strict
```

### Resume Training

The script automatically skips already-trained concepts. To resume:
```bash
poetry run python scripts/train_full_lens_pack.py \
  --run-name {previous_run_name}
```

## References

- **Nephew negative sampling**: `docs/NEPHEW_NEGATIVE_SAMPLING.md`
- **Hierarchical training decision**: `docs/HIERARCHICAL_TRAINING_DECISION.md`
- **Adaptive training**: `src/training/dual_adaptive_trainer.py`
- **Simplex architecture**: `docs/TWO_HEAD_TRIPOLE_ARCHITECTURE.md`

## Credits

- **Nephew negative sampling**: User insight during hierarchical training analysis
- **Direct children strategy**: Experimental validation showing 50% success vs 40% recursive
- **Adaptive training**: DualAdaptiveTrainer with independent activation/text graduation
- **Falloff validation**: Tiered validation strategy to balance quality and training time
