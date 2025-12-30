# HatCat Production Configuration

**Purpose**: The canonical system configuration that ships. All performance claims reference this exact configuration.

**Version**: 1.0 (shipping)

**Status**: Training in progress → validation pending → release

---

## What This Document Is

This is the **single source of truth** for HatCat's production performance. Every claim in documentation, papers, or demos must cite results from this configuration running against the validation suite defined here.

No development history. No experimental results. Just: "This is the system. Here's how it performs."

---

## Reference Configuration Specification

### 1. Model Configuration

```python
MODEL_CONFIG = {
    'model_id': 'google/gemma-3-4b-pt',
    'target_layers': [2],  # Layer 2 only for current reference
    'device': 'cuda',
    'dtype': 'float16',
}
```

**Rationale**:
- Layer 2 chosen for balance between semantic richness and computational efficiency
- Single layer simplifies validation and comparison
- Gemma-3-4b-pt provides stable, reproducible activations

### 2. Lens Architecture

```python
LENS_ARCHITECTURE = {
    'type': 'SimpleMLP',
    'input_dim': 4096,  # Gemma-3-4b hidden size
    'hidden_dims': [256, 128],
    'output_dim': 1,
    'activation': 'ReLU',
    'dropout': 0.1,
}
```

### 3. Training Configuration

```python
TRAINING_CONFIG = {
    'validation_mode': 'falloff',  # FALLOFF STRICT → FALLOFF HIGH → FALLOFF MEDIUM
    'adaptive_scaling': True,
    'initial_samples': 10,  # 5 positive + 5 negative
    'tier_thresholds': {
        'A': {'score': 0.50, 'cycle': 0},   # Top tier, achievable in first cycle
        'B+': {'score': 0.35, 'cycle': 0},
        'B': {'score': 0.23, 'cycle': 1},   # Acceptable at high tier (cycle 1)
        'C+': {'score': 0.15, 'cycle': 1},
    },
    'max_cycles': 2,  # 0→1 (10→30 samples)
    'max_samples_per_cycle': [10, 20],  # Cycle 0: 10, Cycle 1: 20
    'convergence_patience': 3,
    'test_split': 0.2,
}
```

### 4. Validation Benchmark Suite

All lenses in the reference configuration must be validated against these benchmarks:

#### 4.1 Calibration Quality
- **Metric**: Lens confidence on target concept vs others
- **Target Score**: ≥0.50 for A-tier, ≥0.23 for B-tier
- **Test Set**: 40 held-out prompts per concept

#### 4.2 OOD Generalization
- **Metric**: F1 score on prompts from different domains
- **Target**: ≥0.95 average F1
- **Test Set**: Cross-domain prompts not seen during training

#### 4.3 Behavioral vs Definitional Robustness
- **Metric**: Detection accuracy across prompt types
- **Test Types**:
  - Definitional: "What does X mean?"
  - Behavioral (Neutral): "How would someone X?"
  - Behavioral (Prosocial): "How can X help others?"
  - Behavioral (Antisocial): "How can X harm others?"
- **Target**: ≥80% cross-type detection

#### 4.4 Steering Quality
- **Metric**: Coherence under intervention
- **Test Range**: -0.5 to +0.5 (with contamination removal)
- **Target**: 100% coherence, linear Δ vs strength

#### 4.5 Monitoring Overhead
- **Metric**: Per-token latency with cascade loading
- **Test Conditions**:
  - Light load: ~70 lenses
  - Heavy load: ~1,350 lenses
  - Aggressive pruning: top-30
- **Target**: <100ms per token at heavy load

### 5. Concept Coverage

**Reference Concept Set**: 3,278 SUMO concepts (v2 training set)

**Distribution Requirements**:
- **A-tier**: ≥30% of concepts (minimum 983 concepts)
- **B+ or better**: ≥60% of concepts (minimum 1,967 concepts)
- **B or better**: ≥80% of concepts (minimum 2,622 concepts)
- **C+ or better**: ≥95% of concepts (minimum 3,114 concepts)

**Coverage Dimensions**:
- Physical objects and processes
- Abstract concepts (emotions, ethics, reasoning)
- AI safety concepts (deception, control, alignment)
- Temporal and spatial concepts
- Social and relational concepts

### 6. Validation Pipeline

**Required Tests** (in order):

```bash
# 1. Calibration validation
python scripts/validate_trained_lenses.py --config reference_v2.0

# 2. OOD generalization
python scripts/test_ood_generalization.py --config reference_v2.0

# 3. Behavioral robustness
python scripts/test_behavioral_vs_definitional.py --lens-set v2.0

# 4. Steering quality
python scripts/test_steering_quality.py --config reference_v2.0 --range -0.5,0.5

# 5. Monitoring overhead
python scripts/benchmark_monitoring_overhead.py --config reference_v2.0

# 6. Generate report
python scripts/generate_validation_report.py --config reference_v2.0
```

**Output**: Single JSON file with all metrics
```json
{
  "reference_version": "v2.0-alpha",
  "timestamp": "2025-11-16T...",
  "model": "google/gemma-3-4b-pt",
  "total_concepts": 3278,
  "calibration": {...},
  "ood_generalization": {...},
  "behavioral_robustness": {...},
  "steering_quality": {...},
  "monitoring_overhead": {...},
  "grade_distribution": {
    "A": 1015,
    "B+": 982,
    "B": 870,
    "C+": 411
  },
  "overall_quality_score": 0.87
}
```

---

## Quality Gates

**Minimum requirements for reference acceptance**:

1. ✅ **Training completion**: All 3,278 concepts trained
2. ✅ **Grade distribution**: ≥80% at B-tier or better
3. ✅ **OOD F1**: ≥0.95 average
4. ✅ **Behavioral robustness**: ≥80% cross-type detection
5. ✅ **Steering coherence**: 100% at ±0.5 range
6. ✅ **Monitoring latency**: <100ms per token (heavy load)

**Blocker criteria** (automatic reference rejection):
- ❌ <70% concepts at B-tier or better
- ❌ <0.90 average OOD F1
- ❌ <70% behavioral cross-detection
- ❌ <95% steering coherence at ±0.5
- ❌ >150ms per token monitoring overhead

---

## System Updates

When the production configuration changes (improved architecture, better training, etc.), this document is updated to reflect the **current shipping system only**.

Old configurations are not documented here. If you need historical performance data, check git history or ask for the archived validation reports.

---

## Usage

**Claiming Performance**:
All HatCat performance claims must cite the validation report from this configuration:
```
"HatCat Production v1.0 validation report: validation_report_v1.0_20251116.json"
```

**Running Validation**:
```bash
./scripts/run_production_validation.sh
```

This generates the canonical validation report. Don't cherry-pick results from experiments.
