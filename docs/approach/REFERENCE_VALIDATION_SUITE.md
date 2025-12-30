# Reference Configuration Validation Suite

**Purpose**: Comprehensive testing protocol for production configuration. All tests must pass before release.

**Version**: 1.0

**Last Updated**: 2025-11-16

---

## Required Tests for Production Release

All tests run against the reference configuration (Layer 2, 3,278 concepts, adaptive training).

### 1. Lens Calibration Quality

**Script**: `scripts/validate_trained_lenses.py`

**What it tests**:
- Per-concept calibration scores (target vs others)
- Grade distribution (A/B+/B/C+)
- Test set F1 scores

**Pass criteria**:
- ≥80% concepts at B-tier or better
- ≥95% average F1 on test set
- No concepts with <0.15 calibration score

**Output**: `validation_report_v1.0_calibration.json`

---

### 2. OOD Generalization

**Script**: `scripts/test_ood_generalization.py`

**What it tests**:
- Cross-domain prompt performance
- Concept detection on unseen prompt structures
- Robustness to paraphrasing

**Pass criteria**:
- ≥95% average F1 on OOD prompts
- <10% performance drop vs in-domain

**Output**: `validation_report_v1.0_ood.json`

---

### 3. Behavioral vs Definitional Robustness

**Script**: `scripts/test_behavioral_vs_definitional_training.py`

**What it tests**:
- Detection across prompt types:
  - Definitional ("What is X?")
  - Behavioral Neutral ("How would someone X?")
  - Behavioral Prosocial ("How can X help?")
  - Behavioral Antisocial ("How can X harm?")
- Cross-type generalization

**Pass criteria**:
- ≥80% cross-type detection rate
- Documents behavioral coverage gaps

**Output**: `validation_report_v1.0_behavioral.json`

**Known limitation**: Current test uses quasi-behavioral prompts (descriptions of behavior, not actual behavioral elicitation). True behavioral coverage requires future work.

---

### 4. Conceptual Density Map (Activation Topology Analysis)

**Script**: `scripts/analyze_activation_topology.py` (**NEW - needs creation**)

**What it tests**:
- Activation correlation matrix across all concepts
- Co-activation clustering
- Coverage estimation of model activation space
- Blind spot identification
- Cascade risk zones (topologically adjacent concepts)

**Method**:
1. Load test prompts for all 3,278 concepts (~130K prompts total)
2. Extract activations at Layer 2 in batches
3. Compute pairwise correlation matrix (3,278 × 3,278)
4. Cluster analysis (identify dense regions vs sparse regions)
5. Estimate coverage: % of activation space variance explained by lens set
6. Identify high-correlation clusters (>0.7 correlation = cascade risk)

**Pass criteria**:
- Coverage map generated successfully
- Blind spots documented
- Cascade risk zones identified
- No crashes or errors

**Output**: `validation_report_v1.0_topology.json`

**Metrics**:
```json
{
  "activation_space_coverage_estimate": "35-45%",
  "total_concepts": 3278,
  "dense_clusters": [
    {"name": "physical_objects", "size": 850, "coverage": "high"},
    {"name": "ai_psychology", "size": 45, "coverage": "medium"}
  ],
  "blind_spots": [
    {"description": "behavioral_verbs", "estimated_importance": "high"},
    {"description": "relational_dynamics", "estimated_importance": "medium"}
  ],
  "cascade_risk_zones": [
    {"concepts": ["deception", "certainty", "agency"], "correlation": 0.85}
  ],
  "s_tier_candidates_topological": ["list of concepts in high-correlation clusters"]
}
```

**Known limitations**:
- Single-layer analysis (Layer 2 only)
- Does not capture layer-specific topology changes
- Does not test behavioral activation space (only definitional)

---

### 5. Multi-Layer Topology Analysis

**Script**: `scripts/analyze_multilayer_topology.py` (**NEW - needs creation**)

**What it tests**:
- How conceptual neighborhoods change across layers
- Layer-specific concept activation patterns
- Cross-layer stability of concepts

**Method**:
1. Select representative subset of concepts (~500 covering all clusters)
2. Extract activations at layers [2, 6, 10, 15, 20]
3. Compute correlation matrix per layer
4. Measure topology drift: how much do neighborhoods change?
5. Identify layer-specific S-tier candidates

**Pass criteria**:
- Multi-layer map generated successfully
- Layer-drift analysis complete
- Documents layer-specific blind spots

**Output**: `validation_report_v1.0_multilayer.json`

**Known limitations**:
- Computationally expensive (may need sampling)
- Current lens set is single-layer (Layer 2), so detection on other layers is inference only

---

### 6. Steering Quality

**Script**: `scripts/test_steering_quality.py`

**What it tests**:
- Coherence under intervention at range [-0.5, +0.5]
- Linear relationship between strength and semantic shift
- No behavioral collapse

**Pass criteria**:
- 100% coherence at ±0.5 (with contamination removal)
- Linear Δ vs strength (R² > 0.9)

**Output**: `validation_report_v1.0_steering.json`

---

### 7. Monitoring Overhead Benchmark

**Script**: `scripts/benchmark_monitoring_overhead.py`

**What it tests**:
- Per-token latency with cascade loading
- Memory footprint
- Scalability to large lens sets

**Test conditions**:
- Light load: ~70 lenses
- Heavy load: ~1,350 lenses
- Aggressive pruning: top-30

**Pass criteria**:
- <100ms per token at heavy load
- <10GB memory for full cascade

**Output**: `validation_report_v1.0_monitoring.json`

---

### 8. S-Tier Simplex Identification

**Script**: `scripts/identify_s_tier_simplexes.py` (**NEW - needs creation**)

**What it does**:
- Combines semantic simplex review results
- Adds topological S-tier candidates (from topology analysis)
- Manual curation against S-tier criteria
- Validates asymmetric tolerance bounds

**Inputs**:
- Simplex agentic review results (~20-50 semantic simplexes)
- Topological cascade risk zones
- S-tier criteria from SIMPLEX_FRAMEWORK_PRIORITIES.md

**Output**: `s_tier_simplexes_v1.0.json`

**Format**:
```json
{
  "semantic_simplexes": 42,
  "topological_additions": 38,
  "total_s_tier": 80,
  "simplexes": [
    {
      "dimension": "certainty",
      "negative_pole": {"concept": "Confusion", "synset": "..."},
      "neutral_homeostasis": {"concept": "CalibratedUncertainty", "synset": "..."},
      "positive_pole": {"concept": "Certainty", "synset": "..."},
      "source": "semantic",
      "validation_status": "pass"
    }
  ],
  "coverage_gaps": ["behavioral_verbs", "relational_dynamics"]
}
```

---

## Execution Order

Tests must run in this order:

```bash
# 1. Basic quality
./scripts/validate_trained_lenses.py

# 2. Generalization
./scripts/test_ood_generalization.py
./scripts/test_behavioral_vs_definitional_training.py

# 3. Topology analysis (informs S-tier)
./scripts/analyze_activation_topology.py
./scripts/analyze_multilayer_topology.py

# 4. S-tier identification
./scripts/identify_s_tier_simplexes.py

# 5. Performance
./scripts/test_steering_quality.py
./scripts/benchmark_monitoring_overhead.py

# 6. Generate combined report
./scripts/generate_validation_report.py --output validation_report_v1.0.json
```

---

## Combined Validation Report

**File**: `validation_report_v1.0.json`

**Structure**:
```json
{
  "version": "1.0",
  "date": "2025-11-16",
  "configuration": {
    "model": "google/gemma-3-4b-pt",
    "layers": [2],
    "total_concepts": 3278,
    "training_mode": "adaptive_falloff"
  },
  "calibration": { /* from test 1 */ },
  "ood_generalization": { /* from test 2 */ },
  "behavioral_robustness": { /* from test 3 */ },
  "topology": {
    "single_layer": { /* from test 4 */ },
    "multi_layer": { /* from test 5 */ }
  },
  "s_tier": { /* from test 8 */ },
  "steering_quality": { /* from test 6 */ },
  "monitoring_overhead": { /* from test 7 */ },
  "quality_gates": {
    "all_passed": false,
    "blockers": [
      "Behavioral coverage gaps documented but not resolved",
      "Multi-layer topology shows drift - single-layer lenses may miss late-layer concepts"
    ]
  },
  "known_limitations": [
    "Single-layer lenses (Layer 2 only)",
    "Definitional bias in training data (behavioral verbs underrepresented)",
    "~35-45% estimated activation space coverage (blind spots in behavioral/relational spaces)"
  ],
  "recommended_improvements": [
    "Add behavioral verb coverage",
    "Train multi-layer lenses for layer-specific concepts",
    "Generate truly behavioral training data (not descriptive)"
  ]
}
```

---

## Release Criteria

**Minimum viable release** (v1.0):
- ✅ All tests complete without errors
- ✅ Quality gates pass (except documented blockers)
- ✅ Known limitations clearly documented
- ✅ Coverage map shows what we can/can't detect

**NOT required for v1.0**:
- ❌ 100% activation space coverage
- ❌ Multi-layer lenses
- ❌ Behavioral verb coverage
- ❌ All S-tier simplexes trained

**Why**: v1.0 is a **documented, validated baseline**. We know what it does, what it doesn't do, and what needs improvement. That's shippable.

---

## Future Validation Tests (v2.0+)

These tests are valuable but not blocking for v1.0:

1. **Adversarial Prompt Testing**: Steering robustness under jailbreak attempts
2. **Long-Conversation Drift**: Concept activation stability over 100+ turn dialogues
3. **Behavioral Elicitation Coverage**: True behavioral prompts (not descriptions)
4. **Cross-Model Generalization**: Do lenses transfer to other models?
5. **Temporal Dynamics**: Activation sequences and causal chains

These inform v2.0 design but aren't required to ship v1.0.

---

## Scripts to Create

**High priority** (blocking for validation):
1. `scripts/analyze_activation_topology.py` - Conceptual density map
2. `scripts/identify_s_tier_simplexes.py` - S-tier curation
3. `scripts/generate_validation_report.py` - Combined report generator

**Medium priority** (valuable but not blocking):
4. `scripts/analyze_multilayer_topology.py` - Cross-layer analysis

**Low priority** (future work):
5. `scripts/test_adversarial_robustness.py`
6. `scripts/test_long_conversation_drift.py`

---

## Usage

Once all scripts exist:

```bash
# Run full validation suite
./scripts/run_production_validation.sh

# This executes all tests in order and generates:
# - validation_report_v1.0.json (combined)
# - Individual test outputs
# - Pass/fail summary
```

**Time estimate**: ~4-6 hours for full suite (mostly topology analysis)

**Output**: Single source of truth for all performance claims in whitepaper/documentation.
