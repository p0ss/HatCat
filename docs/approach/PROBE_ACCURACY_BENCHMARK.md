# Lens Accuracy Benchmark System

## Overview

A comprehensive benchmarking system to validate lens accuracy across different activation contexts:
- **Prompt activations**: Lens response to input text containing the concept
- **Response activations**: Lens response during model generation of concept-related text
- **Steered activations**: Lens response when steering vectors are applied (up/down)

## Design

### Concept Lens Benchmark

For binary concept classifiers (e.g., "Hat", "Cat", "Geology"):

**CSV Structure:**
```
prompt_concept,detected_concept,context,n_samples,mean_activation,std_activation,peak_activation,min_activation
Hat,Hat,prompt,100,0.95,0.03,0.99,0.87
Hat,Cat,prompt,100,0.02,0.01,0.05,0.00
Hat,Geology,prompt,100,0.01,0.01,0.03,0.00
Hat,Hat,response,100,0.92,0.05,0.98,0.80
Hat,Cat,response,100,0.03,0.02,0.07,0.00
...
Hat,Hat,steered_up,100,0.98,0.02,1.00,0.93
Hat,Hat,steered_down,100,0.05,0.03,0.12,0.00
...
```

**Contexts:**
- `prompt`: Activate lens on prompts containing the concept
- `response`: Activate lens during model generation about the concept
- `steered_up`: Activate lens when steering vector is applied positively
- `steered_down`: Activate lens when steering vector is applied negatively

**Metrics:**
- `n_samples`: Number of test samples (default 100)
- `mean_activation`: Average lens activation
- `std_activation`: Standard deviation
- `peak_activation`: Maximum activation observed
- `min_activation`: Minimum activation observed

**Expected Patterns:**
- **Diagonal (prompt_concept == detected_concept)**: High mean (0.90+), low std
- **Off-diagonal (different concepts)**: Low mean (0.05-), low std
- **Steered up**: Higher than baseline prompt activation
- **Steered down**: Near-zero activation

---

### Simplex Lens Benchmark

For three-pole simplex lenses (e.g., "Hunger" with μ−, μ0, μ+):

**CSV Structure:**
```
prompt_concept,pole_prompted,pole_detected,context,n_samples,mean_activation,std_activation,peak_activation,min_activation
Hunger,negative,negative,prompt,100,0.94,0.03,0.99,0.86
Hunger,negative,neutral,prompt,100,0.05,0.02,0.10,0.00
Hunger,negative,positive,prompt,100,0.02,0.01,0.05,0.00
Hunger,neutral,negative,prompt,100,0.03,0.02,0.08,0.00
Hunger,neutral,neutral,prompt,100,0.93,0.04,0.98,0.82
Hunger,neutral,positive,prompt,100,0.04,0.02,0.09,0.00
...
Hunger,negative,negative,steered_homeostasis,100,0.12,0.05,0.25,0.02
Hunger,negative,neutral,steered_homeostasis,100,0.85,0.06,0.95,0.70
...
```

**Poles:**
- `negative`: μ− (negative pole, e.g., "starving")
- `neutral`: μ0 (homeostatic equilibrium, e.g., "satiated")
- `positive`: μ+ (positive pole, e.g., "overfed")

**Contexts:**
- `prompt`: Activate lens on prompts containing pole-specific text
- `response`: Activate lens during model generation about the pole
- `steered_to_negative`: Apply steering toward μ−
- `steered_to_neutral`: Apply steering toward μ0 (homeostasis)
- `steered_to_positive`: Apply steering toward μ+

**Expected Patterns:**
- **3x3 diagonal blocks**: High activation when prompted == detected pole
- **Off-diagonal within simplex**: Low cross-activation between poles
- **Homeostatic steering**: Strong μ0 activation, suppressed μ−/μ+ activation

---

## Implementation

### 1. Concept Lens Benchmark

**File**: `scripts/benchmark_concept_lenses.py`

**Key functions:**
- `generate_concept_prompts(concept, n_samples)`: Generate diverse prompts containing concept
- `measure_prompt_activations(lens, prompts, model, tokenizer)`: Extract activations from input
- `measure_response_activations(lens, prompts, model, tokenizer)`: Extract activations during generation
- `measure_steered_activations(lens, prompts, steering_vector, model, tokenizer, direction)`: Extract with steering
- `run_concept_benchmark(concepts, lenses, n_samples)`: Full benchmark suite

**Output**: `results/lens_benchmarks/concept_lenses_<timestamp>.csv`

---

### 2. Simplex Lens Benchmark

**File**: `scripts/benchmark_simplex_lenses.py`

**Key functions:**
- `generate_pole_prompts(simplex_dimension, pole_type, n_samples)`: Generate pole-specific prompts
- `measure_simplex_activations(simplex_lenses, prompts, model, tokenizer, context)`: Extract all 3 pole activations
- `measure_homeostatic_steering(simplex_lenses, prompts, model, tokenizer)`: Test steering to μ0
- `run_simplex_benchmark(simplexes, lens_dirs, n_samples)`: Full simplex benchmark suite

**Output**: `results/lens_benchmarks/simplex_lenses_<timestamp>.csv`

---

## Analysis & Visualization

**File**: `scripts/analyze_lens_benchmarks.py`

**Capabilities:**
- **Confusion matrices**: Heatmaps showing cross-concept activation
- **Steering effectiveness**: Compare steered vs baseline activations
- **Lens quality scores**: Aggregate metrics (precision, selectivity, steerability)
- **Anomaly detection**: Flag unexpected activation patterns

**Outputs:**
- `results/lens_benchmarks/analysis_<timestamp>/confusion_matrix.png`
- `results/lens_benchmarks/analysis_<timestamp>/steering_effectiveness.png`
- `results/lens_benchmarks/analysis_<timestamp>/quality_report.json`

---

## Quality Metrics

### Concept Lens Quality

1. **Precision**: Mean activation on correct concept / Mean activation on incorrect concepts
2. **Selectivity**: 1 - (Max off-diagonal activation / Mean diagonal activation)
3. **Steerability**: (Mean steered_up - Mean baseline) / Mean baseline
4. **Stability**: 1 / Mean(std_activation across contexts)

### Simplex Lens Quality

1. **Pole Separation**: Min inter-pole distance in activation space
2. **Homeostatic Precision**: μ0 activation under neutral steering / Max(μ−, μ+) activation
3. **Steering Effectiveness**: Ability to drive system toward target pole
4. **Pole Stability**: Within-pole activation consistency across prompts

---

## Usage Examples

### Benchmark Concept Lenses
```bash
python scripts/benchmark_concept_lenses.py \
  --concepts Hat Cat Geology \
  --n-samples 100 \
  --output results/lens_benchmarks/concepts.csv
```

### Benchmark Simplex Lenses
```bash
python scripts/benchmark_simplex_lenses.py \
  --simplexes Hunger Temperature Arousal \
  --lens-dir results/s_tier_simplexes/run_20251117_082151 \
  --n-samples 100 \
  --output results/lens_benchmarks/simplexes.csv
```

### Analyze Results
```bash
python scripts/analyze_lens_benchmarks.py \
  --concept-csv results/lens_benchmarks/concepts.csv \
  --simplex-csv results/lens_benchmarks/simplexes.csv \
  --output-dir results/lens_benchmarks/analysis
```

---

## Future Extensions

1. **Cross-model validation**: Benchmark lenses on different model architectures
2. **Adversarial testing**: Deliberately confusing prompts (e.g., "The cat wore a hat")
3. **Temporal dynamics**: Track activation changes over token sequence
4. **Layer-wise analysis**: Compare lens performance across model layers
5. **Compositional testing**: Multi-concept prompts (e.g., "geological formation shaped like a cat")
