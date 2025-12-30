# Lens Accuracy Benchmark - Quick Start Guide

## Overview

The lens accuracy benchmark system validates lens performance across multiple contexts:

1. **Prompt activations**: How the lens responds to input text containing the concept
2. **Response activations**: How the lens responds during model generation
3. **Steered activations**: How the lens responds when steering vectors are applied

## Example Usage

### 1. Benchmark Concept Lenses

Test binary concept classifiers (e.g., "Hat", "Cat", "Geology"):

```bash
python scripts/benchmark_concept_lenses.py \
  --concepts Hat Cat Geology \
  --lens-dir results/trained_lenses \
  --n-samples 100 \
  --output results/lens_benchmarks/concepts.csv
```

**Expected CSV Output:**
```csv
prompt_concept,detected_concept,context,n_samples,mean_activation,std_activation,peak_activation,min_activation
Hat,Hat,prompt,100,0.95,0.03,0.99,0.87
Hat,Cat,prompt,100,0.02,0.01,0.05,0.00
Hat,Hat,response,100,0.92,0.05,0.98,0.80
Hat,Hat,steered_up,100,0.98,0.02,1.00,0.93
Hat,Hat,steered_down,100,0.05,0.03,0.12,0.00
...
```

### 2. Benchmark Simplex Lenses

Test three-pole simplex lenses (μ−, μ0, μ+):

```bash
python scripts/benchmark_simplex_lenses.py \
  --simplexes Hunger Temperature Arousal \
  --lens-dir results/s_tier_simplexes/run_20251117_082151 \
  --n-samples 100 \
  --output results/lens_benchmarks/simplexes.csv
```

**Expected CSV Output:**
```csv
simplex_dimension,pole_prompted,pole_detected,context,n_samples,mean_activation,std_activation,peak_activation,min_activation
Hunger,negative,negative,prompt,100,0.94,0.03,0.99,0.86
Hunger,negative,neutral,prompt,100,0.05,0.02,0.10,0.00
Hunger,negative,positive,prompt,100,0.02,0.01,0.05,0.00
Hunger,neutral,neutral,prompt,100,0.93,0.04,0.98,0.82
Hunger,mixed,neutral,steered_to_neutral,100,0.85,0.06,0.95,0.70
...
```

### 3. Analyze & Visualize Results

Generate confusion matrices, steering plots, and quality reports:

```bash
python scripts/analyze_lens_benchmarks.py \
  --concept-csv results/lens_benchmarks/concepts.csv \
  --simplex-csv results/lens_benchmarks/simplexes.csv \
  --output-dir results/lens_benchmarks/analysis
```

**Generated Outputs:**
- `confusion_matrix_prompt.png` - Heatmap of cross-concept activations (prompt context)
- `confusion_matrix_response.png` - Heatmap of cross-concept activations (response context)
- `steering_effectiveness.png` - Bar chart comparing baseline vs steered activations
- `simplex_<name>_prompt.png` - 3x3 pole confusion matrix for each simplex
- `simplex_steering.png` - Homeostatic steering effectiveness
- `concept_quality_report.json` - Quality metrics for each concept

## What to Look For

### Good Concept Lens Patterns

**Diagonal dominance** (prompted == detected):
- High mean activation (0.90+)
- Low standard deviation (0.05-)
- Minimal off-diagonal noise (0.05-)

**Steering effectiveness**:
- `steered_up` > `baseline` by 10%+
- `steered_down` < 0.10

**Example quality scores**:
```json
{
  "Hat": {
    "precision": 47.5,      // Diagonal / mean(off-diagonal)
    "selectivity": 0.95,    // 1 - (max_offdiag / diagonal)
    "steerability": 0.15,   // (steered_up - baseline) / baseline
    "suppression": 0.92     // 1 - (steered_down / baseline)
  }
}
```

### Good Simplex Lens Patterns

**3x3 diagonal blocks**:
- High diagonal (0.90+): Each pole strongly detects itself
- Low off-diagonal (0.10-): Minimal cross-activation between poles

**Homeostatic steering**:
- `steered_to_neutral` → High μ0 activation (0.80+), low μ−/μ+ (0.15-)
- `steered_to_negative` → High μ− activation, low μ0/μ+
- `steered_to_positive` → High μ+ activation, low μ0/μ−

## Quality Metrics

### Concept Lenses

1. **Precision**: Ratio of correct-concept activation to mean incorrect-concept activation
   - Good: 20+
   - Excellent: 50+

2. **Selectivity**: 1 - (max incorrect activation / correct activation)
   - Good: 0.85+
   - Excellent: 0.95+

3. **Steerability**: Relative increase from baseline to steered-up
   - Good: 0.10+
   - Excellent: 0.20+

4. **Suppression**: Relative decrease from baseline to steered-down
   - Good: 0.80+
   - Excellent: 0.95+

### Simplex Lenses

1. **Pole Separation**: Mean activation difference between correct pole and other poles
   - Good: 0.70+
   - Excellent: 0.85+

2. **Homeostatic Precision**: μ0 activation under neutral steering
   - Good: 0.70+
   - Excellent: 0.85+

3. **Cross-pole Suppression**: Max incorrect pole under steering
   - Good: <0.20
   - Excellent: <0.10

## Troubleshooting

### Low diagonal activation (concept doesn't detect itself)
- Check training data quality
- Verify concept definition is clear
- Consider more training samples

### High off-diagonal activation (false positives)
- Concepts may be too similar (e.g., "Cat" and "Feline")
- Need better negative examples in training
- Consider hierarchical relationships

### Poor steering effectiveness
- Lens may not be well-calibrated
- Try different steering strengths
- Check if concept is steerable (abstract concepts harder)

### Simplex poles cross-activate
- Pole definitions may overlap
- Check if examples are distinct
- Verify three-pole structure is appropriate for this dimension

## Advanced: Custom Prompt Generation

For better domain-specific benchmarking, you can modify prompt generation:

**In `benchmark_concept_lenses.py`:**
```python
def generate_concept_prompts(concept: str, n_samples: int, seed: int = 42) -> List[str]:
    # Add domain-specific templates
    templates = [
        "The {concept} is",
        "I saw a {concept}",
        # Add your custom templates here
    ]
    ...
```

**In `benchmark_simplex_lenses.py`:**
```python
def generate_pole_prompts(...):
    # Customize based on simplex dimension and pole type
    if simplex_dimension == "Hunger":
        # Hunger-specific prompts
        ...
```

## Integration with CI/CD

Add benchmark runs to your validation pipeline:

```bash
#!/bin/bash
# Run after training new lenses

# Benchmark
python scripts/benchmark_concept_lenses.py --concepts Hat Cat --n-samples 50 --output latest_bench.csv

# Analyze
python scripts/analyze_lens_benchmarks.py --concept-csv latest_bench.csv --output-dir latest_analysis

# Check quality thresholds
python scripts/validate_quality_thresholds.py --report latest_analysis/concept_quality_report.json
```

## See Also

- [Full Benchmark Design](LENS_ACCURACY_BENCHMARK.md)
- [Lens Training Guide](TRAINING_CODE_CONSOLIDATION.md)
- [Concept Pack Format](CONCEPT_PACK_FORMAT.md)
